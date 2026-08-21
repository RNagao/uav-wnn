"""Codificadores de termômetro (SPEC-01, T-01.2).

A rede n-tuple só enxerga bits, e o codificador precisa **preservar
similaridade**: entradas próximas devem virar padrões com pequena distância de
tupla (Katopodis §3.2.3). O termômetro faz isso por construção — o número de
bits ligados cresce monotonicamente com o valor.

Três decisões vêm de medições do ambiente, não do texto da dissertação:

1. **A faixa real é `[-0,5; 1,5]`**, não `[0,1]` nem o `Box(-1,1)` declarado.
   A observação é `(p_k − p_i + S)/(2S)` e `p_k − p_i` vive em `[-2S, 2S]`.
2. **O padding é `-1`**, fora da faixa dos válidos. Todo limiar fica **acima**
   de `-1`, então o padding vira o padrão **todo-zero** — constante e distinto
   de qualquer valor válido. É isto que dispensa o bit de validade.
3. **A distribuição depende do slot**: o sensor mais próximo concentra perto de
   um valor, o 12º espalha. Daí os esquemas por slot.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

PADDING = -1.0
"""Sentinela de slot vazio do ambiente de referência."""


@runtime_checkable
class Codificador(Protocol):
    """Contrato comum. `ajustar` vê o corpus; `codificar` é puro depois disso."""

    n_bits: int

    def ajustar(self, obs: NDArray[np.float32]) -> "Codificador": ...

    def codificar(self, obs: NDArray[np.float32]) -> NDArray[np.bool_]: ...


def _valido(coluna: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Máscara dos valores que não são padding."""
    return ~np.isclose(coluna, PADDING)


def _aplicar(obs: NDArray[np.float32], limiares: NDArray[np.float64]) -> NDArray[np.bool_]:
    """`bit[i] = (v >= limiar[i])`, por dimensão.

    Como todo limiar é maior que `PADDING`, um slot de padding produz o padrão
    **todo-zero**. E como o primeiro limiar é o mínimo empírico dos válidos,
    nenhum valor válido produz todo-zero — o mínimo acende exatamente um bit.
    Essa separação é o que torna o bit de validade desnecessário, e está
    coberta por teste.
    """
    if obs.ndim == 1:
        obs = obs[None, :]
    # (N, D, 1) >= (1, D, T)  ->  (N, D, T)
    bits = obs[:, :, None] >= limiares[None, :, :]
    return bits.reshape(obs.shape[0], -1)


class TermometroGlobal:
    """E0 — limiares uniformes, os mesmos para todas as dimensões.

    A linha de base ingênua. Se a hipótese de que a distribuição depende do
    slot se confirmar, este esquema desperdiça a maior parte dos bits: quase
    toda a massa de um slot cai num punhado de bins.
    """

    def __init__(self, n_niveis: int) -> None:
        self.n_niveis = n_niveis
        self._limiares: NDArray[np.float64] | None = None
        self.n_bits = 0

    def ajustar(self, obs: NDArray[np.float32]) -> "TermometroGlobal":
        v = obs[_valido(obs)]
        lo, hi = float(v.min()), float(v.max())
        niveis = np.linspace(lo, hi, self.n_niveis)
        self._limiares = np.tile(niveis, (obs.shape[1], 1))
        self.n_bits = obs.shape[1] * self.n_niveis
        return self

    def codificar(self, obs: NDArray[np.float32]) -> NDArray[np.bool_]:
        assert self._limiares is not None, "chame ajustar() antes"
        return _aplicar(obs, self._limiares)

    # -- persistência: os limiares SÃO estado treinado ----------------------

    def salvar(self, caminho) -> None:
        """Grava os limiares.

        Eles **têm** de viajar com o modelo. Reajustar o termômetro num corpus
        diferente daria uma binarização diferente para a mesma observação, e as
        posições de memória do modelo passariam a ser endereçadas por outros bits
        — erro silencioso, que apareceria como "a política não aprendeu".

        É também por isso que o harness de custo da SPEC-03 conta
        `limiares_por_dim` no footprint: são `D × T` floats de estado
        persistente, não constantes reconstruíveis de dois números (o que é o
        caso do E0, e a razão de lá `limiares_por_dim = 0`).
        """
        assert self._limiares is not None, "nada a salvar: chame ajustar() antes"
        np.savez_compressed(caminho, limiares=self._limiares,
                            n_niveis=np.array(self.n_niveis))

    @classmethod
    def carregar(cls, caminho) -> "TermometroQuantil":
        z = np.load(caminho, allow_pickle=False)
        t = cls(int(z["n_niveis"]))
        t._limiares = z["limiares"]
        t.n_bits = t._limiares.shape[0] * t.n_niveis
        return t


class TermometroPorSlot:
    """E1 — limiares lineares, com o suporte empírico de **cada** dimensão."""

    def __init__(self, n_niveis: int) -> None:
        self.n_niveis = n_niveis
        self._limiares: NDArray[np.float64] | None = None
        self.n_bits = 0

    def ajustar(self, obs: NDArray[np.float32]) -> "TermometroPorSlot":
        D = obs.shape[1]
        lim = np.empty((D, self.n_niveis))
        for d in range(D):
            col = obs[:, d]
            v = col[_valido(col)]
            if v.size == 0:  # dimensão só com padding
                v = np.array([0.0, 1.0])
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                hi = lo + 1e-6
            lim[d] = np.linspace(lo, hi, self.n_niveis)
        self._limiares = lim
        self.n_bits = D * self.n_niveis
        return self

    def codificar(self, obs: NDArray[np.float32]) -> NDArray[np.bool_]:
        assert self._limiares is not None, "chame ajustar() antes"
        return _aplicar(obs, self._limiares)

    # -- persistência: os limiares SÃO estado treinado ----------------------

    def salvar(self, caminho) -> None:
        """Grava os limiares.

        Eles **têm** de viajar com o modelo. Reajustar o termômetro num corpus
        diferente daria uma binarização diferente para a mesma observação, e as
        posições de memória do modelo passariam a ser endereçadas por outros bits
        — erro silencioso, que apareceria como "a política não aprendeu".

        É também por isso que o harness de custo da SPEC-03 conta
        `limiares_por_dim` no footprint: são `D × T` floats de estado
        persistente, não constantes reconstruíveis de dois números (o que é o
        caso do E0, e a razão de lá `limiares_por_dim = 0`).
        """
        assert self._limiares is not None, "nada a salvar: chame ajustar() antes"
        np.savez_compressed(caminho, limiares=self._limiares,
                            n_niveis=np.array(self.n_niveis))

    @classmethod
    def carregar(cls, caminho) -> "TermometroQuantil":
        z = np.load(caminho, allow_pickle=False)
        t = cls(int(z["n_niveis"]))
        t._limiares = z["limiares"]
        t.n_bits = t._limiares.shape[0] * t.n_niveis
        return t


class TermometroQuantil:
    """E2 — limiares nos quantis, bins equiprováveis segundo o corpus.

    É o "termômetro distributivo": cada bin recebe a mesma massa de
    probabilidade, então nenhum bit é desperdiçado numa região vazia.
    """

    def __init__(self, n_niveis: int) -> None:
        self.n_niveis = n_niveis
        self._limiares: NDArray[np.float64] | None = None
        self.n_bits = 0

    def ajustar(self, obs: NDArray[np.float32]) -> "TermometroQuantil":
        D = obs.shape[1]
        lim = np.empty((D, self.n_niveis))
        qs = np.linspace(0.0, 1.0, self.n_niveis)
        for d in range(D):
            col = obs[:, d]
            v = col[_valido(col)]
            if v.size == 0:
                v = np.array([0.0, 1.0])
            q = np.quantile(v, qs)
            # Quantis empatados (distribuição concentrada) colapsariam bits;
            # desempata por um epsilon crescente para manter a monotonicidade.
            q = np.maximum.accumulate(q)
            iguais = np.diff(q) == 0
            if iguais.any():
                q = q + np.arange(self.n_niveis) * 1e-9
            lim[d] = q
        self._limiares = lim
        self.n_bits = D * self.n_niveis
        return self

    def codificar(self, obs: NDArray[np.float32]) -> NDArray[np.bool_]:
        assert self._limiares is not None, "chame ajustar() antes"
        return _aplicar(obs, self._limiares)

    # -- persistência: os limiares SÃO estado treinado ----------------------

    def salvar(self, caminho) -> None:
        """Grava os limiares.

        Eles **têm** de viajar com o modelo. Reajustar o termômetro num corpus
        diferente daria uma binarização diferente para a mesma observação, e as
        posições de memória do modelo passariam a ser endereçadas por outros bits
        — erro silencioso, que apareceria como "a política não aprendeu".

        É também por isso que o harness de custo da SPEC-03 conta
        `limiares_por_dim` no footprint: são `D × T` floats de estado
        persistente, não constantes reconstruíveis de dois números (o que é o
        caso do E0, e a razão de lá `limiares_por_dim = 0`).
        """
        assert self._limiares is not None, "nada a salvar: chame ajustar() antes"
        np.savez_compressed(caminho, limiares=self._limiares,
                            n_niveis=np.array(self.n_niveis))

    @classmethod
    def carregar(cls, caminho) -> "TermometroQuantil":
        z = np.load(caminho, allow_pickle=False)
        t = cls(int(z["n_niveis"]))
        t._limiares = z["limiares"]
        t.n_bits = t._limiares.shape[0] * t.n_niveis
        return t


def cobertura_de_bits(n_ram: int, tamanho_tupla: int, n_bits: int) -> float:
    """Fração da entrada que a rede efetivamente lê: `N·n / B`.

    Achado C4: com sorteio independente por RAM, `N·n` pode ficar bem abaixo de
    `B`, e a rede **ignora o resto silenciosamente**. Aumentar `T` sem aumentar
    `N` reduz esta fração — é por isso que as varreduras de `T` e `N` não são
    independentes e precisam rodar juntas.
    """
    return min(1.0, (n_ram * tamanho_tupla) / n_bits)
