"""As duas redes n-tuple da dissertação do Katopodis (2022).

A dissertação define **dois modelos distintos**, não um só — e a leitura do
capítulo 4 fecha a pendência C2, que perguntava como conciliar gradiente
funcional *com* esquecimento. A resposta é que não se concilia: são redes
diferentes, com papéis diferentes no Algoritmo 4.4.

| papel | seção | modelo | guarda | esquecimento |
|---|---|---|---|---|
| ator | §4.3.1 | `RedeGradienteFuncional` | só valor | não |
| crítico | §4.2.1 | `RedeRegressaoEsquecimento` | valor + contador | sim (φ) |

O motivo fica claro no laço de treino: o crítico é regredido para o retorno
`G`, cujo alvo se desloca conforme a política melhora — é a distribuição
mutável que o fator de esquecimento existe para acompanhar. O ator não tem
alvo se movendo assim.

**Armazenamento esparso, por dicionário.** Não é detalhe de implementação: as
tabelas do apêndice A da dissertação usam tuplas de **16, 32 e 64 bits**, e
`2^64` posições por RAM não cabem em array denso. O `NTupleRegressor` denso de
`ntuple.py` continua servindo a SPEC-01 e a SPEC-03, mas fica preso a `n ≲ 16`;
estas classes não.

**Mapeamento por partição.** `randperm(B)` fatiado em `⌈B/n⌉` tuplas, como o
`:uniform_random` do Ramnet.jl. Cada bit é lido **exatamente uma vez**, então a
cobertura é 1,0 por construção — o achado C4 resolvido pela estrutura em vez de
por verificação. Quando `B` não é múltiplo de `n`, a última tupla é mais curta,
e isso é fiel: o Julia usa `Iterators.partition`, que também devolve resto.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MAX_TUPLA = 64
"""Mesmo teto do Ramnet.jl (`max_tuple_size` em 64 bits). Endereços vivem em
`uint64`, então `n > 64` não é representável."""


def mapeamento_particionado(
    n_bits: int, tamanho_tupla: int, seed: int = 0
) -> list[NDArray[np.int64]]:
    """Partição dos bits em `⌈B/n⌉` tuplas — o `:uniform_random` do Ramnet.

    Devolve lista, e não matriz, porque a última tupla pode ser mais curta.
    """
    if tamanho_tupla < 1:
        raise ValueError(f"tamanho_tupla={tamanho_tupla} deve ser >= 1")
    if tamanho_tupla > MAX_TUPLA:
        raise ValueError(f"tamanho_tupla={tamanho_tupla} excede {MAX_TUPLA}")
    if tamanho_tupla > n_bits:
        raise ValueError(f"tamanho_tupla={tamanho_tupla} maior que n_bits={n_bits}")

    ordem = np.random.default_rng(seed).permutation(n_bits)
    return [ordem[i:i + tamanho_tupla] for i in range(0, n_bits, tamanho_tupla)]


class _RedeNTupla:
    """Endereçamento comum às duas redes.

    O endereço de cada RAM é o inteiro cujos bits são os bits da entrada lidos
    por aquela tupla, com peso `2^k` na posição `k` — a mesma convenção do
    `encode` do Ramnet (`value += bit << (i-1)`).
    """

    def __init__(self, n_bits: int, tamanho_tupla: int, n_saidas: int = 1,
                 seed: int = 0) -> None:
        self.n_bits = n_bits
        self.tamanho_tupla = tamanho_tupla
        self.n_saidas = n_saidas
        self.mapeamento = mapeamento_particionado(n_bits, tamanho_tupla, seed)
        self.n_ram = len(self.mapeamento)
        self._potencias = [
            (np.uint64(1) << np.arange(len(t), dtype=np.uint64))
            for t in self.mapeamento
        ]

    @property
    def cobertura(self) -> float:
        """Fração dos bits de entrada lida por alguma RAM. Sempre 1,0 aqui."""
        return len(np.unique(np.concatenate(self.mapeamento))) / self.n_bits

    def enderecos(self, bits: NDArray[np.bool_]) -> NDArray[np.uint64]:
        """(M, n_bits) -> (M, n_ram)."""
        if bits.ndim == 1:
            bits = bits[None, :]
        saida = np.empty((bits.shape[0], self.n_ram), dtype=np.uint64)
        for i, (tupla, pot) in enumerate(zip(self.mapeamento, self._potencias)):
            saida[:, i] = (bits[:, tupla].astype(np.uint64) * pot).sum(axis=1)
        return saida

    def entradas_ocupadas(self) -> int:
        return sum(len(m) for m in self.memoria)

    def reiniciar(self) -> None:
        for m in self.memoria:
            m.clear()


class RedeGradienteFuncional(_RedeNTupla):
    """Ator — §4.3.1, eqs. (4.11) e (4.12).

        v[i]_t(x) = v[i]_{t-1}(x) + α_t · Δ_t,      i = 1..N        (4.11)
        f(x)      = (1/N) Σ_i v[i](x)                               (4.12)

    onde `Δ_t = −∂c/∂z |_{f(x)}` é o negativo do gradiente da perda em relação
    à *predição*. Posições nunca escritas valem zero, então a média é sempre
    sobre `N` — não sobre as visitadas.

    **Média, não soma.** A eq. (4.12) traz o `1/N` explicitamente. É a
    convenção que desacopla a taxa de aprendizado de `N`: com `Δ = −(f − y)`, a
    variação da saída num passo é exatamente `−α(f − y)`, seja qual for `N`.
    """

    def __init__(self, n_bits: int, tamanho_tupla: int, n_saidas: int = 1,
                 seed: int = 0) -> None:
        super().__init__(n_bits, tamanho_tupla, n_saidas, seed)
        self.memoria: list[dict[int, NDArray[np.float64]]] = [
            {} for _ in range(self.n_ram)
        ]

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """(M, n_bits) -> (M, n_saidas)."""
        end = self.enderecos(bits)
        saida = np.zeros((end.shape[0], self.n_saidas))
        for m, linha in enumerate(end):
            for i, e in enumerate(linha):
                v = self.memoria[i].get(int(e))
                if v is not None:
                    saida[m] += v
        return saida / self.n_ram

    def atualizar(self, bits: NDArray[np.bool_], passo: NDArray[np.float64]) -> None:
        """Soma `passo` (já escalado por α) a todas as `N` posições endereçadas.

        É a eq. (4.11) crua. O chamador é quem monta `α_t · Δ_t` — no
        Algoritmo 4.4 esse termo é `α · δ · ∇_f ln f(A|S)`, e não o gradiente de
        uma perda supervisionada.
        """
        passo = np.atleast_1d(np.asarray(passo, dtype=np.float64))
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            v = self.memoria[i].get(chave)
            if v is None:
                self.memoria[i][chave] = passo.copy()
            else:
                v += passo

    def treinar(self, bits: NDArray[np.bool_], alvo: NDArray[np.float64],
                eta: float = 0.1, epocas: int = 1,
                rng: np.random.Generator | None = None) -> None:
        """Regressão supervisionada com perda quadrática — `Δ = −(f − y)`.

        Existe para permitir os proxies da SPEC-01 e o teste diferencial contra
        o `NTupleRegressor` denso. O Algoritmo 4.4 não passa por aqui: ele usa
        `atualizar` diretamente.
        """
        alvo = np.asarray(alvo, dtype=np.float64)
        if alvo.ndim == 1:
            alvo = alvo[:, None]
        rng = rng or np.random.default_rng(0)
        for _ in range(epocas):
            for m in rng.permutation(len(bits)):
                x = bits[m:m + 1]
                f = self.prever(x)[0]
                self.atualizar(x, -eta * (f - alvo[m]))


class RedeRegressaoEsquecimento(_RedeNTupla):
    """Crítico — §4.2.1, eqs. (4.8), (4.9) e (4.10).

        v[i]_{t+1}(x) = φ · v[i]_t(x) + y_{t+1}                      (4.8)
        c[i]_{t+1}(x) = c[i]_t(x) + 1

        f(x) = Σ_i v[i](x) / Σ_i s[i](x)     se Σ s ≠ 0, senão 0     (4.9)

        s[i](x) = (1 − φ^c[i](x)) / (1 − φ)  se c > 0 e φ ≠ 1        (4.10)
                = c[i](x)                    caso contrário

    O denominador é **soma de pesos**, não de contadores: com esquecimento as
    observações passadas contribuem menos, e a média precisa ser ponderada. A
    forma fechada é a soma dos primeiros termos de uma série geométrica, que é
    o que (4.10) diz.

    Com `φ = 1` a rede degenera na regressão n-tuple original — média simples
    sobre as observações —, e (4.10) cai no ramo `c[i](x)` para evitar a divisão
    por zero de `(1 − 1^c)/(1 − 1)`.
    """

    def __init__(self, n_bits: int, tamanho_tupla: int, n_saidas: int = 1,
                 phi: float = 1.0, seed: int = 0) -> None:
        if not 0.0 <= phi <= 1.0:
            raise ValueError(f"phi={phi} fora de [0, 1]")
        super().__init__(n_bits, tamanho_tupla, n_saidas, seed)
        self.phi = float(phi)
        # valor e contador por posição — é o que dobra a memória do crítico em
        # relação à do ator, e o que a contabilidade de T1 precisa refletir.
        self.memoria: list[dict[int, tuple[NDArray, NDArray]]] = [
            {} for _ in range(self.n_ram)
        ]

    def _pesos(self, contador: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eq. (4.10), vetorizada e com o ramo de `φ = 1` tratado."""
        if self.phi == 1.0:
            return contador
        return (1.0 - self.phi ** contador) / (1.0 - self.phi)

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """(M, n_bits) -> (M, n_saidas). Zero onde nenhuma posição foi escrita."""
        end = self.enderecos(bits)
        saida = np.zeros((end.shape[0], self.n_saidas))
        for m, linha in enumerate(end):
            num = np.zeros(self.n_saidas)
            den = np.zeros(self.n_saidas)
            for i, e in enumerate(linha):
                item = self.memoria[i].get(int(e))
                if item is not None:
                    v, c = item
                    num += v
                    den += self._pesos(c)
            saida[m] = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        return saida

    def treinar(self, bits: NDArray[np.bool_], alvo: NDArray[np.float64]) -> None:
        """Eq. (4.8), uma amostra. Multiplica pelo esquecimento, depois soma."""
        alvo = np.atleast_1d(np.asarray(alvo, dtype=np.float64))
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            item = self.memoria[i].get(chave)
            if item is None:
                self.memoria[i][chave] = (alvo.astype(np.float64).copy(),
                                          np.ones(self.n_saidas))
            else:
                v, c = item
                v *= self.phi
                v += alvo
                c += 1.0
