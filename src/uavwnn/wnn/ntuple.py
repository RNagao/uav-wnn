"""Regressor n-tuple mínimo (SPEC-01; a SPEC-04 endurece).

    f_k(x) = (1/N) Σ_i W[k, i, a_i(x)]

Cada RAM `i` lê `n` bits pseudo-aleatórios da entrada binarizada e forma um
endereço `a_i(x)`. É **tile coding com ladrilhamento aleatório**, e é por isso
que herda as garantias de convergência de TD com aproximação linear.

DUAS DECISÕES DE CONVENÇÃO, ambas deliberadas:

**Média, não soma.** Katopodis usa `f = (1/N) Σ v_i`; o protótipo do `PLANO.md`
usava a soma. Com a média e a atualização `v_i -= η(f − y)`, a variação
efetiva da saída é exatamente `Δf = −η(f − y)` — **independente de `N`**. Com a
soma seria `−Nη(f − y)`, e daí a regra `η = c/N` do PRD (RNF-01). Adotamos a
média porque ela **desacopla a taxa de aprendizado de `N`**, que é o que
permite varrer `N` sem confundir capacidade com passo (achado da SPEC-06).

**Sem fator de esquecimento.** Os proxies da SPEC-01 são regressão
supervisionada estacionária: não há distribuição mudando para esquecer. O
esquecimento é da SPEC-04, junto com a decisão C2 sobre como conciliá-lo com o
gradiente funcional.

Armazenamento denso. O esparso, a contabilidade de memória e a quantização são
da SPEC-04 — aqui o que importa é medir o codificador, não o custo do modelo.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class NTupleRegressor:
    """Regressor de `d` saídas sobre entrada binária de `n_bits`."""

    def __init__(
        self,
        n_bits: int,
        n_ram: int = 48,
        tamanho_tupla: int = 12,
        n_saidas: int = 1,
        seed: int = 0,
    ) -> None:
        if tamanho_tupla > n_bits:
            raise ValueError(
                f"tamanho_tupla={tamanho_tupla} maior que n_bits={n_bits}"
            )
        self.n_bits = n_bits
        self.n_ram = n_ram
        self.tamanho_tupla = tamanho_tupla
        self.n_saidas = n_saidas

        rng = np.random.default_rng(seed)
        # Mapeamento das tuplas: quais bits cada RAM lê. Sorteio SEM reposição
        # dentro de cada RAM; RAMs diferentes podem repetir bits.
        self.mapeamento = np.stack(
            [rng.choice(n_bits, size=tamanho_tupla, replace=False) for _ in range(n_ram)]
        )
        self.pesos = np.zeros((n_saidas, n_ram, 1 << tamanho_tupla), dtype=np.float64)
        self._potencias = (1 << np.arange(tamanho_tupla)).astype(np.int64)

    # -- endereçamento ------------------------------------------------------

    def enderecos(self, bits: NDArray[np.bool_]) -> NDArray[np.int64]:
        """(M, n_bits) -> (M, n_ram), o endereço lido por cada RAM."""
        if bits.ndim == 1:
            bits = bits[None, :]
        # (M, n_ram, tamanho_tupla) -> pesa por potências de 2
        selecionados = bits[:, self.mapeamento]
        return selecionados.astype(np.int64) @ self._potencias

    @property
    def cobertura(self) -> float:
        """Fração dos bits de entrada que alguma RAM lê (achado C4)."""
        return len(np.unique(self.mapeamento)) / self.n_bits

    def entradas_ocupadas(self) -> int:
        """Posições de LUT já escritas. Alimenta T1 da §5.5 do PRD."""
        return int((self.pesos != 0).any(axis=0).sum())

    # -- inferência e treino ------------------------------------------------

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """(M, n_bits) -> (M, n_saidas)."""
        end = self.enderecos(bits)
        idx = np.arange(self.n_ram)
        # (M, n_ram) por saída -> média sobre as RAMs
        return np.stack(
            [self.pesos[k][idx, end].mean(axis=1) for k in range(self.n_saidas)],
            axis=1,
        )

    def treinar(
        self,
        bits: NDArray[np.bool_],
        alvo: NDArray[np.float64],
        eta: float = 0.1,
        epocas: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Gradiente funcional com perda quadrática, uma amostra por vez.

        `v_i -= η (f − y)` para as `N` entradas endereçadas. Com a convenção de
        média, `Δf = −η (f − y)` exatamente — o passo efetivo é `η`, sem
        depender de `N`.
        """
        alvo = np.asarray(alvo, dtype=np.float64)
        if alvo.ndim == 1:
            alvo = alvo[:, None]
        rng = rng or np.random.default_rng(0)
        end = self.enderecos(bits)
        idx = np.arange(self.n_ram)

        for _ in range(epocas):
            ordem = rng.permutation(end.shape[0])
            for m in ordem:
                e = end[m]
                for k in range(self.n_saidas):
                    f = self.pesos[k][idx, e].mean()
                    self.pesos[k][idx, e] -= eta * (f - alvo[m, k])
