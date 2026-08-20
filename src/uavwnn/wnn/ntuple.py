"""Regressor n-tuple **denso** — a rede de gradiente funcional de §4.3.1.

Referência: Katopodis (2022), `tmp/947841_compressed.pdf`. Páginas citadas são
as **impressas**; some 11 para achar no PDF.

    f_t(x) = (1/N) Σ_i v[i]_t(x)                                     (4.12)
    v[i]_t(x_t) = v[i]_{t-1}(x_t) + α_t Δ_t                          (4.11)

QUANDO USAR ESTA CLASSE, E QUANDO NÃO

    Esta é a **mesma matemática** da `RedeGradienteFuncional` de `redes.py`, com
    outro armazenamento: array `(d, N, 2^n)` alocado por inteiro em vez de
    dicionário esparso. A troca compra vetorização — `prever` e `treinar`
    processam lotes com indexação avançada do NumPy — e paga o teto de tupla.

    | | `NTupleRegressor` (aqui) | `RedeGradienteFuncional` |
    |---|---|---|
    | armazenamento | denso, `2^n` por RAM | dicionário esparso |
    | tupla máxima prática | `n ≲ 16` | 64, o teto do `Ramnet.jl` |
    | mapeamento | amostragem **por RAM**, sem reposição dentro da RAM | **partição** de todos os bits |
    | cobertura de bits | `N·n/B`, pode ser < 1 | sempre 1,0 |
    | lote | vetorizado | uma amostra por vez |

    O apêndice A da dissertação usa `n ∈ {16, 32, 64}`, fora do alcance daqui.
    Esta classe permanece porque as SPEC-01 e SPEC-03 foram medidas com ela e
    porque a vetorização importa para varrer configurações — mas **o modelo fiel
    é o de `redes.py`**.

    Equivalência verificada em `test_ator_esparso_concorda_com_o_regressor_denso`
    (mesmo mapeamento forçado nas duas), e contra o `Ramnet.jl` em
    `tests/test_diferencial.py`.

DUAS DECISÕES DE CONVENÇÃO

    **Média, não soma.** A eq. (4.12) traz o `1/N` explícito. Com
    `Δ = −η(f − y)`, a variação da saída é exatamente `−η(f − y)`, independente
    de `N` — o que permite varrer `N` sem confundir capacidade com passo. O
    protótipo do `PLANO.md` usava soma, e daí vinha a regra `η = c/N` da RNF-01
    antiga do PRD; a eq. (4.22) normaliza o mapa de características por `1/√N`,
    então `‖φ‖² = 1` e a divisão por `N` não existe. Regra retirada.

    **Sem fator de esquecimento.** O esquecimento pertence ao modelo de §4.2.1
    (o crítico), não a este. A combinação "gradiente funcional com esquecimento"
    não existe na dissertação — ver a pendência C2 no docstring de `redes.py`.
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
        #
        # Diferente do `:uniform_random` do Ramnet.jl, que particiona todos os
        # bits e portanto lê cada um exatamente uma vez. Aqui a cobertura é
        # `N·n/B` e pode ficar abaixo de 1 — o achado C4 do projeto. Ver a
        # propriedade `cobertura` abaixo.
        self.mapeamento = np.stack(
            [rng.choice(n_bits, size=tamanho_tupla, replace=False) for _ in range(n_ram)]
        )
        # O array alocado: `alocado = d × N × 2^n × bytes` do PRD §5.5. É a cota
        # superior teórica, e a razão do teto `n ≲ 16` desta classe.
        self.pesos = np.zeros((n_saidas, n_ram, 1 << tamanho_tupla), dtype=np.float64)
        # Convenção de endereço do `encode` do Ramnet.jl: `value += bit << (i-1)`,
        # isto é, o k-ésimo bit da tupla pesa 2^k.
        self._potencias = (1 << np.arange(tamanho_tupla)).astype(np.int64)

    # -- endereçamento ------------------------------------------------------

    def enderecos(self, bits: NDArray[np.bool_]) -> NDArray[np.int64]:
        """`τ[i](x)` para cada RAM — §3.2.2, p.23. `(M, B) -> (M, N)`.

        Versão vetorizada: `bits[:, mapeamento]` monta `(M, N, n)` de uma vez e
        o produto matricial com as potências de 2 dá todos os endereços. É o que
        esta classe compra em relação à esparsa.
        """
        if bits.ndim == 1:
            bits = bits[None, :]
        # (M, n_ram, tamanho_tupla) -> pesa por potências de 2
        selecionados = bits[:, self.mapeamento]
        return selecionados.astype(np.int64) @ self._potencias

    @property
    def cobertura(self) -> float:
        """Fração dos bits de entrada que alguma RAM lê — achado C4.

        Com amostragem por RAM, `N·n < B` implica bits que **nenhuma** RAM lê:
        informação da observação simplesmente invisível ao modelo. A partição de
        `redes.py` elimina o problema por construção; aqui ele precisa ser
        medido, e é o que a SPEC-01 checa antes de aceitar uma configuração.
        """
        return len(np.unique(self.mapeamento)) / self.n_bits

    def entradas_ocupadas(self) -> int:
        """Posições de LUT já escritas — o **ocupado** de T1 (PRD §5.5).

        Distinto do **alocado** (`d × N × 2^n`) e do **realizado** (footprint da
        estrutura de dados de fato usada). A SPEC-03 mediu que reportar só um dos
        três muda a conclusão por uma ordem de grandeza. Ver
        `src/uavwnn/custo/memoria.py`.
        """
        return int((self.pesos != 0).any(axis=0).sum())

    # -- inferência e treino ------------------------------------------------

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Eq. (4.12): `f(x) = (1/N) Σ_i v[i](x)`. `(M, B) -> (M, d)`.

        O `.mean(axis=1)` é a média sobre as **RAMs**, com denominador `N` fixo —
        posições nunca escritas valem zero e ainda contam. É o que a (4.12) diz.
        """
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
        """Eq. (4.11) com perda quadrática, uma amostra por vez.

        `c(x,y,z) = ½(z − y)²` dá `Δ_t = −(f − y)`, logo `v[i] -= η(f − y)` nas
        `N` posições endereçadas. Com a convenção de média, `Δf = −η(f − y)`
        exatamente — o passo efetivo é `η`, sem depender de `N`.

        Uma amostra por vez, e não em lote, porque a (4.11) é uma recorrência:
        `Δ_t` é avaliado em `f_{t-1}(x_t)`, a predição **antes** da atualização.
        Processar um lote com a predição congelada seria outro algoritmo.
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
                    # f_{t-1}(x_t) — a predição corrente, antes de escrever
                    f = self.pesos[k][idx, e].mean()
                    self.pesos[k][idx, e] -= eta * (f - alvo[m, k])
