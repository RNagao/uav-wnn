"""Validação de consistência — SPEC-07 T-07.6.

A trava, não a verificação (RF-05): um KF mal sintonizado produz erro médio
plausível e covariância mentirosa. Nenhum resultado de RL é olhado antes de NEES
e NIS passarem, para cada `σ` da varredura. Consistência medida antes de qualquer
número de RL é o que separa este trabalho de um relatório com número bonito e
conclusão errada.

Três testes, do que precisa da verdade ao que não precisa:

- **NEES** `ε = (p − x̂)ᵀ P⁻¹ (p − x̂)` — precisa da verdade, só existe em
  simulação. Detecta filtro otimista (acima da faixa) ou conservador (abaixo).
- **NIS** `ν = ỹᵀ S⁻¹ ỹ` — não precisa da verdade; é o que existiria num voo
  real.
- **Brancura da inovação** — o mais informativo dos três: inovação correlacionada
  é a assinatura direta de erro de GPS não branco e de erro de modelo não
  modelado (inércia, discretização), que o NEES médio mascara quando o viés é
  pequeno mas persistente.

**Protocolo Monte Carlo** (T-07.6): média sobre `M` execuções no **mesmo índice
de tempo `k`**, nunca média temporal como teste primário — passos consecutivos
não são independentes. `M·ε̄_k ~ χ²(M·n)`, e a faixa é computada em tempo de
execução por `scipy.stats.chi2`, nunca hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


def nees_por_passo(
    erros: NDArray[np.float64], covs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`ε = eᵀ P⁻¹ e` por amostra.

    `erros` de forma `(..., n)`, `covs` de forma `(..., n, n)`. Devolve `(...,)`.
    """
    erros = np.asarray(erros, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    p_inv = np.linalg.inv(covs)
    return np.einsum("...i,...ij,...j->...", erros, p_inv, erros)


def nis_por_passo(
    inovacoes: NDArray[np.float64], esses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`ν = ỹᵀ S⁻¹ ỹ` por amostra. Mesma forma que `nees_por_passo`."""
    return nees_por_passo(inovacoes, esses)


def faixa_chi2(M: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Faixa de aceitação bilateral do NEES/NIS **médio** sobre `M` execuções.

    `M·ε̄ ~ χ²(M·n)`, logo a faixa a `1−α` para `ε̄` é
    `[χ²_{Mn}(α/2)/M, χ²_{Mn}(1−α/2)/M]`. Com `M = 100`, `n = 2`, `α = 0,05` dá
    ≈ `[1,63; 2,41]` em torno do valor esperado `n = 2`.
    """
    gl = M * n
    return chi2.ppf(alpha / 2, gl) / M, chi2.ppf(1 - alpha / 2, gl) / M


def faixa_chi2_individual(n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Faixa por amostra (uma execução, um passo), para a fração fora do envelope."""
    return chi2.ppf(alpha / 2, n), chi2.ppf(1 - alpha / 2, n)


def autocorrelacao(residuos: NDArray[np.float64], max_lag: int = 20) -> NDArray[np.float64]:
    """Autocorrelação normalizada `ρ(l)`, `l = 1..max_lag`, de uma sequência.

    `residuos` de forma `(K,)` ou `(K, n)` (nesse caso agrega as componentes).
    Sob filtro correto a inovação é branca: `|ρ(l)| ≤ 1,96/√K`.
    """
    r = np.asarray(residuos, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    r = r - r.mean(axis=0, keepdims=True)
    K = r.shape[0]
    var = float((r * r).sum())
    rho = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        rho[lag - 1] = float((r[:-lag] * r[lag:]).sum()) / var if var > 0 else 0.0
    return rho


def limiar_brancura(K: int) -> float:
    """`1,96/√K` — o envelope de ruído branco a 95 % para a autocorrelação."""
    return 1.96 / np.sqrt(K)


@dataclass(frozen=True)
class Consistencia:
    """Resumo de consistência de um canal, num degrau, para um `σ`.

    `media_por_passo` e a faixa são o material da figura `ε̄_k × k`; `veredito`
    é o que RF-05 lê antes de liberar qualquer número de RL.
    """

    media_por_passo: NDArray[np.float64]
    faixa: tuple[float, float]
    fracao_fora: float
    nees_medio: float
    veredito: str  # "consistente" | "otimista" | "conservador"

    def passou(self) -> bool:
        return self.veredito == "consistente"


def avaliar(
    erros: NDArray[np.float64], covs: NDArray[np.float64], alpha: float = 0.05
) -> Consistencia:
    """Consistência de um conjunto de execuções.

    `erros` de forma `(M, K, n)` (M execuções × K passos × dimensão), `covs`
    `(M, K, n, n)`. O veredito segue a tabela de sintomas de T-07.6: acima da
    faixa é **otimista** (`P` menor que o erro — perigoso, a política confia
    demais); abaixo é **conservador** (`P` maior — seguro, mas desperdiça
    informação).
    """
    erros = np.asarray(erros, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    M, K, n = erros.shape
    eps = nees_por_passo(erros, covs)              # (M, K)
    media_por_passo = eps.mean(axis=0)             # (K,)
    lo, hi = faixa_chi2(M, n, alpha)
    lo_ind, hi_ind = faixa_chi2_individual(n, alpha)
    fora = np.mean((eps < lo_ind) | (eps > hi_ind))
    nees_medio = float(media_por_passo.mean())
    if nees_medio > hi:
        veredito = "otimista"
    elif nees_medio < lo:
        veredito = "conservador"
    else:
        veredito = "consistente"
    return Consistencia(
        media_por_passo=media_por_passo,
        faixa=(lo, hi),
        fracao_fora=float(fora),
        nees_medio=nees_medio,
        veredito=veredito,
    )
