"""Bootstrap pareado, IC da diferença e MDE — SPEC-02 T-02.7.

Este módulo é a razão de a SPEC-02 vir antes de qualquer treino. **Todo IC já
reportado neste projeto** — a validação 9/9 do guloso, o `greedy` vs `smart`, a
comparação de custo da SPEC-03, a replicação de §5.2.2, a validação 3/3 do
MADDPG — pressupõe que o procedimento aqui esteja calibrado, e isso nunca foi
verificado. `scripts/spec02_aa.py` verifica.

O QUE DECIDE OS PORTÕES É O IC DA DIFERENÇA PAREADA

Não os ICs marginais. A implicação vale numa direção só:

    ICs marginais que NÃO se sobrepõem  ⇒  há diferença
    ICs marginais que SE sobrepõem      ⇏  não há diferença

Com dados pareados a distância entre as duas leituras é grande, porque a
correlação entre métodos no mesmo cenário é alta. `bootstrap_pareado` existe para
que ninguém precise raciocinar sobre sobreposição de barras.

O ERRO QUE ESTE MÓDULO EXISTE PARA IMPEDIR

Reamostrar os dois métodos **independentemente** destrói o pareamento e infla o
IC — a diferença fica "não significativa" quando é. É um erro de duas linhas que
não aparece em nenhum teste unitário óbvio, e só o teste A/A o pega: reamostragem
independente dá cobertura errada, e nada mais denuncia.

Por isso `bootstrap_pareado` sorteia **um único conjunto de índices por réplica**
e o aplica aos dois vetores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

B_CANONICO = 1000
"""Reamostragens. 1000 segue o protocolo de referência (Katopodis §5.4).

**Ressalva registrada:** com `B = 1000`, cada cauda de 2,5 % é estimada por ~25
réplicas, o que torna os extremos do IC ruidosos na terceira casa. `B = 10_000`
roda onde é barato, e a diferença entre os dois é reportada uma vez para mostrar
que a conclusão não depende disso.
"""


# --------------------------------------------------------------------------
# agregado marginal
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Marginal:
    """Descreve **um** método. Não serve para comparar dois — use `Diferenca`."""

    n: int
    media: float
    ic_lo: float
    ic_hi: float
    mediana: float
    p90: float

    def como_linha(self) -> dict[str, float]:
        return {"n_ep": self.n, "media": self.media, "ic_lo": self.ic_lo,
                "ic_hi": self.ic_hi, "mediana": self.mediana, "p90": self.p90}


def bootstrap_media(
    x: Sequence[float],
    rng: np.random.Generator,
    B: int = B_CANONICO,
    alfa: float = 0.05,
) -> Marginal:
    """Média com IC bootstrap por percentis.

    Mediana e p90 vêm junto porque a distribuição de `t_conc_pen` é **assimétrica
    e censurada** (T-02.5): a média sozinha engana, e a censura em
    `max_episode_length` cria um atomo no extremo direito.
    """
    v = np.asarray(x, dtype=float)
    if v.size == 0:
        raise ValueError("amostra vazia")
    idx = rng.integers(0, v.size, size=(B, v.size))
    medias = v[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return Marginal(
        n=int(v.size), media=float(v.mean()), ic_lo=float(lo), ic_hi=float(hi),
        mediana=float(np.median(v)), p90=float(np.percentile(v, 90)),
    )


# --------------------------------------------------------------------------
# diferença pareada
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Diferenca:
    """`A − B` no mesmo conjunto de cenários. É isto que decide os portões."""

    n_pareado: int
    delta_media: float
    delta_ic_lo: float
    delta_ic_hi: float
    mde_95: float
    taxa_vitoria_a: float
    tv_ic_lo: float
    tv_ic_hi: float
    p_wilcoxon: float = float("nan")
    extras: dict = field(default_factory=dict)

    @property
    def contem_zero(self) -> bool:
        """Regra de decisão pré-registrada: diferença declarada quando `0 ∉ IC`."""
        return self.delta_ic_lo <= 0.0 <= self.delta_ic_hi

    @property
    def conclusivo(self) -> bool:
        return not self.contem_zero

    def como_linha(self) -> dict:
        return {
            "n_pareado": self.n_pareado,
            "delta_media": self.delta_media,
            "delta_ic_lo": self.delta_ic_lo,
            "delta_ic_hi": self.delta_ic_hi,
            "contem_zero": self.contem_zero,
            "mde_95": self.mde_95,
            "taxa_vitoria_a": self.taxa_vitoria_a,
            "tv_ic_lo": self.tv_ic_lo,
            "tv_ic_hi": self.tv_ic_hi,
            "p_wilcoxon": self.p_wilcoxon,
            **self.extras,
        }


def mde_95(a: Sequence[float], b: Sequence[float]) -> float:
    """Efeito mínimo detectável: `1,96 · σ̂_d / √n`.

    **Coluna obrigatória.** Sem ela um resultado nulo é ininterpretável — "não
    achamos diferença" precisa vir acompanhado de "e detectaríamos qualquer
    diferença acima de X".
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 2:
        return float("nan")
    return float(1.96 * d.std(ddof=1) / np.sqrt(d.size))


def bootstrap_pareado(
    a: Sequence[float],
    b: Sequence[float],
    rng: np.random.Generator,
    B: int = B_CANONICO,
    alfa: float = 0.05,
    menor_e_melhor: bool = True,
) -> Diferenca:
    """IC da diferença `média(A) − média(B)`, com pareamento preservado.

    **O ponto do módulo está em três linhas:** sorteia UM conjunto de índices por
    réplica e o aplica aos DOIS vetores. Reamostrar independentemente destruiria
    a correlação entre métodos no mesmo cenário e inflaria o IC.

    `menor_e_melhor` orienta a taxa de vitória: em `t_conc_pen`, A vence quando
    conclui mais cedo.
    """
    va, vb = np.asarray(a, float), np.asarray(b, float)
    if va.shape != vb.shape:
        raise ValueError(f"pareamento exige mesmo tamanho: {va.shape} vs {vb.shape}")
    if va.size == 0:
        raise ValueError("amostra vazia")

    n = va.size
    idx = rng.integers(0, n, size=(B, n))          # UM conjunto por réplica...
    deltas = va[idx].mean(axis=1) - vb[idx].mean(axis=1)   # ...aplicado aos dois
    lo, hi = np.percentile(deltas, [100 * alfa / 2, 100 * (1 - alfa / 2)])

    vitorias = (va < vb) if menor_e_melhor else (va > vb)
    tv = float(vitorias.mean())
    tv_boot = vitorias[idx].mean(axis=1)
    tv_lo, tv_hi = np.percentile(tv_boot, [100 * alfa / 2, 100 * (1 - alfa / 2)])

    return Diferenca(
        n_pareado=int(n),
        delta_media=float(va.mean() - vb.mean()),
        delta_ic_lo=float(lo), delta_ic_hi=float(hi),
        mde_95=mde_95(va, vb),
        taxa_vitoria_a=tv, tv_ic_lo=float(tv_lo), tv_ic_hi=float(tv_hi),
        p_wilcoxon=wilcoxon_pareado(va, vb),
    )


# --------------------------------------------------------------------------
# equivalência (TOST) — SPEC-08 T-08.2
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Equivalencia:
    """TOST: a diferença pareada cabe dentro de ±δ? (SPEC-08 T-08.2).

    `equivalente` é o critério de portão — o IC de (1−2α) da diferença inteiro
    dentro de [−δ, +δ]. `meia_largura` fica ao lado de `delta` de propósito:
    equivalência declarada com `meia_largura > delta` é **falta de poder
    disfarçada de resultado** (a amostra não distingue ±δ), e a spec manda marcar
    a célula como **não informativa** nesse caso.
    """

    n_pareado: int
    delta: float
    delta_media: float
    ic_lo: float
    ic_hi: float
    p_inferior: float   # H0: dif ≤ −δ  (bootstrap: fração de réplicas ≤ −δ)
    p_superior: float   # H0: dif ≥ +δ  (bootstrap: fração de réplicas ≥ +δ)

    @property
    def equivalente(self) -> bool:
        return self.ic_lo >= -self.delta and self.ic_hi <= self.delta

    @property
    def meia_largura(self) -> float:
        return 0.5 * (self.ic_hi - self.ic_lo)

    @property
    def informativa(self) -> bool:
        """Tem poder para declarar equivalência: o IC é mais estreito que ±δ."""
        return self.meia_largura <= self.delta

    def como_linha(self) -> dict:
        return {
            "n_pareado": self.n_pareado, "delta": self.delta,
            "delta_media": self.delta_media,
            "ic_lo": self.ic_lo, "ic_hi": self.ic_hi,
            "meia_largura": self.meia_largura,
            "equivalente": self.equivalente, "informativa": self.informativa,
            "p_inferior": self.p_inferior, "p_superior": self.p_superior,
        }


def tost(
    a: Sequence[float],
    b: Sequence[float],
    delta: float,
    rng: np.random.Generator,
    B: int = B_CANONICO,
    alfa: float = 0.05,
) -> Equivalencia:
    """Dois testes unilaterais (TOST) por bootstrap pareado, margem `delta`.

    Equivalência (`A ≡ B` a menos de ±δ) é declarada quando o IC de **(1−2α)**
    da diferença pareada `média(A) − média(B)` cai INTEIRO em [−δ, +δ] — a forma
    por IC do TOST, no nível α. Preserva o pareamento como `bootstrap_pareado`:
    um conjunto de índices por réplica, aplicado aos dois vetores.

    `delta > 0`; reporta também os dois p-valores unilaterais por bootstrap.
    """
    if delta <= 0:
        raise ValueError(f"a margem δ deve ser positiva: {delta}")
    va, vb = np.asarray(a, float), np.asarray(b, float)
    if va.shape != vb.shape:
        raise ValueError(f"pareamento exige mesmo tamanho: {va.shape} vs {vb.shape}")
    if va.size == 0:
        raise ValueError("amostra vazia")

    n = va.size
    idx = rng.integers(0, n, size=(B, n))
    deltas = va[idx].mean(axis=1) - vb[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [100 * alfa, 100 * (1 - alfa)])   # IC (1−2α)
    return Equivalencia(
        n_pareado=int(n), delta=float(delta),
        delta_media=float(va.mean() - vb.mean()),
        ic_lo=float(lo), ic_hi=float(hi),
        p_inferior=float(np.mean(deltas <= -delta)),
        p_superior=float(np.mean(deltas >= delta)),
    )


# --------------------------------------------------------------------------
# apoios livres de distribuição
# --------------------------------------------------------------------------


def wilcoxon_pareado(a: Sequence[float], b: Sequence[float]) -> float:
    """Wilcoxon dos postos sinalizados. **Apoio, não critério** — o critério é o
    IC da diferença.

    Devolve `nan` quando todas as diferenças são nulas (o teste é indefinido aí,
    e `scipy` levanta).
    """
    from scipy import stats

    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 1 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def mcnemar(sucesso_a: Sequence[bool], sucesso_b: Sequence[bool]) -> dict:
    """Diferença de taxa de sucesso, binária pareada.

    Usa o binomial exato sobre os discordantes, que é o McNemar exato — mais
    honesto que o chi-quadrado quando `b + c` é pequeno, e aqui costuma ser
    (taxas de sucesso perto de 1).
    """
    from scipy import stats

    sa = np.asarray(sucesso_a, bool)
    sb = np.asarray(sucesso_b, bool)
    if sa.shape != sb.shape:
        raise ValueError("pareamento exige mesmo tamanho")
    b = int((sa & ~sb).sum())      # A acerta, B erra
    c = int((~sa & sb).sum())      # B acerta, A erra
    if b + c == 0:
        p = float("nan")           # sem discordantes, o teste é indefinido
    else:
        p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p_mcnemar": p,
            "delta_taxa_sucesso": float(sa.mean() - sb.mean())}


def holm(p_valores: Sequence[float]) -> NDArray[np.float64]:
    """Correção de Holm–Bonferroni.

    **Coluna opcional nas varreduras, nunca aplicada retroativamente às
    primárias.** As comparações primárias são poucas e pré-registradas
    (`política vs guloso` e `WNN vs MLP pareada`); corrigi-las por multiplicidade
    depois de olhar os dados seria trocar o protocolo pelo resultado.
    """
    p = np.asarray(p_valores, float)
    n = p.size
    ordem = np.argsort(p)
    ajustado = np.empty(n)
    corrente = 0.0
    for posicao, i in enumerate(ordem):
        corrente = max(corrente, (n - posicao) * p[i])
        ajustado[i] = min(1.0, corrente)
    return ajustado
