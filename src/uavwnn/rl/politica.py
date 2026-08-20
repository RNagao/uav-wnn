"""Política gaussiana — Katopodis (2022) §4.3.3, p.44 a 47.

Páginas citadas são as **impressas**; some 11 para achar no PDF.

    f(a|s) = (1/η) exp(−½ (f(s) − a)ᵀ Σ⁻¹ (f(s) − a))                (4.34)
    ∇_f ln f(a|s) = Σ⁻¹ (a − f(s)) · k(s, ·)                          (4.35)

CONTEXTO — §4.3.3 oferece TRÊS parametrizações, e usamos a terceira

    | espaço de ação | distribuição | eqs. | usamos? |
    |---|---|---|---|
    | binário `{−1, +1}` | Bernoulli via sigmoide | (4.27)–(4.30) | não |
    | discreto finito | categórica via softmax | (4.31)–(4.33) | não |
    | **contínuo** | **gaussiana** | **(4.34)–(4.35)** | **sim** |

    "As actions are sampled from the parameterized distribution, its support must
    be the same, or at least be close to, the action space" (§4.3.3, p.44). A
    nossa ação é `⟨θ, s⟩` contínua, herdada do MDP de Lamenza sem alteração
    (PRD §5.2), logo a gaussiana é a escolha forçada.

A REDE EMITE A MÉDIA — A DISPERSÃO NÃO É APRENDIDA

    A eq. (4.34) é explícita: *"Σ is a **predefined** covariance matrix"*. O
    apêndice A lista `Σ` como hiperparâmetro constante:

        Mountain Car  Σ = 0,5      Hopper  Σ = 0,3

    Nas tarefas multivariadas o texto usa **identidade escalada** — "scaled
    identity matrices, that is, diagonal matrices where all nonzero elements are
    the same" (§5.1.3, p.53). É por isso que `Σ⁻¹` aparece aqui como escalar em
    vez de inversão de matriz.

    Esta é a diferença de fundo em relação ao `NTNModel` do repositório
    `anbubus`, onde o RLlib **aprende `log_std`** como saída da rede (daí o
    `num_outputs` dobrado). Aquele repositório implementa o capítulo 6 —
    trabalhos futuros —, não o capítulo 4.

O FATOR `k(s, ·)` DA EQ. (4.35) É IMPLÍCITO NA REDE

    A (4.35) traz `∇_f ln f(a|s) = Σ⁻¹(a − f(s)) · k(s, ·)`, onde `k` é a função
    de base do núcleo. Numa máquina de núcleo explícita seria preciso multiplicar
    pela base; na rede n-tuple **escrever nas `N` posições endereçadas por `s`
    já é aplicar a base** — é o conteúdo da eq. (4.19), que mostra que a
    atualização se propaga como `αΔ·k(s, ·)`.

    Por isso `grad_log` devolve apenas o vetor `Σ⁻¹(a − f(s))`, e é
    `RedeGradienteFuncional.atualizar` quem aplica a base.

DIVERGÊNCIA ENTRE O ARTIGO E O CÓDIGO DO AUTOR — pendência P10

    | | amostragem | gradiente |
    |---|---|---|
    | dissertação (4.34)/(4.35) | `N(f(s), Σ)`, logo desvio `√Σ` | `Σ⁻¹(a − f(s))` |
    | `ContinousActionPolicy.jl` | `Σ·z + f(s)`, tratando Σ como **desvio** | `inv(Σ)·(f(s) − a)` |

    O código do autor usa o mesmo número como desvio na amostragem e como
    variância no gradiente. Se `Σ` é a covariância, como (4.34) diz, a
    amostragem correta é `√Σ·z`.

    **Seguimos o artigo.** Com `Σ = 0,25` isso dá desvio 0,5 em vez de 0,25 —
    **fator 2 na exploração**, o que não é desprezível. `escala_como_desvio=True`
    reproduz o comportamento do Julia, para quem quiser comparar.

NÃO CONFUNDIR OS DOIS `φ`/`Σ` DO PROJETO

    A dissertação usa `φ` para o fator de esquecimento (§4.2.1) e `Σ` para a
    covariância da política (§4.3.3). O PRD usa `φ(x)` para o mapa de
    características binário (eq. 4.22). A colisão de símbolos vem do material de
    origem; no código: `phi` é o esquecimento, `escala_covariancia` é a
    covariância, e o mapa não é materializado fora dos testes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PoliticaGaussiana:
    """Política gaussiana de covariância predefinida sobre uma rede n-tuple.

    A rede fornece `f(s)`, a **média**. `Σ` é hiperparâmetro fixo, identidade
    escalada — ver o docstring do módulo.
    """

    def __init__(
        self,
        rede,
        escala_covariancia: float,
        escala_como_desvio: bool = False,
    ) -> None:
        if escala_covariancia <= 0:
            raise ValueError(f"escala_covariancia={escala_covariancia} deve ser > 0")
        self.rede = rede
        self.escala = float(escala_covariancia)
        # P10: se True, reproduz o `ContinousActionPolicy.jl` em vez do artigo.
        self.escala_como_desvio = escala_como_desvio

    @property
    def desvio(self) -> float:
        """Desvio-padrão da amostragem.

        Pelo artigo, `Σ` é **covariância** (eq. 4.34), então o desvio é `√Σ`.
        Com `escala_como_desvio` usamos `Σ` direto, como faz o Julia (P10).
        """
        return self.escala if self.escala_como_desvio else np.sqrt(self.escala)

    @property
    def inv_covariancia(self) -> float:
        """`Σ⁻¹` da eq. (4.35).

        Escalar porque `Σ` é identidade escalada (§5.1.3, p.53). Numa
        covariância cheia isto seria `np.linalg.inv`.
        """
        return 1.0 / self.escala

    def media(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """`f(s)` — a média da gaussiana, saída da rede (eq. 4.12).

        No `ContinousActionPolicy.jl` corresponde a `mode(p, observation)`: a
        média é também a moda, e é a ação escolhida em avaliação determinística.
        """
        return self.rede.prever(bits)[0]

    def amostrar(
        self, bits: NDArray[np.bool_], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Ação amostrada de `N(f(s), Σ)` — a densidade da eq. (4.34).

        Sem tratamento de wrap-around, exatamente como o artigo. Isso é
        adequado ao Hopper (torques numa caixa) e **inadequado** ao nosso rumo
        `θ ∈ [0, 2π)`, onde a topologia é circular: é a pendência **C1** do
        projeto, e é problema nosso, não do artigo. A recomendação registrada é
        normal projetada com `AS = 3` (`u₁, u₂` para o rumo, `μ_s` para a
        velocidade).
        """
        mu = self.media(bits)
        return mu + self.desvio * rng.standard_normal(len(mu))

    def grad_log(
        self, bits: NDArray[np.bool_], acao: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Eq. (4.35): `Σ⁻¹ (a − f(s))`.

        Note o **sinal**: é `(a − f(s))`, então uma vantagem positiva empurra a
        média **na direção da ação tomada**. O `Loss.grad` do Julia devolve
        `inv(Σ)·(pred − a)` e o `train!` soma `−grad`, chegando ao mesmo lugar
        por dois sinais invertidos — vale conferir ao comparar os códigos.

        O fator `k(s, ·)` da equação **não** aparece aqui: ele é aplicado por
        `RedeGradienteFuncional.atualizar`, ver o docstring do módulo.
        """
        return self.inv_covariancia * (np.asarray(acao, float) - self.media(bits))
