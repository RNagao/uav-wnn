"""Política de ação contínua com **normal projetada** — SPEC-05 §4.3 e §4.4.

Resolve a pendência **C1**: a ação de Lamenza é `⟨θ, s⟩` com `θ ∈ [0, 2π)`, e a
gaussiana isotrópica do Katopodis (eq. 4.34) não trata wrap-around. Para o Hopper
— torques numa caixa — isso basta; para um rumo, não.

A IDEIA, E POR QUE ELA RESOLVE

O ator emite `d = 3` componentes, `f = (u₁, u₂, μ_s)`. Amostramos em **R³**:

    z ~ N(f, σ² I₃)

e derivamos a ação por um mapa determinístico:

    θ = atan2(z₂, z₁)          rumo, sem descontinuidade
    s = clip(z₃, 0, 1)         velocidade normalizada

**O gradiente vive em R³, onde não existe wrap.** `∇_f log π(z|o) = (z − f)/σ²`
é a pontuação gaussiana comum, e a topologia circular fica inteiramente no mapa
`atan2`, que é liso e sobrejetivo no círculo. Nenhuma correção de fase, nenhum
caso especial em `θ = 0`.

`‖f₁:₂‖` controla a **concentração** do rumo: norma grande concentra, norma perto
de zero dá rumo quase uniforme. É o análogo do parâmetro de concentração da von
Mises, e sai de graça — não é hiperparâmetro.

    Isso vale para OS DOIS aproximadores. Consequência a registrar: a MLP pareada
    é `D → 32 → 32 → 3`, e não `→ 2` como diz a tabela do PRD §5.5. São +33
    parâmetros — desprezível no total, mas o número publicado tem de ser o certo.

AMOSTRADO ≠ EXECUTADO — a regra que evita o bug mais provável desta spec

> A atualização usa **`z` bruto**, o que a política de fato amostrou. O ambiente
> recebe `(θ, s)` derivado e **recortado**. Nunca o contrário.

Recortar `z₃` e usar o valor recortado no resíduo `(z − f)` viesa o gradiente e
mata o sinal justamente onde a política pede velocidade máxima — que é a região
onde ela vai passar a maior parte do tempo. `amostrar` devolve os dois
explicitamente, em vez de deixar o chamador reconstruir.

`σ_s`: FIXA E ANELADA, NÃO APRENDIDA — §4.4

Quatro razões, e a primeira é a que invalida copiar os números do artigo:

1. **A grandeza transferível é `σ/faixa`, não `σ`.** Katopodis usa `σ² = 0,5` no
   Mountain Car, cuja ação vive em `[−1, 1]` — ou seja `σ ≈ 0,71`, uns 35 % da
   faixa. A nossa ação normalizada vive em `[0, 1]`: o mesmo `σ` produziria
   maioria de amostras recortadas e uma política efetivamente de dois níveis,
   parado ou máximo.
2. **A recompensa penaliza tempo** (`R_t = (Σ visited − Ns)/Ns`), e explorar na
   velocidade interage com `max_seconds_stalled = 30 s`: 15 decisões sem coleta e
   o episódio é podado. Exploração excessiva encurta o episódio e reduz a
   experiência útil.
3. **Aprender `σ` diverge no aproximador errado.** Uma cabeça `log σ` numa rede
   n-tuple herda o modo de falha: onde a vantagem é consistentemente positiva,
   `σ → 0`, o fator `σ⁻²` explode.
4. **Custo de modelo:** `d = 3 → 4` é +33 % na carga de cada posição de memória
   do ator, e ataca T1 diretamente.

**Avaliação é determinística**, e isto não é escolha nossa — é o protocolo de
Lamenza §7 (*"during an evaluation the training is temporarily halted and action
noise is removed"*): usa-se a média, `θ = atan2(f₂, f₁)` e `s = clip(f₃, 0, 1)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base import Acao, V_MAX

D_ATOR = 3
"""`(u₁, u₂, μ_s)`. Vale para os dois aproximadores."""


@dataclass
class NormalProjetada:
    """`z ~ N(f, σ²I₃)` em R³, ação derivada por `atan2` e `clip`.

    `sigma` é fração da faixa da ação, que é unitária depois da normalização —
    então `sigma = 0.1` significa 10 % da faixa. Ver §4.4 sobre por que os
    valores do artigo não são transferíveis.
    """

    sigma: float = 0.10
    sigma_min: float | None = None
    decaimento: int = 1
    """Anelamento linear de `sigma` a `sigma_min` em `decaimento` atualizações.

    Com `σ` fixa a política ótima determinística é inatingível e sobra ruído
    residual; a avaliação usa a média, mas o **treino** precisa convergir. Deixar
    `sigma_min = None` desliga.
    """

    _n: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma={self.sigma} deve ser > 0")
        if self.sigma_min is not None and self.sigma_min <= 0:
            raise ValueError(f"sigma_min={self.sigma_min} deve ser > 0")

    # -- anelamento ---------------------------------------------------------

    @property
    def sigma_corrente(self) -> float:
        if self.sigma_min is None:
            return self.sigma
        frac = min(1.0, self._n / max(1, self.decaimento))
        return (1 - frac) * self.sigma + frac * self.sigma_min

    def contar_atualizacao(self) -> None:
        self._n += 1

    def reiniciar(self) -> None:
        self._n = 0

    # -- o mapa determinístico ---------------------------------------------

    @staticmethod
    def para_acao(z: NDArray[np.float64]) -> Acao:
        """`z ∈ R³ → (θ, s)`. É aqui que a topologia circular é resolvida.

        `atan2` é liso e sobrejetivo no círculo, então não há descontinuidade em
        `θ = 0` e nenhuma correção de fase é necessária. O recorte de `z₃` é
        **só** para o ambiente — a atualização usa `z` bruto.
        """
        z = np.asarray(z, float).ravel()
        if z.size != D_ATOR:
            raise ValueError(f"z tem {z.size} componentes, esperado {D_ATOR}")
        theta = float(np.arctan2(z[1], z[0]))
        s = float(np.clip(z[2], 0.0, 1.0))
        return Acao.de_rumo(theta, s * V_MAX)

    def media(self, f: NDArray[np.float64]) -> Acao:
        """Ação determinística: a média, sem ruído. **É o modo de avaliação.**"""
        return self.para_acao(f)

    def amostrar(
        self, f: NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[Acao, NDArray[np.float64]]:
        """Devolve `(acao_executada, z_bruto)` — os dois, de propósito.

        Separar é o que impede o bug de usar o valor recortado no resíduo. Quem
        chamar recebe as duas coisas e não tem como confundi-las.
        """
        f = np.asarray(f, float).ravel()
        z = f + self.sigma_corrente * rng.standard_normal(D_ATOR)
        return self.para_acao(z), z

    # -- gradiente ----------------------------------------------------------

    def grad_log(
        self, f: NDArray[np.float64], z: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """`∇_f log π(z|o) = (z − f)/σ²`.

        A pontuação gaussiana comum, em R³. **`z` é o bruto** — passar a ação
        recortada aqui é o bug que a §4.3 nomeia.
        """
        f = np.asarray(f, float).ravel()
        z = np.asarray(z, float).ravel()
        s = self.sigma_corrente
        return (z - f) / (s * s)

    # -- diagnóstico obrigatório -------------------------------------------

    @staticmethod
    def concentracao(f: NDArray[np.float64]) -> float:
        """`‖f₁:₂‖` — o quanto o rumo está concentrado.

        **Diagnóstico obrigatório** da §4.3: o histograma disto ao longo do
        treino é o que distingue *"não aprendeu"* de *"colapsou para
        determinístico cedo demais"* — dois diagnósticos com correções opostas.
        Norma perto de zero significa rumo quase uniforme, mesmo com `σ` pequena.
        """
        f = np.asarray(f, float).ravel()
        return float(np.hypot(f[0], f[1]))


@dataclass
class GaussianaSimples:
    """**C0** — gaussiana direta sobre `(θ_norm, s_norm)`, com recorte.

    É o esquema ingênuo, e existe para ser **comparado e batido**. Só faz sentido
    manter no repositório porque o critério de saída da SPEC-05 exige que a
    normal projetada o supere com IC separado: *"se C2 não vencer, a §4.3 estava
    errada e a decisão precisa ser revista antes da SPEC-06 — não depois"*.

    O MODO DE FALHA, que é específico e mensurável

    Uma política que quer rumo perto de `θ_norm = 0` (ou de 1) tem **metade da
    massa de probabilidade recortada** na fronteira. Ela não consegue expressar
    "aponte para leste" sem perder metade das amostras, e o gradiente ali fica
    viesado para dentro do intervalo — empurrando o rumo para longe da fronteira
    em vez de para o alvo.

    Com alvos uniformes no círculo, uma fração fixa dos cenários cai exatamente
    nessa região. É o que o brinquedo de `toy/nav.py` mede.

    Note que `d = 2` aqui, contra `d = 3` da normal projetada. A comparação
    portanto **favorece o C0 em custo** — uma cabeça de menos por posição de
    memória —, o que torna uma vitória do C2 mais forte, não mais fraca.
    """

    sigma: float = 0.10
    sigma_min: float | None = None
    decaimento: int = 1
    _n: int = field(default=0, init=False)

    D = 2

    @property
    def sigma_corrente(self) -> float:
        if self.sigma_min is None:
            return self.sigma
        frac = min(1.0, self._n / max(1, self.decaimento))
        return (1 - frac) * self.sigma + frac * self.sigma_min

    def contar_atualizacao(self) -> None:
        self._n += 1

    @staticmethod
    def para_acao(z: NDArray[np.float64]) -> Acao:
        """Recorta as duas componentes em `[0,1]`. **É aqui que o C0 perde.**"""
        z = np.asarray(z, float).ravel()
        return Acao(theta_norm=float(np.clip(z[0], 0.0, 1.0)),
                    s_norm=float(np.clip(z[1], 0.0, 1.0)))

    def media(self, f: NDArray[np.float64]) -> Acao:
        return self.para_acao(f)

    def amostrar(self, f, rng: np.random.Generator):
        f = np.asarray(f, float).ravel()
        z = f + self.sigma_corrente * rng.standard_normal(self.D)
        return self.para_acao(z), z

    def grad_log(self, f, z) -> NDArray[np.float64]:
        f = np.asarray(f, float).ravel()
        z = np.asarray(z, float).ravel()
        s = self.sigma_corrente
        return (z - f) / (s * s)

    @staticmethod
    def concentracao(f: NDArray[np.float64]) -> float:
        """Não há noção de concentração aqui — `σ` é o único controle. Devolve
        `nan` para o diagnóstico da §4.3 não fingir que existe."""
        return float("nan")
