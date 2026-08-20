"""Algoritmo 4.4 — n-tuple Actor-Critic. Katopodis (2022), §4.3.4, p.47-48.

Páginas citadas são as **impressas**; some 11 para achar no PDF. O pseudocódigo
está na p. 48 (= p. 59 do PDF).

PSEUDOCÓDIGO ORIGINAL, transcrito

    Entrada: rede de valor  — tupla n_value,  neurônios N_value,  esquecimento φ
             rede de política — tupla n_policy, neurônios N_policy, taxa α
             codificação, número de passos m, desconto γ

    para cada episódio:
        S₀ ← estado inicial
        A₀ ~ f_policy(·|S₀)
        t ← 0;  τ ← −1;  T ← ∞
        enquanto τ < T − 1:
            se t < T:
                executa A_t, observa S_{t+1}, R_{t+1}
                se S_{t+1} é terminal:  T ← t + 1
                senão:                  A_{t+1} ~ f_policy(·|S_{t+1})
            τ ← t − m + 1
            se τ ≥ 0:
                G ← Σ_{i=τ+1}^{min(τ+m, T)} γ^{i−τ−1} R_i
                se τ + m < T:  G ← G + γ^m f_value(encode(S_{τ+m}))
                S_code ← encode(S_τ)
                δ ← G − f_value(S_code)
                para i = 1..N_policy:
                    v[i]_policy(S_code) ← v[i]_policy(S_code)
                                          + α δ ∇_f ln f(A_τ|S_code)
                para j = 1..N_value:
                    v[j]_value(S_code) ← φ v[j]_value(S_code) + G
                    c[j]_value(S_code) ← c[j]_value(S_code) + 1
            t ← t + 1

O FUNCIONAL DE RISCO — eq. (4.36), p.47

    R[f_value, f_policy, S_t, A_t, G_{t:t+m}]
        = (G_{t:t+m} + γ^m f_value(S_{t+m}) − f_value(S_t))
          · ∇_f_policy ln f_policy(·|S_t)

    O termo entre parênteses é a **vantagem**. §4.3.4 explica a diferença em
    relação ao REINFORCE de §4.3.2: "the policy gradient is weighted not by a
    state-value estimate but by the difference between a new state-value estimate
    and the current value approximation given by the model". Reduz variância, e é
    o motivo de existir um crítico.

DOIS PONTOS QUE O PSEUDOCÓDIGO ESCONDE

    **1. A vantagem usa o valor ANTES da atualização.** `δ = G − f_value(S_τ)` é
    calculado com a rede no estado corrente; só depois o crítico treina. Ator e
    crítico enxergam a mesma estimativa daquele estado. A ordem das linhas no
    pseudocódigo garante isso, e o `update!` do `ContinousFunctionalAC.jl`
    confirma (`δ = G - predict(critic, ...)` precede `train!(critic, ...)`).

    **2. O crítico ACUMULA `G`, não regride para `G`.** A linha
    `v ← φ v + G` é a eq. (4.8), e é a média ponderada da (4.9) que converte
    acumulação em estimativa. Não é um passo de gradiente na direção do alvo
    `G` — é por isso que o crítico precisa de contador. Confundir os dois daria
    um crítico que parece funcionar e converge para a coisa errada.

O QUE ESTÁ NO CÓDIGO DO AUTOR E **NÃO** NA DISSERTAÇÃO

    O `ContinousFunctionalAC.jl` recozia linearmente a taxa de aprendizado e a
    covariância (`start_lr → end_lr`, `start_cov → end_cov`). O apêndice A lista
    `α` e `Σ` como **constantes**. Onde o código e o artigo divergem, seguimos o
    artigo: o recozimento existe aqui, desligado por padrão.

    Ressalva ao ler o Julia: o default `end_learning_rate = 1e5` no construtor é
    quase certamente erro de digitação por `1e-5` — taxa final maior que a
    inicial contraria o propósito do recozimento.

O QUE LIMITA `α`, MEDIDO

    Não é a fronteira LMS (`0 < α < 2`), que governa a regressão do crítico. É a
    **variância do gradiente de política**. Medido no bandido contextual, 5
    sementes, 20 000 passos: o pior erro cresce monotonicamente com `α` — 0,044
    em `α = 0,002` contra 0,238 em `α = 0,05`, com uma semente divergindo em
    0,05. Uma ordem de grandeza abaixo da fronteira LMS.

    É a instabilidade que a dissertação nomeia no capítulo 6 (p.70) ao sugerir
    TRPO/PPO: "SGD is a simple and often effective algorithm but can lead to
    unstable policy updates due to its **fixed step size**".

VALIDAÇÃO — e o que falta

    As duas redes são idênticas bit a bit ao `Ramnet.jl` (`tests/test_redes.py`).
    **Este laço não tem oráculo diferencial**: validá-lo exigiria rodar o agente
    Julia inteiro, amarrado aos tipos de ambiente e buffer do `RLWorktable.jl`.
    O que existe são invariantes estruturais um a um mais dois testes de
    comportamento, em `tests/test_actor_critic.py`. Registrado em
    `specs/PLANO-TESTES.md` §5.

CODIFICADOR EXTERNO, DE PROPÓSITO

    Recebemos observação já em bits. A dissertação usa termômetro uniforme
    (eq. 3.23) ou circular (eq. 3.24); a SPEC-01 mediu que **quantil por slot**
    vence no nosso problema (4,3° contra 7,6°), e essa escolha é nossa. Manter a
    binarização fora daqui é o que torna o algoritmo indiferente a ela.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Transicao:
    """Uma tupla `(S_code, A, R)` do buffer de `m` passos.

    Corresponde ao `MultiStepDynamicBuffer` do `RLWorktable.jl`. Guardamos o
    estado **já codificado**, porque o pseudocódigo aplica `encode(S_τ)` uma vez
    e usa o resultado nas duas atualizações.
    """

    bits: NDArray[np.bool_]
    acao: NDArray[np.float64]
    recompensa: float


@dataclass
class AtorCriticoNTupla:
    """Algoritmo 4.4, p.48."""

    politica: object
    """`PoliticaGaussiana` sobre uma `RedeGradienteFuncional` — o `f_policy`."""

    critico: object
    """`RedeRegressaoEsquecimento` de uma saída — o `f_value`.

    Uma saída porque é função **de estado**, `V(s)`, não de estado-ação. O
    Algoritmo 4.1 (SARSA) é que precisa de `|A|` saídas; ver `sarsa.py`.
    """

    passos: int
    """`m` do pseudocódigo — horizonte do retorno com bootstrap.

    O apêndice A usa `m` de 5 (Lunar Lander) a 1000 (Hopper). Com `m` igual ao
    comprimento máximo do episódio o bootstrap nunca ocorre e o retorno vira
    Monte Carlo puro.
    """

    desconto: float
    """`γ`. O apêndice A usa 1,0 em CartPole e 0,9999 em Hopper."""

    taxa: float
    """`α` do pseudocódigo — a taxa do **ator**. O crítico não tem taxa: ele
    acumula pela eq. (4.8)."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    taxa_final: float | None = None
    decaimento_taxa: int = 1
    """Recozimento linear de `α`. **NÃO é da dissertação** — vem do
    `ContinousFunctionalAC.jl`. Desligado por padrão (`taxa_final=None`)."""

    def __post_init__(self) -> None:
        if self.passos < 1:
            raise ValueError(f"passos={self.passos} deve ser >= 1")
        if not 0.0 < self.desconto <= 1.0:
            raise ValueError(f"desconto={self.desconto} fora de (0, 1]")
        self._buffer: list[Transicao] = []
        self._n_updates = 0

    # -- taxa ---------------------------------------------------------------

    @property
    def taxa_corrente(self) -> float:
        """`α` no passo atual. Constante, como o apêndice A, salvo recozimento."""
        if self.taxa_final is None:
            return self.taxa
        frac = min(1.0, self._n_updates / max(1, self.decaimento_taxa))
        return (1 - frac) * self.taxa + frac * self.taxa_final

    # -- interação ----------------------------------------------------------

    def agir(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """`A ~ f_policy(·|S)` — as linhas `A₀ ~ ...` e `A_{t+1} ~ ...`."""
        return self.politica.amostrar(bits, self.rng)

    def registrar(self, bits, acao, recompensa: float) -> None:
        """Empilha `(S_t, A_t, R_{t+1})` no buffer de `m` passos."""
        self._buffer.append(Transicao(np.asarray(bits), np.asarray(acao, float),
                                      float(recompensa)))

    # -- atualização --------------------------------------------------------

    def _retorno(self, bootstrap: float) -> float:
        """As duas linhas de `G` do pseudocódigo, de trás para frente.

            G ← Σ_{i=τ+1}^{min(τ+m,T)} γ^{i−τ−1} R_i
            se τ + m < T:  G ← G + γ^m f_value(encode(S_{τ+m}))

        Acumular ao contrário — `G = R + γG` partindo do bootstrap — dá
        exatamente `R_{τ+1} + γR_{τ+2} + … + γ^{m−1}R_{τ+m} + γ^m V(S_{τ+m})`,
        que é a forma fechada das duas linhas juntas. É o que o `update!` do
        `ContinousFunctionalAC.jl` faz (`for transition in
        Iterators.reverse(buffer)`), e evita calcular potências de `γ`.

        `bootstrap = 0` cobre o caso `τ + m ≥ T`, em que a segunda linha não se
        aplica.
        """
        g = bootstrap
        for t in reversed(self._buffer):
            g = t.recompensa + self.desconto * g
        return g

    def _aplicar(self, t: Transicao, g: float) -> None:
        """As quatro últimas linhas do bloco `se τ ≥ 0`, na ordem do texto."""
        # δ ← G − f_value(S_code). ANTES de treinar o crítico: ver ponto 1 do
        # docstring do módulo.
        valor = float(self.critico.prever(t.bits)[0, 0])
        delta = g - valor

        # v_policy ← v_policy + α δ ∇_f ln f(A_τ|S_code)
        # O ∇ vem da eq. (4.35); o fator k(s,·) dela é aplicado implicitamente
        # por `atualizar`, que escreve nas N posições endereçadas (eq. 4.19).
        passo = self.taxa_corrente * delta * self.politica.grad_log(t.bits, t.acao)
        self.politica.rede.atualizar(t.bits, passo)

        # v_value ← φ v_value + G  e  c_value ← c_value + 1  (eq. 4.8).
        # ACUMULA G, não regride para G: ver ponto 2 do docstring do módulo.
        self.critico.treinar(t.bits, np.array([g]))

        self._n_updates += 1

    def passo_intermediario(self, bits_atual: NDArray[np.bool_]) -> None:
        """O caso `τ ≥ 0` com `τ + m < T`: há bootstrap e o episódio continua.

        Chamar a cada transição. Enquanto o buffer não tem `m` elementos, `τ < 0`
        e nada acontece — é o `se τ ≥ 0` do pseudocódigo. Quando enche, o
        bootstrap sai do estado corrente (`S_{τ+m}`) e **só a transição mais
        antiga** é atualizada e descartada; as demais seguem no buffer para as
        chamadas seguintes.
        """
        if len(self._buffer) < self.passos:
            return
        g = self._retorno(float(self.critico.prever(bits_atual)[0, 0]))
        self._aplicar(self._buffer.pop(0), g)

    def fim_de_episodio(self) -> None:
        """A cauda do episódio: `τ + m ≥ T`, logo sem bootstrap.

        O laço do pseudocódigo continua até `τ = T − 1`, o que esvazia o buffer
        atualizando cada transição restante com o retorno até o fim do episódio.
        Percorremos de trás para frente acumulando `G = R + γG` a partir de zero,
        como o branch `if agent.done` do `ContinousFunctionalAC.jl`.
        """
        g = 0.0
        for t in reversed(self._buffer):
            g = t.recompensa + self.desconto * g
            self._aplicar(t, g)
        self._buffer.clear()

    def reiniciar(self) -> None:
        """Equivalente ao `reset!` do agente: esvazia buffer e as duas redes."""
        self._buffer.clear()
        self.politica.rede.reiniciar()
        self.critico.reiniciar()
        self._n_updates = 0
