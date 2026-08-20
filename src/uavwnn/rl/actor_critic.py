"""Algoritmo 4.4 da dissertação — n-tuple Actor-Critic.

Transcrição do pseudocódigo da página 48, com os nomes da dissertação:

    Entrada: rede de valor (tupla n_value, N_value, esquecimento φ)
             rede de política (tupla n_policy, N_policy, taxa α)
             codificação, número de passos m, desconto γ

    para cada episódio:
        S₀ ← estado inicial;  A₀ ~ f_policy(·|S₀);  t ← 0;  τ ← −1;  T ← ∞
        enquanto τ < T − 1:
            se t < T:
                executa A_t, observa S_{t+1}, R_{t+1}
                se S_{t+1} terminal:  T ← t + 1
                senão:                A_{t+1} ~ f_policy(·|S_{t+1})
            τ ← t − m + 1
            se τ ≥ 0:
                G ← Σ_{i=τ+1}^{min(τ+m, T)} γ^{i−τ−1} R_i
                se τ + m < T:  G ← G + γ^m f_value(encode(S_{τ+m}))
                S_code ← encode(S_τ)
                δ ← G − f_value(S_code)
                v_policy(S_code) ← v_policy(S_code) + α δ ∇_f ln f(A_τ|S_code)
                v_value(S_code)  ← φ v_value(S_code) + G
                c_value(S_code)  ← c_value(S_code) + 1
            t ← t + 1

Dois pontos que o pseudocódigo esconde e que só o código do autor esclarece:

**A vantagem usa o valor ANTES da atualização.** `δ = G − f_value(S_τ)` é
calculado com a rede no estado corrente, e só depois o crítico é treinado. Ator
e crítico enxergam a mesma estimativa.

**O crítico não é regredido para G — ele ACUMULA G.** A eq. (4.8) faz
`v ← φ v + G`, e a média ponderada de (4.9) é quem converte acumulação em
estimativa. É diferente de um passo de gradiente para o alvo `G`, e é a razão
de o crítico precisar de contador.

O recozimento da taxa de aprendizado e da covariância que aparece no
`ContinousFunctionalAC.jl` **não está na dissertação** — lá α e Σ são constantes
(apêndice A). Ficam disponíveis, desligados por padrão.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Transicao:
    """Uma tupla `(S, A, R)` do buffer de `m` passos."""

    bits: NDArray[np.bool_]
    acao: NDArray[np.float64]
    recompensa: float


@dataclass
class AtorCriticoNTupla:
    """Algoritmo 4.4. O codificador é externo: recebemos observação já em bits.

    Manter a binarização fora daqui é deliberado — a SPEC-01 mediu que o
    termômetro quantil por slot vence o linear global por 4,3° contra 7,6°, e
    essa escolha é nossa, não da dissertação.
    """

    politica: object
    """`PoliticaGaussiana` sobre uma `RedeGradienteFuncional`."""

    critico: object
    """`RedeRegressaoEsquecimento` de uma saída."""

    passos: int
    """`m` — horizonte do retorno com bootstrap."""

    desconto: float
    """`γ`."""

    taxa: float
    """`α` — taxa de aprendizado do ator."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    taxa_final: float | None = None
    decaimento_taxa: int = 1
    """Recozimento linear de α. NÃO é da dissertação; veja o docstring do módulo."""

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
        if self.taxa_final is None:
            return self.taxa
        frac = min(1.0, self._n_updates / max(1, self.decaimento_taxa))
        return (1 - frac) * self.taxa + frac * self.taxa_final

    # -- interação ----------------------------------------------------------

    def agir(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Amostra `A ~ f_policy(·|S)`."""
        return self.politica.amostrar(bits, self.rng)

    def registrar(self, bits, acao, recompensa: float) -> None:
        self._buffer.append(Transicao(np.asarray(bits), np.asarray(acao, float),
                                      float(recompensa)))

    # -- atualização --------------------------------------------------------

    def _retorno(self, bootstrap: float) -> float:
        """`G` acumulado de trás para frente sobre o buffer inteiro.

        `G = R_{τ+1} + γR_{τ+2} + … + γ^{m−1}R_{τ+m} + γ^m V(S_{τ+m})`, que é a
        forma fechada das duas linhas de `G` no pseudocódigo.
        """
        g = bootstrap
        for t in reversed(self._buffer):
            g = t.recompensa + self.desconto * g
        return g

    def _aplicar(self, t: Transicao, g: float) -> None:
        valor = float(self.critico.prever(t.bits)[0, 0])
        delta = g - valor                                   # vantagem, ANTES do treino
        passo = self.taxa_corrente * delta * self.politica.grad_log(t.bits, t.acao)
        self.politica.rede.atualizar(t.bits, passo)
        self.critico.treinar(t.bits, np.array([g]))         # eq. (4.8): v ← φv + G
        self._n_updates += 1

    def passo_intermediario(self, bits_atual: NDArray[np.bool_]) -> None:
        """Chamar a cada transição: consome uma quando o buffer enche.

        O bootstrap sai do estado corrente, `S_{τ+m}`, e só a transição mais
        antiga é atualizada e descartada — as demais continuam no buffer.
        """
        if len(self._buffer) < self.passos:
            return
        g = self._retorno(float(self.critico.prever(bits_atual)[0, 0]))
        self._aplicar(self._buffer.pop(0), g)

    def fim_de_episodio(self) -> None:
        """Esvazia o buffer sem bootstrap — `τ + m ≥ T`, nada a estimar."""
        g = 0.0
        for t in reversed(self._buffer):
            g = t.recompensa + self.desconto * g
            self._aplicar(t, g)
        self._buffer.clear()

    def reiniciar(self) -> None:
        self._buffer.clear()
        self.politica.rede.reiniciar()
        self.critico.reiniciar()
        self._n_updates = 0
