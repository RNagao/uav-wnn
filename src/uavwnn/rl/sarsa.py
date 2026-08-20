"""Algoritmos 4.1 e 4.2 — SARSA de `m` passos com esquecimento.

Katopodis (2022), §4.2.2, p.35-37. Páginas impressas; some 11 para o PDF.

POR QUE ESTE ALGORITMO ESTÁ NO REPOSITÓRIO

    Não é o algoritmo do nosso problema: a coleta por enxame tem ação contínua, e
    para isso vale o Algoritmo 4.4 (`actor_critic.py`). O 4.1 está aqui porque é
    com ele que o **capítulo 5 da dissertação produziu resultados publicados** —
    figs. 5.3 a 5.6, tabelas 5.1 a 5.7. Sem ele não há como replicar nada.

    Replicação de §5.2.2 em `reports/replica-katopodis-52.md`.

PSEUDOCÓDIGO — Algoritmo 4.1, p.37

    Entrada: tupla n, neurônios N, codificação, esquecimento φ,
             número de passos m, exploração ε, desconto γ

    para cada episódio:
        S₀ ← estado inicial
        A₀ ~ π(·|S₀)                      (Algoritmo 4.2)
        t ← 0;  τ ← −1;  T ← ∞
        enquanto τ < T − 1:
            se t < T:
                executa A_t, observa S_{t+1}, R_{t+1}
                se S_{t+1} é terminal:  T ← t + 1
                senão:                  A_{t+1} ~ π(·|S_{t+1})
            τ ← t − m + 1
            se τ ≥ 0:
                G ← Σ_{i=τ+1}^{min(τ+m,T)} γ^{i−τ−1} R_i
                se τ + m < T:  G ← G + γ^m f(encode(S_{τ+m}), A_{τ+m})
                S_code ← encode(S_τ)
                para i = 1..N:
                    v[i](S_code, A_τ) ← φ v[i](S_code, A_τ) + G
                    c[i](S_code, A_τ) ← c[i](S_code, A_τ) + 1
            t ← t + 1

    Algoritmo 4.2, p.37 — seleção de ação:

        se Uniform(0,1) < ε:  A ← amostra uniforme de A
        senão:                A ← amostra **uniforme** de argmax_a f(encode(S), a)

COMO A FUNÇÃO DE VALOR-AÇÃO É REPRESENTADA — §4.2.2, p.35

    O problema: `Q(s,a)` tem dois argumentos, mas a rede aceita um só padrão
    binário. O texto considera codificar estado e ação juntos, e descarta:

        "a much simpler approach is handling state and action separately. The
         separation could be understood as modeling a state-value function for
         each possible action."

    A fig. 4.3 (p.36) mostra as **duas realizações equivalentes**:

        (a) UM discriminador com posições de memória **vetoriais**
        (b) VÁRIOS discriminadores escalares, um por componente

    Usamos (a) — é o que `n_saidas` já dá. Logo `Q(s,a) = f(encode(s))[a]`, e
    **só o estado é codificado**: "only states require being encoded into a
    binary pattern".

    Consequência que não é detalhe: a atualização escreve em `v[i](S_code, A)`,
    **só a componente da ação tomada**. Daí o contador ser por componente no
    `RedeRegressaoEsquecimento` — um contador compartilhado degradaria a média
    ponderada da eq. (4.9) para as ações menos visitadas. É também por isso que o
    `RegressionNodeMemoryElement{D}` do `Ramnet.jl` tem `count::Vector{Int}` de
    comprimento `D` em vez de um `Int`.

O DETALHE DO ALGORITMO 4.2 QUE É FÁCIL PERDER

    "Sample **uniformly** from arg max". Com a rede zerada no início, **todas** as
    ações valem 0 e o argmax empata em todas. Desempatar pelo índice produziria
    uma política inicial degenerada, sempre a ação 0 — e com `ε = 0,005`, que é o
    valor do quadro de §5.2.3, o agente levaria muito tempo para sair disso.

SARSA, NÃO Q-LEARNING

    O bootstrap usa `f(S_{τ+m}, A_{τ+m})`: o valor da **ação efetivamente
    tomada**, não `max_a`. É o que faz deste um método on-policy — §4.2.2
    situa-o em "generalized policy improvement", com a política definida
    implicitamente pela função de valor-ação.

HIPERPARÂMETROS DO ARTIGO, para referência

    Do quadro de §5.2.3 (p.58), CartPole: `m = 500`, resolução 128, tupla 64,
    `γ = 1,0`, `ε = 0,005`, codificação circular, `φ = 0,85`.
    Note `B = 4 × 128 = 512` e portanto `N = ⌈512/64⌉ = 8` neurônios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SarsaNTupla:
    """Algoritmo 4.1, com seleção de ação pelo Algoritmo 4.2."""

    rede: object
    """`RedeRegressaoEsquecimento` com `n_saidas = |A|` — a fig. 4.3(a)."""

    codificar: object
    """`obs -> bits` de forma `(1, B)`.

    Externo de propósito: é o que permite trocar termômetro (3.23) por circular
    (3.24) sem tocar no algoritmo, que é exatamente o experimento de §5.2.2.
    """

    n_acoes: int
    passos: int
    """`m` do pseudocódigo. O apêndice A usa 500 em CartPole, 50 em Lunar Lander."""

    desconto: float
    """`γ`."""

    exploracao: float
    """`ε` do Algoritmo 4.2. O quadro de §5.2.3 usa 0,005 em CartPole."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.passos < 1:
            raise ValueError(f"passos={self.passos} deve ser >= 1")
        if not 0.0 < self.desconto <= 1.0:
            raise ValueError(f"desconto={self.desconto} fora de (0, 1]")
        if not 0.0 <= self.exploracao <= 1.0:
            raise ValueError(f"exploracao={self.exploracao} fora de [0, 1]")
        # A representação da fig. 4.3(a) exige uma componente de saída por ação.
        if self.rede.n_saidas != self.n_acoes:
            raise ValueError(
                f"rede tem {self.rede.n_saidas} saídas mas há {self.n_acoes} ações")

    # -- Algoritmo 4.2 ------------------------------------------------------

    def valores(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """`f(encode(S), ·)` — o vetor de valores de ação no estado.

        Uma única passada pela rede devolve todas as ações, porque a saída é
        vetorial (fig. 4.3a). Com a realização (b) seriam `|A|` passadas.
        """
        return self.rede.prever(bits)[0]

    def escolher(self, bits: NDArray[np.bool_]) -> int:
        """Algoritmo 4.2: ε-greedy com desempate **uniforme** entre os máximos.

        Ver o docstring do módulo sobre por que o desempate importa.
        """
        if self.rng.random() < self.exploracao:
            return int(self.rng.integers(self.n_acoes))
        q = self.valores(bits)
        empatados = np.flatnonzero(q == q.max())
        return int(self.rng.choice(empatados))

    # -- Algoritmo 4.1 ------------------------------------------------------

    def episodio(self, env, max_passos: int = 500) -> float:
        """Um episódio, treinando online. Devolve o retorno não descontado.

        Segue a estrutura do pseudocódigo literalmente: `t` avança, `τ = t−m+1`
        marca a transição a atualizar, e o laço continua até `τ = T−1` para
        esvaziar a cauda do episódio.

        Índices seguem o texto: `S[t]`, `A[t]` e `R[i]` com `R[0]` inexistente
        (a primeira recompensa é `R_1`, recebida depois de `A_0`). O placeholder
        em `R[0]` mantém o alinhamento com a soma
        `Σ_{i=τ+1}^{min(τ+m,T)} γ^{i−τ−1} R_i`.

        Listas em vez de buffer circular porque os episódios das tarefas do
        capítulo 5 têm no máximo 500 passos, e a clareza vale mais aqui.
        """
        m, gama = self.passos, self.desconto
        obs, _ = env.reset()
        S = [self.codificar(obs)]          # S[0] ← estado inicial, codificado
        A = [self.escolher(S[0])]          # A₀ ~ π(·|S₀)
        R: list[float] = [0.0]             # R[0] não existe; R[i] = R_i
        T = np.inf
        t, tau, total = 0, -1, 0.0

        while tau < T - 1:
            if t < T:
                # executa A_t, observa S_{t+1}, R_{t+1}
                obs, r, terminou, truncou, _ = env.step(A[t])
                R.append(float(r))
                total += float(r)
                if terminou or truncou or t + 1 >= max_passos:
                    T = t + 1              # S_{t+1} é terminal
                else:
                    S.append(self.codificar(obs))
                    A.append(self.escolher(S[t + 1]))   # A_{t+1} ~ π(·|S_{t+1})

            tau = t - m + 1
            if tau >= 0:
                # G ← Σ_{i=τ+1}^{min(τ+m,T)} γ^{i−τ−1} R_i
                fim = int(min(tau + m, T))
                G = sum(gama ** (i - tau - 1) * R[i] for i in range(tau + 1, fim + 1))
                if tau + m < T:
                    # G ← G + γ^m f(encode(S_{τ+m}), A_{τ+m})
                    # SARSA: a ação TOMADA, não o argmax. Ver docstring do módulo.
                    G += gama ** m * float(self.valores(S[tau + m])[A[tau + m]])
                # v[i](S_code, A_τ) ← φ v[i](S_code, A_τ) + G   (eq. 4.8)
                # c[i](S_code, A_τ) ← c[i](S_code, A_τ) + 1
                self.rede.treinar_componente(S[tau], A[tau], G)
            t += 1

        return total
