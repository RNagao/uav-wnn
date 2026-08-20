"""Algoritmos 4.1 e 4.2 da dissertação — SARSA de `m` passos com esquecimento.

Não é o algoritmo que o nosso problema usa: a coleta por enxame tem ação
contínua, e para isso vale o Algoritmo 4.4 (`actor_critic.py`). O 4.1 está aqui
por um motivo específico — é o algoritmo com que o **capítulo 5 da dissertação
produziu resultados publicados**, e sem ele não há como replicar nada. Com ele,
as afirmações de §5.2.1 a §5.2.3 passam de prosa a verificáveis.

Duas escolhas de representação, ambas do §4.2.2:

**A ação indexa a saída, não a entrada.** `Q(s,a) = f(encode(s))[a]`. O texto
oferece duas realizações equivalentes (fig. 4.3): vários discriminadores
escalares, ou um só com posições vetoriais. Usamos a segunda — é o que
`n_saidas` já dá. *"only states require being encoded into a binary pattern"*.

**Contador por componente.** Decorre da anterior: o Algoritmo 4.1 escreve só em
`v[i](S_code, A)`, então cada ação acumula suas próprias visitas. Um contador
compartilhado estragaria a média ponderada de (4.9) para as ações menos
visitadas.

O Algoritmo 4.2 (ε-greedy) tem um detalhe fácil de perder: o desempate entre
ações de valor máximo é **uniforme**, não o primeiro índice. Com a rede zerada
no início, *todas* as ações valem 0, e desempatar pelo índice produziria uma
política inicial degenerada que sempre escolhe a ação 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SarsaNTupla:
    """Algoritmo 4.1, com seleção de ação pelo 4.2.

    O codificador é externo: `codificar(obs) -> (1, B) bool`. Manter a
    binarização fora daqui é o que permite trocar termômetro por circular sem
    tocar no algoritmo, que é exatamente o experimento de §5.2.2.
    """

    rede: object
    """`RedeRegressaoEsquecimento` com `n_saidas = |A|`."""

    codificar: object
    """Função `obs -> bits`, forma `(1, B)`."""

    n_acoes: int
    passos: int
    """`m` — horizonte do retorno com bootstrap."""

    desconto: float
    """`γ`."""

    exploracao: float
    """`ε`."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.passos < 1:
            raise ValueError(f"passos={self.passos} deve ser >= 1")
        if not 0.0 < self.desconto <= 1.0:
            raise ValueError(f"desconto={self.desconto} fora de (0, 1]")
        if not 0.0 <= self.exploracao <= 1.0:
            raise ValueError(f"exploracao={self.exploracao} fora de [0, 1]")
        if self.rede.n_saidas != self.n_acoes:
            raise ValueError(
                f"rede tem {self.rede.n_saidas} saídas mas há {self.n_acoes} ações")

    # -- Algoritmo 4.2 ------------------------------------------------------

    def valores(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """`f(encode(S), ·)` — os valores de ação no estado."""
        return self.rede.prever(bits)[0]

    def escolher(self, bits: NDArray[np.bool_]) -> int:
        """ε-greedy, com desempate **uniforme** entre os máximos."""
        if self.rng.random() < self.exploracao:
            return int(self.rng.integers(self.n_acoes))
        q = self.valores(bits)
        empatados = np.flatnonzero(q == q.max())
        return int(self.rng.choice(empatados))

    # -- Algoritmo 4.1 ------------------------------------------------------

    def episodio(self, env, max_passos: int = 500) -> float:
        """Roda um episódio treinando online. Devolve o retorno não descontado.

        Segue a estrutura do pseudocódigo: `t` avança, `τ = t − m + 1` marca a
        transição a atualizar, e o laço continua até `τ = T − 1` para esvaziar a
        cauda do episódio.
        """
        m, gama = self.passos, self.desconto
        obs, _ = env.reset()
        S = [self.codificar(obs)]
        A = [self.escolher(S[0])]
        R: list[float] = [0.0]                 # R[0] não existe; R[i] = R_i
        T = np.inf
        t, tau, total = 0, -1, 0.0

        while tau < T - 1:
            if t < T:
                obs, r, terminou, truncou, _ = env.step(A[t])
                R.append(float(r))
                total += float(r)
                if terminou or truncou or t + 1 >= max_passos:
                    T = t + 1
                else:
                    S.append(self.codificar(obs))
                    A.append(self.escolher(S[t + 1]))

            tau = t - m + 1
            if tau >= 0:
                fim = int(min(tau + m, T))
                G = sum(gama ** (i - tau - 1) * R[i] for i in range(tau + 1, fim + 1))
                if tau + m < T:                # bootstrap ON-POLICY: usa A_{τ+m}
                    G += gama ** m * float(self.valores(S[tau + m])[A[tau + m]])
                self.rede.treinar_componente(S[tau], A[tau], G)
            t += 1

        return total
