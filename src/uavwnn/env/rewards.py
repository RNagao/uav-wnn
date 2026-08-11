"""Recompensas de Lamenza, verbatim (SPEC-00, T-00.7).

Sem shaping potencial, sem penalidade de borda, sem termo de coordenação
(PRD D3). Qualquer acréscimo aqui exige linha nova na tabela de desvios do
PRD §5.4.

Referência de mecanismo: bloco `# Calculating reward` em
`scenario/003-collaboration/environment.py` (linhas 626-644). O modo usado nos
experimentos publicados é `--reward=punish`, que é exatamente a eq. 5-6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reward_punish(
    visited_now: NDArray[np.bool_],
    visited_prev: NDArray[np.bool_],
) -> float:
    """Lamenza eq. 5-6 (`reward='punish'` no código de referência).

        R_t = 10 * Δvisited            , se Δvisited > 0
              (Σ visited - Ns) / Ns    , caso contrário

    RECOMPENSA DE EQUIPE: a fórmula não tem índice de agente, e o código de
    referência distribui o mesmo escalar a todos (`rewards = {agent: reward
    for agent in self.agents}`). Não é escolha nossa e não é negociável sem
    linha nova na §5.4 do PRD.

    Nota que explica a regra de não poda: quando tudo está visitado, Δ = 0 e o
    ramo "caso contrário" vale exatamente 0, em vez de negativo. Continuar o
    episódio após a coleta completa entrega, portanto, uma sequência de
    recompensas 0 que domina o retorno de qualquer estado incompleto (sempre
    negativo). É consistente com a explicação de Lamenza §5 sobre o "longer
    duration positive reward signal" — leitura nossa, não afirmação do autor.
    """
    n_s = int(visited_now.size)
    after = int(visited_now.sum())
    before = int(visited_prev.sum())
    delta = after - before
    if delta > 0:
        return 10.0 * delta
    return -(n_s - after) / n_s


def reward_exploratory(visited_now: NDArray[np.bool_]) -> float:
    """Lamenza eq. 7-2. Usada SOMENTE na réplica do experimento exploratório
    (§7.1), onde a métrica é taxa de sucesso e não tempo de conclusão.

        R_t = Σ visited / Ns
    """
    return float(visited_now.sum()) / float(visited_now.size)
