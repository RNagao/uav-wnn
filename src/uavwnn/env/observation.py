"""Observação de Lamenza, eq. 5-1 (SPEC-00, T-00.5). FONTE ÚNICA.

Esta é a única implementação do vetor de observação no projeto. O simulador
(SPEC-00) e o gerador analítico (SPEC-01) chamam esta mesma função. Se
houvesse duas implementações, o teste KS do invariante V-1 mediria a diferença
entre dois códigos em vez da diferença entre duas dinâmicas — que é o que se
quer medir.

Referência de mecanismo: `observe_simulation_relative_positions` em
`scenario/003-collaboration/environment.py` (linhas 331-375) do repositório
`Project-GrADyS/gradys-data-collection@dissertation`.

Três propriedades do original que NÃO estão na eq. 5-1 do texto e que só
aparecem no código — e que mudam completamente o desenho do codificador
binário da SPEC-01:

1. O vetor é NORMALIZADO: `(p_k - p_i + S) / (2S)`, com S = scenario_size.
   Valores válidos caem em [0, 1] enquanto |p_k - p_i| <= S.
2. O padding é `-1`, NÃO `(0, 0)`. Como -1 está fora da faixa dos valores
   válidos, padding e "sensor exatamente sobre o UAV" são distinguíveis por
   construção — o bit de validade que a SPEC-00 §5.4 propunha é desnecessário.
3. O índice do agente entra como `k / Na`, não como `k` cru.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

PADDING_VALUE = -1.0
"""Sentinela de slot vazio. Fora de [0, 1], logo não colide com valor válido."""


@runtime_checkable
class StateView(Protocol):
    """Vista do estado entregue ao construtor de observação.

    Este é o ponto de inserção da SPEC-07 (RF-04). Na Etapa 1,
    `TruthStateView` devolve a verdade — que é fiel ao "canal global" de
    Lamenza §3. Na Etapa 2, `NoisyStateView` entra no lugar e a verdade
    permanece no `EpisodeState`, exclusivamente para métricas.

    `build_observation` recebe esta interface e NUNCA o `EpisodeState`, de modo
    que "a verdade não vaza para a política" é verificável pela assinatura.
    """

    def agent_positions(self) -> NDArray[np.float64]:
        """(Na, 2)."""
        ...

    def sensor_positions(self) -> NDArray[np.float64]:
        """(Ns, 2)."""
        ...

    def visited_mask(self) -> NDArray[np.bool_]:
        """(Ns,) — True para sensores já coletados."""
        ...


def build_observation(
    view: StateView,
    agent_index: int,
    n_star_a: int,
    n_star_s: int,
    scenario_size: float,
    id_on_state: bool = True,
) -> NDArray[np.float32]:
    """Constrói a observação do agente `agent_index`.

    Forma: `(2*n_star_a + 2*n_star_s + int(id_on_state),)`.

    A função é PURA: mesma vista, mesma saída, bit a bit. Sem estado, sem RNG,
    sem relógio.
    """
    agents = np.asarray(view.agent_positions(), dtype=np.float64)
    sensors = np.asarray(view.sensor_positions(), dtype=np.float64)
    visited = np.asarray(view.visited_mask(), dtype=bool)

    n_agents = agents.shape[0]
    me = agents[agent_index]

    # --- Sensores não visitados, ordenados por distância crescente ---------
    unvisited = sensors[~visited]
    closest_sensors = np.full((n_star_s, 2), PADDING_VALUE, dtype=np.float64)
    n_fill_s = min(n_star_s, unvisited.shape[0])
    if n_fill_s > 0:
        d = np.linalg.norm(unvisited - me, axis=1)
        # `stable` em vez do quicksort do original: empates são de medida nula
        # com posições contínuas, mas estabilidade dá determinismo de graça.
        order = np.argsort(d, kind="stable")[:n_fill_s]
        closest_sensors[:n_fill_s] = _normalize(me, unvisited[order], scenario_size)

    # --- Demais agentes, ordenados por distância crescente -----------------
    closest_agents = np.full((n_star_a, 2), PADDING_VALUE, dtype=np.float64)
    n_fill_a = min(n_star_a, n_agents - 1)
    if n_fill_a > 0:
        d = np.linalg.norm(agents - me, axis=1)
        # O índice 0 da ordenação é o próprio agente (distância 0); descartado.
        order = np.argsort(d, kind="stable")[1 : n_fill_a + 1]
        closest_agents[:n_fill_a] = _normalize(me, agents[order], scenario_size)

    parts = [closest_agents.reshape(-1), closest_sensors.reshape(-1)]
    if id_on_state:
        parts.append(np.array([agent_index / n_agents], dtype=np.float64))

    return np.concatenate(parts).astype(np.float32)


def _normalize(
    me: NDArray[np.float64],
    others: NDArray[np.float64],
    scenario_size: float,
) -> NDArray[np.float64]:
    """`(p_k - p_i + S) / (2S)`.

    Atenção ao SINAL: é a posição do agente MENOS a do outro (eq. 5-2 e 5-5),
    isto é, o vetor que aponta do outro PARA o agente. É o contrário da
    convenção intuitiva "vetor até o alvo" e é o erro mais fácil de cometer
    neste projeto.

    Nota sobre a faixa: `p_k - p_i` vive em [-2S, 2S], então a saída vive em
    [-0.5, 1.5] e não em [0, 1]. O código de referência declara
    `Box(-1, 1)` para este modo de estado, o que também não bate. Reproduzido
    como está; a SPEC-01 precisa binarizar a faixa REAL, não a declarada.
    """
    return (me - others + scenario_size) / (2.0 * scenario_size)


def nearest_unvisited_direction(obs: NDArray[np.float32], n_star_a: int) -> float | None:
    """Rumo (rad) para o sensor não visitado mais próximo, lido da observação.

    Usado pela `GreedyPolicy` (SPEC-00) e pela clonagem da SPEC-01. Lê apenas o
    vetor de observação — nunca a verdade —, então um erro de sinal em
    `_normalize` aparece como um guloso que FOGE dos sensores: falha ruidosa,
    não silenciosa.

    Devolve `None` se todos os slots de sensor estiverem vazios.
    """
    block = obs[2 * n_star_a :]
    if block.size % 2 == 1:  # descarta o id do agente, se presente
        block = block[:-1]
    rel = block.reshape(-1, 2).astype(np.float64)

    # Slots vazios continuam em PADDING_VALUE em AMBAS as componentes.
    valid = ~np.all(np.isclose(rel, PADDING_VALUE), axis=1)
    if not valid.any():
        return None
    rel = rel[valid]

    # Desfaz a normalização: `(me - other + S)/(2S)` -> `(me - other)/S`.
    delta = rel * 2.0 - 1.0
    idx = int(np.argmin(np.linalg.norm(delta, axis=1)))
    # `delta` aponta do sensor para o agente; o rumo desejado é o oposto.
    return float(np.arctan2(-delta[idx, 1], -delta[idx, 0]))
