"""Estado verdadeiro do episódio e vistas (SPEC-00, T-00.4).

Quem é dono da verdade: o `EpisodeState`, criado pelo driver. Nem
`SensorProtocol` nem `DroneProtocol` mantêm cópia autoritativa de nada — e
nenhuma métrica mora em `protocol.finish()`, porque quando o episódio para
cedo `finish()` não é chamado (o `_finalize_simulation` do gradysim é privado
e só roda pelas condições internas dele).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .scenario import Scenario


class TerminationReason(str, Enum):
    TIME_LIMIT = "time_limit"
    """t >= max_episode_length."""

    STALLED = "stalled"
    """Sem visita nova por max_seconds_stalled.

    Atenção: com `end_when_all_collected=False` (a configuração publicada),
    este é também o motivo pelo qual um episódio BEM-SUCEDIDO termina — 30 s
    após a última visita. "stalled" não implica fracasso; olhe `n_visited`.
    """

    ALL_COLLECTED = "all_collected"
    """Só ocorre com `end_when_all_collected=True`, que NÃO é a configuração
    publicada."""


class EpisodeState:
    """Verdade do episódio. Mutável, de propriedade do driver."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        n_s = scenario.config.n_sensors
        self.visited: NDArray[np.bool_] = np.zeros(n_s, dtype=bool)
        self.visit_time: NDArray[np.float64] = np.full(n_s, np.nan, dtype=np.float64)
        self.stall_duration: float = 0.0
        self.n_decisions: int = 0
        self.terminated: bool = False
        self.reason: TerminationReason | None = None
        self.completion_time: float | None = None
        self._nodes_agents: list = []
        self._nodes_sensors: list = []
        self._sensor_id_to_index: dict[int, int] = {}

    # -- posições: lidas ao vivo de Node.position ---------------------------

    def agent_positions(self) -> NDArray[np.float64]:
        return np.array([n.position[:2] for n in self._nodes_agents], dtype=np.float64)

    def sensor_positions(self) -> NDArray[np.float64]:
        return np.array([n.position[:2] for n in self._nodes_sensors], dtype=np.float64)

    def visited_mask(self) -> NDArray[np.bool_]:
        return self.visited

    # -- visitação ----------------------------------------------------------

    def mark_visited(self, sensor_node_id: int, t: float) -> bool:
        """Idempotente. Devolve True só na primeira vez."""
        idx = self._sensor_id_to_index[sensor_node_id]
        if self.visited[idx]:
            return False
        self.visited[idx] = True
        self.visit_time[idx] = t
        if bool(self.visited.all()) and self.completion_time is None:
            self.completion_time = t
        return True

    @property
    def n_visited(self) -> int:
        return int(self.visited.sum())

    @property
    def all_collected(self) -> bool:
        return bool(self.visited.all())


class TruthStateView:
    """Vista que entrega a VERDADE.

    É fiel ao original: em Lamenza §3 a política recebe o estado exato através
    de um "canal global" (satélite), e o próprio autor lista isso como
    limitação a endereçar no futuro. É exatamente esse buraco que a SPEC-07
    preenche, substituindo esta vista por `NoisyStateView` sem tocar em
    `build_observation`.
    """

    __slots__ = ("_state",)

    def __init__(self, state: EpisodeState) -> None:
        self._state = state

    def agent_positions(self) -> NDArray[np.float64]:
        return self._state.agent_positions()

    def sensor_positions(self) -> NDArray[np.float64]:
        return self._state.sensor_positions()

    def visited_mask(self) -> NDArray[np.bool_]:
        return self._state.visited_mask()


@dataclass(frozen=True)
class EpisodeResult:
    """Métricas cruas de um episódio. Agregação e IC são da SPEC-02."""

    scenario: Scenario
    seed: int
    episode_index: int
    reason: TerminationReason
    end_time: float
    completion_time: float | None
    """Instante da ÚLTIMA visita quando o episódio completou; None se não
    completou. É a métrica "completion time" da fig. 7.3 de Lamenza.

    O código de referência reporta `max_episode_length` quando não completa
    (`"completion_time": self.max_episode_length if not all_collected else ...`),
    isto é, censura em 500 s. Aqui guardamos `None` e deixamos a política de
    censura para a SPEC-02, que precisa reportar as duas leituras.
    """
    n_visited: int
    n_sensors: int
    visit_times: NDArray[np.float64]
    n_decisions: int
    n_events: int
    wall_time: float
    return_sum: float = 0.0

    @property
    def success(self) -> bool:
        return self.n_visited == self.n_sensors

    @property
    def censored_completion_time(self) -> float:
        """Tempo de conclusão com censura em `max_episode_length`.

        É o que o código de referência reporta. Ver SPEC-02 para a discussão
        sobre viés de sobrevivência.
        """
        if self.completion_time is None:
            return self.scenario.config.max_episode_length
        return self.completion_time

    def as_row(self) -> dict[str, object]:
        cfg = self.scenario.config
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "n_agents": cfg.n_agents,
            "n_sensors": cfg.n_sensors,
            "reason": self.reason.value,
            "end_time": self.end_time,
            "completion_time": self.completion_time,
            "censored_completion_time": self.censored_completion_time,
            "success": self.success,
            "n_visited": self.n_visited,
            "n_decisions": self.n_decisions,
            "n_events": self.n_events,
            "wall_time": self.wall_time,
            "return_sum": self.return_sum,
        }
