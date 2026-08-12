"""Configuração, semeadura e laço de episódio sobre o ambiente de referência.

Tudo aqui é **externo** ao `environment.py`: configuração, semeadura e coleta
de métricas. Nada modifica o comportamento do MDP.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from .loader import load_environment


@dataclass(frozen=True)
class EnvConfig:
    """Argumentos do `GrADySEnvironment`.

    Os defaults reproduzem a configuração do `benchmark.py` que gerou os
    resultados do §7.2 — e NÃO os defaults do `main.py`, que são de teste.
    Proveniência em `reports/spec00-parametros.md`.
    """

    num_drones: int
    num_sensors: int

    algorithm_iteration_interval: float = 0.5
    """`--algorithm-iteration-interval=0.5` no benchmark.py.

    A tab. 7.1 da dissertação diz 2 s; o código diz 0,5. Ver o relatório.
    """

    scenario_size: float = 100.0
    max_episode_length: float = 500.0
    max_seconds_stalled: int = 30
    communication_range: float = 20.0
    state_num_closest_sensors: int = 12
    """`--state-num-closest-sensors=12`."""

    state_num_closest_drones: int | None = None
    """`None` = `num_drones - 1`, como no benchmark.py."""

    state_mode: str = "relative"
    id_on_state: bool = True
    min_sensor_priority: float = 1.0
    """`--min-sensor-priority=1`."""

    max_sensor_priority: float = 1.0
    full_random_drone_position: bool = False
    reward: str = "punish"
    """`--reward=punish` = eq. 5-6."""

    speed_action: bool = True
    end_when_all_collected: bool = False
    """`--no-end-when-all-collected`: o episódio NÃO termina ao coletar tudo,
    mas a poda por parada segue valendo."""

    def kwargs(self) -> dict[str, Any]:
        d = asdict(self)
        if d["state_num_closest_drones"] is None:
            d["state_num_closest_drones"] = self.num_drones - 1
        return d

    @property
    def observation_size(self) -> int:
        n_a = self.state_num_closest_drones
        if n_a is None:
            n_a = self.num_drones - 1
        return 2 * n_a + 2 * self.state_num_closest_sensors + int(self.id_on_state)


def LAMENZA_003(num_drones: int, num_sensors: int, phase: str = "eval") -> EnvConfig:
    """A configuração publicada, para a célula pedida.

    ATENÇÃO: treino e avaliação usam configurações DIFERENTES no trabalho de
    referência, e a diferença muda a métrica.

    - `phase="train"` — `benchmark.py`: `--no-end-when-all-collected`. O
      episódio continua após a coleta completa e só acaba pela poda por parada.
      É o que dá o "longer duration positive reward signal" de Lamenza §5.
    - `phase="eval"` — `visualize_performance.py`, o script que gerou a
      fig. 7.3: instancia o ambiente **sem** passar `end_when_all_collected`,
      logo vale o default `True`, e o laço quebra na primeira `info` não vazia.
      `completion_time` é então o instante em que tudo foi coletado.
    - `phase="eval_heuristic"` — o MESMO script avalia a heurística gulosa com
      `algorithm_iteration_interval=0.1`, e não 0,5. Ver a ressalva abaixo.

    **Ressalva sobre a comparação publicada.** Em `visualize_performance.py` o
    MADDPG é avaliado a 0,5 s por decisão e o guloso a 0,1 s — cinco vezes mais
    reativo, e com integrador de mobilidade proporcionalmente mais fino. É uma
    variável não controlada na comparação da fig. 7.3. O efeito favorece o
    guloso (mais oportunidades de reapontar), então a vantagem do MADDPG sobre
    ele é conservadora — mas a assimetria existe e precisa ser declarada quando
    citarmos aqueles números.
    """
    if phase == "train":
        return EnvConfig(
            num_drones=num_drones, num_sensors=num_sensors,
            end_when_all_collected=False,
        )
    if phase == "eval":
        return EnvConfig(
            num_drones=num_drones, num_sensors=num_sensors,
            end_when_all_collected=True,
        )
    if phase == "eval_heuristic":
        return EnvConfig(
            num_drones=num_drones, num_sensors=num_sensors,
            end_when_all_collected=True,
            algorithm_iteration_interval=0.1,
        )
    raise ValueError(f"phase inválida: {phase!r}")


class Policy(Protocol):
    """Recebe a observação de um agente e devolve a ação em [0,1]^2."""

    def __call__(self, observation: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass
class EpisodeSummary:
    seed: int
    steps: int
    collected: int
    n_sensors: int
    completion_time: float
    all_collected: bool
    cause: str | None
    reward_sum: float
    wall_time: float
    collection_times: list[float] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.all_collected

    def as_row(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "collected": self.collected,
            "n_sensors": self.n_sensors,
            "completion_time": self.completion_time,
            "success": self.success,
            "cause": self.cause,
            "reward_sum": self.reward_sum,
            "wall_time": self.wall_time,
        }


def prune_logging_handlers(before: list[logging.Handler]) -> None:
    """Remove handlers que o `Simulator` acrescentou ao logger raiz.

    O `gradysim` adiciona um `StreamHandler` ao logger raiz a cada `Simulator`
    construído e nunca o remove. Como não há `reset()`, cada episódio constrói
    um `Simulator` — 10 000 episódios acumulariam 10 000 handlers, com
    degradação progressiva. Verificado por execução.

    Isto não altera o MDP: é higiene de processo.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)


def make_env(config: EnvConfig):
    """Instancia o `GrADySEnvironment` de referência, sem alteração."""
    env_module = load_environment()
    return env_module.GrADySEnvironment(render_mode=None, **config.kwargs())


def run_episode(
    config: EnvConfig,
    policy: Policy,
    seed: int,
    max_steps: int | None = None,
    on_step: Callable[[int, dict, float], None] | None = None,
) -> EpisodeSummary:
    """Roda um episódio e devolve as métricas cruas.

    A semeadura é feita com `random.seed(seed)` imediatamente antes de
    `reset()`, porque o `reset()` de referência sorteia as posições com o
    `random` da stdlib e **ignora** o argumento `seed` que sua assinatura
    declara. É a única forma de tornar o cenário reprodutível sem tocar no
    código de referência.
    """
    handlers_before = list(logging.getLogger().handlers)
    env = make_env(config)

    if max_steps is None:
        max_steps = int(config.max_episode_length / config.algorithm_iteration_interval) + 2

    random.seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset()
    prune_logging_handlers(handlers_before)

    wall0 = time.perf_counter()
    steps = 0
    reward_sum = 0.0
    final_info: dict[str, Any] = {}

    try:
        for _ in range(max_steps):
            actions = {agent: policy(obs[agent]) for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            steps += 1
            r = float(next(iter(rewards.values()))) if rewards else 0.0
            reward_sum += r
            if on_step is not None:
                on_step(steps, obs, r)
            # Condição de parada VERBATIM do `visualize_performance.py`, o
            # script que gerou a fig. 7.3: quebra na PRIMEIRA `info` não vazia.
            # Capturar a última em vez da primeira daria o instante do fim do
            # episódio, não o da conclusão -- com `end_when_all_collected=False`
            # os dois diferem por `max_seconds_stalled`.
            if infos and "avg_reward" in next(iter(infos.values())):
                final_info = next(iter(infos.values()))
                break
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        wall = time.perf_counter() - wall0
        try:
            env.close()
        except Exception:  # pragma: no cover - close do simulador é best-effort
            pass
        prune_logging_handlers(handlers_before)

    return EpisodeSummary(
        seed=seed,
        steps=steps,
        collected=int(env.sensors_collected),
        n_sensors=config.num_sensors,
        completion_time=float(final_info.get("completion_time", config.max_episode_length)),
        all_collected=bool(final_info.get("all_collected", env.sensors_collected == config.num_sensors)),
        cause=final_info.get("cause"),
        reward_sum=reward_sum,
        wall_time=wall,
        collection_times=list(env.collection_times),
    )
