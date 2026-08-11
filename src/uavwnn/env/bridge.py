"""Ponte simulador <-> treinador e driver de episódio (SPEC-00, T-00.9).

O gradysim é orientado a EVENTOS: `step_simulation()` processa um evento, não
um instante. O RL quer `obs -> ação`. A conciliação é a injeção de política: a
política é chamada de forma síncrona pelo driver, sem corrotinas e sem
inversão de controle, o que é válido para métodos on-policy — o caso do
n-tuple Actor-Critic (PRD D6).

DIVERGÊNCIA DELIBERADA em relação à SPEC-00 T-00.9: a spec propunha um timer
"decide" por agente. O código de referência usa uma BARREIRA GLOBAL de época
(`GrADySHandler.iterate_algorithm`, environment.py 485-508): um evento
periódico marca o fim da época, o driver bombeia `step_simulation()` até ele, e
então TODOS os agentes agem no mesmo instante com a MESMA observação.

Isso é melhor e é o que gerou os números publicados: elimina a dependência da
ordem de disparo entre agentes, que no gradysim não é semanticamente definida
(o `Event.__lt__` compara apenas `timestamp`). Também fecha a pergunta em
aberto nº 10 da SPEC-05, que dependia desta confirmação.

Parar cedo exige bombear o laço manualmente: `start_simulation()` só devolve
quando o event loop esvazia, e ele NUNCA esvazia porque o handler de
mobilidade sempre se reagenda.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import (
    CommunicationHandler,
    CommunicationMedium,
)
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.handler.mobility import MobilityConfiguration, MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import (
    SimulationBuilder,
    SimulationConfiguration,
)

from .observation import build_observation, nearest_unvisited_direction
from .protocols import bind_protocols
from .rewards import reward_exploratory, reward_punish
from .scenario import Scenario, ScenarioConfig, episode_seed, generate_scenario
from .state import EpisodeResult, EpisodeState, TerminationReason, TruthStateView


class Policy(Protocol):
    """Contrato da política. Não conhece simulador, estado verdadeiro nem tempo."""

    def start_episode(self, agent_index: int, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        ...

    def step(
        self, agent_index: int, reward: float, obs: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        ...

    def end_episode(self, agent_index: int, reward: float) -> None:
        ...


class EpisodeContext:
    """Estado compartilhado entre driver e protocolos durante um episódio."""

    __slots__ = ("state", "agent_node_ids", "sensor_node_ids")

    def __init__(self, state: EpisodeState) -> None:
        self.state = state
        self.agent_node_ids: list[int] = []
        self.sensor_node_ids: list[int] = []


class _EpochBarrierHandler(INodeHandler):
    """Marca o fim de cada época de decisão.

    Equivalente ao `GrADySHandler` do código de referência. `inject` dispara
    `iterate` imediatamente, de modo que a época 0 já está marcada quando o
    driver dá o primeiro passo.
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self.epoch_finished = False
        self._event_loop: EventLoop | None = None

    @staticmethod
    def get_label() -> str:
        return "EpochBarrierHandler"

    def inject(self, event_loop: EventLoop) -> None:
        self._event_loop = event_loop
        self._iterate()

    def register_node(self, node: Node) -> None:
        pass

    def _iterate(self) -> None:
        self.epoch_finished = True
        assert self._event_loop is not None
        self._event_loop.schedule_event(
            self._event_loop.current_time + self._interval, self._iterate
        )


class EpisodeRunner:
    """Executa episódios. Um `Simulator` por episódio (o gradysim não tem reset)."""

    def __init__(
        self,
        config: ScenarioConfig,
        root_seed: int = 0,
        exploratory_reward: bool = False,
        record_observations: bool = False,
        on_decision: Callable[[float, EpisodeState], None] | None = None,
    ) -> None:
        self.config = config
        self.root_seed = root_seed
        self.exploratory_reward = exploratory_reward
        self.record_observations = record_observations
        self.on_decision = on_decision
        """Gancho de instrumentação, chamado a cada época com (t, state).

        Recebe o estado VERDADEIRO de propósito: existe para métricas e
        validação (SPEC-01 V-1, SPEC-03 custo), nunca no caminho da política.
        """
        self.recorded: list[NDArray[np.float32]] = []

    # -- construção ---------------------------------------------------------

    def _build(self, scenario: Scenario) -> tuple[object, EpisodeState, EpisodeContext, _EpochBarrierHandler, TimerHandler]:
        cfg = scenario.config
        state = EpisodeState(scenario)
        ctx = EpisodeContext(state)
        sensor_cls, drone_cls = bind_protocols(ctx)

        builder = SimulationBuilder(
            SimulationConfiguration(
                duration=cfg.max_episode_length,
                execution_logging=False,  # formata uma string por evento; custo de 1a ordem
                debug=False,
                real_time=False,
            )
        )
        builder.add_handler(
            CommunicationHandler(
                CommunicationMedium(
                    transmission_range=cfg.communication_range,
                    delay=0.0,
                    failure_rate=0.0,
                )
            )
        )
        builder.add_handler(
            MobilityHandler(MobilityConfiguration(update_rate=cfg.effective_update_rate))
        )
        timer_handler = TimerHandler()
        builder.add_handler(timer_handler)
        barrier = _EpochBarrierHandler(cfg.algorithm_iteration_interval)
        builder.add_handler(barrier)

        # Ordem de inserção igual à do código de referência: sensores primeiro.
        # z = 0 em TODOS os nós: o alcance de comunicação é avaliado em 3D.
        for p in scenario.sensor_positions:
            ctx.sensor_node_ids.append(
                builder.add_node(sensor_cls, (float(p[0]), float(p[1]), 0.0))
            )
        for p in scenario.agent_positions:
            ctx.agent_node_ids.append(
                builder.add_node(drone_cls, (float(p[0]), float(p[1]), 0.0))
            )

        state._sensor_id_to_index = {nid: i for i, nid in enumerate(ctx.sensor_node_ids)}

        handlers_before = list(logging.getLogger().handlers)
        simulator = builder.build()
        # `Simulator.__init__` adiciona um StreamHandler ao logger raiz e nunca
        # o remove. Um Simulator por episódio x 10 000 episódios = 10 000
        # handlers acumulados. Verificado por execução; poda obrigatória.
        _prune_root_handlers(handlers_before)

        state._nodes_agents = [simulator.get_node(i) for i in ctx.agent_node_ids]
        state._nodes_sensors = [simulator.get_node(i) for i in ctx.sensor_node_ids]
        return simulator, state, ctx, barrier, timer_handler

    # -- execução -----------------------------------------------------------

    def run(self, policy: Policy, episode_index: int) -> EpisodeResult:
        cfg = self.config
        rng, seed = episode_seed(self.root_seed, episode_index)
        scenario = generate_scenario(cfg, rng, seed=seed)
        simulator, state, ctx, barrier, timer_handler = self._build(scenario)

        n_a = cfg.effective_n_star_a
        n_s = cfg.n_star_s
        view = TruthStateView(state)
        reward_fn: Callable[[], float] = (
            (lambda: reward_exploratory(state.visited))
            if self.exploratory_reward
            else (lambda: reward_punish(state.visited, visited_prev))
        )

        wall0 = time.perf_counter()
        n_events = 0

        # Um passo inicial para que a época 0 seja consumida e as posições
        # estejam registradas, como no `reset()` de referência.
        if not simulator.step_simulation():
            raise RuntimeError("simulação terminou antes de começar")
        n_events += 1
        barrier.epoch_finished = False

        visited_prev = state.visited.copy()
        started = False
        return_sum = 0.0
        reason: TerminationReason | None = None
        ongoing = True

        while True:
            # --- decisão: todos os agentes, mesmo instante, mesma verdade ---
            if self.on_decision is not None:
                self.on_decision(float(timer_handler.get_current_time()), state)
            reward = reward_fn()
            actions: list[NDArray[np.float32]] = []
            for k in range(cfg.n_agents):
                obs = build_observation(
                    view, k, n_a, n_s, cfg.scenario_size, cfg.id_on_state
                )
                if self.record_observations:
                    self.recorded.append(obs)
                if not started:
                    actions.append(policy.start_episode(k, obs))
                else:
                    actions.append(policy.step(k, reward, obs))
            if started:
                return_sum += reward
            started = True
            state.n_decisions += 1

            for k, u in enumerate(actions):
                node = state._nodes_agents[k]
                node.protocol_encapsulator.protocol.act(float(u[0]), float(u[1]))

            visited_prev = state.visited.copy()

            # --- avança até o fim da época ---------------------------------
            barrier.epoch_finished = False
            while not barrier.epoch_finished:
                ongoing = simulator.step_simulation()
                n_events += 1
                if not ongoing:
                    reason = TerminationReason.TIME_LIMIT
                    break
            if not ongoing:
                break

            # --- terminação ------------------------------------------------
            if state.visited.sum() > visited_prev.sum():
                state.stall_duration = 0.0
            else:
                state.stall_duration += cfg.algorithm_iteration_interval

            if state.all_collected and cfg.end_when_all_collected:
                reason = TerminationReason.ALL_COLLECTED
                break
            if state.stall_duration > cfg.max_seconds_stalled:
                reason = TerminationReason.STALLED
                break

        end_time = timer_handler.get_current_time()
        final_reward = reward_fn()
        for k in range(cfg.n_agents):
            policy.end_episode(k, final_reward)

        return EpisodeResult(
            scenario=scenario,
            seed=seed,
            episode_index=episode_index,
            reason=reason or TerminationReason.TIME_LIMIT,
            end_time=float(end_time),
            completion_time=state.completion_time,
            n_visited=state.n_visited,
            n_sensors=cfg.n_sensors,
            visit_times=state.visit_time.copy(),
            n_decisions=state.n_decisions,
            n_events=n_events,
            wall_time=time.perf_counter() - wall0,
            return_sum=return_sum,
        )


def _prune_root_handlers(before: list[logging.Handler]) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)


# ---------------------------------------------------------------------------
# Políticas mínimas de validação (a versão instrumentada é da SPEC-02)
# ---------------------------------------------------------------------------


class GreedyPolicy:
    """Vai para o sensor não visitado mais próximo, a velocidade máxima.

    Lê a DIREÇÃO da própria observação, nunca a verdade — então um erro de
    sinal em `_normalize` aparece como um guloso que foge dos sensores.

    Mecanismo verbatim de `create_greedy_heuristics` (heuristics.py): recomputa
    o alvo a cada passo e usa `speed = 1` (normalizado, isto é, 15 m/s).
    """

    def __init__(self, n_star_a: int) -> None:
        self.n_star_a = n_star_a
        self._last = np.array([0.0, 1.0], dtype=np.float32)

    def _act(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        theta = nearest_unvisited_direction(obs, self.n_star_a)
        if theta is None:
            return self._last
        u0 = (theta % (2.0 * np.pi)) / (2.0 * np.pi)
        self._last = np.array([u0, 1.0], dtype=np.float32)
        return self._last

    def start_episode(self, agent_index: int, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._act(obs)

    def step(self, agent_index: int, reward: float, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._act(obs)

    def end_episode(self, agent_index: int, reward: float) -> None:
        pass


class RandomPolicy:
    """Piso. Rumo e velocidade uniformes a cada decisão (verbatim de
    `create_random_heuristics`)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def _act(self) -> NDArray[np.float32]:
        return self.rng.uniform(0.0, 1.0, size=(2,)).astype(np.float32)

    def start_episode(self, agent_index: int, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._act()

    def step(self, agent_index: int, reward: float, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._act()

    def end_episode(self, agent_index: int, reward: float) -> None:
        pass
