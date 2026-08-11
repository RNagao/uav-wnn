"""Testes de contrato do MDP (SPEC-00, T-00.12).

A numeração segue a tabela da spec. Onde o código de referência contradisse a
spec, o teste segue o CÓDIGO e o desvio está anotado.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from uavwnn.env import (
    PADDING_VALUE,
    EpisodeRunner,
    GreedyPolicy,
    ScenarioConfig,
    TerminationReason,
    build_observation,
    reward_exploratory,
    reward_punish,
)
from uavwnn.env.protocols import DroneProtocol


class FakeView:
    def __init__(self, agents, sensors, visited):
        self._a = np.asarray(agents, dtype=np.float64)
        self._s = np.asarray(sensors, dtype=np.float64)
        self._v = np.asarray(visited, dtype=bool)

    def agent_positions(self):
        return self._a

    def sensor_positions(self):
        return self._s

    def visited_mask(self):
        return self._v


# --- 1: dimensão -----------------------------------------------------------


@pytest.mark.parametrize("n_agents,expected", [(1, 25), (2, 27), (4, 31), (8, 39)])
def test_01_observation_size(n_agents, expected):
    cfg = ScenarioConfig(n_agents=n_agents, n_sensors=12)
    assert cfg.observation_size == expected

    view = FakeView(
        agents=np.zeros((n_agents, 2)),
        sensors=np.ones((12, 2)),
        visited=np.zeros(12, dtype=bool),
    )
    obs = build_observation(view, 0, cfg.effective_n_star_a, cfg.n_star_s, 100.0)
    assert obs.shape == (expected,)
    assert obs.dtype == np.float32


# --- 2: sinal das coordenadas relativas ------------------------------------


def test_02_relative_sign_and_normalization():
    """Eq. 5-2/5-5: `(x_k - x_i)`, normalizado por `(d + S) / (2S)`."""
    view = FakeView(agents=[[10.0, 0.0]], sensors=[[30.0, 0.0]], visited=[False])
    obs = build_observation(view, 0, n_star_a=0, n_star_s=1, scenario_size=100.0)
    # d = 10 - 30 = -20  ->  (-20 + 100) / 200 = 0.4
    assert obs[0] == pytest.approx(0.4)
    assert obs[1] == pytest.approx(0.5)  # (0 - 0 + 100)/200


# --- 3: ordenação por distância crescente ----------------------------------


def test_03_sorted_by_distance():
    view = FakeView(
        agents=[[0.0, 0.0]],
        sensors=[[50.0, 0.0], [10.0, 0.0], [30.0, 0.0]],
        visited=[False, False, False],
    )
    obs = build_observation(view, 0, n_star_a=0, n_star_s=3, scenario_size=100.0)
    xs = obs[0:6:2]
    # (0 - d + 100)/200 é decrescente em d, então a ordem por distância
    # crescente aparece como valores decrescentes.
    expected = [(0 - d + 100) / 200 for d in (10.0, 30.0, 50.0)]
    assert list(xs) == pytest.approx(expected)


# --- 4: sensores visitados nunca aparecem ----------------------------------


def test_04_visited_never_appears():
    view = FakeView(
        agents=[[0.0, 0.0]],
        sensors=[[10.0, 0.0], [30.0, 0.0]],
        visited=[True, False],
    )
    obs = build_observation(view, 0, n_star_a=0, n_star_s=2, scenario_size=100.0)
    assert obs[0] == pytest.approx((0 - 30 + 100) / 200)
    assert obs[2] == PADDING_VALUE and obs[3] == PADDING_VALUE


# --- 5: padding ------------------------------------------------------------


def test_05_padding_is_minus_one_and_trailing():
    """DESVIO DA SPEC: o padding do original é -1, não (0, 0).

    Como -1 está fora da faixa dos valores válidos, padding e "sensor sobre o
    UAV" são distinguíveis por construção. O bit de validade proposto na
    SPEC-00 §5.4 e na SPEC-01 é DESNECESSÁRIO.
    """
    view = FakeView(agents=[[0.0, 0.0]], sensors=[[0.0, 0.0]], visited=[False])
    obs = build_observation(view, 0, n_star_a=0, n_star_s=3, scenario_size=100.0)
    # Sensor exatamente sobre o UAV -> 0.5, distinguível do padding -1.
    assert obs[0] == pytest.approx(0.5) and obs[1] == pytest.approx(0.5)
    assert list(obs[2:6]) == [PADDING_VALUE] * 4


# --- 6: pureza -------------------------------------------------------------


def test_06_build_observation_is_pure():
    view = FakeView(
        agents=np.array([[1.0, 2.0], [3.0, 4.0]]),
        sensors=np.array([[5.0, 6.0], [7.0, 8.0]]),
        visited=np.array([False, True]),
    )
    a = build_observation(view, 0, 1, 2, 100.0)
    b = build_observation(view, 0, 1, 2, 100.0)
    assert np.array_equal(a, b)


# --- 6b: o próprio agente não entra em P^a ---------------------------------


def test_06b_self_not_in_agent_block():
    view = FakeView(
        agents=[[0.0, 0.0], [10.0, 0.0]],
        sensors=[[80.0, 0.0]],
        visited=[False],
    )
    obs = build_observation(view, 0, n_star_a=1, n_star_s=1, scenario_size=100.0)
    assert obs[0] == pytest.approx((0 - 10 + 100) / 200)  # o outro, não ele mesmo


# --- 7/8: ação e distância percorrida --------------------------------------


@pytest.mark.parametrize("u1,expected_speed", [(0.0, 0.0), (0.5, 7.5), (1.0, 15.0)])
def test_07_action_denormalization(u1, expected_speed):
    cfg = ScenarioConfig(n_agents=1, n_sensors=1)
    assert u1 * cfg.max_speed == pytest.approx(expected_speed)


@pytest.mark.parametrize("u1", [0.0, 0.5, 1.0])
def test_08_distance_travelled_per_interval(u1):
    """Distância percorrida por intervalo = `s * dt`, MEDIDA no simulador.

    Este é o teste que pega o fato 4 da §4.1 caso alguém troque a aplicação da
    ação por `SetVelocityMobilityCommand`: o comando seria aceito sem erro,
    interpretado como `speed = vx`, e só a distância percorrida denunciaria.
    """
    # Episódio curto de propósito: a 15 m/s o UAV atinge a borda em ~7 s, e
    # perto dela o waypoint satura (ver test_08b). Medimos longe da borda.
    cfg = ScenarioConfig(
        n_agents=1,
        n_sensors=1,
        max_seconds_stalled=1e9,
        max_episode_length=3.0,
        spawn_radius=0.0,
    )
    dt = cfg.algorithm_iteration_interval
    expected = u1 * cfg.max_speed * dt

    class FixedPolicy:
        def start_episode(self, k, obs):
            return np.array([0.0, u1], dtype=np.float32)  # rumo +x, u0=0

        def step(self, k, r, obs):
            return np.array([0.0, u1], dtype=np.float32)

        def end_episode(self, k, r):
            pass

    positions: list[np.ndarray] = []
    runner = EpisodeRunner(
        cfg, root_seed=3, on_decision=lambda t, st: positions.append(st.agent_positions()[0].copy())
    )
    runner.run(FixedPolicy(), 0)

    assert len(positions) >= 5
    # Descarta o transiente inicial (ver test_08d): as duas primeiras épocas
    # são anômalas por construção do driver de referência.
    steps = [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(2, len(positions) - 1)]
    assert steps, "sem passos medidos"
    for d in steps:
        assert d == pytest.approx(expected, abs=1e-6), f"esperado {expected}, medido {d}"


def test_08d_initial_epoch_transient_is_faithful():
    """Documenta um transiente do driver de referência, não um bug nosso.

    `reset()` dá UM `step_simulation()` antes da primeira observação. Esse
    passo consome o primeiro evento da fila — um tique de mobilidade em
    `t = update_rate` — de modo que:

    1. a primeira decisão ocorre em `t = update_rate`, não em `t = 0`;
    2. a primeira época é mais curta e vale `speed * (interval - 2*update_rate)`
       de deslocamento, contra `speed * interval` no regime estacionário: o
       tique que abriu a época já passou antes de a ação ser aplicada, e o que
       a fecha é lido depois;
    3. a posição lida numa decisão está um tique de mobilidade ATRASADA,
       porque a barreira e o tique compartilham o mesmo timestamp e a barreira
       é processada primeiro.

    O código de referência (environment.py, reset/step) tem exatamente a mesma
    estrutura, então isto é fidelidade. Fica registrado porque (3) importa para
    a SPEC-07: o KF vai estimar uma posição que a política observa com atraso.
    """
    cfg = ScenarioConfig(
        n_agents=1, n_sensors=1, max_seconds_stalled=1e9, max_episode_length=3.0,
        spawn_radius=0.0,
    )

    class Fixed:
        def start_episode(self, k, obs):
            return np.array([0.0, 1.0], dtype=np.float32)

        def step(self, k, r, obs):
            return np.array([0.0, 1.0], dtype=np.float32)

        def end_episode(self, k, r):
            pass

    times: list[float] = []
    xs: list[float] = []
    runner = EpisodeRunner(
        cfg, root_seed=3,
        on_decision=lambda t, st: (times.append(t), xs.append(float(st.agent_positions()[0][0]))),
    )
    runner.run(Fixed(), 0)

    ur = cfg.effective_update_rate
    dt = cfg.algorithm_iteration_interval

    assert times[0] == pytest.approx(ur)  # (1)
    assert times[1] == pytest.approx(dt)

    assert xs[0] == pytest.approx(0.0)
    first = cfg.max_speed * (dt - 2 * ur)  # (2)
    assert xs[1] - xs[0] == pytest.approx(first, abs=1e-6)

    steady = cfg.max_speed * dt  # (3) regime estacionário
    assert xs[2] - xs[1] == pytest.approx(steady, abs=1e-6)
    assert xs[3] - xs[2] == pytest.approx(steady, abs=1e-6)


def test_08b_waypoint_reference_quirk():
    """A quirk da borda do original: perto de uma borda o UAV quase para,
    mesmo apontando para DENTRO. Reproduzida de propósito."""
    # Em x = 99, indo para -x (para dentro), limite 100.
    x, y = 99.0, 0.0
    dest = DroneProtocol._waypoint(x, y, -1.0, 0.0, 100.0, "reference")
    # `limit - |x|` = 1, então o waypoint fica a apenas 1 m — para dentro.
    assert dest[0] == pytest.approx(98.0)

    corrected = DroneProtocol._waypoint(x, y, -1.0, 0.0, 100.0, "corrected")
    assert corrected[0] == pytest.approx(-100.0)


def test_08c_waypoint_stays_in_bounds():
    for ux, uy in [(1, 0), (0, 1), (-1, 0), (0, -1), (0.7071, 0.7071)]:
        dest = DroneProtocol._waypoint(50.0, -30.0, ux, uy, 100.0, "reference")
        assert -100.0 - 1e-6 <= dest[0] <= 100.0 + 1e-6
        assert -100.0 - 1e-6 <= dest[1] <= 100.0 + 1e-6


def test_07b_zero_speed_no_nan():
    dest = DroneProtocol._waypoint(0.0, 0.0, math.cos(0.0), math.sin(0.0), 100.0, "reference")
    assert all(np.isfinite(dest))


# --- 9: recompensa ---------------------------------------------------------


def test_09_reward_punish():
    prev = np.zeros(12, dtype=bool)
    now = prev.copy()
    now[:2] = True
    assert reward_punish(now, prev) == pytest.approx(20.0)  # Δ=2 -> 20

    v = np.zeros(12, dtype=bool)
    v[:3] = True
    assert reward_punish(v, v) == pytest.approx(-0.75)  # 3 de 12, Δ=0

    full = np.ones(12, dtype=bool)
    assert reward_punish(full, full) == pytest.approx(0.0)  # tudo visitado -> 0


def test_09b_reward_exploratory():
    v = np.zeros(12, dtype=bool)
    v[:3] = True
    assert reward_exploratory(v) == pytest.approx(0.25)


# --- 11/12: coleta por proximidade -----------------------------------------


@pytest.mark.parametrize("offset,should_collect", [(-1.0, True), (+1.0, False)])
def test_11_collection_by_range(offset, should_collect):
    """Sensor a `range - eps` é coletado; a `range + eps` não é."""
    cfg = ScenarioConfig(
        n_agents=1, n_sensors=1, communication_range=20.0, spawn_radius=0.0
    )
    r = cfg.communication_range + offset

    class Still:
        def start_episode(self, k, obs):
            return np.array([0.0, 0.0], dtype=np.float32)

        def step(self, k, rw, obs):
            return np.array([0.0, 0.0], dtype=np.float32)

        def end_episode(self, k, rw):
            pass

    runner = EpisodeRunner(cfg, root_seed=0)
    # Injeta um cenário determinístico com o sensor à distância exata.
    import uavwnn.env.bridge as bridge
    from uavwnn.env.scenario import Scenario

    real = bridge.generate_scenario

    def fake(config, rng, seed=0):
        return Scenario(
            config=config,
            agent_positions=np.zeros((1, 2)),
            sensor_positions=np.array([[r, 0.0]]),
            seed=seed,
        )

    bridge.generate_scenario = fake
    try:
        res = runner.run(Still(), 0)
    finally:
        bridge.generate_scenario = real
    assert res.success is should_collect


# --- 13/14/15: terminação --------------------------------------------------


def test_13_rf02_episode_does_not_end_on_completion():
    """RF-02, leitura (B).

    DESVIO DA SPEC: a SPEC-00 §9.2 propunha a leitura (A) — episódio corre até
    500 s após a coleta completa. O `benchmark.py` de referência passa
    `--no-end-when-all-collected`, o que faz o episódio CONTINUAR após a
    coleta, mas a poda por parada segue valendo: ele acaba
    `max_seconds_stalled` depois da última visita. É a leitura (B).
    """
    cfg = ScenarioConfig(n_agents=4, n_sensors=12)
    assert cfg.end_when_all_collected is False

    runner = EpisodeRunner(cfg, root_seed=1)
    res = runner.run(GreedyPolicy(cfg.effective_n_star_a), 0)
    assert res.success
    assert res.completion_time is not None
    # Continuou após completar...
    assert res.end_time > res.completion_time
    # ...e parou pela regra de parada, ~max_seconds_stalled depois.
    assert res.reason is TerminationReason.STALLED
    assert res.end_time - res.completion_time == pytest.approx(
        cfg.max_seconds_stalled, abs=2 * cfg.algorithm_iteration_interval
    )


def test_13b_end_when_all_collected_true():
    cfg = ScenarioConfig(n_agents=4, n_sensors=12, end_when_all_collected=True)
    runner = EpisodeRunner(cfg, root_seed=1)
    res = runner.run(GreedyPolicy(cfg.effective_n_star_a), 0)
    assert res.success
    assert res.reason is TerminationReason.ALL_COLLECTED


def test_14_stalled_pruning():
    cfg = ScenarioConfig(n_agents=1, n_sensors=4, max_seconds_stalled=5.0)

    class Still:
        def start_episode(self, k, obs):
            return np.array([0.0, 0.0], dtype=np.float32)

        def step(self, k, r, obs):
            return np.array([0.0, 0.0], dtype=np.float32)

        def end_episode(self, k, r):
            pass

    res = EpisodeRunner(cfg, root_seed=7).run(Still(), 0)
    assert res.reason is TerminationReason.STALLED
    assert res.end_time < 20.0


def test_15_time_limit():
    cfg = ScenarioConfig(
        n_agents=1, n_sensors=2, max_episode_length=20.0, max_seconds_stalled=1e9
    )

    class Circler:
        def __init__(self):
            self.t = 0

        def start_episode(self, k, obs):
            return self.step(k, 0.0, obs)

        def step(self, k, r, obs):
            self.t += 1
            return np.array([(self.t * 0.13) % 1.0, 1.0], dtype=np.float32)

        def end_episode(self, k, r):
            pass

    res = EpisodeRunner(cfg, root_seed=11).run(Circler(), 0)
    assert res.reason is TerminationReason.TIME_LIMIT
    assert res.end_time == pytest.approx(20.0, abs=1.0)


# --- 16: RF-04 pré-registro ------------------------------------------------


def test_16_build_observation_takes_only_a_view():
    """A assinatura não pode expor `EpisodeState`. O teste de vazamento
    propriamente dito é da SPEC-07."""
    import inspect

    sig = inspect.signature(build_observation)
    ann = sig.parameters["view"].annotation
    assert "StateView" in str(ann)
    src = inspect.getsource(build_observation)
    assert "EpisodeState" not in src


# --- 17: coerência dentro de uma época -------------------------------------


def test_17_same_epoch_same_observation_inputs():
    """Todos os agentes decidem no mesmo instante, sobre o mesmo estado.

    Confirmado pelo código de referência: a barreira global de época
    (`GrADySHandler`) sincroniza os agentes. Isto fecha a pergunta nº 10 da
    SPEC-05.
    """
    cfg = ScenarioConfig(n_agents=4, n_sensors=12)
    seen: list[tuple[int, ...]] = []

    class Recorder:
        def start_episode(self, k, obs):
            return self.step(k, 0.0, obs)

        def step(self, k, r, obs):
            if k == 0:
                seen.append(tuple())
            return np.array([0.25, 1.0], dtype=np.float32)

        def end_episode(self, k, r):
            pass

    runner = EpisodeRunner(cfg, root_seed=5, record_observations=True)
    res = runner.run(Recorder(), 0)
    recorded = runner.recorded
    # Os 4 primeiros vetores são da mesma época: os blocos de sensores devem
    # descrever o mesmo conjunto de sensores não visitados.
    assert len(recorded) >= 4
    n_a = cfg.effective_n_star_a
    blocks = [tuple(np.round(o[2 * n_a : -1], 6)) for o in recorded[:4]]
    # Diferentes entre si (agentes em posições diferentes), mas todos com o
    # mesmo número de slots preenchidos.
    filled = {int(np.sum(~np.isclose(np.array(b).reshape(-1, 2), PADDING_VALUE).all(axis=1))) for b in blocks}
    assert len(filled) == 1


# --- asserções de configuração (23/24) -------------------------------------


def test_23_interval_must_be_multiple_of_update_rate():
    with pytest.raises(ValueError, match="múltiplo inteiro"):
        ScenarioConfig(n_agents=1, n_sensors=1, update_rate=0.11)


def test_24_heartbeat_must_not_outrun_range():
    with pytest.raises(ValueError, match="heartbeat esparso"):
        # dt = 2 s (o valor da tab. 7.1) com alcance 20 m: o UAV percorre
        # ~29.7 m entre heartbeats e ATRAVESSA a cobertura sem coletar.
        ScenarioConfig(
            n_agents=1, n_sensors=1, algorithm_iteration_interval=2.0,
            communication_range=20.0,
        )


def test_26_update_rate_must_be_dyadic():
    """O handler de mobilidade reagenda por soma repetida; um `update_rate` não
    exato em binário deriva contra a barreira de época e o número de tiques por
    época passa a oscilar (medido com 0.1: 4,5 / 9,0 / 6,0 / 7,5 m em vez de
    7,5 constante). A média fica certa, a dinâmica não."""
    with pytest.raises(ValueError, match="exato em binário"):
        ScenarioConfig(n_agents=1, n_sensors=1, update_rate=0.1)

    # Frações diádicas passam.
    for k in (1, 2, 3, 4, 5):
        ScenarioConfig(n_agents=1, n_sensors=1, update_rate=0.5 / (2**k))
