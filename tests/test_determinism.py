"""Determinismo e higiene (SPEC-00, T-00.12, testes 18-22)."""

from __future__ import annotations

import logging

import numpy as np

from uavwnn.env import EpisodeRunner, GreedyPolicy, ScenarioConfig, episode_seed


def _cfg(**kw):
    return ScenarioConfig(n_agents=2, n_sensors=12, **kw)


# --- 18: mesma semente, mesmo resultado ------------------------------------


def test_18_same_seed_same_result():
    cfg = _cfg()
    a = EpisodeRunner(cfg, root_seed=42).run(GreedyPolicy(cfg.effective_n_star_a), 3)
    b = EpisodeRunner(cfg, root_seed=42).run(GreedyPolicy(cfg.effective_n_star_a), 3)

    assert a.seed == b.seed
    assert a.reason == b.reason
    assert a.end_time == b.end_time
    assert a.n_visited == b.n_visited
    assert a.n_decisions == b.n_decisions
    assert a.n_events == b.n_events
    np.testing.assert_array_equal(a.visit_times, b.visit_times)
    np.testing.assert_array_equal(
        a.scenario.sensor_positions, b.scenario.sensor_positions
    )


# --- 19: episódio i reproduzível isoladamente ------------------------------


def test_19_episode_reproducible_out_of_order():
    cfg = _cfg()
    runner = EpisodeRunner(cfg, root_seed=7)

    seq = [runner.run(GreedyPolicy(cfg.effective_n_star_a), i) for i in range(4)]
    isolated = EpisodeRunner(cfg, root_seed=7).run(GreedyPolicy(cfg.effective_n_star_a), 2)

    assert isolated.seed == seq[2].seed
    np.testing.assert_array_equal(
        isolated.scenario.sensor_positions, seq[2].scenario.sensor_positions
    )
    assert isolated.end_time == seq[2].end_time


def test_19b_episode_seed_distinct_and_stable():
    seeds = [episode_seed(123, i)[1] for i in range(64)]
    assert len(set(seeds)) == 64
    assert episode_seed(123, 10)[1] == seeds[10]


# --- 20: nenhum RNG global no caminho do episódio --------------------------


def test_20_no_global_rng_dependency():
    """Alterar o estado do `random` da stdlib e do `np.random` legado não pode
    mudar o episódio."""
    import random

    cfg = _cfg()
    random.seed(1)
    np.random.seed(1)
    a = EpisodeRunner(cfg, root_seed=9).run(GreedyPolicy(cfg.effective_n_star_a), 0)

    random.seed(999)
    np.random.seed(999)
    b = EpisodeRunner(cfg, root_seed=9).run(GreedyPolicy(cfg.effective_n_star_a), 0)

    assert a.end_time == b.end_time
    np.testing.assert_array_equal(a.visit_times, b.visit_times)


# --- 21: vazamento de handlers de logging ----------------------------------


def test_21_no_logging_handler_leak():
    """`Simulator.__init__` adiciona um StreamHandler ao logger raiz e nunca o
    remove. Verificado por execução: 3 Simulators -> 3 handlers. Sem a poda,
    10 000 episódios acumulam 10 000 handlers."""
    cfg = _cfg()
    runner = EpisodeRunner(cfg, root_seed=13)
    policy = GreedyPolicy(cfg.effective_n_star_a)

    runner.run(policy, 0)
    before = len(logging.getLogger().handlers)
    for i in range(1, 25):
        runner.run(policy, i)
    after = len(logging.getLogger().handlers)

    assert after == before, f"vazou {after - before} handlers em 24 episódios"


# --- 22: sem crescimento descontrolado de objetos --------------------------


def test_22_no_simulator_leak():
    import gc

    from gradysim.simulator.simulation import Simulator

    cfg = _cfg()
    runner = EpisodeRunner(cfg, root_seed=17)
    policy = GreedyPolicy(cfg.effective_n_star_a)

    runner.run(policy, 0)
    gc.collect()
    before = sum(1 for o in gc.get_objects() if isinstance(o, Simulator))

    for i in range(1, 11):
        runner.run(policy, i)
    gc.collect()
    after = sum(1 for o in gc.get_objects() if isinstance(o, Simulator))

    assert after - before <= 1, f"{after - before} Simulators retidos após 10 episódios"


# --- 25: todos os nós em z = 0 --------------------------------------------


def test_25_all_nodes_at_z_zero():
    """O alcance de comunicação é avaliado em 3D; com z != 0 o raio efetivo no
    solo seria sqrt(r^2 - h^2)."""
    cfg = _cfg()
    runner = EpisodeRunner(cfg, root_seed=19)

    import uavwnn.env.bridge as bridge

    captured: list[tuple] = []
    real_builder_cls = bridge.SimulationBuilder

    class Spy(real_builder_cls):
        def add_node(self, protocol, position):
            captured.append(position)
            return super().add_node(protocol, position)

    bridge.SimulationBuilder = Spy
    try:
        runner.run(GreedyPolicy(cfg.effective_n_star_a), 0)
    finally:
        bridge.SimulationBuilder = real_builder_cls

    assert captured
    assert all(p[2] == 0.0 for p in captured)
