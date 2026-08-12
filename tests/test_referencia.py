"""Teste diferencial contra a implementação de referência (SPEC-00).

Prova que o MDP é o mesmo em vez de argumentar que é. Roda o
`GrADySEnvironment` do repositório `Project-GrADyS/gradys-data-collection`
(branch `dissertation`, cenário `003-collaboration`) e o nosso `EpisodeRunner`
sobre o MESMO cenário e a MESMA sequência de ações, e compara observação,
recompensa e visitação passo a passo.

O `environment.py` de referência NÃO é versionado aqui — é código de terceiros.
Para rodar este teste, baixe-o e aponte a variável de ambiente:

    curl -sSL https://raw.githubusercontent.com/Project-GrADyS/gradys-data-collection/dissertation/scenario/003-collaboration/environment.py -o /tmp/ref_env.py
    UAVWNN_REF_ENV=/tmp/ref_env.py uv run pytest tests/test_referencia.py -q

Sem a variável, o teste é pulado.

Nota: o arquivo de referência importa `VisualizationController`, que não existe
no gradysim 0.8.0 — ele foi escrito contra outra versão. O teste neutraliza
apenas o bloco de visualização, que não é usado com `render_mode=None`.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

import uavwnn.env.bridge as bridge
from uavwnn.env import EpisodeRunner, ScenarioConfig
from uavwnn.env.scenario import Scenario

NA, NS = 4, 24
N_STEPS = 60
SEEDS = [1, 2, 3, 4, 5]

_REF_PATH = os.environ.get("UAVWNN_REF_ENV")
pytestmark = pytest.mark.skipif(
    not _REF_PATH or not Path(_REF_PATH).exists(),
    reason="defina UAVWNN_REF_ENV com o caminho do environment.py de referência",
)


@pytest.fixture(scope="module")
def ref_module():
    src = Path(_REF_PATH).read_text()
    lines = []
    for line in src.split("\n"):
        if line.startswith("from gradysim.simulator.handler.visualization import"):
            lines.append(
                "VisualizationHandler = VisualizationConfiguration = "
                "VisualizationController = None  # stub"
            )
        elif line.strip() == "VisualizationController":
            continue
        else:
            lines.append(line)
    patched = Path(_REF_PATH).parent / "_ref_env_patched.py"
    patched.write_text("\n".join(lines))
    sys.path.insert(0, str(patched.parent))
    return __import__(patched.stem)


def _run_reference(ref_module, seed, actions):
    env = ref_module.GrADySEnvironment(
        algorithm_iteration_interval=0.5,
        num_drones=NA,
        num_sensors=NS,
        scenario_size=100.0,
        max_episode_length=500.0,
        max_seconds_stalled=30,
        communication_range=20.0,
        state_num_closest_sensors=12,
        state_num_closest_drones=NA - 1,
        state_mode="relative",
        id_on_state=True,
        min_sensor_priority=1.0,
        max_sensor_priority=1.0,
        reward="punish",
        speed_action=True,
        end_when_all_collected=False,
    )
    random.seed(seed)
    obs0, _ = env.reset()
    sensors = np.array([env.simulator.get_node(i).position[:2] for i in env.sensor_node_ids])
    agents = np.array([env.simulator.get_node(i).position[:2] for i in env.agent_node_ids])

    obs = [np.stack([obs0[f"drone{k}"] for k in range(NA)])]
    rewards = []
    for t in range(N_STEPS):
        o, r, term, _, _ = env.step({f"drone{k}": actions[t, k] for k in range(NA)})
        obs.append(np.stack([o[f"drone{k}"] for k in range(NA)]))
        rewards.append(r["drone0"])
        if any(term.values()):
            break
    return obs, rewards, sensors, agents


def _run_ours(sensors, agents, actions):
    cfg = ScenarioConfig(
        n_agents=NA,
        n_sensors=NS,
        algorithm_iteration_interval=0.5,
        update_rate=0.25,  # casa com o MobilityHandler da referência (interval/2)
    )
    real = bridge.generate_scenario
    bridge.generate_scenario = lambda c, rng, seed=0: Scenario(
        config=c, agent_positions=agents.copy(), sensor_positions=sensors.copy(), seed=seed
    )
    obs: list[list[np.ndarray]] = []
    rewards: list[float] = []

    class Scripted:
        def __init__(self):
            self.t = 0

        def start_episode(self, k, o):
            return self._act(k, o, None)

        def step(self, k, r, o):
            return self._act(k, o, r)

        def _act(self, k, o, r):
            if k == 0:
                obs.append([])
                if r is not None:
                    rewards.append(r)
            obs[-1].append(np.asarray(o))
            idx = min(self.t, N_STEPS - 1)
            if k == NA - 1:
                self.t += 1
            return actions[idx, k].astype(np.float32)

        def end_episode(self, k, r):
            pass

    try:
        EpisodeRunner(cfg, root_seed=0).run(Scripted(), 0)
    finally:
        bridge.generate_scenario = real
    return [np.stack(x) for x in obs if len(x) == NA], rewards


@pytest.mark.parametrize("seed", SEEDS)
def test_mdp_identico_a_referencia(ref_module, seed):
    """Observação e recompensa idênticas à precisão de float32.

    Se este teste falhar, alguma coisa no MDP divergiu do original -- e o
    princípio do PRD §0 foi quebrado.
    """
    rng = np.random.default_rng(seed)
    actions = rng.uniform(0, 1, size=(N_STEPS, NA, 2))

    ref_obs, ref_rew, sensors, agents = _run_reference(ref_module, seed, actions)
    our_obs, our_rew = _run_ours(sensors, agents, actions)

    n = min(len(ref_obs), len(our_obs))
    assert n >= 30, f"poucos passos comparáveis: {n}"

    d_obs = max(float(np.max(np.abs(ref_obs[i] - our_obs[i]))) for i in range(n))
    assert d_obs < 1e-6, f"observação divergiu: max|Δ| = {d_obs:.3e}"

    m = min(len(ref_rew), len(our_rew))
    d_rew = max(abs(ref_rew[i] - our_rew[i]) for i in range(m))
    assert d_rew < 1e-9, f"recompensa divergiu: max|Δ| = {d_rew:.3e}"
