"""Teste diferencial contra a implementação de referência.

Roda `GrADySEnvironment` (scenario/003-collaboration/environment.py do
repositório Project-GrADyS/gradys-data-collection@dissertation) e o nosso
`EpisodeRunner` sobre EXATAMENTE o mesmo cenário e a mesma sequência de ações,
e compara observação, recompensa e visitação passo a passo.

O objetivo não é mostrar que o código é parecido — é medir se o MDP é o mesmo.

Uso:
    uv run python scripts/spec00_diff_referencia.py <caminho-do-ref_env.py>
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from uavwnn.env import EpisodeRunner, ScenarioConfig
from uavwnn.env.scenario import Scenario
import uavwnn.env.bridge as bridge

NA, NS = 4, 24
N_STEPS = 60
SEEDS = [1, 2, 3, 4, 5]


def make_ref(ref_module, interval: float):
    return ref_module.GrADySEnvironment(
        algorithm_iteration_interval=interval,
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


def run_pair(ref_module, seed: int, actions: np.ndarray):
    """actions: (N_STEPS, NA, 2) em [0,1]."""
    interval = 0.5

    # ---- referência -------------------------------------------------------
    env = make_ref(ref_module, interval)
    random.seed(seed)
    ref_obs0, _ = env.reset()
    sensors = np.array(
        [env.simulator.get_node(i).position[:2] for i in env.sensor_node_ids]
    )
    agents = np.array(
        [env.simulator.get_node(i).position[:2] for i in env.agent_node_ids]
    )

    ref_obs = [np.stack([ref_obs0[f"drone{k}"] for k in range(NA)])]
    ref_rew, ref_vis = [], []
    for t in range(N_STEPS):
        a = {f"drone{k}": actions[t, k] for k in range(NA)}
        o, r, term, trunc, info = env.step(a)
        ref_obs.append(np.stack([o[f"drone{k}"] for k in range(NA)]))
        ref_rew.append(r["drone0"])
        ref_vis.append(
            sum(
                env.simulator.get_node(i).protocol_encapsulator.protocol.has_collected
                for i in env.sensor_node_ids
            )
        )
        if any(term.values()):
            break

    # ---- nosso ------------------------------------------------------------
    # `update_rate = interval/2` para casar com o MobilityHandler da referência.
    cfg = ScenarioConfig(
        n_agents=NA, n_sensors=NS, algorithm_iteration_interval=interval,
        update_rate=interval / 2,
    )

    real_gen = bridge.generate_scenario
    bridge.generate_scenario = lambda c, rng, seed=0: Scenario(
        config=c, agent_positions=agents.copy(), sensor_positions=sensors.copy(), seed=seed
    )

    our_obs, our_rew, our_vis = [], [], []

    class Scripted:
        def __init__(self):
            self.t = 0

        def start_episode(self, k, obs):
            return self._act(k, obs, None)

        def step(self, k, r, obs):
            return self._act(k, obs, r)

        def _act(self, k, obs, r):
            if k == 0:
                our_obs.append([])
                if r is not None:
                    our_rew.append(r)
            our_obs[-1].append(np.asarray(obs))
            idx = min(self.t, N_STEPS - 1)
            if k == NA - 1:
                self.t += 1
            return actions[idx, k].astype(np.float32)

        def end_episode(self, k, r):
            pass

    runner = EpisodeRunner(
        cfg, root_seed=0,
        on_decision=lambda t, st: our_vis.append(int(st.visited.sum())),
    )
    try:
        runner.run(Scripted(), 0)
    finally:
        bridge.generate_scenario = real_gen

    our_obs_arr = [np.stack(x) for x in our_obs if len(x) == NA]
    return ref_obs, ref_rew, ref_vis, our_obs_arr, our_rew, our_vis


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not path.exists():
        print(f"uso: {sys.argv[0]} <caminho-do-ref_env.py>")
        return 2
    sys.path.insert(0, str(path.parent))
    ref_module = __import__(path.stem)

    rng = np.random.default_rng(0)
    print(f"{'seed':>5} {'passos':>7} {'max|Δobs|':>11} {'max|Δrec|':>11} {'Δvisitados':>11}")
    worst_obs = worst_rew = 0.0
    for seed in SEEDS:
        actions = rng.uniform(0, 1, size=(N_STEPS, NA, 2))
        ro, rr, rv, oo, orw, ov = run_pair(ref_module, seed, actions)

        n = min(len(ro), len(oo))
        d_obs = max(
            float(np.max(np.abs(ro[i] - oo[i]))) for i in range(n)
        ) if n else float("nan")
        m = min(len(rr), len(orw))
        d_rew = max(abs(rr[i] - orw[i]) for i in range(m)) if m else float("nan")
        k = min(len(rv), len(ov) - 1)
        d_vis = max(abs(rv[i] - ov[i + 1]) for i in range(k)) if k > 0 else 0

        worst_obs = max(worst_obs, d_obs)
        worst_rew = max(worst_rew, d_rew)
        print(f"{seed:>5} {n:>7} {d_obs:>11.2e} {d_rew:>11.2e} {d_vis:>11}")

    print()
    print(f"pior |Δobs| = {worst_obs:.3e}   pior |Δrecompensa| = {worst_rew:.3e}")
    ok = worst_obs < 1e-6 and worst_rew < 1e-9
    print("VEREDITO:", "MDP idêntico" if ok else "DIVERGENTE — investigar")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
