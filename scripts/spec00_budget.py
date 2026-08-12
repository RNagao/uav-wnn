"""SPEC-00 T-00.10 — custo de simulação no ambiente de referência (P2).

O `update_rate` não é mais decisão nossa: o `environment.py` de referência o
fixa em `algorithm_iteration_interval / 2`. Esta medição, portanto, só responde
"quanto custa", e não mais "que valor escolher".

Uso: uv run python scripts/spec00_budget.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from uavwnn.refenv import LAMENZA_003, load_heuristics, run_episode

GRID_A = [1, 2, 4, 8]
GRID_S = [12, 24, 36]
N_EPISODES = 20


def main() -> int:
    h = load_heuristics()
    print("Ambiente de referência (submodule), guloso de `heuristics.py`.")
    print(f"{'Na':>3} {'Ns':>3} {'passos':>8} {'s/ep':>8} {'sucesso':>8} "
          f"{'t_conc':>8} {'h/50M':>8}")

    rows = []
    for n_a in GRID_A:
        for n_s in GRID_S:
            cfg = LAMENZA_003(n_a, n_s)
            greedy = h.create_greedy_heuristics(n_a - 1, cfg.state_num_closest_sensors)
            t0 = time.perf_counter()
            res = [run_episode(cfg, greedy, seed=i) for i in range(N_EPISODES)]
            sec = (time.perf_counter() - t0) / N_EPISODES
            steps = float(np.mean([r.steps for r in res]))
            succ = float(np.mean([r.success for r in res]))
            ct = float(np.mean([r.completion_time for r in res]))
            hours = 50e6 / (steps * n_a) * sec / 3600
            print(f"{n_a:>3} {n_s:>3} {steps:>8.1f} {sec:>8.4f} {succ:>8.2f} "
                  f"{ct:>8.1f} {hours:>8.1f}")
            rows.append(dict(n_agents=n_a, n_sensors=n_s, steps=steps,
                             sec_per_ep=sec, success=succ, completion=ct,
                             hours_50M=hours))

    print()
    lo = min(rows, key=lambda r: r["hours_50M"])
    hi = max(rows, key=lambda r: r["hours_50M"])
    print(f"faixa: {lo['hours_50M']:.1f} h (Na={lo['n_agents']},Ns={lo['n_sensors']}) "
          f"a {hi['hours_50M']:.1f} h (Na={hi['n_agents']},Ns={hi['n_sensors']})")
    print("PRD §6.1 estimava 8-30 h em um núcleo.")

    with open("reports/spec00-orcamento.json", "w") as f:
        json.dump({"cost": rows, "n_episodes": N_EPISODES}, f, indent=2)
    print("\nartefato: reports/spec00-orcamento.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
