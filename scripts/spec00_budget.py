"""SPEC-00 T-00.10 — custo de simulação e fidelidade de `update_rate` (P2).

Duas perguntas distintas e frequentemente confundidas:
  (i)  quanto custa;
  (ii) o resultado muda.

Uso: uv run python scripts/spec00_budget.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from uavwnn.env import EpisodeRunner, GreedyPolicy, ScenarioConfig

GRID_A = [1, 2, 4, 8]
GRID_S = [12, 24, 36]
N_EPISODES = 20


def run_cell(n_a: int, n_s: int, update_rate: float | None, n_ep: int, seed: int = 0):
    cfg = ScenarioConfig(n_agents=n_a, n_sensors=n_s, update_rate=update_rate)
    runner = EpisodeRunner(cfg, root_seed=seed)
    policy = GreedyPolicy(cfg.effective_n_star_a)
    t0 = time.perf_counter()
    rows = [runner.run(policy, i) for i in range(n_ep)]
    wall = time.perf_counter() - t0
    return cfg, rows, wall


def main() -> int:
    print("=" * 78)
    print("(i) CUSTO — grade completa, update_rate de referência (interval/2)")
    print("=" * 78)
    print(f"{'Na':>3} {'Ns':>3} {'eventos/ep':>11} {'modelo':>9} {'s/ep':>8} "
          f"{'decisões':>9} {'sucesso':>8} {'t_conc':>8}")

    cost_rows = []
    for n_a in GRID_A:
        for n_s in GRID_S:
            cfg, rows, wall = run_cell(n_a, n_s, None, N_EPISODES)
            ev = np.mean([r.n_events for r in rows])
            sec = wall / len(rows)
            dec = np.mean([r.n_decisions for r in rows])
            succ = np.mean([r.success for r in rows])
            ct = np.mean([r.censored_completion_time for r in rows])
            t_sim = np.mean([r.end_time for r in rows])
            # Modelo da SPEC-00 §4.1 fato 5.
            model = (t_sim / cfg.effective_update_rate) * (1 + n_a + n_s)
            print(f"{n_a:>3} {n_s:>3} {ev:>11.0f} {model:>9.0f} {sec:>8.4f} "
                  f"{dec:>9.1f} {succ:>8.2f} {ct:>8.1f}")
            cost_rows.append(dict(n_agents=n_a, n_sensors=n_s, events=float(ev),
                                  model=float(model), sec_per_ep=float(sec),
                                  decisions=float(dec), success=float(succ),
                                  completion=float(ct)))

    print()
    print("=" * 78)
    print("(ii) FIDELIDADE — update_rate contra a referência fina (0.01)")
    print("=" * 78)
    print("Política determinística (guloso), mesmas sementes, célula 4x24.")
    print(f"{'update_rate':>12} {'razão':>6} {'eventos/ep':>11} {'s/ep':>8} "
          f"{'ordem idêntica':>15} {'Δt_conc(s)':>11}")

    base_cfg, base_rows, base_wall = run_cell(4, 24, 0.01, N_EPISODES)
    base_order = [tuple(np.argsort(np.nan_to_num(r.visit_times, nan=1e9))) for r in base_rows]
    base_ct = np.array([r.censored_completion_time for r in base_rows])

    fid_rows = []
    for ur in [0.01, 0.025, 0.05, 0.1, 0.125, 0.25, 0.5]:
        cfg, rows, wall = run_cell(4, 24, ur, N_EPISODES)
        order = [tuple(np.argsort(np.nan_to_num(r.visit_times, nan=1e9))) for r in rows]
        same = sum(a == b for a, b in zip(order, base_order))
        ct = np.array([r.censored_completion_time for r in rows])
        dct = float(np.max(np.abs(ct - base_ct)))
        ev = np.mean([r.n_events for r in rows])
        print(f"{ur:>12.3f} {cfg.algorithm_iteration_interval/ur:>6.0f} {ev:>11.0f} "
              f"{wall/len(rows):>8.4f} {same:>10}/{len(rows):<4} {dct:>11.2f}")
        fid_rows.append(dict(update_rate=ur, events=float(ev),
                             sec_per_ep=float(wall / len(rows)),
                             same_order=int(same), n=len(rows), max_dct=dct))

    print()
    print("=" * 78)
    print("(iii) EXTRAPOLAÇÃO — 50 M passos de decisão (orçamento de Lamenza)")
    print("=" * 78)
    ref = [c for c in cost_rows if c["n_agents"] == 4 and c["n_sensors"] == 24][0]
    for c in cost_rows:
        # 1 passo de decisão por agente por época, como no contador do CleanRL.
        steps_per_ep = c["decisions"] * c["n_agents"]
        hours = 50e6 / steps_per_ep * c["sec_per_ep"] / 3600
        c["hours_50M"] = float(hours)
    worst = max(cost_rows, key=lambda c: c["hours_50M"])
    best = min(cost_rows, key=lambda c: c["hours_50M"])
    for c in sorted(cost_rows, key=lambda c: -c["hours_50M"])[:5]:
        print(f"  Na={c['n_agents']} Ns={c['n_sensors']}: {c['hours_50M']:8.1f} h/núcleo")
    print(f"\n  faixa: {best['hours_50M']:.1f} h (Na={best['n_agents']},Ns={best['n_sensors']}) "
          f"a {worst['hours_50M']:.1f} h (Na={worst['n_agents']},Ns={worst['n_sensors']})")
    print("  PRD §6.1 estimava 8-30 h em um núcleo.")

    out = {"cost": cost_rows, "fidelity": fid_rows, "n_episodes": N_EPISODES}
    with open("reports/spec00-orcamento.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nartefato: reports/spec00-orcamento.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
