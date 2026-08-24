"""Imagem do path — trajetórias de uma política WNN num cenário (1 episódio).

Roda a política (compartilhada por drone) num episódio e desenha as trajetórias
dos drones + os sensores (coletados × não coletados). Útil para inspeção
qualitativa; o veredito quantitativo é a avaliação pareada, não a figura.

    uv run python scripts/spec06_path.py --checkpoint <dir> --na 4 --ns 24 \
        --seed 0 --saida reports/img/spec06-path/a4s24.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar  # noqa: E402
from uavwnn.refenv import LAMENZA_003, make_env  # noqa: E402

CORES = ["#1C7293", "#F2A93B", "#B0637C", "#6A994E",
         "#8E7DBE", "#D1495B", "#4A6FA5", "#E07A5F"]


def rodar_e_gravar(cfg, pol, seed: int):
    random.seed(seed); np.random.seed(seed)
    env = make_env(cfg); obs, _ = env.reset()
    aids = list(env.agent_node_ids)
    traj = {aid: [tuple(env.simulator.get_node(aid).position[:2])] for aid in aids}
    passos_max = int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2
    ct, ok = float(cfg.max_episode_length), False
    for _ in range(passos_max):
        obs, _, term, trunc, infos = env.step({a: pol(obs[a]) for a in env.agents})
        for aid in aids:
            traj[aid].append(tuple(env.simulator.get_node(aid).position[:2]))
        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            ct = float(info.get("completion_time", ct))
            ok = bool(info.get("all_collected", False))
            break
        if any(term.values()) or any(trunc.values()):
            break
    sensores = []
    for sid in env.sensor_node_ids:
        n = env.simulator.get_node(sid)
        sensores.append((tuple(n.position[:2]),
                         bool(n.protocol_encapsulator.protocol.has_collected)))
    return traj, sensores, ct, ok


def plotar(traj, sensores, ct, ok, saida: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    naovis = [p for p, c in sensores if not c]
    vis = [p for p, c in sensores if c]
    if naovis:
        xy = np.array(naovis)
        ax.scatter(xy[:, 0], xy[:, 1], facecolors="none", edgecolors="#B0637C",
                   s=55, linewidths=1.3, label="sensor não coletado", zorder=2)
    if vis:
        xy = np.array(vis)
        ax.scatter(xy[:, 0], xy[:, 1], c="#6A994E", s=45, marker="o",
                   label="sensor coletado", zorder=2)
    for i, (aid, pts) in enumerate(traj.items()):
        p = np.array(pts); c = CORES[i % len(CORES)]
        ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.7, alpha=0.9,
                label=f"drone{i}", zorder=3)
        ax.scatter(*p[0], c=c, s=110, marker="*", edgecolors="k", zorder=4)
        ax.scatter(*p[-1], c=c, s=55, marker="s", edgecolors="k", zorder=4)
    ncol = len(vis)
    ax.set_title(f"{titulo}\ncoletados {ncol}/{len(sensores)} · "
                 f"{'SUCESSO' if ok else 'incompleto'} · CT {ct:.0f} s"
                 f"   (★ início · ■ fim)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(saida, dpi=120); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--na", type=int, required=True)
    ap.add_argument("--ns", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", type=Path, required=True)
    a = ap.parse_args()
    cfg = LAMENZA_003(a.na, a.ns, phase="eval")
    pol, _ = montar("ntupla_dissertacao", cfg, a.checkpoint)
    traj, sensores, ct, ok = rodar_e_gravar(cfg, pol, a.seed)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    plotar(traj, sensores, ct, ok, a.saida,
           f"WNN Na={a.na} Ns={a.ns} (semente {a.seed})")
    print(f"coletados {sum(1 for _, c in sensores if c)}/{len(sensores)} · "
          f"sucesso={ok} · CT {ct:.1f} s")
    print(f"imagem: {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
