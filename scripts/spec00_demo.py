"""SPEC-00 — Demo interativa do ambiente de referência.

Serve dois propósitos: deixar você inspecionar um episódio passo a passo, e
mostrar o laço cru de condução do ambiente — que é o mesmo que o
`visualize_performance.py` de referência usa.

Uso:
    uv run python scripts/spec00_demo.py
    uv run python scripts/spec00_demo.py --agents 8 --sensors 36 --seed 3
    uv run python scripts/spec00_demo.py --policy random
    uv run python scripts/spec00_demo.py --plot trajetorias.png
    uv run python scripts/spec00_demo.py --show-obs
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from uavwnn.refenv import LAMENZA_003, SPAWN_HALF_WIDTH, load_heuristics, make_env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--sensors", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", choices=["greedy", "smart", "random"], default="greedy")
    ap.add_argument("--phase", choices=["eval", "eval_heuristic", "train"], default="eval")
    ap.add_argument("--plot", metavar="ARQUIVO.png", help="salva o gráfico das trajetórias")
    ap.add_argument("--show-obs", action="store_true", help="imprime o vetor de observação do drone 0")
    ap.add_argument("--every", type=int, default=10, help="imprime uma linha a cada N passos")
    args = ap.parse_args()

    cfg = LAMENZA_003(args.agents, args.sensors, phase=args.phase)
    h = load_heuristics()
    n_a = args.agents - 1
    policy = {
        "greedy": h.create_greedy_heuristics,
        "smart": h.create_smart_heuristics,
        "random": h.create_random_heuristics,
    }[args.policy](n_a, cfg.state_num_closest_sensors)

    print(f"cenário   Na={cfg.num_drones}  Ns={cfg.num_sensors}  "
          f"alcance={cfg.communication_range:.0f} m  área={2*cfg.scenario_size:.0f}x{2*cfg.scenario_size:.0f} m")
    print(f"decisão   {cfg.algorithm_iteration_interval} s   "
          f"observação {cfg.observation_size} valores   política '{args.policy}'   semente {args.seed}")
    print(f"fase      {args.phase}  (end_when_all_collected={cfg.end_when_all_collected})\n")

    # -- laço cru: é este que o visualize_performance.py de referência usa ----
    env = make_env(cfg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    obs, _ = env.reset()

    # A verdade é lida direto dos nós do simulador. Isto é legítimo AQUI porque
    # é instrumentação: a política nunca vê estas posições, só o `obs`.
    sensores = np.array([env.simulator.get_node(i).position[:2] for i in env.sensor_node_ids])
    trilhas: list[list[np.ndarray]] = [[] for _ in range(cfg.num_drones)]

    def posicoes():
        return np.array([env.simulator.get_node(i).position[:2] for i in env.agent_node_ids])

    for k, pos in enumerate(posicoes()):
        trilhas[k].append(pos.copy())

    print(f"{'passo':>6} {'t (s)':>7} {'coletados':>10} {'recompensa':>11}  drone 0 (x, y)")
    print("-" * 62)

    passo = 0
    while True:
        acoes = {a: policy(obs[a]) for a in env.agents}
        obs, recompensas, term, trunc, infos = env.step(acoes)
        passo += 1

        for k, pos in enumerate(posicoes()):
            trilhas[k].append(pos.copy())

        r = float(next(iter(recompensas.values())))
        col = int(env.sensors_collected)
        t = passo * cfg.algorithm_iteration_interval
        p0 = trilhas[0][-1]

        if passo % args.every == 0 or col == cfg.num_sensors:
            print(f"{passo:>6} {t:>7.1f} {col:>6}/{cfg.num_sensors:<3} {r:>11.3f}  "
                  f"({p0[0]:>7.1f}, {p0[1]:>7.1f})")
            if args.show_obs:
                v = obs["drone0"]
                print(f"         obs[{v.size}]  min={v.min():+.3f} max={v.max():+.3f}  "
                      f"padding(-1): {int(np.isclose(v, -1.0).sum())} de {v.size}")

        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            break
        if any(term.values()) or any(trunc.values()):
            info = {}
            break

    print("-" * 62)
    print(f"causa do fim        {info.get('cause')}")
    print(f"todos coletados     {info.get('all_collected')}")
    print(f"tempo de conclusão  {info.get('completion_time'):.2f} s")
    print(f"passos de decisão   {passo}")
    print(f"distância voada     {sum(np.linalg.norm(np.diff(np.array(t), axis=0), axis=1).sum() for t in trilhas):.0f} m "
          f"(soma dos {cfg.num_drones} UAVs)")

    if args.plot:
        salvar_grafico(args.plot, trilhas, sensores, env, cfg, args)
    else:
        print("\n(use --plot trajetorias.png para ver os caminhos)")

    env.close()
    return 0


def salvar_grafico(arquivo, trilhas, sensores, env, cfg, args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("\nmatplotlib não instalado. Rode: uv add --dev matplotlib")
        return

    coletado = np.array([
        env.simulator.get_node(i).protocol_encapsulator.protocol.has_collected
        for i in env.sensor_node_ids
    ])
    S = cfg.scenario_size
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # Sensores: preenchidos = coletados, vazios = perdidos.
    ax.scatter(sensores[~coletado, 0], sensores[~coletado, 1], s=42, facecolors="none",
               edgecolors="#B03A48", linewidths=1.4, label="sensor não coletado", zorder=3)
    ax.scatter(sensores[coletado, 0], sensores[coletado, 1], s=26, c="#2B2F36",
               marker="x", linewidths=1.3, label="sensor coletado", zorder=3)

    cores = plt.cm.tab10.colors
    for k, t in enumerate(trilhas):
        t = np.array(t)
        cor = cores[k % 10]
        # A faixa larga é o alcance de comunicação varrido — mesma leitura da
        # fig. 7.4 de Lamenza, cujas áreas semitransparentes são esse raio.
        ax.plot(t[:, 0], t[:, 1], color=cor, lw=2 * cfg.communication_range * 72 / (2.2 * S * 6.4),
                alpha=0.13, solid_capstyle="round", zorder=1)
        ax.plot(t[:, 0], t[:, 1], color=cor, lw=1.3, label=f"UAV {k}", zorder=2)
        ax.plot(t[0, 0], t[0, 1], "o", color=cor, ms=5, mec="white", mew=1, zorder=4)

    # Área de largada: quadrado de lado 2*SPAWN_HALF_WIDTH (o original usa
    # random.uniform(-2, 2), não um disco).
    ax.add_patch(Rectangle((-SPAWN_HALF_WIDTH, -SPAWN_HALF_WIDTH),
                          2 * SPAWN_HALF_WIDTH, 2 * SPAWN_HALF_WIDTH,
                          fill=False, ec="#8A929C", ls=":", lw=1.1, zorder=4))
    ax.set_xlim(-S * 1.02, S * 1.02); ax.set_ylim(-S * 1.02, S * 1.02)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{cfg.num_drones} UAVs · {cfg.num_sensors} sensores · política {args.policy} · semente {args.seed}",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.16, lw=0.6)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=160)
    print(f"\ngráfico salvo: {arquivo}")
    print("compare com a fig. 7.4 de Lamenza:")
    print("  vendor/gradys-data-collection/scenario/003-collaboration/statistics/paths/")


if __name__ == "__main__":
    sys.exit(main())
