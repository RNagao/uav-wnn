"""SPEC-00 — Visualização 3D ao vivo do episódio.

O `gradysim` traz um visualizador: o simulador sobe um servidor **WebSocket**
em `localhost:5678` (num processo separado) e um visualizador web hospedado se
conecta a ele, mostrando os nós se movendo em 3D em tempo real.

    1. rode este script
    2. abra https://project-gradys.github.io/gradys-sim-nextgen-visualization/
    3. o visualizador conecta sozinho em localhost:5678

Com `render_mode="visual"` o ambiente de referência roda em **tempo real** — um
episódio de 40 s simulados leva 40 s de relógio. É o que se quer para assistir,
e é a razão de este script não servir para medir nada.

No WSL2 o `localhost` é encaminhado para o Windows, então o navegador do
Windows alcança o servidor normalmente.

Uso:
    uv run python scripts/spec00_visualizar.py
    uv run python scripts/spec00_visualizar.py --agents 8 --sensors 36 --policy smart
    uv run python scripts/spec00_visualizar.py --policy greedy   # veja-os voando colados
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from uavwnn.refenv import LAMENZA_003, load_environment, load_heuristics

VISUALIZADOR = "https://project-gradys.github.io/gradys-sim-nextgen-visualization/"

AZUL = (60, 130, 246)
VERMELHO = (200, 60, 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--sensors", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", choices=["greedy", "smart", "random"], default="greedy")
    ap.add_argument("--espera", type=float, default=12.0,
                    help="segundos para você abrir o navegador antes de começar")
    args = ap.parse_args()

    cfg = LAMENZA_003(args.agents, args.sensors)
    env_mod = load_environment()
    h = load_heuristics()
    policy = {
        "greedy": h.create_greedy_heuristics,
        "smart": h.create_smart_heuristics,
        "random": h.create_random_heuristics,
    }[args.policy](args.agents - 1, cfg.state_num_closest_sensors)

    env = env_mod.GrADySEnvironment(render_mode="visual", **cfg.kwargs())
    random.seed(args.seed)
    np.random.seed(args.seed)
    obs, _ = env.reset()

    # Deixa os papéis legíveis: UAVs azuis e maiores, sensores vermelhos.
    # O ambiente de referência já pinta de verde o sensor no instante da coleta.
    try:
        env.controller.resize_nodes(3)
        for i in env.agent_node_ids:
            env.controller.paint_node(i, AZUL)
        for i in env.sensor_node_ids:
            env.controller.paint_node(i, VERMELHO)
        for i in env.agent_node_ids:
            env.controller.show_node_id(i, True)
    except Exception as e:  # o controlador é best-effort
        print(f"(não foi possível estilizar os nós: {e})")

    print(f"\n  servidor WebSocket em localhost:5678")
    print(f"  abra:  {VISUALIZADOR}\n")
    print(f"  {args.agents} UAVs (azul) · {args.sensors} sensores (vermelho → verde ao coletar)")
    print(f"  política '{args.policy}' · semente {args.seed} · roda em tempo real\n")
    for restante in range(int(args.espera), 0, -1):
        print(f"\r  começando em {restante:2d}s ...", end="", flush=True)
        time.sleep(1)
    print("\r  começando agora.        \n")

    passo = 0
    while True:
        obs, recompensas, term, trunc, infos = env.step(
            {a: policy(obs[a]) for a in env.agents}
        )
        passo += 1
        col = int(env.sensors_collected)
        print(f"\r  t={passo * cfg.algorithm_iteration_interval:6.1f}s   "
              f"coletados {col:>3}/{cfg.num_sensors}", end="", flush=True)

        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            break
        if any(term.values()) or any(trunc.values()):
            info = {}
            break

    print(f"\n\n  fim: {info.get('cause')} · conclusão em {info.get('completion_time', float('nan')):.1f} s")
    print("  (o visualizador congela no último quadro; feche a aba quando quiser)")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
