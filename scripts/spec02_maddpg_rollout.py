"""Baseline MADDPG — tempo de conclusão e trajetórias, dos pesos publicados.

Complementa `spec02_baseline_maddpg.py`, que mede tamanho e latência sem rodar
nada. Aqui a política **roda de verdade** no ambiente de referência, e é o que
fecha a terceira figura de mérito.

    tamanho do modelo   → `spec02_baseline_maddpg.py`
    tempo de inferência → `spec02_baseline_maddpg.py`
    tempo de conclusão  → **aqui**
    trajetórias         → **aqui**

POR QUE NÃO USAR O `test.py` DO SUBMODULE DIRETO

Ele faz quase isto, e serviu de referência para o protocolo. Três razões para
reimplementar:

1. **Os caminhos dele estão desatualizados** — aponta para `runs/results__…`
   quando os checkpoints estão em `runs/results/results__…`. O `plot_paths.py`,
   do mesmo diretório, tem os caminhos certos. Rodar o `test.py` como está
   falha.
2. **`from main import Actor` executa o `main.py`**, que faz `tyro.cli` no nível
   de módulo e tenta configurar `wandb`. Transcrever a classe `Actor` (12 linhas)
   é mais barato que contornar isso.
3. **Queremos os nossos artefatos e a nossa semeadura.** O `run_episode` do
   `refenv` semeia com `random.seed` antes do `reset()`, porque o `reset()` de
   referência ignora o argumento `seed` que declara. Sem isso não há
   reprodutibilidade, e o `test.py` não faz.

A CONFIGURAÇÃO É A MESMA — verificado parâmetro por parâmetro

    algorithm_iteration_interval  0.5     scenario_size            100
    max_seconds_stalled           30      state_num_closest_sensors 12
    min_sensor_priority           1       state_num_closest_drones  Na − 1
    full_random_drone_position    False   speed_action              True

`test.py` **não** passa `end_when_all_collected`, então vale o default `True` do
ambiente — que é exatamente o nosso `phase="eval"`, o mesmo do
`visualize_performance.py` que gerou a fig. 7.3.

Uso:
    uv run --group baseline python scripts/spec02_maddpg_rollout.py
    uv run --group baseline python scripts/spec02_maddpg_rollout.py --episodios 200
    uv run --group baseline python scripts/spec02_maddpg_rollout.py --celulas 2:12,4:24
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec00_demo import salvar_grafico_lamenza  # noqa: E402
from spec02_baseline_maddpg import checkpoints  # noqa: E402
from uavwnn.refenv import LAMENZA_003, make_env, run_episode  # noqa: E402

LARGURA_ATOR, N_ACOES = 256, 2


def carregar_ator(caminho: Path, obs_dim: int):
    """Devolve um callable compatível com o protocolo `Policy` do `refenv`.

    Transcrição do `Actor` do `main.py`. Os buffers `action_scale` e
    `action_bias` **vêm do checkpoint** — são as constantes que mapeiam
    `tanh ∈ [−1,1]` no espaço de ação do ambiente. Reconstruí-los com valores
    inventados daria ações fora de faixa e um baseline silenciosamente errado.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        sys.exit("torch é necessário: uv sync --group baseline")

    torch.set_num_threads(1)

    class Ator(nn.Module):
        def __init__(self, obs: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(obs, LARGURA_ATOR)
            self.fc2 = nn.Linear(LARGURA_ATOR, LARGURA_ATOR)
            self.fc_mu = nn.Linear(LARGURA_ATOR, N_ACOES)
            self.register_buffer("action_scale", torch.zeros(N_ACOES))
            self.register_buffer("action_bias", torch.zeros(N_ACOES))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias

    rede = Ator(obs_dim)
    rede.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True)[0])
    rede.eval()

    def politica(observacao):
        with torch.no_grad():
            return rede(torch.as_tensor(observacao, dtype=torch.float32)).numpy()

    politica.rede = rede            # para inspeção
    return politica


def trilhas_de_um_episodio(cfg, politica, seed: int):
    """Roda um episódio guardando as posições ABSOLUTAS por época de decisão.

    A observação é relativa e normalizada, então não serve para desenhar
    trajetória. As posições saem do simulador
    (`env.simulator.get_node(i).position`), que é o mesmo caminho que o
    `plot_paths.py` de referência usa — e, como só alimenta figura e nunca a
    política, não viola a RF-04.

    Espelha o laço de `rodar_silencioso` do `spec00_demo.py`, inclusive a
    condição de parada na PRIMEIRA `info` não vazia.
    """
    env = make_env(cfg)
    random.seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset()

    sensores = np.array(
        [env.simulator.get_node(i).position[:2] for i in env.sensor_node_ids]
    )
    trilhas: list[list] = [[] for _ in range(cfg.num_drones)]
    info: dict = {}

    while True:
        for k, i in enumerate(env.agent_node_ids):
            trilhas[k].append(np.array(env.simulator.get_node(i).position[:2]))
        obs, _, term, trunc, infos = env.step(
            {a: politica(obs[a]) for a in env.agents}
        )
        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            break
        if any(term.values()) or any(trunc.values()):
            break

    coletado = np.array([
        env.simulator.get_node(i).protocol_encapsulator.protocol.has_collected
        for i in env.sensor_node_ids
    ])
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass
    return np.array([np.array(t) for t in trilhas]), sensores, coletado, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=50,
                    help="por célula; o test.py de referência usa 200")
    ap.add_argument("--celulas", default="",
                    help="subconjunto, ex. '2:12,4:24'; vazio = as 9")
    ap.add_argument("--sem-figuras", action="store_true")
    ap.add_argument("--dir-figuras", default="reports/img/maddpg")
    ap.add_argument("--saida", default="reports/spec02-maddpg-rollout.json")
    a = ap.parse_args()

    disponiveis = {(na, ns): p for na, ns, p in checkpoints()}
    if not disponiveis:
        sys.exit("nenhum checkpoint; git submodule update --init --recursive")

    if a.celulas:
        pedidas = []
        for item in a.celulas.split(","):
            na, ns = (int(x) for x in item.split(":"))
            if (na, ns) not in disponiveis:
                sys.exit(f"sem checkpoint para Na={na}, Ns={ns}")
            pedidas.append((na, ns))
    else:
        pedidas = sorted(disponiveis)

    Path(a.dir_figuras).mkdir(parents=True, exist_ok=True)
    print(f"MADDPG dos pesos publicados · {a.episodios} episódios por célula · "
          f"phase='eval' (= config do test.py)\n")
    print(f"{'célula':>10} {'conclusão (s)':>16} {'sucesso':>9} "
          f"{'coleta média':>13} {'s/ep':>7}")

    artefato = {"config": {"episodios": a.episodios, "phase": "eval",
                           "fonte": "pesos publicados no submodule"},
                "celulas": {}}

    for na, ns in pedidas:
        cfg = LAMENZA_003(na, ns, phase="eval")
        politica = carregar_ator(disponiveis[(na, ns)], cfg.observation_size)

        t0 = time.perf_counter()
        resumos = [run_episode(cfg, politica, seed=s) for s in range(a.episodios)]
        parede = (time.perf_counter() - t0) / a.episodios

        ct = np.array([r.completion_time for r in resumos], dtype=float)
        ok = np.array([r.all_collected for r in resumos], dtype=bool)
        coleta = np.array([np.mean(r.collection_times) if r.collection_times else np.nan
                           for r in resumos], dtype=float)

        # IC95% da média por bootstrap — as células são independentes, então o
        # intervalo é por célula e não pareado.
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(ct, len(ct), replace=True).mean()
                         for _ in range(5_000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        print(f"{f'a_{na}-s_{ns}':>10} {ct.mean():>8.1f} [{lo:.1f}, {hi:.1f}] "
              f"{ok.mean():>8.1%} {np.nanmean(coleta):>12.1f} {parede:>6.2f}s")

        artefato["celulas"][f"a_{na}-s_{ns}"] = {
            "num_drones": na, "num_sensores": ns,
            "completion_time_medio": float(ct.mean()),
            "completion_time_ic95": [float(lo), float(hi)],
            "completion_time_por_episodio": ct.tolist(),
            "taxa_sucesso": float(ok.mean()),
            "coleta_media": float(np.nanmean(coleta)),
            "segundos_por_episodio": float(parede),
        }

        if not a.sem_figuras:
            T, sensores, coletado, _ = trilhas_de_um_episodio(cfg, politica, seed=0)
            # Separação média entre pares de UAVs, o número que mede coordenação.
            # A SPEC-00 mediu ~0,1 m para o `greedy` contra 20 m de alcance: ele
            # voa em formação. Aqui é o contraste direto, e é o que transforma a
            # figura em evidência em vez de ilustração.
            if cfg.num_drones > 1:
                passos = min(len(t) for t in T)
                pares = [np.linalg.norm(T[i][:passos] - T[j][:passos], axis=1)
                         for i in range(cfg.num_drones)
                         for j in range(i + 1, cfg.num_drones)]
                sep = float(np.mean(pares))
                artefato["celulas"][f"a_{na}-s_{ns}"]["separacao_media_m"] = sep
                print(f"           separação média entre UAVs: {sep:.1f} m "
                      f"(greedy mediu ~0,1 m; alcance = 20 m)")
            arq = Path(a.dir_figuras) / f"trajetoria-maddpg-{na}a{ns}s.png"
            classe = type("Args", (), {"agents": na, "sensors": ns, "seed": 0})
            salvar_grafico_lamenza(str(arq), T, sensores, cfg, classe)
            artefato["celulas"][f"a_{na}-s_{ns}"]["figura"] = str(arq)

    with open(a.saida, "w") as f:
        json.dump(artefato, f, indent=2)
    print(f"\nartefato: {a.saida}")
    if not a.sem_figuras:
        print(f"figuras:  {a.dir_figuras}/")
    print("\nRessalva: o guloso da fig. 7.3 foi avaliado a 0,1 s por decisão e o")
    print("MADDPG a 0,5 s. Ao confrontar as duas colunas, declarar a assimetria.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
