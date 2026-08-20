"""Baseline MADDPG — tamanho do modelo e tempo de inferência, dos pesos liberados.

**Não treina nada.** O submodule de referência traz os pesos que Lamenza
publicou: 9 checkpoints, `Na ∈ {2,4,8} × Ns ∈ {12,24,36}`, em
`vendor/gradys-data-collection/scenario/003-collaboration/runs/results/`.
Carregar e medir é minutos; retreinar seriam ~7 dias de GPU por célula.

Isto estreita a **D7** do PRD sem derrubá-la. A razão original dela —
*"reproduzi-lo exigiria replicar exatamente simulador, observação e recompensa;
qualquer divergência invalidaria a comparação silenciosamente"* — **não se
aplica aqui**: não reproduzimos nada, carregamos os pesos dele no ambiente dele.
O que continua valendo é que **não retreinamos**.

O QUE ESTE SCRIPT MEDE, E O QUE NÃO

    tamanho do modelo   → do `state_dict`, exato          MEDIDO aqui
    tempo de inferência → benchmark do forward             MEDIDO aqui
    operações           → contagem analítica das camadas   MEDIDO aqui
    tempo de conclusão  → exige rodar a política no env    **NÃO** — é T-02.x,
                          via `test.py` do submodule

RESSALVA DE JUSTIÇA NA LATÊNCIA

    Os microssegundos daqui são PyTorch em CPU, com sobrecarga de arcabouço e de
    Python. A nossa WNN em NumPy carrega sobrecarga análoga. **A comparação
    honesta e independente de arcabouço é a contagem de operações** (T3), não o
    tempo de parede — e é exatamente por isso que a pendência T-03.6 (paridade
    de arcabouço) existe. Reportar só o tempo de parede favoreceria quem tem a
    biblioteca mais otimizada, não o modelo mais barato.

NÃO HÁ CHECKPOINT PARA `Na = 1`

    As células publicadas começam em 2 agentes. A Etapa 1 do nosso plano começa
    em `Na = 1`, e ali **não existe baseline MADDPG** — a comparação com Lamenza
    só é possível de `Na = 2` em diante.

Uso:
    uv run --group baseline python scripts/spec02_baseline_maddpg.py
    uv run --group baseline python scripts/spec02_baseline_maddpg.py --repeticoes 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

RAIZ_CKPT = Path(
    "vendor/gradys-data-collection/scenario/003-collaboration/runs/results"
)

# Defaults do `main.py` do submodule (linhas 108-109). Não são palpite: são o
# que gerou os pesos que estamos medindo.
LARGURA_ATOR = 256
LARGURA_CRITICO = 512
N_ACOES = 2


def _torch():
    """Importa torch com mensagem útil, em vez de ImportError cru."""
    try:
        import torch
    except ImportError:
        sys.exit(
            "torch é necessário para ler os checkpoints (.cleanrl_model é um\n"
            "arquivo do torch.save). Ele é um grupo opcional para não pesar na\n"
            "instalação padrão:\n\n"
            "    uv sync --group baseline\n"
        )
    return torch


def checkpoints() -> list[tuple[int, int, Path]]:
    """`(Na, Ns, caminho)` de cada checkpoint publicado, ordenado."""
    achados = []
    for p in glob(str(RAIZ_CKPT / "*" / "*.cleanrl_model")):
        # nome: `a_4-s_24-checkpoint4999.cleanrl_model`
        nome = Path(p).name.split("-checkpoint")[0]
        na = int(nome.split("-")[0].removeprefix("a_"))
        ns = int(nome.split("-")[1].removeprefix("s_"))
        achados.append((na, ns, Path(p)))
    return sorted(achados)


def tamanho(caminho: Path) -> dict:
    """Parâmetros e bytes do ator e do crítico, do `state_dict`.

    O checkpoint é uma tupla `(ator, crítico)` — é assim que o `test.py` do
    submodule o lê (`torch.load(...)[0]`).

    `action_scale` e `action_bias` são **buffers registrados**, não parâmetros
    aprendidos: constantes de reescala derivadas do espaço de ação. Contam no
    arquivo mas não no tamanho do modelo, e por isso saem da conta.
    """
    torch = _torch()
    ator, critico = torch.load(caminho, map_location="cpu", weights_only=True)

    def conta(sd, prefixos):
        return sum(int(v.numel()) for k, v in sd.items()
                   if any(k.startswith(p) for p in prefixos))

    p_ator = conta(ator, ("fc1", "fc2", "fc_mu"))
    p_critico = conta(critico, ("fc1", "fc2", "fc3"))
    obs = int(ator["fc1.weight"].shape[1])
    return {
        "obs_dim": obs,
        "params_ator": p_ator,
        "params_critico": p_critico,
        "bytes_ator_fp32": p_ator * 4,
        "bytes_critico_fp32": p_critico * 4,
        "bytes_arquivo": os.path.getsize(caminho),
        "camadas_ator": [obs, LARGURA_ATOR, LARGURA_ATOR, N_ACOES],
    }


def operacoes_ator(obs: int) -> dict:
    """Contagem analítica, mesma convenção de `uavwnn.custo.operacoes`.

    Uma multiplicação e uma soma em ponto flutuante por peso, mais o viés, mais
    uma comparação por unidade oculta (o ReLU). O `tanh` da saída entra como
    transcendental, um por dimensão de ação.
    """
    camadas = [obs, LARGURA_ATOR, LARGURA_ATOR, N_ACOES]
    mult = soma = comp = 0
    for i, (ent, sai) in enumerate(zip(camadas[:-1], camadas[1:])):
        mult += ent * sai
        soma += ent * sai + sai
        if i < len(camadas) - 2:
            comp += sai
    return {"mult_fp": mult, "soma_fp": soma, "comparacoes": comp,
            "transcendental_fp": N_ACOES, "total": mult + soma + comp + N_ACOES}


def latencia(caminho: Path, obs: int, repeticoes: int, aquecimento: int) -> dict:
    """Mediana e p99 do forward, batch 1, uma thread.

    `set_num_threads(1)` porque a decisão de um agente é sequencial: paralelismo
    de biblioteca inflaria a comparação com um modelo que não tem o que
    paralelizar. `no_grad` porque é inferência.
    """
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(1)

    class Ator(nn.Module):
        """Transcrição do `Actor` do `main.py` do submodule."""

        def __init__(self, obs: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(obs, LARGURA_ATOR)
            self.fc2 = nn.Linear(LARGURA_ATOR, LARGURA_ATOR)
            self.fc_mu = nn.Linear(LARGURA_ATOR, N_ACOES)
            self.register_buffer("action_scale", torch.ones(N_ACOES))
            self.register_buffer("action_bias", torch.zeros(N_ACOES))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias

    sd = torch.load(caminho, map_location="cpu", weights_only=True)[0]
    rede = Ator(obs)
    rede.load_state_dict(sd)
    rede.eval()

    x = torch.randn(1, obs)
    with torch.no_grad():
        for _ in range(aquecimento):
            rede(x)
        amostras = np.empty(repeticoes)
        for i in range(repeticoes):
            t0 = time.perf_counter_ns()
            rede(x)
            amostras[i] = time.perf_counter_ns() - t0
    us = amostras / 1000.0
    return {"mediana_us": float(np.median(us)),
            "p99_us": float(np.percentile(us, 99)),
            "min_us": float(us.min())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeticoes", type=int, default=20_000)
    ap.add_argument("--aquecimento", type=int, default=2_000)
    ap.add_argument("--saida", default="reports/spec02-baseline-maddpg.json")
    a = ap.parse_args()

    encontrados = checkpoints()
    if not encontrados:
        sys.exit(
            f"nenhum checkpoint em {RAIZ_CKPT}\n"
            "o submodule está inicializado?  git submodule update --init --recursive"
        )

    print(f"{len(encontrados)} checkpoints publicados · ator "
          f"{LARGURA_ATOR}×{LARGURA_ATOR} · crítico "
          f"{LARGURA_CRITICO}×{LARGURA_CRITICO} (defaults do main.py)\n")
    print(f"{'célula':>10} {'obs':>4} {'params':>8} {'ator':>10} "
          f"{'mediana':>10} {'p99':>9} {'mult fp':>9}")

    artefato = {"config": {"largura_ator": LARGURA_ATOR,
                           "largura_critico": LARGURA_CRITICO,
                           "n_acoes": N_ACOES,
                           "repeticoes": a.repeticoes,
                           "threads": 1, "batch": 1, "dtype": "fp32"},
                "celulas": {}}

    # A latência só depende de `obs`, e `obs` só depende de `Na` — medir uma vez
    # por `Na` em vez de nove vezes evita 6 benchmarks redundantes.
    lat_por_obs: dict[int, dict] = {}

    for na, ns, caminho in encontrados:
        t = tamanho(caminho)
        obs = t["obs_dim"]
        if obs not in lat_por_obs:
            lat_por_obs[obs] = latencia(caminho, obs, a.repeticoes, a.aquecimento)
        lat = lat_por_obs[obs]
        ops = operacoes_ator(obs)

        print(f"{f'a_{na}-s_{ns}':>10} {obs:>4} {t['params_ator']:>8} "
              f"{t['bytes_ator_fp32']/1024:>7.1f} KB "
              f"{lat['mediana_us']:>7.1f} µs {lat['p99_us']:>6.1f} µs "
              f"{ops['mult_fp']:>9}")
        artefato["celulas"][f"a_{na}-s_{ns}"] = {
            "num_drones": na, "num_sensores": ns,
            "tamanho": t, "latencia": lat, "operacoes": ops,
        }

    print("\nA observação confere com a fórmula do PRD `D = 2(Na−1) + 25`:")
    for obs in sorted(lat_por_obs):
        na = (obs - 25) // 2 + 1
        print(f"  Na={na}: {2 * (na - 1) + 25} previsto, {obs} medido")

    print("\nRessalvas, e as duas primeiras precisam ir para o relatório:")
    print("  1. NÃO há checkpoint para Na = 1 — as células começam em 2 agentes.")
    print("  2. A latência é PyTorch em CPU, com sobrecarga de arcabouço. A")
    print("     comparação independente de arcabouço é a contagem de operações")
    print("     (T3), não o tempo de parede. Ver pendência T-03.6.")
    print("  3. Tempo de CONCLUSÃO não sai daqui: exige rodar a política no")
    print("     ambiente, via o `test.py` do submodule.")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.saida, "w") as f:
        json.dump(artefato, f, indent=2)
    print(f"\nartefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
