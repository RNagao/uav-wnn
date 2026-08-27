"""Custo dos três modelos: memória, operações e tempo de inferência.

Mede os TRÊS na mesma execução e com o mesmo protocolo. Juntar números de
origens diferentes — T1 de um relatório, latência de outro — é como a razão de
tamanho acabou comparando checkpoints com orçamentos de treino distintos.

TRÊS EIXOS, e eles não concordam:

    T1 memória    quanto ocupa            → o MADDPG ganha
    T3 operações  quanto calcula          → a WNN ganha por 4 ordens
    T2 latência   quanto demora           → depende do que se mede

O T2 tem duas leituras. Em CICLOS (independente de hardware) a WNN ganha; em
wall-clock ela perde, porque compara NumPy com PyTorch — arcabouços, não
algoritmos. As duas ficam na tabela, rotuladas.

    uv run --group baseline python scripts/spec06_custo_comparado.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

CELULAS = ((1, 12), (2, 24))
REPS = 3000


def rss_mb() -> float:
    linha = [l for l in open("/proc/self/status") if l.startswith("VmRSS")][0]
    return int(linha.split()[1]) / 1024


def bytes_em_disco(d: Path) -> int:
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


def medir_wnn(rot: str, na: int, ns: int) -> dict | None:
    from spec06_campanha_wnn import montar

    from uavwnn.custo.memoria import custo_ntuple_de_rede
    from uavwnn.custo.operacoes import CICLOS, ciclos, operacoes_ntuple
    from uavwnn.refenv import LAMENZA_003, make_env, prune_logging_handlers

    T = int(rot.split("T")[1])
    ck = Path(f"models/deploy/T{T}/a{na}s{ns if na > 1 else 12}")
    if not ck.exists():
        return None
    cfg = LAMENZA_003(na, ns, phase="eval")

    gc.collect(); antes = rss_mb()
    pol, ap = montar("ntupla_dissertacao", cfg, ck, plana=True)
    gc.collect(); depois = rss_mb()

    n_ram, n_tup = ap.ator.n_ram, ap.ator.tamanho_tupla
    occ = ap.ator.chaves.size
    d_obs = cfg.observation_size
    cm = custo_ntuple_de_rede(ap.ator, variante="ordenado", limiares_por_dim=T,
                              n_dims=d_obs, rotulo=rot)
    op = operacoes_ntuple(n_ram=n_ram, tamanho_tupla=n_tup, n_cabecas=3,
                          n_dims=d_obs, n_niveis=T)

    import logging
    H = list(logging.getLogger().handlers)
    env = make_env(cfg); obs, _ = env.reset(); prune_logging_handlers(H)
    x = obs[next(iter(obs))]
    for _ in range(200):
        pol(x)
    t = np.empty(REPS)
    for i in range(REPS):
        t0 = time.perf_counter_ns(); pol(x); t[i] = time.perf_counter_ns() - t0
    try:
        env.close()
    except Exception:                                    # pragma: no cover
        pass
    return {"modelo": rot, "na": na, "ns": ns,
            "params": int(occ), "t1_kb": cm.persistente / 1024,
            "disco_kb": bytes_em_disco(ck) / 1024, "rss_mb": depois - antes,
            "mult_fp": op.mult_fp, "soma_fp": op.soma_fp,
            "lookups": op.lookups, "soma_int": op.soma_int,
            "ciclos": {hw: ciclos(op, hw) for hw in CICLOS},
            "us_med": float(np.median(t) / 1000),
            "us_p99": float(np.percentile(t, 99) / 1000)}


def medir_maddpg(na: int, ns: int) -> dict | None:
    from spec02_baseline_maddpg import checkpoints, operacoes_ator, tamanho
    from spec02_maddpg_rollout import carregar_ator

    from uavwnn.custo.memoria import custo_mlp
    from uavwnn.custo.operacoes import CICLOS, ciclos, operacoes_mlp
    from uavwnn.refenv import LAMENZA_003

    disp = {(a, s): p for a, s, p in checkpoints()}
    if (na, ns) not in disp:
        return None
    cfg = LAMENZA_003(na, ns, phase="eval")
    d = cfg.observation_size

    gc.collect(); antes = rss_mb()
    pol = carregar_ator(disp[(na, ns)], d)
    gc.collect(); depois = rss_mb()

    cst = tamanho(disp[(na, ns)])
    cm = custo_mlp([d, 256, 256, 2], rotulo="maddpg")
    op = operacoes_mlp([d, 256, 256, 2])

    import torch
    x = torch.randn(1, d)
    with torch.no_grad():
        for _ in range(200):
            pol.rede(x)
        t = np.empty(REPS)
        for i in range(REPS):
            t0 = time.perf_counter_ns(); pol.rede(x); t[i] = time.perf_counter_ns() - t0
    return {"modelo": "maddpg", "na": na, "ns": ns,
            "params": cst["params_ator"], "t1_kb": cm.persistente / 1024,
            "disco_kb": disp[(na, ns)].stat().st_size / 1024,
            "rss_mb": depois - antes,
            "mult_fp": op.mult_fp, "soma_fp": op.soma_fp,
            "lookups": 0, "soma_int": 0,
            "ciclos": {hw: ciclos(op, hw) for hw in CICLOS},
            "us_med": float(np.median(t) / 1000),
            "us_p99": float(np.percentile(t, 99) / 1000)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=Path("reports/spec06-custo-comparado.json"))
    a = ap.parse_args()

    linhas = []
    for na, ns in CELULAS:
        for rot in ("maddpg", "wnn-T16", "wnn-T32"):
            r = medir_maddpg(na, ns) if rot == "maddpg" else medir_wnn(rot, na, ns)
            if r:
                linhas.append(r)

    p = print
    def cel(r):
        return f"{r['na']}d/{r['ns']}s"
    p("\n=== T1 — MEMÓRIA ===\n")
    p(f"  {'célula':>9} {'modelo':<9}{'params':>10}{'T1 (KB)':>10}"
      f"{'disco (KB)':>12}{'RSS (MB)':>10}")
    for r in linhas:
        p(f"  {cel(r):>9} {r['modelo']:<9}"
          f"{r['params']:>10,}{r['t1_kb']:>10.0f}{r['disco_kb']:>12.0f}{r['rss_mb']:>10.1f}")

    p("\n=== T3 — OPERAÇÕES POR DECISÃO ===\n")
    p(f"  {'célula':>9} {'modelo':<9}{'mult. fp':>11}{'soma fp':>10}"
      f"{'consultas':>11}{'soma int':>10}")
    for r in linhas:
        p(f"  {cel(r):>9} {r['modelo']:<9}{r['mult_fp']:>11,}"
          f"{r['soma_fp']:>10,}{r['lookups']:>11,}{r['soma_int']:>10,}")

    p("\n=== T2 — TEMPO DE INFERÊNCIA ===\n")
    hws = list(linhas[0]["ciclos"])
    p(f"  {'célula':>9} {'modelo':<9}" + "".join(f"{h.split(' ')[0]:>16}" for h in hws)
      + f"{'wall (µs)':>12}{'p99':>9}")
    for r in linhas:
        p(f"  {cel(r):>9} {r['modelo']:<9}"
          + "".join(f"{r['ciclos'][h]:>16,}" for h in hws)
          + f"{r['us_med']:>12.1f}{r['us_p99']:>9.1f}")
    p("\n  ciclos: independente de hardware (RNF-12). wall-clock: NumPy × PyTorch,")
    p("  arcabouços diferentes — serve para mostrar que a implementação inverte o")
    p("  veredito dos ciclos, não para comparar algoritmos.")

    a.json.write_text(json.dumps(linhas, indent=2, ensure_ascii=False))
    p(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
