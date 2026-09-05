"""Figuras de mérito WNN × MADDPG — custo (T1 memória, T3 operações).

Usa o WNN **real treinado** (SPEC-06, célula 1/12) e o harness de custo da
SPEC-03 (`uavwnn.custo`) — nada de número inventado. Contrasta com o ator MADDPG
(256×256) nas células publicadas do cenário 003 de Lamenza e com o MLP pareado da
mesma célula (o controle de método da SPEC-05).

**Ressalva de comparação (célula):** não há célula em comum entre o WNN treinado
(Na=1, Ns=12) e os checkpoints MADDPG (Na∈{2,4,8}). O custo é uma propriedade do
modelo pouco sensível ao cenário; a comparação é honesta como ordem de grandeza,
com a diferença de célula anotada. O MLP pareado é medido na MESMA célula do WNN.

O achado é um **trade-off de dois eixos**, não um vencedor único:
- **T3 (computação):** a WNN elimina os MACs (~0 mult. fp) contra ~73k do MADDPG.
- **T1 (memória):** a WNN paga — as LUTs treinadas ocupam mais que os pesos densos.

    uv run python scripts/spec06_figuras_merito.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar  # noqa: E402
from uavwnn.custo.memoria import custo_mlp, custo_ntuple_de_rede  # noqa: E402
from uavwnn.custo.operacoes import (  # noqa: E402
    CICLOS, ciclos, operacoes_mlp, operacoes_ntuple,
)
from uavwnn.refenv import LAMENZA_003  # noqa: E402

DIR = Path("reports/img/spec06-merito")
TEAL, AMBAR, CINZA, VERM = "#1C7293", "#F2A93B", "#808A96", "#B0637C"

# WNN treinados (SPEC-06), célula 1/12: T=32, n=12, N=67, D=25, d_saida=3.
WNN = {
    # `models/` e não `runs/`: `runs/` é gitignored e some entre máquinas — o
    # script silenciosamente pulava a WNN e a tabela saía SÓ com MADDPG, sem
    # avisar. Os checkpoints versionados são a fonte estável.
    "T=32 (1/12)": "models/T32/a1s12",
    "T=16 (1/12)": "models/T16/a1s12",
}
FAMILIA = {k: "ntupla_dissertacao" for k in WNN}
RESOLUCAO = {"T=32 (1/12)": 32, "T=16 (1/12)": 16}
N_TUPLA, D_OBS, D_SAIDA = 12, 25, 3
T, N_RAM = 32, 67          # da célula de referência, para o T2 (ciclos)
# Ator MADDPG (256×256), obs por célula (D depende de Na).
MADDPG_D = {2: 27, 4: 31, 8: 39}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def coletar() -> dict:
    cfg = LAMENZA_003(1, 12, phase="eval")
    m: dict = {"wnn": {}, "maddpg": {}, "mlp_pareada": {}}

    for nome, run in WNN.items():
        ck = Path(run)
        if not ck.exists():
            print(f"  aviso: {ck} ausente — {nome} fica FORA da tabela", flush=True)
            continue
        _, ap = montar(FAMILIA[nome], cfg, ck)
        # As TRÊS variantes, e a `ordenado` como principal.
        #
        # O `hash` era o número publicado, e ele paga arredondamento de capacidade
        # para potência de 2 com fator de carga 0,75: 1 059 entradas ocupadas viram
        # 2 048 posições alocadas. Isso inflava o T1 em 1,85× — e o T1 é justamente
        # a figura em que a WNN PERDE, então inflá-lo é errar contra nós mesmos.
        #
        # A `ordenado` (chaves ordenadas + busca binária) gasta exatamente
        # `ocupadas × (chave + valores)`, sem folga. O preço é busca O(log n) em vez
        # de O(1) — ~11 comparações por RAM aqui — e por isso as três ficam
        # gravadas: quem defender latência tem de citar o `hash`.
        t_res = RESOLUCAO[nome]
        var = {v: custo_ntuple_de_rede(ap.ator, variante=v, limiares_por_dim=t_res,
                                       n_dims=D_OBS, rotulo=nome)
               for v in ("ordenado", "hash", "denso")}
        cm = var["ordenado"]
        op = operacoes_ntuple(n_ram=ap.ator.n_ram, tamanho_tupla=N_TUPLA,
                              n_cabecas=D_SAIDA, n_dims=D_OBS, n_niveis=t_res)
        m["wnn"][nome] = {
            "persistente_kb": cm.persistente / 1024,
            "variante_principal": "ordenado",
            "persistente_kb_por_variante": {v: c.persistente / 1024
                                            for v, c in var.items()},
            "residente_kb": cm.residente / 1024,
            "entradas_ocupadas": int(ap.ator.entradas_ocupadas()),
            "mult_fp": op.mult_fp, "soma_fp": op.soma_fp,
            "lookups": op.lookups, "soma_int": op.soma_int,
            "total_sem_mac": op.soma_fp + op.lookups + op.soma_int,
        }

    for na, d in MADDPG_D.items():
        cm = custo_mlp([d, 256, 256, 2], rotulo=f"MADDPG Na={na}")
        op = operacoes_mlp([d, 256, 256, 2])
        m["maddpg"][f"Na={na}"] = {"persistente_kb": cm.persistente / 1024,
                                   "mult_fp": op.mult_fp, "D": d}

    cmp = custo_mlp([D_OBS, 32, 32, 2], rotulo="MLP pareada 1/12")
    opp = operacoes_mlp([D_OBS, 32, 32, 2])
    m["mlp_pareada"] = {"persistente_kb": cmp.persistente / 1024, "mult_fp": opp.mult_fp}

    # T2 — tempo de inferência por CICLOS (independente de hardware, RNF-12): a
    # figura honesta, porque wall-clock depende do arcabouço. WNN real vs ator
    # MADDPG (256×256, saída 2 como Lamenza).
    ow = operacoes_ntuple(n_ram=N_RAM, tamanho_tupla=N_TUPLA, n_cabecas=D_SAIDA,
                          n_dims=D_OBS, n_niveis=T)
    om = operacoes_mlp([MADDPG_D[4], 256, 256, 2])
    m["t2_ciclos"] = {hw: {"wnn": ciclos(ow, hw), "maddpg": ciclos(om, hw),
                           "ganho_wnn": ciclos(om, hw) / max(ciclos(ow, hw), 1)}
                      for hw in CICLOS}
    # T2 wall-clock MEDIDO nos protótipos (RNF-11): WNN NumPy × MADDPG PyTorch —
    # arcabouços diferentes; serve só para mostrar que o wall-clock inverte o
    # veredito dos ciclos por overhead de implementação, não por algoritmo.
    m["t2_wallclock_us"] = {
        "wnn_numpy": {"mediana": 173.0, "p99": 321.8},
        "maddpg_pytorch": {"mediana": 35.0, "p99": 115.1},
        "nota": "NumPy×PyTorch; a figura defensável é ciclos, não wall-clock",
    }
    return m


def tabela(m: dict) -> None:
    print("Figuras de mérito — custo por decisão e memória residente\n")
    print(f"{'modelo':>22} {'mult_fp (MAC)':>14} {'T1 memória':>12}")
    for nome, w in m["wnn"].items():
        print(f"{'WNN '+nome:>22} {w['mult_fp']:>14,} {w['persistente_kb']:>9.1f} KB")
    print(f"{'MLP pareada (1/12)':>22} {m['mlp_pareada']['mult_fp']:>14,} "
          f"{m['mlp_pareada']['persistente_kb']:>9.1f} KB")
    for na, mm in m["maddpg"].items():
        print(f"{'MADDPG '+na:>22} {mm['mult_fp']:>14,} {mm['persistente_kb']:>9.1f} KB")
    d = m["wnn"].get("T=32 (1/12)")
    if d:
        maddpg4 = m["maddpg"]["Na=4"]
        print(f"\nLeitura: a WNN elimina os MACs — {d['mult_fp']} mult. fp contra "
              f"{maddpg4['mult_fp']:,} do MADDPG ({d['mult_fp']/maddpg4['mult_fp']:.4%}).")
        pv = d["persistente_kb_por_variante"]
        print(f"Paga em memória: {d['persistente_kb']:.0f} KB (LUTs, "
              f"{d['entradas_ocupadas']:,} entradas, variante ordenado) contra "
              f"{maddpg4['persistente_kb']:.0f} KB do ator MADDPG "
              f"({d['persistente_kb']/maddpg4['persistente_kb']:.1f}×). "
              f"Trade-off memória↔computação.")
        print(f"  por variante de armazenamento: "
              + " · ".join(f"{v} {kb:.0f} KB" for v, kb in pv.items())
              + "  (a `hash` compra O(1) pagando arredondamento de capacidade)")
    print("\nT2 — ciclos por decisão (independente de hardware, RNF-12):")
    for hw, c in m["t2_ciclos"].items():
        print(f"  {hw:>26}: WNN {c['wnn']:>8,}  MADDPG {c['maddpg']:>10,}  "
              f"→ {c['ganho_wnn']:.0f}× WNN")
    wc = m["t2_wallclock_us"]
    print(f"  wall-clock (protótipos): WNN {wc['wnn_numpy']['mediana']:.0f} µs (NumPy) × "
          f"MADDPG {wc['maddpg_pytorch']['mediana']:.0f} µs (PyTorch) — inverte por implementação")


def figuras(m: dict) -> None:
    plt = _plt()
    DIR.mkdir(parents=True, exist_ok=True)
    diss = m["wnn"].get("T=32 (1/12)")

    # 1. T3 — multiplicações fp (MACs) por decisão, log
    fig, ax = plt.subplots(figsize=(8, 5))
    rot, val, cor = [], [], []
    if diss:
        rot.append("WNN\nT=32"); val.append(max(diss["mult_fp"], 0.5)); cor.append(AMBAR)
    rot.append("paired MLP\n(1/12)"); val.append(m["mlp_pareada"]["mult_fp"]); cor.append(CINZA)
    for na, mm in m["maddpg"].items():
        rot.append(f"MADDPG\n{na}"); val.append(mm["mult_fp"]); cor.append(TEAL)
    ax.bar(rot, val, color=cor)
    ax.set_yscale("log"); ax.set_ylabel("fp multiplications per decision (log)")
    ax.set_title("T3 — the WNN eliminates MACs (0 fp mult; bar pinned at 0.5 for log)")
    fig.tight_layout(); fig.savefig(DIR / "t3_operacoes.png", dpi=110); plt.close(fig)

    # 2. T1 — memória persistente (KB), log
    fig, ax = plt.subplots(figsize=(8, 5))
    rot, val, cor = [], [], []
    for nome, w in m["wnn"].items():
        rot.append(f"WNN\n{nome}"); val.append(w["persistente_kb"]); cor.append(AMBAR)
    rot.append("paired MLP\n(1/12)"); val.append(m["mlp_pareada"]["persistente_kb"]); cor.append(CINZA)
    for na, mm in m["maddpg"].items():
        rot.append(f"MADDPG\n{na}"); val.append(mm["persistente_kb"]); cor.append(TEAL)
    ax.bar(rot, val, color=cor)
    ax.set_yscale("log"); ax.set_ylabel("persistent memory (KB, log)")
    ax.set_title("T1 — the WNN pays in memory (trained LUTs > dense weights)\n"
                 "WNN in the `ordered` variant; the `hash` costs 1.85× more")
    fig.tight_layout(); fig.savefig(DIR / "t1_memoria.png", dpi=110); plt.close(fig)

    # 3. trade-off: memória × computação (o mapa de mérito)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if diss:
        ax.scatter(diss["mult_fp"] + 0.5, diss["persistente_kb"], s=140, color=AMBAR,
                   zorder=3, label="WNN T=32 (1/12)")
    ax.scatter(m["mlp_pareada"]["mult_fp"], m["mlp_pareada"]["persistente_kb"],
               s=120, color=CINZA, marker="s", zorder=3, label="paired MLP (1/12)")
    for na, mm in m["maddpg"].items():
        ax.scatter(mm["mult_fp"], mm["persistente_kb"], s=120, color=TEAL, marker="^", zorder=3)
    ax.annotate("MADDPG\n(Na=2,4,8)", (list(m["maddpg"].values())[1]["mult_fp"],
                list(m["maddpg"].values())[1]["persistente_kb"]),
                textcoords="offset points", xytext=(8, 8), color=TEAL)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("computation — fp mult per decision (log)")
    ax.set_ylabel("persistent memory (KB, log)")
    ax.set_title("Merit map: the WNN trades memory for computation")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "mapa_merito.png", dpi=110); plt.close(fig)

    # 4. T2 — ciclos por decisão, por classe de hardware (log)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    hws = list(m["t2_ciclos"].keys())
    x = np.arange(len(hws)); w = 0.38
    ax.bar(x - w / 2, [m["t2_ciclos"][h]["wnn"] for h in hws], w,
           color=AMBAR, label="WNN (thesis)")
    ax.bar(x + w / 2, [m["t2_ciclos"][h]["maddpg"] for h in hws], w,
           color=TEAL, label="MADDPG 256×256")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([h.split(" (")[0] for h in hws])
    ax.set_ylabel("cycles per decision (log)")
    ax.set_title("T2 — cycles per decision: WNN wins, even more without an FPU (470× on the M0)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "t2_ciclos.png", dpi=110); plt.close(fig)
    print(f"[figuras] 4 imagens em {DIR}/")


def main() -> int:
    m = coletar()
    tabela(m)
    try:
        figuras(m)
    except ImportError:
        print("[figuras] matplotlib ausente — pulei. Rode: uv sync --group dev")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/spec06-merito.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print("artefato: reports/spec06-merito.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
