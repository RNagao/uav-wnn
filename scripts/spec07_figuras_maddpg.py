"""Figuras de mérito e imagens da run do MADDPG sob ruído.

Consome os artefatos JSON já gerados (sem re-rodar o simulador — o princípio da
SPEC-10: figura vem de artefato, não de nova medição) e produz:

- **Figuras de mérito com o filtro embutido (RNF-05 / SPEC-08 T-08.10):** T1
  (estado do KF somado ao ator) e T3 (flops do KF contra os MACs do MADDPG),
  impressas como tabela e como figura. O ponto: o custo do filtro é uma
  **constante pequena e fixa**, independente da capacidade do modelo.
- **Imagens** em `reports/img/spec07-maddpg/`: comparação com Lamenza, degradação
  × σ, NEES no ambiente real, RMSE do filtro.

    uv run python scripts/spec07_figuras_maddpg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavwnn.fusion.kalman import FiltroKalmanControle
from uavwnn.fusion.consistencia import faixa_chi2_individual

# MADDPG, dos pesos publicados (reports/spec02-baseline-maddpg.md).
MADDPG_MULT_FP = {2: 72960, 4: 73984, 8: 76032}   # multiplicações fp por decisão (ator)
MADDPG_ATOR_KB = {2: 287.0, 4: 291.0, 8: 299.0}   # tamanho do ator, fp32

DIR = Path("reports/img/spec07-maddpg")
TEAL, AMBAR, CINZA, VERM = "#1C7293", "#F2A93B", "#808A96", "#B0637C"


def _ler(p):
    return json.loads(Path(p).read_text())


def figuras_de_merito() -> dict:
    """T1 e T3 do KF embutido, relativas ao MADDPG (RNF-05).

    O KF é **termo comum** (a MLP pareada precisaria do mesmo filtro), então na
    comparação de H1 ele cancela; aqui o contraste é contra o MADDPG só para
    dimensionar o quão pequeno o filtro é ao lado da política.
    """
    kf = FiltroKalmanControle.custo_operacoes()
    # T1: estado residente do KF por drone = x̂ (2) + P simétrica (3) floats fp32
    kf_bytes_por_drone = (2 + 3) * 4
    merito = {"kf_flops": kf.total, "kf_mult_fp": kf.mult_fp, "kf_div_fp": kf.div_fp,
              "kf_bytes_por_drone": kf_bytes_por_drone, "por_Na": {}}
    print("Figuras de mérito com o filtro embutido (RNF-05)\n")
    print("T3 — operações por decisão")
    print(f"  KF+controle: {kf.total} flops ({kf.mult_fp} mult, {kf.soma_fp} soma, {kf.div_fp} div)")
    print(f"{'Na':>4} {'MADDPG mult_fp':>15} {'KF flops':>9} {'KF / MADDPG':>12}")
    for na in (2, 4, 8):
        frac = kf.total / MADDPG_MULT_FP[na]
        merito["por_Na"][na] = {"maddpg_mult_fp": MADDPG_MULT_FP[na],
                                "fracao_flops": frac,
                                "fracao_bytes": kf_bytes_por_drone / (MADDPG_ATOR_KB[na] * 1024)}
        print(f"{na:>4} {MADDPG_MULT_FP[na]:>15} {kf.total:>9} {frac:>11.3%}")
    print("\nT1 — memória residente")
    print(f"  estado do KF por drone: {kf_bytes_por_drone} bytes "
          f"(x̂ 2 + P sim. 3, fp32) — contra {MADDPG_ATOR_KB[4]:.0f} KB do ator MADDPG "
          f"({merito['por_Na'][4]['fracao_bytes']:.4%})")
    print("\nLeitura: o filtro é constante e desprezível ao lado da política — o custo")
    print("que a alegação 'sem MACs' precisa declarar (RNF-05) é ~0,07% do ator do MADDPG.\n")
    return merito


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def imagens(comp: dict, ladder: dict, merito: dict) -> None:
    plt = _plt()
    DIR.mkdir(parents=True, exist_ok=True)
    celulas = ["a_2-s_24", "a_4-s_24", "a_8-s_24"]

    # 1. Comparação com Lamenza (barras: publicado / limpo / bruto σ10 / KF σ10)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(celulas)); w = 0.2
    pub = [comp["celulas"][c]["publicado"] for c in celulas]
    d0 = [comp["celulas"][c]["degrau0"]["ct"] for c in celulas]
    d1 = [comp["celulas"][c]["sigmas"]["10"]["degrau1"]["ct"] for c in celulas]
    d4 = [comp["celulas"][c]["sigmas"]["10"]["degrau4"]["ct"] for c in celulas]
    ax.bar(x - 1.5 * w, pub, w, label="Lamenza (published)", color=CINZA)
    ax.bar(x - 0.5 * w, d0, w, label="step 0 (clean)", color=TEAL)
    ax.bar(x + 0.5 * w, d1, w, label="step 1 (raw, σ=10)", color=VERM)
    ax.bar(x + 1.5 * w, d4, w, label="step 4 (KF, σ=10)", color=AMBAR)
    ax.set_xticks(x); ax.set_xticklabels(celulas)
    ax.set_ylabel("completion time (s)")
    ax.set_title("MADDPG under noise vs Lamenza — the KF restores the published level")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "comparacao_lamenza.png", dpi=110); plt.close(fig)

    # 2. Degradação × σ (Δ do tempo contra o limpo), bruto vs KF
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    sig = [0, 10, 20]
    for ax, c in zip(axes, celulas):
        cel = comp["celulas"][c]
        base = cel["degrau0"]["ct"]
        bruto = [0.0] + [cel["sigmas"][str(s)]["degrau1"]["ct"] - base for s in (10, 20)]
        kf = [0.0] + [cel["sigmas"][str(s)]["degrau4"]["ct"] - base for s in (10, 20)]
        ax.plot(sig, bruto, "o-", color=VERM, label="raw (no filter)")
        ax.plot(sig, kf, "s-", color=AMBAR, label="KF+control")
        ax.axhline(0, color=CINZA, lw=1, ls=":")
        ax.set_title(c); ax.set_xlabel("GPS σ (m)")
    axes[0].set_ylabel("Δ time vs clean (s)")
    axes[0].legend(fontsize=8)
    axes[1].axvline(10, color=CINZA, alpha=0.3, lw=8)  # marca a grade declarada
    fig.suptitle("Completion-time degradation vs σ — raw grows, the KF stays flat")
    fig.tight_layout()
    fig.savefig(DIR / "degradacao_vs_sigma.png", dpi=110); plt.close(fig)

    # 3. NEES do filtro no ambiente real (da run da escada, 9 células × σ)
    cels9 = sorted(ladder["celulas"].keys())
    sigmas = [5, 10]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cels9)); w = 0.38
    for j, s in enumerate(sigmas):
        nees = [ladder["celulas"][c][f"sigma_{s}"]["degrau4"]["nees_medio"] for c in cels9]
        ax.bar(x + (j - 0.5) * w, nees, w, label=f"σ={s}")
    lo, hi = faixa_chi2_individual(2)
    ax.axhline(2.0, color=CINZA, ls=":", lw=1, label="NEES=2 target")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("a_", "").replace("-s_", "×") for c in cels9], rotation=45)
    ax.set_ylabel("mean NEES (log)")
    ax.set_title("KF NEES in the real environment — optimistic with Q=0 (above 2), worse at smaller σ")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "nees_real.png", dpi=110); plt.close(fig)

    # 4. RMSE do filtro vs a medida crua σ√2
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, s in enumerate(sigmas):
        rmse = [ladder["celulas"][c][f"sigma_{s}"]["degrau4"]["rmse_estimativa"] for c in cels9]
        ax.bar(x + (j - 0.5) * w, rmse, w, label=f"KF σ={s}")
        ax.axhline(s * np.sqrt(2), color=(VERM if s == 10 else AMBAR), ls="--", lw=1,
                   label=f"raw σ={s} (σ√2={s*np.sqrt(2):.1f})")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("a_", "").replace("-s_", "×") for c in cels9], rotation=45)
    ax.set_ylabel("position RMSE (m)")
    ax.set_title("KF error on real trajectories — well below the raw measurement")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "rmse_real.png", dpi=110); plt.close(fig)

    # 5. Figura de mérito: custo do KF contra o ator do MADDPG (T3)
    fig, ax = plt.subplots(figsize=(8, 5))
    nas = [2, 4, 8]
    ax.bar([f"MADDPG\nNa={na}" for na in nas], [MADDPG_MULT_FP[na] for na in nas],
           color=TEAL, label="actor fp mults (MADDPG)")
    ax.axhline(merito["kf_flops"], color=AMBAR, lw=2, label=f"KF total = {merito['kf_flops']} flops")
    ax.set_yscale("log"); ax.set_ylabel("operations per decision (log)")
    ax.set_title("Figure of merit T3 — the filter is negligible next to the policy")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "custo_com_filtro.png", dpi=110); plt.close(fig)

    print(f"[imagens] 5 figuras em {DIR}/")


def main() -> int:
    comp = _ler("reports/spec07-maddpg-vs-lamenza.json")
    ladder = _ler("reports/spec07-maddpg-ruido.json")
    merito = figuras_de_merito()
    try:
        imagens(comp, ladder, merito)
    except ImportError:
        print("[imagens] matplotlib ausente — pulei. Rode: uv sync --group dev")
    Path("reports/spec07-maddpg-merito.json").write_text(
        json.dumps(merito, indent=2, ensure_ascii=False))
    print("artefato: reports/spec07-maddpg-merito.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
