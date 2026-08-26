"""Compila a campanha comparável numa tabela por cenário.

Lê os `-episodios.csv` (não os agregados) porque a comparação entre modelos é
PAREADA: só faz sentido comparar episódios que viram o MESMO cenário, e a
semente é o que identifica isso. Os agregados já perderam essa informação.

    uv run python scripts/spec06_campanha_tabela.py
    uv run python scripts/spec06_campanha_tabela.py --md > reports/spec06-campanha.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from uavwnn.eval.stats import bootstrap_pareado  # noqa: E402

DIR = Path("reports/campanha")
MODELOS = ("maddpg", "wnn-T16", "wnn-T32")
REGIMES = (("limpo", "sem ruído"),
           ("sigma10-sem-kf", "σ=10 · SEM KF"),
           ("sigma10-com-kf", "σ=10 · COM KF"))
CELULAS = ((1, 12), (2, 12), (2, 24), (2, 36))
SEED = 20260826


def ler(modelo: str, na: int, ns: int, regime: str) -> dict[int, dict] | None:
    f = DIR / f"{modelo}-a{na}s{ns}-{regime}-episodios.csv"
    if not f.exists():
        return None
    return {int(r["seed"]): r for r in csv.DictReader(f.open())}


def resumo(eps: dict) -> tuple[float, float, float, int]:
    """`(CT_censurado, CT_sucesso, SR, n)`.

    `CT_cens` conta o fracasso como o horizonte; `CT_succ` só os que
    completaram. Reportar só o segundo premia quem desiste dos cenários
    difíceis.
    """
    ct = np.array([float(r["completion_time"]) for r in eps.values()])
    ok = np.array([int(r["sucesso"]) for r in eps.values()], bool)
    return (float(ct.mean()),
            float(ct[ok].mean()) if ok.any() else float("nan"),
            float(ok.mean()), int(ct.size))


def pareado(a: dict, b: dict, rng) -> dict | None:
    """`b − a` no tempo, só nas sementes que os dois viram."""
    comuns = sorted(set(a) & set(b))
    if not comuns:
        return None
    ca = np.array([float(a[s]["completion_time"]) for s in comuns])
    cb = np.array([float(b[s]["completion_time"]) for s in comuns])
    d = bootstrap_pareado(cb, ca, rng).como_linha()
    d["n"] = len(comuns)
    d["sr_a"] = float(np.mean([int(a[s]["sucesso"]) for s in comuns]))
    d["sr_b"] = float(np.mean([int(b[s]["sucesso"]) for s in comuns]))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", action="store_true", help="saída em Markdown")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    p = print

    if a.md:
        p("# Campanha comparável — MADDPG × WNN T=16 × WNN T=32\n")
        p("100 runs × 200 episódios por célula. Todos os modelos veem as **mesmas**")
        p("sementes, e no regime com ruído o mesmo episódio cai no mesmo degrau —")
        p("a comparação é pareada em tudo.\n")
        p("`CT_cens` conta o fracasso como o horizonte (500 s); `CT_succ` só os que")
        p("completaram. Reportar só o segundo premiaria quem desiste dos difíceis.\n")

    # --- tabela principal
    p("\n## Desempenho por cenário\n" if a.md else "\n=== DESEMPENHO POR CENÁRIO ===\n")
    if a.md:
        p("| Na | Ns | ruído | modelo | CT_cens | CT_succ | SR |")
        p("|---|---|---|---|---|---|---|")
    else:
        p(f"  {'Na':>3}{'Ns':>4} {'ruído':<14} {'modelo':<9}"
          f"{'CT_cens':>9}{'CT_succ':>9}{'SR':>8}")
    for na, ns in CELULAS:
        for reg, rot in REGIMES:
            for m in MODELOS:
                eps = ler(m, na, ns, reg)
                if eps is None:
                    continue
                cc, cs, sr, n = resumo(eps)
                if a.md:
                    p(f"| {na} | {ns} | {rot} | {m} | {cc:.1f} s | {cs:.1f} s | {sr:.1%} |")
                else:
                    p(f"  {na:>3}{ns:>4} {rot:<14} {m:<9}"
                      f"{cc:>8.1f}s{cs:>8.1f}s{sr:>7.1%}")

    # --- o KF ajuda?
    p("\n## O KF ajuda?\n" if a.md else "\n=== O KF AJUDA? (com KF − sem KF, pareado) ===\n")
    if a.md:
        p("Diferença de tempo, pareada pelas mesmas sementes. **Negativo = o KF é")
        p("mais rápido.** IC 95 % por bootstrap pareado.\n")
        p("| Na | Ns | modelo | n | sem KF | com KF | Δ tempo | IC 95 % | SR sem→com |")
        p("|---|---|---|---|---|---|---|---|---|")
    for na, ns in CELULAS:
        for m in MODELOS:
            sem, com = ler(m, na, ns, "sigma10-sem-kf"), ler(m, na, ns, "sigma10-com-kf")
            if not sem or not com:
                continue
            d = pareado(sem, com, rng)
            if d is None:
                continue
            cs = np.mean([float(r["completion_time"]) for r in sem.values()])
            cc = np.mean([float(r["completion_time"]) for r in com.values()])
            ic = f"[{d['delta_ic_lo']:+.2f}, {d['delta_ic_hi']:+.2f}]"
            marca = "" if d["contem_zero"] else " **"
            if a.md:
                p(f"| {na} | {ns} | {m} | {d['n']:,} | {cs:.2f} s | {cc:.2f} s | "
                  f"{d['delta_media']:+.2f} s{marca} | {ic} | "
                  f"{d['sr_a']:.1%} → {d['sr_b']:.1%} |")
            else:
                p(f"  {na:>3}{ns:>4} {m:<9} n={d['n']:>6,} "
                  f"{cs:>7.2f}s →{cc:>7.2f}s  Δ={d['delta_media']:+6.2f}s {ic:>18}"
                  f"  SR {d['sr_a']:.1%}→{d['sr_b']:.1%}")
    if a.md:
        p("\n`**` = o IC não contém zero.")

    # --- o ruído ajuda?
    p("\n## O ruído ajuda? (dithering)\n" if a.md
      else "\n=== O RUÍDO AJUDA? (sem KF − limpo, pareado) ===\n")
    if a.md:
        p("Diferença entre o ruído bruto e o teto limpo. **SR positivo = o ruído")
        p("melhora**, que é o achado da SPEC-08.\n")
        p("| Na | Ns | modelo | limpo | σ=10 sem KF | Δ SR |")
        p("|---|---|---|---|---|---|")
    for na, ns in CELULAS:
        for m in MODELOS:
            lim, sem = ler(m, na, ns, "limpo"), ler(m, na, ns, "sigma10-sem-kf")
            if not lim or not sem:
                continue
            _, _, sr_l, _ = resumo(lim)
            _, _, sr_s, _ = resumo(sem)
            if a.md:
                p(f"| {na} | {ns} | {m} | {sr_l:.1%} | {sr_s:.1%} | "
                  f"**{(sr_s-sr_l)*100:+.1f} pp** |")
            else:
                p(f"  {na:>3}{ns:>4} {m:<9} {sr_l:>7.1%} → {sr_s:>7.1%}"
                  f"   Δ = {(sr_s-sr_l)*100:+5.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
