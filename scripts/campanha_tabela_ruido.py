"""Tabela de ruído — média ± desvio de taxa de sucesso e completion time por
(modelo, σ, uso de KF), no formato LaTeX da dissertação.

É o `campanha_resumo.py` agregado para preencher a tabela de cenários de ruído.
Cada linha da tabela é UM (modelo, σ, KF); como a tabela não separa por Ns, este
script **junta as células Ns∈{12,24,36}** — concatena as corridas dos três CSVs e
tira média/desvio sobre o conjunto (mesmo grão do `campanha_resumo`: uma corrida
por linha, `taxa_sucesso` e `completion_time_medio`).

    modelo  ->  CSV `reports/campanha/{rotulo}-a{na}s{ns}-sigma{σ}-{sem-kf|com-kf}.csv`
                (na=2 por padrão; use `--na` para outra célula)
    MADDPG  ->  maddpg      | T16 -> wnn-T16 | T32 -> wnn-T32
    no KF   ->  sem-kf (degrau 1, medida crua)
    with KF ->  com-kf (degrau 4, KF+controle)

CUIDADO COM O CT AGREGADO. O completion time cresce com Ns, então juntar Ns=12/24/36
num número só dá um desvio inflado pela diferença ENTRE Ns, não só pela variação
entre corridas. É o que a tabela pede (uma linha por modelo/σ/KF), mas se quiser o
número por célula use `--ns 24` (ou 12/36). A taxa de sucesso é fração e agrega bem.

Células incompletas (campanha ainda rodando) entram com o n disponível e são
marcadas `parcial`; células sem nenhum CSV saem como `pendente`.

    uv run python scripts/campanha_tabela_ruido.py                 # tabela + LaTeX
    uv run python scripts/campanha_tabela_ruido.py --ns 24         # só Ns=24
    uv run python scripts/campanha_tabela_ruido.py --sigmas 10 25  # só σ 10 e 25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from campanha_resumo import COL_SUCESSO, COL_TEMPO, _primeira_presente, resumo  # noqa: E402

PASTA_PADRAO = Path(__file__).resolve().parent.parent / "reports" / "campanha"

# (rótulo na tabela, rótulo do arquivo)
MODELOS = [("MADDPG", "maddpg"), ("T16", "wnn-T16"), ("T32", "wnn-T32")]
# (rótulo na tabela, sufixo do arquivo)
KF = [("no KF", "sem-kf"), ("with KF", "com-kf")]
NS_PADRAO = [12, 24, 36]


def _ler(caminho: Path) -> list[dict[str, str]]:
    if not caminho.exists():
        return []
    with caminho.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _coluna(linhas: list[dict[str, str]], candidatas) -> np.ndarray:
    if not linhas:
        return np.array([], float)
    col = _primeira_presente(list(linhas[0].keys()), candidatas)
    if col is None:
        return np.array([], float)
    vals = []
    for ln in linhas:
        bruto = (ln.get(col) or "").strip()
        if bruto:
            try:
                vals.append(float(bruto))
            except ValueError:
                pass
    return np.asarray(vals, float)


def coletar(rotulo: str, sigma: int, reg: str, ns_list: list[int], pasta: Path, na: int):
    """Concatena as corridas dos Ns pedidos para (modelo, σ, KF)."""
    suc, tmp, presentes, faltando = [], [], [], []
    for ns in ns_list:
        linhas = _ler(pasta / f"{rotulo}-a{na}s{ns}-sigma{sigma}-{reg}.csv")
        if not linhas:
            faltando.append(ns)
            continue
        presentes.append(ns)
        suc.append(_coluna(linhas, COL_SUCESSO))
        tmp.append(_coluna(linhas, COL_TEMPO))
    s = np.concatenate(suc) if suc else np.array([], float)
    t = np.concatenate(tmp) if tmp else np.array([], float)
    return s, t, presentes, faltando


def _fmt_sr(r: dict) -> str:
    return f"{r['media']*100:.2f}\\% $\\pm$ {r['desvio']*100:.2f}"


def _fmt_ct(r: dict) -> str:
    return f"{r['media']:.2f} $\\pm$ {r['desvio']:.2f} s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigmas", nargs="+", type=int, default=[10, 25, 50])
    ap.add_argument("--na", type=int, default=2, help="número de drones das células (padrão: 2)")
    ap.add_argument("--ns", nargs="+", type=int, default=NS_PADRAO,
                    help=f"quais Ns juntar (padrão: {NS_PADRAO})")
    ap.add_argument("--dir", type=Path, default=PASTA_PADRAO)
    ap.add_argument("--runs-esperados", type=int, default=100,
                    help="runs/célula esperados; abaixo disso marca 'parcial' (padrão: 100)")
    a = ap.parse_args(argv)

    esperado = a.runs_esperados * len(a.ns)
    console, latex = [], []

    for kf_rot, reg in KF:
        for mod_rot, rotulo in MODELOS:
            for sigma in a.sigmas:
                s, t, pres, falta = coletar(rotulo, sigma, reg, a.ns, a.dir, a.na)
                if s.size == 0:
                    sr_cell = ct_cell = "\\textit{pendente}"
                    nota = "pendente (sem CSV)"
                else:
                    rs, rt = resumo(s), resumo(t)
                    sr_cell, ct_cell = _fmt_sr(rs), _fmt_ct(rt)
                    parcial = s.size < esperado or falta
                    nota = (f"n={s.size}"
                            + (f" PARCIAL (falta Ns {falta})" if falta else "")
                            + (" PARCIAL" if (s.size < esperado and not falta) else ""))
                console.append((f"{mod_rot} σ={sigma} {kf_rot}", sr_cell, ct_cell, nota))
                latex.append(f"{mod_rot} ($\\mathit{{\\sigma}}$={sigma}, {kf_rot}) "
                             f"& {sr_cell} & {ct_cell}\\\\")

    # ---- console: legível, com o n de cada célula ----
    print(f"agregando Ns={a.ns}  (esperado {esperado} corridas/célula)\n")
    print(f"{'célula':>26} | {'Success Rate':>22} | {'Completion Time':>22} | estado")
    print("-" * 96)
    for cel, sr, ct, nota in console:
        sr_p = sr.replace("\\%", "%").replace("$\\pm$", "±")
        ct_p = ct.replace("$\\pm$", "±").replace("\\textit{pendente}", "pendente")
        print(f"{cel:>26} | {sr_p:>22} | {ct_p:>22} | {nota}")

    # ---- LaTeX pronto para colar ----
    print("\n" + "=" * 96 + "\nTABELA LaTeX (colar):\n")
    print("\\begin{table}")
    print("\\centering")
    print("\\caption{WNNs and MADDPG noise scenarios results}\\label{tab1}")
    print("\\label{tab:noise_uav_results}")
    print("\\begin{tabular}{|l|l|l|}")
    print("\\hline")
    print("Model & Success Rate & Completion Time\\\\")
    print("\\hline")
    for linha in latex:
        print(linha)
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
