"""Progresso da campanha comparável (`spec06_campanha_completa.py`).

Lê os CSVs — não os logs — porque cada run é anexada assim que fecha, e o CSV é
a mesma fonte que o relatório vai usar. O ETA sai do tempo por run já gravado
(`wall_s`), que é o único preditor honesto: a vazão muda por ordem de grandeza
entre `Na=1` e `Na=8`, e uma taxa global mentiria nos dois extremos.

    uv run python scripts/spec06_campanha_painel.py
    watch -n 120 'uv run python scripts/spec06_campanha_painel.py'
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR = RAIZ / "reports" / "campanha"
ALVO = 100


def _barra(f: float, larg: int = 14) -> str:
    c = int(max(0.0, min(1.0, f)) * larg)
    return "█" * c + "░" * (larg - c)


def _hm(seg: float) -> str:
    if seg != seg or seg < 0 or seg > 96 * 3600:
        return "  >96h"
    h, m = divmod(int(seg) // 60, 60)
    return f"{h:>3}h{m:02d}m" if h else f"    {m:>2}m"


def main() -> int:
    if not DIR.exists():
        print("campanha ainda não começou"); return 0
    print(f"{time.strftime('%d/%m %H:%M:%S')}   CAMPANHA COMPARÁVEL\n")
    print(f"   {'':14} {'':>6} {'modelo · célula · regime':<34} "
          f"{'CT médio':>10} {'SR':>7} {'resta':>8}")
    faltas = []
    for arq in sorted(DIR.glob("*.csv")):
        if arq.name.endswith("-episodios.csv"):
            continue
        linhas = list(csv.DictReader(arq.open(newline="")))
        if not linhas:
            continue
        n = len(linhas)
        # média das runs — a figura que a campanha existe para produzir
        ct = sum(float(x["completion_time_medio"]) for x in linhas) / n
        sr = sum(float(x["taxa_sucesso"]) for x in linhas) / n
        wall = sum(float(x["wall_s"]) for x in linhas) / n
        eta = wall * (ALVO - n)
        faltas.append(eta)
        rot = arq.stem.replace("-", " · ", 1).replace("-", " · ")
        marca = "PRONTO" if n >= ALVO else f"{n:>3}/{ALVO}"
        print(f"   {_barra(n / ALVO)} {marca:>6} {rot:<34} "
              f"{ct:>9.1f}s {sr:>6.1%} {_hm(eta) if n < ALVO else '':>8}")
    if faltas:
        print(f"\n   -> a última fecha em ~{_hm(max(faltas)).strip()} "
              f"(6 em paralelo; a soma sequencial seria maior)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
