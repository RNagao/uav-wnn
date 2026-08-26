"""Monitor em tempo real da campanha comparável (`spec06_campanha_completa.py`).

    uv run python scripts/spec06_campanha_painel.py
    watch -n 60 -c 'uv run python scripts/spec06_campanha_painel.py'

O UNIVERSO VEM DA FILA, NÃO DOS ARQUIVOS. Ler só os CSVs existentes mostraria
"3 de 3 prontas" quando faltam 84 — o painel tem de saber o que ainda não
começou. Por isso chama `spec06_campanha_fila.sh --listar`, que é a mesma fonte
que o executor usa.

O ETA DE CADA CORRIDA sai do `wall_s` já gravado por run, não de uma taxa global:
a vazão muda por ordem de grandeza entre `Na=1/Ns=12` e `Na=8/Ns=36`, e uma taxa
única mentiria nos dois extremos. O ETA GLOBAL divide o trabalho restante pela
vazão agregada observada — e é otimista por construção, porque as corridas
pesadas tendem a ficar para o fim.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR = RAIZ / "reports" / "campanha"
FILA = RAIZ / "scripts" / "spec06_campanha_fila.sh"
ALVO = 100
LOG_TREINO = sorted(RAIZ.glob("reports/spec06-multi-*.log"))

_LINHA = re.compile(r"^(\S+)\s+Na=(\d+)\s+Ns=(\d+)\s+(\S+)\s+(\d+)/100\s+(.*)$")


def _barra(f: float, larg: int = 12) -> str:
    c = int(max(0.0, min(1.0, f)) * larg)
    return "█" * c + "░" * (larg - c)


def _hm(seg: float) -> str:
    if seg != seg or seg < 0 or seg > 96 * 3600:
        return "  >96h"
    h, m = divmod(int(seg) // 60, 60)
    return f"{h:>3}h{m:02d}m" if h else f"    {m:>2}m"


def universo() -> list[dict]:
    """A fila inteira, do próprio script que a executa."""
    try:
        saida = subprocess.run(["bash", str(FILA), "--listar"], capture_output=True,
                               text=True, timeout=60, cwd=RAIZ).stdout
    except Exception as e:                      # noqa: BLE001
        print(f"não consegui ler a fila ({e})"); return []
    itens = []
    for l in saida.splitlines():
        m = _LINHA.match(l.rstrip())
        if m:
            rot, na, ns, reg, feitas, st = m.groups()
            itens.append({"rotulo": rot, "na": int(na), "ns": int(ns), "regime": reg,
                          "estado": st.strip(),
                          "transf": "transf" in st})
    return itens


def progresso(it: dict) -> tuple[int, float, float, float]:
    """`(runs, wall_medio, ct_medio, sr_medio)` do CSV daquela corrida."""
    arq = DIR / f"{it['rotulo']}-a{it['na']}s{it['ns']}-{it['regime']}.csv"
    if not arq.exists():
        return 0, 0.0, float("nan"), float("nan")
    L = list(csv.DictReader(arq.open(newline="")))
    if not L:
        return 0, 0.0, float("nan"), float("nan")
    n = len(L)
    w = sum(float(x["wall_s"]) for x in L) / n
    ct = sum(float(x["completion_time_medio"]) for x in L) / n
    sr = sum(float(x["taxa_sucesso"]) for x in L) / n
    return n, w, ct, sr


def _cmdlines() -> list[str]:
    """Linhas de comando completas, de `/proc`, NÃO do `ps`.

    O `ps` respeita `$COLUMNS` e TRUNCA — e o `watch` define `COLUMNS` para a
    largura do terminal. Sob `watch -n 60`, as linhas vinham cortadas em ~120
    caracteres, o `--ruido` (que fica além disso) sumia, e o painel declarava
    "0 em voo" com oito processos a 99,9 % de CPU. `/proc/<pid>/cmdline` não tem
    formatação para atrapalhar.
    """
    saida = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            saida.append(
                (d / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8", "replace"))
        except OSError:
            continue                            # o processo morreu no meio da leitura
    return saida


def em_voo() -> set[tuple[str, int, int, str]]:
    """Corridas com processo vivo — para separar 'andando' de 'parada no meio'."""
    vivos = set()
    for l in _cmdlines():
        if "spec06_campanha_completa.py" not in l or ".venv/bin/python3" not in l:
            continue
        r = re.search(r"--rotulo (\S+).*--na (\d+).*--ns (\d+).*--ruido (\S+)", l)
        if not r:
            continue
        reg = r.group(4)
        if reg == "sorteio":
            s = re.search(r"--sigmas (\S+)", l); d = re.search(r"--degraus (\S+)", l)
            sig = (s.group(1) if s else "10").replace(",", "-")
            reg = f"sigma{sig}-" + {"1": "sem-kf", "4": "com-kf"}.get(
                d.group(1) if d else "", "mix")
        vivos.add((r.group(1), int(r.group(2)), int(r.group(3)), reg))
    return vivos


def main() -> int:
    itens = universo()
    if not itens:
        return 1
    vivos = em_voo()
    print(f"{time.strftime('%d/%m %H:%M:%S')}   CAMPANHA COMPARÁVEL\n")

    feitas = fila = bloqueadas = 0
    runs_ok = runs_tot = 0
    voando, parciais = [], []
    for it in itens:
        n, w, ct, sr = progresso(it)
        it.update(runs=n, wall=w, ct=ct, sr=sr)
        chave = (it["rotulo"], it["na"], it["ns"], it["regime"])
        it["voando"] = chave in vivos
        runs_tot += ALVO
        runs_ok += min(n, ALVO)
        if "PULA" in it["estado"]:
            bloqueadas += 1
        elif n >= ALVO:
            feitas += 1
        elif it["voando"]:
            voando.append(it)
        elif n > 0:
            parciais.append(it)
        else:
            fila += 1

    frac = runs_ok / runs_tot if runs_tot else 0.0
    print(f"   {_barra(frac, 28)} {frac:>5.1%}   "
          f"{runs_ok:,}/{runs_tot:,} runs  ·  {runs_ok*200:,} episódios\n")
    print(f"   {feitas} prontas · {len(voando)} em voo · {len(parciais)} paradas no meio"
          f" · {fila} na fila · {bloqueadas} bloqueadas\n")

    def linha(it, marca=""):
        rot = f"{it['rotulo']} a{it['na']}s{it['ns']} {it['regime']}"
        tt = "ᵗ" if it["transf"] else " "
        eta = it["wall"] * (ALVO - it["runs"]) if it["wall"] else float("nan")
        met = (f"{it['ct']:>7.1f}s {it['sr']:>6.1%}" if it["ct"] == it["ct"]
               else f"{'—':>7} {'—':>6}")
        print(f"   {_barra(it['runs']/ALVO)} {it['runs']:>3}/{ALVO} "
              f"{rot:<34}{tt} {met} {_hm(eta) if marca != 'x' else '  parada':>8}")

    if voando:
        print("   EM VOO")
        for it in sorted(voando, key=lambda x: -x["runs"]):
            linha(it)
    if parciais:
        print("\n   PARADAS NO MEIO (relançar a fila as retoma do cache)")
        for it in sorted(parciais, key=lambda x: -x["runs"]):
            linha(it, "x")

    # ETA global pela vazão agregada observada, não por corrida
    if voando:
        vazao = sum(1.0 / it["wall"] for it in voando if it["wall"] > 0)  # runs/s
        resta = runs_tot - runs_ok - bloqueadas * ALVO
        if vazao > 0:
            print(f"\n   restam {resta:,} runs · vazão {vazao*3600:.0f} runs/h"
                  f" · ETA global ~{_hm(resta / vazao).strip()}"
                  "  (otimista: as pesadas ficam para o fim)")
    if bloqueadas:
        print(f"\n   {bloqueadas} bloqueadas — sem checkpoint ou sem atestado; "
              "a fila as pula e avisa")

    # "em curso" pela EXISTÊNCIA DO PROCESSO, não pelo log: um treino morto por
    # fora não deixa marcador nenhum, e o painel o anunciava vivo por horas.
    treinos = [c for c in _cmdlines()
               if re.search(r"python3?\s+scripts/spec06_treino_multiagente\.py", c)]
    for lg in LOG_TREINO:
        ult = [l for l in lg.read_text(errors="replace").splitlines()
               if l.startswith("[")]
        if not ult:
            continue
        alvo = lg.stem.split("-")[-1].lstrip("a")          # "…-T32-a8" -> "8"
        vivo = any(f"--na {alvo} " in c for c in treinos)
        estado = "em curso" if vivo else "PARADO"
        print(f"\n   treino {estado} · {lg.stem}: {ult[-1].strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
