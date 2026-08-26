"""Mantém a campanha rodando SEM derrubar o treino do Algoritmo 4.4.

O PROBLEMA QUE ELE RESOLVE

A fila inteira (8 corridas) derrubou o treino de 105 para 49 passos/s — medido.
A causa é contenção de **cache**, não de CPU: o treino varre LUTs de centenas de
MB a cada passo e as corridas despejam essas linhas continuamente. `nice` arbitra
tempo de CPU e não faz nada contra isso, então a única defesa é limitar QUANTAS
correm ao mesmo tempo — e o número certo não dá para adivinhar, tem de ser medido.

COMO ELE MEDE

A taxa vem das linhas de progresso do treino, e o supervisor **descarta as janelas
que contêm avaliação intercalada**: elas rodam a cada 250 mil passos, custam tempo,
e derrubam a taxa sozinhas. Sem esse cuidado o supervisor leria a queda de uma
avaliação como contenção e recuaria à toa — foi o que quase me enganou ao medir a
segunda corrida (91 passos/s numa janela que continha a avaliação de 1 M).

A REGRA

    taxa < `base × 0,90`   →  recua (para de repor as que fecham)
    taxa ≥ `base × 0,97` duas janelas seguidas  →  sobe uma

Recuar NÃO mata corrida em andamento: só deixa de repor. Matar desperdiçaria a
run em voo, e o excesso se resolve sozinho quando a corrente terminar.

    uv run python scripts/spec06_campanha_supervisor.py            # fica em pé
    uv run python scripts/spec06_campanha_supervisor.py --uma-vez  # um ciclo só
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FILA = RAIZ / "scripts" / "spec06_campanha_fila.sh"
UMA = RAIZ / "scripts" / "spec06_campanha_uma.sh"
LOG_TREINO = RAIZ / "reports" / "spec06-multi-T32-a8.log"
DIARIO = RAIZ / "reports" / "campanha" / "logs" / "_supervisor.log"
ALVO_RUNS = 100
INTERVALO_AVAL = 250_000

_PROG = re.compile(r"\[\s*[\d.]+%\]\s+([\d,]+)/[\d,]+ · ([\d,]+) ep · ([\d.]+) passos/s")
_FILA = re.compile(r"^(\S+)\s+Na=(\d+)\s+Ns=(\d+)\s+(\S+)\s+(\d+)/100\s+(.*)$")
_REG = {"limpo": "limpo", "sigma10-sem-kf": "sem-kf", "sigma10-com-kf": "com-kf"}


def diz(msg: str) -> None:
    linha = f"{time.strftime('%d/%m %H:%M:%S')}  {msg}"
    print(linha, flush=True)
    DIARIO.parent.mkdir(parents=True, exist_ok=True)
    with DIARIO.open("a") as f:
        f.write(linha + "\n")


# --- treino ------------------------------------------------------------------

def janelas() -> list[tuple[int, float, bool]]:
    """`(passo, taxa_marginal, contem_avaliacao)` por linha de progresso.

    A taxa é MARGINAL entre linhas consecutivas, reconstruída da cumulativa que o
    treino imprime: a cumulativa mente por baixo quando a corrida desacelera, e é
    justamente a desaceleração que se quer detectar.
    """
    if not LOG_TREINO.exists():
        return []
    P = []
    for l in LOG_TREINO.read_text(errors="replace").splitlines():
        m = _PROG.search(l)
        if m:
            P.append((int(m.group(1).replace(",", "")), float(m.group(3))))
    saida = []
    for i in range(1, len(P)):
        (s0, r0), (s1, r1) = P[i - 1], P[i]
        t0, t1 = s0 / max(r0, 1e-9), s1 / max(r1, 1e-9)
        if t1 <= t0:
            continue
        aval = any(s0 < k <= s1
                   for k in range(INTERVALO_AVAL, s1 + INTERVALO_AVAL, INTERVALO_AVAL))
        saida.append((s1, (s1 - s0) / (t1 - t0), aval))
    return saida


def treino_vivo() -> bool:
    return any(re.search(r"python3?\s+scripts/spec06_treino_multiagente\.py", c)
               for c in cmdlines())


# --- processos ---------------------------------------------------------------

def cmdlines() -> list[str]:
    """De `/proc`, não do `ps`: o `ps` trunca conforme `$COLUMNS` e some com os
    argumentos que interessam quando roda sob `watch`."""
    saida = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            saida.append((d / "cmdline").read_bytes()
                         .replace(b"\x00", b" ").decode("utf-8", "replace"))
        except OSError:
            continue
    return saida


def em_voo() -> list[tuple[str, int, int, str]]:
    """Só o python do venv com o script como ARQUIVO — não o wrapper `uv`, que
    também casaria, nem um `python -c` cujo FONTE cite o nome do script (foi
    assim que contei três corridas inexistentes)."""
    vivos = []
    for c in cmdlines():
        if not re.search(r"\.venv/bin/python3\s+scripts/spec06_campanha_completa\.py", c):
            continue
        m = re.search(r"--rotulo (\S+).*?--na (\d+).*?--ns (\d+).*?--ruido (\S+)", c)
        if not m:
            continue
        reg = m.group(4)
        if reg == "sorteio":
            d = re.search(r"--degraus (\S+)", c)
            reg = {"1": "sem-kf", "4": "com-kf"}.get(d.group(1) if d else "", "mix")
        vivos.append((m.group(1), int(m.group(2)), int(m.group(3)), reg))
    return vivos


# --- fila --------------------------------------------------------------------

def pendentes() -> list[tuple[str, int, int, str]]:
    """O que ainda falta, na ordem da fila, já sem o bloqueado e o pronto."""
    try:
        s = subprocess.run(["bash", str(FILA), "--listar"], capture_output=True,
                           text=True, timeout=120, cwd=RAIZ).stdout
    except Exception as e:                                   # noqa: BLE001
        diz(f"não li a fila: {e}")
        return []
    out = []
    for l in s.splitlines():
        m = _FILA.match(l.rstrip())
        if not m:
            continue
        rot, na, ns, reg, feitas, st = m.groups()
        if "PULA" in st or int(feitas) >= ALVO_RUNS:
            continue
        out.append((rot, int(na), int(ns), _REG.get(reg, reg)))
    return out


FALHAS_ATE_DESISTIR = 2
"""Quantas vezes tentar a mesma corrida antes de tirá-la da fila.

Sem isto o supervisor entra em laço: a corrida morre na inicialização (sem
metadado, sem atestado, sem checkpoint), ele vê "0 em voo" e relança a MESMA —
quatro vezes em 90 segundos, no caso que motivou esta guarda. Duas tentativas
cobrem falha transitória; a terceira seria teimosia."""


def lancar(alvo: tuple[str, int, int, str], falhas: dict) -> bool:
    """Lança e CONFIRMA que subiu — `bash &` devolve 0 mesmo se o filho morrer.

    A confirmação é olhar o processo depois de um instante: um erro de
    inicialização mata o Python em menos de um segundo, e sem esta checagem o
    supervisor contaria como sucesso.
    """
    rot, na, ns, reg = alvo
    r = subprocess.run(["bash", str(UMA), rot, str(na), str(ns), reg],
                       capture_output=True, text=True, cwd=RAIZ)
    if r.returncode != 0:
        falhas[alvo] = falhas.get(alvo, 0) + 1
        diz(f"falhou lançar {rot} a{na}s{ns} {reg}: "
            f"{r.stdout.strip()} {r.stderr.strip()}")
        return False
    time.sleep(8)
    if alvo not in em_voo():
        falhas[alvo] = falhas.get(alvo, 0) + 1
        n = falhas[alvo]
        log = RAIZ / "reports" / "campanha" / "logs"
        diz(f"{rot} a{na}s{ns} {reg} MORREU na inicialização "
            f"(tentativa {n}/{FALHAS_ATE_DESISTIR}) — ver {log}/"
            + ("desistindo desta corrida" if n >= FALHAS_ATE_DESISTIR else ""))
        return False
    diz(f"lançada {rot} a{na}s{ns} {reg}")
    falhas.pop(alvo, None)
    return True


# --- laço --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvo", type=int, default=1, help="corridas simultâneas iniciais")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--base", type=float, default=None,
                    help="passos/s do treino sozinho; padrão: mediana das janelas "
                         "limpas mais antigas do log")
    ap.add_argument("--piso", type=float, default=0.90, help="recua abaixo disto")
    ap.add_argument("--teto", type=float, default=0.97, help="sobe acima disto")
    ap.add_argument("--ciclo", type=int, default=90, help="segundos entre ciclos")
    ap.add_argument("--uma-vez", action="store_true")
    a = ap.parse_args()

    J = [j for j in janelas() if not j[2]]
    if a.base is None:
        if len(J) < 3:
            diz("sem janelas limpas suficientes para a base — passe --base")
            return 1
        prim = sorted(x[1] for x in J[:6])
        a.base = prim[len(prim) // 2]
    diz(f"supervisor no ar · base {a.base:.0f} passos/s · alvo {a.alvo} · máx {a.max} "
        f"· recua <{a.base*a.piso:.0f} · sobe >{a.base*a.teto:.0f}")

    visto = J[-1][0] if J else 0
    bons = 0
    alvo = a.alvo
    falhas: dict = {}          # corrida → tentativas que morreram
    while True:
        vivos = em_voo()
        # 1) o treino ainda existe? se acabou, a restrição some
        if not treino_vivo() and alvo < a.max:
            alvo = a.max
            diz(f"treino terminou — alvo livre em {alvo}")

        # 2) nova janela limpa do treino → decide subir ou recuar
        J = [j for j in janelas() if not j[2]]
        if J and J[-1][0] > visto:
            passo, taxa, _ = J[-1]
            visto = passo
            frac = taxa / a.base
            if frac < a.piso and alvo > 1:
                alvo -= 1; bons = 0
                diz(f"treino em {taxa:.0f} p/s ({frac:.0%} da base) — RECUA para "
                    f"alvo {alvo}; não mato as em voo, só deixo de repor")
            elif frac >= a.teto:
                bons += 1
                if bons >= 2 and alvo < a.max:
                    alvo += 1; bons = 0
                    diz(f"treino em {taxa:.0f} p/s ({frac:.0%}) duas janelas — "
                        f"SOBE para alvo {alvo}")
                else:
                    diz(f"treino em {taxa:.0f} p/s ({frac:.0%}) · alvo {alvo} "
                        f"· em voo {len(vivos)}")
            else:
                bons = 0
                diz(f"treino em {taxa:.0f} p/s ({frac:.0%}) — mantém alvo {alvo}")

        # 3) repõe até o alvo
        if len(vivos) < alvo:
            ocupadas = set(vivos)
            for cand in pendentes():
                if cand in ocupadas or falhas.get(cand, 0) >= FALHAS_ATE_DESISTIR:
                    continue
                if lancar(cand, falhas):
                    break
            else:
                mortas = sum(1 for v in falhas.values() if v >= FALHAS_ATE_DESISTIR)
                if not vivos:
                    diz(f"nada pendente e nada em voo — campanha completa"
                        + (f" ({mortas} corridas desistidas)" if mortas else ""))
                    return 0

        if a.uma_vez:
            return 0
        time.sleep(a.ciclo)


if __name__ == "__main__":
    raise SystemExit(main())
