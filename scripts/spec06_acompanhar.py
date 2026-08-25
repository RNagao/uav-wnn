"""Painel único das campanhas da SPEC-06 — grades da dissertação e do PPO, e a
escalada multiagente.

Substitui três scripts que faziam pedaços disto: `spec06_eta_ppo.py`, que trazia a
previsão do PPO, e `spec06_parciais.py`, que trazia o efeito marginal por eixo. Um
só, porque a pergunta é sempre a mesma — o que está rodando, quanto falta, e o que
já dá para dizer — e tê-la em três lugares fazia cada um responder um terço.

ONDE CADA CAMPANHA ESCREVE, E POR QUE ISSO OBRIGA A LER DOIS LUGARES

    Algoritmo 4.4     `metrics.jsonl` por corrida, a cada avaliação intercalada. É
    (spec06_busca)    ali que está o progresso DENTRO de uma corrida — o
                      `.progresso.jsonl` só ganha linha quando ela TERMINA, e numa
                      corrida de 10 M isso é horas de silêncio.

    PPO               cada ponto é um subprocesso em ambiente efêmero, com log
    (spec06_grid_ppo) próprio em `<raiz>/<ponto>/saida.log`. O registro da grade só
                      ganha linha ao fim do ponto.

O QUE OLHAR, E O QUE NÃO

`cob` é a **última** avaliação, sem seleção — a estimativa honesta de onde a política
está agora. `máx` é o máximo sobre todas, e carrega maldição do vencedor: medido
+0,9 pp na grade de desempate. Para julgar progresso use `cob`; `máx` só diz onde ela
já esteve.

As duas saem do banco `val`. O número que vai para o relatório sai do `test`, e só
existe quando a corrida termina.

    uv run python scripts/spec06_acompanhar.py
    uv run python scripts/spec06_acompanhar.py --eixos     # efeito marginal
    uv run python scripts/spec06_acompanhar.py --multi     # só a escalada
    watch -n 120 'uv run python scripts/spec06_acompanhar.py'

A escalada multiagente **não tem `metrics.jsonl`**: cada célula é um processo com
log próprio, e o progresso está no texto que ela imprime. Por isso o modo `multi`
lê o log, e não um JSONL — a alternativa seria mudar o treino só para o painel.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

MODELO = {"ntupla_dissertacao": "dissertação", "ntupla_anbubus": "anbubus",
          "mlp": "MLP"}
"""Nome curto por família. Sem ele, duas linhas com os mesmos hiperparâmetros são
indistinguíveis — e as famílias convertem a taxa de formas diferentes (média contra
soma), então confundi-las é confundir `Δf`."""

REFERENCIAS = ("  aleatório 3,3 % cob · PPO não treinado 10,8 % cob · "
               "guloso SR 100 %\n  dissertação a 200 mil: 50,1 % cob · SR 21,3 %")


# --- leitura ----------------------------------------------------------------

def _ler(p: Path) -> list[dict]:
    """JSONL tolerante: a última linha pode estar sendo escrita agora."""
    if not p.exists():
        return []
    fora = []
    for linha in p.read_text().splitlines():
        if not linha.strip():
            continue
        try:
            fora.append(json.loads(linha))
        except json.JSONDecodeError:
            pass
    return fora


def _hm(seg: float) -> str:
    """`4h12m`. Acima de 48 h a estimativa não merece essa precisão."""
    if seg != seg or seg < 0 or seg > 48 * 3600:
        return "  >48h"
    h, m = divmod(int(seg) // 60, 60)
    return f"{h:>3}h{m:02d}m" if h else f"    {m:>2}m"


def _barra(f: float, larg: int = 16) -> str:
    c = int(max(0.0, min(1.0, f)) * larg)
    return "█" * c + "░" * (larg - c)


def _taxa_marginal(linhas: list[dict], janela: int = 3) -> float:
    """Passos por segundo na JANELA recente, não na média cumulativa.

    A cumulativa mente por baixo quando a corrida desacelera: numa corrida real
    medi 464 -> 417 -> 361 acumulado enquanto a marginal era **152,7**. A causa está
    medida — o `prever` do crítico custa por ACERTO, e a taxa de acerto cresce com a
    ocupação da tabela.
    """
    if len(linhas) < 2:
        return 0.0
    a, b = linhas[max(0, len(linhas) - 1 - janela)], linhas[-1]
    dp, dt = b["passo"] - a["passo"], b["parede_s"] - a["parede_s"]
    return dp / dt if dt > 0 else 0.0


# --- escalada multiagente ---------------------------------------------------

_PROG = re.compile(r"\[\s*([\d.]+)%\]\s+([\d,]+)/([\d,]+).*?([\d.]+) passos/s")
_EVAL = re.compile(r"eval @ ([\d,]+): cob ([\d.]+) · sr ([\d.]+)")


def multi(logs: list[Path]) -> None:
    """Estado das células da escalada, uma linha por log.

    A taxa vem do próprio log (cumulativa, que é o que a célula imprime); o ETA é
    recalculado sobre a JANELA das duas últimas linhas de progresso, porque a
    cumulativa mente por baixo quando a tabela enche — o mesmo motivo de
    `_taxa_marginal`.
    """
    achou = [p for p in logs if p.exists()]
    if not achou:
        print("escalada multiagente: nenhum log encontrado\n")
        return
    print("ESCALADA MULTIAGENTE")
    fins = []
    for p in sorted(achou):
        txt = p.read_text(errors="replace")
        prog = _PROG.findall(txt)
        ev = _EVAL.findall(txt)
        cab = re.search(r"Na=(\d+) Ns=(\d+) · T=(\d+) · N=(\d+)", txt)
        rot = (f"Na={cab.group(1)} Ns={cab.group(2)} T={cab.group(3)} "
               f"N={cab.group(4)}" if cab else p.stem)
        if not prog:
            print(f"   {_barra(0)}   0%        │ {rot:<22} │ iniciando…")
            continue
        parou = "parada antecipada" in txt or "checkpoint (melhor):" in txt
        frac = float(prog[-1][0]) / 100.0
        passos = int(prog[-1][1].replace(",", ""))
        alvo = int(prog[-1][2].replace(",", ""))
        # taxa marginal entre as duas últimas linhas, via passos/s cumulativo
        taxa = float(prog[-1][3])
        if len(prog) > 1:
            p0, p1 = prog[max(0, len(prog) - 4)], prog[-1]
            s0, s1 = int(p0[1].replace(",", "")), int(p1[1].replace(",", ""))
            t0 = s0 / max(float(p0[3]), 1e-9)
            t1 = s1 / max(float(p1[3]), 1e-9)
            if t1 > t0:
                taxa = (s1 - s0) / (t1 - t0)
        cob = (f"cob {float(ev[-1][1]):>5.1%}  sr {float(ev[-1][2]):>5.1%}"
               if ev else "sem eval ainda")
        mx = f" máx {max(float(e[1]) for e in ev):>5.1%}" if ev else ""
        # Uma corrida que já parou não tem ETA: mostrar um faz o painel prometer
        # trabalho que não existe, e a soma "a última termina em" fica errada.
        if parou:
            fim = re.search(r"Na=\d+ Ns=\d+: (.*)", txt)
            print(f"   {_barra(1.0)} PAROU {' ':>6} │ {rot:<22} │ "
                  f"{passos:>9,} passos │ {cob}{mx}")
            if fim:
                print(f"   {'':>22} └─ teste: {fim.group(1)}")
            continue
        eta = (alvo - passos) / max(taxa, 1e-9)
        fins.append(eta)
        fim = time.strftime("%d/%m %H:%M", time.localtime(time.time() + eta))
        print(f"   {_barra(frac)} {frac:>4.0%} {_hm(eta)} │ {rot:<22} │ "
              f"{taxa:>5.0f} p/s │ {cob}{mx} │ ~{fim}")
    if fins:
        print(f"   -> a última termina em ~{_hm(max(fins)).strip()}\n")
    else:
        print()


# --- grade do Algoritmo 4.4 -------------------------------------------------

def alg44(raiz: Path, progresso: Path, alvo: int, total: int) -> None:
    prontas = _ler(progresso)
    # RESOLVE os dois lados: o registro grava caminho RELATIVO (como veio de
    # `--raiz`) e o `rglob` devolve ABSOLUTO. Comparar strings cruas nunca casa, e o
    # efeito é silencioso — corridas já abortadas reaparecem como "vivas", exibindo
    # a última avaliação de antes de morrer.
    feitas = {Path(x["dir"]).resolve() for x in prontas if x.get("dir")}
    vivas = []
    if raiz.exists():
        for m in raiz.rglob("metrics.jsonl"):
            if m.parent.resolve() in feitas:
                continue
            cfg = m.parent / "config.json"
            linhas = _ler(m)
            if linhas and cfg.exists():
                # A semente é o último componente do caminho,
                # `runs/<exp>/<config_hash>/<seed_train>/`, como a §4.7 manda.
                vivas.append((json.loads(cfg.read_text())["config"], linhas,
                              m.parent.name))

    def _pt(c):
        return (c["aproximador"], c["passo_efetivo"], c["sigma"], c["resolucao"],
                c["idx_critico"])

    n_pt = len({_pt(c) for c, _, _ in vivas} |
               {_pt(x) for x in prontas if "aproximador" in x})
    print(f"── ALGORITMO 4.4 · {len(prontas)} paradas / {len(vivas)} vivas de "
          f"{total} corridas · {n_pt} pontos ──")
    if not prontas and not vivas:
        print("   nada em `" + str(raiz) + "`")
        return

    if prontas:
        ab = sum(1 for x in prontas if x.get("abortou"))
        print(f"   PARADAS: {ab} por instabilidade · {len(prontas) - ab} "
              f"completaram o orçamento")
        for x in sorted(prontas, key=lambda y: -y["cobertura"]):
            mod = MODELO.get(x["aproximador"], x["aproximador"])
            if x.get("abortou"):
                # O passo em que a assertiva da §4.2 estourou separa "divergiu logo"
                # de "aprendeu e depois divergiu".
                m = re.search(r"treino (\d+)", x["abortou"])
                est = (f"ABORTOU no passo {int(m.group(1)):>9,}" if m
                       else "ABORTOU")
            else:
                est = f"completou {x['passos']:>9,} passos"
            print(f"   {'':16} {'':6} │ {mod:>11} s{x.get('seed_train', '?')} │ "
                  f"passo {x['passo_efetivo']:.1e} σ{x['sigma']:.2f} "
                  f"T{x['resolucao']:>3} crit{x['idx_critico']} │ "
                  f"cob {x['cobertura']:>6.1%}   SR {x['sr']:>5.1%} │ {est}")

    etas = []
    # Ordena por PONTO e depois por semente: os pares do mesmo ponto ficam
    # adjacentes e a variância entre sementes salta à vista.
    for cfg, L, sd in sorted(vivas, key=lambda t: (
            t[0]["aproximador"], -t[0]["passo_efetivo"], t[0]["resolucao"], t[2])):
        u = L[-1]
        frac = u["passo"] / alvo
        taxa = _taxa_marginal(L)
        eta = (alvo - u["passo"]) / taxa if taxa > 0 else float("nan")
        etas.append(eta)
        antes = [x["cobertura"] for x in L[-6:-1]]
        seta = "="
        if antes:
            d = u["cobertura"] - sum(antes) / len(antes)
            seta = "↑" if d > 0.02 else "↓" if d < -0.02 else "="
        mod = MODELO.get(cfg.get("aproximador", ""), "?")
        print(f"   {_barra(frac)} {frac:>4.0%} {_hm(eta)} │ {mod:>11} s{sd} │ "
              f"passo {cfg['passo_efetivo']:.1e} σ{cfg['sigma']:.2f} "
              f"T{cfg['resolucao']:>3} crit{cfg['idx_critico']} │ "
              f"cob {u['cobertura']:>6.1%} {seta} SR {u['sr']:>5.1%} │ "
              f"máx {max(x['cobertura'] for x in L):>6.1%} │ {taxa:>5.0f} p/s")
    fin = [e for e in etas if e == e]
    if fin:
        print(f"   -> a última termina em ~{_hm(max(fin)).strip()} "
              f"(rodam em paralelo, então vale a MAIOR)")


# --- grade do PPO -----------------------------------------------------------

def ppo(progresso: Path, raiz: Path, proc: int, sementes: int, janela: int) -> None:
    feitas = _ler(progresso)
    # Total DEDUZIDO de `pontos()` da própria grade, não fixo: quando o espaço muda
    # de 12 para 60 pontos, um total codificado passa a mentir em silêncio.
    total = None
    try:
        import importlib.util
        esp = importlib.util.spec_from_file_location(
            "grid_ppo", RAIZ / "scripts" / "spec06_grid_ppo.py")
        mod = importlib.util.module_from_spec(esp)
        sys.modules["grid_ppo"] = mod
        esp.loader.exec_module(mod)
        total = len(mod.pontos()) * sementes
    except Exception:      # noqa: BLE001 — o painel não pode cair por causa disto
        pass

    frac = len(feitas) / total if total else 0.0
    print(f"\n── BRAÇO PPO · {len(feitas)}"
          + (f"/{total}" if total else "") + " pontos ──")
    print(f"   {_barra(frac)} {frac:>4.0%}")

    if feitas and total and total > len(feitas):
        # ETA por ONDAS, não por vazão: os pontos rodam `proc` a cada vez, e a
        # última onda pode ficar incompleta. Média das últimas `janela`, não de
        # todas, para acompanhar mudança de carga.
        rec = [x["parede_s"] for x in feitas[-janela:] if x.get("parede_s")]
        if rec:
            medio = sum(rec) / len(rec)
            ondas = math.ceil((total - len(feitas)) / proc)
            falta = ondas * medio
            fim = time.strftime("%H:%M", time.localtime(time.time() + falta))
            print(f"   {medio / 60:.1f} min/ponto (últimos {len(rec)}) · "
                  f"{ondas} ondas de {proc} · resta {_hm(falta).strip()} (~{fim})")

    fal = [x for x in feitas if x.get("falhou")]
    if fal:
        print(f"   {len(fal)} FALHARAM: "
              + ", ".join(sorted({x["falhou"] for x in fal})))
    viv = [x for x in feitas if not x.get("falhou")]
    for x in sorted(viv, key=lambda y: -y["cobertura"])[:6]:
        print(f"   {x['rotulo']:<30} cob {x['cobertura']:>6.1%} SR {x['sr']:>5.1%}")

    # EM VOO: o log por ponto, que o `capture_output` antigo jogava fora. Sem isto
    # eram horas sem saber se um ponto estava andando, travado ou só lento.
    if raiz.exists():
        prontos = {x["rotulo"] for x in feitas}
        voo = []
        for log in raiz.glob("*/saida.log"):
            if log.parent.name in prontos:
                continue
            ult = [l for l in log.read_text().splitlines()
                   if "/" in l and "%" in l]
            if ult:
                voo.append((log.parent.name, ult[-1].strip()))
        if voo:
            print(f"\n   em voo ({len(voo)}):")
            for nome, linha in sorted(voo):
                print(f"     {nome:<30} {linha}")


# --- efeito marginal por eixo ----------------------------------------------

def eixos(progresso: Path, chaves: tuple[str, ...], metrica: str,
          rotulo: str) -> None:
    """Média da métrica por nível de cada eixo — onde o sinal está.

    NÃO é análise pareada: mede a marginal, que mistura os outros eixos. Serve para
    ver ONDE olhar, não para concluir. Foi confiar numa marginal que me fez declarar
    `σ = 0,35` melhor antes de controlar passo e crítico — e ao controlar, os três
    pares diretos dispersavam de -40 a +26 pp.
    """
    L = [x for x in _ler(progresso)
         if not x.get("falhou") and not x.get("abortou")]
    if not L or metrica not in L[0]:
        return
    print(f"\n── EFEITO MARGINAL · {rotulo} ({len(L)} corridas, `{metrica}`) ──")
    for k in chaves:
        if k not in L[0]:
            continue
        d = collections.defaultdict(list)
        for x in L:
            d[x[k]].append(x[metrica])
        linha = " · ".join(f"{n}={sum(v) / len(v):.1%}" for n, v in sorted(d.items()))
        print(f"   {k:>16}: {linha}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz-alg44", default="runs/spec06-longa")
    ap.add_argument("--progresso-alg44",
                    default="reports/spec06-longa.progresso.jsonl")
    ap.add_argument("--alvo-alg44", type=int, default=10_000_000,
                    help="`--b-sweep` da corrida, para a barra")
    ap.add_argument("--total-alg44", type=int, default=12)
    ap.add_argument("--raiz-ppo", default="runs/spec06-ppo")
    ap.add_argument("--progresso-ppo", default="reports/spec06-ppo.progresso.jsonl")
    ap.add_argument("--processos-ppo", type=int, default=4)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--janela", type=int, default=8)
    ap.add_argument("--eixos", action="store_true",
                    help="acrescenta o efeito marginal por eixo das duas grades")
    ap.add_argument("--metrica", default="cobertura")
    ap.add_argument("--logs-multi", default="reports/spec06-multi-*.log",
                    help="glob dos logs da escalada multiagente")
    ap.add_argument("--multi", action="store_true",
                    help="mostra SÓ a escalada multiagente")
    a = ap.parse_args()

    print(f"{time.strftime('%d/%m %H:%M:%S')}\n{REFERENCIAS}\n")
    logs = sorted(RAIZ.glob(a.logs_multi))
    if a.multi:
        multi(logs)
        return 0
    multi(logs)
    alg44(RAIZ / a.raiz_alg44, RAIZ / a.progresso_alg44, a.alvo_alg44,
          a.total_alg44)
    ppo(RAIZ / a.progresso_ppo, RAIZ / a.raiz_ppo, a.processos_ppo, a.sementes,
        a.janela)
    if a.eixos:
        eixos(RAIZ / a.progresso_alg44,
              ("passo_efetivo", "sigma", "resolucao", "tupla", "idx_critico"),
              a.metrica, "Algoritmo 4.4")
        eixos(RAIZ / a.progresso_ppo,
              ("lr", "resolucao", "tupla", "vf_loss_coeff"), a.metrica, "PPO")
    print("\n`cob` é a ÚLTIMA avaliação, sem seleção; `máx` carrega maldição do "
          "vencedor (+0,9 pp medido).\nAs duas saem do banco `val` — o número do "
          "relatório sai do `test`, ao fim da corrida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
