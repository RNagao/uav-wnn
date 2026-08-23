"""Painel ao vivo das corridas da SPEC-06 — Algoritmo 4.4 e braço PPO juntos.

POR QUE ELE EXISTE

As duas campanhas escrevem em lugares diferentes e nenhuma dá barra de progresso:

    Alg 4.4 (`spec06_busca.py`)  grava `metrics.jsonl` por corrida, a cada
                                 avaliação intercalada — é ali que está o progresso
                                 DENTRO de uma corrida. O `.progresso.jsonl` só
                                 recebe uma linha quando a corrida TERMINA, e numa
                                 corrida de 10 M isso é horas de silêncio.
    PPO (`spec06_grid_ppo.py`)   cada ponto é um subprocesso em ambiente efêmero;
                                 o registro só ganha linha ao fim de cada ponto.

Então "acompanhar" significa ler `metrics.jsonl` de quem está vivo e o
`.progresso.jsonl` de quem terminou, e é isso que este script faz.

    uv run python scripts/spec06_acompanhar.py
    watch -n 60 'uv run python scripts/spec06_acompanhar.py'

O QUE OLHAR, E O QUE NÃO OLHAR

A coluna `cob` é a **última** avaliação — sem seleção, é a estimativa honesta de onde
a política está agora. A coluna `máx` é o máximo sobre todas as avaliações, e carrega
maldição do vencedor: medido +0,9 pp na grade de desempate. Para julgar progresso,
use `cob`; `máx` só diz onde ela já esteve.

`cob` e `SR` saem do banco **`val`** (o de seleção), não do `test`. O número que vai
para o relatório é do `test`, e só existe quando a corrida termina.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _barra(fracao: float, largura: int = 18) -> str:
    cheio = int(max(0.0, min(1.0, fracao)) * largura)
    return "█" * cheio + "░" * (largura - cheio)


def _ler(p: Path) -> list[dict]:
    if not p.exists():
        return []
    fora = []
    for linha in p.read_text().splitlines():
        if not linha.strip():
            continue
        try:
            fora.append(json.loads(linha))
        except json.JSONDecodeError:
            pass          # última linha cortada por uma escrita em curso
    return fora


def alg44(raiz: Path, progresso: Path, alvo: int, total: int) -> None:
    """Corridas do Algoritmo 4.4: vivas por `metrics.jsonl`, prontas pelo registro."""
    prontas = _ler(progresso)
    # RESOLVE os dois lados: o registro grava caminho RELATIVO (`runs/…`, como veio
    # de `--raiz`) e o `rglob` devolve ABSOLUTO. Comparar as strings cruas nunca
    # casa, e o efeito é silencioso e enganoso — uma corrida já abortada reaparece
    # como "viva", exibindo a última avaliação de antes de morrer.
    feitas = {Path(x["dir"]).resolve() for x in prontas if x.get("dir")}
    vivas = []
    for m in raiz.rglob("metrics.jsonl"):
        if m.parent.resolve() in feitas:
            continue
        linhas = _ler(m)
        cfg = m.parent / "config.json"
        if linhas and cfg.exists():
            vivas.append((json.loads(cfg.read_text())["config"], linhas))

    print(f"── ALGORITMO 4.4 · {len(prontas)} prontas / {len(vivas)} vivas "
          f"de {total} ──")
    if prontas:
        ab = sum(1 for x in prontas if x.get("abortou"))
        vv = [x for x in prontas if not x.get("abortou")]
        print(f"   prontas: {ab} abortadas · {len(vv)} completas"
              + (f" · melhor cobertura {max(x['cobertura'] for x in vv):.1%}"
                 if vv else ""))
    for cfg, L in sorted(vivas, key=lambda t: -t[1][-1]["cobertura"]):
        u = L[-1]
        frac = u["passo"] / alvo
        # tendência: sobe, oscila ou cai? Compara a última com a média das 5 antes.
        antes = [x["cobertura"] for x in L[-6:-1]]
        seta = "="
        if antes:
            d = u["cobertura"] - sum(antes) / len(antes)
            seta = "↑" if d > 0.02 else "↓" if d < -0.02 else "="
        print(f"   {_barra(frac)} {frac:>4.0%} │ passo {cfg['passo_efetivo']:.0e} "
              f"σ{cfg['sigma']:.2f} T{cfg['resolucao']:>2} │ cob {u['cobertura']:>6.1%} "
              f"{seta} SR {u['sr']:>5.1%} │ máx {max(x['cobertura'] for x in L):>6.1%}")


def ppo(progresso: Path, total: int) -> None:
    """Braço PPO: só há registro ao fim de cada ponto, então mostra o que fechou."""
    feitas = _ler(progresso)
    print(f"\n── BRAÇO PPO · {len(feitas)} de {total} pontos ──")
    frac = len(feitas) / total if total else 0.0
    print(f"   {_barra(frac)} {frac:>4.0%}")
    if not feitas:
        print("   nenhum ponto fechou ainda — cada um só escreve ao terminar.")
        return
    falhas = [x for x in feitas if x.get("falhou")]
    if falhas:
        print(f"   {len(falhas)} FALHARAM: "
              + ", ".join(sorted({x["falhou"] for x in falhas})))
    vivas = [x for x in feitas if not x.get("falhou")]
    for x in sorted(vivas, key=lambda y: -y["cobertura"])[:6]:
        print(f"   {x['rotulo']:<28} cob {x['cobertura']:>6.1%} SR {x['sr']:>5.1%} "
              f"· {x['parede_s'] / 60:>5.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz-longa", default="runs/spec06-longa")
    ap.add_argument("--progresso-longa", default="reports/spec06-longa.progresso.jsonl")
    ap.add_argument("--alvo-longa", type=int, default=10_000_000,
                    help="`--b-sweep` da corrida longa, para a barra")
    ap.add_argument("--total-longa", type=int, default=12)
    ap.add_argument("--progresso-ppo", default="reports/spec06-ppo.progresso.jsonl")
    ap.add_argument("--total-ppo", type=int, default=24)
    a = ap.parse_args()

    print(f"{time.strftime('%d/%m %H:%M:%S')} · referências: aleatório 3,3 % de "
          f"cobertura · guloso SR 100 %\n")
    alg44(RAIZ / a.raiz_longa, RAIZ / a.progresso_longa, a.alvo_longa, a.total_longa)
    ppo(RAIZ / a.progresso_ppo, a.total_ppo)
    print("\n`cob` é a ÚLTIMA avaliação, sem seleção. `máx` carrega maldição do "
          "vencedor (+0,9 pp medido).\nAs duas saem do banco `val`; o número do "
          "relatório sai do `test`, ao fim da corrida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
