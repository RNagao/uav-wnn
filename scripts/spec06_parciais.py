"""Lê o progresso da grade **enquanto ela roda**, e agrega o parcial.

    uv run python scripts/spec06_parciais.py
    uv run python scripts/spec06_parciais.py --top 15 --por-eixo

O `spec06_busca.py` grava uma linha em `*.progresso.jsonl` a cada corrida concluída,
com `imap_unordered` — então o parcial é legível a qualquer momento, sem esperar as
horas todas. Este script existe porque ler o JSONL à mão não distingue o que importa:

- **quantas sementes** cada ponto já tem (um ponto com `n = 1` não é comparável a um
  com `n = 3`, e a diferença some se só a média for mostrada);
- **abortos**, que são informação sobre a estabilidade do ponto e não perda;
- **efeito por eixo**, que é o que diz onde gastar o próximo orçamento.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

EIXOS = ("passo_efetivo", "sigma", "resolucao", "tupla", "passos_m")


def ler(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--progresso", default="reports/spec06-busca.progresso.jsonl")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--por-eixo", action="store_true",
                    help="efeito marginal de cada eixo — onde gastar o próximo")
    a = ap.parse_args()

    linhas = ler(Path(a.progresso))
    if not linhas:
        print(f"nada em {a.progresso} ainda")
        return 0

    print(f"{len(linhas)} corridas concluídas")
    por_modelo = collections.Counter(x["aproximador"] for x in linhas)
    print("  por modelo: " + ", ".join(f"{k}={v}" for k, v in sorted(por_modelo.items())))
    ab = sum(x.get("abortou") is not None for x in linhas)
    if ab:
        print(f"  abortos: {ab} ({ab/len(linhas):.0%}) — informação sobre "
              f"estabilidade, não perda")

    # agrega por (modelo, ponto)
    grupos = collections.defaultdict(list)
    for x in linhas:
        grupos[(x["aproximador"], x["id"])].append(x)

    tabela = []
    for (m, i), g in grupos.items():
        cob = np.array([x["cobertura"] for x in g], float)
        tabela.append({
            "modelo": m, "id": i, "n": len(g),
            "cobertura": float(cob.mean()),
            "dp": float(cob.std(ddof=1)) if len(cob) > 1 else 0.0,
            "sr": float(np.mean([x["sr"] for x in g])),
            "ct": float(np.mean([x["ct_cens"] for x in g])),
            "abortos": sum(x.get("abortou") is not None for x in g),
            **{k: g[0].get(k) for k in EIXOS},
            "critico": g[0].get("phi", g[0].get("razao_critico")),
        })
    tabela.sort(key=lambda t: -t["cobertura"])

    print(f"\nMELHORES {a.top} (por cobertura no test)")
    print(f"{'modelo':>18} {'id':>4} {'n':>2} {'cobertura':>12} {'SR':>6} "
          f"{'CT':>6} {'passo':>8} {'σ':>5} {'T':>3} {'n_t':>3} {'m':>3} {'crít':>5}")
    for t in tabela[:a.top]:
        print(f"{t['modelo']:>18} {t['id']:>4} {t['n']:>2} "
              f"{t['cobertura']:>7.1%}±{t['dp']:>4.1%} {t['sr']:>5.1%} "
              f"{t['ct']:>6.0f} {t['passo_efetivo']:>8.1e} {t['sigma']:>5.2f} "
              f"{t['resolucao']:>3} {t['tupla']:>3} {t['passos_m']:>3} "
              f"{t['critico']:>5}")

    print(f"\nMELHOR POR MODELO (só pontos com todas as sementes)")
    n_max = max(t["n"] for t in tabela)
    for m in sorted(por_modelo):
        cand = [t for t in tabela if t["modelo"] == m and t["n"] == n_max]
        if not cand:
            print(f"  {m}: nenhum ponto completo ainda (máx {n_max} sementes)")
            continue
        b = cand[0]
        print(f"  {m}: passo {b['passo_efetivo']:.1e} σ {b['sigma']:.2f} "
              f"T {b['resolucao']} n {b['tupla']} m {b['passos_m']} "
              f"crít {b['critico']} -> cobertura {b['cobertura']:.1%}±{b['dp']:.1%}, "
              f"SR {b['sr']:.1%}")

    if a.por_eixo:
        print("\nEFEITO MARGINAL POR EIXO — média de cobertura por nível")
        print("(é o que diz onde o próximo orçamento rende; um eixo achatado não")
        print(" precisa de mais resolução)")
        for eixo in EIXOS:
            vals = collections.defaultdict(list)
            for x in linhas:
                if eixo in x:
                    vals[x[eixo]].append(x["cobertura"])
            if len(vals) < 2:
                continue
            ordenado = sorted(vals.items())
            amplitude = (max(np.mean(v) for _, v in ordenado)
                         - min(np.mean(v) for _, v in ordenado))
            corpo = "  ".join(
                f"{k if not isinstance(k, float) else f'{k:.2e}' if k < 0.01 else f'{k:.2f}'}"
                f"={np.mean(v):.1%}" for k, v in ordenado)
            print(f"  {eixo:>15} (Δ {amplitude:>5.1%})  {corpo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
