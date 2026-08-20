"""Campanha dos três baselines — SPEC-02 T-02.9 e T-02.10.

Roda `random_uniforme`, `guloso` e `kmeans_tsp` nos **mesmos** cenários e produz
os três artefatos: `agregado.csv`, `comparacoes.csv` e `manifest.json`, mais o
`episodios.csv` de grão fino (fora do git).

O QUE ESTE SCRIPT VERIFICA, ALÉM DE MEDIR

**Ordenação esperada** (T-02.9 §1), em todas as células:

    LB-1  ≤  kmeans_tsp  ≤  guloso  ≤  random_uniforme

Cada uma tem leitura própria quando falha:

- **LB-1 violado** → bug. O harness já levanta `AssertionError` por episódio.
- **`kmeans_tsp > guloso`** → provável falha do **executor** de waypoints, não do
  planejador. O par `(t_plan_s, t_conc_s)` diz qual dos dois: se `t_plan` é bom e
  `t_conc` é ruim, o gargalo é a execução.
- **`guloso > random`** → inversão de sinal no guloso, o modo de falha que a
  T-02.9 §3 nomeia.

Uso:
    uv run python scripts/spec02_baselines.py
    uv run python scripts/spec02_baselines.py --n-episodios 100 --celulas 2:12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from uavwnn.agents import CaminhadaAleatoria, Guloso, KMeansTSP
from uavwnn.eval import ArvoreSementes, disjuntos, resumir_episodios
from uavwnn.eval.harness import (
    agregado,
    avaliar,
    comparacoes,
    escrever_csv,
    escrever_episodios,
    escrever_manifesto,
)
from uavwnn.refenv import LAMENZA_003

# Células do critério de saída: `Na ∈ {1,2,4} × Ns = 12`.
CELULAS = [(1, 12), (2, 12), (4, 12)]

# Comparações **pré-registradas**. As adjacentes na ordenação esperada.
PARES = [("kmeans_tsp", "guloso"), ("guloso", "random_uniforme"),
         ("kmeans_tsp", "random_uniforme")]
PRIMARIAS = [("guloso", "random_uniforme")]
"""H6/G1 se mede contra o guloso; a comparação com a política aprendida entra na
SPEC-06. Aqui a primária é a que valida o harness."""

ORDEM_ESPERADA = ["kmeans_tsp", "guloso", "random_uniforme"]


def politicas_novas() -> dict:
    """Instâncias frescas por célula — políticas guardam estado no `reset`."""
    return {"random_uniforme": CaminhadaAleatoria(),
            "guloso": Guloso(),
            "kmeans_tsp": KMeansTSP()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodios", type=int, default=500)
    ap.add_argument("--celulas", default="",
                    help="ex. '2:12,4:12'; vazio = Na ∈ {1,2,4} × Ns = 12")
    ap.add_argument("--raiz", type=int, default=20260809)
    ap.add_argument("--dir-saida", default="results/spec02-baselines")
    a = ap.parse_args()

    celulas = CELULAS
    if a.celulas:
        celulas = [tuple(int(x) for x in item.split(":"))
                   for item in a.celulas.split(",")]

    arv = ArvoreSementes(a.raiz)
    # Invariante da T-02.6 §1, verificado antes de gastar tempo de simulação.
    if not disjuntos(arv.cenarios("treino", a.n_episodios),
                     arv.cenarios("avaliacao", a.n_episodios)):
        sys.exit("sementes de treino e avaliação se sobrepõem — vazamento")

    saida = Path(a.dir_saida)
    print(f"SPEC-02 · {a.n_episodios} episódios pareados · raiz {a.raiz}")
    print(f"células: {celulas}\n")

    todo_agregado, todas_comparacoes = [], []
    problemas: list[str] = []
    t_inicio = time.perf_counter()

    for na, ns in celulas:
        cfg = LAMENZA_003(na, ns, phase="eval")
        pols = politicas_novas()
        print(f"=== Na={na}, Ns={ns} ===")
        t0 = time.perf_counter()
        res = avaliar(pols, cfg, arv, a.n_episodios)
        parede = time.perf_counter() - t0

        linhas_ag = agregado(res, arv)
        linhas_cmp = comparacoes(res, arv, PARES, PRIMARIAS)
        todo_agregado += linhas_ag
        todas_comparacoes += linhas_cmp

        escrever_episodios(saida / f"Na{na}_Ns{ns}" / "episodios.csv", res, cfg, pols)
        escrever_manifesto(saida / f"Na{na}_Ns{ns}" / "manifest.json", arv, cfg,
                           pols, a.n_episodios, parede)

        # ---- T-02.9 §1: ordenação esperada -------------------------------
        por_id = {l["politica"]: l for l in linhas_ag}
        media = {k: por_id[k]["media_t_conc_pen"] for k in ORDEM_ESPERADA}
        lb = por_id["guloso"]["media_lb1"]
        print(f"\n  ordenação: LB-1 {lb:.2f} ≤ "
              + " ≤ ".join(f"{k} {media[k]:.2f}" for k in ORDEM_ESPERADA))

        if lb > media["kmeans_tsp"] + 1e-6:
            problemas.append(f"Na{na}_Ns{ns}: LB-1 acima do kmeans_tsp — BUG")
        for x, y in zip(ORDEM_ESPERADA, ORDEM_ESPERADA[1:]):
            if media[x] > media[y]:
                problemas.append(f"Na{na}_Ns{ns}: {x} pior que {y}")
        for c in linhas_cmp:
            if c["subconjunto"] != "todos":
                continue
            par = (c["politica_a"], c["politica_b"])
            adjacente = par in list(zip(ORDEM_ESPERADA, ORDEM_ESPERADA[1:]))
            marca = "conclusivo" if not c["contem_zero"] else "INCONCLUSIVO"
            print(f"    {par[0]:>12} − {par[1]:<16} Δ={c['delta_media']:>9.2f} "
                  f"IC=[{c['delta_ic_lo']:>8.2f},{c['delta_ic_hi']:>8.2f}] "
                  f"MDE={c['mde_95']:>6.2f} tv={c['taxa_vitoria_a']:.2f} {marca}")
            if adjacente and c["contem_zero"]:
                problemas.append(f"Na{na}_Ns{ns}: {par[0]}−{par[1]} inconclusivo")

        # diagnósticos de coordenação
        print(f"    {'':>12}   Gini das visitas / coletas redundantes:")
        for l in linhas_ag:
            print(f"    {l['politica']:>12}   {l['gini_visitas']:>5.3f} / "
                  f"{l['coletas_redundantes']:>5.1f}")
        print()

    escrever_csv(saida / "agregado.csv", todo_agregado)
    escrever_csv(saida / "comparacoes.csv", todas_comparacoes)

    print("=" * 78)
    if problemas:
        print("PROBLEMAS na validação do harness:")
        for p in problemas:
            print(f"  - {p}")
    else:
        print("Ordenação esperada confirmada em todas as células, com as")
        print("comparações adjacentes conclusivas (0 ∉ IC).")
    print(f"\nartefatos: {saida}/agregado.csv, comparacoes.csv, "
          f"Na*_Ns*/{{episodios.csv,manifest.json}}")
    print(f"tempo: {(time.perf_counter() - t_inicio)/60:.1f} min")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
