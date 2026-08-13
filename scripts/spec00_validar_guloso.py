"""SPEC-00 — Validação contra os resultados publicados.

Roda a heurística gulosa do próprio trabalho de referência, no ambiente de
referência, e compara o tempo médio de conclusão com os valores da fig. 7.3 da
dissertação (barras "Heuristic").

É o teste de que a montagem está correta: se o guloso reproduz a curva
publicada, então submodule, versão do simulador, configuração e laço de
avaliação estão todos certos. Nenhuma política aprendida é necessária.

Configuração espelhada de `visualize_performance.py`, que gerou a figura:
`algorithm_iteration_interval=0.1` para a heurística (e não 0,5, que é o da
política), `end_when_all_collected` no default `True`, e quebra na primeira
`info` não vazia.

Uso:
    uv run python scripts/spec00_validar_guloso.py [n_episodios]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from uavwnn.refenv import LAMENZA_003, load_heuristics, run_episode

# Lidos das figuras publicadas em
# vendor/.../statistics/completion_times/completion_times-{2,4,8}a-greedy.png
# (barras "Heuristic"). Precisão de leitura ~0,3 s.
PUBLICADO: dict[tuple[int, int], float] = {
    (2, 12): 29.6, (2, 24): 38.2, (2, 36): 43.2,
    (4, 12): 28.2, (4, 24): 35.4, (4, 36): 39.7,
    (8, 12): 27.6, (8, 24): 33.2, (8, 36): 36.7,
}

TOLERANCIA_REL = 0.10
"""Faixa aceita: 10 % do valor publicado.

Justificativa: os valores publicados foram lidos de um gráfico de barras
(precisão ~0,3 s), o script de referência não semeia o gerador, e o desvio
padrão do tempo de conclusão é da ordem de 20 % da média — com 500 episódios o
erro padrão da média fica em ~1 %.
"""


def main() -> int:
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    h = load_heuristics()

    print(f"Guloso de referência, {n_ep} episódios por célula, dt = 0,1 s "
          "(como em visualize_performance.py)\n")
    print(f"{'Na':>3} {'Ns':>3} {'medido':>8} {'IC95':>16} {'publicado':>10} "
          f"{'erro':>8} {'':>4}")

    linhas = []
    falhas = []
    for (n_a, n_s), pub in sorted(PUBLICADO.items()):
        cfg = LAMENZA_003(n_a, n_s, phase="eval_heuristic")
        greedy = h.create_greedy_heuristics(n_a - 1, cfg.state_num_closest_sensors)

        t0 = time.perf_counter()
        res = [run_episode(cfg, greedy, seed=i) for i in range(n_ep)]
        wall = time.perf_counter() - t0

        ct = np.array([r.completion_time for r in res])
        media = float(ct.mean())
        ic = 1.96 * float(ct.std(ddof=1)) / np.sqrt(n_ep)
        succ = float(np.mean([r.success for r in res]))
        erro_rel = (media - pub) / pub
        ok = abs(erro_rel) <= TOLERANCIA_REL
        if not ok:
            falhas.append((n_a, n_s, media, pub, erro_rel))

        print(f"{n_a:>3} {n_s:>3} {media:>8.2f} "
              f"[{media - ic:>6.2f},{media + ic:>6.2f}] {pub:>10.1f} "
              f"{erro_rel:>+7.1%} {'ok' if ok else 'FORA':>5}")
        linhas.append(dict(n_agents=n_a, n_sensors=n_s, medido=media, ic95=ic,
                           publicado=pub, erro_rel=erro_rel, sucesso=succ,
                           n_episodios=n_ep, wall=wall))

    print()
    erros = np.array([abs(l["erro_rel"]) for l in linhas])
    print(f"erro relativo: mediana {np.median(erros):.1%}, máximo {erros.max():.1%}")
    vies = np.mean([l["erro_rel"] for l in linhas])
    print(f"viés médio (nosso − publicado): {vies:+.1%}")
    print(f"taxa de sucesso mínima: {min(l['sucesso'] for l in linhas):.3f}")

    with open("reports/spec00-validacao-guloso.json", "w") as f:
        json.dump({"linhas": linhas, "tolerancia": TOLERANCIA_REL}, f, indent=2)
    print("\nartefato: reports/spec00-validacao-guloso.json")

    if falhas:
        print(f"\n{len(falhas)} célula(s) fora da tolerância de {TOLERANCIA_REL:.0%}:")
        for n_a, n_s, m, p, e in falhas:
            print(f"  Na={n_a} Ns={n_s}: medido {m:.2f} vs publicado {p:.1f} ({e:+.1%})")
        return 1
    print("\nVEREDITO: guloso reproduz a fig. 7.3 dentro da tolerância.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
