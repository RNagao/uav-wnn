"""MADDPG sob ruído × os números publicados de Lamenza.

Coloca lado a lado, nas células com valor publicado (`Ns = 24`): o número de
Lamenza, o degrau 0 (nossa reprodução limpa), o degrau 1 (ruído bruto) e o degrau
4 (KF+controle). A pergunta: **quanto o ruído de GPS degrada o MADDPG publicado, e
o filtro o devolve ao patamar de Lamenza?**

Pareado por episódio (mesmo cenário nos três degraus; degraus 1 e 4 compartilham a
realização de ruído por CRN), com IC da diferença por bootstrap pareado — o mesmo
procedimento que o teste A/A da SPEC-02 calibrou.

    uv run --group baseline python scripts/spec07_maddpg_vs_lamenza.py
    uv run --group baseline python scripts/spec07_maddpg_vs_lamenza.py --episodios 100 --sigmas 10,20

Rótulos (iguais aos do experimento-mãe): é H3/fusão, não H1 (MADDPG é referência,
D7); `Na ∈ {2,4,8}`, `Ns = 24`; regime L→R (treino limpo, avaliação ruidosa).
`σ = 20` está **fora da grade declarada** {1,3,5,10} — entra só como estresse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec02_baseline_maddpg import checkpoints  # noqa: E402
from spec02_maddpg_rollout import carregar_ator  # noqa: E402
from spec07_maddpg_ruido import rodar_episodio  # noqa: E402
from uavwnn.eval.stats import bootstrap_pareado  # noqa: E402
from uavwnn.refenv import LAMENZA_003  # noqa: E402

# Fig. 7.3 de Lamenza, lidas das barras (só Ns=24), precisão ~0,3 s.
PUBLICADO = {(2, 24): 29.6, (4, 24): 13.8, (8, 24): 9.7}


def _serie(cfg, politica, degrau, sigma, episodios):
    cts, oks = [], []
    for s in range(episodios):
        ct, ok, _ = rodar_episodio(cfg, politica, degrau, sigma, seed=s)
        cts.append(ct)
        oks.append(ok)
    return np.array(cts), np.array(oks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=100)
    ap.add_argument("--sigmas", default="10,20")
    ap.add_argument("--saida", default="reports/spec07-maddpg-vs-lamenza.json")
    a = ap.parse_args()

    disp = {(na, ns): p for na, ns, p in checkpoints()}
    if not disp:
        sys.exit("nenhum checkpoint; git submodule update --init --recursive")
    sigmas = [float(x) for x in a.sigmas.split(",")]
    rng = np.random.default_rng(20260815)
    celulas = [(2, 24), (4, 24), (8, 24)]

    print(f"MADDPG sob ruído × Lamenza · {a.episodios} episódios pareados · Ns=24\n")
    print("H3/fusão (não H1); regime L→R; σ=20 fora da grade (estresse).\n")

    artefato = {"config": {"episodios": a.episodios, "sigmas": sigmas}, "celulas": {}}

    for na, ns in celulas:
        cfg = LAMENZA_003(na, ns, phase="eval")
        pol = carregar_ator(disp[(na, ns)], cfg.observation_size)
        pub = PUBLICADO[(na, ns)]

        ct0, ok0 = _serie(cfg, pol, 0, 0.0, a.episodios)
        print(f"a_{na}-s_{ns}  (Lamenza publicado: {pub:.1f} s)")
        print(f"  degrau 0 (limpo):  {ct0.mean():.1f} s   sucesso {ok0.mean():.0%}   "
              f"(vs publicado: {ct0.mean()-pub:+.1f} s)")

        cel = artefato["celulas"][f"a_{na}-s_{ns}"] = {
            "publicado": pub, "degrau0": {"ct": float(ct0.mean()), "sucesso": float(ok0.mean())},
            "sigmas": {},
        }

        for sigma in sigmas:
            ct1, ok1 = _serie(cfg, pol, 1, sigma, a.episodios)
            ct4, ok4 = _serie(cfg, pol, 4, sigma, a.episodios)
            # diferenças pareadas (menor é melhor)
            deg = bootstrap_pareado(ct1, ct0, rng)   # d1 − d0: quanto o ruído estraga
            res = bootstrap_pareado(ct4, ct0, rng)   # d4 − d0: quanto sobra após o KF
            rec = bootstrap_pareado(ct1, ct4, rng)   # d1 − d4: quanto o KF recupera
            frac = (ct1.mean() - ct4.mean()) / (ct1.mean() - ct0.mean()) \
                if abs(ct1.mean() - ct0.mean()) > 1e-6 else float("nan")

            marca = "  [estresse, fora da grade]" if sigma not in (1, 3, 5, 10) else ""
            print(f"  σ={sigma:.0f}{marca}")
            print(f"    degrau 1 (bruto):  {ct1.mean():.1f} s   sucesso {ok1.mean():.0%}   "
                  f"Δ(d1−d0) = {deg.delta_media:+.2f} [{deg.delta_ic_lo:+.2f}; {deg.delta_ic_hi:+.2f}]"
                  f"{'  (degrada)' if deg.conclusivo and deg.delta_media > 0 else '  (≈ limpo)'}")
            print(f"    degrau 4 (KF):     {ct4.mean():.1f} s   sucesso {ok4.mean():.0%}   "
                  f"Δ(d4−d0) = {res.delta_media:+.2f} [{res.delta_ic_lo:+.2f}; {res.delta_ic_hi:+.2f}]"
                  f"{'  (≈ limpo → recuperou)' if res.contem_zero else ''}")
            print(f"    KF recupera Δ(d1−d4) = {rec.delta_media:+.2f} "
                  f"[{rec.delta_ic_lo:+.2f}; {rec.delta_ic_hi:+.2f}]  "
                  f"({frac:.0%} do estrago)")

            cel["sigmas"][f"{sigma:g}"] = {
                "degrau1": {"ct": float(ct1.mean()), "sucesso": float(ok1.mean())},
                "degrau4": {"ct": float(ct4.mean()), "sucesso": float(ok4.mean())},
                "delta_d1_d0": deg.como_linha(),
                "delta_d4_d0": res.como_linha(),
                "delta_d1_d4": rec.como_linha(),
                "fracao_recuperada": float(frac),
            }
        print()

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.saida, "w") as f:
        json.dump(artefato, f, indent=2, ensure_ascii=False)
    print(f"artefato: {a.saida}")
    print("\nRessalva: os valores publicados são de interval=0,5 s e lidos das barras")
    print("(~0,3 s). A comparação honesta é degrau 0 ≈ publicado (validação) e o efeito")
    print("do ruído/KF como diferença PAREADA contra o próprio degrau 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
