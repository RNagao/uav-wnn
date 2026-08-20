"""SPEC-03 — Harness de custo: T1 tamanho, T3 operações.

Resolve a pendência **T1** com medição, não estimativa, usando a configuração
que a SPEC-01 escolheu (E2 quantil, T=8, N=48, n=12).

Uso:
    uv run python scripts/spec03_custo.py
    uv run python scripts/spec03_custo.py --episodios 30   # mais rápido
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from uavwnn.custo import custo_mlp, custo_ntuple, ocupacao_por_ram
from uavwnn.custo.operacoes import CICLOS, ciclos, operacoes_mlp, operacoes_ntuple
from uavwnn.encoding import TermometroQuantil, coletar_corpus
from uavwnn.wnn import NTupleRegressor

sys.path.insert(0, "scripts")
from spec01_estudo import alvo_guloso, erro_angular  # noqa: E402

T, N, N_TUPLA = 8, 48, 12


def treina_mlp(X, Y, ocultas=32, epocas=600, lr=3e-3, lote=256, seed=0):
    """MLP pareada em NumPy puro, com Adam.

    Adam e 600 épocas não são luxo: com descida simples e 400 épocas a MLP
    ficava em 23°, e reportar isso seria construir um espantalho. O teste de
    convergência está no relatório.
    """
    r = np.random.default_rng(seed)
    dims = [X.shape[1], ocultas, ocultas, 2]
    W = [r.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
    b = [np.zeros(x) for x in dims[1:]]
    mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(x) for x in b]; vb = [np.zeros_like(x) for x in b]
    t = 0
    for _ in range(epocas):
        idx = r.permutation(len(X))
        for s in range(0, len(idx), lote):
            j = idx[s:s + lote]; xb, yb = X[j], Y[j]
            h1 = np.maximum(0, xb @ W[0] + b[0])
            h2 = np.maximum(0, h1 @ W[1] + b[1])
            o = h2 @ W[2] + b[2]
            g = 2 * (o - yb) / len(j)
            gW = [None] * 3; gb = [None] * 3
            gW[2] = h2.T @ g; gb[2] = g.sum(0); g2 = (g @ W[2].T) * (h2 > 0)
            gW[1] = h1.T @ g2; gb[1] = g2.sum(0); g1 = (g2 @ W[1].T) * (h1 > 0)
            gW[0] = xb.T @ g1; gb[0] = g1.sum(0)
            t += 1
            for i in range(3):
                for p, gr, m, v in ((W[i], gW[i], mW, vW), (b[i], gb[i], mb, vb)):
                    m[i][:] = 0.9 * m[i] + 0.1 * gr
                    v[i][:] = 0.999 * v[i] + 0.001 * gr ** 2
                    p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def prev_mlp(Wb, X):
    W, b = Wb
    h1 = np.maximum(0, X @ W[0] + b[0])
    h2 = np.maximum(0, h1 @ W[1] + b[1])
    return h2 @ W[2] + b[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=60)
    args = ap.parse_args()

    corpus = coletar_corpus([(1, 24)], n_episodios=args.episodios)
    X = corpus.obs; D = X.shape[1]
    Y = alvo_guloso(X, 0, 12)
    rng = np.random.default_rng(0); perm = rng.permutation(len(X))
    tr, te = perm[:int(0.75 * len(X))], perm[int(0.75 * len(X)):]
    print(f"corpus {len(X)} observações, D={D}, treino {len(tr)}\n")

    # -- T1: ocupação e curva de saturação ---------------------------------
    enc = TermometroQuantil(T).ajustar(X[tr]); B = enc.codificar(X)
    print("=" * 72)
    print("T1  Curva de ocupação (RNF-08) — satura?")
    print("=" * 72)
    net = NTupleRegressor(n_bits=enc.n_bits, n_ram=N, tamanho_tupla=N_TUPLA,
                          n_saidas=2, seed=1)
    curva, vistas, prev = [], 0, None
    print(f"{'amostras':>10} {'ocupadas':>10} {'elasticidade':>13}")
    for marco in (200, 500, 1000, 2000, 4000, len(tr)):
        lote = tr[vistas:marco]
        if len(lote) == 0:
            continue
        net.treinar(B[lote], Y[lote], eta=0.5, epocas=1, rng=np.random.default_rng(3))
        vistas = marco
        occ = int(ocupacao_por_ram(net.pesos).sum())
        el = None if prev is None else (np.log(occ) - np.log(prev[1])) / (np.log(vistas) - np.log(prev[0]))
        curva.append([vistas, occ, el])
        print(f"{vistas:>10} {occ:>10} {'' if el is None else f'{el:>13.3f}'}")
        prev = (vistas, occ)
    satura = curva[-1][2] is not None and curva[-1][2] < 0.05
    print(f"\n  saturou (elasticidade < 0,05)? {'SIM' if satura else 'NÃO'}"
          f"  ->  a alegação de tamanho sai com o horizonte declarado\n")

    # -- comparação pareada -------------------------------------------------
    net.treinar(B[tr], Y[tr], eta=0.5, epocas=8, rng=np.random.default_rng(3))
    e_wnn = float(np.median(erro_angular(net.prever(B[te]), Y[te])))
    occ = int(ocupacao_por_ram(net.pesos).sum())

    mlp = treina_mlp(X[tr].astype(np.float64), Y[tr])
    e_mlp = float(np.median(erro_angular(prev_mlp(mlp, X[te].astype(np.float64)), Y[te])))

    print("=" * 72)
    print("T1  Comparação pareada — mesma tarefa, mesmo corpus, ambos convergidos")
    print("=" * 72)
    c_mlp = custo_mlp([D, 32, 32, 2], rotulo="MLP pareada")
    print(f"{'modelo':>26} {'erro':>7} {'persistente':>13} {'residente':>11}")
    print(f"{'MLP 25-32-32-2':>26} {e_mlp:>6.1f}° {c_mlp.persistente/1024:>10.1f} KB "
          f"{c_mlp.residente/1024:>8.1f} KB")
    custos = {}
    for var in ("ordenado", "hash", "denso"):
        c = custo_ntuple(net.pesos, enc.n_bits, N_TUPLA, N, variante=var,
                         limiares_por_dim=T, n_dims=D, rotulo=f"WNN {var}")
        custos[var] = c
        print(f"{'WNN E2/T8/N48 ' + var:>26} {e_wnn:>6.1f}° {c.persistente/1024:>10.1f} KB "
              f"{c.residente/1024:>8.1f} KB")

    c_mad = custo_mlp([D, 256, 256, 2], rotulo="ator MADDPG")
    print(f"\nrazão dupla (D12), persistente, variante mais econômica:")
    melhor = custos["ordenado"]
    print(f"  {melhor.persistente/c_mlp.persistente:.1f}x MLP pareada  |  "
          f"{melhor.persistente/c_mad.persistente:.2f}x ator MADDPG "
          f"(ESTIMATIVA: P4 aberta, 256x256 do CleanRL)")

    # -- T3: operações e ciclos --------------------------------------------
    print()
    print("=" * 72)
    print("T3  Operações por decisão, e ciclos por classe de alvo")
    print("=" * 72)
    w = operacoes_ntuple(N, N_TUPLA, 2, n_dims=D, n_niveis=T)
    m = operacoes_mlp([D, 32, 32, 2])
    print(f"  multiplicações fp no aproximador:  WNN {w.mult_fp - 4:>5}   MLP {m.mult_fp - 4:>5}")
    print(f"  total de operações:                WNN {w.total:>5}   MLP {m.total:>5}\n")
    print(f"{'classe de alvo':>26} {'WNN':>9} {'MLP':>9} {'razão':>8}")
    ciclos_por_alvo = {}
    for alvo in CICLOS:
        cw, cm = ciclos(w, alvo), ciclos(m, alvo)
        ciclos_por_alvo[alvo] = [cw, cm]
        print(f"{alvo:>26} {cw:>9} {cm:>9} {cm/cw:>7.2f}x")

    artefato = {
        "config": {"T": T, "N": N, "n": N_TUPLA, "D": D, "encoding": "E2 quantil",
                   "Na": 1, "Ns": 24, "n_obs": len(X)},
        "t1": {
            "ocupadas": occ,
            "curva_ocupacao": curva,
            "saturou": bool(satura),
            "erro_clonagem_graus": {"wnn": e_wnn, "mlp": e_mlp},
            "persistente_bytes": {
                **{f"wnn_{k}": v.persistente for k, v in custos.items()},
                "mlp_pareada": c_mlp.persistente,
                "ator_maddpg_estimado": c_mad.persistente,
            },
            "detalhe_wnn_ordenado": custos["ordenado"].detalhe,
        },
        "t3": {"wnn": w.como_linha(), "mlp": m.como_linha(),
               "ciclos_por_alvo": ciclos_por_alvo},
    }
    with open("reports/spec03-custo.json", "w") as f:
        json.dump(artefato, f, indent=2)
    print("\nartefato: reports/spec03-custo.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
