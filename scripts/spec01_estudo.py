"""SPEC-01 — Estudo de binarização.

Escolhe, com evidência medida, o codificador binário que alimenta a rede
n-tuple: esquema, resolução `T` e o par `(T, N)`. Antes de treinar qualquer
política.

O filtro decisivo é T-01.5: se a codificação não sustenta **clonar o guloso**
— política determinística, simples e que a política aprendida precisa bater —
ela não vai sustentar RL. Custa minutos; descobrir isso dentro do laço de
treino custaria dias.

Uso:
    uv run python scripts/spec01_estudo.py            # varredura completa
    uv run python scripts/spec01_estudo.py --rapido   # grade reduzida
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from uavwnn.encoding import (
    PADDING,
    TermometroGlobal,
    TermometroPorSlot,
    TermometroQuantil,
    coletar_corpus,
)
from uavwnn.refenv import LAMENZA_003, load_heuristics
from uavwnn.wnn import NTupleRegressor

ESQUEMAS = {
    "E0 global": TermometroGlobal,
    "E1 por slot": TermometroPorSlot,
    "E2 quantil": TermometroQuantil,
}


def alvo_guloso(obs: np.ndarray, n_star_a: int, n_star_s: int) -> np.ndarray:
    """Ação do guloso em cada observação, como `(cos θ, sin θ)`.

    Regredir o ângulo em duas componentes evita a descontinuidade em 0/2π —
    média linear de ângulos é inválida, e é o mesmo problema que a SPEC-05
    enfrenta no ator (decisão C1).
    """
    h = load_heuristics()
    guloso = h.create_greedy_heuristics(n_star_a, n_star_s)
    thetas = np.array([guloso(o)[0] * 2 * np.pi - np.pi for o in obs])
    return np.stack([np.cos(thetas), np.sin(thetas)], axis=1)


def erro_angular(pred: np.ndarray, alvo: np.ndarray) -> np.ndarray:
    """Erro angular absoluto em graus, entre dois pares (cos, sin)."""
    a = np.arctan2(pred[:, 1], pred[:, 0])
    b = np.arctan2(alvo[:, 1], alvo[:, 0])
    d = np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))
    return np.degrees(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--episodios", type=int, default=40)
    args = ap.parse_args()

    Ts = [8, 16] if args.rapido else [8, 16, 32]
    Ns = [16, 48] if args.rapido else [16, 48, 128]
    n_tupla = 12

    # -- corpus -------------------------------------------------------------
    cfg = LAMENZA_003(1, 24)
    n_star_a, n_star_s = 0, cfg.state_num_closest_sensors
    print(f"coletando corpus: Na=1, Ns=24, {args.episodios} episódios por política...")
    t0 = time.perf_counter()
    corpus = coletar_corpus([(1, 24)], n_episodios=args.episodios)
    print(f"  {len(corpus)} observações em {time.perf_counter()-t0:.0f}s "
          f"(guloso {int((corpus.politica==0).sum())}, aleatório {int((corpus.politica==1).sum())})")

    X = corpus.obs
    Y = alvo_guloso(X, n_star_a, n_star_s)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    corte = int(0.75 * len(X))
    tr, te = perm[:corte], perm[corte:]
    print(f"  treino {len(tr)}, teste {len(te)}\n")

    # -- T-01.1: a distribuição depende do slot? ---------------------------
    print("=" * 74)
    print("T-01.1  Distribuição por slot de sensor")
    print("=" * 74)
    amps = []
    for j in range(n_star_s):
        col = X[:, 2 * j]
        v = col[~np.isclose(col, PADDING)]
        if v.size < 50:
            continue
        p05, p95 = np.percentile(v, [5, 95])
        amps.append(p95 - p05)
    print(f"amplitude p05–p95:  slot 1 = {amps[0]:.3f}   slot {len(amps)} = {amps[-1]:.3f}"
          f"   razão = {amps[-1]/amps[0]:.2f}x")
    print("A distribuição depende fortemente do slot: o sensor mais próximo")
    print("concentra, o mais distante espalha. Binning global desperdiça bits")
    print("justamente no slot 1 — que é o que dirige a ação.\n")

    # -- T-01.4 + T-01.5: cobertura de bits e clonagem do guloso -----------
    print("=" * 74)
    print("T-01.5  Clonagem do guloso  (o filtro decisivo)   +   T-01.4 cobertura")
    print("=" * 74)
    print(f"{'esquema':>12} {'T':>4} {'N':>5} {'bits':>6} {'N·n/B':>7} "
          f"{'cobert.':>8} {'err.med':>8} {'err.p90':>8} {'ocupadas':>9}")

    linhas = []
    for nome, cls in ESQUEMAS.items():
        for T in Ts:
            enc = cls(T).ajustar(X[tr])
            B = enc.codificar(X)
            for N in Ns:
                net = NTupleRegressor(
                    n_bits=enc.n_bits, n_ram=N, tamanho_tupla=n_tupla,
                    n_saidas=2, seed=1,
                )
                net.treinar(B[tr], Y[tr], eta=0.25, epocas=4,
                            rng=np.random.default_rng(3))
                err = erro_angular(net.prever(B[te]), Y[te])
                razao = N * n_tupla / enc.n_bits
                linhas.append(dict(
                    esquema=nome, T=T, N=N, bits=enc.n_bits,
                    razao_Nn_B=float(razao), cobertura=float(net.cobertura),
                    erro_mediano=float(np.median(err)),
                    erro_p90=float(np.percentile(err, 90)),
                    ocupadas=net.entradas_ocupadas(),
                ))
                print(f"{nome:>12} {T:>4} {N:>5} {enc.n_bits:>6} {razao:>7.2f} "
                      f"{net.cobertura:>7.0%} {np.median(err):>7.1f}° "
                      f"{np.percentile(err,90):>7.1f}° {net.entradas_ocupadas():>9}")

    # -- veredito -----------------------------------------------------------
    melhor = min(linhas, key=lambda r: r["erro_mediano"])
    print()
    print("=" * 74)
    print(f"melhor: {melhor['esquema']}  T={melhor['T']}  N={melhor['N']}  "
          f"→ erro mediano {melhor['erro_mediano']:.1f}°")

    por_esq = {}
    for r in linhas:
        por_esq.setdefault(r["esquema"], []).append(r["erro_mediano"])
    print("\nmelhor erro mediano por esquema:")
    for k, v in por_esq.items():
        print(f"  {k:>12}: {min(v):>6.1f}°")

    baixa = [r for r in linhas if r["cobertura"] < 0.999]
    if baixa:
        pior = min(baixa, key=lambda r: r["cobertura"])
        print(f"\ncobertura de bits abaixo de 100 % em {len(baixa)} de {len(linhas)} "
              f"configurações (mínima {pior['cobertura']:.0%} em "
              f"T={pior['T']}, N={pior['N']}): a rede ignora parte da entrada.")

    with open("reports/spec01-binarizacao.json", "w") as f:
        json.dump({"linhas": linhas, "amplitudes_por_slot": [float(a) for a in amps]},
                  f, indent=2)
    print("\nartefato: reports/spec01-binarizacao.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
