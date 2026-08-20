"""Teste diferencial: `NTupleRegressor` contra o código Julia do Katopodis.

O núcleo n-tuple era validado só por propriedades que eu mesmo escolhi (RMSE
numa regressão sintética, identidade `Δf = −η(f−y)`). Elas passariam mesmo se
eu tivesse entendido errado a semântica do modelo do Katopodis. Com o código
dele em mãos há um **oráculo independente**, e é ele que decide.

O teste força os dois lados ao mesmo mapeamento de tuplas. Do lado Julia isso
se faz com `RawBitsEncoder{UInt16}` — cada elemento da entrada carrega 16 bits,
e `indices_to_segment_offset` traduz índice global de bit para
`(segmento, deslocamento)` exatamente como o empacotamento little-endian que
usamos aqui. Exige `B = N · n`, para que `cld(input_len·res, n)` dê `N` nós.

Uso:
    uv run python scripts/spec04_diferencial.py --exportar <dir>
    <julia> scripts/diferencial.jl <dir>
    uv run python scripts/spec04_diferencial.py --comparar <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uavwnn.wnn import NTupleRegressor

# B = N*n é obrigatório para o partitioner do Ramnet dar exatamente N nós.
CONFIGS = [
    dict(nome="B64_n8_N8", n_bits=64, tamanho_tupla=8, n_ram=8, n_saidas=2),
    dict(nome="B96_n12_N8", n_bits=96, tamanho_tupla=12, n_ram=8, n_saidas=2),
    dict(nome="B128_n8_N16", n_bits=128, tamanho_tupla=8, n_ram=16, n_saidas=3),
]
ETA, M_TREINO, M_TESTE = 0.35, 400, 60


def _monta(cfg, seed=0):
    """Rede, dados e a ordem de visita — tudo determinístico e exportável."""
    rng = np.random.default_rng(seed)
    net = NTupleRegressor(
        n_bits=cfg["n_bits"], n_ram=cfg["n_ram"],
        tamanho_tupla=cfg["tamanho_tupla"], n_saidas=cfg["n_saidas"], seed=7,
    )
    bits = rng.random((M_TREINO + M_TESTE, cfg["n_bits"])) > 0.5
    alvo = rng.normal(size=(M_TREINO + M_TESTE, cfg["n_saidas"]))
    # A mesma ordem que `treinar` usaria internamente com este gerador.
    ordem = np.random.default_rng(3).permutation(M_TREINO)
    return net, bits, alvo, ordem


def exportar(destino: Path) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        net, bits, alvo, ordem = _monta(cfg)
        d = destino / cfg["nome"]
        d.mkdir(exist_ok=True)

        # Índice GLOBAL de bit lido por cada RAM, em ordem RAM-major: é o que o
        # Julia vai fatiar com `Iterators.partition(segments, n)`.
        np.savetxt(d / "mapeamento.txt", net.mapeamento.ravel(), fmt="%d")
        np.savetxt(d / "bits.txt", bits.astype(np.int8), fmt="%d")
        np.savetxt(d / "alvo.txt", alvo, fmt="%.17g")
        np.savetxt(d / "ordem.txt", ordem, fmt="%d")
        (d / "meta.txt").write_text(
            f"{cfg['n_bits']} {cfg['tamanho_tupla']} {cfg['n_ram']} "
            f"{cfg['n_saidas']} {M_TREINO} {M_TESTE} {ETA}\n"
        )

        net.treinar(bits[:M_TREINO], alvo[:M_TREINO], eta=ETA, epocas=1,
                    rng=np.random.default_rng(3))
        np.savetxt(d / "pred_python.txt", net.prever(bits[M_TREINO:]), fmt="%.17g")
        print(f"exportado {cfg['nome']}")
    return 0


def comparar(origem: Path) -> int:
    pior = 0.0
    print(f"{'config':>14} {'|Δ| máx':>12} {'|Δ| médio':>12} {'escala':>10}")
    for cfg in CONFIGS:
        d = origem / cfg["nome"]
        jl = d / "pred_julia.txt"
        if not jl.exists():
            print(f"{cfg['nome']:>14}   FALTA pred_julia.txt — rode o lado Julia")
            return 1
        py = np.loadtxt(d / "pred_python.txt", ndmin=2)
        ju = np.loadtxt(jl, ndmin=2)
        if py.shape != ju.shape:
            print(f"{cfg['nome']:>14}   FORMA {py.shape} != {ju.shape}")
            return 1
        dif = np.abs(py - ju)
        pior = max(pior, dif.max())
        print(f"{cfg['nome']:>14} {dif.max():>12.3e} {dif.mean():>12.3e} "
              f"{np.abs(py).max():>10.3f}")

    # Ambos em float64 sobre a mesma sequência de operações: o desvio aceitável
    # é o de reassociação em ponto flutuante, não o de "modelos parecidos".
    ok = pior < 1e-9
    print(f"\npior desvio: {pior:.3e}  ->  {'EQUIVALENTES' if ok else 'DIVERGEM'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exportar", type=Path)
    ap.add_argument("--comparar", type=Path)
    a = ap.parse_args()
    if a.exportar:
        return exportar(a.exportar)
    if a.comparar:
        return comparar(a.comparar)
    ap.error("use --exportar ou --comparar")


if __name__ == "__main__":
    sys.exit(main())
