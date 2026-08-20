"""Teste diferencial das duas redes da dissertação contra o código do Katopodis.

`spec04_diferencial.py` valida o `NTupleRegressor` denso. Este valida as redes
esparsas de `wnn/redes.py`, que são as fiéis ao capítulo 4:

- `RedeGradienteFuncional`  contra  `FunctionalDiscriminator`   (§4.3.1)
- `RedeRegressaoEsquecimento` contra `RegressionDiscriminator`  (§4.2.1, com φ)

A segunda é a que mais precisava de oráculo: a média ponderada da eq. (4.9) com
o denominador de série geométrica da (4.10) tem várias formas plausíveis de
errar — somar contadores em vez de pesos, aplicar φ na ordem trocada, esquecer o
ramo `φ = 1` — e todas produzem números que *parecem* certos.

Aqui o mapeamento é o **mesmo particionamento** dos dois lados
(`randperm` fatiado), então não é preciso sobrescrever `segments`/`offsets`
como no outro script: basta exportar a partição e reconstruí-la no Julia.

Uso:
    uv run python scripts/spec04_diferencial_redes.py --exportar <dir>
    <julia> scripts/diferencial_redes.jl <dir>
    uv run python scripts/spec04_diferencial_redes.py --comparar <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uavwnn.wnn import RedeGradienteFuncional, RedeRegressaoEsquecimento

# B = N*n mantém o particionamento exato (sem tupla curta no fim), o que deixa
# a reconstrução do lado Julia trivial. A cauda curta é exercitada pelos testes
# unitários, não aqui.
CONFIGS = [
    dict(nome="ator_B64_n8", papel="ator", n_bits=64, tupla=8, saidas=2, phi=None),
    dict(nome="ator_B96_n12", papel="ator", n_bits=96, tupla=12, saidas=3, phi=None),
    dict(nome="critico_B64_n8_phi085", papel="critico", n_bits=64, tupla=8,
         saidas=1, phi=0.85),
    dict(nome="critico_B96_n12_phi100", papel="critico", n_bits=96, tupla=12,
         saidas=1, phi=1.0),
    dict(nome="critico_B64_n8_phi050", papel="critico", n_bits=64, tupla=8,
         saidas=1, phi=0.5),
]
M_TREINO, M_TESTE, ETA = 400, 50, 0.35


def _monta(cfg, seed=0):
    rng = np.random.default_rng(seed)
    bits = rng.random((M_TREINO + M_TESTE, cfg["n_bits"])) > 0.5
    alvo = rng.normal(size=(M_TREINO + M_TESTE, cfg["saidas"]))
    if cfg["papel"] == "ator":
        rede = RedeGradienteFuncional(cfg["n_bits"], cfg["tupla"],
                                      n_saidas=cfg["saidas"], seed=7)
    else:
        rede = RedeRegressaoEsquecimento(cfg["n_bits"], cfg["tupla"],
                                         n_saidas=cfg["saidas"],
                                         phi=cfg["phi"], seed=7)
    return rede, bits, alvo


def _treina(cfg, rede, bits, alvo):
    """Sequência de treino idêntica à que o lado Julia vai executar.

    Para o ator usamos a atualização crua da eq. (4.11) com um passo arbitrário
    — e não a regressão supervisionada — porque é assim que o Algoritmo 4.4 o
    usa, e porque um passo que não depende da predição elimina o acoplamento
    entre as duas implementações no teste.
    """
    for m in range(M_TREINO):
        x = bits[m:m + 1]
        if cfg["papel"] == "ator":
            f = rede.prever(x)[0]
            rede.atualizar(x, -ETA * (f - alvo[m]))
        else:
            rede.treinar(x, alvo[m])


def exportar(destino: Path) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        d = destino / cfg["nome"]
        d.mkdir(exist_ok=True)
        rede, bits, alvo = _monta(cfg)

        np.savetxt(d / "mapeamento.txt",
                   np.concatenate(rede.mapeamento), fmt="%d")
        np.savetxt(d / "bits.txt", bits.astype(np.int8), fmt="%d")
        np.savetxt(d / "alvo.txt", alvo, fmt="%.17g")
        (d / "meta.txt").write_text(
            f"{cfg['n_bits']} {cfg['tupla']} {rede.n_ram} {cfg['saidas']} "
            f"{M_TREINO} {M_TESTE} {ETA} "
            f"{'ator' if cfg['papel'] == 'ator' else 'critico'} "
            f"{cfg['phi'] if cfg['phi'] is not None else 1.0}\n"
        )
        _treina(cfg, rede, bits, alvo)
        np.savetxt(d / "pred_python.txt", rede.prever(bits[M_TREINO:]), fmt="%.17g")
        print(f"exportado {cfg['nome']}  (ocupadas={rede.entradas_ocupadas()})")
    return 0


def comparar(origem: Path) -> int:
    pior = 0.0
    print(f"{'config':>24} {'|Δ| máx':>12} {'|Δ| médio':>12} {'escala':>9}")
    for cfg in CONFIGS:
        d = origem / cfg["nome"]
        jl = d / "pred_julia.txt"
        if not jl.exists():
            print(f"{cfg['nome']:>24}   FALTA pred_julia.txt")
            return 1
        py = np.loadtxt(d / "pred_python.txt", ndmin=2)
        ju = np.loadtxt(jl, ndmin=2)
        if py.shape != ju.shape:
            print(f"{cfg['nome']:>24}   FORMA {py.shape} != {ju.shape}")
            return 1
        dif = np.abs(py - ju)
        pior = max(pior, dif.max())
        print(f"{cfg['nome']:>24} {dif.max():>12.3e} {dif.mean():>12.3e} "
              f"{np.abs(py).max():>9.3f}")
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
