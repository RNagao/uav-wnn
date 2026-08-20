"""Replicação de §5.2.2 da dissertação: codificação circular contra termômetro.

    "For both tasks, the circular thermometer reaped better results than the
     simple one, with the difference being more pronounced with lunar lander."
    — §5.2.2

Hiperparâmetros como o quadro de §5.2.3 lista para CartPole: `m = 500`,
resolução 128, tupla 64, `γ = 1,0`, `ε = 0,005`, esquecimento 0,85. Só o
codificador varia — é o desenho do experimento do texto.

Duas diferenças em relação ao original, ambas por orçamento e declaradas:

- **Menos episódios e menos réplicas.** O artigo usa 50 réplicas e 1 M passos.
  Aqui o padrão é 300 episódios e 8 sementes, o que dá para ver a diferença de
  regime mas não reproduz as curvas finais.
- **Recorte da observação em `[l, u]`.** A eq. (3.24) pressupõe `x` no intervalo
  onde a variável é definida; fora dele o circular **dá a volta** e apelida
  valores extremos em valores centrais. Sem recorte a comparação mediria essa
  patologia em vez do esquema. O termômetro satura em vez de dar a volta, então
  o recorte o afeta menos — o que já é assimetria a favor dele, não contra.

Uso:
    uv run python scripts/replica_katopodis_52.py
    uv run python scripts/replica_katopodis_52.py --episodios 600 --sementes 12
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from uavwnn.encoding import CircularArtigo, TermometroArtigo
from uavwnn.rl import SarsaNTupla
from uavwnn.wnn import RedeRegressaoEsquecimento

# Quadro de §5.2.3 (a), CartPole.
RESOLUCAO, TUPLA, PASSOS = 128, 64, 500
DESCONTO, EXPLORACAO, ESQUECIMENTO = 1.0, 0.005, 0.85
FAIXA = (-1.5, 1.5)


def _codificador(esquema: str, resolucao: int):
    cls = CircularArtigo if esquema == "circular" else TermometroArtigo
    enc = cls(FAIXA[0], FAIXA[1], resolucao)
    limite = FAIXA[1] - 1e-6

    def codificar(obs):
        return enc.codificar(np.clip(obs, FAIXA[0], limite)[None, :])

    return codificar, enc.n_bits_para(4)


def roda(esquema: str, episodios: int, semente: int, resolucao: int = RESOLUCAO,
         tupla: int = TUPLA, passos: int = PASSOS,
         exploracao: float = EXPLORACAO) -> list[float]:
    import gymnasium as gym

    codificar, n_bits = _codificador(esquema, resolucao)
    rede = RedeRegressaoEsquecimento(n_bits, tupla, n_saidas=2,
                                     phi=ESQUECIMENTO, seed=semente)
    agente = SarsaNTupla(rede=rede, codificar=codificar, n_acoes=2,
                         passos=passos, desconto=DESCONTO,
                         exploracao=exploracao,
                         rng=np.random.default_rng(semente))
    env = gym.make("CartPole-v1")
    env.reset(seed=semente)
    retornos = [agente.episodio(env, max_passos=passos) for _ in range(episodios)]
    env.close()
    return retornos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=300)
    ap.add_argument("--sementes", type=int, default=8)
    ap.add_argument("--resolucao", type=int, default=RESOLUCAO)
    ap.add_argument("--tupla", type=int, default=TUPLA)
    ap.add_argument("--passos", type=int, default=PASSOS)
    ap.add_argument("--exploracao", type=float, default=EXPLORACAO)
    ap.add_argument("--saida", default="reports/replica-katopodis-52.json")
    a = ap.parse_args()

    print(f"CartPole-v1 · R={a.resolucao} n={a.tupla} m={a.passos} "
          f"γ={DESCONTO} ε={a.exploracao} φ={ESQUECIMENTO}")
    print(f"{a.episodios} episódios × {a.sementes} sementes\n")

    resultado = {}
    for esquema in ("simples", "circular"):
        curvas = np.array([roda(esquema, a.episodios, s, a.resolucao, a.tupla,
                                a.passos, a.exploracao)
                           for s in range(a.sementes)])
        # média do último quinto, por semente: o "regime final" do texto
        final = curvas[:, -a.episodios // 5:].mean(axis=1)
        resultado[esquema] = dict(
            curva_media=curvas.mean(axis=0).tolist(),
            final_por_semente=final.tolist(),
            final_media=float(final.mean()),
            final_desvio=float(final.std(ddof=1)),
            area=float(curvas.mean()),
        )
        print(f"{esquema:>9}  final {final.mean():>7.1f} ± {final.std(ddof=1):>5.1f}"
              f"   área {curvas.mean():>7.1f}"
              f"   por semente {np.round(final, 0).tolist()}")

    c, s = resultado["circular"], resultado["simples"]
    ganho = c["final_media"] / s["final_media"]
    # bootstrap pareado por semente: as sementes são as mesmas nos dois braços
    dif = np.array(c["final_por_semente"]) - np.array(s["final_por_semente"])
    rng = np.random.default_rng(0)
    reamostras = np.array([rng.choice(dif, len(dif), replace=True).mean()
                           for _ in range(10_000)])
    lo, hi = np.percentile(reamostras, [2.5, 97.5])
    print(f"\ncircular / simples = {ganho:.2f}x")
    print(f"diferença pareada: {dif.mean():+.1f}  IC95% [{lo:+.1f}, {hi:+.1f}]")
    veredito = ("circular VENCE" if lo > 0 else
                "simples VENCE" if hi < 0 else "EMPATE (IC cruza zero)")
    print(f"veredito: {veredito}")

    resultado["comparacao"] = dict(razao=ganho, dif_media=float(dif.mean()),
                                   ic95=[float(lo), float(hi)], veredito=veredito)
    with open(a.saida, "w") as f:
        json.dump(resultado, f, indent=2)
    print(f"\nartefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
