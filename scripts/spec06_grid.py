"""Varredura de hiperparâmetros em `Na = 1`, sem ruído — e a escolha do modelo.

Antes de comparar, treinar. Esta varredura roda os **três** aproximadores na célula
base da Etapa 1 e escolhe o melhor de cada um, para que a comparação de H1 seja entre
modelos **sintonizados** e não entre um sintonizado e dois espantalhos.

    uv run python scripts/spec06_grid.py --processos 12 --b-sweep 500000

A REGRA DE JUSTIÇA, E ELA TEM CONSEQUÊNCIA ARITMÉTICA

A T-06.3 é explícita: *"um pareamento em que a WNN foi varrida em 15 pontos e a MLP em
2 não é pareamento; é vantagem de esforço disfarçada de vantagem de método. Regra: os
dois lados recebem o mesmo número de pontos de varredura e o mesmo `B_max`, e o
orçamento gasto por lado é reportado."*

Então: **6 pontos por modelo**, os mesmos dois eixos (`α`, `σ`), o mesmo `B_sweep`, as
mesmas sementes, o mesmo banco. O `φ` do crítico com esquecimento **não** entra na
grade — ele existe só num dos três, e incluí-lo daria 12 pontos a ele contra 6 aos
outros. Fica como **sensibilidade no vencedor**, medida e não usada para selecionar.

AS FAIXAS DE `α` SÃO POR FAMÍLIA, E ISSO É MAIS JUSTO, NÃO MENOS

Uma faixa comum de `α` handicaparia dois dos três, por três razões medidas:

1. **A MLP usa Adam**, cuja taxa nominal não é o deslocamento. A SPEC-03 mediu o ótimo
   dela em ~1e-4 e que a curva é **não monotônica com ótimo agudo** — um palpite 30×
   acima fez a MLP parecer 5× pior do que é.
2. **O `anbubus` está na convenção de SOMA.** Somar `passo` às `N` posições muda `f` em
   `passo·N`, então a mesma taxa nominal dá um passo `N` vezes maior. Com `N = 67`, os
   `2e-3` da nossa convenção de média equivalem a **3,0e-5** na dele. A grade usa os
   equivalentes, de modo que o **passo efetivo em `f`** seja o mesmo nos dois — é o que
   faz a varredura medir o aproximador em vez de medir a convenção.
3. Cada faixa cobre a mesma **razão** (16×) em torno do valor de referência da família.

SELEÇÃO POR COBERTURA, NÃO POR `CT_cens` — e a razão foi medida

`CT_cens` satura em 500 s enquanto nenhum episódio conclui, e `SR` fica em 0 % até o
primeiro sucesso — **a caminhada aleatória também dá `SR = 0 %`**. Selecionar por
métrica achatada é selecionar ao acaso.

A `cobertura` (fração média de sensores coletados) tem sinal desde o primeiro passo e é
monótona em relação às duas. Ela **seleciona**; `CT_cens` e `SR` são **reportados**.
Quando um modelo passa a concluir episódios (`cobertura → 1`), `CT_cens` volta a ter
sinal e a seleção deve migrar para ele — a condição está registrada aqui e verificada
na saída.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_treinar import (  # noqa: E402
    BITS_POR_ESCALAR,
    REPLICAS,
    _cfgs,
    _ctx,
    avaliar,
    corpus_de_observacoes,
    rodar_episodios_de_treino,
)
from uavwnn.agents import APROXIMADORES, BASELINES  # noqa: E402
from uavwnn.agents.ac_continuo import AtorCriticoContinuo  # noqa: E402
from uavwnn.agents.base import Observacao  # noqa: E402
from uavwnn.agents.politica_continua import NormalProjetada  # noqa: E402
from uavwnn.campanha import RECOMPENSAS, Corrida, PapeisDeSemente, carregar_bancos  # noqa: E402
from uavwnn.campanha.treino import ParadaAntecipada  # noqa: E402
from uavwnn.encoding import TermometroQuantil  # noqa: E402
from uavwnn.refenv import run_episode  # noqa: E402

TAMANHO_TUPLA = 12
SIGMAS = (0.10, 0.20)

TAXAS = {
    # convenção de MÉDIA (eq. 4.12): o passo em `f` é exatamente `α`
    "ntupla": (5e-4, 2e-3, 8e-3),
    # convenção de SOMA: o passo em `f` é `α·N`. Com N = 67 estes são os
    # equivalentes exatos dos de cima — ver `taxa_equivalente`.
    "ntupla_funcional": (7.5e-6, 3.0e-5, 1.2e-4),
    # Adam: a taxa nominal não é o deslocamento. Faixa em torno do ótimo que a
    # SPEC-03 mediu (~1e-4), cobrindo a mesma razão de 16×.
    "mlp": (1e-4, 4e-4, 1.6e-3),
}
"""Mesma **razão** (16×) em cada família, em torno do valor de referência dela."""


def _montar(aprox: str, cfg, sem: PapeisDeSemente, corpus, taxa: float, sigma: float):
    d_obs = cfg.observation_size
    if aprox == "mlp":
        ap = APROXIMADORES[aprox](n_bits_ou_dim=d_obs, seed=sem.seed_train,
                                  taxa_ator=taxa, taxa_critico=taxa)
        codificar = lambda v: np.asarray(v, float).ravel()          # noqa: E731
        termo = None
    else:
        termo = TermometroQuantil(BITS_POR_ESCALAR).ajustar(corpus)
        kw = dict(n_bits=termo.n_bits, seed=sem.seed_tuple,
                  tamanho_tupla=TAMANHO_TUPLA, taxa_ator=taxa,
                  n_dims_codificador=d_obs, limiares_por_dim=BITS_POR_ESCALAR)
        if aprox == "ntupla_funcional":
            kw["taxa_critico"] = taxa
        ap = APROXIMADORES[aprox](**kw)
        codificar = lambda v: termo.codificar(np.asarray(v, float)[None, :])[0]  # noqa: E731
    pol = AtorCriticoContinuo(aproximador=ap, codificar=codificar,
                              politica=NormalProjetada(sigma, None),
                              id=f"grid_{aprox}", treinando=True)
    return ap, termo, pol


def uma_corrida(args: tuple) -> dict:
    """Um ponto da grade, uma semente. Isolada para rodar em processo próprio."""
    aprox, taxa, sigma, seed_train, replica, b_sweep, n_val, n_test, raiz = args
    r = REPLICAS[replica]
    na, ns = 1, 12
    cfg_tr, cfg_av = _cfgs(replica, na, ns)
    bancos = carregar_bancos(na, ns)
    val = bancos.val.sementes[:n_val]
    test = bancos.test.sementes[:n_test]
    ctx_tr, ctx_av = _ctx(cfg_tr), _ctx(cfg_av)
    corpus = corpus_de_observacoes(cfg_tr)
    sem = PapeisDeSemente(seed_train=seed_train)

    ap, termo, pol = _montar(aprox, cfg_tr, sem, corpus, taxa, sigma)
    cfg_corrida = {"grade": True, "replica": replica, "aproximador": aprox,
                   "na": na, "ns": ns, "taxa": taxa, "sigma": sigma,
                   "tamanho_tupla": TAMANHO_TUPLA, "b_sweep": b_sweep,
                   "recompensa": r["recompensa"], "gamma": 0.99}
    corrida = Corrida(experimento=f"spec06-grid-{replica}", config=cfg_corrida,
                      sementes=sem, bancos=bancos, raiz=Path(raiz))
    corrida.preparar()

    rng_cen = sem.rng_cenarios_treino()
    recompensa = RECOMPENSAS[r["recompensa"]]
    intervalo = max(1, b_sweep // 20)
    parada = ParadaAntecipada(maior_e_melhor=True)
    melhor_val, passos = -np.inf, 0
    t0 = time.perf_counter()
    abortou = None

    try:
        while passos < b_sweep:
            info = rodar_episodios_de_treino(
                cfg_tr, pol, ctx_tr, recompensa, rng_cen,
                min(intervalo, b_sweep - passos), r["escala_recompensa"])
            passos += int(info["passos"])
            m = avaliar(cfg_av, pol, ctx_av, val)
            melhor_val = max(melhor_val, m["cobertura"])
            if parada.registrar(m["cobertura"]):
                break
    except Exception as e:
        # A asserção de estabilidade da §4.2 é BUG, não resultado (§4.3) — mas num
        # ponto de grade ela é informação sobre o ponto: aquele `α` é instável.
        # Registrar e seguir é o que permite ver a fronteira de estabilidade em vez
        # de perder a grade inteira.
        abortou = f"{type(e).__name__}: {e}"

    final = avaliar(cfg_av, pol, ctx_av, test)
    corrida.gravar_eval_test(final["por_episodio"])
    return {"aproximador": aprox, "taxa": taxa, "sigma": sigma,
            "seed_train": seed_train, "passos": passos,
            "val_cobertura": float(melhor_val) if np.isfinite(melhor_val) else 0.0,
            "cobertura": final["cobertura"], "sr": final["sr"],
            "ct_cens": final["ct_cens"], "ct_succ": final["ct_succ"],
            "abortou": abortou, "parede_s": time.perf_counter() - t0,
            "dir": str(corrida.dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replica", choices=sorted(REPLICAS), default="1b",
                    help="1b é O PROBLEMA (eq. 5-6, CT_cens); 1a é o exploratório")
    ap.add_argument("--modelos", default="ntupla,ntupla_funcional,mlp")
    ap.add_argument("--sementes", type=int, default=3, help="`R` da §4.3")
    ap.add_argument("--b-sweep", type=int, default=500_000)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--processos", type=int, default=12)
    ap.add_argument("--raiz", default="runs")
    ap.add_argument("--saida", default="reports/spec06-grid.json")
    a = ap.parse_args()

    modelos = a.modelos.split(",")
    tarefas = [(m, t, s, sd, a.replica, a.b_sweep, a.n_val, a.n_test, a.raiz)
               for m in modelos for t in TAXAS[m] for s in SIGMAS
               for sd in range(a.sementes)]

    print(f"varredura · réplica {a.replica} · Na=1 Ns=12 · sem ruído")
    print(f"{len(modelos)} modelos × {len(TAXAS[modelos[0]])} taxas × "
          f"{len(SIGMAS)} sigmas × {a.sementes} sementes = {len(tarefas)} corridas")
    print(f"B_sweep = {a.b_sweep:,} passos · {a.processos} processos")
    print("pontos por modelo: "
          f"{len(TAXAS[modelos[0]]) * len(SIGMAS)} (pareado, T-06.3)\n")

    t0 = time.perf_counter()
    with mp.Pool(a.processos) as pool:
        res = pool.map(uma_corrida, tarefas)
    parede = time.perf_counter() - t0

    print(f"\n{'modelo':>18} {'taxa':>9} {'σ':>5} {'cob val':>8} {'cob test':>9} "
          f"{'SR':>6} {'CT_cens':>8} {'abortos':>8}")
    melhores = {}
    for m in modelos:
        for t in TAXAS[m]:
            for s in SIGMAS:
                g = [x for x in res if x["aproximador"] == m and x["taxa"] == t
                     and x["sigma"] == s]
                cv = float(np.mean([x["val_cobertura"] for x in g]))
                ct = float(np.mean([x["cobertura"] for x in g]))
                sr = float(np.mean([x["sr"] for x in g]))
                cc = float(np.mean([x["ct_cens"] for x in g]))
                ab = sum(x["abortou"] is not None for x in g)
                print(f"{m:>18} {t:>9.1e} {s:>5.2f} {cv:>8.1%} {ct:>9.1%} "
                      f"{sr:>5.1%} {cc:>8.1f} {ab:>4}/{len(g)}")
                if m not in melhores or cv > melhores[m]["val_cobertura"]:
                    melhores[m] = {"taxa": t, "sigma": s, "val_cobertura": cv,
                                   "cobertura": ct, "sr": sr, "ct_cens": cc,
                                   "abortos": ab, "n": len(g)}

    print(f"\n{'MELHOR DE CADA (por cobertura no val)':>44}")
    print(f"{'modelo':>18} {'taxa':>9} {'σ':>5} {'cob val':>8} {'cob test':>9} "
          f"{'SR':>6} {'CT_cens':>8}")
    for m, d in melhores.items():
        print(f"{m:>18} {d['taxa']:>9.1e} {d['sigma']:>5.2f} "
              f"{d['val_cobertura']:>8.1%} {d['cobertura']:>9.1%} "
              f"{d['sr']:>5.1%} {d['ct_cens']:>8.1f}")

    campeao = max(melhores.items(), key=lambda kv: kv[1]["val_cobertura"])
    print(f"\nvencedor: {campeao[0]} (taxa {campeao[1]['taxa']:.1e}, "
          f"σ {campeao[1]['sigma']:.2f})")
    if campeao[1]["cobertura"] < 0.99:
        print("  RESSALVA: cobertura < 99 %, então `CT_cens` ainda está saturado e")
        print("  a seleção por ele seria ao acaso. Quando um modelo passar a concluir,")
        print("  a seleção deve migrar para `CT_cens` — é a condição registrada.")

    art = {"config": vars(a), "taxas": {k: list(v) for k, v in TAXAS.items()},
           "sigmas": list(SIGMAS), "tamanho_tupla": TAMANHO_TUPLA,
           "parede_s": parede, "corridas": res, "melhores": melhores,
           "vencedor": campeao[0]}
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nparede: {parede / 60:.1f} min · artefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
