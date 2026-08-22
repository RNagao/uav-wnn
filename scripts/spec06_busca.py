"""Busca aleatória pareada sobre **todos** os hiperparâmetros — `Na = 1`, sem ruído.

A grade fatorial completa não cabe, e isso é aritmética: `α(3) × σ(3) × T(3) × n(5) ×
crítico(3) × m(4)` = **540 pontos** × 3 sementes = 1 620 corridas ≈ 26 h. A própria
T-06.4 diz o mesmo — *"a grade completa não cabe, e isso é aritmética, não preguiça"*.

Ela prescreve **descida coordenada a coordenada**. Este script faz outra coisa e o
desvio é declarado: **busca aleatória**, sorteando os seis eixos em conjunto.

POR QUE ALEATÓRIA, E NÃO COORDENADA

1. **A coordenada não vê interação.** A T-06.4 admite isso (*"a limitação da busca
   coordenada fica declarada no relatório"*), e a primeira grade deu motivo para
   suspeitar: `σ` mudou o resultado por **20×** enquanto `α` mudou por 15× — dois
   eixos de efeito grande, que é o regime em que a coordenada mais erra.
2. **Cobre todos os eixos com qualquer orçamento.** Com poucos eixos importantes, a
   aleatória encontra melhores pontos que a grade para o mesmo número de corridas —
   é o resultado clássico de Bergstra e Bengio (2012), e a nossa situação é
   exatamente a dele: dos seis eixos, `σ` sozinho separou 3 % de 72 %.
3. **Ela alimenta a coordenada.** O melhor ponto daqui é o ponto de partida da
   descida da T-06.4, que fica mais curta por começar mais perto.

O PAREAMENTO É MAIS FORTE DO QUE NA GRADE

Os dois modelos recebem os **mesmos sorteios** de `(σ, T, n, m, passo_efetivo)` — não
apenas o mesmo número de pontos. Só duas coisas diferem, e as duas por necessidade:

    taxa        convertida para a convenção de cada família, POR AMOSTRA
    crítico     `φ` na dissertação; razão `taxa_crítico/taxa_ator` no anbubus

Isso torna a comparação **pareada por configuração**, e não só por orçamento: cada
ponto tem um par exato no outro modelo.

A CONVERSÃO DA TAXA TEM DE SER POR AMOSTRA — e é o eixo `n` que obriga

Somar `passo` às `N` posições endereçadas muda `f` em `passo` na convenção de média
(eq. 4.12) e em `passo·N` na de soma. Como `N = ⌈25·T/n⌉` e a busca varia `T` e `n`,
**`N` varia de 12 a 200** no espaço amostrado. Uma tabela fixa de taxas equivalentes
seria válida para um único `(T, n)`.

Então o que se sorteia é o **passo efetivo em `f`** — a grandeza que tem o mesmo
significado nas duas convenções — e cada família recebe a taxa que o produz:

    média:  α = passo_efetivo
    soma:   α = passo_efetivo / N

O CRÍTICO DE CADA FAMÍLIA, E POR QUE NÃO É O MESMO EIXO

O crítico da dissertação (§4.2.1) **acumula** retorno pela eq. (4.8) e não tem taxa —
o hiperparâmetro dele é `φ`, o fator de esquecimento. O do `anbubus` é gradiente
funcional treinado por descida, e o hiperparâmetro dele é a **taxa**. Não há como dar
o mesmo eixo aos dois; dar **um eixo de crítico a cada** é o pareamento honesto, e a
cardinalidade é igual.
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
from spec06_treinar import REPLICAS, _cfgs, _ctx, avaliar, corpus_de_observacoes, \
    rodar_episodios_de_treino  # noqa: E402
from uavwnn.agents import APROXIMADORES  # noqa: E402
from uavwnn.agents.ac_continuo import AtorCriticoContinuo  # noqa: E402
from uavwnn.agents.politica_continua import NormalProjetada  # noqa: E402
from uavwnn.campanha import RECOMPENSAS, Corrida, PapeisDeSemente, carregar_bancos  # noqa: E402
from uavwnn.campanha.treino import ParadaAntecipada  # noqa: E402
from uavwnn.encoding import TermometroQuantil  # noqa: E402
from uavwnn.refenv.runner import EnvConfig  # noqa: E402

# --- o espaço de busca ------------------------------------------------------

PASSO_EFETIVO = (5e-5, 4e-3)
"""Faixa log-uniforme do **passo em `f`**, não da taxa nominal.

Bracketa o vencedor da primeira grade (5e-4 na convenção de média, que É o passo
efetivo) com uma ordem de grandeza para cada lado. O limite superior fica em 4e-3
porque 8e-3 falhou em todas as sementes e 2e-3 falhou em `σ = 0,10`."""

SIGMA = (0.15, 0.60)
"""Uniforme. A primeira grade mediu um **degrau** entre 0,10 (3 %) e 0,20 (72 %), com
0,20 no teto da faixa — evidência de que o ótimo está acima. Sobe até 0,60."""

RESOLUCAO = (16, 32, 64)
"""`T` do termômetro. A T-06.4 pede `{8, 32, 128}`; 8 é grosseiro demais para 25
dimensões e 128 dá `B = 3200` bits, cujo custo por decisão não cabe no orçamento."""

TUPLA = (8, 12, 16, 24, 32)
"""`n`. Cobre a faixa do PRD (`{8,10,12}`) **e** a do Katopodis em tarefa contínua
(`{16,32,64}`), que a pergunta aberta 2 pedia para unir."""

PASSOS_M = (4, 8, 16, 32)
"""`m` do retorno de `m` passos. Faixa própria: o apêndice A não reporta `m` para as
tarefas contínuas."""

PHI = (0.70, 0.85, 0.99, 1.0)
"""Esquecimento do crítico da dissertação. `1,0` é o controle sem esquecimento."""

RAZAO_CRITICO = (0.25, 1.0, 4.0, 16.0)
"""`taxa_crítico / taxa_ator` no crítico do `anbubus`. Mesma cardinalidade que `PHI`
— é o eixo de crítico daquela família."""


def amostrar(n: int, seed: int = 20260921) -> list[dict]:
    """Sorteia `n` configurações, **compartilhadas** pelos dois modelos.

    Semente fixa: a busca precisa ser reproduzível como qualquer outra corrida, e um
    sorteio irreproduzível transformaria o vencedor em anedota.
    """
    g = np.random.default_rng(seed)
    fora = []
    for i in range(n):
        fora.append({
            "id": i,
            # log-uniforme: a escala de um passo de gradiente é multiplicativa
            "passo_efetivo": float(np.exp(g.uniform(*np.log(PASSO_EFETIVO)))),
            "sigma": float(g.uniform(*SIGMA)),
            "resolucao": int(g.choice(RESOLUCAO)),
            "tupla": int(g.choice(TUPLA)),
            "passos_m": int(g.choice(PASSOS_M)),
            # os índices do eixo de crítico são compartilhados; cada família traduz
            "idx_critico": int(g.integers(len(PHI))),
        })
    return fora


def _montar(aprox: str, cfg: EnvConfig, sem: PapeisDeSemente, amostra: dict):
    """Instancia o modelo, convertendo a taxa para a convenção da família."""
    d_obs = cfg.observation_size
    termo = TermometroQuantil(amostra["resolucao"]).ajustar(
        corpus_de_observacoes(cfg))
    comum = dict(n_bits=termo.n_bits, tamanho_tupla=amostra["tupla"],
                 seed=sem.seed_tuple, n_dims_codificador=d_obs,
                 limiares_por_dim=amostra["resolucao"])

    # `N` depende de (T, n), então a conversão é POR AMOSTRA — ver o cabeçalho.
    from uavwnn.wnn.redes import RedeGradienteFuncional
    n_ram = RedeGradienteFuncional(termo.n_bits, amostra["tupla"], seed=0).n_ram
    passo = amostra["passo_efetivo"]

    if aprox == "ntupla":
        ap = APROXIMADORES[aprox](taxa_ator=passo, phi=PHI[amostra["idx_critico"]],
                                  **comum)
        detalhe = {"phi": PHI[amostra["idx_critico"]], "taxa_ator": passo}
    else:
        taxa = passo / n_ram
        razao = RAZAO_CRITICO[amostra["idx_critico"]]
        ap = APROXIMADORES[aprox](taxa_ator=taxa, taxa_critico=taxa * razao, **comum)
        detalhe = {"razao_critico": razao, "taxa_ator": taxa}

    codificar = lambda v: termo.codificar(np.asarray(v, float)[None, :])[0]  # noqa: E731
    pol = AtorCriticoContinuo(aproximador=ap, codificar=codificar,
                              politica=NormalProjetada(amostra["sigma"], None),
                              passos=amostra["passos_m"], id=f"busca_{aprox}",
                              treinando=True)
    return ap, termo, pol, {**detalhe, "n_ram": n_ram, "n_bits": termo.n_bits}


def uma(args: tuple) -> dict:
    aprox, amostra, seed_train, replica, b, n_val, n_test, raiz = args
    r = REPLICAS[replica]
    cfg_tr, cfg_av = _cfgs(replica, 1, 12)
    bancos = carregar_bancos(1, 12)
    val, test = bancos.val.sementes[:n_val], bancos.test.sementes[:n_test]
    ctx_tr, ctx_av = _ctx(cfg_tr), _ctx(cfg_av)
    sem = PapeisDeSemente(seed_train=seed_train)

    try:
        ap, termo, pol, det = _montar(aprox, cfg_tr, sem, amostra)
    except Exception as e:
        return {"aproximador": aprox, **amostra, "seed_train": seed_train,
                "abortou": f"montagem: {type(e).__name__}: {e}", "cobertura": 0.0,
                "sr": 0.0, "ct_cens": float(cfg_av.max_episode_length),
                "val_cobertura": 0.0, "passos": 0, "parede_s": 0.0}

    cfg_corrida = {"busca": True, "replica": replica, "aproximador": aprox,
                   "na": 1, "ns": 12, "b_sweep": b, **amostra, **det}
    corrida = Corrida(experimento=f"spec06-busca-{replica}", config=cfg_corrida,
                      sementes=sem, bancos=bancos, raiz=Path(raiz))
    corrida.preparar()

    rng_cen = sem.rng_cenarios_treino()
    recompensa = RECOMPENSAS[r["recompensa"]]
    # `// 40` e não `// 20`: com 20 avaliações a guarda de MINIMO_AVALIACOES=20
    # tornava a parada antecipada impossível — ela só poderia disparar quando o
    # orçamento já tivesse acabado. Medido na primeira grade: nenhuma corrida parou.
    intervalo = max(1, b // 40)
    parada = ParadaAntecipada(maior_e_melhor=True)
    melhor, passos, abortou = 0.0, 0, None
    t0 = time.perf_counter()
    try:
        while passos < b:
            info = rodar_episodios_de_treino(
                cfg_tr, pol, ctx_tr, recompensa, rng_cen,
                min(intervalo, b - passos), r["escala_recompensa"])
            passos += int(info["passos"])
            m = avaliar(cfg_av, pol, ctx_av, val)
            melhor = max(melhor, m["cobertura"])
            if parada.registrar(m["cobertura"]):
                break
    except Exception as e:
        # Aborto é INFORMAÇÃO sobre o ponto (aquele passo é instável), não perda da
        # busca. É o que faz a fronteira de estabilidade aparecer no resultado.
        abortou = f"{type(e).__name__}: {e}"

    final = avaliar(cfg_av, pol, ctx_av, test)
    corrida.gravar_eval_test(final["por_episodio"])
    return {"aproximador": aprox, **amostra, **det, "seed_train": seed_train,
            "passos": passos, "val_cobertura": float(melhor),
            "cobertura": final["cobertura"], "sr": final["sr"],
            "ct_cens": final["ct_cens"], "ct_succ": final["ct_succ"],
            "abortou": abortou, "parede_s": time.perf_counter() - t0,
            "dir": str(corrida.dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replica", choices=sorted(REPLICAS), default="1b")
    ap.add_argument("--modelos", default="ntupla,ntupla_funcional")
    ap.add_argument("--amostras", type=int, default=24,
                    help="configurações sorteadas, COMPARTILHADAS pelos modelos")
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--b-sweep", type=int, default=400_000)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--processos", type=int, default=12)
    ap.add_argument("--seed-busca", type=int, default=20260921)
    ap.add_argument("--raiz", default="runs")
    ap.add_argument("--saida", default="reports/spec06-busca.json")
    a = ap.parse_args()

    modelos = a.modelos.split(",")
    amostras = amostrar(a.amostras, a.seed_busca)
    tarefas = [(m, s, sd, a.replica, a.b_sweep, a.n_val, a.n_test, a.raiz)
               for m in modelos for s in amostras for sd in range(a.sementes)]

    print(f"busca aleatória pareada · réplica {a.replica} · Na=1 Ns=12 · sem ruído")
    print(f"6 eixos · {a.amostras} sorteios COMPARTILHADOS · {len(modelos)} modelos "
          f"× {a.sementes} sementes = {len(tarefas)} corridas")
    print(f"B_sweep = {a.b_sweep:,} · {a.processos} processos\n")
    print("espaço: passo_efetivo log-U(5e-5, 4e-3) · σ U(0,15, 0,60) · "
          f"T {RESOLUCAO} · n {TUPLA} · m {PASSOS_M}")
    print(f"        crítico: φ {PHI} (dissertação) | razão {RAZAO_CRITICO} (anbubus)\n")

    t0 = time.perf_counter()
    with mp.Pool(a.processos) as pool:
        # `chunksize=1` de propósito. O padrão fatia a lista em blocos contíguos, e
        # a lista está ordenada por modelo — o que faria a `ntupla` rodar inteira
        # antes de a `ntupla_funcional` começar. O total seria o mesmo, mas por uma
        # hora só um modelo teria resultado, e nenhuma leitura parcial seria
        # comparativa. Com blocos de 1 os dois avançam juntos.
        res = pool.map(uma, tarefas, chunksize=1)
    parede = time.perf_counter() - t0

    import collections
    por = collections.defaultdict(list)
    for x in res:
        por[(x["aproximador"], x["id"])].append(x)

    print(f"{'modelo':>18} {'id':>3} {'passo':>8} {'σ':>5} {'T':>4} {'n':>3} "
          f"{'m':>3} {'crít':>6} {'N':>4} {'cob':>7} {'SR':>6} {'ab':>4}")
    linhas = []
    for (m, i), g in sorted(por.items(),
                            key=lambda kv: -float(np.mean([x["cobertura"]
                                                           for x in kv[1]]))):
        a0 = g[0]
        cob = float(np.mean([x["cobertura"] for x in g]))
        sr = float(np.mean([x["sr"] for x in g]))
        ab = sum(x["abortou"] is not None for x in g)
        crit = a0.get("phi", a0.get("razao_critico"))
        linhas.append({"modelo": m, "id": i, "cobertura": cob, "sr": sr,
                       "abortos": ab, "n": len(g), **{k: a0[k] for k in
                       ("passo_efetivo", "sigma", "resolucao", "tupla", "passos_m")},
                       "critico": crit, "n_ram": a0.get("n_ram")})
        print(f"{m:>18} {i:>3} {a0['passo_efetivo']:>8.1e} {a0['sigma']:>5.2f} "
              f"{a0['resolucao']:>4} {a0['tupla']:>3} {a0['passos_m']:>3} "
              f"{crit:>6} {a0.get('n_ram', 0):>4} {cob:>6.1%} {sr:>5.1%} "
              f"{ab:>2}/{len(g)}")

    print(f"\n{'MELHOR POR MODELO':>40}")
    for m in modelos:
        cand = [l for l in linhas if l["modelo"] == m]
        if not cand:
            continue
        b = max(cand, key=lambda l: l["cobertura"])
        print(f"  {m}: passo {b['passo_efetivo']:.2e} · σ {b['sigma']:.3f} · "
              f"T {b['resolucao']} · n {b['tupla']} · m {b['passos_m']} · "
              f"crítico {b['critico']} · N {b['n_ram']}")
        print(f"      -> cobertura {b['cobertura']:.1%}, SR {b['sr']:.1%}, "
              f"abortos {b['abortos']}/{b['n']}")

    art = {"config": vars(a), "espaco": {
        "passo_efetivo": list(PASSO_EFETIVO), "sigma": list(SIGMA),
        "resolucao": list(RESOLUCAO), "tupla": list(TUPLA),
        "passos_m": list(PASSOS_M), "phi": list(PHI),
        "razao_critico": list(RAZAO_CRITICO)},
        "amostras": amostras, "pontos": linhas, "corridas": res,
        "parede_s": parede}
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nparede: {parede/60:.1f} min · artefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
