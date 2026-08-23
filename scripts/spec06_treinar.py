"""T-06.1 a T-06.5 — o treinador da campanha, um laço para todas as réplicas.

O que muda entre as tarefas é **configuração**, não código: ramificar por tarefa
faria cada réplica ter o seu laço, e a comparação entre elas mediria os laços.

    uv run python scripts/spec06_treinar.py --replica 1a --celulas 1:12 --sementes 5

RÉPLICAS

    1a   §7.1 exploratório: `N*_a = 0` (cegos aos demais), recompensa **eq. 7-2**,
         horizonte permissivo (P5: 500 s, medido), métrica primária **SR**,
         alvo publicado **100 %**
    1b   §7.2 tempo de conclusão: observação completa (eq. 5-1), recompensa
         **eq. 5-6**, `max_episode_length = 500 s`, métrica primária **CT_cens**

A eq. 7-2 é calculada **fora** do ambiente (`campanha.recompensas`), porque o
ambiente não a implementa. Não é modificar o problema: a recompensa de aprendizado é
escolha do agente, e o ambiente expõe `sensors_collected`.

O QUE ESTE SCRIPT NÃO FAZ, DE PROPÓSITO

Não toca no banco `test` durante o treino. A seleção de checkpoint usa `val`, e o
`test` entra uma vez, no fim, com o checkpoint já escolhido. Lamenza avalia durante o
treino e reporta o modelo carregado dessas avaliações (§7, p36) — seleção sobre o
conjunto de reporte. Separamos e declaramos que separamos.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import multiprocessing as mp  # noqa: E402
from uavwnn.agents import APROXIMADORES, BASELINES  # noqa: E402
from uavwnn.agents.politica_continua import NormalProjetada  # noqa: E402
from uavwnn.agents.ac_continuo import AtorCriticoContinuo  # noqa: E402
from uavwnn.agents.base import ContextoEpisodio, Observacao  # noqa: E402
from uavwnn.campanha import (  # noqa: E402
    RECOMPENSAS,
    Corrida,
    PapeisDeSemente,
    carregar_bancos,
    treinar,
)
from uavwnn.encoding import TermometroQuantil  # noqa: E402
from uavwnn.refenv import LAMENZA_003, make_env, run_episode  # noqa: E402
from uavwnn.refenv.runner import EnvConfig  # noqa: E402

GAMMA = 0.99
"""De Lamenza (tab. 7.1). Fixo, não varrido — critério de saída da SPEC-05."""

ESCALA_RECOMPENSA = 1.0 - GAMMA
"""Normalização do alvo de valor por `(1−γ)`, **prescrita pela T-06.1** e registrada
aqui antes da campanha, como ela exige.

O RISCO QUE ELA RESOLVE, E ELE FOI MEDIDO

A T-06.1 previu: *"a recompensa 7-2 é não negativa e monotonicamente não decrescente
dentro do episódio. Com `γ = 0,99` e horizonte permissivo, `V` se aproxima de
`≈ 100 × fração_visitada` … os alvos de TD são grandes desde o começo."*

Medido: sem a normalização, a asserção de estabilidade da SPEC-05 §4.2 aborta no
**passo 998** com `passo 0,6742 excede c_max=0,5`. A asserção funcionou como
projetada — a §4.3 classifica isso como **bug, não resultado**: *"corrigir e
re-executar a configuração inteira"*.

**Não é alteração do MDP**, e a T-06.1 já diz por quê: um fator escalar positivo na
recompensa não muda a política ótima, então isto é hiperparâmetro do aproximador e
**não** entra na tabela de desvios do PRD §5.4. Escalar a recompensa e não só o alvo
do crítico é o que mantém `δ` e o passo do ator na mesma escala — normalizar um lado
só desequilibraria os dois.

Com `γ = 0,99`, `1/(1−γ) = 100`: é exatamente o fator que a previsão da T-06.1
antecipou."""

# MÉTRICA DE SELEÇÃO ≠ MÉTRICA REPORTADA, e a razão foi medida.
#
# A §4.4 usa a métrica de validação para escolher checkpoint e disparar a parada; a
# T-06.1 reporta `SR`. Usar `SR` para as duas coisas não funciona nesta escala:
# medido, a **caminhada aleatória também dá SR = 0,0 %**, e `CT_cens` satura em 500 s
# para todo mundo. Com as duas achatadas, a seleção de checkpoint seleciona ao acaso,
# e o critério de falha catastrófica ("não supera a aleatória com IC separado")
# dispara para qualquer corrida que ainda não aprendeu.
#
# A `cobertura` — fração média de sensores coletados ao fim do episódio — tem sinal
# desde o primeiro passo, é **o próprio objetivo** da eq. 7-2, e é monótona em relação
# a `SR` (cobertura = 1 ⟺ sucesso). Selecionar por ela e reportar `SR` não muda nada
# do que é publicado: a §4.4 escolhe a métrica de validação, e nada obriga que ela
# seja a de reporte.
REPLICAS = {
    "1a": {"cegos": True, "recompensa": "eq_7_2", "horizonte": 500.0,
           "metrica": "cobertura", "reportada": "sr",
           "maior_e_melhor": True, "alvo_publicado": 1.0,
           "escala_recompensa": ESCALA_RECOMPENSA},
    "1b": {"cegos": False, "recompensa": "eq_5_6", "horizonte": 500.0,
           "metrica": "ct_cens", "reportada": "ct_cens",
           "maior_e_melhor": False, "alvo_publicado": None,
           # A eq. 5-6 já é O(1) fora das coletas e dá +10 nelas; o problema de
           # escala é específico da 7-2 acumulada sob horizonte permissivo.
           "escala_recompensa": 1.0},
}
BITS_POR_ESCALAR = 32

# --- configuração provisória vencedora da varredura ------------------------
#
# Escolhida sobre a PARCIAL do nível `desempate` (160 de 720 corridas), para
# destravar desenvolvimento em paralelo. **Provisória**: a grade não fechou, e o
# portão G1 da SPEC-06 segue não declarado.
#
# O motivo de existirem estas constantes é que os padrões não serviam, e um deles
# era uma armadilha: a `NormalProjetada` tem `sigma = 0,1` por padrão, e a
# primeira grade mediu que **as 18 corridas com σ = 0,10 abortaram**, todas, pela
# assertiva de estabilidade. Não é exploração insuficiente — é que o passo efetivo
# é `~α·δ/σ`, com σ no DENOMINADOR, então metade do σ dobra o passo. Treinar com o
# padrão daria divergência garantida.
#
# A escolha, eixo por eixo, sobre a parcial:
#
#   passo 5e-4  O maior passo amplamente estável. Medido na dissertação:
#               3e-4 -> 0/16 abortos e 14,0 % de cobertura das vivas
#               5e-4 -> 2/19 abortos e 26,1 %      <- escolhido
#               8e-4 -> 9/15 abortos e 37,7 %      (melhor, e 60 % de mortalidade)
#               1,5e-3 e 3e-3 -> 15/15 abortos, os dois
#   σ 0,20      O ponto de 3 sementes com melhor cobertura usa 0,20 (49,3 %).
#   T 32, n 12  `n` domina: com n=12 a cobertura é 31,7 % (T16) e 35,7 % (T32);
#               com n=24 cai para 4,9 % e 10,7 %. T=32 custa 1,26× de T=16.
#   φ 0,85      Melhor cobertura das vivas (27,8 % contra 22,4 % de 0,70 e
#               15,4 % de 0,99). O esquecimento NÃO estabiliza: 0,70 aborta 11/21.
#   m 8         Não varrido — limitação declarada. É o padrão da classe.
#   normalizar  Desligado: medido, PIORA os dois braços.
#
# RESSALVA: variância entre sementes é alta neste ponto (19,0 %, 54,8 %, 74,1 %).
# Uma única semente não é sinal.
PASSO_EFETIVO = 5e-4
"""Passo em `f` por atualização. Na convenção de MÉDIA (eq. 4.12) é a taxa direta."""

SIGMA = 0.20
"""Desvio da normal projetada. **Não** use o padrão 0,1 da classe — diverge."""

TAMANHO_TUPLA = 12
"""`n` da varredura, e precisa ser EXPLÍCITO porque os padrões das duas famílias
divergem: `NTuplaDissertacaoAprox` usa 12 e `NTuplaAnbubusAprox` usa **8**.

Sem passar isto, a configuração vencedora só valia para a dissertação — o anbubus
rodava com `n = 8`, ou `N = 100` em vez de 67, uma rede que a grade nunca mediu. E
como a conversão de taxa divide por `N`, o passo efetivo saía 1,49× acima do
validado. Dois defeitos que se compunham em instabilidade construída.

`n = 12` em todos os seis melhores pontos das duas famílias; nenhum usa 24."""


def _cfgs(replica: str, na: int, ns: int) -> tuple[EnvConfig, EnvConfig]:
    """`(treino, avaliação)`. As duas fases diferem e a diferença muda a métrica.

    Treino usa `end_when_all_collected=False` — o `--no-end-when-all-collected` do
    `benchmark.py`, que dá os 30,5 s de experiência pós-conclusão medidos em
    `spec06_bancos.py`. Avaliação usa `True`, que é o protocolo do
    `visualize_performance.py` que gerou a fig. 7.3.
    """
    r = REPLICAS[replica]
    base = {"num_drones": na, "num_sensors": ns,
            "max_episode_length": r["horizonte"]}
    if r["cegos"]:
        base["state_num_closest_drones"] = 0
    treino = EnvConfig(**{**LAMENZA_003(na, ns, "train").kwargs(), **base})
    aval = EnvConfig(**{**LAMENZA_003(na, ns, "eval").kwargs(), **base})
    return treino, aval


def _ctx(cfg: EnvConfig) -> ContextoEpisodio:
    return ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval,
        max_episode_length=cfg.max_episode_length,
        state_num_closest_drones=cfg.state_num_closest_drones)


def ap_taxa(aprox: str, n_bits: int) -> float:
    """Converte `PASSO_EFETIVO` para a taxa nominal da convenção da família.

    Somar `passo` às `N` posições endereçadas muda `f` em `passo` na convenção de
    média e em `passo·N` na de soma. Como o `anbubus` usa soma, a taxa dele tem de
    ser dividida por `N` para produzir o MESMO deslocamento — e `N` depende de
    `(T, n)`, então a conversão é por configuração, não uma tabela fixa.
    """
    if aprox == "ntupla_anbubus":
        import math
        # `TAMANHO_TUPLA`, não um 12 fixo: com o literal, `n = 8` daria taxa 1,49×
        # grande demais e `n = 24` daria 0,51×, porque `N = ceil(n_bits/n)`.
        return PASSO_EFETIVO / math.ceil(n_bits / TAMANHO_TUPLA)
    return PASSO_EFETIVO


def montar_agente(aprox: str, cfg: EnvConfig, sem: PapeisDeSemente,
                  corpus: np.ndarray):
    """Aproximador + codificador + política. `seed_tuple` governa o ladrilhamento."""
    d_obs = cfg.observation_size
    if aprox == "mlp":
        ap = APROXIMADORES[aprox](n_bits_ou_dim=d_obs, seed=sem.seed_train)
        codificar = lambda v: np.asarray(v, float).ravel()          # noqa: E731
        termo = None
    else:
        termo = TermometroQuantil(BITS_POR_ESCALAR).ajustar(corpus)
        # `seed=seed_tuple`: é o ladrilhamento, e a D-06.1 o quer variando com a
        # repetição. Passar `seed_train` aqui acoplaria tuplas e ruído.
        ap = APROXIMADORES[aprox](n_bits=termo.n_bits, seed=sem.seed_tuple,
                                  n_dims_codificador=d_obs,
                                  limiares_por_dim=BITS_POR_ESCALAR,
                                  tamanho_tupla=TAMANHO_TUPLA,
                                  taxa_ator=ap_taxa(aprox, termo.n_bits))
        codificar = lambda v: termo.codificar(np.asarray(v, float)[None, :])[0]  # noqa: E731
    politica = AtorCriticoContinuo(aproximador=ap, codificar=codificar,
                                   politica=NormalProjetada(SIGMA, None),
                                   id=f"wnn_{aprox}", treinando=True)
    return ap, termo, politica, codificar


def corpus_de_observacoes(cfg: EnvConfig, n_episodios: int = 3,
                          seed: int = 99_991) -> np.ndarray:
    """Corpus para ajustar o termômetro quantil (E2).

    Colhido com **caminhada aleatória**, e de cenários fora dos bancos: ajustar os
    limiares em cenários de avaliação seria vazamento — os limiares são estado
    treinado (a SPEC-03 os conta no footprint), então caberiam no mesmo cuidado que
    o checkpoint.
    """
    ctx = _ctx(cfg)
    aleat = BASELINES["random_uniforme"]()
    aleat.reset(ctx, np.random.default_rng(seed))
    obs: list[np.ndarray] = []

    def coletar(observacao):
        obs.append(np.asarray(observacao, float).ravel())
        return aleat.act(Observacao.decodificar(observacao, ctx)).como_array()

    for s in range(n_episodios):
        run_episode(cfg, coletar, seed=seed + s)
    return np.array(obs)


def rodar_episodios_de_treino(cfg: EnvConfig, politica, ctx, recompensa,
                              rng_cen, orcamento: int,
                              escala: float = 1.0) -> dict:
    """Treina até gastar `orcamento` passos de ambiente.

    A recompensa vem de `recompensa(colhidos, colhidos_antes, Ns)` e **não** do
    ambiente: é o que permite a eq. 7-2 do §7.1, que o ambiente não implementa.
    """
    passos = 0
    passos_max = int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2
    retornos = []
    while passos < orcamento:
        s = int(rng_cen.integers(2_000_000_000, 4_000_000_000))
        env = make_env(cfg)
        random.seed(s)
        np.random.seed(s)
        obs, _ = env.reset()
        politica.reset(ctx, politica._rng or np.random.default_rng(s))
        antes = 0
        total = 0.0
        for _ in range(passos_max):
            acoes = {}
            for a in env.agents:
                acoes[a] = politica.act(Observacao.decodificar(obs[a], ctx)).como_array()
            obs, _, term, trunc, _ = env.step(acoes)
            colhidos = int(env.sensors_collected)
            r = escala * recompensa(colhidos, antes, cfg.num_sensors)
            antes = colhidos
            total += r
            politica.registrar(r)
            passos += 1
            if any(term.values()) or any(trunc.values()):
                break
            politica.passo_intermediario(Observacao.decodificar(
                obs[env.agents[0]], ctx))
            if passos >= orcamento:
                break
        politica.fim_de_episodio()
        retornos.append(total)
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return {"passos": passos, "recompensa_media": float(np.mean(retornos))}


def avaliar(cfg: EnvConfig, politica, ctx, sementes) -> dict:
    """Determinística — a **média** da gaussiana, sem ruído (§4.2).

    Devolve o agregado e o **por episódio**, que é o insumo do bootstrap pareado.
    """
    politica.treinando = False
    try:
        linhas = []
        for s in sementes:
            politica.reset(ctx, np.random.default_rng(int(s)))

            def pol(observacao, _p=politica, _c=ctx):
                return _p.act(Observacao.decodificar(observacao, _c)).como_array()

            r = run_episode(cfg, pol, seed=int(s))
            linhas.append({"seed": int(s), "ct": r.completion_time,
                           "sucesso": bool(r.all_collected), "passos": r.steps,
                           "cobertura": r.collected / r.n_sensors,
                           "recompensa": r.reward_sum})
    finally:
        politica.treinando = True

    ct = np.array([x["ct"] for x in linhas], float)
    ok = np.array([x["sucesso"] for x in linhas], bool)
    return {"sr": float(ok.mean()), "ct_cens": float(ct.mean()),
            "ct_succ": float(ct[ok].mean()) if ok.any() else float("nan"),
            "cobertura": float(np.mean([x["cobertura"] for x in linhas])),
            "recompensa_media": float(np.mean([x["recompensa"] for x in linhas])),
            "por_episodio": linhas}


def _uma_corrida(args: tuple) -> dict:
    """Uma corrida completa — **em processo próprio**, ver o comentário em `main`.

    Recebe só dados planos e reconstrói tudo, porque o que atravessa a fronteira de
    processo precisa ser serializável e porque reconstruir é o que garante que o
    filho não herde estado. O corpus do termômetro é determinístico (semente fixa em
    `corpus_de_observacoes`), então recalculá-lo por semente não muda resultado —
    custa três episódios.
    """
    (replica, aproximador, na, ns, b_max, seed_train, n_val, n_test, raiz,
     ref_sr, ref_ct, ref_cob) = args
    r = REPLICAS[replica]
    cfg_tr, cfg_av = _cfgs(replica, na, ns)
    bancos = carregar_bancos(na, ns)
    val = bancos.val.sementes[:n_val] if n_val else bancos.val.sementes
    test = bancos.test.sementes[:n_test] if n_test else bancos.test.sementes
    ctx_tr, ctx_av = _ctx(cfg_tr), _ctx(cfg_av)
    corpus = corpus_de_observacoes(cfg_tr)
    recompensa = RECOMPENSAS[r["recompensa"]]

    sem = PapeisDeSemente(seed_train=seed_train)
    aprox, termo, politica, _ = montar_agente(aproximador, cfg_tr, sem, corpus)
    cfg_corrida = {"replica": replica, "aproximador": aproximador,
                   "na": na, "ns": ns, "b_max": b_max,
                   "recompensa": r["recompensa"],
                   "horizonte": r["horizonte"], "cegos": r["cegos"],
                   "escala_recompensa": r["escala_recompensa"],
                   "gamma": GAMMA,
                   "bits_por_escalar": BITS_POR_ESCALAR}
    corrida = Corrida(experimento=f"spec06-{replica}", config=cfg_corrida,
                      sementes=sem, bancos=bancos, raiz=Path(raiz))
    print(f"  semente {seed_train} -> {corrida.dir}", flush=True)

    rng_cen = sem.rng_cenarios_treino()

    def salvar(destino: Path, _ap=aprox, _t=termo):
        destino.mkdir(parents=True, exist_ok=True)
        _ap.salvar(destino)
        if _t is not None:
            _t.salvar(destino / "termometro.npz")

    resumo = treinar(
        corrida,
        passo_de_treino=lambda orc, _c=cfg_tr, _p=politica, _x=ctx_tr,
        _r=recompensa, _g=rng_cen, _e=r["escala_recompensa"]:
            rodar_episodios_de_treino(_c, _p, _x, _r, _g, orc, _e),
        avaliar_val=lambda _c=cfg_av, _p=politica, _x=ctx_av, _v=val: {
            k: v for k, v in avaliar(_c, _p, _x, _v).items()
            if k != "por_episodio"},
        b_max=b_max,
        salvar_checkpoint=salvar,
        metrica=r["metrica"], maior_e_melhor=r["maior_e_melhor"],
        ocupadas=lambda _a=aprox: (
            _a.ator.entradas_ocupadas()
            if hasattr(_a.ator, "entradas_ocupadas")
            else _a.ator.n_parametros()),
    )

    # AGORA, e só agora, o banco `test`.
    final = avaliar(cfg_av, politica, ctx_av, test)
    corrida.gravar_eval_test(final["por_episodio"])
    # Pela COBERTURA e não por `SR`: com a aleatória em SR = 0 %, o critério por SR
    # dispara para qualquer corrida sub-treinada e deixa de distinguir "falhou" de
    # "não terminou de treinar".
    catastrofica = (final["cobertura"] <= ref_cob if r["maior_e_melhor"]
                    else final["ct_cens"] >= ref_ct)
    print(f"    test: SR={final['sr']:.1%} cob={final['cobertura']:.1%} "
          f"CT_cens={final['ct_cens']:.1f} CT_succ={final['ct_succ']:.1f}"
          f"{'  FALHA CATASTRÓFICA' if catastrofica else ''}", flush=True)

    return {"na": na, "ns": ns, "seed_train": seed_train,
            "dir": str(corrida.dir), "config_hash": corrida.config_hash,
            **{k: v for k, v in resumo.items() if k != "historico_val"},
            "test": {k: v for k, v in final.items() if k != "por_episodio"},
            "falha_catastrofica": bool(catastrofica),
            "referencia_aleatoria": {"sr": ref_sr, "ct_cens": ref_ct,
                                     "cobertura": ref_cob}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replica", choices=sorted(REPLICAS), default="1a")
    ap.add_argument("--aproximador", choices=sorted(APROXIMADORES), default="ntupla_dissertacao")
    ap.add_argument("--celulas", default="1:12")
    ap.add_argument("--sementes", type=int, default=5, help="`R` da §4.3")
    ap.add_argument("--b-max", type=int, default=19_500_000,
                    help="passos por corrida; padrão = o B_max medido em T-06.0")
    ap.add_argument("--n-val", type=int, default=None, help="reduz o banco val")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--raiz", default="runs")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()

    r = REPLICAS[a.replica]
    celulas = [tuple(int(x) for x in c.split(":")) for c in a.celulas.split(",")]
    saida = Path(a.saida or f"reports/spec06-{a.replica}-{a.aproximador}.json")
    art: dict = {"replica": a.replica, "aproximador": a.aproximador,
                 "config": vars(a), "protocolo": r, "corridas": []}

    print(f"réplica {a.replica} · {a.aproximador} · B_max={a.b_max:,} · "
          f"R={a.sementes}")
    print(f"recompensa {r['recompensa']} × {r['escala_recompensa']} · "
          f"seleção por {r['metrica']}, reporta {r['reportada']} · horizonte {r['horizonte']:.0f} s · N*_a="
          f"{'0 (cegos)' if r['cegos'] else 'Na−1'}")
    if a.replica != "1b":
        print(f"  AVISO: `PASSO_EFETIVO`={PASSO_EFETIVO:.1e} e `SIGMA`={SIGMA} foram "
              f"medidos na réplica **1b** (grade `desempate`).\n"
              f"  Você está na {a.replica}, que usa outra recompensa "
              f"({r['recompensa']}, escala {r['escala_recompensa']:.3g}).\n"
              f"  A escala de `δ` difere, e é `α·δ·∇log π` que a assertiva da §4.2 "
              f"mede — a estabilidade não transfere de graça.\n")
    if r["escala_recompensa"] != 1.0:
        print(f"  escala {r['escala_recompensa']} = (1−γ): normalização do alvo de "
              f"valor prescrita pela T-06.1.\n  Sem ela a asserção da §4.2 aborta "
              f"— medido, no passo 998.\n")

    # POOL DE UM, RECICLADO A CADA CORRIDA — e a razão foi medida na SPEC-06.
    #
    # O processo que treina **degrada 1,55× por corrida**, de forma geométrica:
    # medido na grade de desempate, o mesmo trabalho passou de 2,0 para 50,8
    # ms/passo ao longo de 9 corridas no mesmo worker, com composição idêntica,
    # memória sem pressão, swap zerado e `espera-io` em 0 %. Um worker recém-criado
    # fazia 4,3 ms/passo enquanto os antigos faziam 51,6, na mesma contenção.
    #
    # A causa não é construção de ambiente (medi a curva até 40 mil ambientes num
    # processo e ela é plana). A assinatura — RSS estável, sem swap, buscas 12× mais
    # lentas — é de fragmentação de heap por construir e descartar centenas de MB de
    # tabelas hash a cada corrida.
    #
    # AQUI SERIA PIOR QUE NA GRADE. A T-06.5 escala `Na = 1 → 2 → 4 → 8` com `R = 5`,
    # ou seja até 20 corridas SEQUENCIAIS de `b_max = 19,5 M` passos no mesmo
    # processo. A vigésima rodaria `1,55¹⁹ ≈ 12 000×` mais lenta — a campanha não
    # terminaria, e o sintoma pareceria "a WNN não escala com `Na`", que é
    # exatamente a conclusão que este trabalho quer medir.
    #
    # `mp.Pool(1, maxtasksperchild=1)` mantém a semântica sequencial — uma corrida
    # por vez, na mesma ordem — e dá heap novo a cada uma. O `fork` faz o filho
    # herdar a memória do pai, então custa milissegundos. O `maxtasks` é checado
    # depois de `completed += 1`, então nenhuma corrida é interrompida no meio.
    with mp.Pool(1, maxtasksperchild=1) as pool:
        for na, ns in celulas:
            cfg_av = _cfgs(a.replica, na, ns)[1]
            bancos = carregar_bancos(na, ns)
            val = bancos.val.sementes[:a.n_val] if a.n_val else bancos.val.sementes
            ctx_av = _ctx(cfg_av)

            # Referência da falha catastrófica (§4.4): a caminhada aleatória. Fica no
            # pai porque não constrói tabela — não é fonte de fragmentação.
            aleat = BASELINES["random_uniforme"]()
            aleat.reset(ctx_av, np.random.default_rng(0))

            def pol_aleat(observacao, _p=aleat, _c=ctx_av):
                return _p.act(Observacao.decodificar(observacao, _c)).como_array()

            ref = [run_episode(cfg_av, pol_aleat, seed=int(s)) for s in val]
            ref_sr = float(np.mean([x.all_collected for x in ref]))
            ref_ct = float(np.mean([x.completion_time for x in ref]))
            ref_cob = float(np.mean([x.collected / x.n_sensors for x in ref]))
            print(f"[{na}a{ns}s] caminhada aleatória no val: SR={ref_sr:.1%} "
                  f"cobertura={ref_cob:.1%} CT={ref_ct:.1f} s")
            print("           falha catastrófica medida pela COBERTURA — SR e CT saturam")

            for seed_train in range(a.sementes):
                art["corridas"].append(pool.apply(_uma_corrida, ((
                    a.replica, a.aproximador, na, ns, a.b_max, seed_train,
                    a.n_val, a.n_test, a.raiz, ref_sr, ref_ct, ref_cob),)))

    n_falhas = sum(c["falha_catastrofica"] for c in art["corridas"])
    print(f"\nfalhas/R = {n_falhas}/{len(art['corridas'])}  "
          f"(§4.3: nenhuma corrida descartada)")
    art["falhas"] = n_falhas
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"artefato: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
