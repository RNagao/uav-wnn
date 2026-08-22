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
from uavwnn.campanha import (  # noqa: E402
    FAIXA_PADRONIZADA,
    RECOMPENSAS,
    Corrida,
    NormalizadorMovel,
    PapeisDeSemente,
    carregar_bancos,
)
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


# --- a grade fatorial, em dois níveis de cardinalidade ---------------------
#
# O custo é aritmética, e ele decide o desenho (12 núcleos, ~6,5 min/corrida a
# 200 mil passos):
#
#   completo  passo(3)×σ(3)×T(3)×n(5)×m(4)×crítico(4) = 2 160 pontos
#             × 2 modelos × 1 semente =  4 320 corridas ->   39 h
#             × 2 modelos × 3 sementes = 12 960 corridas ->  117 h
#   reduzido  passo(3)×σ(3)×T(2)×n(3)×m(2)×crítico(2) =   216 pontos
#             × 2 modelos × 1 semente =    432 corridas ->  3,9 h
#             × 2 modelos × 3 sementes =  1 296 corridas -> 11,7 h
#
# Os dois são **fatoriais completos** — o reduzido não é uma amostra do completo,
# é o mesmo cruzamento com menos níveis por eixo. A diferença é resolução, não
# cobertura, e é isso que o torna uma escolha de orçamento em vez de um atalho.

GRADE = {
    # --- grade de DESEMPATE: a decisão não decidiu, e disse por quê ---------
    #
    # A grade `decisao` (108 corridas) deu Δ = +0,7 pp com IC [-2,9; +3,8] — inclui
    # zero. Mas separando por regime de estabilidade o quadro é nítido:
    #
    #   estável   (passo ≤ 5e-4)   Δ = +4,3 pp  IC [+2,0; +6,3]  11/12 pontos
    #   instável  (passo = 1,5e-3) Δ = -6,7 pp  IC [-12,3; -1,3]  1/6 pontos
    #
    # e o instável não é média comparável: morreram 13/18 corridas da dissertação
    # contra 8/18 do anbubus, e as mortas entram com a cobertura do momento da
    # morte. Cada eixo abaixo responde a um achado daquela grade.
    #
    # O QUE ENTRA, E POR QUÊ
    #
    # `passo` de 3e-4 a 3e-3, cinco níveis. É o eixo da INTERAÇÃO: a dissertação
    #   pica em 5e-4 e cai; o anbubus sobe monotonicamente (7,7 -> 15,9 -> 18,5 %)
    #   e ainda estava subindo no topo da faixa. Bracketa os dois ótimos em vez de
    #   um só — foi exatamente esse viés que fez as 192 corridas antigas (passo =
    #   2e-4, o único nível onde a dissertação domina os seis pontos) parecerem
    #   veredito geral.
    #
    # `crítico` volta com três níveis, e é o eixo que mais importa aqui: é o único
    #   em que as duas famílias diferem por construção (φ acumulativo contra taxa
    #   de descida), nunca foi variado, e é o suspeito da mortalidade. O crítico da
    #   dissertação acumula pela eq. (4.8), o que produz picos de δ maiores — e é
    #   δ·∇log π que a assertiva mede. Se for isso, φ = 0,70 estabiliza.
    #
    # `T` e `n` voltam com dois níveis. Fixá-los em (16, 12) na grade de decisão
    #   limitou o melhor ponto a 31 % de cobertura, contra 77,5 % que a grade de
    #   dois eixos tinha alcançado com outro (T, n).
    #
    # O QUE SAI, E A ECONOMIA É MERECIDA
    #
    # `normalizar` sai, fixado em False: medido, ele PIORA os dois (dissertação
    #   15,8 -> 13,5 %, anbubus 14,9 -> 13,2 %). A hipótese de que a estagnação do
    #   anbubus era artefato do nosso codificador está refutada — era o passo. Sair
    #   corta a grade pela metade, e essa é a economia paga pela medição.
    #
    # `m` fica fixo em 8 e isso é uma LIMITAÇÃO DECLARADA, não uma conclusão: ele
    #   entra em δ, que é o que dispara a assertiva, então poderia interagir com a
    #   mortalidade. Cruzá-lo dobraria para 27 h. Fica para a coordenada da T-06.4.
    #
    # ORÇAMENTO: 200 mil passos, não 75 mil. Na grade de decisão o SR foi 0 % nas
    # 108 corridas — ninguém completou um episódio, então ela mediu "quem chega
    # mais longe antes de desistir", não quem resolve a tarefa. A grade de dois
    # eixos chegou a 34 % de SR com 200 mil.
    #
    #   5 × 2 × 2 × 2 × 3 = 120 pontos × 2 modelos × 3 sementes = 720 corridas
    #   a 13 min/corrida medidos em 12 processos -> ~13,5 h
    "desempate": {
        "passo_efetivo": (3e-4, 5e-4, 8e-4, 1.5e-3, 3e-3),
        "sigma": (0.20, 0.35),
        "resolucao": (16, 32),
        "tupla": (12, 24),
        "passos_m": (8,),
        "idx_critico": (0, 1, 2),
        "normalizar": (False,),
    },
    # --- grade de DECISÃO: só para escolher entre os dois modelos -----------
    #
    # Não é triagem de hiperparâmetro; é um teste pareado com um objetivo único.
    # Por isso varre apenas os eixos em que a evidência anterior era fina ou
    # confundida, e **fixa** os demais em valores defensáveis:
    #
    #   varridos   passo (3) × σ (3) × normalizar (2) = 18 pontos
    #   fixos      T = 16, n = 12 (N = 34, custo médio-baixo)
    #              m = 8  (o mais barato dos dois níveis)
    #              crítico: φ = 0,85 (o melhor nas 4 tarefas do apêndice A) e
    #                       razão = 1,0 (crítico na mesma taxa do ator, o default)
    #
    # POR QUE ESTES TRÊS EIXOS
    #
    # `passo` e `σ` são os dominantes medidos, e as 192 corridas anteriores
    # cobriam **um único nível de cada** (2e-4 e 0,20) — o resultado pareado
    # valia para um canto do espaço. `normalizar` entra porque a estagnação do
    # anbubus pode ser artefato do nosso codificador, e sem cruzá-lo com passo e
    # σ não há como separar "saturou" de "pipeline errado".
    #
    # RESSALVA REGISTRADA: `α` e `σ` continuam **confundidos** — o passo efetivo é
    # `~α·δ/σ`, então variar `σ` muda exploração E tamanho do passo. Para ESTA
    # pergunta isso é tolerável, porque o confundimento é idêntico nos dois braços
    # (mesma política, mesmo σ) e portanto não favorece nenhum. Para concluir qual
    # `σ` é melhor, seria preciso reparametrizar.
    "decisao": {
        "passo_efetivo": (2e-4, 5e-4, 1.5e-3),
        "sigma": (0.20, 0.35, 0.50),
        "resolucao": (16,),
        "tupla": (12,),
        "passos_m": (8,),
        "idx_critico": (1,),
        "normalizar": (False, True),
    },
    "reduzido": {
        "passo_efetivo": (2e-4, 5e-4, 1.5e-3),
        "sigma": (0.20, 0.35, 0.50),
        "resolucao": (16, 32),
        "tupla": (8, 12, 24),
        "passos_m": (8, 16),
        "idx_critico": (1, 2),
        "normalizar": (False, True),
    },
    "completo": {
        # `passo` centrado no vencedor da primeira grade (5e-4 efetivo)
        "passo_efetivo": (2e-4, 5e-4, 1.5e-3),
        # `σ` a partir de 0,20, que era o TETO da faixa anterior e onde estava o
        # degrau — a faixa antiga já foi medida como estrangulada
        "sigma": (0.20, 0.35, 0.50),
        "resolucao": (16, 32, 64),
        "tupla": (8, 12, 16, 24, 32),
        "passos_m": (4, 8, 16, 32),
        "idx_critico": (0, 1, 2, 3),
        "normalizar": (False, True),
    },
}
"""O eixo `normalizar` carrega o **par coerente** de cada pipeline, e não a
normalização isolada — ver `campanha.normalizacao` para por que isolar não é
possível:

    False   `TermometroQuantil` ajustado ao corpus cru        (nosso)
    True    normalização móvel + termômetro de faixa fixa     (do `anbubus`)

Ele existe porque o braço `ntupla_anbubus` **estagnou** em ~37 % de cobertura (+0,8
pontos ao dobrar o orçamento, contra +14 do outro braço) usando o nosso codificador,
enquanto todas as configurações dos autores usam `MeanStdFilter`. Sem este eixo, não
há como distinguir "saturou" de "codificador errado"."""


def grade(nivel: str) -> list[dict]:
    """Produto cartesiano dos seis eixos — a grade **completa** naquele nível.

    Os pontos são **compartilhados** pelos dois modelos, como na busca aleatória:
    cada configuração tem par exato no outro modelo, e a comparação é pareada por
    configuração e não só por orçamento.
    """
    import itertools
    eixos = GRADE[nivel]
    chaves = list(eixos)
    fora = []
    for i, combo in enumerate(itertools.product(*(eixos[k] for k in chaves))):
        fora.append({"id": i, **dict(zip(chaves, combo))})
    return fora


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
            "normalizar": bool(g.integers(2)),
        })
    return fora


def _montar(aprox: str, cfg: EnvConfig, sem: PapeisDeSemente, amostra: dict):
    """Instancia o modelo, convertendo a taxa para a convenção da família."""
    d_obs = cfg.observation_size
    corpus = corpus_de_observacoes(cfg)
    normaliza = bool(amostra.get("normalizar", False))
    if normaliza:
        # Pipeline do `anbubus`: normalização móvel + faixa FIXA. O normalizador é
        # semeado no corpus para não começar de estatística vazia — com `n = 1` o
        # desvio é 1 por convenção e as primeiras observações sairiam sem escala.
        nz = NormalizadorMovel(d_obs)
        for v in corpus:
            nz.atualizar(v)
        termo = TermometroQuantil(amostra["resolucao"]).ajustar(
            np.tile(np.linspace(*FAIXA_PADRONIZADA, 512)[:, None], (1, d_obs)))
    else:
        nz = None
        termo = TermometroQuantil(amostra["resolucao"]).ajustar(corpus)
    comum = dict(n_bits=termo.n_bits, tamanho_tupla=amostra["tupla"],
                 seed=sem.seed_tuple, n_dims_codificador=d_obs,
                 limiares_por_dim=amostra["resolucao"])

    # `N` depende de (T, n), então a conversão é POR AMOSTRA — ver o cabeçalho.
    from uavwnn.wnn.redes import RedeGradienteFuncional
    n_ram = RedeGradienteFuncional(termo.n_bits, amostra["tupla"], seed=0).n_ram
    passo = amostra["passo_efetivo"]

    if aprox == "ntupla_dissertacao":
        ap = APROXIMADORES[aprox](taxa_ator=passo, phi=PHI[amostra["idx_critico"]],
                                  **comum)
        detalhe = {"phi": PHI[amostra["idx_critico"]], "taxa_ator": passo}
    else:
        taxa = passo / n_ram
        razao = RAZAO_CRITICO[amostra["idx_critico"]]
        ap = APROXIMADORES[aprox](taxa_ator=taxa, taxa_critico=taxa * razao, **comum)
        detalhe = {"razao_critico": razao, "taxa_ator": taxa}

    if nz is None:
        codificar = lambda v: termo.codificar(np.asarray(v, float)[None, :])[0]  # noqa: E731
    else:
        # `atualizar` segue a fase: no treino a estatística acompanha a deriva; na
        # avaliação ela congela, senão o agente se calibraria nos episódios de
        # avaliação — vazamento.
        def codificar(v, _t=termo, _n=nz):
            return _t.codificar(_n.aplicar(v, atualizar=not _n.congelado)[None, :])[0]

    pol = AtorCriticoContinuo(aproximador=ap, codificar=codificar,
                              politica=NormalProjetada(amostra["sigma"], None),
                              passos=amostra["passos_m"], id=f"busca_{aprox}",
                              treinando=True)
    return ap, termo, pol, {**detalhe, "n_ram": n_ram, "n_bits": termo.n_bits,
                            "normalizador": nz}


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
        nz = det.pop("normalizador", None)
    except Exception as e:
        return {"aproximador": aprox, **amostra, "seed_train": seed_train,
                "abortou": f"montagem: {type(e).__name__}: {e}", "cobertura": 0.0,
                "sr": 0.0, "ct_cens": float(cfg_av.max_episode_length),
                "val_cobertura": 0.0, "passos": 0, "parede_s": 0.0,
                "b_sweep": b}

    cfg_corrida = {"busca": True, "replica": replica, "aproximador": aprox,
                   "na": 1, "ns": 12, "b_sweep": b, **amostra, **det}
    corrida = Corrida(experimento=f"spec06-busca-{replica}", config=cfg_corrida,
                      sementes=sem, bancos=bancos, raiz=Path(raiz))
    corrida.preparar()

    def salvar_checkpoint(rotulo: str) -> None:
        """Grava ator, crítico, termômetro **e** normalizador.

        Os três últimos são estado **treinado**, não configuração: os limiares do
        termômetro quantil vêm de um corpus, e a estatística do normalizador vem do
        treino. Um checkpoint sem eles carregaria o modelo com a entrada em outra
        escala — erro silencioso, com predições fora de faixa e nenhuma exceção.
        """
        dest = corrida.dir / "checkpoints" / rotulo
        dest.mkdir(parents=True, exist_ok=True)
        ap.salvar(dest)
        termo.salvar(dest / "termometro.npz")
        if nz is not None:
            nz.salvar(dest / "normalizador.npz")

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
            if nz is not None:
                nz.congelar()
            m = avaliar(cfg_av, pol, ctx_av, val)
            if nz is not None:
                nz.descongelar()
            if m["cobertura"] > melhor:
                melhor = m["cobertura"]
                # Checkpoint de MELHOR validação, como a §4.4 manda — não o último.
                # Sem isto a grade produzia métricas e nenhum modelo, e escolher o
                # vencedor obrigaria a retreinar cada finalista do zero.
                salvar_checkpoint("melhor")
            # A curva por avaliação, que a primeira grade NÃO gravava — eu escrevi
            # um laço próprio e esqueci o `registrar`. Sem ela não há como
            # distinguir "estabilizou baixo" de "ainda subia quando o orçamento
            # acabou", que é exatamente o que se precisa saber de um ponto ruim.
            from uavwnn.campanha import Avaliacao
            corrida.registrar(Avaliacao(
                passo=passos, recompensa_media=float(m.get("recompensa_media", 0.0)),
                sr=float(m["sr"]), ct_cens=float(m["ct_cens"]),
                ct_succ=float(m["ct_succ"]), cobertura=float(m["cobertura"]),
                ocupadas=int(ap.ator.entradas_ocupadas()),
                parede_s=time.perf_counter() - t0,
                passos_por_s=passos / max(1e-9, time.perf_counter() - t0)))
            if parada.registrar(m["cobertura"]):
                break
    except Exception as e:
        # Aborto é INFORMAÇÃO sobre o ponto (aquele passo é instável), não perda da
        # busca. É o que faz a fronteira de estabilidade aparecer no resultado.
        abortou = f"{type(e).__name__}: {e}"

    if nz is not None:
        nz.congelar()
    salvar_checkpoint("final")
    final = avaliar(cfg_av, pol, ctx_av, test)
    corrida.gravar_eval_test(final["por_episodio"])
    return {"aproximador": aprox, **amostra, **det, "seed_train": seed_train,
            "b_sweep": b,
            "passos": passos, "val_cobertura": float(melhor),
            "cobertura": final["cobertura"], "sr": final["sr"],
            "ct_cens": final["ct_cens"], "ct_succ": final["ct_succ"],
            "abortou": abortou, "parede_s": time.perf_counter() - t0,
            "dir": str(corrida.dir),
            "checkpoint_mb": round(sum(
                f.stat().st_size for f in (corrida.dir / "checkpoints").rglob("*")
                if f.is_file()) / 1e6, 3)}


_EIXOS = frozenset({"passo_efetivo", "sigma", "resolucao", "tupla",
                    "passos_m", "idx_critico", "normalizar"})
"""Os eixos que identificam uma corrida, e **só** eles.

DUAS EXCLUSÕES, POR MOTIVOS DIFERENTES

`phi`, `taxa_ator`, `n_ram` e as métricas estão no registro de progresso e **não**
podem entrar: os três primeiros são **derivados** do eixo de crítico e do passo por
família, então incluí-los faria a chave depender do modelo, e duas corridas do mesmo
ponto nunca casariam.

`id` fica fora porque ele é o índice do produto cartesiano, e **o índice muda com o
nível da grade**: a configuração `passo 5e-4, σ 0,20, T 16, n 12, crit 1` é `id=6` no
nível `decisao` e `id=25` no `desempate`. Com `id` na chave, dois níveis que
compartilham configurações e orçamento refariam o trabalho — o que importa quando a
descida coordenada da T-06.4 rodar no mesmo orçamento do desempate e sobrepor com
ele. Os eixos já determinam o ponto; o `id` só numera.

Na busca aleatória o `id` numera sorteios, mas dois sorteios com valores idênticos
**são** a mesma configuração — reaproveitar um pelo outro é inócuo, não é confusão.
"""


def _chave(rec: dict, eixos) -> tuple:
    """Identidade de uma corrida: modelo × semente × eixos.

    Ordena os eixos para que a chave não dependa da ordem de inserção do dict — em
    Python ela é estável, mas o JSON de um arquivo escrito por outra versão do
    script não precisa preservá-la.
    """
    return (rec["aproximador"], rec["seed_train"],
            tuple(sorted((k, rec[k]) for k in eixos)))


def _retomar(progresso, tarefas, b_sweep: int) -> tuple[list[dict], str]:
    """Lê o progresso existente e devolve só o que serve para ESTA grade.

    Três filtros, e cada um evita um jeito diferente de contaminar o resultado:

    1. **Orçamento.** Um registro treinado por outro `b_sweep` é outra corrida —
       o `config_hash` da §4.7 já separa as pastas por isso. Registros sem o campo
       `b_sweep` vêm de uma versão anterior do script, que não o gravava; são
       recusados em vez de presumidos, porque presumir aqui misturaria corridas de
       75 mil passos com as de 200 mil e o comparativo cairia por terra.
    2. **Pertinência.** Um registro cuja chave não está nas tarefas desta grade
       sobrou de outra e não entra no relatório final.
    3. **Duplicidade.** Se o arquivo tem a mesma corrida duas vezes (retomada de
       retomada), vale a última.

    Corridas ABORTADAS contam como concluídas e **não** são refeitas: o aborto é
    determinação, não falha de infraestrutura — mesma semente, mesmo ponto, mesmo
    estouro da assertiva de estabilidade.
    """
    alvos = {_chave({"aproximador": t[0], **t[1], "seed_train": t[2]},
                    t[1].keys() & _EIXOS)
             for t in tarefas}
    bons, notas = {}, []
    sem_campo = fora_orc = alheios = 0
    for linha in progresso.read_text().splitlines():
        if not linha.strip():
            continue
        try:
            x = json.loads(linha)
        except json.JSONDecodeError:
            continue          # última linha truncada por um Ctrl-C no meio da escrita
        if "b_sweep" not in x:
            sem_campo += 1
            continue
        if x["b_sweep"] != b_sweep:
            fora_orc += 1
            continue
        k = _chave(x, x.keys() & _EIXOS)
        if k not in alvos:
            alheios += 1
            continue
        bons[k] = x
    if sem_campo:
        notas.append(f"{sem_campo} sem campo b_sweep (versão antiga) descartadas")
    if fora_orc:
        notas.append(f"{fora_orc} de outro orçamento descartadas")
    if alheios:
        notas.append(f"{alheios} de outra grade descartadas")
    return list(bons.values()), " · ".join(notas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replica", choices=sorted(REPLICAS), default="1b")
    ap.add_argument("--modelos", default="ntupla_dissertacao,ntupla_anbubus")
    ap.add_argument("--modo", choices=("grade", "aleatoria"), default="grade",
                    help="`grade` = fatorial completo; `aleatoria` = sorteio")
    ap.add_argument("--niveis", choices=tuple(GRADE), default="reduzido",
                    help="cardinalidade da grade fatorial — ver o custo impresso")
    ap.add_argument("--amostras", type=int, default=24,
                    help="só no modo `aleatoria`: sorteios COMPARTILHADOS")
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--b-sweep", type=int, default=400_000)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--processos", type=int, default=12)
    ap.add_argument("--seed-busca", type=int, default=20260921)
    ap.add_argument("--raiz", default="runs")
    ap.add_argument("--saida", default="reports/spec06-busca.json")
    ap.add_argument("--retomar", action="store_true",
                    help="pula corridas já concluídas no .progresso.jsonl "
                         "existente, em vez de truncá-lo e refazer tudo")
    a = ap.parse_args()

    modelos = a.modelos.split(",")
    if a.modo == "grade":
        amostras = grade(a.niveis)
    else:
        amostras = amostrar(a.amostras, a.seed_busca)
    # MODELO NO LAÇO INTERNO: o `pool` despacha em ordem, então com o modelo no
    # laço externo a primeira rede rodaria inteira antes de a segunda começar. O
    # total seria o mesmo, mas nenhuma leitura parcial seria comparativa —
    # medido: 15 corridas concluídas, 15 da mesma rede. `chunksize` não resolve:
    # ele muda o lote, não a ordem.
    tarefas = [(m, s, sd, a.replica, a.b_sweep, a.n_val, a.n_test, a.raiz)
               for s in amostras for sd in range(a.sementes) for m in modelos]
    # EMBARALHA com semente fixa, e a razão foi medida.
    #
    # O produto cartesiano deixa os eixos em blocos: `resolucao` é o 3º, então `T`
    # alterna a cada 144 corridas. Com 12 workers, os 12 ficam quase sempre na
    # MESMA configuração de `T` — e aí o tempo por corrida mede **autocontenção do
    # bloco**, não o custo daquele ponto. Medido: `T=32` pareceu **18× mais lento**
    # que `T=16`, quando isolado é **1,7×**, e eu quase recomendei cortar um eixo
    # com base nisso.
    #
    # Pior que a medição errada: um parcial lido no meio do caminho fica enviesado
    # para o bloco corrente, e a leitura comparativa deixa de valer. Semente fixa
    # preserva a reprodutibilidade.
    import random as _r

    _r.Random(a.seed_busca).shuffle(tarefas)

    rot = (f"grade fatorial COMPLETA ({a.niveis})" if a.modo == "grade"
           else f"busca aleatória ({a.amostras} sorteios)")
    print(f"{rot} · réplica {a.replica} · Na=1 Ns=12 · sem ruído")
    print(f"{len(amostras)} pontos COMPARTILHADOS × {len(modelos)} modelos "
          f"× {a.sementes} sementes = {len(tarefas)} corridas")
    print(f"B_sweep = {a.b_sweep:,} · {a.processos} processos")
    if a.modo == "grade":
        print("eixos: " + " × ".join(f"{k}({len(v)})" for k, v in
                                     GRADE[a.niveis].items()))
    est = len(tarefas) * 6.5 * (a.b_sweep / 200_000) / a.processos / 60
    print(f"estimativa: ~{est:.1f} h de parede")
    print(f"progresso ao vivo: {Path(a.saida).with_suffix('.progresso.jsonl')}\n")

    # `imap_unordered` e não `map`: os resultados chegam **à medida que terminam**,
    # em vez de só no fim. É o que permite gravar o progresso e ler parciais sem
    # esperar as horas todas.
    progresso = Path(a.saida).with_suffix(".progresso.jsonl")
    progresso.parent.mkdir(parents=True, exist_ok=True)
    res = []
    if a.retomar and progresso.exists():
        res, desc = _retomar(progresso, tarefas, a.b_sweep)
        feitas = {_chave(x, x.keys() & _EIXOS) for x in res}
        antes = len(tarefas)
        tarefas = [t for t in tarefas
                   if _chave({"aproximador": t[0], **t[1], "seed_train": t[2]},
                             t[1].keys() & _EIXOS) not in feitas]
        print(f"RETOMANDO: {len(res)} corridas aproveitadas de "
              f"{antes} · {len(tarefas)} a fazer" + (f" · {desc}" if desc else ""))
        if not tarefas:
            print("nada a fazer — a grade já está completa.")
    else:
        progresso.write_text("")
    t0 = time.perf_counter()
    with mp.Pool(a.processos) as pool:
        for i, x in enumerate(pool.imap_unordered(uma, tarefas, chunksize=1), 1):
            res.append(x)
            with open(progresso, "a") as f:
                f.write(json.dumps(x) + "\n")
            if i % 5 == 0 or i == len(tarefas):
                dec = time.perf_counter() - t0
                resta = dec / i * (len(tarefas) - i)
                print(f"  [{i:>5}/{len(tarefas)}] {i/len(tarefas):>5.1%} · "
                      f"decorrido {dec/60:>5.1f} min · resta ~{resta/60:>5.1f} min",
                      flush=True)
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
        # `None` quando a corrida abortou na MONTAGEM: o retorno curto não tem
        # `phi` nem `razao_critico`, porque o modelo nunca chegou a existir.
        crit = a0.get("phi", a0.get("razao_critico"))
        crit_txt = "—" if crit is None else f"{crit:g}"
        linhas.append({"modelo": m, "id": i, "cobertura": cob, "sr": sr,
                       "abortos": ab, "n": len(g), **{k: a0[k] for k in
                       ("passo_efetivo", "sigma", "resolucao", "tupla", "passos_m")},
                       "critico": crit, "n_ram": a0.get("n_ram")})
        print(f"{m:>18} {i:>3} {a0['passo_efetivo']:>8.1e} {a0['sigma']:>5.2f} "
              f"{a0['resolucao']:>4} {a0['tupla']:>3} {a0['passos_m']:>3} "
              f"{crit_txt:>6} {a0.get('n_ram') or 0:>4} {cob:>6.1%} {sr:>5.1%} "
              f"{ab:>2}/{len(g)}")

    print(f"\n{'MELHOR POR MODELO':>40}")
    for m in modelos:
        cand = [l for l in linhas if l["modelo"] == m]
        if not cand:
            continue
        b = max(cand, key=lambda l: l["cobertura"])
        b_crit = "—" if b["critico"] is None else f"{b['critico']:g}"
        print(f"  {m}: passo {b['passo_efetivo']:.2e} · σ {b['sigma']:.3f} · "
              f"T {b['resolucao']} · n {b['tupla']} · m {b['passos_m']} · "
              f"crítico {b_crit} · N {b['n_ram'] or 0}")
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
