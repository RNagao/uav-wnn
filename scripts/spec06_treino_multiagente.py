"""Treino MULTIAGENTE da WNN (dissertação) — política compartilhada, Algoritmo 4.4.

**Parameter sharing correto:** UMA rede (aproximador) compartilhada por todos os
drones; **cada drone tem a sua própria trajetória de AC** (buffer + `_ultimo`), e
todas atualizam as MESMAS LUTs. Isto corrige o laço single-agente, em que um único
`registrar()` por passo (sobre `self._ultimo`, sobrescrito a cada `act()`) treinava
só o ÚLTIMO drone.

Fiel ao Lamenza: **um** conjunto de hiperparâmetros (a config vencedora da T-06.4),
**uma rede por célula**, só o cenário (Na, Ns) muda. Recompensa de time (eq. 7-2 /
punish), escala `(1−γ)` como na SPEC-06.

**Ressalva de método (documentada):** o MADDPG do Lamenza é CTDE — crítico
CENTRALIZADO (vê obs+ações de todos). O AC padrão aqui tem crítico DESCENTRALIZADO
on-policy (Algoritmo 4.4). O **ator implantado** é descentralizado e compartilhado
nos dois, logo comparável no deploy; os algoritmos de treino diferem, e a
comparação WNN×MADDPG é cross-método (a comparação "mesmo algoritmo" do projeto é
WNN-AC vs MLP-AC).

**Crítico centralizado (CTDE), desde 2026-10-28 — `--critico centralizado`.** A
campanha mediu que os drones da WNN voam a 0,5–0,85 da separação do MADDPG, e a
causa é o crítico `V(x_i)` que não vê os outros drones nem as ações deles (sem
atribuição de crédito). `uavwnn/agents/ctde.py` dá ao treino um crítico n-tuple
sobre o estado conjunto — `--contrafactual 0`: `V_c(s)` com δ compartilhado;
`--contrafactual K>0`: `Q_c(s, a)` com vantagem contrafactual por drone (COMA),
o análogo do `∂Q/∂a_i` do MADDPG. O ator, a observação, o deploy e o problema de
Lamenza NÃO mudam. A avaliação passa a registrar `separacao_m` (coordenação) e,
no CTDE, `credito.dispersao_credito` (0 = todos recebem o mesmo sinal).

    uv run python scripts/spec06_treino_multiagente.py --na 2 --ns 24 --smoke
    uv run python scripts/spec06_treino_multiagente.py --na 2 --ns 24 --b-max 10000000
    uv run python scripts/spec06_treino_multiagente.py --na 4 --ns 12,24,36 --resolucao 16 \
        --config-de reports/spec06-busca-a4s12-24-36.json --critico centralizado \
        --contrafactual 4 --b-max 10000000 --saida runs/spec06-ctde/T16-a4s12-24-36-ctde4

UMA GRADE POR `Na`, E O `Ns` REVEZADO

Quando a configuração vem de uma grade própria daquele `Na` (`spec06_busca.py
--na 2 --ns 12,24,36`), o treino longo deve rodar no MESMO regime em que ela foi
selecionada: `--config-de` lê o melhor ponto da grade para o `T` pedido, e `--ns`
em lista reveza o número de sensores por episódio com uma rede só.

    uv run python scripts/spec06_treino_multiagente.py --na 2 --ns 12,24,36 \
        --resolucao 16 --config-de reports/spec06-busca-a2s12-24-36.json \
        --b-max 50000000 --saida runs/spec06-50M/T16-a2s12-24-36
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_treinar import (  # noqa: E402
    RECOMPENSAS, REPLICAS, _cfgs, _ctx, corpus_de_observacoes,
)
from uavwnn.agents import APROXIMADORES, AtorCriticoContinuo  # noqa: E402
from uavwnn.agents.base import Observacao  # noqa: E402
from uavwnn.agents.ctde import CriticoCentral, TimeAtorCritico  # noqa: E402
from uavwnn.agents.politica_continua import NormalProjetada  # noqa: E402
from uavwnn.campanha import ParadaAntecipada, carregar_bancos  # noqa: E402
from uavwnn.encoding import TermometroQuantil  # noqa: E402
from uavwnn.refenv import make_env, prune_logging_handlers  # noqa: E402

# Config vencedora (T-06.4), fixa em todas as células — o análogo do conjunto
# único de hiperparâmetros do Lamenza.
PASSO, SIGMA, T_RES, N_TUPLA, PHI, M, GAMMA, D_SAIDA = 8e-4, 0.35, 32, 12, 0.85, 8, 0.99, 3
"""`T_RES` é o **padrão**, não uma constante: a campanha de 1 drone produziu dois
modelos que empatam em SR e diferem só em `T` — `T = 32` (67 RAMs) e `T = 16`
(34 RAMs, metade da memória, indistinguível sob ruído nas 16 células da escada
L→R). A escalada tem de rodar os dois, senão a fronteira custo×desempenho existe
em `Na = 1` e desaparece no multiagente. Use `--resolucao`; todo o resto é fixo,
como o conjunto único de hiperparâmetros do Lamenza exige."""
SEED_TREINO_BASE = 2_000_000


HP_PADRAO = {"passo": PASSO, "sigma": SIGMA, "tupla": N_TUPLA, "phi": PHI, "m": M}
"""Os hiperparâmetros que a grade varre, com o vencedor da T-06.4 como padrão.

Viraram parâmetro (`--passo`, `--sigma`, `--tupla`, `--phi`, `--m`, ou de uma vez
`--config-de <json da grade>`) porque a escalada passou a ter UMA grade por `Na`:
o modelo de `Na = 2` treina com o vencedor da grade de `Na = 2`, não com o de
`Na = 1`. Os padrões continuam os de sempre, então nenhuma corrida já feita muda."""


def _rotulo_ns(nss: tuple[int, ...]) -> str:
    """`24` para um Ns só, `12-24-36` para o revezamento — é o sufixo da célula."""
    return "-".join(str(n) for n in nss)


def _ler_ns(txt: str) -> tuple[int, ...]:
    """`"24"`, `"12,24,36"` ou `"12-24-36"` -> `(12, 24, 36)`, sem repetição, na ordem."""
    return tuple(dict.fromkeys(int(v) for v in str(txt).replace("-", ",").split(",")
                               if v.strip()))


def config_da_grade(arquivo: Path, t_res: int | None = None,
                    modelo: str = "ntupla_dissertacao") -> dict:
    """O melhor ponto do artefato de `spec06_busca.py`, restrito a `T = t_res`.

    Restringe a `T` porque a escalada treina os DOIS — `T = 16` e `T = 32` — e o
    melhor ponto global da grade só diria qual dos dois venceu; o que se quer é
    o melhor DE CADA UM. `cobertura` é a métrica de seleção da grade (média entre
    sementes e, com revezamento, entre células).
    """
    art = json.loads(Path(arquivo).read_text())
    pontos = [p for p in art["pontos"] if p["modelo"] == modelo
              and (t_res is None or int(p["resolucao"]) == int(t_res))]
    if not pontos:
        raise SystemExit(f"{arquivo}: nenhum ponto de {modelo}"
                         + (f" com T={t_res}" if t_res else ""))
    p = max(pontos, key=lambda x: x["cobertura"])
    return {"passo": float(p["passo_efetivo"]), "sigma": float(p["sigma"]),
            "tupla": int(p["tupla"]), "phi": float(p["critico"]),
            "m": int(p["passos_m"]), "resolucao": int(p["resolucao"]),
            "cobertura_grade": float(p["cobertura"]), "sr_grade": float(p["sr"]),
            "abortos_grade": f"{p['abortos']}/{p['n']}", "id_grade": p["id"],
            "grade": str(arquivo),
            "celula_grade": art.get("celula")}


def construir(cfg, na: int, t_res: int = T_RES, retomar: Path | None = None,
              c_max: float | None = 0.5, hp: dict | None = None,
              corpus=None):
    """Rede compartilhada + `na` trajetórias de AC (uma por drone).

    Se ``retomar`` for um dir de checkpoint, carrega ator/crítico/termômetro dali
    (em vez de sortear rede nova e reajustar o termômetro) e **continua** o treino
    de onde parou — usado para retomar um parcial em outra máquina.

    Ao retomar, `T` vem do **termômetro gravado**, não do argumento: os limiares
    são estado treinado, e montar a rede com um `T` diferente do que gerou os
    endereços daria uma rede que lê as LUTs nos lugares errados — o mesmo defeito
    que o `carregar` do `RedeGradienteFuncional` corrige para o mapeamento.

    `hp` são os hiperparâmetros da grade (ver `HP_PADRAO`); `corpus` pronto é para
    o revezamento de `Ns`, em que o termômetro tem de ver todas as células.
    """
    hp = {**HP_PADRAO, **(hp or {})}
    d_obs = cfg.observation_size
    if retomar is not None:
        termo = TermometroQuantil.carregar(Path(retomar) / "termometro.npz")
        gravado = termo.n_bits // d_obs
        if gravado != t_res:
            print(f"  T do checkpoint = {gravado} (o --resolucao {t_res} foi "
                  f"ignorado: os limiares são estado treinado)", flush=True)
        t_res = gravado
    else:
        termo = TermometroQuantil(t_res).ajustar(
            corpus if corpus is not None else corpus_de_observacoes(cfg))
    ap = APROXIMADORES["ntupla_dissertacao"](
        n_bits=termo.n_bits, tamanho_tupla=hp["tupla"], phi=hp["phi"],
        taxa_ator=hp["passo"], seed=0, d_saida=D_SAIDA, n_dims_codificador=d_obs,
        limiares_por_dim=t_res)
    if retomar is not None:
        rp = Path(retomar)
        ap.ator = type(ap.ator).carregar(next(rp.glob("*ator.npz")))
        ap.critico = type(ap.critico).carregar(next(rp.glob("*critico.npz")))
    if c_max != 0.5:
        # A asserção TreinoInstavel é NOSSA tripwire de sanidade (§4.2), não
        # parte do Algoritmo 4.4 — que não recorta a atualização. Ela existe
        # para pegar divergência cedo; aos 64 M de atualizações do T32-a2s24
        # um pico isolado de δ chegou a 0,5317 e derrubou a corrida a 40 M.
        # Como a retomada é determinística por semente de episódio, relançar
        # sem mudar nada reproduz o MESMO crash — daí o botão. Desligar não
        # altera nenhuma atualização abaixo do teto; apenas deixa passar as
        # que a asserção teria abortado. A curva de validação (a cada 1,25 M)
        # é quem vigia divergência de verdade.
        ap.ator.c_max = c_max
        print(f"  c_max = {c_max} (padrão 0,5) — asserção de estabilidade "
              f"{'DESLIGADA' if c_max is None else 'ajustada'}", flush=True)

    def codificar(v, _t=termo):
        return _t.codificar(np.asarray(v, float)[None, :])[0]

    acs = [AtorCriticoContinuo(
        aproximador=ap, codificar=codificar,
        politica=NormalProjetada(hp["sigma"], None),
        passos=hp["m"], desconto=GAMMA, id=f"wnn_a{i}", treinando=True)
        for i in range(na)]
    return ap, termo, acs


def avaliar_multi(cfg, ap, termo, ctx, sementes) -> dict:
    """Avaliação determinística multiagente: o ator compartilhado por drone."""
    def codificar(v, _t=termo):
        return _t.codificar(np.asarray(v, float)[None, :])[0]
    aval = AtorCriticoContinuo(aproximador=ap, codificar=codificar,
                               politica=NormalProjetada(SIGMA, None), passos=M,
                               desconto=GAMMA, id="wnn_eval", treinando=False)
    cts, oks, cov, seps = [], [], [], []
    for s in sementes:
        random.seed(int(s)); np.random.seed(int(s))
        env = make_env(cfg); obs, _ = env.reset()
        ct, ok = float(cfg.max_episode_length), False
        dist = []
        for _ in range(int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2):
            # Separação média entre drones a cada decisão — a medida direta da
            # coordenação (drones que dividem a área ficam longe; os que competem
            # pelos mesmos sensores ficam perto). É o que a correção CTDE tem de mover.
            if cfg.num_drones > 1:
                P = np.array([env.simulator.get_node(i).position[:2] for i in env.agent_node_ids])
                dist.append(float(np.mean([np.linalg.norm(P[i] - P[j])
                                           for i in range(len(P)) for j in range(i + 1, len(P))])))
            acoes = {a: aval.act(Observacao.decodificar(obs[a], ctx)).como_array()
                     for a in env.agents}
            obs, _, term, trunc, infos = env.step(acoes)
            if infos and "avg_reward" in next(iter(infos.values())):
                info = next(iter(infos.values()))
                ct = float(info.get("completion_time", cfg.max_episode_length))
                ok = bool(info.get("all_collected", False)); break
            if any(term.values()) or any(trunc.values()):
                break
        cov.append(int(env.sensors_collected) / cfg.num_sensors)
        cts.append(ct); oks.append(ok)
        if dist:
            seps.append(float(np.mean(dist)))
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return {"sr": float(np.mean(oks)), "ct_cens": float(np.mean(cts)),
            "cobertura": float(np.mean(cov)), "n": len(cts),
            "separacao_m": float(np.mean(seps)) if seps else float("nan")}


def avaliar_celulas(celulas: dict, ap, termo, sementes_por_ns: dict) -> dict:
    """`avaliar_multi` em cada `Ns`; o agregado é a MÉDIA entre células.

    Média simples, não ponderada por episódios: cada `Ns` é um problema, e a
    seleção de checkpoint não deve pender para o que tem mais sementes. Com um
    `Ns` só é o `avaliar_multi` de sempre, sem `por_ns`.
    """
    por = {ns: avaliar_multi(c["cfg_av"], ap, termo, c["ctx_av"],
                             sementes_por_ns[ns]) for ns, c in celulas.items()}
    if len(por) == 1:
        return next(iter(por.values()))
    agg = {k: float(np.mean([m[k] for m in por.values()]))
           for k in ("sr", "ct_cens", "cobertura", "separacao_m")}
    agg["n"] = sum(m["n"] for m in por.values())
    agg["por_ns"] = {str(ns): m for ns, m in por.items()}
    return agg


def treinar(na: int, nss: tuple[int, ...] | int, b_max: int, replica: str,
            saida: Path, progresso_cada: int, t_res: int = T_RES,
            retomar: Path | None = None, sem_parada: bool = False,
            passos_previos: int | None = None, val_n: int = 20,
            c_max: float | None = 0.5, hp: dict | None = None,
            guarda_colapso: bool = True, critico: str = "descentralizado",
            contrafactual: int = 0, res_acao: int = 16) -> dict:
    """`sem_parada` gasta o orçamento inteiro — a avaliação continua rodando.

    A §4.4 para em platô da `cobertura`, e ela satura em 1,0 cedo (medido: 1,5 M
    passos no `a2s24`, contra os 6,75 M em que a paciência de 8 finalmente
    disparou). Com `sem_parada` a curva de avaliação continua sendo gravada até o
    fim, que é o que permite responder se os passos além do platô da cobertura
    ainda compram `SR` — a métrica que não satura.

    `nss` em lista REVEZA o número de sensores a cada episódio (round-robin, na
    ordem dada), com uma rede só: `Ns` não muda a largura da observação — entram
    sempre os 12 sensores mais próximos — e a recompensa da eq. 5-6 já é
    normalizada por `Ns`. A validação roda em cada `Ns` e o checkpoint `melhor` é
    pela média. É o mesmo revezamento do `spec06_busca.py`, para que o treino
    longo aconteça no regime em que a configuração foi selecionada.
    """
    nss = (nss,) if isinstance(nss, int) else tuple(nss)
    hp = {**HP_PADRAO, **(hp or {})}
    if critico not in ("descentralizado", "centralizado"):
        raise ValueError(f"critico={critico!r}")
    if critico == "centralizado" and na < 2:
        print("  aviso: crítico centralizado com Na = 1 é o descentralizado com a "
              "própria ação na entrada — não há outro drone a ver", flush=True)
    r = REPLICAS[replica]
    escala = r["escala_recompensa"]
    recompensa = RECOMPENSAS[r["recompensa"]]
    celulas = {}
    for ns in nss:
        cfg_tr, cfg_av = _cfgs(replica, na, ns)
        celulas[ns] = {"cfg_tr": cfg_tr, "cfg_av": cfg_av, "ctx": _ctx(cfg_tr),
                       "ctx_av": _ctx(cfg_av), "bancos": carregar_bancos(na, ns)}
    primeira = celulas[nss[0]]
    corpus = (None if retomar is not None or len(nss) == 1 else
              np.concatenate([corpus_de_observacoes(c["cfg_tr"])
                              for c in celulas.values()]))
    ap, termo, acs = construir(primeira["cfg_tr"], na, t_res, retomar, c_max=c_max,
                               hp=hp, corpus=corpus)
    t_res = termo.n_bits // primeira["cfg_tr"].observation_size   # após retomar
    time_ctde = None
    if critico == "centralizado":
        # GUARD RAIL: o crítico central tem de ver TODOS os drones. A entrada é
        # Na·(bits da observação + bits da ação); qualquer outra coisa é um
        # crítico que não enxerga o time, e o treino degeneraria para o
        # descentralizado sem avisar.
        com_acoes = contrafactual > 0        # K = 0 → V_c(s); K > 0 → Q_c(s, a) (ver ctde.py)
        if retomar is not None and (Path(retomar) / CriticoCentral.NOME).exists():
            cc = CriticoCentral.carregar(Path(retomar), termo.n_bits, na, res_acao, com_acoes)
        else:
            cc = CriticoCentral(termo.n_bits, na, hp["tupla"], hp["phi"], seed=0,
                                resolucao_acao=res_acao, com_acoes=com_acoes)
        assert cc.n_bits == na * (termo.n_bits + cc.n_bits_acao), "crítico central não cobre o time"
        time_ctde = TimeAtorCritico(ap, acs[0].codificar, cc, na=na, sigma=hp["sigma"],
                                    passos=hp["m"], desconto=GAMMA, contrafactual=contrafactual)
    if retomar is not None:
        print(f"RETOMANDO de {retomar} (ator/crítico/termômetro carregados)",
              flush=True)
    rot_ns = _rotulo_ns(nss)
    print(f"multiagente · Na={na} Ns={rot_ns}"
          f"{' (revezando por episódio)' if len(nss) > 1 else ''} · T={t_res} · "
          f"N={ap.ator.n_ram} RAMs · B={termo.n_bits} bits · {na} trajetórias "
          f"compartilhando a rede · passo {hp['passo']:g} σ {hp['sigma']:g} "
          f"n {hp['tupla']} φ {hp['phi']:g} m {hp['m']}", flush=True)
    if time_ctde is not None:
        cc = time_ctde.critico
        print(f"  crítico CENTRALIZADO (CTDE): {'Q_c(s, a)' if cc.com_acoes else 'V_c(s)'} n-tuple com "
              f"{cc.n_bits} bits = {na}·({termo.n_bits} obs + {cc.n_bits_acao} ação) · "
              f"{cc.rede.n_ram} RAMs · contrafactual K = {contrafactual}"
              f"{' (δ compartilhado, sem crédito por drone)' if contrafactual == 0 else ' (vantagem por drone)'}",
              flush=True)

    handlers = list(__import__("logging").getLogger().handlers)
    val = {ns: c["bancos"].val.sementes[:val_n] for ns, c in celulas.items()}
    intervalo = max(1, b_max // 40)
    parada = ParadaAntecipada(maior_e_melhor=True)
    avisou_plato = False
    colapsos, colapsou = 0, None
    saida.mkdir(parents=True, exist_ok=True)
    melhor_dir = saida / "checkpoints" / "melhor"
    ultimo_dir = saida / "checkpoints" / "ultimo"
    config = {"na": na, "ns": nss[0] if len(nss) == 1 else list(nss),
              "passo": hp["passo"], "sigma": hp["sigma"], "T": t_res,
              "n": hp["tupla"], "phi": hp["phi"], "m": hp["m"], "gamma": GAMMA,
              "replica": replica, "parameter_sharing": True,
              "recompensa_por_na": True,
              "retomado_de": str(retomar) if retomar else None,
              "critico": ("descentralizado (Alg. 4.4)" if time_ctde is None else
                          f"centralizado (CTDE: Q(s,a) n-tuple, contrafactual K={contrafactual}, "
                          f"res_acao={res_acao})"),
              **{k: v for k, v in hp.items() if k not in HP_PADRAO}}

    def _salvar(dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        ap.salvar(dest)
        termo.salvar(dest / "termometro.npz")
        if time_ctde is not None:
            time_ctde.critico.salvar(dest)      # só nos checkpoints de treino

    passos, ep, melhor = 0, 0, -1.0
    ep_por_ns = {ns: 0 for ns in nss}
    if retomar is not None:
        # o `procedencia.json` fica na raiz da corrida, não em `checkpoints/final`
        base_ret = Path(retomar)
        proc_ant = next((c for c in (base_ret / "procedencia.json",
                                     base_ret.parent / "procedencia.json",
                                     base_ret.parent.parent / "procedencia.json")
                         if c.exists()), base_ret / "procedencia.json")
        if proc_ant.exists():
            d = json.loads(proc_ant.read_text())
            passos = int(d.get("passos", 0))
            ep = int(d.get("episodios", 0))
            melhor = float(d.get("melhor_val_cobertura", -1.0))
        if passos_previos is not None:
            passos = passos_previos
        print(f"retomando de {passos:,} passos, {ep:,} episódios "
              f"(melhor-val {melhor:.3f}) · alvo {b_max:,}", flush=True)
        if passos >= b_max:
            raise SystemExit(
                f"--b-max {b_max:,} já foi alcançado ({passos:,} passos gravados). "
                f"É ORÇAMENTO TOTAL, não incremento.")
    ultimo_prog = ultimo_eval = passos_iniciais = passos
    t0 = time.perf_counter()
    while passos < b_max:
        ns = nss[ep % len(nss)]                       # revezamento por episódio
        c = celulas[ns]
        cfg_tr, ctx = c["cfg_tr"], c["ctx"]
        passos_max = int(cfg_tr.max_episode_length
                         / cfg_tr.algorithm_iteration_interval) + 2
        s = SEED_TREINO_BASE + ep
        random.seed(s); np.random.seed(s)
        env = make_env(cfg_tr); obs, _ = env.reset()
        prune_logging_handlers(handlers)
        agentes = list(env.agents)
        if time_ctde is not None:
            time_ctde.reset([np.random.default_rng(s + 7919 * i) for i in range(len(agentes))])
        else:
            for i, _ in enumerate(agentes):
                acs[i].reset(ctx, np.random.default_rng(s + 7919 * i))
        antes = 0
        for _ in range(passos_max):
            if time_ctde is not None:
                # CTDE: as observações de TODOS entram juntas; o fechamento da
                # transição mais antiga (bootstrap Q(s', a')) acontece dentro de agir().
                lista = time_ctde.agir([obs[a] for a in agentes])
                acoes = {a: lista[i] for i, a in enumerate(agentes)}
            else:
                acoes = {a: acs[i].act(Observacao.decodificar(obs[a], ctx)).como_array()
                         for i, a in enumerate(agentes)}
            obs, _, term, trunc, _ = env.step(acoes)
            colhidos = int(env.sensors_collected)
            # Normaliza a recompensa de time por Na: cada drone recebe a fração
            # `rew/Na`, então o sinal TOTAL na rede compartilhada (Na drones ×
            # rew/Na) fica invariante a Na. Sem isto, o sinal escala com Na, os
            # retornos inflam e o δ estoura o c_max (medido: aborto em Na=2). É
            # escalar positivo → não muda a política ótima (como a escala (1−γ)).
            rew = escala * recompensa(colhidos, antes, cfg_tr.num_sensors) / len(agentes)
            antes = colhidos
            if time_ctde is not None:
                time_ctde.registrar(rew)
            else:
                for i in range(len(agentes)):       # CADA drone fecha sua transição
                    acs[i].registrar(rew)
            passos += 1                              # passo de AMBIENTE (todos os drones)
            if any(term.values()) or any(trunc.values()):
                break
            if time_ctde is None:
                for i, a in enumerate(agentes):      # e atualiza com o SEU próximo obs
                    acs[i].passo_intermediario(Observacao.decodificar(obs[a], ctx))
            if passos >= b_max:
                break
        if time_ctde is not None:
            time_ctde.fim_de_episodio()
        else:
            for ac in acs:
                ac.fim_de_episodio()
        ep += 1
        ep_por_ns[ns] += 1
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
        if passos - ultimo_prog >= progresso_cada:
            ultimo_prog = passos
            taxa = (passos - passos_iniciais) / max(1e-9, time.perf_counter() - t0)
            print(f"[{passos/b_max:5.1%}] {passos:,}/{b_max:,} · {ep:,} ep · "
                  f"{taxa:.0f} passos/s · ETA {(b_max-passos)/max(taxa,1)/3600:.1f} h", flush=True)
        # early-stop: avaliação periódica no val + checkpoint de melhor validação
        if passos - ultimo_eval >= intervalo:
            ultimo_eval = passos
            m = avaliar_celulas(celulas, ap, termo, val)
            eh_melhor = m["cobertura"] > melhor
            if eh_melhor:
                melhor = m["cobertura"]; _salvar(melhor_dir)
            por_ns_txt = ("  " + " ".join(f"s{k}:{v['cobertura']:.2f}" for k, v in
                                          m["por_ns"].items())
                          if "por_ns" in m else "")
            cred = time_ctde.estatisticas() if time_ctde is not None else None
            cred_txt = (f" · crédito disp {cred['dispersao_credito']:.2f}"
                        f" |A| {np.mean(cred['abs_vantagem']):.2f}") if cred else ""
            print(f"   eval @ {passos:,}: cob {m['cobertura']:.3f} · sr {m['sr']:.3f}"
                  f" · CT {m['ct_cens']:.1f} s · sep {m.get('separacao_m', float('nan')):.1f} m"
                  f"{cred_txt}{por_ns_txt}"
                  f"{'  (melhor)' if eh_melhor else ''}", flush=True)
            # A curva também em JSONL: o log é prosa, e uma corrida de 31 h
            # precisa ser plotável sem reparsear texto. `ocupadas` entra porque é
            # o que diz se a LUT ainda está crescendo depois do platô da cobertura.
            _salvar(ultimo_dir)
            (ultimo_dir / "procedencia.json").write_text(json.dumps({
                "passos": passos, "episodios": ep,
                "melhor_val_cobertura": melhor,
                "config": config,
                "parcial": True, "fonte": "spec06_treino_multiagente.py"}, indent=2))
            with (saida / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "passo": passos, "episodios": ep, "cobertura": m["cobertura"],
                    "sr": m["sr"], "ct_cens": m["ct_cens"], "n_val": m["n"],
                    "separacao_m": m.get("separacao_m"),
                    **({"por_ns": m["por_ns"]} if "por_ns" in m else {}),
                    **({"ocupadas_critico_central": time_ctde.critico.entradas_ocupadas(),
                        "credito": cred}
                       if time_ctde is not None else {}),
                    "ocupadas": sum(len(x) for x in ap.ator.memoria),
                    "parede_s": round(time.perf_counter() - t0, 1),
                    "melhor": bool(eh_melhor)}) + "\n")
            # GUARDA DE COLAPSO. Com `c_max` desligado nada mais aborta a corrida,
            # e as duas de `Na = 4` (passo 1,5e-3) mostraram o que isso custa:
            # pico a 1,25–2,5 M, cobertura a 3,5 % — o nível da caminhada
            # aleatória — a partir de 6 e 12 M, e 20–27 M de passos queimados até
            # a parada antecipada ler aquilo como "platô". Três avaliações
            # seguidas abaixo de um quarto do melhor, depois de o melhor ter
            # passado de 0,5, não é oscilação: é a política que divergiu e não
            # volta. O `melhor` fica; o `final` dessa corrida não é artefato.
            colapsos = (colapsos + 1
                        if (melhor > 0.5 and m["cobertura"] < melhor / 4) else 0)
            if guarda_colapso and colapsos >= 3:
                colapsou = passos
                print(f"   COLAPSO: cobertura {m['cobertura']:.3f} contra melhor "
                      f"{melhor:.3f} em 3 avaliações seguidas — encerrando em "
                      f"{passos:,} passos (--sem-guarda-colapso para seguir)",
                      flush=True)
                break
            plato = parada.registrar(m["cobertura"])
            if plato and not sem_parada:
                print(f"   parada antecipada (platô) em {passos:,} passos", flush=True)
                break
            if plato and not avisou_plato:
                avisou_plato = True
                print(f"   platô da cobertura em {passos:,} passos "
                      f"(--sem-parada: seguindo até {b_max:,})", flush=True)

    # final: salva o estado final e avalia o MELHOR checkpoint (fix 539b61e)
    _salvar(saida / "checkpoints" / "final")
    if not melhor_dir.exists():
        _salvar(melhor_dir)                     # parou antes da 1ª melhora
    ap.ator = type(ap.ator).carregar(next(melhor_dir.glob("*ator.npz")))
    ap.critico = type(ap.critico).carregar(next(melhor_dir.glob("*critico.npz")))
    termo_m = TermometroQuantil.carregar(melhor_dir / "termometro.npz")
    test = {ns: c["bancos"].test.sementes[:min(100, len(c["bancos"].test.sementes))]
            for ns, c in celulas.items()}
    metr = avaliar_celulas(celulas, ap, termo_m, test)
    proc = {"ator": {"n_saidas": D_SAIDA}, "resolucao": t_res,
            "config": config,
            "passos": passos, "episodios": ep,
            **({"episodios_por_ns": {str(k): v for k, v in ep_por_ns.items()}}
               if len(nss) > 1 else {}),
            "melhor_val_cobertura": melhor, "colapsou_em": colapsou,
            "metrica_test": metr, "fonte": "spec06_treino_multiagente.py"}
    # Poda pós-treino PADRÃO: `checkpoints/deploy` podado e verificado no VAL.
    # Nunca falha o treino — um erro aqui perde o artefato de deploy, não a
    # corrida — e nunca toca `final`/`melhor`. Com revezamento, a verificação
    # é no MAIOR `Ns`, o cenário mais difícil da lista.
    try:
        if colapsou is not None:
            raise RuntimeError(f"corrida colapsou em {colapsou:,} passos; o "
                               f"`final` não é artefato de implantação")
        from spec06_exportar_deploy import deploy_pos_treino
        ns_dep = max(nss)
        dep = deploy_pos_treino(saida / "checkpoints" / "final", na, ns_dep,
                                sementes=celulas[ns_dep]["bancos"].val.sementes[:100])
        proc["deploy"] = {k: dep[k] for k in
                          ("poda", "deploy_kb", "sr_cheio", "sr_podado")
                          if k in dep} | ({"motivo": dep["motivo"]}
                                          if "motivo" in dep else {})
        print(f"deploy podado ({dep.get('poda', 0):.0%}): "
              f"{dep['deploy_kb']:.0f} KB · SR {dep['sr_cheio']:.1%}"
              f"→{dep['sr_podado']:.1%} (val, Ns={ns_dep})"
              + (f" · {dep['motivo']}" if "motivo" in dep else ""), flush=True)
    except Exception as e:                                    # noqa: BLE001
        proc["deploy"] = {"erro": str(e)}
        print(f"aviso: deploy pós-treino falhou ({e}); final/melhor intactos",
              flush=True)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False))
    (melhor_dir / "procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False))
    print(f"\nNa={na} Ns={rot_ns}: SR {metr['sr']:.0%} · CT {metr['ct_cens']:.1f} s · "
          f"cobertura {metr['cobertura']:.1%} · melhor-val {melhor:.1%} "
          f"({metr['n']} sementes)"
          + (f" · COLAPSOU em {colapsou:,} passos — métricas do checkpoint "
             f"`melhor`, o `final` não serve" if colapsou is not None else ""))
    if "por_ns" in metr:
        for k, v in metr["por_ns"].items():
            print(f"   Ns={k}: SR {v['sr']:.0%} · CT {v['ct_cens']:.1f} s · "
                  f"cobertura {v['cobertura']:.1%}")
    print(f"checkpoint (melhor): {melhor_dir}")
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--na", type=int, required=True)
    ap.add_argument("--ns", default="24",
                    help="número de sensores, ou LISTA (`12,24,36`) para revezar "
                         "por episódio com uma rede só — o regime da grade com "
                         "`--ns` em lista")
    ap.add_argument("--b-max", type=int, default=10_000_000)
    ap.add_argument("--replica", default="1b")
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--progresso-cada", type=int, default=50_000)
    ap.add_argument("--resolucao", type=int, default=T_RES,
                    help=f"T do termômetro; {T_RES} (padrão) ou 16 (econômico). "
                         "Ignorado com --retomar: T vem do checkpoint.")
    ap.add_argument("--config-de", type=Path, default=None,
                    help="artefato JSON de `spec06_busca.py`: usa o melhor ponto "
                         "DAQUELA grade com T = --resolucao (passo, σ, n, φ, m). "
                         "Flags explícitas abaixo têm precedência.")
    ap.add_argument("--passo", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--tupla", type=int, default=None)
    ap.add_argument("--phi", type=float, default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--retomar", type=Path, default=None,
                    help="dir de checkpoint (ex.: models/T32/em-treino/a8s24) para "
                         "continuar o treino em vez de recomeçar do zero")
    ap.add_argument("--c-max", default="0.5",
                    help="teto do passo efetivo (asserção TreinoInstavel da "
                         "§4.2). 'none' desliga — usado para retomar o "
                         "T32-a2s24, que caiu num pico isolado de δ a 40 M e "
                         "reproduziria o mesmo crash pela mesma semente.")
    ap.add_argument("--val-n", type=int, default=20,
                    help="sementes do banco val por avaliação (máx. 100). Com 20, "
                         "a SR tem resolução de 5 pp e um único episódio move a "
                         "curva — pouco para decidir se os passos além do platô "
                         "da cobertura ainda compram SR. 100 custa 0,65 %% do "
                         "orçamento de 50 M.")
    ap.add_argument("--passos-previos", type=int, default=None,
                    help="passos já treinados, quando o checkpoint de --retomar "
                         "não tem procedencia.json (ex.: os do grid). --b-max é "
                         "ORÇAMENTO TOTAL, então isto desconta do que falta.")
    ap.add_argument("--sem-parada", action="store_true",
                    help="ignora a parada antecipada da §4.4 e gasta --b-max "
                         "inteiro; a avaliação periódica continua rodando")
    ap.add_argument("--critico", choices=("descentralizado", "centralizado"),
                    default="descentralizado",
                    help="`centralizado` = CTDE: Q(s, a) n-tuple sobre observações e "
                         "ações de TODOS os drones, só no treino; o ator e o deploy "
                         "não mudam (ver uavwnn/agents/ctde.py)")
    ap.add_argument("--contrafactual", type=int, default=0,
                    help="K > 0: vantagem contrafactual por drone (COMA-lite) com K "
                         "amostras da própria política; 0 = δ compartilhado")
    ap.add_argument("--res-acao", type=int, default=16,
                    help="níveis do termômetro de faixa fixa da ação no crítico central")
    ap.add_argument("--sem-guarda-colapso", action="store_true",
                    help="não encerra a corrida quando a cobertura de validação "
                         "cai a menos de 1/4 do melhor por 3 avaliações seguidas")
    ap.add_argument("--smoke", action="store_true", help="b-max=5000, avaliação curta")
    a = ap.parse_args()
    if a.smoke:
        a.b_max = min(a.b_max, 5000); a.progresso_cada = 1000
    nss = _ler_ns(a.ns)
    if not nss or min(nss) < 1:
        ap.error(f"--ns inválido: {a.ns!r}")

    hp: dict = {}
    if a.config_de is not None:
        hp = config_da_grade(a.config_de, a.resolucao)
        print(f"config da grade {a.config_de} (T={a.resolucao}, ponto "
              f"{hp['id_grade']}): passo {hp['passo']:g} σ {hp['sigma']:g} "
              f"n {hp['tupla']} φ {hp['phi']:g} m {hp['m']} · cobertura "
              f"{hp['cobertura_grade']:.1%} SR {hp['sr_grade']:.1%} abortos "
              f"{hp['abortos_grade']}", flush=True)
        hp.pop("resolucao")
    for k in ("passo", "sigma", "tupla", "phi", "m"):
        if getattr(a, k) is not None:
            hp[k] = getattr(a, k)
    hp_ef = {**HP_PADRAO, **hp}

    # O sufixo só aparece fora do padrão: as células de `T = 32` já publicadas
    # mantêm o caminho que `modelo-e-hiperparametros.md` registra.
    sufixo = "" if a.resolucao == T_RES else f"-T{a.resolucao}"
    if a.critico == "centralizado":
        sufixo += f"-ctde{a.contrafactual}"
    saida = a.saida or Path(
        f"runs/spec06-multi/ntupla_dissertacao-a{a.na}s{_rotulo_ns(nss)}{sufixo}")
    print(f"SPEC-06 escalada multiagente · "
          f"{'config da grade' if hp else 'config vencedora (T-06.4)'} "
          f"(passo {hp_ef['passo']:g}, σ {hp_ef['sigma']:g}, T {a.resolucao}, "
          f"n {hp_ef['tupla']}, φ {hp_ef['phi']:g}, m {hp_ef['m']})\n", flush=True)
    treinar(a.na, nss, a.b_max, a.replica, saida, a.progresso_cada,
            a.resolucao, a.retomar, a.sem_parada, a.passos_previos, a.val_n,
            None if str(a.c_max).lower() == "none" else float(a.c_max), hp=hp,
            guarda_colapso=not a.sem_guarda_colapso, critico=a.critico,
            contrafactual=a.contrafactual, res_acao=a.res_acao)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
