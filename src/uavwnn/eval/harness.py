"""Harness de avaliação — SPEC-02 T-02.8 e T-02.9.

**O harness não distingue baseline de política aprendida.** Recebe qualquer
`Policy`, roda os mesmos cenários para todas, aplica a mesma censura e o mesmo
bootstrap, e produz a mesma tabela. É o que torna a comparação de H1 e H6
possível sem caminhos separados por método.

TRÊS INVARIANTES QUE O HARNESS APLICA, NÃO SUPÕE

1. **Privilégio é declarado.** `observar_verdade` só é chamado para políticas com
   `usa_estado_verdadeiro = True`. É a RF-04 antecipada aqui.
2. **Pareamento.** Todas as políticas veem **exatamente** os mesmos
   `seed_cenario`, vindos do ramo `avaliacao` da árvore de sementes.
3. **LB-1 é limite.** Resultado abaixo dele é bug, e há asserção.

DESVIO DECLARADO EM RELAÇÃO À SPEC

A T-02.8 pede `episodios.parquet`. Usamos **CSV**, para não acrescentar `pyarrow`
como dependência por causa de um arquivo. O grão, as colunas e a exclusão do git
são os mesmos; só o formato muda. Se o volume crescer ao ponto de importar, a
troca é local a duas funções.

LAÇO PRÓPRIO, E POR QUÊ

Não reusamos `refenv.run_episode` porque ele expõe só `(passo, obs, recompensa)`
ao callback, e o harness precisa de mais: posições verdadeiras por passo (para as
privilegiadas e para o LB-1), visitas por agente, coletas redundantes e distância
percorrida. A T-02.1 já previa isso — *"contrato requisitado à SPEC-00"*. O laço
aqui é o mesmo do `run_episode`, incluindo a semeadura com `random.seed` antes do
`reset()` (o `reset()` de referência **ignora** o `seed` que declara) e a parada
na **primeira** `info` não vazia.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..agents.base import (
    ContextoEpisodio,
    EstadoVerdadeiro,
    Observacao,
    Policy,
)
from ..refenv import EnvConfig, make_env, prune_logging_handlers
from .metricas import MetricasEpisodio, lb1, resumir_episodios
from .sementes import ArvoreSementes
from .stats import bootstrap_media, bootstrap_pareado, mcnemar


def contexto_de(cfg: EnvConfig) -> ContextoEpisodio:
    """`ContextoEpisodio` a partir da configuração do ambiente.

    São parâmetros do cenário (tab. 3.1 e 7.1), não estado — a política pode
    saber a constante de normalização sem que isso seja privilégio.
    """
    return ContextoEpisodio(
        num_drones=cfg.num_drones,
        num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval,
        max_episode_length=cfg.max_episode_length,
        state_num_closest_sensors=cfg.state_num_closest_sensors,
    )


def _verdade(env) -> EstadoVerdadeiro:
    return EstadoVerdadeiro(
        pos_sensores=np.array(
            [env.simulator.get_node(i).position[:2] for i in env.sensor_node_ids]),
        pos_agentes=np.array(
            [env.simulator.get_node(i).position[:2] for i in env.agent_node_ids]),
        coletado=np.array(
            [env.simulator.get_node(i).protocol_encapsulator.protocol.has_collected
             for i in env.sensor_node_ids], dtype=bool),
    )


def rodar_episodio(
    politica: Policy,
    cfg: EnvConfig,
    seed_cenario: int,
    idx_episodio: int,
    rng_politica: np.random.Generator,
) -> MetricasEpisodio:
    """Um episódio, um método. Devolve o grão de `episodios.csv`."""
    ctx = contexto_de(cfg)
    handlers = list(logging.getLogger().handlers)
    env = make_env(cfg)

    politica.reset(ctx, rng_politica)
    privilegiada = bool(getattr(politica, "usa_estado_verdadeiro", False))

    random.seed(seed_cenario)
    np.random.seed(seed_cenario)
    obs, _ = env.reset()
    prune_logging_handlers(handlers)

    v0 = _verdade(env)
    lb = lb1(v0.pos_sensores, v0.pos_agentes, cfg.communication_range, ctx.v_max)

    pos_ant = v0.pos_agentes.copy()
    dist = np.zeros(cfg.num_drones)
    visitas = np.zeros(cfg.num_drones, dtype=int)
    redundantes = 0
    coletado_ant = v0.coletado.copy()

    max_passos = int(cfg.max_episode_length / ctx.dt) + 2
    passos, t, info = 0, 0.0, {}

    try:
        for _ in range(max_passos):
            estado = _verdade(env)
            if privilegiada:
                politica.observar_verdade(estado)     # invariante 1

            acoes = {}
            for nome in env.agents:
                o = Observacao.decodificar(obs[nome], ctx, t)
                acoes[nome] = politica.act(o).como_array()

            obs, _r, term, trunc, infos = env.step(acoes)
            passos += 1
            t += ctx.dt

            depois = _verdade(env)
            dist += np.linalg.norm(depois.pos_agentes - pos_ant, axis=1)
            pos_ant = depois.pos_agentes.copy()

            # Atribuição de visita: quem estava no alcance quando o sensor virou
            # coletado. Dois ou mais no alcance = coleta redundante, que é a
            # assinatura quantitativa do "todos vão para o mesmo sensor".
            novos = np.flatnonzero(depois.coletado & ~coletado_ant)
            for s in novos:
                d = np.linalg.norm(depois.pos_agentes - depois.pos_sensores[s], axis=1)
                no_alcance = np.flatnonzero(d <= cfg.communication_range)
                if no_alcance.size == 0:
                    no_alcance = np.array([int(np.argmin(d))])
                visitas[no_alcance[0]] += 1
                if no_alcance.size > 1:
                    redundantes += 1
            coletado_ant = depois.coletado.copy()

            if infos and "avg_reward" in next(iter(infos.values())):
                info = next(iter(infos.values()))
                break
            if any(term.values()) or any(trunc.values()):
                break
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
        prune_logging_handlers(handlers)

    sucesso = bool(info.get("all_collected", False))
    t_conc = float(info.get("completion_time", float("nan")))
    n_vis = int(coletado_ant.sum())
    motivo = ("sucesso" if sucesso
              else info.get("cause", "timeout" if passos >= max_passos else "stalled"))

    ep = MetricasEpisodio(
        idx_episodio=idx_episodio, seed_cenario=seed_cenario, sucesso=sucesso,
        n_visitados=n_vis, n_sensores=cfg.num_sensors,
        t_conc_s=t_conc, t_fim_s=t, motivo_fim=str(motivo), n_decisoes=passos,
        dist_por_agente_m=[float(x) for x in dist],
        visitas_por_agente=[int(x) for x in visitas],
        coletas_redundantes=redundantes, lb1_s=lb,
        t_plan_s=float(np.mean(list(
            politica.metadados().get("t_plan_por_agente", {}).values()) or [np.nan])),
    )

    # Invariante 3: LB-1 é limite. Abaixo dele é bug, não descoberta.
    if ep.sucesso and ep.t_conc_s < lb - 1e-6:
        raise AssertionError(
            f"t_conc={ep.t_conc_s:.3f}s < LB-1={lb:.3f}s na semente "
            f"{seed_cenario} — limite violado, logo há bug no LB-1 ou no laço")
    return ep


# --------------------------------------------------------------------------
# campanha
# --------------------------------------------------------------------------


@dataclass
class Resultado:
    celula: str
    por_politica: dict[str, list[MetricasEpisodio]]
    sementes: list[int]


def avaliar(
    politicas: dict[str, Policy],
    cfg: EnvConfig,
    arvore: ArvoreSementes,
    n_episodios: int,
    verboso: bool = True,
) -> Resultado:
    """Roda todas as políticas nos **mesmos** cenários. Invariante 2."""
    sementes = arvore.cenarios("avaliacao", n_episodios)
    celula = f"Na{cfg.num_drones}_Ns{cfg.num_sensors}"
    fora: dict[str, list[MetricasEpisodio]] = {}

    for pid, pol in politicas.items():
        t0 = time.perf_counter()
        eps = [
            rodar_episodio(pol, cfg, s, i, arvore.politica(pid, i))
            for i, s in enumerate(sementes)
        ]
        fora[pid] = eps
        if verboso:
            r = resumir_episodios(eps)
            print(f"  {pid:>16} {r.media_t_conc_pen:>8.2f}s  "
                  f"sucesso {r.taxa_sucesso:>6.1%}  "
                  f"LB-1 {r.media_lb1:>6.2f}s  "
                  f"({time.perf_counter() - t0:.0f}s)")
    return Resultado(celula=celula, por_politica=fora, sementes=sementes)


def comparacoes(
    res: Resultado,
    arvore: ArvoreSementes,
    pares: Sequence[tuple[str, str]],
    primarias: Sequence[tuple[str, str]] = (),
) -> list[dict]:
    """Uma linha por par (A, B), no subconjunto `todos` e em `sucesso_comum`."""
    linhas = []
    for a, b in pares:
        ea, eb = res.por_politica[a], res.por_politica[b]
        pa = np.array([e.t_conc_pen_s for e in ea])
        pb = np.array([e.t_conc_pen_s for e in eb])
        sa = np.array([e.sucesso for e in ea])
        sb = np.array([e.sucesso for e in eb])

        for subconjunto in ("todos", "sucesso_comum"):
            if subconjunto == "todos":
                m = np.ones(len(pa), bool)
            else:
                m = sa & sb
                if m.sum() < 10:
                    continue          # degenera; a T-02.5 manda não reportar
            rng = arvore.bootstrap(f"{res.celula}-{a}-{b}-{subconjunto}")
            d = bootstrap_pareado(pa[m], pb[m], rng)
            mc = mcnemar(sa[m], sb[m])
            linhas.append({
                "celula": res.celula, "politica_a": a, "politica_b": b,
                "subconjunto": subconjunto,
                **d.como_linha(), **mc,
                "primaria": (a, b) in primarias or (b, a) in primarias,
            })
    return linhas


def agregado(res: Resultado, arvore: ArvoreSementes) -> list[dict]:
    linhas = []
    for pid, eps in res.por_politica.items():
        r = resumir_episodios(eps)
        rng = arvore.bootstrap(f"{res.celula}-{pid}")
        m = bootstrap_media([e.t_conc_pen_s for e in eps], rng)
        ok = np.array([e.sucesso for e in eps], float)
        ts = bootstrap_media(ok, arvore.bootstrap(f"{res.celula}-{pid}-ts"))
        gini = float(np.nanmean([e.gini_visitas for e in eps]))
        linhas.append({
            "celula": res.celula, "politica": pid, **r.como_linha(),
            "ic_lo": m.ic_lo, "ic_hi": m.ic_hi,
            "ts_ic_lo": ts.ic_lo, "ts_ic_hi": ts.ic_hi,
            "gini_visitas": gini,
            "coletas_redundantes": float(np.mean(
                [e.coletas_redundantes for e in eps])),
        })
    return linhas


# --------------------------------------------------------------------------
# artefatos
# --------------------------------------------------------------------------


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip() or "?"
    except Exception:  # pragma: no cover
        return "?"


def escrever_csv(caminho: Path, linhas: Iterable[dict]) -> None:
    linhas = list(linhas)
    if not linhas:
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


def escrever_episodios(caminho: Path, res: Resultado, cfg: EnvConfig,
                       politicas: dict[str, Policy]) -> None:
    """Grão fino, **fora do git**. Nenhuma coluna de tempo de parede aqui: o
    determinismo exige que duas execuções produzam o arquivo idêntico, e latência
    vive em `custo.parquet` sob protocolo da SPEC-03."""
    linhas = []
    for pid, eps in res.por_politica.items():
        priv = bool(getattr(politicas[pid], "usa_estado_verdadeiro", False))
        for e in eps:
            d = asdict(e)
            d.update({
                "celula": res.celula, "Na": cfg.num_drones, "Ns": cfg.num_sensors,
                "politica": pid, "usa_estado_verdadeiro": priv,
                "t_conc_pen_s": e.t_conc_pen_s,
                "gini_visitas": e.gini_visitas,
                "communication_range": cfg.communication_range,
                "dt": cfg.algorithm_iteration_interval,
                "v_max": contexto_de(cfg).v_max,
                "max_episode_length": cfg.max_episode_length,
                "max_seconds_stalled": cfg.max_seconds_stalled,
                "sigma_gps": float("nan"), "degrau_fusao": "n/a",
                "git_commit": _commit(),
            })
            d["dist_por_agente_m"] = json.dumps(
                [round(x, 4) for x in d["dist_por_agente_m"]])
            d["visitas_por_agente"] = json.dumps(d["visitas_por_agente"])
            linhas.append(d)
    escrever_csv(caminho, linhas)


def escrever_manifesto(caminho: Path, arvore: ArvoreSementes, cfg: EnvConfig,
                       politicas: dict[str, Policy], n_episodios: int,
                       parede_s: float) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w") as f:
        json.dump({
            "raiz_sementes": arvore.raiz,
            "n_episodios": n_episodios,
            "config_env": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                               else str(v)) for k, v in vars(cfg).items()},
            "politicas": {pid: p.metadados() for pid, p in politicas.items()},
            "git_commit": _commit(),
            "parede_s": round(parede_s, 2),
            "modo": "desempenho",
            "nota_formato": "episodios em CSV, não parquet — desvio declarado "
                            "no docstring de eval/harness.py",
        }, f, indent=2, default=str)
