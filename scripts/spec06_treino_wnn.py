"""SPEC-06 — treino da WNN (Algoritmo 4.4) no ambiente real, uma configuração.

Modelo provisório escolhido enquanto o grid search não fecha: a rede da
dissertação (`ntupla` — gradiente funcional §4.3.1 + NTRN com esquecimento
§4.2.1), com

    passo (taxa do ator) = 5e-4      σ = 0,2        T = 32
    n (tupla) = 12  →  N = ⌈B/n⌉      φ (crítico) = 0,85
    m = 8            γ = 0,99         d_saída = 3 (normal projetada)

na célula `Na = 1, Ns = 12` (B = 25·32 = 800, logo N = 67).

**Orçamento de Lamenza:** `benchmark.py` treina com `--total-timesteps=50000000`
= **50 M passos** por célula (~7 dias de GPU no original). Para `Na=1, Ns=12`
isso é ≈ 730 mil episódios (68,5 passos/ep). Este script treina com o **mesmo
número de passos**, salva o checkpoint no formato que `spec06_campanha_wnn.py`
consome, e faz checkpoints periódicos (retomáveis) porque é uma corrida longa.

    uv run python scripts/spec06_treino_wnn.py --saida reports/ckpt/ntupla-a1s12
    uv run python scripts/spec06_treino_wnn.py --passos-totais 5000000        # mais curto
    uv run python scripts/spec06_treino_wnn.py --retomar --saida reports/ckpt/ntupla-a1s12

Fase **train** (`end_when_all_collected=False`), o regime do `benchmark.py` — o
episódio segue após a coleta completa (o "longer duration positive reward
signal" de Lamenza §5), podado só por `max_seconds_stalled`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from uavwnn.agents import APROXIMADORES, AtorCriticoContinuo, CaminhadaAleatoria
from uavwnn.agents.base import ContextoEpisodio, Observacao
from uavwnn.agents.politica_continua import NormalProjetada
from uavwnn.encoding import TermometroQuantil
from uavwnn.refenv import LAMENZA_003, make_env, prune_logging_handlers

SEED_TREINO_BASE = 1_000_000   # disjunto das sementes de avaliação (0..499)


@dataclass
class ConfigTreino:
    num_drones: int = 1
    num_sensores: int = 12
    resolucao: int = 32          # T
    tamanho_tupla: int = 12      # n
    phi: float = 0.85
    taxa_ator: float = 5e-4      # o "passo"
    sigma: float = 0.20
    passos_m: int = 8            # m
    desconto: float = 0.99       # γ
    d_saida: int = 3
    passos_totais: int = 50_000_000   # o número de Lamenza
    seed: int = 0


def _ctx(cfg_env) -> ContextoEpisodio:
    return ContextoEpisodio(
        num_drones=cfg_env.num_drones, num_sensores=cfg_env.num_sensors,
        scenario_size=cfg_env.scenario_size,
        communication_range=cfg_env.communication_range,
        dt=cfg_env.algorithm_iteration_interval)


def ajustar_termometro(cfg_env, resolucao: int, seed: int, n_ep: int = 5) -> TermometroQuantil:
    """Ajusta o termômetro quantil num corpus de caminhada aleatória.

    Os limiares são estado treinado e viajam com o modelo (viram parte do
    checkpoint). Reajustá-los depois num corpus diferente endereçaria outras
    posições de memória — erro silencioso. Aqui é ajustado UMA vez, antes do
    treino, e congelado.
    """
    from uavwnn.refenv import run_episode
    ctx = _ctx(cfg_env)
    corpus: list[np.ndarray] = []
    aleat = CaminhadaAleatoria()
    aleat.reset(ctx, np.random.default_rng(seed))

    def coletar(observacao):
        corpus.append(np.asarray(observacao, float).ravel())
        return aleat.act(Observacao.decodificar(observacao, ctx)).como_array()

    for s in range(n_ep):
        run_episode(cfg_env, coletar, seed=20_000 + s)
    return TermometroQuantil(resolucao).ajustar(np.array(corpus))


def construir(cfg: ConfigTreino, cfg_env, termo: TermometroQuantil):
    """Monta o aproximador `ntupla` e o laço AC contínuo (Algoritmo 4.4)."""
    ap = APROXIMADORES["ntupla_dissertacao"](
        n_bits=termo.n_bits, tamanho_tupla=cfg.tamanho_tupla, phi=cfg.phi,
        taxa_ator=cfg.taxa_ator, seed=cfg.seed, d_saida=cfg.d_saida,
        n_dims_codificador=cfg_env.observation_size, limiares_por_dim=cfg.resolucao)

    def codificar(v):
        return termo.codificar(np.asarray(v, float)[None, :])[0]

    politica = AtorCriticoContinuo(
        aproximador=ap, codificar=codificar, id="wnn_ntupla",
        politica=NormalProjetada(sigma=cfg.sigma),
        passos=cfg.passos_m, desconto=cfg.desconto, treinando=True)
    return politica, ap


def salvar_checkpoint(saida: Path, ap, termo, cfg: ConfigTreino, passos: int,
                      episodios: int, metrica: dict) -> None:
    """Grava no formato que `spec06_campanha_wnn.py` consome (ap + termômetro +
    procedencia.json)."""
    saida.mkdir(parents=True, exist_ok=True)
    ap.salvar(saida)
    termo.salvar(saida / "termometro.npz")
    proc = {
        "ator": {"n_saidas": cfg.d_saida},   # o campanha lê isto para montar a rede
        "resolucao": cfg.resolucao,          # e isto para o codificador
        "config": asdict(cfg),
        "passos_feitos": passos,
        "episodios": episodios,
        "metrica_recente": metrica,
        "fonte": "spec06_treino_wnn.py (Algoritmo 4.4, ambiente real, fase train)",
    }
    (saida / "procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False))


def treinar(cfg: ConfigTreino, saida: Path, checkpoint_cada: int, retomar: bool,
            progresso_cada: int = 50_000) -> None:
    cfg_env = LAMENZA_003(cfg.num_drones, cfg.num_sensores, phase="train")
    ctx = _ctx(cfg_env)
    rng = np.random.default_rng(cfg.seed)

    # -- termômetro + aproximador (novo ou retomado) -----------------------
    passos_feitos, ep = 0, 0
    if retomar and (saida / "procedencia.json").exists():
        proc = json.loads((saida / "procedencia.json").read_text())
        termo = TermometroQuantil.carregar(saida / "termometro.npz")
        politica, ap = construir(cfg, cfg_env, termo)
        ap.carregar(saida)                    # substitui redes pelas gravadas
        passos_feitos = int(proc.get("passos_feitos", 0))
        ep = int(proc.get("episodios", 0))
        print(f"retomando de {passos_feitos:,} passos, {ep:,} episódios")
    else:
        print(f"ajustando o termômetro (T={cfg.resolucao}) num corpus aleatório...")
        termo = ajustar_termometro(cfg_env, cfg.resolucao, cfg.seed)
        politica, ap = construir(cfg, cfg_env, termo)
        print(f"B = {termo.n_bits} bits, N = {ap.ator.n_ram} RAMs, n = {cfg.tamanho_tupla}")

    politica.reset(ctx, rng)

    n_ep_lamenza = cfg.passos_totais // 68  # estimativa só para a mensagem
    print(f"\nAlvo: {cfg.passos_totais:,} passos (≈ {n_ep_lamenza:,} episódios) — "
          f"o orçamento de Lamenza para esta célula.\n")

    # -- laço de treino ----------------------------------------------------
    janela = deque(maxlen=500)   # (completion_time, sucesso) recentes
    handlers = list(logging.getLogger().handlers)
    env = make_env(cfg_env)
    t0 = time.perf_counter()
    ultimo_ckpt = passos_feitos
    ultimo_prog = passos_feitos
    agente = env.agents[0] if hasattr(env, "agents") and env.agents else "drone0"

    try:
        while passos_feitos < cfg.passos_totais:
            seed_ep = SEED_TREINO_BASE + ep
            random.seed(seed_ep)
            np.random.seed(seed_ep)
            obs, _ = env.reset()
            prune_logging_handlers(handlers)
            agente = next(iter(obs))
            o = Observacao.decodificar(obs[agente], ctx)
            info: dict = {}

            while True:
                acao = politica.act(o)
                obs, rewards, term, trunc, infos = env.step({agente: acao.como_array()})
                politica.registrar(float(next(iter(rewards.values()))) if rewards else 0.0)
                o = Observacao.decodificar(obs[agente], ctx)
                politica.passo_intermediario(o)
                passos_feitos += 1
                if infos and "avg_reward" in next(iter(infos.values())):
                    info = next(iter(infos.values()))
                    break
                if any(term.values()) or any(trunc.values()):
                    break
                if passos_feitos >= cfg.passos_totais:
                    break
            politica.fim_de_episodio()
            janela.append((float(info.get("completion_time", cfg_env.max_episode_length)),
                           bool(info.get("all_collected", False))))
            ep += 1

            if passos_feitos - ultimo_prog >= progresso_cada:
                ultimo_prog = passos_feitos
                _reportar(cfg, passos_feitos, ep, janela, t0)
            if passos_feitos - ultimo_ckpt >= checkpoint_cada:
                ultimo_ckpt = passos_feitos
                salvar_checkpoint(saida, ap, termo, cfg, passos_feitos, ep, _metrica(janela))
                print(f"  ↳ checkpoint salvo em {saida} ({passos_feitos:,} passos)")
    except KeyboardInterrupt:
        print("\ninterrompido — salvando o checkpoint atual...")
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
        prune_logging_handlers(handlers)

    salvar_checkpoint(saida, ap, termo, cfg, passos_feitos, ep, _metrica(janela))
    _reportar(cfg, passos_feitos, ep, janela, t0)
    print(f"\ncheckpoint final: {saida}")
    print(f"  ator: {ap.ator.entradas_ocupadas():,} entradas · "
          f"crítico: {ap.critico.entradas_ocupadas():,}")


def _metrica(janela) -> dict:
    if not janela:
        return {}
    ct = np.array([c for c, _ in janela])
    ok = np.array([s for _, s in janela])
    return {"completion_time_medio": float(ct.mean()),
            "taxa_sucesso": float(ok.mean()), "n_janela": len(janela)}


def _reportar(cfg, passos, ep, janela, t0) -> None:
    dt = time.perf_counter() - t0
    taxa = passos / dt if dt > 0 else 0.0
    restam = (cfg.passos_totais - passos) / taxa if taxa > 0 else float("inf")
    m = _metrica(janela)
    frac = passos / cfg.passos_totais
    print(f"[{frac:5.1%}] {passos:,}/{cfg.passos_totais:,} passos · {ep:,} ep · "
          f"{taxa:.0f} passos/s · ETA {restam/3600:.1f} h · "
          f"CT {m.get('completion_time_medio', float('nan')):.1f} s · "
          f"sucesso {m.get('taxa_sucesso', float('nan')):.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saida", type=Path, default=Path("reports/ckpt/ntupla-a1s12"))
    ap.add_argument("--passos-totais", type=int, default=50_000_000,
                    help="orçamento de Lamenza = 50M; reduza para testar")
    ap.add_argument("--checkpoint-cada", type=int, default=2_000_000)
    ap.add_argument("--progresso-cada", type=int, default=50_000,
                    help="passos entre linhas de progresso no log (~3 min cada)")
    ap.add_argument("--retomar", action="store_true", help="continua de um checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = ConfigTreino(passos_totais=a.passos_totais, seed=a.seed)
    print("SPEC-06 — treino da WNN (ntupla, Algoritmo 4.4), célula Na=1 Ns=12")
    print(f"  passo={cfg.taxa_ator}  σ={cfg.sigma}  T={cfg.resolucao}  "
          f"n={cfg.tamanho_tupla}  φ={cfg.phi}  m={cfg.passos_m}  γ={cfg.desconto}\n")
    treinar(cfg, a.saida, a.checkpoint_cada, a.retomar, a.progresso_cada)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
