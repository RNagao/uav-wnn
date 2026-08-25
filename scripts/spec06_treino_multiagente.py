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
CENTRALIZADO (vê obs+ações de todos). Este AC tem crítico DESCENTRALIZADO on-policy
(Algoritmo 4.4). O **ator implantado** é descentralizado e compartilhado nos dois,
logo comparável no deploy; os algoritmos de treino diferem, e a comparação
WNN×MADDPG é cross-método (a comparação "mesmo algoritmo" do projeto é WNN-AC vs
MLP-AC).

    uv run python scripts/spec06_treino_multiagente.py --na 2 --ns 24 --smoke
    uv run python scripts/spec06_treino_multiagente.py --na 2 --ns 24 --b-max 10000000
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


def construir(cfg, na: int, t_res: int = T_RES, retomar: Path | None = None):
    """Rede compartilhada + `na` trajetórias de AC (uma por drone).

    Se ``retomar`` for um dir de checkpoint, carrega ator/crítico/termômetro dali
    (em vez de sortear rede nova e reajustar o termômetro) e **continua** o treino
    de onde parou — usado para retomar um parcial em outra máquina.

    Ao retomar, `T` vem do **termômetro gravado**, não do argumento: os limiares
    são estado treinado, e montar a rede com um `T` diferente do que gerou os
    endereços daria uma rede que lê as LUTs nos lugares errados — o mesmo defeito
    que o `carregar` do `RedeGradienteFuncional` corrige para o mapeamento.
    """
    d_obs = cfg.observation_size
    if retomar is not None:
        termo = TermometroQuantil.carregar(Path(retomar) / "termometro.npz")
        gravado = termo.n_bits // d_obs
        if gravado != t_res:
            print(f"  T do checkpoint = {gravado} (o --resolucao {t_res} foi "
                  f"ignorado: os limiares são estado treinado)", flush=True)
        t_res = gravado
    else:
        termo = TermometroQuantil(t_res).ajustar(corpus_de_observacoes(cfg))
    ap = APROXIMADORES["ntupla_dissertacao"](
        n_bits=termo.n_bits, tamanho_tupla=N_TUPLA, phi=PHI, taxa_ator=PASSO,
        seed=0, d_saida=D_SAIDA, n_dims_codificador=d_obs, limiares_por_dim=t_res)
    if retomar is not None:
        rp = Path(retomar)
        ap.ator = type(ap.ator).carregar(next(rp.glob("*ator.npz")))
        ap.critico = type(ap.critico).carregar(next(rp.glob("*critico.npz")))

    def codificar(v, _t=termo):
        return _t.codificar(np.asarray(v, float)[None, :])[0]

    acs = [AtorCriticoContinuo(
        aproximador=ap, codificar=codificar, politica=NormalProjetada(SIGMA, None),
        passos=M, desconto=GAMMA, id=f"wnn_a{i}", treinando=True) for i in range(na)]
    return ap, termo, acs


def avaliar_multi(cfg, ap, termo, ctx, sementes) -> dict:
    """Avaliação determinística multiagente: o ator compartilhado por drone."""
    def codificar(v, _t=termo):
        return _t.codificar(np.asarray(v, float)[None, :])[0]
    aval = AtorCriticoContinuo(aproximador=ap, codificar=codificar,
                               politica=NormalProjetada(SIGMA, None), passos=M,
                               desconto=GAMMA, id="wnn_eval", treinando=False)
    cts, oks, cov = [], [], []
    for s in sementes:
        random.seed(int(s)); np.random.seed(int(s))
        env = make_env(cfg); obs, _ = env.reset()
        ct, ok = float(cfg.max_episode_length), False
        for _ in range(int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2):
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
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return {"sr": float(np.mean(oks)), "ct_cens": float(np.mean(cts)),
            "cobertura": float(np.mean(cov)), "n": len(cts)}


def treinar(na: int, ns: int, b_max: int, replica: str, saida: Path,
            progresso_cada: int, t_res: int = T_RES,
            retomar: Path | None = None) -> dict:
    cfg_tr, cfg_av = _cfgs(replica, na, ns)
    ctx = _ctx(cfg_tr)
    r = REPLICAS[replica]
    escala = r["escala_recompensa"]
    recompensa = RECOMPENSAS[r["recompensa"]]
    ap, termo, acs = construir(cfg_tr, na, t_res, retomar)
    t_res = termo.n_bits // cfg_tr.observation_size     # o efetivo, após retomar
    if retomar is not None:
        print(f"RETOMANDO de {retomar} (ator/crítico/termômetro carregados)",
              flush=True)
    print(f"multiagente · Na={na} Ns={ns} · T={t_res} · N={ap.ator.n_ram} RAMs · "
          f"B={termo.n_bits} bits · {na} trajetórias compartilhando a rede",
          flush=True)

    passos_max = int(cfg_tr.max_episode_length / cfg_tr.algorithm_iteration_interval) + 2
    handlers = list(__import__("logging").getLogger().handlers)
    ctx_av = _ctx(cfg_av)
    val = carregar_bancos(na, ns).val.sementes[:20]
    intervalo = max(1, b_max // 40)
    parada = ParadaAntecipada(maior_e_melhor=True)
    melhor_dir = saida / "checkpoints" / "melhor"

    def _salvar(dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        ap.salvar(dest)
        termo.salvar(dest / "termometro.npz")

    passos, ep, ultimo_prog, ultimo_eval, melhor = 0, 0, 0, 0, -1.0
    t0 = time.perf_counter()
    while passos < b_max:
        s = SEED_TREINO_BASE + ep
        random.seed(s); np.random.seed(s)
        env = make_env(cfg_tr); obs, _ = env.reset()
        prune_logging_handlers(handlers)
        agentes = list(env.agents)
        for i, _ in enumerate(agentes):
            acs[i].reset(ctx, np.random.default_rng(s + 7919 * i))
        antes = 0
        for _ in range(passos_max):
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
            for i in range(len(agentes)):           # CADA drone fecha sua transição
                acs[i].registrar(rew)
            passos += 1                              # passo de AMBIENTE (todos os drones)
            if any(term.values()) or any(trunc.values()):
                break
            for i, a in enumerate(agentes):          # e atualiza com o SEU próximo obs
                acs[i].passo_intermediario(Observacao.decodificar(obs[a], ctx))
            if passos >= b_max:
                break
        for ac in acs:
            ac.fim_de_episodio()
        ep += 1
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
        if passos - ultimo_prog >= progresso_cada:
            ultimo_prog = passos
            taxa = passos / max(1e-9, time.perf_counter() - t0)
            print(f"[{passos/b_max:5.1%}] {passos:,}/{b_max:,} · {ep:,} ep · "
                  f"{taxa:.0f} passos/s · ETA {(b_max-passos)/max(taxa,1)/3600:.1f} h", flush=True)
        # early-stop: avaliação periódica no val + checkpoint de melhor validação
        if passos - ultimo_eval >= intervalo:
            ultimo_eval = passos
            m = avaliar_multi(cfg_av, ap, termo, ctx_av, val)
            eh_melhor = m["cobertura"] > melhor
            if eh_melhor:
                melhor = m["cobertura"]; _salvar(melhor_dir)
            print(f"   eval @ {passos:,}: cob {m['cobertura']:.3f} · sr {m['sr']:.3f}"
                  f"{'  (melhor)' if eh_melhor else ''}", flush=True)
            if parada.registrar(m["cobertura"]):
                print(f"   parada antecipada (platô) em {passos:,} passos", flush=True)
                break

    # final: salva o estado final e avalia o MELHOR checkpoint (fix 539b61e)
    _salvar(saida / "checkpoints" / "final")
    if not melhor_dir.exists():
        _salvar(melhor_dir)                     # parou antes da 1ª melhora
    ap.ator = type(ap.ator).carregar(next(melhor_dir.glob("*ator.npz")))
    ap.critico = type(ap.critico).carregar(next(melhor_dir.glob("*critico.npz")))
    termo_m = TermometroQuantil.carregar(melhor_dir / "termometro.npz")
    bancos = carregar_bancos(na, ns)
    n_test = min(100, len(bancos.test.sementes))
    metr = avaliar_multi(cfg_av, ap, termo_m, ctx_av, bancos.test.sementes[:n_test])
    proc = {"ator": {"n_saidas": D_SAIDA}, "resolucao": t_res,
            "config": {"na": na, "ns": ns, "passo": PASSO, "sigma": SIGMA,
                       "T": t_res, "n": N_TUPLA, "phi": PHI, "m": M, "gamma": GAMMA,
                       "replica": replica, "parameter_sharing": True,
                       "recompensa_por_na": True,
                       "retomado_de": str(retomar) if retomar else None,
                       "critico": "descentralizado (Alg. 4.4)"},
            "passos": passos, "episodios": ep, "melhor_val_cobertura": melhor,
            "metrica_test": metr, "fonte": "spec06_treino_multiagente.py"}
    # Poda pós-treino PADRÃO: `checkpoints/deploy` podado e verificado no VAL.
    # Nunca falha o treino — um erro aqui perde o artefato de deploy, não a
    # corrida — e nunca toca `final`/`melhor`.
    try:
        from spec06_exportar_deploy import deploy_pos_treino
        dep = deploy_pos_treino(saida / "checkpoints" / "final", na, ns,
                                sementes=bancos.val.sementes[:100])
        proc["deploy"] = {k: dep[k] for k in
                          ("poda", "deploy_kb", "sr_cheio", "sr_podado")
                          if k in dep} | ({"motivo": dep["motivo"]}
                                          if "motivo" in dep else {})
        print(f"deploy podado ({dep.get('poda', 0):.0%}): "
              f"{dep['deploy_kb']:.0f} KB · SR {dep['sr_cheio']:.1%}"
              f"→{dep['sr_podado']:.1%} (val)"
              + (f" · {dep['motivo']}" if "motivo" in dep else ""), flush=True)
    except Exception as e:                                    # noqa: BLE001
        proc["deploy"] = {"erro": str(e)}
        print(f"aviso: deploy pós-treino falhou ({e}); final/melhor intactos",
              flush=True)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False))
    (melhor_dir / "procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False))
    print(f"\nNa={na} Ns={ns}: SR {metr['sr']:.0%} · CT {metr['ct_cens']:.1f} s · "
          f"cobertura {metr['cobertura']:.1%} · melhor-val {melhor:.1%} "
          f"({metr['n']} sementes)")
    print(f"checkpoint (melhor): {melhor_dir}")
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--na", type=int, required=True)
    ap.add_argument("--ns", type=int, default=24)
    ap.add_argument("--b-max", type=int, default=10_000_000)
    ap.add_argument("--replica", default="1b")
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--progresso-cada", type=int, default=50_000)
    ap.add_argument("--resolucao", type=int, default=T_RES,
                    help=f"T do termômetro; {T_RES} (padrão) ou 16 (econômico). "
                         "Ignorado com --retomar: T vem do checkpoint.")
    ap.add_argument("--retomar", type=Path, default=None,
                    help="dir de checkpoint (ex.: models/T32/em-treino/a8s24) para "
                         "continuar o treino em vez de recomeçar do zero")
    ap.add_argument("--smoke", action="store_true", help="b-max=5000, avaliação curta")
    a = ap.parse_args()
    if a.smoke:
        a.b_max = min(a.b_max, 5000); a.progresso_cada = 1000
    # O sufixo só aparece fora do padrão: as células de `T = 32` já publicadas
    # mantêm o caminho que `modelo-e-hiperparametros.md` registra.
    sufixo = "" if a.resolucao == T_RES else f"-T{a.resolucao}"
    saida = a.saida or Path(
        f"runs/spec06-multi/ntupla_dissertacao-a{a.na}s{a.ns}{sufixo}")
    print(f"SPEC-06 escalada multiagente · config vencedora "
          f"(passo {PASSO}, σ {SIGMA}, T {a.resolucao}, n {N_TUPLA}, φ {PHI}, m {M})\n",
          flush=True)
    treinar(a.na, a.ns, a.b_max, a.replica, saida, a.progresso_cada,
            a.resolucao, a.retomar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
