"""SPEC-08 — escada de ablação 0–4 em L→R (transferência), célula 1×12.

Carrega uma política WNN treinada (o degrau 0 = a política da Etapa 1) e a avalia
sob cada estimador da escada, no regime **L→R**: treino limpo, avaliação ruidosa.
Não treina nada — a mesma política é avaliada sob os degraus 1–4 por σ, no banco
de teste compartilhado (CRN). Os parâmetros de filtro (q_a do degrau 3, q_w do
degrau 4) vêm do atestado do gate.

    uv run python scripts/spec08_escada.py --checkpoint <dir>/checkpoints/melhor
    uv run python scripts/spec08_escada.py --checkpoint <dir> --smoke

**Bloqueio B1 (RF-05):** recusa-se a rodar sem `results/etapa2/gate_rf05.json`
aprovado e coerente com o pré-registro. É o mecanismo da §4 — não uma intenção.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar  # noqa: E402  (loader oficial checkpoint→política)
from uavwnn.campanha import carregar_bancos  # noqa: E402
from uavwnn.fusion import criar_ambiente_ruidoso  # noqa: E402
from uavwnn.fusion.consistencia import nees_por_passo  # noqa: E402
from uavwnn.fusion.ruido import ConfigRuido  # noqa: E402
from uavwnn.refenv import LAMENZA_003  # noqa: E402

GATE = Path("results/etapa2/gate_rf05.json")
PRE = Path("results/etapa2/analysis_plan.sha256")


def _b1_guard() -> dict:
    """Bloqueio B1: sem atestado RF-05 aprovado, nenhuma rodada de RL começa."""
    if not GATE.exists():
        sys.exit("B1 (RF-05): results/etapa2/gate_rf05.json ausente — "
                 "rode scripts/spec08_gate_rf05.py antes.")
    g = json.loads(GATE.read_text())
    if g.get("status") != "aprovado":
        sys.exit(f"B1 (RF-05): atestado reprovado (status={g.get('status')!r}). "
                 "Nenhuma rodada de RL desta spec executa.")
    if PRE.exists():
        esperado = PRE.read_text().split()[0]
        if g.get("hash_experimento") not in (None, esperado):
            sys.exit("B1: hash do experimento no atestado diverge do pré-registro "
                     "— o desenho mudou; refaça o gate.")
    return g


def _params_filtro(degrau: int, gate: dict) -> dict:
    """Parâmetros do estimador do degrau, tirados do atestado (melhores q)."""
    if degrau == 2:
        return {"alpha": 0.5}                       # varredura de α: T-08.5 (adiada)
    if degrau == 3:
        return {"q_a": float(gate["degrau3"]["melhor"])}
    if degrau == 4:
        return {"q_w": float(gate["degrau4"]["melhor"])}
    return {}


def rodar_episodio(cfg, politica, degrau, sigma, seed, gate):
    """Um episódio: devolve (tempo_conclusao_censurado, sucesso, log_estimacao)."""
    random.seed(seed); np.random.seed(seed)
    amb = criar_ambiente_ruidoso(
        cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=sigma),
        **_params_filtro(degrau, gate))
    amb.indice_ruido = seed                          # CRN: mesmo z entre degraus
    obs, _ = amb.reset()
    ct, ok = float(cfg.max_episode_length), False
    for _ in range(400):
        obs, _, term, trunc, infos = amb.step({a: politica(obs[a]) for a in amb.agents})
        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            ct = float(info.get("completion_time", cfg.max_episode_length))
            ok = bool(info.get("all_collected", False))
            break
        if any(term.values()) or any(trunc.values()):
            break
    return ct, ok, amb.log_estimacao


def _nees_medio(log) -> float | None:
    covs = [x for x in log if "cov" in x]
    if not covs:
        return None
    er = np.array([x["erro"] for x in covs])
    co = np.array([x["cov"] for x in covs])
    return float(np.mean(nees_por_passo(er, co)))


def _serie(cfg, politica, degrau, sigma, sementes, gate):
    cts, oks, nees = [], [], []
    for sd in sementes:
        ct, ok, log = rodar_episodio(cfg, politica, degrau, sigma, int(sd), gate)
        cts.append(ct); oks.append(ok)
        nm = _nees_medio(log)
        if nm is not None:
            nees.append(nm)
    saida = {"ct_medio": float(np.mean(cts)), "sr": float(np.mean(oks)),
             "n_ep": len(cts)}
    if nees:
        saida["nees_medio"] = float(np.mean(nees))
    return saida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="dir do checkpoint (…/checkpoints/melhor)")
    ap.add_argument("--aproximador", default=None,
                    help="família; padrão: lê do config.json do run")
    ap.add_argument("--sigmas", default="1,3,5,10")
    ap.add_argument("--degraus", default="0,1,2,3,4")
    ap.add_argument("--episodios", type=int, default=500)
    ap.add_argument("--smoke", action="store_true",
                    help="σ=5, degraus 0–4, 3 episódios — valida load + escada + B1")
    ap.add_argument("--saida", type=Path, default=Path("results/etapa2/escada_lr.json"))
    a = ap.parse_args()

    gate = _b1_guard()

    if a.smoke:
        a.sigmas, a.degraus, a.episodios = "5", "0,1,2,3,4", min(a.episodios, 3)

    aprox = a.aproximador
    if aprox is None:
        cfgjson = a.checkpoint.parent.parent / "config.json"
        aprox = json.loads(cfgjson.read_text())["config"]["aproximador"]

    cfg = LAMENZA_003(1, 12, phase="eval")
    politica, _ = montar(aprox, cfg, a.checkpoint)

    sigmas = [float(x) for x in a.sigmas.split(",")]
    degraus = [int(x) for x in a.degraus.split(",")]
    sementes = carregar_bancos(1, 12).test.sementes[:a.episodios]

    print(f"escada L→R · {aprox} · degraus {degraus} · σ {sigmas} · "
          f"{len(sementes)} episódios")
    print(f"B1 OK — gate {gate['status']} (q_a={gate['degrau3']['melhor']}, "
          f"q_w={gate['degrau4']['melhor']})\n")

    artefato = {"config": {"aproximador": aprox, "checkpoint": str(a.checkpoint),
                           "sigmas": sigmas, "degraus": degraus,
                           "episodios": len(sementes), "regime": "LR"},
                "hash_filtro": gate.get("hash_filtro"),
                "celulas": {}}

    # degrau 0 independe de σ — mede uma vez (teto)
    base = None
    if 0 in degraus:
        base = _serie(cfg, politica, 0, 0.0, sementes, gate)
        print(f"degrau 0 (teto): CT {base['ct_medio']:.1f} s · SR {base['sr']:.0%}")

    for sigma in sigmas:
        cel = artefato["celulas"][f"sigma_{sigma:g}"] = {}
        if base is not None:
            cel["degrau0"] = base
        for degrau in degraus:
            if degrau == 0:
                continue
            r = _serie(cfg, politica, degrau, sigma, sementes, gate)
            cel[f"degrau{degrau}"] = r
            extra = f" · NEES {r['nees_medio']:.2f}" if "nees_medio" in r else ""
            print(f"σ={sigma:g} degrau {degrau}: CT {r['ct_medio']:.1f} s · "
                  f"SR {r['sr']:.0%}{extra}")
        print()

    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(artefato, indent=2, ensure_ascii=False))
    print(f"artefato: {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
