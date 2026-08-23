"""SPEC-08 T-08.1 — atestado de consistência (RF-05) e não vazamento (RF-04).

O **bloqueio B1** da Etapa 2: nenhuma rodada de RL desta spec pode começar antes
que este atestado exista com `status: aprovado`. Mede NEES/NIS por degrau (2, 3,
4) sob ruído com uma **política gulosa de referência** — a consistência do filtro
é agnóstica à política, então não precisa do WNN treinado; depende do filtro e da
mobilidade. Varre `q_a` no degrau 3 e `q_w` no degrau 4, roda o teste RF-04, e
grava `results/etapa2/gate_rf05.json` com o **hash da configuração de filtro**
validada.

    uv run python scripts/spec08_gate_rf05.py --episodios 100     # atestado
    uv run python scripts/spec08_gate_rf05.py --episodios 20      # smoke

Regras (T-08.1): o **degrau 3 pode reprovar legitimamente** (mal especificado, não
mal sintonizado) — registra-se "inconsistente por descasamento" e não bloqueia. O
que bloqueia é o **degrau 4** reprovar, ou o RF-04 falhar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from uavwnn.fusion import criar_ambiente_ruidoso
from uavwnn.fusion.consistencia import avaliar
from uavwnn.fusion.ruido import ConfigRuido
from uavwnn.refenv import LAMENZA_003, load_heuristics

SIGMA_ANCORA = 5.0
Q_A_GRADE = [0.5, 2.0, 5.0, 20.0, 50.0, 200.0]     # degrau 3 (T-08.1)
Q_W_GRADE = [0.0, 0.25, 1.0, 4.0]                  # degrau 4: descasamento de atuação
SAIDA = Path("results/etapa2/gate_rf05.json")


def _coletar(cfg, degrau, sigma, episodios, **kw):
    """Roda o guloso por `episodios` no ambiente ruidoso e empilha os logs.

    Devolve arrays alinhados truncando ao menor `K` (episódios têm comprimentos
    diferentes): `(erros[M,K,2], covs[M,K,2,2]|None, inov[M,J,2]|None, S[...]|None)`.
    """
    greedy = load_heuristics().create_greedy_heuristics(0, 12)
    erros_ep, covs_ep, inov_ep, ess_ep = [], [], [], []
    for e in range(episodios):
        random.seed(e); np.random.seed(e)
        amb = criar_ambiente_ruidoso(
            cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=sigma), **kw)
        amb.indice_ruido = e
        obs, _ = amb.reset()
        for _ in range(400):
            obs, _, term, trunc, infos = amb.step({a: greedy(obs[a]) for a in amb.agents})
            if (infos and "avg_reward" in next(iter(infos.values()))) \
                    or any(term.values()) or any(trunc.values()):
                break
        log = amb.log_estimacao
        if not log:
            continue
        erros_ep.append(np.array([x["erro"] for x in log]))
        if all("cov" in x for x in log):
            covs_ep.append(np.array([x["cov"] for x in log]))
        inov = [x for x in log if "inovacao" in x]
        if inov:
            inov_ep.append(np.array([x["inovacao"] for x in inov]))
            ess_ep.append(np.array([x["S"] for x in inov]))

    def _empilha(lst):
        if not lst:
            return None
        k = min(len(a) for a in lst)
        return np.stack([a[:k] for a in lst]) if k > 0 else None

    return _empilha(erros_ep), _empilha(covs_ep), _empilha(inov_ep), _empilha(ess_ep)


def _rmse(erros: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(erros**2, axis=-1))))


def _consistencia(cfg, degrau, episodios, **kw) -> dict:
    """NEES (erro,P) e NIS (inovação,S) de um degrau, mais RMSE."""
    erros, covs, inov, ess = _coletar(cfg, degrau, SIGMA_ANCORA, episodios, **kw)
    out: dict = {"rmse": _rmse(erros) if erros is not None else None,
                 "n_execucoes": 0 if erros is None else int(erros.shape[0]),
                 "n_passos": 0 if erros is None else int(erros.shape[1])}
    if covs is not None and erros is not None:
        c = avaliar(erros, covs)
        out["nees"] = {"medio": c.nees_medio, "faixa": list(c.faixa),
                       "fracao_fora": c.fracao_fora, "veredito": c.veredito}
    if inov is not None and ess is not None:
        c = avaliar(inov, ess)
        out["nis"] = {"medio": c.nees_medio, "faixa": list(c.faixa),
                      "fracao_fora": c.fracao_fora, "veredito": c.veredito}
    return out


def _varre(cfg, degrau, grade, chave, episodios) -> dict:
    """Varre o parâmetro do filtro e escolhe o melhor (NEES mais perto de n=2)."""
    tentativas = {}
    melhor_q, melhor_dist = None, float("inf")
    for q in grade:
        r = _consistencia(cfg, degrau, episodios, **{chave: q})
        tentativas[str(q)] = r
        nees = r.get("nees", {}).get("medio")
        if nees is not None:
            dist = abs(nees - 2.0)
            if dist < melhor_dist:
                melhor_dist, melhor_q = dist, q
    return {"grade": chave, "melhor": melhor_q,
            "resultado_melhor": tentativas.get(str(melhor_q), {}),
            "tentativas": tentativas}


def _rodar_rf04() -> dict:
    """RF-04 (B2): roda o teste de não vazamento e registra verde/vermelho."""
    proc = subprocess.run(
        ["uv", "run", "pytest", "tests/test_vazamento.py", "-q"],
        capture_output=True, text=True)
    return {"verde": proc.returncode == 0,
            "resumo": (proc.stdout.strip().splitlines() or ["?"])[-1]}


def _hash_filtro(dt: float, q_w: float) -> str:
    cfg_filtro = {"modelo": "KF 2 estados x=[x,y], predicao x+u*dt",
                  "R": "sigma_gps^2 * I2", "Q": "q_w*dt + sigma_v0^2*dt^2",
                  "dt": dt, "q_w": q_w, "sigma_v0": 0.0, "kappa": 0.0,
                  "sigma_ancora": SIGMA_ANCORA}
    return hashlib.sha256(
        json.dumps(cfg_filtro, sort_keys=True).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=100)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    a = ap.parse_args()

    cfg = LAMENZA_003(1, 12, phase="eval")
    dt = float(cfg.algorithm_iteration_interval)
    print(f"Gate RF-05/RF-04 · célula 1×12 · σ_âncora={SIGMA_ANCORA} · "
          f"{a.episodios} episódios (guloso de referência)\n")

    print("degrau 2 (passa-baixa)…", flush=True)
    d2 = _consistencia(cfg, 2, a.episodios, alpha=0.5)
    print(f"  RMSE {d2['rmse']:.2f} m (sem P — NEES/NIS não se aplica)\n")

    print("degrau 3 (KF vel. constante) — varrendo q_a…", flush=True)
    d3 = _varre(cfg, 3, Q_A_GRADE, "q_a", a.episodios)
    v3 = d3["resultado_melhor"].get("nees", {}).get("veredito", "?")
    print(f"  melhor q_a={d3['melhor']} · NEES {v3}\n")

    print("degrau 4 (KF+controle) — varrendo q_w…", flush=True)
    d4 = _varre(cfg, 4, Q_W_GRADE, "q_w", a.episodios)
    nees4 = d4["resultado_melhor"].get("nees", {})
    nis4 = d4["resultado_melhor"].get("nis", {})
    print(f"  melhor q_w={d4['melhor']} · NEES {nees4.get('veredito','?')} "
          f"({nees4.get('medio', float('nan')):.2f}) · "
          f"NIS {nis4.get('veredito','?')}\n")

    print("RF-04 (não vazamento)…", flush=True)
    rf04 = _rodar_rf04()
    print(f"  {'verde' if rf04['verde'] else 'VERMELHO'} — {rf04['resumo']}\n")

    # aprovação: degrau 4 consistente (NEES) E RF-04 verde. Degrau 3 pode reprovar.
    degrau4_ok = nees4.get("veredito") == "consistente"
    status = "aprovado" if (degrau4_ok and rf04["verde"]) else "reprovado"

    atestado = {
        "status": status,
        "hash_filtro": _hash_filtro(dt, d4["melhor"] if d4["melhor"] is not None else 0.0),
        "hash_experimento": Path("results/etapa2/analysis_plan.sha256").read_text().split()[0]
        if Path("results/etapa2/analysis_plan.sha256").exists() else None,
        "sigma_ancora": SIGMA_ANCORA, "dt": dt, "episodios": a.episodios,
        "degrau2": d2, "degrau3": d3, "degrau4": d4, "rf04": rf04,
        "regras": {
            "degrau4_consistente": degrau4_ok,
            "degrau3_pode_reprovar": True,
            "_": "aprovado exige degrau 4 NEES consistente E RF-04 verde",
        },
    }
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(atestado, indent=2, ensure_ascii=False))
    print(f"=== {status.upper()} === → {a.saida}")
    print(f"hash_filtro: {atestado['hash_filtro'][:16]}…")
    return 0 if status == "aprovado" else 1


if __name__ == "__main__":
    raise SystemExit(main())
