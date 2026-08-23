"""MADDPG sob ruído de GPS, com o KF construído na SPEC-07.

Avalia o **filtro** numa política forte e pronta, no ambiente real — desacoplando
a validação da fusão da qualidade da WNN da Etapa 1 (o risco D-06.3 do "piso de
mensurabilidade": degradar ruído sobre uma política ruim mede zero contra zero).

Roda a escada nos degraus **0** (sem ruído, teto), **1** (ruído bruto, piso) e
**4** (KF + entrada de controle), por célula e por σ, e mede o tempo de conclusão
mais o NEES do filtro sobre as trajetórias reais.

    uv run --group baseline python scripts/spec07_maddpg_ruido.py
    uv run --group baseline python scripts/spec07_maddpg_ruido.py --celulas 4:24,8:36 --sigmas 5,10

**Três rótulos de honestidade** (já discutidos no plano):
1. Isto é H3 (fusão), **não H1** — o MADDPG é referência de literatura (D7), não
   baseline pareado.
2. **Não há checkpoint para `Na = 1`** — a base da Etapa 2 é single-drone; aqui as
   células são `Na ∈ {2,4,8}`, com ruído só no GPS próprio (Etapa 2, D13), **não**
   o canal correlacionado da Etapa 3.
3. Regime **L→R** (treino limpo → avaliação ruidosa): o MADDPG nunca viu ruído, e
   o KF é o que o transfere sem retreino.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec02_baseline_maddpg import checkpoints  # noqa: E402
from spec02_maddpg_rollout import carregar_ator  # noqa: E402
from uavwnn.fusion import criar_ambiente_ruidoso  # noqa: E402
from uavwnn.fusion.consistencia import faixa_chi2_individual, nees_por_passo  # noqa: E402
from uavwnn.fusion.ruido import ConfigRuido  # noqa: E402
from uavwnn.refenv import LAMENZA_003  # noqa: E402


def rodar_episodio(cfg, politica, degrau, sigma, seed, q_w=0.0):
    """Um episódio no ambiente ruidoso; devolve conclusão, sucesso e o log do KF."""
    amb = criar_ambiente_ruidoso(
        cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=sigma), q_w=q_w
    )
    amb.indice_ruido = seed  # CRN: mesmo σ e semente → mesma realização entre degraus
    random.seed(seed)
    np.random.seed(seed)
    obs, _ = amb.reset()
    info: dict = {}
    while True:
        obs, _, term, trunc, infos = amb.step({a: politica(obs[a]) for a in amb.agents})
        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            break
        if any(term.values()) or any(trunc.values()):
            break
    try:
        amb.close()
    except Exception:  # pragma: no cover
        pass
    ct = float(info.get("completion_time", cfg.max_episode_length))
    ok = bool(info.get("all_collected", False))
    return ct, ok, amb.log_estimacao


def _agrega(cfg, politica, degrau, sigma, episodios):
    cts, oks, logs = [], [], []
    for s in range(episodios):
        ct, ok, log = rodar_episodio(cfg, politica, degrau, sigma, seed=s)
        cts.append(ct)
        oks.append(ok)
        logs.extend(log)
    saida = {
        "completion_time": float(np.mean(cts)),
        "taxa_sucesso": float(np.mean(oks)),
    }
    if logs:  # degrau 4: NEES do filtro nas trajetórias reais
        erros = np.array([e["erro"] for e in logs])
        covs = np.array([e["cov"] for e in logs])
        eps = nees_por_passo(erros, covs)
        lo, hi = faixa_chi2_individual(2)
        saida["nees_medio"] = float(eps.mean())
        saida["nees_frac_dentro"] = float(np.mean((eps >= lo) & (eps <= hi)))
        saida["rmse_estimativa"] = float(np.sqrt(np.mean(np.sum(erros**2, axis=1))))
    return saida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=20)
    ap.add_argument("--celulas", default="", help="ex. '4:24,8:36'; vazio = as 9")
    ap.add_argument("--sigmas", default="5,10", help="σ do GPS, em metros")
    ap.add_argument("--saida", default="reports/spec07-maddpg-ruido.json")
    a = ap.parse_args()

    disponiveis = {(na, ns): p for na, ns, p in checkpoints()}
    if not disponiveis:
        sys.exit("nenhum checkpoint; git submodule update --init --recursive")
    if a.celulas:
        pedidas = []
        for item in a.celulas.split(","):
            na, ns = (int(x) for x in item.split(":"))
            if (na, ns) not in disponiveis:
                sys.exit(f"sem checkpoint para Na={na}, Ns={ns}")
            pedidas.append((na, ns))
    else:
        pedidas = sorted(disponiveis)
    sigmas = [float(x) for x in a.sigmas.split(",")]

    print(f"MADDPG sob ruído · {a.episodios} episódios/condição · "
          f"escada degraus 0/1/4 · σ ∈ {sigmas}\n")
    print("H3 no ambiente real (fusão sobre política de referência); NÃO é H1 (D7).\n")

    artefato = {"config": {"episodios": a.episodios, "sigmas": sigmas}, "celulas": {}}

    cab = f"{'célula':>9} {'σ':>4} {'d0 (s)':>8} {'d1 (s)':>8} {'d4 (s)':>8} " \
          f"{'d1→d4':>7} {'d4 vs d0':>9} {'NEES':>6} {'RMSE(m)':>8}"
    print(cab)
    print("-" * len(cab))

    for na, ns in pedidas:
        cfg = LAMENZA_003(na, ns, phase="eval")
        politica = carregar_ator(disponiveis[(na, ns)], cfg.observation_size)
        cel = artefato["celulas"][f"a_{na}-s_{ns}"] = {"num_drones": na, "num_sensores": ns}

        d0 = _agrega(cfg, politica, degrau=0, sigma=0.0, episodios=a.episodios)
        cel["degrau0"] = d0

        for sigma in sigmas:
            d1 = _agrega(cfg, politica, degrau=1, sigma=sigma, episodios=a.episodios)
            d4 = _agrega(cfg, politica, degrau=4, sigma=sigma, episodios=a.episodios)
            cel[f"sigma_{sigma:g}"] = {"degrau1": d1, "degrau4": d4}
            # razões: quanto o KF recupera do estrago do ruído
            ganho = d1["completion_time"] / d4["completion_time"] if d4["completion_time"] else float("inf")
            vs0 = d4["completion_time"] / d0["completion_time"] if d0["completion_time"] else float("inf")
            print(f"{f'a_{na}-s_{ns}':>9} {sigma:>4.0f} "
                  f"{d0['completion_time']:>8.1f} {d1['completion_time']:>8.1f} "
                  f"{d4['completion_time']:>8.1f} {ganho:>6.2f}× {vs0:>8.2f}× "
                  f"{d4.get('nees_medio', float('nan')):>6.2f} {d4.get('rmse_estimativa', float('nan')):>8.2f}")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.saida, "w") as f:
        json.dump(artefato, f, indent=2, ensure_ascii=False)
    print(f"\nartefato: {a.saida}")
    print("\nLeitura: d1→d4 = quanto o KF recupera do estrago do ruído; d4 vs d0 =")
    print("o quanto sobra em relação ao teto sem ruído (ideal → 1,00×). NEES ~2 e")
    print("RMSE são a consistência do filtro medida nas trajetórias REAIS, não na bancada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
