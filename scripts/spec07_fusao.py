"""SPEC-07 — ruído no GPS, KF com entrada de controle, NEES/NIS e detector.

Roda na bancada sintética (sem `gradysim`): a escada de ablação, a consistência
por `σ`, a potência do detector e a contagem de custo. Fecha os critérios de
saída 4 (achado A3), 1 (NEES/NIS na faixa), 3 (potência do detector) e 8 (custo)
da SPEC-07 §6.

    uv run python scripts/spec07_fusao.py                 # M=100, ~segundos
    uv run python scripts/spec07_fusao.py --execucoes 300 # mais sementes

Escopo pós-D13: um canal só, o GPS do drone. O canal do sensor, o filtro de
informação, a H4 e a T3 saíram.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uavwnn.fusion import (
    CanalGPS,
    ConfigRuido,
    FiltroGanhoEstacionario,
    FiltroKalmanControle,
    FiltroVelocidadeConstante,
    PassaBaixa,
    autocorrelacao,
    avaliar,
    faixa_chi2,
    limiar_brancura,
    nis_por_passo,
    rodar_filtro,
    trajetoria_b1,
    trajetoria_b2,
)

SIGMAS = [1.0, 3.0, 5.0, 10.0]
Q_A_GRID = [0.5, 2.0, 5.0, 20.0, 50.0, 200.0]
DT_DEC = 2.0


def _trajetorias(M: int, semente: int, n_sensores: int = 12):
    """B2 (guloso analítico), para a escada — estatística on-policy."""
    return [
        trajetoria_b2(np.random.default_rng(semente + ep), n_sensores=n_sensores)
        for ep in range(M)
    ]


def _trajetorias_b1(M: int, semente: int, n_passos: int = 120):
    """B1 (comando sintético), para NEES/NIS e detector.

    Comprimento fixo, porque o protocolo Monte Carlo de T-07.6 exige a média
    sobre `M` execuções no **mesmo índice de tempo `k`** — e os episódios do
    guloso (B2) terminam em instantes diferentes, o que truncaria a análise aos
    poucos passos iniciais e esconderia o acúmulo de viés correlacionado.
    """
    return [
        trajetoria_b1(np.random.default_rng(semente + 10_000 + ep),
                      n_passos=n_passos, sigma_salto=np.pi)
        for ep in range(M)
    ]


def _canal(sigma: float, ep: int, modelo: str = "branco", raiz: int = 20260815):
    from uavwnn.fusion import fluxo_ruido
    cfg = ConfigRuido(sigma_gps=sigma, modelo=modelo)
    return CanalGPS(cfg, fluxo_ruido(raiz, "gps", 0, ep))


def escada(trajs, sigma: float, dt_gps: float = DT_DEC) -> dict:
    """RMSE médio por degrau, num `σ`. Degrau 3 usa o melhor `q_a` da grade."""
    # degrau 3: melhor q_a por RMSE, como T-07.10 pede (senão vira espantalho)
    melhor_qa, melhor_rmse_vc = None, np.inf
    for qa in Q_A_GRID:
        rmses = []
        for ep, traj in enumerate(trajs):
            f = FiltroVelocidadeConstante(sigma_gps=sigma, q_a=qa)
            rmses.append(rodar_filtro(traj, _canal(sigma, ep), f, dt_gps=dt_gps).rmse_filtrado)
        m = float(np.mean(rmses))
        if m < melhor_rmse_vc:
            melhor_rmse_vc, melhor_qa = m, qa

    def media_rmse(fabrica) -> float:
        vals = []
        for ep, traj in enumerate(trajs):
            r = rodar_filtro(traj, _canal(sigma, ep), fabrica(), dt_gps=dt_gps)
            vals.append(r.rmse_filtrado)
        return float(np.mean(vals))

    # degrau 1 (bruto) = erro da medida crua, comum a todos
    cru = float(np.mean([
        rodar_filtro(traj, _canal(sigma, ep), FiltroKalmanControle(sigma_gps=sigma), dt_gps=dt_gps).rmse_cru
        for ep, traj in enumerate(trajs)
    ]))

    return {
        "sigma": sigma,
        "d1_bruto": cru,
        "d2_ema": media_rmse(lambda: PassaBaixa(alpha=0.5, usar_comando=False)),
        "d2_ema_cmd": media_rmse(lambda: PassaBaixa(alpha=0.5, usar_comando=True)),
        "d3_kf_vc": melhor_rmse_vc,
        "d3_q_a": melhor_qa,
        "d4_kf_controle": media_rmse(lambda: FiltroKalmanControle(sigma_gps=sigma, q_w=0.0)),
        "d4_ganho_estac": media_rmse(lambda: FiltroGanhoEstacionario(sigma_gps=sigma, q_w=0.05)),
    }


def consistencia_degrau4(trajs, sigma: float) -> dict:
    """NEES/NIS do KF+controle. `Q = 0`: na bancada o modelo é exato (T-07.5)."""
    erros, covs, nis_todos = [], [], []
    for ep, traj in enumerate(trajs):
        f = FiltroKalmanControle(sigma_gps=sigma, q_w=0.0)
        r = rodar_filtro(traj, _canal(sigma, ep), f)
        erros.append(r.erros)
        covs.append(r.covs)
        if len(r.inovacoes):
            nis_todos.append(nis_por_passo(r.inovacoes, r.esses))
    n_min = min(e.shape[0] for e in erros)
    erros = np.array([e[:n_min] for e in erros])
    covs = np.array([c[:n_min] for c in covs])
    c = avaliar(erros, covs)
    nis_medio = float(np.mean(np.concatenate(nis_todos))) if nis_todos else float("nan")
    return {
        "sigma": sigma,
        "nees_medio": c.nees_medio,
        "faixa": list(c.faixa),
        "veredito": c.veredito,
        "fracao_fora": c.fracao_fora,
        "nis_medio": nis_medio,
    }


def potencia_detector(trajs, sigma: float = 5.0) -> list[dict]:
    """T-07.7 — desintonias injetadas e o que o NEES deve acusar.

    Sem inércia na bancada, o `Q` correto é 0, então perturbar `Q` não é o
    caminho: perturba-se `R` (o filtro acredita num ruído diferente do real) e
    injeta-se ruído Gauss-Markov contra um filtro que assume branco.
    """
    linhas = []

    def roda(sigma_filtro, modelo="branco"):
        erros, covs, resid = [], [], []
        for ep, traj in enumerate(trajs):
            f = FiltroKalmanControle(sigma_gps=sigma_filtro, q_w=0.0)
            r = rodar_filtro(traj, _canal(sigma, ep, modelo=modelo), f)
            erros.append(r.erros)
            covs.append(r.covs)
            resid.append(r.inovacoes)
        n_min = min(e.shape[0] for e in erros)
        c = avaliar(np.array([e[:n_min] for e in erros]), np.array([cc[:n_min] for cc in covs]))
        # brancura: autocorrelação lag 1 média sobre execuções
        rho1 = float(np.mean([abs(autocorrelacao(rr, max_lag=1)[0]) for rr in resid if len(rr) > 2]))
        K = n_min
        return c, rho1, limiar_brancura(K)

    base, rho_base, lim = roda(sigma)  # R correto
    linhas.append({"injecao": "R correto", "nees": base.nees_medio, "veredito": base.veredito,
                   "rho1": rho_base, "limiar_brancura": lim})
    c_otim, rho, _ = roda(sigma / 2)   # R/4 → otimista
    linhas.append({"injecao": "R/4 (subestima ruido)", "nees": c_otim.nees_medio,
                   "veredito": c_otim.veredito, "rho1": rho, "limiar_brancura": lim})
    c_cons, rho, _ = roda(sigma * 2)   # 4R → conservador
    linhas.append({"injecao": "4R (superestima ruido)", "nees": c_cons.nees_medio,
                   "veredito": c_cons.veredito, "rho1": rho, "limiar_brancura": lim})
    c_gm, rho_gm, _ = roda(sigma, modelo="gauss_markov")  # ruído correlacionado
    linhas.append({"injecao": "Gauss-Markov, filtro branco", "nees": c_gm.nees_medio,
                   "veredito": c_gm.veredito, "rho1": rho_gm, "limiar_brancura": lim})
    return linhas


def custo() -> dict:
    kf = FiltroKalmanControle.custo_operacoes()
    ge = FiltroGanhoEstacionario.custo_operacoes()
    macs_mlp = 25 * 32 + 32 * 32 + 32 * 3  # ator pareado D→32→32→3, D=25
    return {
        "kf_controle": {**kf.como_linha(), "fracao_dos_macs_mlp": kf.total / macs_mlp},
        "ganho_estacionario": ge.como_linha(),
        "macs_mlp_pareada": macs_mlp,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SPEC-07 — fusão local")
    ap.add_argument("--execucoes", type=int, default=100, help="M execuções Monte Carlo")
    ap.add_argument("--semente", type=int, default=20260815)
    ap.add_argument("--saida", type=Path, default=Path("reports/spec07-fusao.json"))
    args = ap.parse_args()

    trajs = _trajetorias(args.execucoes, args.semente)        # B2, escada
    trajs_b1 = _trajetorias_b1(args.execucoes, args.semente)  # B1, NEES/detector

    esc = [escada(trajs, s) for s in SIGMAS]
    cons = [consistencia_degrau4(trajs_b1, s) for s in SIGMAS]
    det = potencia_detector(trajs_b1)
    cst = custo()

    # -- impressão ----------------------------------------------------------
    print(f"\nSPEC-07 — bancada B2 (guloso analítico), M={args.execucoes}, dt_dec={DT_DEC}s\n")

    print("Escada de ablação — RMSE de posição (m), por σ e degrau")
    print(f"{'σ':>4} | {'1 bruto':>8} | {'2 EMA':>7} | {'2 EMA+u':>8} | "
          f"{'3 KF-VC':>8} | {'4 KF+u':>8} | {'4 ganho∞':>8}")
    for e in esc:
        print(f"{e['sigma']:>4.0f} | {e['d1_bruto']:>8.2f} | {e['d2_ema']:>7.2f} | "
              f"{e['d2_ema_cmd']:>8.2f} | {e['d3_kf_vc']:>8.2f} | "
              f"{e['d4_kf_controle']:>8.2f} | {e['d4_ganho_estac']:>8.2f}")

    print("\nAchado A3 — ganho do KF+controle sobre a medida crua, por σ")
    for e in esc:
        g = e["d1_bruto"] / e["d4_kf_controle"] if e["d4_kf_controle"] > 0 else float("inf")
        gvc = e["d3_kf_vc"] / e["d4_kf_controle"] if e["d4_kf_controle"] > 0 else float("inf")
        print(f"  σ={e['sigma']:>4.0f}: {e['d1_bruto']:.2f} → {e['d4_kf_controle']:.3f} m "
              f"({g:.1f}×);  bate o KF-VC ({e['d3_kf_vc']:.2f} m) por {gvc:.1f}×")

    print("\nConsistência do degrau 4 (NEES/NIS), por σ")
    for c in cons:
        print(f"  σ={c['sigma']:>4.0f}: NEES {c['nees_medio']:.2f} "
              f"faixa [{c['faixa'][0]:.2f}; {c['faixa'][1]:.2f}] → {c['veredito']:<12} "
              f"| NIS {c['nis_medio']:.2f} | fora {c['fracao_fora']*100:.1f}%")

    print("\nPotência do detector (T-07.7) — σ=5, o filtro deve acusar cada desintonia")
    for d in det:
        print(f"  {d['injecao']:<28}: NEES {d['nees']:.2f} → {d['veredito']:<12} "
              f"| |ρ(1)| {d['rho1']:.3f} (limiar {d['limiar_brancura']:.3f})")

    print("\nCusto (T-07.9)")
    kf = cst["kf_controle"]
    print(f"  KF+controle: {kf['total']} flops/decisão "
          f"(mult {kf['mult_fp']}, soma {kf['soma_fp']}, div {kf['div_fp']}) "
          f"= {kf['fracao_dos_macs_mlp']*100:.1f}% dos {cst['macs_mlp_pareada']} MACs da MLP pareada")
    print(f"  ganho estacionário: {cst['ganho_estacionario']['total']} flops, sem divisão")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps({
        "execucoes": args.execucoes, "semente": args.semente, "dt_dec": DT_DEC,
        "escada": esc, "consistencia": cons, "detector": det, "custo": cst,
    }, indent=2, ensure_ascii=False))
    print(f"\nArtefato: {args.saida}")


if __name__ == "__main__":
    main()
