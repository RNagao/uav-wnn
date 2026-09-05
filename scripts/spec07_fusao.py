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


def _curva_consistencia(trajs, sigma: float, fabrica):
    """`(Consistencia, nis_medio)` de um filtro sobre `trajs`, no mesmo índice `k`.

    Genérico no filtro para servir ao degrau 4 (KF+controle) e ao degrau 3
    (KF-VC), cujo NEES é a evidência de que ele é **inconsistente**, não só pior.
    """
    erros, covs, nis_todos = [], [], []
    for ep, traj in enumerate(trajs):
        r = rodar_filtro(traj, _canal(sigma, ep), fabrica())
        erros.append(r.erros)
        covs.append(r.covs)
        if len(r.inovacoes):
            nis_todos.append(nis_por_passo(r.inovacoes, r.esses))
    n_min = min(e.shape[0] for e in erros)
    c = avaliar(np.array([e[:n_min] for e in erros]), np.array([cc[:n_min] for cc in covs]))
    nis_medio = float(np.mean(np.concatenate(nis_todos))) if nis_todos else float("nan")
    return c, nis_medio


def consistencia_degrau4(trajs, sigma: float) -> dict:
    """NEES/NIS do KF+controle. `Q = 0`: na bancada o modelo é exato (T-07.5)."""
    c, nis = _curva_consistencia(trajs, sigma, lambda: FiltroKalmanControle(sigma_gps=sigma, q_w=0.0))
    return {
        "sigma": sigma, "nees_medio": c.nees_medio, "faixa": list(c.faixa),
        "veredito": c.veredito, "fracao_fora": c.fracao_fora, "nis_medio": nis,
    }


def varredura_kfvc(trajs, sigma: float) -> dict:
    """NEES × RMSE do KF-VC ao longo da grade de `q_a` — refina a T-07.10.

    A spec previa inconsistência para **todo** `q_a`. Medido, o quadro é mais
    rico e mais interessante: no `q_a` natural (pequeno) o KF-VC é **otimista**
    (a previsão vale), e só fica consistente inflando `q_a` a ponto de mal
    filtrar. Ele **não pode ser consistente e preciso ao mesmo tempo** — essa é a
    assinatura de 'mal especificado', não de 'mal sintonizado'.
    """
    linhas = []
    faixa = None
    for qa in Q_A_GRID:
        c, _ = _curva_consistencia(trajs, sigma, lambda qa=qa: FiltroVelocidadeConstante(sigma_gps=sigma, q_a=qa))
        faixa = list(c.faixa)
        rmses = [
            rodar_filtro(traj, _canal(sigma, ep), FiltroVelocidadeConstante(sigma_gps=sigma, q_a=qa)).rmse_filtrado
            for ep, traj in enumerate(trajs)
        ]
        linhas.append({
            "q_a": qa, "nees_medio": c.nees_medio, "veredito": c.veredito,
            "fracao_fora": c.fracao_fora, "rmse": float(np.mean(rmses)),
        })
    return {"sigma": sigma, "faixa": faixa, "varredura": linhas}


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


def gerar_figuras(esc, cons, kfvc, trajs_b1, dir_saida: Path, sigma_fig: float = 10.0) -> None:
    """Figuras do relatório (T-07.11 e §8). Recomputa as séries por passo.

    Pula com aviso se matplotlib não estiver disponível — as figuras são do grupo
    `dev` e não bloqueiam a medição.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[figuras] matplotlib ausente — pulei. Rode: uv sync --group dev")
        return

    dir_saida.mkdir(parents=True, exist_ok=True)
    TEAL, AMBAR, CINZA = "#1C7293", "#F2A93B", "#808A96"

    # 1. escada — barras de RMSE por σ e degrau
    fig, ax = plt.subplots(figsize=(9, 5))
    degraus = [("d1_bruto", "1 raw"), ("d2_ema", "2 EMA"), ("d2_ema_cmd", "2 EMA+u"),
               ("d3_kf_vc", "3 KF-VC"), ("d4_kf_controle", "4 KF+u")]
    sig = [e["sigma"] for e in esc]
    x = np.arange(len(sig))
    larg = 0.16
    for i, (chave, rot) in enumerate(degraus):
        ax.bar(x + (i - 2) * larg, [e[chave] for e in esc], larg, label=rot)
    ax.set_xticks(x); ax.set_xticklabels([f"σ={s:.0f}" for s in sig])
    ax.set_ylabel("position RMSE (m)"); ax.set_yscale("log")
    ax.set_title("Ablation ladder — RMSE by step and σ (testbed B2)")
    ax.legend(ncol=5, fontsize=8); fig.tight_layout()
    fig.savefig(dir_saida / "escada.png", dpi=110); plt.close(fig)

    # 2. NEES ε̄_k × k — KF+controle (na faixa) vs KF-VC no q_a natural (otimista)
    c4, _ = _curva_consistencia(trajs_b1, sigma_fig, lambda: FiltroKalmanControle(sigma_gps=sigma_fig, q_w=0.0))
    qa3 = 2.0  # q_a natural (pequeno); no RMSE-ótimo o KF-VC vira consistente-mas-impreciso
    c3, _ = _curva_consistencia(trajs_b1, sigma_fig, lambda: FiltroVelocidadeConstante(sigma_gps=sigma_fig, q_a=qa3))
    fig, ax = plt.subplots(figsize=(9, 5))
    k4 = np.arange(1, len(c4.media_por_passo) + 1)
    ax.plot(k4, c4.media_por_passo, color=TEAL, label="step 4 (KF+control)")
    ax.plot(np.arange(1, len(c3.media_por_passo) + 1), c3.media_por_passo, color=AMBAR,
            label=f"step 3 (KF-VC, q_a={qa3:g})")
    ax.axhspan(c4.faixa[0], c4.faixa[1], color=CINZA, alpha=0.25, label="χ² 95% band")
    ax.axhline(2.0, color=CINZA, ls=":", lw=1)
    ax.set_xlabel("decision step k"); ax.set_ylabel("mean NEES ε̄ₖ")
    ax.set_title(f"Per-step consistency, σ={sigma_fig:.0f} m — KF-VC optimistic at natural q_a")
    ax.legend(); fig.tight_layout()
    fig.savefig(dir_saida / "nees_degraus.png", dpi=110); plt.close(fig)

    # 3. correlograma da inovação — branco vs Gauss-Markov
    def correl_media(modelo):
        rhos = []
        for ep, traj in enumerate(trajs_b1):
            f = FiltroKalmanControle(sigma_gps=sigma_fig, q_w=0.0)
            r = rodar_filtro(traj, _canal(sigma_fig, ep, modelo=modelo), f)
            if len(r.inovacoes) > 21:
                rhos.append(autocorrelacao(r.inovacoes, max_lag=20))
        return np.mean(rhos, axis=0)
    rho_b = correl_media("branco"); rho_gm = correl_media("gauss_markov")
    K = trajs_b1[0].n_passos; lim = limiar_brancura(K)
    fig, ax = plt.subplots(figsize=(9, 5))
    lags = np.arange(1, 21)
    ax.bar(lags - 0.2, rho_b, 0.4, color=TEAL, label="white (correct filter)")
    ax.bar(lags + 0.2, rho_gm, 0.4, color=AMBAR, label="Gauss-Markov (white filter)")
    ax.axhline(lim, color=CINZA, ls="--", lw=1, label=f"envelope ±1.96/√K = ±{lim:.3f}")
    ax.axhline(-lim, color=CINZA, ls="--", lw=1)
    ax.set_xlabel("lag l"); ax.set_ylabel("autocorrelation ρ(l)")
    ax.set_title(f"Innovation whiteness, σ={sigma_fig:.0f} m — the detector flags the correlation")
    ax.legend(); fig.tight_layout()
    fig.savefig(dir_saida / "correlograma.png", dpi=110); plt.close(fig)

    # 4. erro(k) com o ideal σ√(2/(k+1)) — o transiente (T-07.11)
    erros_k, trp_k = [], []
    for ep, traj in enumerate(trajs_b1):
        f = FiltroKalmanControle(sigma_gps=sigma_fig, q_w=0.0)
        r = rodar_filtro(traj, _canal(sigma_fig, ep), f)
        erros_k.append(np.linalg.norm(r.erros, axis=1))
        trp_k.append(np.sqrt(np.trace(r.covs, axis1=1, axis2=2)))
    erro_rms = np.sqrt(np.mean(np.array(erros_k) ** 2, axis=0))
    trp = np.mean(trp_k, axis=0)
    k = np.arange(1, len(erro_rms) + 1)
    ideal = sigma_fig * np.sqrt(2.0 / (k + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, erro_rms, color=TEAL, label="measured RMS error")
    ax.plot(k, trp, color=AMBAR, ls="--", label="√tr P (filter prediction)")
    ax.plot(k, ideal, color=CINZA, ls=":", label="ideal σ√(2/(k+1))")
    ax.axhline(sigma_fig * np.sqrt(2), color=CINZA, alpha=0.4, lw=1, label="raw measurement σ√2")
    ax.set_xlabel("decision step k"); ax.set_ylabel("position error (m)")
    ax.set_title(f"Filter transient, σ={sigma_fig:.0f} m — the episode mean hides the onset")
    ax.legend(); fig.tight_layout()
    fig.savefig(dir_saida / "erro_transiente.png", dpi=110); plt.close(fig)

    # 5. viés do passa-baixa × α (T-07.10)
    alphas = [0.1, 0.25, 0.5, 0.75, 1.0]
    v, dt = 15.0, 2.0
    medido = []
    for a in alphas:
        pb = PassaBaixa(alpha=a, usar_comando=False); pb.iniciar(np.zeros(2), 0.0)
        p = np.zeros(2); u = np.array([v, 0.0])
        for j in range(1, 200):
            p = p + u * dt; pb.comando(u); pb.atualizar(p, float(j * dt))
        medido.append(float(p[0] - pb.media[0]))
    teorico = [v * dt * (1 - a) / a for a in alphas]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, teorico, color=CINZA, ls="--", label="theoretical v·dt·(1−α)/α")
    ax.scatter(alphas, medido, color=AMBAR, zorder=3, label="measured")
    ax.set_xlabel("EMA α"); ax.set_ylabel("steady-state lag (m)")
    ax.set_title("Step 2 without command is biased, not just less efficient")
    ax.legend(); fig.tight_layout()
    fig.savefig(dir_saida / "passa_baixa_alpha.png", dpi=110); plt.close(fig)

    # 6. detector: NEES × desintonia de R cruzando a faixa (T-07.7)
    fatores = [0.25, 0.5, 1.0, 2.0, 4.0]  # R' = fator²·R (filtro acredita σ·fator)
    neeses = []
    for fat in fatores:
        c, _ = _curva_consistencia(trajs_b1, sigma_fig,
                                   lambda fat=fat: FiltroKalmanControle(sigma_gps=sigma_fig * fat, q_w=0.0))
        neeses.append(c.nees_medio)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fatores, neeses, "o-", color=TEAL)
    ax.axhspan(c4.faixa[0], c4.faixa[1], color=CINZA, alpha=0.25, label="χ² 95% band")
    ax.set_xscale("log"); ax.set_xlabel("filter σ / true σ"); ax.set_ylabel("mean NEES")
    ax.set_title("Detector power — wrong R pulls NEES out of the band")
    ax.annotate("optimistic\n(P too small)", (0.25, neeses[0]), fontsize=8, va="bottom")
    ax.annotate("conservative", (4.0, neeses[-1]), fontsize=8, ha="right", va="top")
    ax.legend(); fig.tight_layout()
    fig.savefig(dir_saida / "detector_nees_R.png", dpi=110); plt.close(fig)

    # 7. KF-VC: o tradeoff NEES × RMSE por q_a (refina T-07.10)
    qas = [l["q_a"] for l in kfvc["varredura"]]
    nees_qa = [l["nees_medio"] for l in kfvc["varredura"]]
    rmse_qa = [l["rmse"] for l in kfvc["varredura"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qas, nees_qa, "o-", color=TEAL, label="NEES")
    ax.axhspan(kfvc["faixa"][0], kfvc["faixa"][1], color=CINZA, alpha=0.25, label="χ² 95% band")
    ax.set_xscale("log"); ax.set_xlabel("q_a (KF-VC)"); ax.set_ylabel("mean NEES", color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL)
    ax2 = ax.twinx()
    ax2.plot(qas, rmse_qa, "s--", color=AMBAR, label="RMSE")
    ax2.axhline(sigma_fig * np.sqrt(2), color=CINZA, alpha=0.4, lw=1)
    ax2.set_ylabel("position RMSE (m)", color=AMBAR); ax2.tick_params(axis="y", labelcolor=AMBAR)
    ax.set_title(f"Step 3, σ={sigma_fig:.0f} — optimistic at natural q_a, consistent only without filtering")
    fig.tight_layout()
    fig.savefig(dir_saida / "kfvc_tradeoff.png", dpi=110); plt.close(fig)

    print(f"\n[figuras] 7 figuras em {dir_saida}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="SPEC-07 — fusão local")
    ap.add_argument("--execucoes", type=int, default=100, help="M execuções Monte Carlo")
    ap.add_argument("--semente", type=int, default=20260815)
    ap.add_argument("--saida", type=Path, default=Path("reports/spec07-fusao.json"))
    ap.add_argument("--sem-figuras", action="store_true", help="não gera as figuras")
    ap.add_argument("--dir-figuras", type=Path, default=Path("reports/img/spec07"))
    args = ap.parse_args()

    trajs = _trajetorias(args.execucoes, args.semente)        # B2, escada
    trajs_b1 = _trajetorias_b1(args.execucoes, args.semente)  # B1, NEES/detector

    esc = [escada(trajs, s) for s in SIGMAS]
    cons = [consistencia_degrau4(trajs_b1, s) for s in SIGMAS]
    # degrau 3: NEES × RMSE ao longo de q_a, em σ=10 — o tradeoff do KF-VC
    kfvc = varredura_kfvc(trajs_b1, 10.0)
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

    print(f"\nDegrau 3 (KF-VC) — NEES × RMSE por q_a, σ={kfvc['sigma']:.0f} "
          f"(faixa [{kfvc['faixa'][0]:.2f}; {kfvc['faixa'][1]:.2f}])")
    print("  refina T-07.10: otimista no q_a natural, consistente só inflando Q a ponto de mal filtrar")
    for l in kfvc["varredura"]:
        print(f"  q_a={l['q_a']:>5g}: NEES {l['nees_medio']:>5.2f} → {l['veredito']:<12} "
              f"| RMSE {l['rmse']:.2f} m")

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

    if not args.sem_figuras:
        gerar_figuras(esc, cons, kfvc, trajs_b1, args.dir_figuras)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps({
        "execucoes": args.execucoes, "semente": args.semente, "dt_dec": DT_DEC,
        "escada": esc, "consistencia": cons, "kfvc_tradeoff": kfvc,
        "detector": det, "custo": cst,
    }, indent=2, ensure_ascii=False))
    print(f"\nArtefato: {args.saida}")


if __name__ == "__main__":
    main()
