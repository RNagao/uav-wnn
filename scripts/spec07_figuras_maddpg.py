"""Figuras de mérito e imagens da run do MADDPG sob ruído.

Consome os artefatos JSON já gerados (sem re-rodar o simulador — o princípio da
SPEC-10: figura vem de artefato, não de nova medição) e produz:

- **Figuras de mérito com o filtro embutido (RNF-05 / SPEC-08 T-08.10):** T1
  (estado do KF somado ao ator) e T2 (latência empírica MEDIDA de um ciclo do KF
  contra a decisão do ator MADDPG, protocolo da SPEC-03), impressas como tabela e
  como figura. O ponto: o custo do filtro é uma **constante pequena e fixa**,
  independente da capacidade do modelo.
- **Imagens** em `reports/img/spec07-maddpg/`: comparação com Lamenza, degradação
  × σ, NEES no ambiente real, RMSE do filtro.

    uv run python scripts/spec07_figuras_maddpg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavwnn.fusion.kalman import FiltroKalmanControle
from uavwnn.fusion.consistencia import faixa_chi2_individual

# MADDPG, dos pesos publicados (reports/spec02-baseline-maddpg.md).
MADDPG_ATOR_KB = {2: 287.0, 4: 291.0, 8: 299.0}   # tamanho do ator, fp32
# Célula do artefato de latência da SPEC-03 (única com WNN deploy E MADDPG).
CELULA_LATENCIA = (2, 24)
# Orçamento de tempo real: uma decisão a cada 2 s (PRD §5.5). O KF é termo COMUM
# (WNN e MADDPG usam o mesmo filtro), contabilizado à parte contra esse prazo.
INTERVALO_DECISAO_S = 2.0

DIR = Path("reports/img/spec07-maddpg")
TEAL, AMBAR, CINZA, VERM = "#1C7293", "#F2A93B", "#808A96", "#B0637C"


def _ler(p):
    return json.loads(Path(p).read_text())


def _latencia_ator_maddpg(na: int = CELULA_LATENCIA[0],
                          ns: int = CELULA_LATENCIA[1]) -> dict | None:
    """Latência MEDIDA da decisão do ator MADDPG, do artefato da SPEC-03.

    Baseline honesto para dimensionar o KF: o mesmo protocolo (um processo por
    modelo, mesma observação, menor mediana de 8×1000) que mede a WNN mede o ator.
    """
    art = Path(f"reports/spec03-latencia-memoria-a{na}s{ns}.json")
    if not art.exists():
        print(f"  aviso: {art} ausente — rode a SPEC-03 para a célula {na}d/{ns}s "
              f"antes; a latência da política fica de fora", flush=True)
        return None
    for r in json.loads(art.read_text()):
        if r["modelo"] == "maddpg":
            return {"celula": f"a{na}s{ns}", "mediana_us": r["mediana_us"],
                    "p99_us": r["p99_us"]}
    return None


def _medir_kf_latencia(sigma_gps: float = 10.0, dt: float = 2.0) -> dict:
    """Latência empírica de UM ciclo do KF (comando + predição + correção), por drone.

    Mesmo espírito da SPEC-03: aquece, cronometra em blocos e reporta a MENOR
    mediana de bloco (o estimador sem contenção) mais o p99 global. O KF é
    determinístico e minúsculo — a medida é o custo real que ele soma à decisão da
    política, para poder dizer 'desprezível' com o relógio, e não só com contagem
    de flops (que subestima o filtro por construção).
    """
    import time

    rng = np.random.default_rng(12345)
    passos = 4096
    ang = rng.uniform(0.0, 2 * np.pi, passos)
    us = 15.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)   # comandos de rumo
    zs = rng.normal(0.0, sigma_gps, size=(passos, 2))          # medidas ruidosas
    kf = FiltroKalmanControle(sigma_gps=sigma_gps, q_w=1.0)
    kf.iniciar(zs[0], t0=0.0)
    est = {"t": 0.0, "k": 0}

    def um_ciclo():
        k = est["k"] % passos
        est["t"] += dt
        kf.comando(us[k])
        kf.atualizar(zs[k], est["t"])
        est["k"] += 1

    for _ in range(1000):                                      # aquecimento
        um_ciclo()
    blocos, reps = 8, 2000
    tudo, medianas = [], []
    for _ in range(blocos):
        t = np.empty(reps)
        for i in range(reps):
            t0 = time.perf_counter_ns(); um_ciclo(); t[i] = time.perf_counter_ns() - t0
        arr = t / 1000.0
        medianas.append(float(np.median(arr))); tudo.append(arr)
    return {"mediana_us": min(medianas),
            "p99_us": float(np.percentile(np.concatenate(tudo), 99)),
            "sigma_gps": sigma_gps, "dt": dt}


def figuras_de_merito() -> dict:
    """T1 (memória) e T2 (latência empírica MEDIDA) do KF, contabilizado À PARTE.

    O KF é **termo comum**: WNN e MADDPG podem usar o MESMO filtro, então na
    comparação de H1 ele **cancela** — não é cobrado de nenhuma das políticas. Por
    isso é medido de forma autônoma e referido ao **orçamento de tempo real** (uma
    decisão a cada 2 s), que é o que o deploy precisa cumprir, independente de qual
    aproximador roda. A latência de uma política entra só como referência de escala.

    A figura antiga contava multiplicações em ponto flutuante (T3). Foi trocada
    pela **latência empírica**: contar multiplicação subestima o KF por construção
    (ele mal multiplica); o relógio é o custo real, e contra o prazo de 2 s ele é
    desprezível. (O protótipo do KF é NumPy — 2×2 dominado por overhead de dispatch;
    o número absoluto é do protótipo, não do algoritmo, mas o veredito contra o
    orçamento de 2 s vale com folga de 4 ordens de grandeza.)
    """
    # T1: estado residente do KF por drone = x̂ (2) + P simétrica (3) floats fp32
    kf_bytes_por_drone = (2 + 3) * 4
    kf_lat = _medir_kf_latencia()
    pol = _latencia_ator_maddpg()          # referência de escala, não denominador
    orcamento_us = INTERVALO_DECISAO_S * 1e6
    frac_orc = kf_lat["mediana_us"] / orcamento_us
    merito = {"kf_bytes_por_drone": kf_bytes_por_drone,
              "kf_latencia": kf_lat, "politica_latencia": pol,
              "intervalo_decisao_s": INTERVALO_DECISAO_S,
              "fracao_orcamento": frac_orc, "por_Na": {}}

    print("Figuras de mérito — o KF contabilizado à parte (termo comum a WNN e MADDPG)\n")
    print("T2 — latência empírica por decisão (MEDIDA, protocolo SPEC-03)")
    print(f"  KF (comando+predição+correção): {kf_lat['mediana_us']:.2f} µs "
          f"(mediana)  {kf_lat['p99_us']:.2f} µs (p99)")
    print(f"  contra o orçamento de {INTERVALO_DECISAO_S:.0f} s por decisão: "
          f"{frac_orc:.4%} — desprezível contra o prazo de tempo real")
    if pol:
        print(f"  (escala: uma decisão do ator MADDPG é ~{pol['mediana_us']:.0f} µs, "
              f"SPEC-03/{pol['celula']} — mesma ordem, ambas << 2 s)")

    print("\nT1 — memória residente")
    for na in (2, 4, 8):
        merito["por_Na"][na] = {
            "fracao_bytes": kf_bytes_por_drone / (MADDPG_ATOR_KB[na] * 1024)}
    print(f"  estado do KF por drone: {kf_bytes_por_drone} bytes (x̂ 2 + P sim. 3, fp32)")

    print("\nLeitura: o KF é um custo fixo e pequeno, COMUM aos dois aproximadores")
    print(f"(cancela em H1). Contra o prazo de {INTERVALO_DECISAO_S:.0f} s da decisão é "
          f"~{frac_orc:.3%}; e o estado residente por drone são {kf_bytes_por_drone} bytes.\n")
    return merito


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def imagens(comp: dict, ladder: dict, merito: dict) -> None:
    plt = _plt()
    DIR.mkdir(parents=True, exist_ok=True)
    celulas = ["a_2-s_24", "a_4-s_24", "a_8-s_24"]

    # 1. Comparação com Lamenza (barras: publicado / limpo / bruto σ10 / KF σ10)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(celulas)); w = 0.2
    pub = [comp["celulas"][c]["publicado"] for c in celulas]
    d0 = [comp["celulas"][c]["degrau0"]["ct"] for c in celulas]
    d1 = [comp["celulas"][c]["sigmas"]["10"]["degrau1"]["ct"] for c in celulas]
    d4 = [comp["celulas"][c]["sigmas"]["10"]["degrau4"]["ct"] for c in celulas]
    ax.bar(x - 1.5 * w, pub, w, label="Lamenza (published)", color=CINZA)
    ax.bar(x - 0.5 * w, d0, w, label="step 0 (clean)", color=TEAL)
    ax.bar(x + 0.5 * w, d1, w, label="step 1 (raw, σ=10)", color=VERM)
    ax.bar(x + 1.5 * w, d4, w, label="step 4 (KF, σ=10)", color=AMBAR)
    ax.set_xticks(x); ax.set_xticklabels(celulas)
    ax.set_ylabel("completion time (s)")
    ax.set_title("MADDPG under noise vs Lamenza — the KF restores the published level")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "comparacao_lamenza.png", dpi=110); plt.close(fig)

    # 2. Degradação × σ (Δ do tempo contra o limpo), bruto vs KF
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    sig = [0, 10, 20]
    for ax, c in zip(axes, celulas):
        cel = comp["celulas"][c]
        base = cel["degrau0"]["ct"]
        bruto = [0.0] + [cel["sigmas"][str(s)]["degrau1"]["ct"] - base for s in (10, 20)]
        kf = [0.0] + [cel["sigmas"][str(s)]["degrau4"]["ct"] - base for s in (10, 20)]
        ax.plot(sig, bruto, "o-", color=VERM, label="raw (no filter)")
        ax.plot(sig, kf, "s-", color=AMBAR, label="KF+control")
        ax.axhline(0, color=CINZA, lw=1, ls=":")
        ax.set_title(c); ax.set_xlabel("GPS σ (m)")
    axes[0].set_ylabel("Δ time vs clean (s)")
    axes[0].legend(fontsize=8)
    axes[1].axvline(10, color=CINZA, alpha=0.3, lw=8)  # marca a grade declarada
    fig.suptitle("Completion-time degradation vs σ — raw grows, the KF stays flat")
    fig.tight_layout()
    fig.savefig(DIR / "degradacao_vs_sigma.png", dpi=110); plt.close(fig)

    # 3. NEES do filtro no ambiente real (da run da escada, 9 células × σ)
    cels9 = sorted(ladder["celulas"].keys())
    sigmas = [5, 10]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cels9)); w = 0.38
    for j, s in enumerate(sigmas):
        nees = [ladder["celulas"][c][f"sigma_{s}"]["degrau4"]["nees_medio"] for c in cels9]
        ax.bar(x + (j - 0.5) * w, nees, w, label=f"σ={s}")
    lo, hi = faixa_chi2_individual(2)
    ax.axhline(2.0, color=CINZA, ls=":", lw=1, label="NEES=2 target")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("a_", "").replace("-s_", "×") for c in cels9], rotation=45)
    ax.set_ylabel("mean NEES (log)")
    ax.set_title("KF NEES in the real environment — optimistic with Q=0 (above 2), worse at smaller σ")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "nees_real.png", dpi=110); plt.close(fig)

    # 4. RMSE do filtro vs a medida crua σ√2
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, s in enumerate(sigmas):
        rmse = [ladder["celulas"][c][f"sigma_{s}"]["degrau4"]["rmse_estimativa"] for c in cels9]
        ax.bar(x + (j - 0.5) * w, rmse, w, label=f"KF σ={s}")
        ax.axhline(s * np.sqrt(2), color=(VERM if s == 10 else AMBAR), ls="--", lw=1,
                   label=f"raw σ={s} (σ√2={s*np.sqrt(2):.1f})")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("a_", "").replace("-s_", "×") for c in cels9], rotation=45)
    ax.set_ylabel("position RMSE (m)")
    ax.set_title("KF error on real trajectories — well below the raw measurement")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "rmse_real.png", dpi=110); plt.close(fig)

    # 5. Figura de mérito: latência MEDIDA do KF contra o orçamento de 2 s (T2).
    #    O KF é termo COMUM (WNN e MADDPG usam o mesmo) — contabilizado à parte, não
    #    cobrado de nenhuma política. A referência é o prazo de decisão de tempo real;
    #    a política entra só como escala (mesma ordem do KF, ambas << 2 s).
    fig, ax = plt.subplots(figsize=(8.5, 5))
    orc_us = merito["intervalo_decisao_s"] * 1e6
    kf_lat = merito["kf_latencia"]["mediana_us"]
    pol = merito.get("politica_latencia")
    rot, val, cor = ["KF cycle\n(cmd+pred+corr)"], [kf_lat], [AMBAR]
    if pol:
        rot.append(f"MADDPG actor\n({pol['celula']}, for scale)")
        val.append(pol["mediana_us"]); cor.append(TEAL)
    rot.append(f"decision budget\n({merito['intervalo_decisao_s']:.0f} s)")
    val.append(orc_us); cor.append(CINZA)
    barras = ax.bar(rot, val, color=cor)
    for b, v in zip(barras, val):
        txt = f"{v / 1e6:.0f} s" if v >= 1e5 else f"{v:.0f} µs"
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_yscale("log"); ax.set_ylabel("measured latency per decision (µs, log)")
    ax.set_title("The KF is a shared constant, negligible vs the 2 s budget\n"
                 "measured (SPEC-03 protocol); the KF cancels between WNN and MADDPG in H1")
    fig.tight_layout()
    fig.savefig(DIR / "latencia_kf.png", dpi=110); plt.close(fig)

    print(f"[imagens] 5 figuras em {DIR}/")


def main() -> int:
    comp = _ler("reports/spec07-maddpg-vs-lamenza.json")
    ladder = _ler("reports/spec07-maddpg-ruido.json")
    merito = figuras_de_merito()
    try:
        imagens(comp, ladder, merito)
    except ImportError:
        print("[imagens] matplotlib ausente — pulei. Rode: uv sync --group dev")
    Path("reports/spec07-maddpg-merito.json").write_text(
        json.dumps(merito, indent=2, ensure_ascii=False))
    print("artefato: reports/spec07-maddpg-merito.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
