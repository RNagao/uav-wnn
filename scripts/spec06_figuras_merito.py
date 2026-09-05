"""Figuras de mérito WNN × MADDPG — custo (T1 memória, T2 inferência).

Usa o WNN **real treinado** (SPEC-06, célula 2/24) e o harness de custo da
SPEC-03 (`uavwnn.custo`) — nada de número inventado. Contrasta com o ator MADDPG
(256×256) da **mesma célula** (a2s24, Na=2) do cenário 003 de Lamenza.

**Comparação na mesma célula (a2s24, Na=2/Ns=24).** WNN e MADDPG são ambos dessa
célula — mesma tarefa, mesma observação, mesma resolução. T1 (memória), ciclos,
FLOPs e a latência empírica (SPEC-03) casam todos no mesmo modelo e na mesma célula,
sem a ressalva de "célula diferente" que versões anteriores (WNN em Na=1) carregavam.

O achado é um **trade-off de dois eixos**, não um vencedor único:
- **T2 (inferência):** a WNN é mais barata — em ciclos modelados (independente de
  hardware, RNF-12) e na **latência empírica MEDIDA** (SPEC-03), onde ela é ~2×
  mais rápida que o ator MADDPG no relógio, na mesma célula.
- **T1 (memória):** a WNN paga — as LUTs treinadas ocupam mais que os pesos densos.

    uv run python scripts/spec06_figuras_merito.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar  # noqa: E402
from uavwnn.custo.memoria import custo_mlp, custo_ntuple_de_rede  # noqa: E402
from uavwnn.custo.operacoes import (  # noqa: E402
    CICLOS, ciclos, operacoes_mlp, operacoes_ntuple,
)
from uavwnn.refenv import LAMENZA_003  # noqa: E402

DIR = Path("reports/img/spec06-merito")
TEAL, AMBAR, CINZA, VERM = "#1C7293", "#F2A93B", "#808A96", "#B0637C"

# WNN treinados (SPEC-06), célula 2/24 — a MESMA do MADDPG e da latência (a2s24):
# T=32, n=12, N=72, D=27, d_saida=3.
WNN = {
    # Modelos de DEPLOY — o artefato que de fato voa (ator + termômetro
    # exportados), sob `models/` (versionado, estável entre máquinas; `runs/` é
    # gitignored e some). Célula **a2s24 (Na=2)**, a MESMA do checkpoint MADDPG e da
    # latência medida (SPEC-03): T1, ciclos, FLOPs e latência casam no mesmo modelo
    # E na mesma célula — sem a ressalva de "célula diferente".
    "T=32 (2/24)": "models/deploy/T32/a2s24",
    "T=16 (2/24)": "models/deploy/T16/a2s24",
}
FAMILIA = {k: "ntupla_dissertacao" for k in WNN}
RESOLUCAO = {"T=32 (2/24)": 32, "T=16 (2/24)": 16}
N_TUPLA, D_OBS, D_SAIDA = 12, 27, 3
T, N_RAM = 32, 72          # da célula de referência (a2s24 T=32), para o T2 (ciclos)
# Ator MADDPG (256×256), obs por célula (D depende de Na). Só Na=2: é a célula
# (a2s24) com latência medida e o comparador único do WNN×MADDPG; Na=4/8 eram
# ~iguais em tamanho/operações e só repetiam a linha.
MADDPG_D = {2: 27}
# Latência empírica (T2): única célula com WNN de deploy E checkpoint MADDPG.
CELULA_LATENCIA = (2, 24)
# Alvo do eixo de computação do mapa de mérito (o mesmo hardware da medição).
HW_MAPA = "x86-64 (com FPU e SIMD)"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _latencia_medida(na: int = CELULA_LATENCIA[0],
                     ns: int = CELULA_LATENCIA[1]) -> dict | None:
    """Latência MEDIDA (não modelada) da decisão, do artefato da SPEC-03.

    A SPEC-03 mede a decisão completa (obs → ação) com um processo por modelo, a
    MESMA observação para os três e a menor mediana de 8×1000 repetições. Aqui só
    consumimos esse artefato — o número honesto de wall-clock, que substituiu o
    valor hardcoded que contradizia a própria medição (dizia a WNN mais lenta; a
    medição mostra a WNN ~2× mais rápida).
    """
    art = Path(f"reports/spec03-latencia-memoria-a{na}s{ns}.json")
    if not art.exists():
        print(f"  aviso: {art} ausente — rode `uv run --group baseline python "
              f"scripts/spec03_latencia_memoria.py --na {na} --ns {ns}` antes, "
              f"para a figura de latência empírica", flush=True)
        return None
    linhas = json.loads(art.read_text())
    por_modelo = {r["modelo"]: {"mediana_us": r["mediana_us"], "p99_us": r["p99_us"]}
                  for r in linhas}
    return {"celula": f"a{na}s{ns}", "por_modelo": por_modelo}


def _perfil_ops(op) -> dict:
    """Perfil de operações por decisão + FLOPs em duas convenções.

    Exposto como APOIO (transparência), não como veredito: o FLOP subestima a WNN
    por construção — o trabalho dela é `lookups` + `soma_int` + `op_bits`, que NÃO
    são FLOPs. `flops_mult` (só multiplicação) é a convenção enviesada; `flops_total`
    (mult+add) é a padrão de ML. A coluna `nao_flop` é o que qualquer FLOP esconde.
    """
    return {
        "mult_fp": op.mult_fp, "soma_fp": op.soma_fp, "div_fp": op.div_fp,
        "transcendental_fp": op.transcendental_fp,
        "lookups": op.lookups, "soma_int": op.soma_int, "op_bits": op.op_bits,
        "comparacoes": op.comparacoes,
        "flops_mult": op.mult_fp,
        "flops_total": op.mult_fp + op.soma_fp + op.div_fp + op.transcendental_fp,
        "nao_flop": op.lookups + op.soma_int + op.op_bits,
    }


def coletar() -> dict:
    cfg = LAMENZA_003(2, 24, phase="eval")
    m: dict = {"wnn": {}, "maddpg": {}}

    for nome, run in WNN.items():
        ck = Path(run)
        if not ck.exists():
            print(f"  aviso: {ck} ausente — {nome} fica FORA da tabela", flush=True)
            continue
        _, ap = montar(FAMILIA[nome], cfg, ck)
        # As TRÊS variantes, e a `ordenado` como principal.
        #
        # O `hash` era o número publicado, e ele paga arredondamento de capacidade
        # para potência de 2 com fator de carga 0,75: 1 059 entradas ocupadas viram
        # 2 048 posições alocadas. Isso inflava o T1 em 1,85× — e o T1 é justamente
        # a figura em que a WNN PERDE, então inflá-lo é errar contra nós mesmos.
        #
        # A `ordenado` (chaves ordenadas + busca binária) gasta exatamente
        # `ocupadas × (chave + valores)`, sem folga. O preço é busca O(log n) em vez
        # de O(1) — ~11 comparações por RAM aqui — e por isso as três ficam
        # gravadas: quem defender latência tem de citar o `hash`.
        t_res = RESOLUCAO[nome]
        var = {v: custo_ntuple_de_rede(ap.ator, variante=v, limiares_por_dim=t_res,
                                       n_dims=D_OBS, rotulo=nome)
               for v in ("ordenado", "hash", "denso")}
        cm = var["ordenado"]
        op = operacoes_ntuple(n_ram=ap.ator.n_ram, tamanho_tupla=N_TUPLA,
                              n_cabecas=D_SAIDA, n_dims=D_OBS, n_niveis=t_res)
        m["wnn"][nome] = {
            "persistente_kb": cm.persistente / 1024,
            "variante_principal": "ordenado",
            "persistente_kb_por_variante": {v: c.persistente / 1024
                                            for v, c in var.items()},
            "residente_kb": cm.residente / 1024,
            "entradas_ocupadas": int(ap.ator.entradas_ocupadas()),
            # Computação em CICLOS (não mais mult. fp): a WNN quase não multiplica,
            # então contar multiplicação a subestimava por construção. Ciclos somam
            # TODAS as operações (lookups, somas int) e é o que o mapa de mérito usa.
            "ciclos_ref": ciclos(op, HW_MAPA),
            # Perfil de FLOPs como APOIO (não veredito) — ver _perfil_ops.
            "operacoes": _perfil_ops(op),
        }

    for na, d in MADDPG_D.items():
        cm = custo_mlp([d, 256, 256, 2], rotulo=f"MADDPG Na={na}")
        op = operacoes_mlp([d, 256, 256, 2])
        m["maddpg"][f"Na={na}"] = {"persistente_kb": cm.persistente / 1024,
                                   "ciclos_ref": ciclos(op, HW_MAPA), "D": d,
                                   "operacoes": _perfil_ops(op)}

    # T2 — tempo de inferência por CICLOS (independente de hardware, RNF-12): a
    # figura honesta, porque wall-clock depende do arcabouço. WNN real vs ator
    # MADDPG (256×256, saída 2 como Lamenza).
    ow = operacoes_ntuple(n_ram=N_RAM, tamanho_tupla=N_TUPLA, n_cabecas=D_SAIDA,
                          n_dims=D_OBS, n_niveis=T)
    om = operacoes_mlp([MADDPG_D[2], 256, 256, 2])
    m["t2_ciclos"] = {hw: {"wnn": ciclos(ow, hw), "maddpg": ciclos(om, hw),
                           "ganho_wnn": ciclos(om, hw) / max(ciclos(ow, hw), 1)}
                      for hw in CICLOS}
    # T2 latência empírica MEDIDA (SPEC-03, RNF-11): decisão completa obs→ação, um
    # processo por modelo, mesma observação, menor mediana de 8×1000. Consumida do
    # artefato — nada re-medido nem hardcoded aqui.
    m["latencia_us"] = _latencia_medida()
    return m


def tabela(m: dict) -> None:
    print("Figuras de mérito — memória (T1) e inferência (T2)\n")
    print(f"{'modelo':>22} {'T1 memória':>12}")
    for nome, w in m["wnn"].items():
        print(f"{'WNN '+nome:>22} {w['persistente_kb']:>9.1f} KB")
    for na, mm in m["maddpg"].items():
        print(f"{'MADDPG '+na:>22} {mm['persistente_kb']:>9.1f} KB")
    d = m["wnn"].get("T=32 (2/24)")
    if d:
        maddpg2 = m["maddpg"]["Na=2"]
        pv = d["persistente_kb_por_variante"]
        print(f"\nLeitura (T1): a WNN paga em memória — {d['persistente_kb']:.0f} KB "
              f"(LUTs, {d['entradas_ocupadas']:,} entradas, variante ordenado) contra "
              f"{maddpg2['persistente_kb']:.0f} KB do ator MADDPG "
              f"({d['persistente_kb']/maddpg2['persistente_kb']:.1f}×). "
              f"Trade-off memória↔computação.")
        print(f"  por variante de armazenamento: "
              + " · ".join(f"{v} {kb:.0f} KB" for v, kb in pv.items())
              + "  (a `hash` compra O(1) pagando arredondamento de capacidade)")
    print("\nT2 — ciclos por decisão (modelado, independente de hardware, RNF-12):")
    for hw, c in m["t2_ciclos"].items():
        print(f"  {hw:>26}: WNN {c['wnn']:>8,}  MADDPG {c['maddpg']:>10,}  "
              f"→ {c['ganho_wnn']:.0f}× WNN")
    lat = m.get("latencia_us")
    if lat:
        pm = lat["por_modelo"]
        print(f"\nT2 — latência empírica MEDIDA (SPEC-03, célula {lat['celula']}, "
              f"menor mediana de 8×1000):")
        for modelo, v in pm.items():
            print(f"  {modelo:>10}: {v['mediana_us']:>7.1f} µs (mediana)  "
                  f"{v['p99_us']:>7.1f} µs (p99)")
        wnn = [v["mediana_us"] for k, v in pm.items() if k.startswith("wnn")]
        mad = pm.get("maddpg")
        if wnn and mad:
            print(f"  → a WNN é ~{mad['mediana_us']/min(wnn):.1f}× mais rápida que o "
                  f"ator MADDPG no relógio (NumPy × PyTorch, mesma célula)")

    # Perfil de operações — APOIO, não veredito. FLOP subestima a WNN (o trabalho
    # dela é lookup + inteiro, que não é FLOP); por isso a coluna nao_flop existe.
    print("\nPerfil de operações por decisão (apoio; o veredito de computação é a "
          "latência acima). Colunas por tipo — some as fp p/ o total, ou olhe só "
          "'mult fp' p/ a convenção só-multiplicação:")
    print(f"  {'modelo':>14} | {'mult fp':>8} {'soma fp':>8} {'div fp':>7} "
          f"| {'lookups':>8} {'soma int':>9} {'bits':>7}")
    linhas_perfil = ([(f"WNN {n}", w["operacoes"]) for n, w in m["wnn"].items()]
                     + [(f"MADDPG {na}", mm["operacoes"]) for na, mm in m["maddpg"].items()])
    for nome, o in linhas_perfil:
        print(f"  {nome:>14} | {o['mult_fp']:>8,} {o['soma_fp']:>8,} {o['div_fp']:>7,} "
              f"| {o['lookups']:>8,} {o['soma_int']:>9,} {o['op_bits']:>7,}")
    diss_op = m["wnn"].get("T=32 (2/24)", {}).get("operacoes")
    mad2_op = m["maddpg"].get("Na=2", {}).get("operacoes")
    if diss_op and mad2_op:
        print(f"  → só multiplicação: {mad2_op['flops_mult']/max(diss_op['flops_mult'],1):,.0f}× "
              f"(enviesado) · mult+add: {mad2_op['flops_total']/diss_op['flops_total']:.0f}× · "
              f"mas ambos ESCONDEM {diss_op['nao_flop']:,} ops não-FLOP da WNN "
              f"(lookup+inteiro+bits) — por isso o relógio medido dá só ~2×.")


def figuras(m: dict) -> None:
    plt = _plt()
    DIR.mkdir(parents=True, exist_ok=True)
    diss = m["wnn"].get("T=32 (2/24)")

    # 1. T2 — latência empírica MEDIDA (SPEC-03): mediana em barra, p99 na haste.
    #    Substitui a antiga T3 (mult. fp): contar multiplicação favorecia a WNN por
    #    construção (ela quase não multiplica); o relógio medido é o custo real.
    lat = m.get("latencia_us")
    if lat:
        pm = lat["por_modelo"]
        ordem = [k for k in ("wnn-T16", "wnn-T32", "maddpg") if k in pm]
        fig, ax = plt.subplots(figsize=(8.5, 5))
        med = [pm[k]["mediana_us"] for k in ordem]
        p99 = [pm[k]["p99_us"] for k in ordem]
        cor = [AMBAR if k.startswith("wnn") else TEAL for k in ordem]
        barras = ax.bar(ordem, med, color=cor, width=0.55)
        ax.vlines(range(len(ordem)), med, p99, color="#333", lw=1.5)
        ax.scatter(range(len(ordem)), p99, color="#333", s=18, zorder=3, label="p99")
        for b, mm in zip(barras, med):
            ax.annotate(f"{mm:.0f} µs", (b.get_x() + b.get_width() / 2, mm),
                        ha="center", va="bottom", fontsize=10, weight="bold")
        ax.set_ylabel("measured latency per decision (µs)")
        ax.set_title("Measured inference latency (SPEC-03, cell "
                     f"{lat['celula']})\nlowest median of 8×1000 · one process/model "
                     "· NumPy WNN × PyTorch MADDPG", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(DIR / "latencia_empirica.png", dpi=110); plt.close(fig)

    # 2. T1 — memória persistente (KB), log
    fig, ax = plt.subplots(figsize=(8, 5))
    rot, val, cor = [], [], []
    for nome, w in m["wnn"].items():
        rot.append(f"WNN\n{nome}"); val.append(w["persistente_kb"]); cor.append(AMBAR)
    for na, mm in m["maddpg"].items():
        rot.append(f"MADDPG\n{na}"); val.append(mm["persistente_kb"]); cor.append(TEAL)
    ax.bar(rot, val, color=cor)
    ax.set_yscale("log"); ax.set_ylabel("persistent memory — model size (KB, log)")
    ax.set_title("Model size (persistent memory per model): scales with resolution\n"
                 "same cell a2s24 · T=32 ≈2× MADDPG, T=16 ≈ties · `hash` variant ~1.9×",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(DIR / "t1_memoria.png", dpi=110); plt.close(fig)

    # 3. trade-off: memória × computação (o mapa de mérito), computação em CICLOS
    hw_curto = HW_MAPA.split(" (")[0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if diss:
        ax.scatter(diss["ciclos_ref"], diss["persistente_kb"], s=140, color=AMBAR,
                   zorder=3, label="WNN T=32 (2/24)")
    for na, mm in m["maddpg"].items():
        ax.scatter(mm["ciclos_ref"], mm["persistente_kb"], s=120, color=TEAL, marker="^", zorder=3)
    ax.annotate("MADDPG\n(Na=2)", (list(m["maddpg"].values())[0]["ciclos_ref"],
                list(m["maddpg"].values())[0]["persistente_kb"]),
                textcoords="offset points", xytext=(8, 8), color=TEAL)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"computation — cycles per decision ({hw_curto}, log)")
    ax.set_ylabel("persistent memory (KB, log)")
    ax.set_title("Merit map: the WNN trades memory for computation")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "mapa_merito.png", dpi=110); plt.close(fig)

    # 4. T2 — ciclos por decisão, por classe de hardware (log)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    hws = list(m["t2_ciclos"].keys())
    x = np.arange(len(hws)); w = 0.38
    ax.bar(x - w / 2, [m["t2_ciclos"][h]["wnn"] for h in hws], w,
           color=AMBAR, label="WNN (thesis)")
    ax.bar(x + w / 2, [m["t2_ciclos"][h]["maddpg"] for h in hws], w,
           color=TEAL, label="MADDPG 256×256")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([h.split(" (")[0] for h in hws])
    ax.set_ylabel("cycles per decision (log)")
    ax.set_title("Inference cost — modeled cycles per decision: WNN wins, even more without an FPU (470× on the M0)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(DIR / "t2_ciclos.png", dpi=110); plt.close(fig)
    n = 4 if m.get("latencia_us") else 3
    print(f"[figuras] {n} imagens em {DIR}/")


def main() -> int:
    m = coletar()
    tabela(m)
    try:
        figuras(m)
    except ImportError:
        print("[figuras] matplotlib ausente — pulei. Rode: uv sync --group dev")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/spec06-merito.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print("artefato: reports/spec06-merito.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
