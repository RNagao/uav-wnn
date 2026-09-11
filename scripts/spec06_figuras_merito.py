"""Figuras de mérito WNN × MADDPG — custo (T1 pico de RAM, T2 inferência), numa célula só.

Todos os modelos comparados vêm da **mesma célula** `(Na, Ns)`: os atores WNN de
deploy em `models/deploy/T{T}/a{Na}s{Ns}` e o ator MADDPG (256×256) com a
observação dessa célula (`D = cfg.observation_size`). Nada de célula diferente por
barra — uma figura de memória que põe a WNN de 1 drone ao lado do MADDPG de 2 não
compara os métodos, compara tarefas.

**T1 é o pico de RAM MEDIDO** — `VmHWM` do processo que carrega o modelo e decide,
um processo por modelo, do artefato da SPEC-03 (`scripts/spec03_latencia_memoria.py`).
Não há mais contagem de bytes de artefato ("persistente"): a pergunta que a figura
responde é "quanta RAM o dispositivo precisa ter para esta rede rodar", e isso se
mede, não se modela. A latência empírica (T2) vem do mesmo artefato; os ciclos
modelados (T2, RNF-12) e o perfil de operações vêm do harness `uavwnn.custo.operacoes`.

O achado é um **trade-off de dois eixos**, não um vencedor único:
- **T2 (inferência):** a WNN é mais barata — em ciclos modelados e na latência
  MEDIDA, onde ela é ~2× mais rápida que o ator MADDPG no relógio.
- **T1 (pico de RAM):** o processo da WNN (NumPy) tem pico ~10× menor que o do
  MADDPG (PyTorch); descontado o arcabouço, o acréscimo do modelo é maior na WNN
  (o load constrói a rede de dicionários antes de achatá-la em LUT).

    uv run python scripts/spec06_figuras_merito.py                 # célula padrão a2s24
    uv run python scripts/spec06_figuras_merito.py --na 4 --ns 24  # outra célula
    uv run python scripts/spec06_figuras_merito.py --resolucoes 16 # só T=16

A célula padrão escreve `reports/spec06-merito.json` e `reports/img/spec06-merito/`;
qualquer outra célula recebe o sufixo `-a{Na}s{Ns}` para não sobrescrever a canônica.
Sem o artefato de medição da célula, a tabela sai sem T1 e sem latência — rode antes
`uv run --group baseline python scripts/spec03_latencia_memoria.py --na N --ns S`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar  # noqa: E402
from uavwnn.custo.operacoes import (  # noqa: E402
    CICLOS, ciclos, operacoes_mlp, operacoes_ntuple,
)
from uavwnn.refenv import LAMENZA_003  # noqa: E402

CELULA_PADRAO = (2, 24)
RESOLUCOES_PADRAO = (16, 32)
FAMILIA = "ntupla_dissertacao"
N_TUPLA, D_SAIDA = 12, 3
MADDPG_OCULTAS = [256, 256]
MADDPG_SAIDA = 2
TEAL, AMBAR, CINZA, VERM = "#1C7293", "#F2A93B", "#808A96", "#B0637C"
# Alvo do eixo de computação do mapa de mérito (o mesmo hardware da medição).
HW_MAPA = "x86-64 (com FPU e SIMD)"
CHAVES_RAM = ("pico_total_kb", "pico_arcabouco_kb", "pico_modelo_kb",
              "rss_apos_carga_kb", "arcabouco")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def rotulo_celula(na: int, ns: int) -> str:
    return f"a{na}s{ns}"


def caminho_deploy(t: int, na: int, ns: int) -> Path:
    return Path(f"models/deploy/T{t}/{rotulo_celula(na, ns)}")


def _procedencia(ck: Path) -> dict:
    p = ck / "procedencia.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _medicao(na: int, ns: int) -> dict[str, dict]:
    """Latência e pico de RAM MEDIDOS, do artefato da SPEC-03 da célula.

    A SPEC-03 mede a decisão completa (obs → ação) com um processo por modelo, a
    MESMA observação para os três (calculada fora dos filhos) e a menor mediana de
    8×1000 repetições; o pico de RAM é o `VmHWM` desse processo. Aqui só consumimos
    esse artefato, indexado por modelo (`wnn-T16`, `wnn-T32`, `maddpg`).
    """
    art = Path(f"reports/spec03-latencia-memoria-{rotulo_celula(na, ns)}.json")
    if not art.exists():
        print(f"  aviso: {art} ausente — rode `uv run --group baseline python "
              f"scripts/spec03_latencia_memoria.py --na {na} --ns {ns}`; a tabela "
              "sai SEM T1 (pico de RAM) e sem latência", flush=True)
        return {}
    return {r["modelo"]: r for r in json.loads(art.read_text())}


def _agregado_rep(na: int, ns: int) -> tuple[dict, int] | None:
    """Maior agregado de repetições da célula (`spec03-pico-ram-rep*-a{na}s{ns}.json`).

    É de onde saem a média e o desvio do pico total: N processos novos por modelo.
    Sem o arquivo, a tabela e as figuras usam a medição única.
    """
    import re
    melhor = None
    for f in Path("reports").glob(f"spec03-pico-ram-rep*-{rotulo_celula(na, ns)}.json"):
        m = re.search(r"rep(\d+)-", f.name)
        n = int(m.group(1)) if m else 0
        if melhor is None or n > melhor[0]:
            melhor = (n, f)
    if melhor is None:
        return None
    d = json.loads(melhor[1].read_text())
    return d["agregado"], d["repeticoes"]


def _ram(med: dict | None, rep: dict | None = None) -> dict:
    """Pico de RAM do modelo: a medição única mais, se houver, média ± desvio das
    repetições (`pico_total_media_kb`, `pico_total_desvio_kb`, `n_rep`)."""
    if not med:
        return {}
    r = {k: med[k] for k in CHAVES_RAM if k in med}
    if rep and "pico_total_kb" in rep:
        r["pico_total_media_kb"] = rep["pico_total_kb"]["media"]
        r["pico_total_desvio_kb"] = rep["pico_total_kb"]["desvio"]
        r["n_rep"] = rep["n"]
    return r


def _pico(ram: dict) -> float:
    """O pico total a citar: média das repetições se houver, senão a medição única."""
    return ram.get("pico_total_media_kb", ram["pico_total_kb"])


def _perfil_ops(op) -> dict:
    """Perfil por TIPO de operação — sem pré-fixar uma convenção de FLOP.

    `flops_mult` é a convenção "só multiplicação" (enviesada a favor da WNN),
    `flops_total` soma mult+add+div+transcendental, e `nao_flop` é o trabalho
    (lookups, somas inteiras, bits) que nenhuma convenção de FLOP enxerga.
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


def coletar(na: int, ns: int, resolucoes: tuple[int, ...]) -> dict:
    cel = rotulo_celula(na, ns)
    cfg = LAMENZA_003(na, ns, phase="eval")
    d_obs = cfg.observation_size
    medido = _medicao(na, ns)
    rep = _agregado_rep(na, ns)
    agg = rep[0] if rep else {}
    m: dict = {"celula": {"na": na, "ns": ns, "rotulo": cel, "d_obs": d_obs},
               "wnn": {}, "maddpg": {}, "repeticoes": rep[1] if rep else None}

    for t in resolucoes:
        nome = f"T={t} ({cel})"
        ck = caminho_deploy(t, na, ns)
        if not ck.exists():
            print(f"  aviso: {ck} ausente — {nome} fica FORA da tabela", flush=True)
            continue
        _, ap = montar(FAMILIA, cfg, ck)
        op = operacoes_ntuple(n_ram=ap.ator.n_ram, tamanho_tupla=N_TUPLA,
                              n_cabecas=D_SAIDA, n_dims=d_obs, n_niveis=t)
        proc = _procedencia(ck)
        m["wnn"][nome] = {
            "resolucao": t,
            "caminho": str(ck),
            "passos_treino": proc.get("passos"),
            "podado": bool((proc.get("deploy") or {}).get("poda")),
            "n_ram": int(ap.ator.n_ram),
            "entradas_ocupadas": int(ap.ator.entradas_ocupadas()),
            # T1 — pico de RAM MEDIDO (SPEC-03), não bytes de artefato.
            "ram": _ram(medido.get(f"wnn-T{t}"), agg.get(f"wnn-T{t}")),
            # Computação em CICLOS: a WNN quase não multiplica, então contar
            # multiplicação a subestimava por construção. Ciclos somam TODAS as
            # operações (lookups, somas int) e é o que o mapa de mérito usa.
            "ciclos_ref": ciclos(op, HW_MAPA),
            "operacoes": _perfil_ops(op),
            "_op": op,
        }

    camadas = [d_obs, *MADDPG_OCULTAS, MADDPG_SAIDA]
    om = operacoes_mlp(camadas)
    m["maddpg"][cel] = {"ram": _ram(medido.get("maddpg"), agg.get("maddpg")),
                        "params": sum(a * b + b for a, b in zip(camadas[:-1], camadas[1:])),
                        "ciclos_ref": ciclos(om, HW_MAPA), "D": d_obs,
                        "camadas": camadas, "operacoes": _perfil_ops(om)}

    # T2 — ciclos por decisão por classe de hardware (RNF-12), WNN de referência =
    # a maior resolução carregada da célula (a mais cara em lookups).
    ref = _wnn_referencia(m)
    if ref:
        ow = ref["_op"]
        m["t2_ciclos"] = {hw: {"wnn": ciclos(ow, hw), "maddpg": ciclos(om, hw),
                               "ganho_wnn": ciclos(om, hw) / max(ciclos(ow, hw), 1)}
                          for hw in CICLOS}
        m["t2_referencia_wnn"] = _nome_referencia(m)
    for w in m["wnn"].values():
        w.pop("_op", None)
    if medido:
        m["latencia_us"] = {
            "celula": cel,
            "por_modelo": {k: {"mediana_us": v["mediana_us"], "p99_us": v["p99_us"]}
                           for k, v in medido.items()},
        }
    else:
        m["latencia_us"] = None
    return m


def _nome_referencia(m: dict) -> str | None:
    if not m["wnn"]:
        return None
    return max(m["wnn"], key=lambda k: m["wnn"][k]["resolucao"])


def _wnn_referencia(m: dict) -> dict | None:
    nome = _nome_referencia(m)
    return m["wnn"][nome] if nome else None


def _unico_maddpg(m: dict) -> tuple[str, dict]:
    (k, v), = m["maddpg"].items()
    return k, v


def _mb(kb: float) -> str:
    return f"{kb / 1024:,.1f} MB"


def tabela(m: dict) -> None:
    cel = m["celula"]["rotulo"]
    print(f"Figuras de mérito — pico de RAM (T1) e inferência (T2), célula {cel} "
          f"(Na={m['celula']['na']}, Ns={m['celula']['ns']}, D={m['celula']['d_obs']})\n")
    n_rep = m.get("repeticoes")
    col = f"pico total (média ± dp, n={n_rep})" if n_rep else "pico total"
    print(f"{'modelo':>22} {col:>32} {'entradas':>10} {'treino':>8}")
    _, mad = _unico_maddpg(m)

    def linha_ram(rot, ram, n, treino, extra=""):
        if ram and "pico_total_media_kb" in ram:
            v = f"{ram['pico_total_media_kb']/1024:,.2f} ± {ram['pico_total_desvio_kb']/1024:,.2f} MB"
            print(f"{rot:>22} {v:>32} {n:>10,} {treino:>8}{extra}")
        elif ram:
            print(f"{rot:>22} {_mb(ram['pico_total_kb']):>32} {n:>10,} {treino:>8}{extra}")
        else:
            print(f"{rot:>22} {'— (sem medição)':>32} {n:>10,} {treino:>8}{extra}")

    for nome, w in m["wnn"].items():
        passos = f"{w['passos_treino']/1e6:.2f}M" if w["passos_treino"] else "?"
        pod = "" if w["podado"] else "  (não podado)"
        linha_ram("WNN " + nome, w["ram"], w["entradas_ocupadas"], passos, pod)
    for k, mm in m["maddpg"].items():
        linha_ram("MADDPG " + k, mm["ram"], mm["params"], "50M")

    if m["wnn"] and mad["ram"]:
        print("\nLeitura (T1 = pico de RAM do processo, VmHWM), mesma célula:")
        for nome, w in m["wnn"].items():
            if not w["ram"]:
                continue
            r = _pico(w["ram"]) / _pico(mad["ram"])
            print(f"  WNN {nome}: pico {_mb(_pico(w['ram']))} = {r:.2f}× o "
                  f"processo do MADDPG ({_mb(_pico(mad['ram']))})")
        print(f"  arcabouço: WNN em {next(iter(m['wnn'].values()))['ram'].get('arcabouco', '?')}, "
              f"MADDPG em {mad['ram'].get('arcabouco', '?')} — o pico do MADDPG é quase "
              "todo biblioteca (RNF-11: arcabouços diferentes); a decomposição "
              "arcabouço/modelo fica no relatório da SPEC-03")
        orcamentos = {w["passos_treino"] for w in m["wnn"].values()}
        if any(p and p < 50_000_000 for p in orcamentos):
            print("  ATENÇÃO: WNN desta célula treinada com menos passos que o MADDPG "
                  "(50M); a LUT cresce com o treino, então o pico é um piso, não o "
                  "número final.")

    if "t2_ciclos" in m:
        print(f"\nT2 — ciclos por decisão (modelado, RNF-12; WNN = {m['t2_referencia_wnn']}):")
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
        madl = pm.get("maddpg")
        if wnn and madl:
            print(f"  → a WNN é ~{madl['mediana_us']/min(wnn):.1f}× mais rápida que o "
                  f"ator MADDPG no relógio (NumPy × PyTorch, mesma célula)")

    print("\nPerfil de operações por decisão (apoio; o veredito de computação é a "
          "latência acima). Colunas por tipo — some as fp p/ o total, ou olhe só "
          "'mult fp' p/ a convenção só-multiplicação:")
    print(f"  {'modelo':>20} | {'mult fp':>8} {'soma fp':>8} {'div fp':>7} "
          f"| {'lookups':>8} {'soma int':>9} {'bits':>7}")
    linhas_perfil = ([(f"WNN {n}", w["operacoes"]) for n, w in m["wnn"].items()]
                     + [(f"MADDPG {k}", mm["operacoes"]) for k, mm in m["maddpg"].items()])
    for nome, o in linhas_perfil:
        print(f"  {nome:>20} | {o['mult_fp']:>8,} {o['soma_fp']:>8,} {o['div_fp']:>7,} "
              f"| {o['lookups']:>8,} {o['soma_int']:>9,} {o['op_bits']:>7,}")
    ref = _wnn_referencia(m)
    if ref:
        ro, mo = ref["operacoes"], mad["operacoes"]
        print(f"  → só multiplicação: {mo['flops_mult']/max(ro['flops_mult'],1):,.0f}× "
              f"(enviesado) · mult+add: {mo['flops_total']/ro['flops_total']:.0f}× · "
              f"mas ambos ESCONDEM {ro['nao_flop']:,} ops não-FLOP da WNN "
              f"(lookup+inteiro+bits) — por isso o relógio medido dá só ~2×.")


def figuras(m: dict, saida: Path) -> None:
    plt = _plt()
    saida.mkdir(parents=True, exist_ok=True)
    cel = m["celula"]["rotulo"]
    na, ns = m["celula"]["na"], m["celula"]["ns"]
    _, mad = _unico_maddpg(m)
    n_fig = 0
    com_ram = bool(mad["ram"]) and all(w["ram"] for w in m["wnn"].values())

    # 1. T2 — latência empírica MEDIDA (SPEC-03): mediana em barra, p99 na haste.
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
        ax.set_title(f"Measured inference latency (SPEC-03, cell {lat['celula']})\n"
                     "lowest median of 8×1000 · one process/model · NumPy WNN × PyTorch MADDPG",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(saida / "latencia_empirica.png", dpi=110); plt.close(fig)
        n_fig += 1

    # 2. T1 — pico de RAM MEDIDO (MB, log), MESMA célula para todas as barras.
    #    SÓ o pico total: uma barra por modelo, média ± desvio das repetições quando
    #    o agregado da célula existe. Log porque o PyTorch põe o processo do MADDPG
    #    uma ordem de grandeza acima das WNN.
    if com_ram:
        fig, ax = plt.subplots(figsize=(8, 5))
        rot = [f"MADDPG\nNa={na}, Ns={ns}"]
        rams = [mad["ram"]]
        cor = [TEAL]
        for nome, w in m["wnn"].items():
            rot.append(f"WNN T={w['resolucao']}\nNa={na}, Ns={ns}")
            rams.append(w["ram"])
            cor.append(AMBAR)
        com_rep = all("pico_total_media_kb" in r for r in rams)
        tot = [_pico(r) / 1024 for r in rams]
        err = [r.get("pico_total_desvio_kb", 0.0) / 1024 for r in rams]
        barras = ax.bar(rot, tot, color=cor, width=0.55,
                        yerr=err if com_rep else None, capsize=5, ecolor="#333")
        for b, t, e in zip(barras, tot, err):
            txt = f"{t:,.1f} ± {e:,.1f} MB" if com_rep else f"{t:,.1f} MB"
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, t + e), ha="center",
                        va="bottom", fontsize=10, weight="bold", xytext=(0, 3),
                        textcoords="offset points")
        ax.set_yscale("log"); ax.set_ylim(10, max(tot) * 4)
        ax.set_ylabel("peak RSS of the process — VmHWM (MB, log)")
        razoes = " · ".join(f"T={w['resolucao']} {_pico(w['ram'])/_pico(mad['ram']):.2f}×"
                            for w in m["wnn"].values())
        sub = (f"mean ± sd over {m['repeticoes']} fresh processes per model" if com_rep
               else "one process per model")
        ax.set_title(f"Peak RAM to load and run the actor, same cell {cel}: "
                     f"WNN / MADDPG = {razoes}\n"
                     f"WNN in {rams[-1].get('arcabouco', 'NumPy')}, MADDPG in "
                     f"{mad['ram'].get('arcabouco', 'PyTorch')} · {sub}", fontsize=10)
        fig.tight_layout(); fig.savefig(saida / "t1_pico_ram.png", dpi=110); plt.close(fig)
        n_fig += 1

        # 3. trade-off: pico de RAM × computação (o mapa de mérito), computação em CICLOS
        hw_curto = HW_MAPA.split(" (")[0]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for nome, w in m["wnn"].items():
            ax.scatter(w["ciclos_ref"], _pico(w["ram"]) / 1024, s=140,
                       color=AMBAR, zorder=3, label=f"WNN {nome}")
        ax.scatter(mad["ciclos_ref"], _pico(mad["ram"]) / 1024, s=120,
                   color=TEAL, marker="^", zorder=3, label=f"MADDPG ({cel})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"computation — cycles per decision ({hw_curto}, log)")
        ax.set_ylabel("peak RAM of the process (MB, log)")
        ax.set_title(f"Merit map, cell {cel}: measured peak RAM × modeled cycles")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(saida / "mapa_merito.png", dpi=110); plt.close(fig)
        n_fig += 1
    else:
        print("  [figuras] sem medição de pico de RAM da célula — t1_pico_ram.png e "
              "mapa_merito.png não gerados")

    print(f"[figuras] {n_fig} imagens em {saida}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--na", type=int, default=CELULA_PADRAO[0], help="número de drones")
    ap.add_argument("--ns", type=int, default=CELULA_PADRAO[1], help="número de sensores")
    ap.add_argument("--resolucoes", type=int, nargs="+", default=list(RESOLUCOES_PADRAO),
                    help="resoluções T do termômetro a incluir (deploy em models/deploy/T{T}/)")
    args = ap.parse_args()

    m = coletar(args.na, args.ns, tuple(args.resolucoes))
    if not m["wnn"]:
        print(f"nenhum ator WNN de deploy na célula {m['celula']['rotulo']} — nada a comparar")
        return 1
    tabela(m)
    padrao = (args.na, args.ns) == CELULA_PADRAO
    sufixo = "" if padrao else f"-{m['celula']['rotulo']}"
    saida = Path("reports/img/spec06-merito" + sufixo)
    try:
        figuras(m, saida)
    except ImportError:
        print("[figuras] matplotlib ausente — pulei. Rode: uv sync --group dev")
    Path("reports").mkdir(exist_ok=True)
    art = Path(f"reports/spec06-merito{sufixo}.json")
    art.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"artefato: {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
