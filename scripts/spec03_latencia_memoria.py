#!/usr/bin/env python3
"""Latência de inferência e PICO de RAM das redes — MEDIDOS, não modelados.

Complementa o `spec03_custo_comparado.py`, que reporta o custo MODELADO (T2 por
ciclos, perfil de operações). Aqui é o que o processo de verdade paga: o relógio
de parede da decisão e o pico de memória residente (RSS) que carregar e executar
cada rede custa.

A figura de memória é UMA: o **pico de RSS** do processo que carrega o modelo e
decide (`VmHWM` em `/proc/self/status` — a marca d'água alta do kernel). Não é a
contagem de bytes do artefato nem o "persistente" da SPEC-03: é o que o
dispositivo precisa ter de RAM para a rede rodar. O pico é lido em três pontos:

    base        → o interpretador, antes de qualquer import
    arcabouço   → depois dos imports (NumPy; ou NumPy + PyTorch), antes de peso
    total       → ao fim do laço de inferência, com o modelo carregado e decidindo

O acréscimo do modelo é `total − arcabouço`: o quanto carregar a rede (com os
transientes do load) e executá-la eleva o pico acima do que o arcabouço já pagou.

DUAS DECISÕES DE PROTOCOLO, e o porquê delas:

1. **Um processo por modelo.** Medir os três intercalados no mesmo laço distorce
   duas vezes: quem roda logo depois do forward do torch paga os caches que as
   matrizes 256×256 dele despejam (~2× no seguinte — medido), e o pico de RSS é
   monotônico dentro de um processo — depois de importar o torch, nenhum modelo
   medido em seguida consegue mostrar um pico próprio. Um drone carrega UM
   modelo; o processo isolado é o cenário real.

2. **A mesma observação para todos, calculada FORA dos filhos.** Construir o
   ambiente levanta o simulador inteiro (~15 MB), e ele entraria no pico do
   filho; o pai constrói o ambiente uma vez, grava a primeira observação do
   `reset` (semente fixa) e cada filho só a lê. Observação real, não `randn`: o
   custo da WNN depende de quais endereços são tocados, e ruído gaussiano
   tocaria endereços vazios, medindo o caso mais barato.

Uso:
    uv run --group baseline python scripts/spec03_latencia_memoria.py
    (--na/--ns escolhem a célula; sem MADDPG publicado a célula roda só as WNN)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BLOCOS, REPS_BLOCO = 8, 1000
SEED_OBS = 12345


def _status_kb(campo: str) -> int:
    with open("/proc/self/status") as fh:
        return next(int(l.split()[1]) for l in fh if l.startswith(campo))


def _pico_kb() -> int:
    """Pico de RSS do processo até agora (`VmHWM`) — nunca desce."""
    return _status_kb("VmHWM")


def _rss_kb(aparar: bool = False) -> int:
    if aparar:
        # O load da WNN constrói a rede de DICIONÁRIOS e a converte em LUT plana;
        # os dicts morrem, mas o glibc retém as arenas e o RSS não desce. O trim
        # devolve as páginas, e a leitura passa a ser o que FICA residente. O
        # pico (`VmHWM`) não é afetado pelo trim — é justamente por isso que ele
        # é a figura: o transiente do load conta.
        import ctypes
        import gc
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:  # pragma: no cover — libc não-glibc
            pass
    return _status_kb("VmRSS")


def _observacao(cfg):
    """Semente fixa, primeira observação do reset. Roda no PAI, uma vez."""
    import numpy as np
    from uavwnn.refenv import make_env
    random.seed(SEED_OBS); np.random.seed(SEED_OBS)
    env = make_env(cfg); obs, _ = env.reset()
    x = obs[next(iter(obs))]
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass
    return np.asarray(x)


def _cronometrar(f):
    """Mediana do bloco MAIS LIMPO, p99 global.

    Um bloco único de 8 mil repetições herda o que estiver pulsando na máquina
    naquele meio minuto (medido: a mesma rede deu 17,0 µs e 34,7 µs em rodadas
    consecutivas com 4 treinos ao fundo). Vários blocos curtos espalham a
    exposição, e a MENOR mediana de bloco é o estimador do custo sem contenção —
    a contenção só soma, nunca subtrai. O p99 fica global de propósito: cauda é
    exatamente onde a contenção pertence.
    """
    import numpy as np
    for _ in range(500):
        f()
    tudo = []
    medianas = []
    for _ in range(BLOCOS):
        t = np.empty(REPS_BLOCO)
        for i in range(REPS_BLOCO):
            t0 = time.perf_counter_ns(); f(); t[i] = time.perf_counter_ns() - t0
        us = t / 1000.0
        medianas.append(float(np.median(us)))
        tudo.append(us)
    return min(medianas), float(np.percentile(np.concatenate(tudo), 99))


def _memoria(pico_base: int, pico_arcabouco: int, rss_carga: int,
             pico_carga: int, pico_total: int) -> dict:
    return {"pico_base_kb": pico_base,
            "pico_arcabouco_kb": pico_arcabouco,
            "rss_apos_carga_kb": rss_carga,
            "pico_apos_carga_kb": pico_carga,
            "pico_total_kb": pico_total,
            "pico_modelo_kb": pico_total - pico_arcabouco}


# --------------------------- filhos (um por modelo) ---------------------------

def _filho_wnn(na: int, ns: int, t_res: int, obs: Path) -> dict:
    pico0 = _pico_kb()
    import numpy as np
    sys.path.insert(0, str(RAIZ / "scripts"))
    from spec06_campanha_wnn import montar                # importa uavwnn junto
    from uavwnn.refenv import LAMENZA_003
    pico_imports = _pico_kb()

    ckpt = RAIZ / "models" / "deploy" / f"T{t_res}" / f"a{na}s{ns}"
    cfg = LAMENZA_003(num_drones=na, num_sensors=ns)
    pol, ap = montar("ntupla_dissertacao", cfg, ckpt, plana=True)
    rss_carga = _rss_kb(aparar=True)
    pico_carga = _pico_kb()
    x = np.load(obs)

    med, p99 = _cronometrar(lambda: pol(x))
    pico_fim = _pico_kb()
    return {"modelo": f"wnn-T{t_res}", "mediana_us": med, "p99_us": p99,
            "params": int(ap.ator.n_entradas), "arcabouco": "NumPy",
            **_memoria(pico0, pico_imports, rss_carga, pico_carga, pico_fim)}


def _filho_maddpg(na: int, ns: int, obs: Path) -> dict:
    pico0 = _pico_kb()
    import numpy as np
    import torch
    sys.path.insert(0, str(RAIZ / "scripts"))
    from spec02_baseline_maddpg import checkpoints
    from spec02_maddpg_rollout import carregar_ator
    from uavwnn.refenv import LAMENZA_003
    pico_imports = _pico_kb()

    disp = {(a, s): p for a, s, p in checkpoints()}
    if (na, ns) not in disp:
        return {"modelo": "maddpg", "ausente": True}
    cfg = LAMENZA_003(num_drones=na, num_sensors=ns)
    pol = carregar_ator(disp[(na, ns)], cfg.observation_size)
    rss_carga = _rss_kb(aparar=True)
    pico_carga = _pico_kb()
    x = np.load(obs)

    params = sum(v.numel() for v in pol.rede.state_dict().values())
    with torch.no_grad():
        med, p99 = _cronometrar(lambda: pol(x))
    pico_fim = _pico_kb()
    return {"modelo": "maddpg", "mediana_us": med, "p99_us": p99,
            "params": int(params), "arcabouco": "PyTorch",
            **_memoria(pico0, pico_imports, rss_carga, pico_carga, pico_fim)}


# ------------------------------- orquestrador --------------------------------

def _rodar_filho(alvo: str, na: int, ns: int, obs: Path) -> dict:
    r = subprocess.run(
        [sys.executable, __file__, "--filho", alvo, "--na", str(na),
         "--ns", str(ns), "--obs", str(obs)],
        capture_output=True, text=True, cwd=RAIZ)
    if r.returncode != 0:
        raise RuntimeError(f"filho {alvo} falhou:\n{r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--na", type=int, default=2)
    ap.add_argument("--ns", type=int, default=24)
    ap.add_argument("--filho", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--obs", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--so-figuras", action="store_true",
                    help="não mede: regenera relatório e PNGs do JSON já gravado")
    ap.add_argument("--repeticoes", type=int, default=1,
                    help="N > 1: roda cada filho N vezes (processo novo a cada vez) e "
                         "grava média/desvio por modelo em spec03-pico-ram-repN-*.json/.md; "
                         "o artefato canônico (N=1) não é tocado")
    a = ap.parse_args()

    saida = RAIZ / "reports" / f"spec03-latencia-memoria-a{a.na}s{a.ns}.json"
    if a.so_figuras and a.repeticoes > 1:
        base = RAIZ / "reports" / f"spec03-pico-ram-rep{a.repeticoes}-a{a.na}s{a.ns}"
        d = json.loads(Path(f"{base}.json").read_text())
        for f in _artefatos_rep(d["amostras"], d["agregado"], a.na, a.ns, a.repeticoes):
            print(f"-> {f.relative_to(RAIZ)}")
        return 0
    if a.so_figuras:
        linhas = json.loads(saida.read_text())
        for f in [_markdown(linhas, a.na, a.ns)] + _graficos(linhas, a.na, a.ns):
            print(f"-> {f.relative_to(RAIZ)}")
        return 0

    if a.filho:                                           # modo filho: só JSON
        if a.filho == "maddpg":
            print(json.dumps(_filho_maddpg(a.na, a.ns, a.obs)))
        else:
            print(json.dumps(_filho_wnn(a.na, a.ns, int(a.filho[1:]), a.obs)))
        return 0

    # A observação é calculada AQUI, uma vez: o simulador fica fora do pico dos
    # filhos, e os três leem exatamente o mesmo vetor.
    import numpy as np
    from uavwnn.refenv import LAMENZA_003
    cfg = LAMENZA_003(num_drones=a.na, num_sensors=a.ns)
    with tempfile.TemporaryDirectory(prefix="spec03-obs-") as tmp:
        obs = Path(tmp) / f"obs-a{a.na}s{a.ns}.npy"
        np.save(obs, _observacao(cfg))
        if a.repeticoes > 1:
            return _repetir(a.na, a.ns, obs, a.repeticoes)
        linhas = []
        for alvo in ("T16", "T32", "maddpg"):
            try:
                r = _rodar_filho(alvo, a.na, a.ns, obs)
            except RuntimeError as e:
                print(f"  {alvo}: {e}", file=sys.stderr)
                continue
            if not r.get("ausente"):
                linhas.append(r)
            else:
                print(f"  maddpg: sem checkpoint para Na={a.na} Ns={a.ns} "
                      "(Lamenza publica Na∈{2,4,8})")

    print(f"\n=== célula {a.na}d/{a.ns}s · mesma observação · "
          f"um processo por modelo · {BLOCOS}×{REPS_BLOCO} repetições ===\n")
    print(f"  {'modelo':<9} {'mediana':>9} {'p99':>9} {'params':>9}"
          f"{'pico total':>12}{'arcabouço':>11}{'modelo':>9}{'pós-carga':>11}")
    for r in linhas:
        print(f"  {r['modelo']:<9} {r['mediana_us']:>7.1f}µs {r['p99_us']:>7.1f}µs"
              f" {r['params']:>9,}{r['pico_total_kb']/1024:>9,.1f} MB"
              f"{r['pico_arcabouco_kb']/1024:>8,.1f} MB"
              f"{r['pico_modelo_kb']/1024:>6,.1f} MB"
              f"{r['rss_apos_carga_kb']/1024:>8,.1f} MB")
    print("""
  pico total = VmHWM ao fim do laço de inferência — o que o processo que carrega
               o modelo e decide precisou de RAM, no máximo
  arcabouço  = VmHWM depois dos imports, antes de qualquer peso (torch ≈ 0,5 GB)
  modelo     = pico total − arcabouço: quanto carregar e executar a rede eleva o
               pico acima do que o arcabouço já pagou
  pós-carga  = RSS estável com o modelo carregado (após malloc_trim)""")

    saida.write_text(json.dumps(linhas, indent=2))
    print(f"\n-> {saida.relative_to(RAIZ)}")
    for f in [_markdown(linhas, a.na, a.ns)] + _graficos(linhas, a.na, a.ns):
        print(f"-> {f.relative_to(RAIZ)}")
    return 0


CAMPOS_REP = (("pico_total_kb", "pico total", 1024, "MB"),
              ("pico_arcabouco_kb", "arcabouço", 1024, "MB"),
              ("pico_modelo_kb", "acréscimo do modelo", 1024, "MB"),
              ("rss_apos_carga_kb", "RSS pós-carga", 1024, "MB"),
              ("mediana_us", "latência mediana", 1, "µs"),
              ("p99_us", "latência p99", 1, "µs"))


def _repetir(na: int, ns: int, obs: Path, n: int) -> int:
    """N medições independentes por modelo — um processo NOVO a cada uma.

    Repetir dentro do mesmo processo não mede nada: o `VmHWM` é monotônico, e a
    segunda leitura seria igual à primeira. Cada repetição paga o import inteiro
    (torch ≈ 3 s) — é o preço de ter N amostras do pico de verdade. A agregação é
    média e desvio-padrão amostral (ddof=1), com mínimo e máximo para a cauda.
    """
    import numpy as np
    amostras: dict[str, list[dict]] = {}
    for i in range(n):
        for alvo in ("T16", "T32", "maddpg"):
            try:
                r = _rodar_filho(alvo, na, ns, obs)
            except RuntimeError as e:
                print(f"  rep {i + 1} {alvo}: {e}", file=sys.stderr)
                continue
            if r.get("ausente"):
                continue
            amostras.setdefault(r["modelo"], []).append(r)
        print(f"  rep {i + 1}/{n} concluída", file=sys.stderr, flush=True)

    agg = {}
    for modelo, rs in amostras.items():
        agg[modelo] = {"n": len(rs), "arcabouco": rs[0]["arcabouco"]}
        for campo, *_ in CAMPOS_REP:
            v = np.array([r[campo] for r in rs], dtype=float)
            agg[modelo][campo] = {"media": float(v.mean()),
                                  "desvio": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                  "min": float(v.min()), "max": float(v.max())}

    print(f"\n=== célula {na}d/{ns}s · {n} medições por modelo · um processo novo "
          f"por medição · média ± desvio (min–max) ===\n")
    for campo, rot, div, un in CAMPOS_REP:
        print(f"  {rot}:")
        for modelo, g in agg.items():
            e = g[campo]
            print(f"    {modelo:<9} {e['media']/div:>9,.2f} ± {e['desvio']/div:>6,.2f} {un}"
                  f"   ({e['min']/div:,.2f}–{e['max']/div:,.2f})   n={g['n']}")

    base = RAIZ / "reports" / f"spec03-pico-ram-rep{n}-a{na}s{ns}"
    Path(f"{base}.json").write_text(json.dumps(
        {"celula": f"a{na}s{ns}", "repeticoes": n, "agregado": agg,
         "amostras": amostras}, indent=2))
    print(f"\n-> {Path(str(base) + '.json').relative_to(RAIZ)}")
    for f in _artefatos_rep(amostras, agg, na, ns, n):
        print(f"-> {f.relative_to(RAIZ)}")
    return 0


def _artefatos_rep(amostras: dict, agg: dict, na: int, ns: int, n: int) -> list[Path]:
    """Relatório e box plot das N medições — escritos do JSON, nada transcrito."""
    import datetime
    base = RAIZ / "reports" / f"spec03-pico-ram-rep{n}-a{na}s{ns}"

    def linha(cels):
        return "| " + " | ".join(str(c) for c in cels) + " |"
    L = [f"# Pico de RAM e latência — {na}d/{ns}s, {n} medições por modelo", "",
         f"Gerado por `scripts/spec03_latencia_memoria.py --repeticoes {n}` em "
         f"{datetime.date.today().isoformat()}. Cada medição é um **processo novo** "
         f"(o `VmHWM` é monotônico; repetir no mesmo processo não mede nada), com a "
         f"mesma observação pré-calculada no pai e o protocolo de latência da SPEC-03 "
         f"(menor mediana de {BLOCOS}×{REPS_BLOCO}). Média ± desvio-padrão amostral; "
         f"entre parênteses, mínimo–máximo.", ""]
    for campo, rot, div, un in CAMPOS_REP:
        L += [f"## {rot} ({un})", "",
              linha(["modelo", "média ± desvio", "min–max", "n"]), linha(["---"] * 4)]
        for modelo, g in agg.items():
            e = g[campo]
            L.append(linha([f"{modelo} ({g['arcabouco']})",
                            f"**{e['media']/div:,.2f} ± {e['desvio']/div:,.2f}**",
                            f"{e['min']/div:,.2f}–{e['max']/div:,.2f}", g["n"]]))
        L.append("")
    L += [f"![box plot das {n} medições]({base.name}.png)", ""]
    md = Path(f"{base}.md"); md.write_text("\n".join(L))
    saidas = [md]
    try:
        saidas.append(_boxplot_rep(amostras, na, ns, n, Path(f"{base}.png")))
    except ImportError:
        print("  [boxplot] matplotlib ausente — pulei. Rode: uv sync --group dev")
    return saidas


def _boxplot_rep(amostras: dict, na: int, ns: int, n: int, saida: Path) -> Path:
    """Box plot por modelo das N medições: pico total, acréscimo do modelo e
    latência mediana. Caixa = quartis, bigodes 1,5×IQR, pontos = TODAS as
    amostras (jitter), para que a dispersão real e os outliers fiquem visíveis.
    O pico total vai em log — o PyTorch põe o MADDPG uma ordem de grandeza acima."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordem = [m for m in ("wnn-T16", "wnn-T32", "maddpg") if m in amostras]
    cores = [CORES.get(m, "#888") for m in ordem]
    paineis = (("pico_total_kb", 1024, "peak RSS of the process (MB, log)", True),
               ("pico_modelo_kb", 1024, "model increment to the peak (MB)", False),
               ("mediana_us", 1, "latency per decision, median (µs)", False))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, (campo, div, rot, log) in zip(axes, paineis):
        dados = [np.array([r[campo] for r in amostras[m]], float) / div for m in ordem]
        bp = ax.boxplot(dados, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "#222", "lw": 1.4},
                        whiskerprops={"color": "#555"}, capprops={"color": "#555"})
        for caixa, c in zip(bp["boxes"], cores):
            caixa.set(facecolor=c, alpha=0.35, edgecolor=c)
        for i, (d, c) in enumerate(zip(dados, cores), start=1):
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(d)), d, s=9, color=c,
                       alpha=0.55, zorder=3, linewidths=0)
            ax.annotate(f"{d.mean():,.1f} ± {d.std(ddof=1):,.1f}",
                        (i, d.max()), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 4), textcoords="offset points")
        ax.set_xticks(range(1, len(ordem) + 1), ordem)
        ax.set_ylabel(rot)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(min(d.min() for d in dados) * 0.6, max(d.max() for d in dados) * 2.5)
        else:
            ax.margins(y=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{n} measurements per model, one fresh process each — "
                 f"{na} drone{'s' if na > 1 else ''} · {ns} sensors\n"
                 "box = quartiles, whiskers 1.5×IQR, dots = all samples; "
                 "label = mean ± sd", fontsize=10)
    fig.tight_layout()
    fig.savefig(saida, dpi=160); plt.close(fig)
    return saida


def _markdown(linhas: list[dict], na: int, ns: int) -> Path:
    """O relatório completo, escrito PELA MEDIÇÃO — nada transcrito à mão.

    Números copiados de uma execução para um texto foi como esta sessão publicou
    uma latência de laço intercalado: o erro mora na transcrição. Aqui a mesma
    execução que mediu escreve a tabela.
    """
    import datetime
    import platform

    def linha(cels):
        return "| " + " | ".join(str(c) for c in cels) + " |"

    def mb(kb):
        return f"{kb / 1024:,.1f} MB"

    L = [f"# Latência e pico de RAM — {na}d/{ns}s, medidos",
         "",
         f"Gerado por `scripts/spec03_latencia_memoria.py` em "
         f"{datetime.date.today().isoformat()} · {platform.machine()} · "
         f"mesma observação para os três · um processo por modelo · "
         f"menor mediana de {BLOCOS}×{REPS_BLOCO}.",
         "",
         "## Tempo de inferência (decisão completa: obs → ação)",
         "",
         linha(["modelo", "mediana", "p99"]),
         linha(["---"] * 3)]
    for r in linhas:
        L.append(linha([r["modelo"], f"{r['mediana_us']:.1f} µs",
                        f"{r['p99_us']:.1f} µs"]))
    base = next((r for r in linhas if r["modelo"] == "maddpg"), None)
    if base:
        for r in linhas:
            if r["modelo"] != "maddpg":
                L.append(f"\n- {r['modelo']} é "
                         f"**{base['mediana_us'] / r['mediana_us']:.1f}×** "
                         f"mais rápida que o MADDPG")

    L += ["", "## Pico de RAM do processo (VmHWM)",
          "",
          linha(["modelo", "arcabouço", "pico total", "acréscimo do modelo",
                 "RSS pós-carga"]),
          linha(["---"] * 5)]
    for r in linhas:
        L.append(linha([f"{r['modelo']} ({r['arcabouco']})",
                        mb(r["pico_arcabouco_kb"]), f"**{mb(r['pico_total_kb'])}**",
                        mb(r["pico_modelo_kb"]), mb(r["rss_apos_carga_kb"])]))
    L += ["",
          "- **pico total**: a marca d'água alta de RSS do processo ao fim do laço "
          "de inferência — o que o dispositivo precisa ter de RAM para carregar o "
          "modelo e decidir. É a figura de memória do projeto;",
          "- **arcabouço**: o pico depois dos imports e antes de qualquer peso — "
          "NumPy na WNN, NumPy + PyTorch no MADDPG. É custo real de quem implanta "
          "com aquele arcabouço, mas é biblioteca, não rede;",
          "- **acréscimo do modelo**: `pico total − arcabouço` — quanto carregar a "
          "rede (com os transientes do load: na WNN, a rede de dicionários que "
          "vira LUT plana) e executá-la eleva o pico;",
          "- **RSS pós-carga**: o residente estável com o modelo carregado, após "
          "`malloc_trim` — o que FICA, não o pico;",
          "- o simulador fica fora: a observação é calculada no processo pai e lida "
          "pelos filhos de um `.npy`.",
          "",
          f"![tempos](spec03-latencia-a{na}s{ns}.png)",
          f"![pico de RAM](spec03-pico-ram-a{na}s{ns}.png)", ""]

    f = RAIZ / "reports" / f"spec03-latencia-memoria-a{na}s{ns}.md"
    f.write_text("\n".join(L))
    return f


CORES = {"wnn-T16": "#2a9d8f", "wnn-T32": "#457b9d", "maddpg": "#e76f51"}


def _graficos(linhas: list[dict], na: int, ns: int) -> list[Path]:
    """Dois PNGs: os tempos medidos e o pico de RAM por processo.

    O pico é uma barra por modelo, em escala log — o arcabouço do MADDPG
    (PyTorch) fica uma ordem de grandeza acima de tudo o mais, e o log é o que
    deixa as duas WNN legíveis ao lado dele. A parte do arcabouço fica
    hachurada, para que se veja o quanto do pico é biblioteca e o quanto é rede.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nomes = [r["modelo"] for r in linhas]
    cores = [CORES.get(n, "#888") for n in nomes]
    rot = f"{na} drone{'s' if na > 1 else ''} · {ns} sensors"
    saidas = []

    # -- tempos: mediana como barra, p99 como haste --
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    med = [r["mediana_us"] for r in linhas]
    p99 = [r["p99_us"] for r in linhas]
    barras = ax.bar(nomes, med, color=cores, width=0.55)
    ax.vlines(range(len(nomes)), med, p99, color="#333", lw=1.5)
    ax.scatter(range(len(nomes)), p99, color="#333", s=18, zorder=3,
               label="p99")
    for b, m in zip(barras, med):
        ax.annotate(f"{m:.1f} µs", (b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylabel("time per decision (µs)")
    ax.set_title(f"Inference time — {rot}\n"
                 "same observation · one process per model · lowest median "
                 "of 8×1000", fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    f = RAIZ / "reports" / f"spec03-latencia-a{na}s{ns}.png"
    fig.savefig(f, dpi=160); plt.close(fig); saidas.append(f)

    # -- pico de RAM: arcabouço (hachurado) ao lado do pico total, em log --
    #    Empilhar não funciona em log: a fatia do arcabouço vira uma linha no pé
    #    da barra. Lado a lado, as duas leituras ficam comparáveis entre modelos.
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    arc = [r["pico_arcabouco_kb"] / 1024 for r in linhas]
    tot = [r["pico_total_kb"] / 1024 for r in linhas]
    x = np.arange(len(nomes)); lg = 0.36
    ax.bar(x - lg / 2, arc, lg, color=cores, alpha=0.35, hatch="//",
           edgecolor=cores, label="framework peak (imports, no weights)")
    ax.bar(x + lg / 2, tot, lg, color=cores,
           label="total peak (model loaded + inference)")
    for i, (a_, t, r) in enumerate(zip(arc, tot, linhas)):
        ax.annotate(f"{a_:,.0f}", (i - lg / 2, a_), ha="center", va="bottom",
                    fontsize=8, xytext=(0, 2), textcoords="offset points")
        ax.annotate(f"{t:,.0f} MB\n+{r['pico_modelo_kb']/1024:,.1f} MB model",
                    (i + lg / 2, t), ha="center", va="bottom", fontsize=8,
                    weight="bold", xytext=(0, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(10, max(tot) * 4)
    ax.set_xticks(x, nomes)
    ax.set_ylabel("peak RSS of the process (MB, log)")
    ax.set_title(f"Peak RAM to load and run the actor — {rot}\n"
                 "VmHWM · one process per model · observation precomputed "
                 "outside the child", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    f = RAIZ / "reports" / f"spec03-pico-ram-a{na}s{ns}.png"
    fig.savefig(f, dpi=160); plt.close(fig); saidas.append(f)
    return saidas


if __name__ == "__main__":
    raise SystemExit(main())
