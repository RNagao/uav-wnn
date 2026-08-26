#!/usr/bin/env python3
"""Latência de inferência e gasto de memória das três redes — MEDIDOS, não modelados.

Complementa o `spec06_custo_comparado.py`, que reporta o custo MODELADO (T1 pela
contabilidade de bytes, T2 por ciclos). Aqui é o que o processo de verdade paga:
o relógio de parede da decisão e o RSS que carregar cada rede custa.

DUAS DECISÕES DE PROTOCOLO, e o porquê delas:

1. **Um processo por modelo.** Medir os três intercalados no mesmo laço distorce:
   quem roda logo depois do forward do torch paga os caches que as matrizes
   256×256 dele despejam (~2× no seguinte — medido, e a penalidade se move junto
   quando se troca a ordem do laço). Um drone carrega UM modelo; o processo
   isolado é o cenário real. O isolamento também torna o ΔRSS atribuível: o RSS
   é lido antes dos imports, depois deles (o arcabouço) e depois do load (a rede).

2. **A mesma observação para todos.** Cada filho reconstrói o ambiente com a
   mesma semente e usa a primeira observação do reset — real, não `randn`: o
   custo da WNN depende de quais endereços são tocados, e ruído gaussiano
   tocaria endereços vazios, medindo o caso mais barato.

Uso:
    uv run --group baseline python scripts/spec06_latencia_memoria.py
    (--na/--ns escolhem a célula; sem MADDPG publicado a célula roda só as WNN)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BLOCOS, REPS_BLOCO = 8, 1000
SEED_OBS = 12345


def _rss_kb(aparar: bool = False) -> int:
    if aparar:
        # O load da WNN constrói a rede de DICIONÁRIOS e a converte em LUT plana;
        # os dicts morrem, mas o glibc retém as arenas e o RSS não desce. O trim
        # devolve as páginas, e o ΔRSS passa a medir o que FICA residente.
        import ctypes
        import gc
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:  # pragma: no cover — libc não-glibc
            pass
    with open("/proc/self/status") as fh:
        return next(int(l.split()[1]) for l in fh if l.startswith("VmRSS"))


def _observacao(cfg):
    """A mesma nos três filhos: semente fixa, primeira observação do reset.

    Chamada DEPOIS da leitura de RSS pós-carga: construir o ambiente levanta o
    simulador inteiro (~15 MB), e medi-lo dentro do ΔRSS do modelo foi o erro da
    primeira versão deste script — a WNN aparecia com ΔRSS 18 MB para 346 KB de
    arrays.
    """
    import numpy as np
    from uavwnn.refenv import make_env
    random.seed(SEED_OBS); np.random.seed(SEED_OBS)
    env = make_env(cfg); obs, _ = env.reset()
    x = obs[next(iter(obs))]
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass
    return x


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


# --------------------------- filhos (um por modelo) ---------------------------

def _filho_wnn(na: int, ns: int, t_res: int) -> dict:
    rss0 = _rss_kb()
    import numpy as np                                    # noqa: F401
    sys.path.insert(0, str(RAIZ / "scripts"))
    from spec06_campanha_wnn import montar                # importa uavwnn junto
    rss_imports = _rss_kb()

    from uavwnn.refenv import LAMENZA_003
    ckpt = RAIZ / "models" / "deploy" / f"T{t_res}" / f"a{na}s{ns}"
    cfg = LAMENZA_003(num_drones=na, num_sensors=ns)
    pol, ap = montar("ntupla_dissertacao", cfg, ckpt, plana=True)
    rss_carregado = _rss_kb(aparar=True)
    x = _observacao(cfg)

    # bytes dos arrays vivos: ator (LUT plana) + termômetro — o artefato inteiro
    from uavwnn.encoding import TermometroQuantil
    termo = TermometroQuantil.carregar(ckpt / "termometro.npz")
    lut = ap.ator
    arrays = (lut.chaves, lut.valores, lut._desloc, lut._tuplas, lut._pot_mat,
              termo.limiares)
    vivos_kb = sum(a.nbytes for a in arrays) / 1024

    med, p99 = _cronometrar(lambda: pol(x))
    return {"modelo": f"wnn-T{t_res}", "mediana_us": med, "p99_us": p99,
            "rss_base_kb": rss0, "rss_arcabouco_kb": rss_imports - rss0,
            "rss_modelo_kb": rss_carregado - rss_imports,
            "arrays_vivos_kb": vivos_kb,
            "disco_kb": sum(f.stat().st_size for f in ckpt.iterdir()) / 1024,
            "params": int(lut.chaves.size)}


def _filho_maddpg(na: int, ns: int) -> dict:
    rss0 = _rss_kb()
    import numpy as np                                    # noqa: F401
    import torch
    sys.path.insert(0, str(RAIZ / "scripts"))
    from spec02_baseline_maddpg import checkpoints
    from spec02_maddpg_rollout import carregar_ator
    rss_imports = _rss_kb()

    disp = {(a, s): p for a, s, p in checkpoints()}
    if (na, ns) not in disp:
        return {"modelo": "maddpg", "ausente": True}
    from uavwnn.refenv import LAMENZA_003
    cfg = LAMENZA_003(num_drones=na, num_sensors=ns)
    pol = carregar_ator(disp[(na, ns)], cfg.observation_size)
    rss_carregado = _rss_kb(aparar=True)
    x = _observacao(cfg)

    sd = pol.rede.state_dict()
    vivos_kb = sum(v.numel() * v.element_size() for v in sd.values()) / 1024
    params = sum(v.numel() for v in sd.values())

    with torch.no_grad():
        med, p99 = _cronometrar(lambda: pol(x))
    return {"modelo": "maddpg", "mediana_us": med, "p99_us": p99,
            "rss_base_kb": rss0, "rss_arcabouco_kb": rss_imports - rss0,
            "rss_modelo_kb": rss_carregado - rss_imports,
            "arrays_vivos_kb": vivos_kb,
            # o .cleanrl_model traz ator E crítico — anotado na tabela
            "disco_kb": disp[(na, ns)].stat().st_size / 1024,
            "params": int(params)}


# ------------------------------- orquestrador --------------------------------

def _rodar_filho(alvo: str, na: int, ns: int) -> dict:
    r = subprocess.run(
        [sys.executable, __file__, "--filho", alvo, "--na", str(na),
         "--ns", str(ns)],
        capture_output=True, text=True, cwd=RAIZ)
    if r.returncode != 0:
        raise RuntimeError(f"filho {alvo} falhou:\n{r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--na", type=int, default=2)
    ap.add_argument("--ns", type=int, default=24)
    ap.add_argument("--filho", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.filho:                                           # modo filho: só JSON
        if a.filho == "maddpg":
            print(json.dumps(_filho_maddpg(a.na, a.ns)))
        else:
            print(json.dumps(_filho_wnn(a.na, a.ns, int(a.filho[1:]))))
        return 0

    linhas = []
    for alvo in ("T16", "T32", "maddpg"):
        try:
            r = _rodar_filho(alvo, a.na, a.ns)
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
          f"{'rede (KB)':>11} {'ΔRSS (KB)':>11} {'arcabouço':>12} {'disco':>9}")
    for r in linhas:
        print(f"  {r['modelo']:<9} {r['mediana_us']:>7.1f}µs {r['p99_us']:>7.1f}µs"
              f" {r['params']:>9,}{r['arrays_vivos_kb']:>11,.0f}"
              f"{r['rss_modelo_kb']:>11,.0f}{r['rss_arcabouco_kb']/1024:>9,.0f} MB"
              f"{r['disco_kb']:>8,.0f}K")
    print("""
  rede (KB)  = arrays vivos do ator + termômetro (MADDPG: tensores do ator)
  ΔRSS       = o que FICA residente após o load (com malloc_trim) — modelo mais
               os módulos que o caminho de load puxa; por isso é > rede (KB)
  arcabouço  = o que os imports custam antes de qualquer peso (torch ≈ 0,5 GB)
  disco      = o artefato em disco; o .cleanrl_model traz ator E CRÍTICO juntos""")

    saida = RAIZ / "reports" / f"spec06-latencia-memoria-a{a.na}s{a.ns}.json"
    saida.write_text(json.dumps(linhas, indent=2))
    print(f"\n-> {saida.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
