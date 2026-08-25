"""Exporta o artefato de IMPLANTAÇÃO de um modelo WNN.

O que está em `models/` é o checkpoint de TREINO: ator + crítico, endereços em
`uint64` e valores em `float64`. Para voar, nada disso é necessário:

  * o **crítico** só existe para treinar (ele estima "quão boa é esta situação"
    e nunca escolhe ação) — 32 % do pacote, inútil no deploy;
  * o **endereço** tem `tamanho_tupla` bits (12 → 4096 valores), então `uint16`
    basta e o `uint64` desperdiça 4×;
  * os **valores** cabem em `float32`, que é a precisão que o modelo de custo da
    SPEC-03 já assume ao reportar o T1.

Medido no ator de 2 drones: 4722 KB → 1773 KB, 2,7× menor.

O `float32` MUDA as predições nos últimos bits (erro relativo máximo 5,9e-08,
mediano 2,1e-08 — dentro do eps do `float32`, 1,2e-07). Por isso este script
**verifica** antes de declarar sucesso: compara as ações do ator nos dois
formatos sobre observações reais e recusa exportar se a diferença passar do
tolerado.

    uv run python scripts/spec06_exportar_deploy.py models/T32/a2s24 --na 2 --ns 24
    uv run python scripts/spec06_exportar_deploy.py --todos
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

RAIZ = Path(__file__).resolve().parents[1]
TOL_REL = 1e-6
"""Tolerância na AÇÃO, não no peso. `float32` erra ~6e-08 por valor; a ação passa
por uma soma de `N` termos e uma divisão, então o erro se acumula um pouco. 1e-6
dá três ordens de folga sobre o erro por valor e ainda é 10⁴ vezes menor que a
menor diferença de ação que o ambiente distingue."""


def exportar(origem: Path, destino: Path) -> dict:
    """Grava só o ator, com `uint16`/`float32`. Devolve o resumo de tamanhos."""
    fonte = next(origem.glob("*ator.npz"))
    z = np.load(fonte, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    n = int(cfg["tamanho_tupla"])
    if n > 16:
        raise SystemExit(f"tamanho_tupla={n} não cabe em uint16 — ajuste o dtype")

    end = z["enderecos"]
    if end.size and int(end.max()) >= (1 << 16):
        raise SystemExit(f"endereço máximo {int(end.max())} não cabe em uint16")

    destino.mkdir(parents=True, exist_ok=True)
    saida = {k: z[k] for k in z.files}
    saida["enderecos"] = end.astype(np.uint16)
    saida["valores"] = z["valores"].astype(np.float32)
    # `mapeamento` indexa bits de entrada (n_bits ≤ 1248 aqui) — uint16 basta.
    saida["mapeamento"] = z["mapeamento"].astype(np.uint16)
    alvo = destino / fonte.name
    np.savez_compressed(alvo, **saida)
    for extra in ("termometro.npz", "familia.txt", "procedencia.json", "config.json"):
        f = origem / extra
        if f.exists():
            (destino / extra).write_bytes(f.read_bytes())

    antes = sum(f.stat().st_size for f in origem.iterdir() if f.is_file())
    depois = sum(f.stat().st_size for f in destino.iterdir() if f.is_file())
    return {"origem_kb": antes / 1024, "deploy_kb": depois / 1024,
            "reducao": antes / max(depois, 1)}


def verificar(origem: Path, destino: Path, na: int, ns: int, n_obs: int = 400) -> dict:
    """Compara as AÇÕES dos dois formatos sobre observações reais do ambiente.

    Observações reais, não vetores aleatórios: o custo e o resultado do WNN
    dependem de QUAIS endereços são tocados, e ruído gaussiano tocaria endereços
    vazios — mediria o caso trivial em que os dois formatos concordam por serem
    ambos zero.
    """
    import logging
    from spec06_campanha_wnn import montar
    from uavwnn.refenv import LAMENZA_003, make_env, prune_logging_handlers

    cfg = LAMENZA_003(na, ns, phase="eval")
    H = list(logging.getLogger().handlers)
    p64, _ = montar("ntupla_dissertacao", cfg, origem)
    p32, _ = montar("ntupla_dissertacao", cfg, destino)

    difs, vistos = [], 0
    for s in range(50):
        random.seed(s); np.random.seed(s)
        env = make_env(cfg); obs, _ = env.reset(); prune_logging_handlers(H)
        for _ in range(60):
            for a in env.agents:
                x = obs[a]
                d = np.abs(np.asarray(p64(x), float) - np.asarray(p32(x), float))
                difs.append(d.max()); vistos += 1
            if vistos >= n_obs:
                break
            obs, _, term, trunc, infos = env.step(
                {a: p64(obs[a]) for a in env.agents})
            if (infos and "avg_reward" in next(iter(infos.values()))) \
               or any(term.values()) or any(trunc.values()):
                break
        try:
            env.close()
        except Exception:                                    # pragma: no cover
            pass
        if vistos >= n_obs:
            break
    d = np.array(difs)
    return {"n_obs": int(d.size), "dif_max": float(d.max()),
            "dif_mediana": float(np.median(d)), "ok": bool(d.max() <= TOL_REL)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origem", nargs="?", type=Path)
    ap.add_argument("--na", type=int); ap.add_argument("--ns", type=int)
    ap.add_argument("--todos", action="store_true",
                    help="exporta models/T*/a*s* para models/deploy/")
    ap.add_argument("--destino", type=Path, default=None)
    a = ap.parse_args()

    if a.todos:
        alvos = []
        for d in sorted((RAIZ / "models").glob("T*/a*s*")):
            if "em-treino" in str(d) or not any(d.glob("*ator.npz")):
                continue
            m = d.name  # aNsM
            na = int(m.split("s")[0][1:]); ns = int(m.split("s")[1])
            alvos.append((d, RAIZ / "models" / "deploy" / d.parent.name / d.name, na, ns))
    else:
        if a.origem is None or a.na is None or a.ns is None:
            sys.exit("informe origem --na --ns, ou use --todos")
        dst = a.destino or (RAIZ / "models" / "deploy" / a.origem.parent.name / a.origem.name)
        alvos = [(a.origem, dst, a.na, a.ns)]

    falhou = False
    print(f"{'modelo':<22}{'treino':>10}{'deploy':>10}{'redução':>9}"
          f"{'dif. máx':>12}  veredito")
    for origem, destino, na, ns in alvos:
        t = exportar(origem, destino)
        v = verificar(origem, destino, na, ns)
        falhou |= not v["ok"]
        rot = f"{origem.parent.name}/{origem.name}"
        print(f"{rot:<22}{t['origem_kb']:>8.0f}K{t['deploy_kb']:>9.0f}K"
              f"{t['reducao']:>8.1f}×{v['dif_max']:>12.2e}  "
              f"{'OK' if v['ok'] else 'FALHOU (> ' + f'{TOL_REL:.0e}' + ')'}")
    if falhou:
        print("\nalgum export mudou a ação além do tolerado — NÃO publique estes.")
        return 1
    print(f"\n-> models/deploy/  ({len(alvos)} modelos, ação idêntica até {TOL_REL:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
