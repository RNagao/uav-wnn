"""Gera as imagens de path da WNN para TODOS os cenários, de uma vez.

É o laço em volta do `spec06_path.py` — reaproveita o mesmo renderizador (roda a
política num episódio, desenha as trajetórias e os sensores) e só percorre a grade
de cenários, resolvendo o checkpoint por (variante, T, Na) e escolhendo uma semente
que complete a coleta quando existir.

GRADE. Cada Na tem UM checkpoint (a rede é compartilhada por drone e `d_obs`
depende só de Na, não de Ns), então o mesmo artefato desenha qualquer Ns. Por
padrão desenha só as células com baseline — Na=1 em Ns=12 (single-drone) e Na=2 em
Ns∈{12,24,36}, transferindo a rede treinada em Ns=24. Na=4 e Na=8 têm checkpoint e
saem com `--na 4 8`, mas ficam fora do padrão.

CHECKPOINT. `--variante deploy` (padrão) usa o artefato PODADO que a tese reporta —
`models/deploy/T{t}/…`; `--variante cheio` usa o de treino em `models/T{t}/…`. O
Ns de treino é 12 para Na=1 e 24 para os demais. Cenário sem checkpoint é pulado
com aviso (ex.: T32/a8, cujo treino ficou em 37 %).

SEMENTE. Sem `--seed`, tenta `--tentativas` sementes (0,1,2,…) e fica com a melhor:
sucesso primeiro, depois maior cobertura, depois menor completion time. Isso é para
inspeção qualitativa — o veredito quantitativo é a avaliação pareada, não a figura.
Com `--seed S`, usa só aquela, sem busca.

    # padrão: deploy, T16, todas as células, melhor de 6 sementes
    uv run python scripts/spec06_paths_todos.py

    # cheio e podado, os dois T, semente fixa
    uv run python scripts/spec06_paths_todos.py --variante cheio deploy --t 16 32 --seed 0

    # só as células single e Na=2, tentando 12 sementes
    uv run python scripts/spec06_paths_todos.py --na 1 2 --tentativas 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from spec06_path import plotar, rodar_e_gravar  # noqa: E402
from spec06_campanha_wnn import montar  # noqa: E402
from uavwnn.refenv import LAMENZA_003, make_env  # noqa: E402  (garante refenv importável)

RAIZ = Path(__file__).resolve().parents[1]
SAIDA_PADRAO = RAIZ / "reports" / "img" / "spec06-path"

# Na -> Ns a desenhar. `d_obs` depende só de Na, então o mesmo checkpoint vale para
# qualquer Ns; a grade abaixo é a da campanha comparável.
GRADE = {1: [12], 2: [12, 24, 36], 4: [12, 24, 36], 8: [12, 24, 36]}


def ns_treino(na: int) -> int:
    """O Ns em que a família de Na drones foi treinada — 12 no single, 24 nos demais."""
    return 12 if na == 1 else 24


def checkpoint(variante: str, t: int, na: int) -> Path:
    ns_tr = ns_treino(na)
    if variante == "deploy":
        return RAIZ / "models" / "deploy" / f"T{t}" / f"a{na}s{ns_tr}"
    return RAIZ / "models" / f"T{t}" / f"a{na}s{ns_tr}"


def escolher_semente(cfg, pol, tentativas: int, semente_fixa: int | None):
    """Roda 1..N sementes e devolve a melhor: (sucesso, cobertura, −CT). Fixa pula a busca."""
    sementes = [semente_fixa] if semente_fixa is not None else list(range(tentativas))
    melhor = None  # (chave, seed, traj, sensores, ct, ok)
    for s in sementes:
        traj, sensores, ct, ok = rodar_e_gravar(cfg, pol, s)
        coletados = sum(1 for _, c in sensores if c)
        chave = (int(ok), coletados, -ct)
        if melhor is None or chave > melhor[0]:
            melhor = (chave, s, traj, sensores, ct, ok)
        if ok and semente_fixa is None:
            break  # já achou uma que completa; não precisa varrer o resto
    _, s, traj, sensores, ct, ok = melhor
    return s, traj, sensores, ct, ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", nargs="+", choices=["cheio", "deploy"], default=["deploy"],
                    help="qual artefato desenhar (padrão: deploy — o podado que a tese reporta)")
    ap.add_argument("--t", nargs="+", type=int, choices=[16, 32], default=[16],
                    help="resolução(ões) do termômetro T (padrão: 16)")
    ap.add_argument("--na", nargs="+", type=int, choices=sorted(GRADE),
                    help="filtra por número de drones (padrão: todos da grade)")
    ap.add_argument("--ns", nargs="+", type=int,
                    help="filtra por número de sensores (padrão: todos da grade por Na)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixa a semente e pula a busca (padrão: busca a melhor)")
    ap.add_argument("--tentativas", type=int, default=6,
                    help="quantas sementes tentar quando não há --seed (padrão: 6)")
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO,
                    help=f"pasta de saída (padrão: {SAIDA_PADRAO})")
    a = ap.parse_args()

    nas = a.na or [1, 2]   # padrão: só single-drone e Na=2 (as células com baseline)
    a.saida.mkdir(parents=True, exist_ok=True)

    resumo, pulados = [], []
    for variante in a.variante:
        for t in a.t:
            for na in nas:
                ns_lista = [ns for ns in GRADE[na] if a.ns is None or ns in a.ns]
                if not ns_lista:
                    continue
                ckpt = checkpoint(variante, t, na)
                if not ckpt.exists() or not any(ckpt.glob("*ator.npz")):
                    pulados.append(f"a{na} · T{t} · {variante} (sem checkpoint em {ckpt})")
                    continue
                # A política independe de Ns, mas a montagem lê o config do checkpoint;
                # monta uma vez por (variante,T,Na) e reusa em todos os Ns.
                pol = None
                for ns in ns_lista:
                    cfg = LAMENZA_003(na, ns, phase="eval")
                    if pol is None:
                        pol, _ = montar("ntupla_dissertacao", cfg, ckpt)
                    s, traj, sensores, ct, ok = escolher_semente(
                        cfg, pol, a.tentativas, a.seed)
                    nome = f"wnn-a{na}s{ns}-T{t}-{variante}.png"
                    destino = a.saida / nome
                    coletados = sum(1 for _, c in sensores if c)
                    plotar(traj, sensores, ct, ok, destino,
                           f"WNN {variante} · Na={na} Ns={ns} · T{t} (seed {s})")
                    resumo.append((f"a{na}s{ns}", f"T{t}", variante, s,
                                   f"{coletados}/{len(sensores)}", ok, ct, nome))
                    print(f"✓ a{na}s{ns} T{t} {variante:6s} · semente {s} · "
                          f"{coletados}/{len(sensores)} · {'SUCESSO' if ok else 'incompleto'} · "
                          f"CT {ct:.0f}s → {nome}")

    print(f"\n{len(resumo)} imagens em {a.saida}")
    if resumo:
        print(f"\n{'cenário':>9}  {'T':>3}  {'variante':>8}  {'sem':>3}  "
              f"{'coletados':>9}  {'ok':>3}  {'CT(s)':>6}")
        for cen, t, var, s, cob, ok, ct, _ in resumo:
            print(f"{cen:>9}  {t:>3}  {var:>8}  {s:>3}  {cob:>9}  "
                  f"{'sim' if ok else 'não':>3}  {ct:>6.0f}")
    for p in pulados:
        print(f"· pulado: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
