#!/usr/bin/env python3
"""Promove o `final` de uma corrida de 50 M para `models/` — decisão de projeto.

A política, decidida com a curva do T16-a1s12 na mesa: **o checkpoint que vale é
o `final` de 50 M**, sem seleção pelo melhor ponto da curva de validação. A curva
oscila SR 0,90–0,99 num platô desde ~13 M, então qualquer ponto é um sorteio da
mesma banda — e o `final` é o único que existe em todas as corridas (as `a1s12`
subiram antes do checkpoint periódico) e o único que não reintroduz seleção
sobre validação.

O que a promoção faz, nesta ordem:

1. recusa corrida incompleta (`procedencia.passos < 50 M`) — promover parcial
   por engano foi o modo de falha que já misturou orçamentos numa comparação;
2. avalia o `final` no banco `test` (100 sementes) e grava em
   `metrica_test_final`: o `metrica_test` que o treinador escreve é do `melhor`
   (fix 539b61e), e o número reportado tem de ser do artefato promovido;
3. arquiva o modelo anterior em `models/6M75/` — nada é apagado;
4. instala `final` (cheio) em `models/T{T}/a{...}` e o `deploy` podado em
   `models/deploy/T{T}/a{...}`, com a procedência junto.

Uso:  uv run python scripts/spec06_promover_50m.py T16-a1s12 [T32-a2s24 ...]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ALVO_PASSOS = 50_000_000


def _avaliar_final(cel: str, na: int, ns: int, t_res: int, final: Path) -> dict:
    sys.path.insert(0, str(RAIZ / "scripts"))
    from spec06_treino_multiagente import _cfgs, _ctx, avaliar_multi, construir
    from uavwnn.campanha import carregar_bancos

    cfg_tr, cfg_av = _cfgs("1b", na, ns)
    ap, termo, _ = construir(cfg_tr, na, t_res=t_res, retomar=final)
    test = carregar_bancos(na, ns).test.sementes[:100]
    m = avaliar_multi(cfg_av, ap, termo, _ctx(cfg_av), test)
    return {k: float(v) if isinstance(v, float) else v for k, v in m.items()}


def promover(cel: str) -> None:
    t_res = int(cel.split("-")[0][1:])
    a = cel.split("-")[1]                       # "a1s12"
    na, ns = int(a[1:a.index("s")]), int(a[a.index("s") + 1:])
    run = RAIZ / "runs" / "spec06-50M" / cel
    proc = json.loads((run / "procedencia.json").read_text())
    if int(proc.get("passos", 0)) < ALVO_PASSOS:
        raise SystemExit(f"{cel}: só {proc.get('passos', 0):,} passos — "
                         f"corrida incompleta NÃO é promovida")

    final = run / "checkpoints" / "final"
    deploy = run / "checkpoints" / "deploy"
    for d in (final, deploy):
        if not d.exists():
            raise SystemExit(f"{cel}: falta {d}")

    print(f"[{cel}] avaliando o final no test (100 sementes)…", flush=True)
    proc["metrica_test_final"] = _avaliar_final(cel, na, ns, t_res, final)
    proc["promovido_de"] = str(run.relative_to(RAIZ))
    print(f"[{cel}] final@test: {proc['metrica_test_final']}", flush=True)

    for raiz, novo in ((RAIZ / "models", final),
                       (RAIZ / "models" / "deploy", deploy)):
        atual = raiz / f"T{t_res}" / f"a{na}s{ns}"
        arquivo = RAIZ / "models" / "6M75" / atual.relative_to(RAIZ / "models")
        if atual.exists():
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            if arquivo.exists():
                raise SystemExit(f"{cel}: {arquivo} já existe — não sobrescrevo")
            shutil.move(str(atual), str(arquivo))
            print(f"[{cel}] arquivado: {atual.relative_to(RAIZ)} -> "
                  f"{arquivo.relative_to(RAIZ)}", flush=True)
        shutil.copytree(novo, atual)
        (atual / "procedencia.json").write_text(json.dumps(proc, indent=2))
        print(f"[{cel}] instalado: {atual.relative_to(RAIZ)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for cel in sys.argv[1:]:
        promover(cel)
