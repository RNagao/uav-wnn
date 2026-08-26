"""SPEC-08 — escada de ablação 0–4 em L→R (transferência), célula 1×12.

Carrega uma política WNN treinada (o degrau 0 = a política da Etapa 1) e a avalia
sob cada estimador da escada, no regime **L→R**: treino limpo, avaliação ruidosa.
Não treina nada — a mesma política é avaliada sob os degraus 1–4 por σ, no banco
de teste compartilhado (CRN por semente). Os parâmetros de filtro (q_a do degrau
3, q_w do degrau 4) vêm do atestado do gate.

Produz, por σ, os testes de H3 do bloco L→R: **superioridade** do degrau 4 sobre
1/2/3 (IC bootstrap pareado excluindo zero, Holm) e **equivalência** 4 ≡ 0 (TOST,
margem δ = 0,10·(t̄₁ − t̄₀)).

    uv run python scripts/spec08_escada.py --checkpoint <dir>/checkpoints/melhor
    uv run python scripts/spec08_escada.py --checkpoint <dir> --smoke

**Bloqueio B1 (RF-05):** recusa-se a rodar sem `results/etapa2/gate_rf05.json`
aprovado e coerente com o pré-registro.

Ressalva declarada: com **uma** semente de treino (o modelo da Etapa 1), o IC é
condicional a essa semente; o veredito pleno pede ≥5 (SPEC-08 T-08.2).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar  # noqa: E402  (loader oficial checkpoint→política)
from uavwnn.campanha import carregar_bancos  # noqa: E402
from uavwnn.eval.stats import bootstrap_pareado, holm, tost  # noqa: E402
from uavwnn.fusion import criar_ambiente_ruidoso  # noqa: E402
from uavwnn.fusion.consistencia import nees_por_passo  # noqa: E402
from uavwnn.fusion.ruido import ConfigRuido  # noqa: E402
from uavwnn.refenv import LAMENZA_003  # noqa: E402

GATE = Path("results/etapa2/gate_rf05.json")
PRE = Path("results/etapa2/analysis_plan.sha256")
SEED_ANALISE = 20260823


def _b1_guard(gate: Path = GATE) -> dict:
    """Bloqueio B1: sem atestado RF-05 aprovado, nenhuma rodada de RL começa.

    O caminho é parâmetro porque o atestado é **por modelo**: a consistência do
    filtro vale sob a mobilidade da política que será avaliada (T-08.1), e
    políticas diferentes (T = 16 × T = 32) exigem atestados diferentes. Um caminho
    fixo faria a escada de um modelo rodar com o `q_w` do outro.
    """
    if not gate.exists():
        sys.exit(f"B1 (RF-05): {gate} ausente — "
                 "rode scripts/spec08_gate_rf05.py antes.")
    g = json.loads(gate.read_text())
    if g.get("status") != "aprovado":
        sys.exit(f"B1 (RF-05): atestado reprovado (status={g.get('status')!r}). "
                 "Nenhuma rodada de RL desta spec executa.")
    if PRE.exists():
        esperado = PRE.read_text().split()[0]
        if g.get("hash_experimento") not in (None, esperado):
            sys.exit("B1: hash do experimento no atestado diverge do pré-registro "
                     "— o desenho mudou; refaça o gate.")
    return g


def _params_filtro(degrau: int, gate: dict) -> dict:
    """Parâmetros do estimador do degrau, tirados do atestado (melhores q)."""
    if degrau == 2:
        return {"alpha": 0.5}                       # varredura de α: T-08.5 (adiada)
    if degrau == 3:
        return {"q_a": float(gate["degrau3"]["melhor"])}
    if degrau == 4:
        return {"q_w": float(gate["degrau4"]["melhor"])}
    return {}


def _nees_medio(log) -> float | None:
    covs = [x for x in log if "cov" in x]
    if not covs:
        return None
    er = np.array([x["erro"] for x in covs])
    co = np.array([x["cov"] for x in covs])
    return float(np.mean(nees_por_passo(er, co)))


def rodar_cell(cfg, politica, degrau, sigma, sementes, gate):
    """Uma célula (degrau, σ): **um** ambiente, reset por episódio (rápido).

    Devolve `(cts[N], oks[N], nees_medio|None)` pareados por semente (CRN:
    `indice_ruido = semente`).
    """
    amb = criar_ambiente_ruidoso(
        cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=sigma),
        **_params_filtro(degrau, gate))
    cts, oks, nees = [], [], []
    for sd in sementes:
        random.seed(int(sd)); np.random.seed(int(sd))
        amb.indice_ruido = int(sd)
        obs, _ = amb.reset()
        ct, ok = float(cfg.max_episode_length), False
        for _ in range(400):
            obs, _, term, trunc, infos = amb.step({a: politica(obs[a]) for a in amb.agents})
            if infos and "avg_reward" in next(iter(infos.values())):
                info = next(iter(infos.values()))
                ct = float(info.get("completion_time", cfg.max_episode_length))
                ok = bool(info.get("all_collected", False))
                break
            if any(term.values()) or any(trunc.values()):
                break
        cts.append(ct); oks.append(ok)
        nm = _nees_medio(amb.log_estimacao)
        if nm is not None:
            nees.append(nm)
    return np.array(cts), np.array(oks), (float(np.mean(nees)) if nees else None)


def _resumo(cts, oks, nees):
    r = {"ct_medio": float(cts.mean()), "sr": float(oks.mean()), "n_ep": int(cts.size)}
    if nees is not None:
        r["nees_medio"] = nees
    return r


def _analise_sigma(ct0, ct, rng) -> dict:
    """Testes de H3 (L→R) para um σ: degradação, superioridade 4>{1,2,3}, TOST 4≡0."""
    deg = {d: bootstrap_pareado(ct[d], ct0, rng).como_linha() for d in ct}
    sup = {}
    ps = []
    for d in (1, 2, 3):
        if d not in ct:
            continue
        diff = bootstrap_pareado(ct[4], ct[d], rng)   # δ = t̄₄ − t̄_d; 4 supera se <0
        sup[d] = {**diff.como_linha(),
                  "4_supera": bool(diff.conclusivo and diff.delta_media < 0)}
        ps.append(diff.p_wilcoxon)
    if ps:
        for (d, _), p_adj in zip(((d, sup[d]) for d in (1, 2, 3) if d in sup), holm(ps)):
            sup[d]["p_holm"] = float(p_adj)
    dano = float(ct[1].mean() - ct0.mean()) if 1 in ct else 0.0
    delta = max(0.10 * abs(dano), 1e-9)
    eq = tost(ct[4], ct0, delta, rng).como_linha() if 4 in ct else None
    return {"degradacao_vs_teto": deg, "superioridade_4": sup,
            "equivalencia_4_vs_0": eq, "dano_bruto_s": dano, "delta": delta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--aproximador", default=None,
                    help="família; padrão: lê do config.json do run")
    ap.add_argument("--sigmas", default="1,3,5,10")
    ap.add_argument("--degraus", default="0,1,2,3,4")
    ap.add_argument("--episodios", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--saida", type=Path, default=Path("results/etapa2/escada_lr.json"))
    ap.add_argument("--gate", type=Path, default=GATE,
                    help="atestado RF-05 do MODELO avaliado (ver _b1_guard)")
    a = ap.parse_args()

    gate = _b1_guard(a.gate)
    if a.smoke:
        a.sigmas, a.degraus, a.episodios = "5", "0,1,2,3,4", min(a.episodios, 3)

    aprox = a.aproximador
    if aprox is None:
        cfgjson = a.checkpoint.parent.parent / "config.json"
        aprox = json.loads(cfgjson.read_text())["config"]["aproximador"]

    cfg = LAMENZA_003(1, 12, phase="eval")
    # A escada roda 17 células × 500 episódios sobre a MESMA política, sem
    # nunca escrever nela — o caso exato para a LUT plana.
    politica, _ = montar(aprox, cfg, a.checkpoint, plana=True)

    sigmas = [float(x) for x in a.sigmas.split(",")]
    degraus = [int(x) for x in a.degraus.split(",")]
    sementes = carregar_bancos(1, 12).test.sementes[:a.episodios]
    rng = np.random.default_rng(SEED_ANALISE)

    print(f"escada L→R · {aprox} · degraus {degraus} · σ {sigmas} · "
          f"{len(sementes)} episódios (1 semente de treino)")
    print(f"B1 OK — gate {gate['status']} (q_a={gate['degrau3']['melhor']}, "
          f"q_w={gate['degrau4']['melhor']})\n")

    artefato = {"config": {"aproximador": aprox, "checkpoint": str(a.checkpoint),
                           "sigmas": sigmas, "degraus": degraus,
                           "episodios": len(sementes), "regime": "LR",
                           "sementes_treino": 1},
                "hash_filtro": gate.get("hash_filtro"),
                "gate": str(a.gate), "celulas": {}}

    ct0 = ok0 = None
    if 0 in degraus:
        ct0, ok0, _ = rodar_cell(cfg, politica, 0, 0.0, sementes, gate)
        print(f"degrau 0 (teto): CT {ct0.mean():.1f} s · SR {ok0.mean():.0%}")

    for sigma in sigmas:
        ct_d, ok_d = {}, {}
        cel = artefato["celulas"][f"sigma_{sigma:g}"] = {}
        if ct0 is not None:
            cel["degrau0"] = _resumo(ct0, ok0, None)
        for degrau in degraus:
            if degrau == 0:
                continue
            cts, oks, nees = rodar_cell(cfg, politica, degrau, sigma, sementes, gate)
            ct_d[degrau], ok_d[degrau] = cts, oks
            cel[f"degrau{degrau}"] = _resumo(cts, oks, nees)
            extra = f" · NEES {nees:.2f}" if nees is not None else ""
            print(f"σ={sigma:g} degrau {degrau}: CT {cts.mean():.1f} s · "
                  f"SR {oks.mean():.0%}{extra}")
        if ct0 is not None and ct_d:
            cel["analise"] = _analise_sigma(ct0, ct_d, rng)
            e = cel["analise"]["equivalencia_4_vs_0"]
            s = cel["analise"]["superioridade_4"]
            if e is not None:
                print(f"   4≡0: {'EQUIVALENTE' if e['equivalente'] else 'não'} "
                      f"(δ={e['delta']:.2f}, meia-largura={e['meia_largura']:.2f}, "
                      f"{'informativa' if e['informativa'] else 'NÃO informativa'})")
            sup_ok = [d for d in (1, 2, 3) if s.get(d, {}).get("4_supera")]
            print(f"   4 supera: {sup_ok or 'nenhum'}")
        print()

    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(artefato, indent=2, ensure_ascii=False))
    print(f"artefato: {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
