"""T-06.0 — orçamento de simulação medido, `B_max`, e o veredito sobre P2.

**É a primeira tarefa da SPEC-06 e ela bloqueia as outras**, porque `B_max` sai
daqui e todo o dimensionamento da campanha sai de `B_max`.

O PRD §6.1 é honesto quanto ao status do número: 0,11 s por episódio de 120 s com
4 UAVs / 12 sensores, extrapolado para **8-30 h num núcleo** para 50 M passos —
isto é, **≈ 460-1 700 passos/s**. É estimativa, e é isso que se confronta.

A DECOMPOSIÇÃO É O PONTO, NÃO O AGREGADO

| parcela | WNN | MLP | distingue os métodos? |
|---|---|---|---|
| simulação (`gradysim`, eventos, mobilidade) | comum | comum | **não** |
| montagem da observação (`O(Ns log Ns)`) | comum | comum | **não** |
| codificação (termômetro) | só WNN | — | sim |
| passo de aprendizado | consultas + somas | MACs | **sim** |

> Se a simulação dominar, o tempo de parede de treino é **comum aos dois lados** e
> não sustenta a alegação "sem GPU". O que sustenta é a parcela de aprendizado,
> medida isolada. As duas são reportadas: parede total (contra os 7 dias de GPU por
> célula de Lamenza) e parede da parcela de aprendizado (a comparação limpa).
> Misturá-las repete o erro que a §5.5 proíbe para T2.

A montagem da observação **é** um método do ambiente
(`observe_simulation_relative_positions`), então é cronometrada **direto**, e não
estimada por subtração. Simulação e montagem ficariam indistinguíveis se só
`env.step()` fosse medido.

P2 NÃO É TESTÁVEL COMO POSTA — e é achado, não desistência

A T-06.0 pede: *"fidelidade de `update_rate = 0,1` contra `0,01`"*. Mas
`update_rate` **não é parâmetro livre**: o ambiente o deriva
(`environment.py:473`),

    update_rate = algorithm_iteration_interval / 2

Então variar `update_rate` exige variar `algorithm_iteration_interval`, o que muda
**também a frequência de decisão**. As duas são inseparáveis sem editar o ambiente,
e editar o ambiente é proibido pela regra do projeto (clonar o cenário 3 sem
modificação). Um teste que variasse as duas juntas e atribuísse a diferença ao
integrador de mobilidade estaria errado por confusão de variáveis.

O que **é** testável, e responde a pergunta prática que motivou P2: qual o efeito do
par `(aii, update_rate = aii/2)` sobre `CT`. Isso quantifica de passagem a
assimetria que a SPEC-00 registrou — o `visualize_performance.py` avalia o MADDPG a
`aii = 0,5` e o guloso a `0,1`, cinco vezes mais reativo, variável não controlada na
fig. 7.3.

Uso:
    uv run python scripts/spec06_orcamento.py --episodios 3
    uv run python scripts/spec06_orcamento.py --so-p2 --episodios 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from uavwnn.agents import APROXIMADORES  # noqa: E402
from uavwnn.agents.base import ContextoEpisodio, Observacao  # noqa: E402
from uavwnn.encoding import TermometroQuantil  # noqa: E402
from uavwnn.eval import ArvoreSementes, bootstrap_pareado  # noqa: E402
from uavwnn.refenv import LAMENZA_003, make_env, run_episode  # noqa: E402

CELULAS = [(na, ns) for na in (1, 2, 4, 8) for ns in (12, 24, 36)]
"""`Na × Ns` da T-06.0. Inclui `Na = 1`, que é a célula base da Etapa 1 e **não
tem** checkpoint de MADDPG — o baseline daquela célula é o guloso."""

INTERVALOS = (0.5, 2.0)
"""`algorithm_iteration_interval`, e os dois valores existem porque as fontes
discordam: o `benchmark.py` que gerou o §7.2 usa **0,5** (e é o default do nosso
`EnvConfig`), a tab. 7.1 da dissertação diz **2**. A vazão em passos de ambiente
por segundo escala direto com isso — a 0,5 um episódio de 500 s tem 1000 decisões,
a 2 s tem 250 —, então reportar um número só seria reportar a fonte escolhida."""

BITS_POR_ESCALAR = 32
PASSOS_ESTIMADOS_PRD = 50_000_000
FAIXA_PRD_HORAS = (8.0, 30.0)


def _ctx(cfg) -> ContextoEpisodio:
    return ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval)


def medir_celula(na: int, ns: int, aii: float, episodios: int,
                 seed: int = 0) -> dict:
    """Vazão e as quatro parcelas, para uma célula.

    A ação é **constante** no laço do ambiente: o que se mede aqui é o custo do
    ambiente, e uma política de verdade acrescentaria a sua própria latência à
    parcela errada.
    """
    cfg = LAMENZA_003(num_drones=na, num_sensors=ns, phase="train")
    cfg = type(cfg)(**{**cfg.kwargs(), "algorithm_iteration_interval": aii})
    ctx = _ctx(cfg)
    d_obs = cfg.observation_size

    # -- 1. ambiente: `env.step()` com ação constante ----------------------
    env = make_env(cfg)
    random.seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset()
    acao = {a: np.array([0.5, 1.0], np.float32) for a in obs}

    passos = 0
    t_env = 0.0
    t_obs = 0.0
    n_obs = 0
    for _ in range(episodios):
        obs, _ = env.reset()
        while True:
            t0 = time.perf_counter()
            obs, _, term, trunc, info = env.step(acao)
            t_env += time.perf_counter() - t0
            passos += 1
            # A montagem da observação é método do ambiente, então dá para
            # cronometrá-la DIRETO. Ela já ocorreu dentro do `step`; chamá-la de
            # novo mede o mesmo trabalho sem alterar o estado da simulação.
            if n_obs < 200:
                t1 = time.perf_counter()
                env.observe_simulation_relative_positions()
                t_obs += time.perf_counter() - t1
                n_obs += 1
            if (info and "avg_reward" in next(iter(info.values()))) \
                    or any(term.values()) or any(trunc.values()):
                break
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass

    # -- 2. codificação: termômetro sobre observações reais ----------------
    rng = np.random.default_rng(seed)
    corpus = rng.uniform(-1.0, 1.5, (512, d_obs))
    termo = TermometroQuantil(BITS_POR_ESCALAR).ajustar(corpus)
    amostra = corpus[:200]
    t0 = time.perf_counter()
    for v in amostra:
        termo.codificar(v[None, :])
    t_cod = (time.perf_counter() - t0) / len(amostra)

    # -- 3. aprendizado: forward + atualização, por família ----------------
    aprendizado = {}
    for nome in ("ntupla", "mlp"):
        if nome == "mlp":
            ap = APROXIMADORES[nome](n_bits_ou_dim=d_obs)
            entradas = [v for v in amostra]
        else:
            ap = APROXIMADORES[nome](n_bits=termo.n_bits,
                                     n_dims_codificador=d_obs,
                                     limiares_por_dim=BITS_POR_ESCALAR)
            entradas = [termo.codificar(v[None, :])[0] for v in amostra]
        d = np.zeros(3)
        d[0] = 0.01
        for x in entradas[:20]:                      # aquece
            ap.f(x)
        t0 = time.perf_counter()
        for x in entradas:
            ap.f(x)
            ap.ascender_ator(x, d)
            ap.regredir_critico(x, 1.0)
        aprendizado[nome] = (time.perf_counter() - t0) / len(entradas)

    por_passo_env = t_env / max(1, passos)
    return {
        "na": na, "ns": ns, "aii": aii, "passos": passos, "episodios": episodios,
        "decisoes_por_episodio": passos / episodios,
        "s_ambiente": por_passo_env,
        "s_montagem_obs": t_obs / max(1, n_obs),
        "s_codificacao": t_cod,
        "s_aprendizado_ntupla": aprendizado["ntupla"],
        "s_aprendizado_mlp": aprendizado["mlp"],
        "vazao_ambiente": 1.0 / por_passo_env if por_passo_env else float("nan"),
        "vazao_wnn": 1.0 / (por_passo_env + t_cod + aprendizado["ntupla"]),
        "vazao_mlp": 1.0 / (por_passo_env + aprendizado["mlp"]),
    }


def p2_efeito_do_intervalo(episodios: int, celulas=((1, 12), (8, 36))) -> list[dict]:
    """O que P2 de fato pode medir: efeito do par `(aii, update_rate = aii/2)`.

    Guloso sobre os **mesmos** cenários, dois intervalos. Pareado por semente, que é
    o que permite IC estreito com poucos episódios.
    """
    from uavwnn.agents import BASELINES

    arv = ArvoreSementes(20260821)
    saida = []
    for na, ns in celulas:
        ct = {}
        for aii in (0.5, 0.1):
            cfg = LAMENZA_003(num_drones=na, num_sensors=ns, phase="eval")
            cfg = type(cfg)(**{**cfg.kwargs(), "algorithm_iteration_interval": aii})
            ctx = _ctx(cfg)
            guloso = BASELINES["guloso"]()
            vals = []
            for s in range(episodios):
                guloso.reset(ctx, np.random.default_rng(s))

                def politica(observacao, _g=guloso, _c=ctx):
                    return _g.act(Observacao.decodificar(observacao, _c)).como_array()

                vals.append(run_episode(cfg, politica, seed=s).completion_time)
            ct[aii] = np.array(vals, float)

        d = bootstrap_pareado(ct[0.1], ct[0.5], arv.bootstrap(f"p2-{na}-{ns}"))
        from scipy.stats import ks_2samp
        ks = ks_2samp(ct[0.1], ct[0.5])
        media = float(ct[0.5].mean())
        saida.append({
            "na": na, "ns": ns, "n": episodios,
            "ct_aii_050": media, "ct_aii_010": float(ct[0.1].mean()),
            "delta": d.delta_media, "ic": [d.delta_ic_lo, d.delta_ic_hi],
            "contem_zero": bool(d.contem_zero),
            "delta_rel": abs(d.delta_media) / media if media else float("nan"),
            "ks_p": float(ks.pvalue),
            "aceita": bool(d.contem_zero and abs(d.delta_media) / media < 0.01
                           and ks.pvalue > 0.05),
        })
    return saida


def b_max(vazao: float, nucleos: int, horas: float, config: int, sementes: int) -> float:
    """`B_max = v·W/(C·R)`, com `W` em segundos-núcleo."""
    return vazao * nucleos * horas * 3600.0 / (config * sementes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=3,
                    help="episódios por célula na medição de vazão")
    ap.add_argument("--episodios-p2", type=int, default=30)
    ap.add_argument("--nucleos", type=int, default=12)
    ap.add_argument("--horas", type=float, default=72.0,
                    help="orçamento de parede alocado à campanha")
    ap.add_argument("--configuracoes", type=int, default=27,
                    help="`C` da conta: pontos da varredura T × n × N")
    ap.add_argument("--sementes", type=int, default=5, help="`R` da conta")
    ap.add_argument("--so-p2", action="store_true")
    ap.add_argument("--celulas", default=None, help="ex. '1:12,8:36'")
    ap.add_argument("--saida", default="reports/spec06-orcamento.json")
    a = ap.parse_args()

    art: dict = {"config": vars(a)}

    if not a.so_p2:
        celulas = CELULAS
        if a.celulas:
            celulas = [tuple(int(x) for x in c.split(":"))
                       for c in a.celulas.split(",")]
        print("=" * 84)
        print("VAZÃO POR CÉLULA E POR PARCELA (µs por decisão)")
        print("=" * 84)
        print(f"{'Na':>3} {'Ns':>3} {'aii':>5} {'dec/ep':>7} {'ambiente':>9} "
              f"{'obs':>7} {'codif':>7} {'apr.WNN':>8} {'apr.MLP':>8} "
              f"{'v.WNN':>8} {'v.MLP':>8}")
        linhas = []
        for aii in INTERVALOS:
            for na, ns in celulas:
                r = medir_celula(na, ns, aii, a.episodios)
                linhas.append(r)
                print(f"{na:>3} {ns:>3} {aii:>5.1f} {r['decisoes_por_episodio']:>7.0f} "
                      f"{r['s_ambiente']*1e6:>9.0f} {r['s_montagem_obs']*1e6:>7.0f} "
                      f"{r['s_codificacao']*1e6:>7.0f} "
                      f"{r['s_aprendizado_ntupla']*1e6:>8.0f} "
                      f"{r['s_aprendizado_mlp']*1e6:>8.0f} "
                      f"{r['vazao_wnn']:>8.0f} {r['vazao_mlp']:>8.0f}")
        art["vazao"] = linhas

        # -- quem domina? ---------------------------------------------------
        print("\nFRAÇÃO DO TEMPO POR PARCELA — é isto que decide a narrativa")
        print(f"{'Na':>3} {'Ns':>3} {'aii':>5} {'ambiente':>9} {'codif':>7} "
              f"{'aprend.':>8}")
        for r in linhas:
            tot = r["s_ambiente"] + r["s_codificacao"] + r["s_aprendizado_ntupla"]
            print(f"{r['na']:>3} {r['ns']:>3} {r['aii']:>5.1f} "
                  f"{r['s_ambiente']/tot:>8.1%} {r['s_codificacao']/tot:>6.1%} "
                  f"{r['s_aprendizado_ntupla']/tot:>7.1%}")

        # -- veredito sobre a estimativa do PRD -----------------------------
        base = [r for r in linhas if r["na"] == 4 and r["ns"] == 12] or linhas
        v = float(np.mean([r["vazao_wnn"] for r in base]))
        horas = PASSOS_ESTIMADOS_PRD / v / 3600.0
        lo, hi = FAIXA_PRD_HORAS
        if horas < lo:
            veredito = f"PESSIMISTA por {lo/horas:.1f}×"
        elif horas > hi:
            veredito = f"OTIMISTA por {horas/hi:.1f}×"
        else:
            veredito = "CONFIRMADA"
        print(f"\nESTIMATIVA DO PRD §6.1 ({lo:.0f}-{hi:.0f} h para "
              f"{PASSOS_ESTIMADOS_PRD/1e6:.0f} M passos): {veredito}")
        print(f"  vazão medida (Na=4, Ns=12, WNN): {v:.0f} passos/s "
              f"-> {horas:.1f} h num núcleo")
        print(f"  a faixa do PRD equivale a {PASSOS_ESTIMADOS_PRD/(hi*3600):.0f}"
              f"-{PASSOS_ESTIMADOS_PRD/(lo*3600):.0f} passos/s")
        art["veredito_prd"] = {"vazao_medida": v, "horas_50M": horas,
                               "veredito": veredito}

        bm = b_max(v, a.nucleos, a.horas, a.configuracoes, a.sementes)
        print(f"\nB_max = v·W/(C·R) = {v:.0f} × {a.nucleos}×{a.horas:.0f} h / "
              f"({a.configuracoes}×{a.sementes}) = {bm/1e6:.1f} M passos "
              f"por configuração-semente")
        if bm < 10e6:
            print("  ABAIXO de ~10 M: acionar a linha de corte do PRD §8 na ordem")
            print("  publicada (varreduras n × N primeiro). NÃO reduzir R — reduzir")
            print("  R destrói a única defesa contra o problema do §7.3.")
        art["b_max"] = {"valor": bm, "vazao": v, "nucleos": a.nucleos,
                        "horas": a.horas, "configuracoes": a.configuracoes,
                        "sementes": a.sementes, "abaixo_do_corte": bool(bm < 10e6)}

    # -- P2 ----------------------------------------------------------------
    print("\n" + "=" * 84)
    print("P2 — efeito do par (aii, update_rate = aii/2) sobre CT")
    print("=" * 84)
    print("`update_rate` NÃO é parâmetro livre: o ambiente o deriva de")
    print("`algorithm_iteration_interval/2` (environment.py:473). Variá-lo isolado")
    print("exigiria editar o ambiente, o que a regra do projeto proíbe. O que se")
    print("mede é o par — e isso quantifica a assimetria da fig. 7.3.\n")
    p2 = p2_efeito_do_intervalo(a.episodios_p2)
    print(f"{'Na':>3} {'Ns':>3} {'CT@0,5':>8} {'CT@0,1':>8} {'Δ':>8} "
          f"{'IC95%':>20} {'|Δ|/média':>10} {'KS p':>8} {'aceita':>7}")
    for r in p2:
        print(f"{r['na']:>3} {r['ns']:>3} {r['ct_aii_050']:>8.1f} "
              f"{r['ct_aii_010']:>8.1f} {r['delta']:>+8.2f} "
              f"[{r['ic'][0]:>+7.2f},{r['ic'][1]:>+7.2f}] "
              f"{r['delta_rel']:>10.2%} {r['ks_p']:>8.3f} "
              f"{'sim' if r['aceita'] else 'NÃO':>7}")
    art["p2"] = p2
    art["p2_nota"] = ("update_rate é derivado de algorithm_iteration_interval/2 "
                      "(environment.py:473), então não é variável independente. "
                      "P2 como posta não é testável sem editar o ambiente; o que "
                      "se mede é o efeito do par.")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nartefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
