"""Bancos de episódios, horizonte permissivo (P5), e a verificação da §7.1.

Três coisas que a SPEC-06 exige **antes** da primeira corrida de treino:

1. **Bancos `test` (500) e `val` (100), disjuntos e versionados** (§4.2), com hash
   registrado. Gerados uma vez por célula e gravados em `runs/bancos/`.

2. **`max_episode_length_1a` (P5)**, por definição operacional em vez de chute:
   `3 × CT_p99(guloso)`, múltiplos de 100 s, piso 500, teto 2000 — com a
   **verificação funcional** de que sob esse horizonte a taxa de sucesso do guloso
   é ≥ 99 %. É o que "permissivo" significa no texto: o horizonte não é o gargalo.

3. **A interação `max_seconds_stalled` × RF-02**, que a T-06.1 manda confirmar
   contra o código de referência antes de 1a rodar:

   > O RF-02 manda não podar quando todos os sensores são coletados; mas
   > `max_seconds_stalled = 30 s` poda 30 s após a última visita, já que nenhuma
   > visita nova ocorre. O efeito líquido é que o episódio termina 30 s depois de
   > completo, e o horizonte permissivo só atua sobre episódios **incompletos**.

   Isso define quanta "experiência pós-conclusão" o agente recebe, e Lamenza atribui
   a isso ganho grande de desempenho (§5). Medimos em vez de supor.

Uso:
    uv run python scripts/spec06_bancos.py --celulas 1:12,2:12,4:12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from uavwnn.agents import BASELINES  # noqa: E402
from uavwnn.agents.base import ContextoEpisodio, Observacao  # noqa: E402
from uavwnn.campanha import carregar_bancos, horizonte_permissivo  # noqa: E402
from uavwnn.refenv import LAMENZA_003, run_episode  # noqa: E402


def _cfg(na: int, ns: int, horizonte: float | None = None, cegos: bool = True,
         fase: str = "eval"):
    """Config de 1a: `N*_a = 0` (agentes cegos aos demais, §7.1).

    `fase` importa e a distinção é sutil: em `eval` o episódio **termina** ao
    coletar tudo (`end_when_all_collected=True`), então não há experiência
    pós-conclusão nenhuma. Em `train` ele segue, e é aí que o
    `max_seconds_stalled` age. Medir o excedente em `eval` daria zero por
    construção — foi o erro da primeira medição."""
    base = LAMENZA_003(num_drones=na, num_sensors=ns, phase=fase).kwargs()
    if cegos:
        base["state_num_closest_drones"] = 0
    if horizonte is not None:
        base["max_episode_length"] = horizonte
    from uavwnn.refenv.runner import EnvConfig
    return EnvConfig(**base)


def _guloso_no_banco(cfg, sementes) -> dict:
    """Guloso sobre um banco. Devolve `CT` por episódio e a taxa de sucesso."""
    ctx = ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval,
        max_episode_length=cfg.max_episode_length,
        state_num_closest_drones=cfg.state_num_closest_drones)
    pol = BASELINES["guloso"]()
    ct, ok, passos = [], [], []
    for s in sementes:
        pol.reset(ctx, np.random.default_rng(int(s)))

        def politica(observacao, _p=pol, _c=ctx):
            return _p.act(Observacao.decodificar(observacao, _c)).como_array()

        r = run_episode(cfg, politica, seed=int(s))
        ct.append(r.completion_time)
        ok.append(r.all_collected)
        passos.append(r.steps)
    return {"ct": np.array(ct, float), "sucesso": np.array(ok, bool),
            "passos": np.array(passos, int)}


def excedente_pos_conclusao(cfg, sementes, n: int = 40) -> dict:
    """Quanto tempo o episódio segue **depois** de tudo coletado.

    Mede o efeito líquido de `max_seconds_stalled` contra o RF-02: quanta
    experiência pós-conclusão o agente recebe. Lamenza atribui a isso ganho grande
    de desempenho (§5), então o número importa.

    NÃO USA `run_episode`, E ISSO É O PONTO. Aquele runner quebra na **primeira
    `info` não vazia**, verbatim do `visualize_performance.py` — é um runner de
    **avaliação**, e o instante que ele registra é o da conclusão, não o do fim do
    episódio. Medir o excedente por ele daria **zero por construção**, o que este
    script já produziu duas vezes antes de a causa ficar clara.

    Aqui o laço vai até `terminations`/`truncations`, e o instante da conclusão sai
    de `env.sensors_collected` alcançando `num_sensors`.
    """
    import random

    from uavwnn.refenv import make_env

    ctx = ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval,
        max_episode_length=cfg.max_episode_length,
        state_num_closest_drones=cfg.state_num_closest_drones)
    pol = BASELINES["guloso"]()
    dur, exced = [], []
    passos_max = int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2

    for s in list(sementes)[:n]:
        env = make_env(cfg)
        random.seed(int(s))
        np.random.seed(int(s))
        obs, _ = env.reset()
        pol.reset(ctx, np.random.default_rng(int(s)))
        t_conclusao = None
        passos = 0
        for _ in range(passos_max):
            acoes = {a: pol.act(Observacao.decodificar(obs[a], ctx)).como_array()
                     for a in env.agents}
            obs, _, term, trunc, _ = env.step(acoes)
            passos += 1
            if t_conclusao is None and env.sensors_collected == cfg.num_sensors:
                t_conclusao = passos * cfg.algorithm_iteration_interval
            if any(term.values()) or any(trunc.values()):
                break
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
        if t_conclusao is None:
            continue
        duracao = passos * cfg.algorithm_iteration_interval
        dur.append(duracao)
        exced.append(duracao - t_conclusao)

    if not exced:
        return {"n": 0}
    e = np.array(exced, float)
    return {"n": len(e), "media": float(e.mean()), "mediana": float(np.median(e)),
            "min": float(e.min()), "max": float(e.max()),
            "duracao_media": float(np.mean(dur))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--celulas", default="1:12,2:12,4:12")
    ap.add_argument("--n-test", type=int, default=None,
                    help="reduz o banco de teste (padrão: 500, o do §7.2)")
    ap.add_argument("--saida", default="reports/spec06-bancos.json")
    a = ap.parse_args()

    celulas = [tuple(int(x) for x in c.split(":")) for c in a.celulas.split(",")]
    art: dict = {"config": vars(a), "celulas": {}}

    print("=" * 78)
    print("BANCOS test/val — §4.2, disjuntos e versionados")
    print("=" * 78)
    print(f"{'célula':>8} {'test':>6} {'val':>5} {'hash test':>18} {'hash val':>18}")
    pares = {}
    for na, ns in celulas:
        par = carregar_bancos(na, ns)
        pares[(na, ns)] = par
        print(f"{f'{na}a{ns}s':>8} {len(par.test):>6} {len(par.val):>5} "
              f"{par.test.hash:>18} {par.val.hash:>18}")
        art["celulas"][f"a{na}s{ns}"] = {"bancos": par.como_dict()}

    print("\n" + "=" * 78)
    print("P5 — `max_episode_length_1a` por definição operacional")
    print("=" * 78)
    print("horizonte = 3 × CT_p99(guloso), múltiplos de 100 s, piso 500, teto 2000")
    print("verificação: taxa de sucesso do guloso ≥ 99 % sob o horizonte escolhido\n")
    print(f"{'célula':>8} {'CT_p99@500':>11} {'horizonte':>10} {'SR@horiz':>9} "
          f"{'permissivo?':>12}")
    for na, ns in celulas:
        sem = pares[(na, ns)].test.sementes
        if a.n_test:
            sem = sem[:a.n_test]
        # 1º passo: distribuição do guloso no horizonte da tab. 7.1 (500 s)
        base = _guloso_no_banco(_cfg(na, ns), sem)
        p99 = float(np.percentile(base["ct"], 99))
        h = horizonte_permissivo(base["ct"])
        # 2º passo: verificar que o horizonte escolhido não é o gargalo
        sob_h = base if h == 500.0 else _guloso_no_banco(_cfg(na, ns, h), sem)
        sr = float(sob_h["sucesso"].mean())
        ok = sr >= 0.99
        print(f"{f'{na}a{ns}s':>8} {p99:>11.1f} {h:>10.0f} {sr:>8.1%} "
              f"{'sim' if ok else 'NÃO':>12}")
        art["celulas"][f"a{na}s{ns}"]["p5"] = {
            "ct_p99_em_500": p99, "horizonte": h, "sr_guloso": sr,
            "permissivo": bool(ok), "n_episodios": len(sem),
            "ct_medio": float(base["ct"].mean()),
            "sr_em_500": float(base["sucesso"].mean())}

    print("\n" + "=" * 78)
    print("EXCEDENTE PÓS-CONCLUSÃO — `max_seconds_stalled` × RF-02")
    print("=" * 78)
    print("a T-06.1 prevê ~30 s; medido sobre episódios bem-sucedidos do guloso,")
    print("na fase `train` (`end_when_all_collected=False`) — em `eval` o episódio")
    print("termina na coleta e o excedente seria zero por construção.\n")
    print(f"{'célula':>8} {'n':>4} {'média':>7} {'mediana':>8} {'min':>6} "
          f"{'max':>6} {'duração':>8}")
    for na, ns in celulas:
        # `fase="train"`: é a configuração em que o episódio NÃO termina ao
        # coletar tudo, e portanto a única em que o excedente existe.
        e = excedente_pos_conclusao(_cfg(na, ns, fase="train"),
                                    pares[(na, ns)].test.sementes)
        if not e["n"]:
            print(f"{f'{na}a{ns}s':>8} — nenhum episódio bem-sucedido")
            continue
        print(f"{f'{na}a{ns}s':>8} {e['n']:>4} {e['media']:>7.1f} "
              f"{e['mediana']:>8.1f} {e['min']:>6.1f} {e['max']:>6.1f} "
              f"{e['duracao_media']:>8.1f}")
        art["celulas"][f"a{na}s{ns}"]["excedente_pos_conclusao"] = e

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nartefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
