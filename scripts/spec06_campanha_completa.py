"""Campanha comparável — MADDPG × WNN T=16 × WNN T=32, limpo e sob ruído.

MESMO FORMATO DOS CSVs DE `reports/maddpg`, e é isso que a torna comparável:

    1 episódio  →  um tempo de conclusão
    1 RUN       →  `E` episódios (200, o `total_runs` do `test.py` de Lamenza)
    1 campanha  →  `R` runs (100), e o desvio é SOBRE AS RUNS

O desvio entre runs mede a variabilidade do número publicado; o desvio entre
episódios mede outra coisa e é muito maior. Confundi-los muda a conclusão.

SEMEADURA — E POR QUE ELA É A MESMA PARA TODOS OS MODELOS

    run `r` usa as sementes `[r·E, (r+1)·E)`

Idêntico ao `spec02_maddpg_campanha.py`, de propósito: com a mesma semente os
modelos veem o MESMO cenário, e a comparação é pareada. Trocar o esquema de
semeadura entre modelos transformaria diferença de cenário em diferença de
método.

O REGIME DE RUÍDO É SORTEADO POR EPISÓDIO, E A CONDIÇÃO FICA GRAVADA

`--ruido sorteio` desenha, para cada episódio, um ponto da escada COMPLETA
(degrau 0 limpo, mais degraus 1–4 × σ ∈ {1,3,5,10} — 17 condições). O sorteio é
função **só da semente do episódio**, então:

  * o mesmo episódio recebe a mesma condição em TODOS os modelos (pareamento);
  * a média da campanha é o desempenho sob regime aleatório de ruído;
  * e como cada episódio grava a sua condição no CSV `-episodios.csv`, dá para
    fatiar depois por degrau e por σ — 20 000 episódios ÷ 17 ≈ 1 180 por célula,
    sem rodar 17 campanhas.

Rodar as 17 condições cheias custaria 17×; sortear e gravar custa 1× e entrega
as duas leituras. O preço é que cada célula da escada tem 1 180 episódios em vez
de 20 000 — suficiente para o efeito medido na SPEC-08 (dezenas de segundos).

Os parâmetros do filtro (`q_a` do degrau 3, `q_w` do degrau 4) saem do atestado
RF-05 **daquele modelo naquela célula**: a consistência vale sob a mobilidade da
política avaliada, e usar o `q_w` de outra política é medir o filtro errado.

    uv run --group baseline python scripts/spec06_campanha_completa.py \
        --modelo maddpg --na 2 --ns 24 --ruido sorteio --gate <atestado.json>

    uv run python scripts/spec06_campanha_completa.py \
        --modelo wnn --rotulo wnn-T16 --na 1 --ns 12 --ruido limpo \
        --checkpoint runs/.../checkpoints/final
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

RAIZ = Path(__file__).resolve().parents[1]
SEED_SORTEIO = 20260825
"""Semente do sorteio de condição. Fixa e separada das sementes de cenário: o
sorteio tem de ser reprodutível E independente do modelo."""

NOME_DEGRAU = {0: "limpo", 1: "ruído bruto, SEM filtro", 2: "passa-baixa",
               3: "KF velocidade constante", 4: "KF + controle"}

_APELIDO = {1: "sem-kf", 4: "com-kf"}
"""Rótulo dos dois degraus que a comparação principal usa. `1` é o piso (a medida
crua) e `4` é a proposta; `sem-kf`/`com-kf` diz isso no nome do arquivo, que é o
que alguém lê ao abrir a pasta. Os degraus 2 e 3 caem no genérico `d2`/`d3` —
existem para a escada da SPEC-08, não para esta comparação."""


def _rotulo_regime(degraus: list[int], sigmas: list[float]) -> str:
    s = "sigma" + "-".join(f"{x:g}" for x in sigmas)
    if len(degraus) == 1:
        d = degraus[0]
        return f"{s}-{_APELIDO.get(d, f'd{d}')}"
    return s


def montar_escada(degraus: list[int], sigmas: list[float]) -> list[tuple[int, float]]:
    """As condições sorteáveis. O degrau 0 é limpo, então σ não se aplica e ele
    entra **uma vez só** — repeti-lo por σ daria peso extra ao caso sem ruído."""
    fora = [(0, 0.0)] if 0 in degraus else []
    return fora + [(d, s) for d in degraus if d != 0 for s in sigmas]


def condicao_do_episodio(seed: int, escada: list[tuple[int, float]]) -> tuple[int, float]:
    """Condição da escada para esta semente. **Só depende da semente.**

    É o que garante pareamento entre modelos: MADDPG e WNN avaliam o episódio
    `s` sob o mesmo degrau e o mesmo σ, então a diferença medida é do método.

    Depende também do TAMANHO da escada — restringir σ muda o sorteio de todos
    os episódios. Por isso o rótulo do regime carrega o σ: campanhas com escadas
    diferentes não podem ir para o mesmo CSV.
    """
    rng = np.random.default_rng(SEED_SORTEIO + int(seed))
    return escada[int(rng.integers(len(escada)))]


# --- políticas ---------------------------------------------------------------

def montar_politica(a, cfg):
    """`(politica, custo, ops, latencia_fn)` para o modelo pedido."""
    if a.modelo == "maddpg":
        from spec02_baseline_maddpg import checkpoints, operacoes_ator, tamanho
        from spec02_maddpg_rollout import carregar_ator
        disp = {(na, ns): p for na, ns, p in checkpoints()}
        if (a.na, a.ns) not in disp:
            sys.exit(f"sem checkpoint MADDPG para Na={a.na} Ns={a.ns}. "
                     "Lamenza publica Na∈{2,4,8}×Ns∈{12,24,36} — não há Na=1.")
        pol = carregar_ator(disp[(a.na, a.ns)], cfg.observation_size)
        cst = tamanho(disp[(a.na, a.ns)])
        ops = operacoes_ator(cst["obs_dim"])
        return pol, cst, ops, lambda: _latencia_torch(pol, cfg.observation_size)

    from spec06_campanha_wnn import montar
    from uavwnn.custo.memoria import custo_ntuple_de_rede
    from uavwnn.custo.operacoes import operacoes_ntuple
    # `plana=True`: a campanha só INFERE, e a LUT plana custa 8× menos RAM e
    # 3,6× menos tempo no `prever`. Com 6 corridas em paralelo, cada uma com um
    # modelo carregado, a diferença é de 570 MB para 72 MB.
    pol, ap = montar(a.aproximador, cfg, a.checkpoint, plana=True)
    import json as _json
    # `T` e `n` vêm do checkpoint, não de constante: a campanha compara T=16 com
    # T=32, e assumir um dos dois faria o custo do outro sair errado.
    proc = _json.loads((a.checkpoint / "procedencia.json").read_text()) \
        if (a.checkpoint / "procedencia.json").exists() else {}
    d_obs, n_tup, d_saida = cfg.observation_size, ap.ator.tamanho_tupla, 3
    t_res = int(proc.get("resolucao") or _res_do_config(a.checkpoint, d_obs))
    cm = custo_ntuple_de_rede(ap.ator, variante="hash", limiares_por_dim=t_res,
                              n_dims=d_obs, rotulo=f"{a.rotulo} a{a.na}s{a.ns}")
    op = operacoes_ntuple(n_ram=ap.ator.n_ram, tamanho_tupla=n_tup,
                          n_cabecas=d_saida, n_dims=d_obs, n_niveis=t_res)
    # `params_ator` do MADDPG conta pesos; o análogo do WNN é a ENTRADA OCUPADA
    # da LUT — só essas custam memória, e é o número que `spec06-merito.json`
    # reporta. `CustoMemoria` não o expõe, então vem da rede.
    ocupadas = sum(len(m) for m in ap.ator.memoria)
    cst = {"params_ator": int(ocupadas), "bytes_ator_fp32": int(cm.persistente)}
    return pol, cst, {"mult_fp": op.mult_fp}, lambda: _latencia_wnn(pol, cfg)


def _res_do_config(ckpt: Path, d_obs: int) -> int:
    """`T` do checkpoint. Do metadado se houver; senão, DERIVADO do próprio `.npz`.

    O metadado externo é frágil: copiar um checkpoint sem levar o `config.json`
    junto deixa a corrida sem `T` e ela morre na inicialização — aconteceu com
    `models/T32/a1s12`, e o supervisor relançou a mesma corrida quatro vezes.

    Mas `T` não precisa de arquivo externo: o `.npz` do ator grava `n_bits`, e
    `n_bits = d_obs · T` por construção do termômetro. Derivar é mais robusto que
    procurar, e a divisão exata serve de verificação.
    """
    import json as _json
    for cand in (ckpt / "config.json", ckpt.parent.parent / "config.json",
                 ckpt.parent / "config.json"):
        if cand.exists():
            return int(_json.loads(cand.read_text())["config"]["resolucao"])
    z = np.load(next(ckpt.glob("*ator.npz")), allow_pickle=False)
    n_bits = int(_json.loads(str(z["config"]))["n_bits"])
    if n_bits % d_obs:
        raise SystemExit(f"{ckpt}: n_bits={n_bits} não é múltiplo de d_obs={d_obs}; "
                         "não dá para derivar T — restaure o config.json")
    return n_bits // d_obs


def _latencia_torch(politica, obs_dim: int, repeticoes: int = 2000):
    import torch
    rede = politica.rede
    x = torch.randn(1, obs_dim)
    with torch.no_grad():
        for _ in range(200):
            rede(x)
        t = np.empty(repeticoes)
        for i in range(repeticoes):
            t0 = time.perf_counter_ns(); rede(x); t[i] = time.perf_counter_ns() - t0
    us = t / 1000.0
    return float(np.median(us)), float(np.percentile(us, 99))


def _latencia_wnn(politica, cfg, repeticoes: int = 2000):
    """Mesma medida do MADDPG: `repeticoes` forwards sobre uma observação fixa.

    Usa uma observação REAL (não ruído gaussiano): o custo do WNN depende de
    quais endereços são tocados, e um vetor aleatório tocaria endereços vazios,
    medindo o caso mais barato. Ver a vetorização do `prever` — o custo cresce
    com a taxa de ACERTO.
    """
    from uavwnn.refenv import make_env
    env = make_env(cfg); obs, _ = env.reset()
    x = obs[next(iter(obs))]
    for _ in range(200):
        politica(x)
    t = np.empty(repeticoes)
    for i in range(repeticoes):
        t0 = time.perf_counter_ns(); politica(x); t[i] = time.perf_counter_ns() - t0
    try:
        env.close()
    except Exception:  # pragma: no cover
        pass
    us = t / 1000.0
    return float(np.median(us)), float(np.percentile(us, 99))


# --- ambientes ---------------------------------------------------------------

class Ambientes:
    """Cache de ambientes por `(degrau, σ)` — instanciar por episódio dominaria
    o custo, e `criar_ambiente_ruidoso` cria uma CLASSE a cada chamada."""

    def __init__(self, cfg, gate, limpo: bool):
        self.cfg, self.gate, self.limpo, self._c = cfg, gate, limpo, {}

    def _params(self, degrau: int) -> dict:
        if degrau == 2:
            return {"alpha": 0.5}
        if degrau == 3:
            return {"q_a": float(self.gate["degrau3"]["melhor"])}
        if degrau == 4:
            return {"q_w": float(self.gate["degrau4"]["melhor"])}
        return {}

    def obter(self, degrau: int, sigma: float):
        chave = (degrau, sigma)
        if chave not in self._c:
            if self.limpo or degrau == 0:
                from uavwnn.refenv import make_env
                self._c[chave] = make_env(self.cfg)
            else:
                from uavwnn.fusion import criar_ambiente_ruidoso
                from uavwnn.fusion.ruido import ConfigRuido
                self._c[chave] = criar_ambiente_ruidoso(
                    self.cfg, degrau=degrau,
                    config_ruido=ConfigRuido(sigma_gps=sigma),
                    **self._params(degrau))
        return self._c[chave]


def rodar_episodio(amb, cfg, politica, seed: int, handlers):
    from uavwnn.refenv import prune_logging_handlers
    random.seed(int(seed)); np.random.seed(int(seed))
    if hasattr(amb, "indice_ruido"):
        amb.indice_ruido = int(seed)     # CRN: mesma semente → mesma realização
    obs, _ = amb.reset()
    prune_logging_handlers(handlers)
    ct, ok, coleta = float(cfg.max_episode_length), False, float("nan")
    for _ in range(int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2):
        obs, _, term, trunc, infos = amb.step(
            {a: politica(obs[a]) for a in amb.agents})
        if infos and "avg_reward" in next(iter(infos.values())):
            info = next(iter(infos.values()))
            ct = float(info.get("completion_time", cfg.max_episode_length))
            ok = bool(info.get("all_collected", False))
            # `avg_collection_time`, não `collection_times`: o ambiente já
            # devolve a média do episódio. Ler a chave inexistente gravava NaN
            # em toda a coluna e o `nanmean` da run devolvia NaN.
            v = info.get("avg_collection_time")
            if v is not None and float(v) == float(v):
                coleta = float(v)
            break
        if any(term.values()) or any(trunc.values()):
            break
    return ct, ok, coleta


# --- campanha ----------------------------------------------------------------

COLUNAS = ["run", "modelo", "num_drones", "num_sensores", "treinado_em_ns",
           "transferencia", "regime", "episodios",
           "seed_inicial", "completion_time_medio", "completion_time_std_episodios",
           "taxa_sucesso", "coleta_media", "separacao_media_m", "params_ator",
           "bytes_ator_fp32", "mult_fp", "latencia_mediana_us", "latencia_p99_us",
           "wall_s"]
COL_EP = ["run", "modelo", "num_drones", "num_sensores", "seed", "degrau",
          "sigma", "completion_time", "sucesso", "coleta_media"]


def separacao(T, num_drones: int) -> float:
    if num_drones < 2 or not T:
        return float("nan")
    passos = min(len(t) for t in T)
    pares = [np.linalg.norm(np.array(T[i][:passos]) - np.array(T[j][:passos]), axis=1)
             for i in range(num_drones) for j in range(i + 1, num_drones)]
    return float(np.mean(pares)) if pares else float("nan")


def trilhas(amb, cfg, politica, seed, handlers):
    """Posições absolutas por época de decisão, para a separação média."""
    from uavwnn.refenv import prune_logging_handlers
    random.seed(int(seed)); np.random.seed(int(seed))
    if hasattr(amb, "indice_ruido"):
        amb.indice_ruido = int(seed)
    obs, _ = amb.reset()
    prune_logging_handlers(handlers)
    T = [[] for _ in range(cfg.num_drones)]
    for _ in range(int(cfg.max_episode_length / cfg.algorithm_iteration_interval) + 2):
        for k, i in enumerate(amb.agent_node_ids):
            T[k].append(np.array(amb.simulator.get_node(i).position[:2]))
        obs, _, term, trunc, infos = amb.step(
            {a: politica(obs[a]) for a in amb.agents})
        if infos and "avg_reward" in next(iter(infos.values())):
            break
        if any(term.values()) or any(trunc.values()):
            break
    return T


def ler_cache(arq: Path, episodios: int, colunas: list[str]) -> dict[int, dict]:
    """Runs já feitas. Recusa cache com `E` diferente — média sobre runs de
    tamanhos distintos não significa nada — e cache de outro ESQUEMA.

    A checagem de esquema vem antes da de `E` porque é ela que dá mensagem útil:
    lendo um CSV de 18 colunas com o leitor de 20, o `episodios` some e o erro
    vira um `KeyError` sem relação com a causa.
    """
    if not arq.exists() or not arq.stat().st_size:
        return {}
    with arq.open(newline="") as f:
        atual = next(csv.reader(f), [])
    if atual != colunas:
        faltam = [c for c in colunas if c not in atual]
        sys.exit(f"{arq}: cabeçalho é de outro esquema ({len(atual)} colunas "
                 f"contra {len(colunas)}"
                 + (f"; faltam {faltam}" if faltam else "") + ").\n"
                 "Migre o arquivo ou apague-o para refazer as runs.")
    feitas = {}
    with arq.open(newline="") as f:
        for lin in csv.DictReader(f):
            if int(lin["episodios"]) != episodios:
                sys.exit(f"{arq}: cache com episodios={lin['episodios']}, "
                         f"pedido {episodios}. Apague o arquivo ou ajuste.")
            feitas[int(lin["run"])] = lin
    return feitas


def anexar(arq: Path, colunas: list[str], linhas: list[dict]) -> None:
    """Anexa, RECUSANDO escrever sob um cabeçalho de outro esquema.

    Sem esta guarda o arquivo fica misto e ninguém percebe: quando as colunas de
    transferência entraram, a fila retomou CSVs já começados e gravou linhas de
    20 campos sob cabeçalhos de 18. O `csv.DictWriter` não reclama — ele só
    escreve os valores em ordem — e o defeito só apareceu ao tentar reler.
    """
    arq.parent.mkdir(parents=True, exist_ok=True)
    if arq.exists() and arq.stat().st_size:
        with arq.open(newline="") as f:
            atual = next(csv.reader(f), [])
        if atual and atual != colunas:
            faltam = [c for c in colunas if c not in atual]
            sys.exit(
                f"{arq}: cabeçalho é de outro esquema "
                f"({len(atual)} colunas contra {len(colunas)}"
                + (f"; faltam {faltam}" if faltam else "") + ").\n"
                "Anexar aqui produziria um arquivo misto. Migre o arquivo ou "
                "apague-o para refazer as runs.")
    else:
        with arq.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=colunas).writeheader()
    with arq.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=colunas).writerows(linhas)


def main() -> int:
    import logging
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=("maddpg", "wnn"), required=True)
    ap.add_argument("--rotulo", default=None, help="nome no CSV; padrão = --modelo")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--aproximador", default="ntupla_dissertacao")
    ap.add_argument("--na", type=int, required=True)
    ap.add_argument("--ns", type=int, required=True)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--episodios", type=int, default=200)
    ap.add_argument("--ruido", choices=("limpo", "sorteio"), default="limpo")
    ap.add_argument("--sigmas", default="1,3,5,10",
                    help="σ do GPS sorteáveis, em metros")
    ap.add_argument("--degraus", default="1,2,3,4",
                    help="estimadores sorteáveis. O degrau 0 (limpo) fica de fora "
                         "por padrão: o regime `limpo` já o mede, com os 20 mil "
                         "episódios inteiros em vez de uma fração do sorteio.")
    ap.add_argument("--gate", type=Path, default=None,
                    help="atestado RF-05 DESTE modelo nesta célula (degraus 3 e 4)")
    ap.add_argument("--treinado-em-ns", type=int, default=None,
                    help="Ns em que a rede foi TREINADA, se diferente de --ns. "
                         "Marca a corrida como transferência no CSV.")
    ap.add_argument("--dir-saida", type=Path, default=Path("reports/campanha"))
    a = ap.parse_args()

    a.rotulo = a.rotulo or a.modelo
    if a.modelo == "wnn" and a.checkpoint is None:
        sys.exit("--checkpoint é obrigatório para --modelo wnn")
    gate = {}
    if a.ruido == "sorteio":
        if a.gate is None or not a.gate.exists():
            sys.exit("--gate é obrigatório com --ruido sorteio: os degraus 3 e 4 "
                     "usam q_a/q_w do atestado RF-05 DESTE modelo nesta célula.")
        import json
        gate = json.loads(a.gate.read_text())
        if gate.get("status") != "aprovado":
            sys.exit(f"atestado reprovado ({gate.get('status')!r}) — ver spec08_gate_rf05.py")

    from uavwnn.refenv import LAMENZA_003
    cfg = LAMENZA_003(a.na, a.ns, phase="eval")
    politica, custo, ops, latencia = montar_politica(a, cfg)

    # `d_obs` depende só de Na, NÃO de Ns — então uma rede treinada em Ns=24 roda
    # em Ns=12/36 sem retreino. Mas isso é TRANSFERÊNCIA, e o MADDPG tem
    # checkpoint próprio para cada uma das 9 células (joga em casa nas três).
    # Sem esta coluna a assimetria some no CSV e a tabela final mentiria.
    treinado_ns = a.treinado_em_ns if a.treinado_em_ns is not None else a.ns
    transf = treinado_ns != a.ns
    if transf:
        print(f"  TRANSFERÊNCIA: rede treinada em Ns={treinado_ns}, avaliada em "
              f"Ns={a.ns} (d_obs={cfg.observation_size} é o mesmo)", flush=True)
    ambientes = Ambientes(cfg, gate, limpo=(a.ruido == "limpo"))
    handlers = list(logging.getLogger().handlers)

    degraus = [int(x) for x in a.degraus.split(",")]
    sigmas = [float(x) for x in a.sigmas.split(",")]
    escada = montar_escada(degraus, sigmas)
    # O rótulo carrega a escada inteira: sortear de {σ=10} e de {σ=1,3,5,10}, ou
    # fixar o degrau 1 em vez do 4, dá campanhas diferentes — escrevê-las no
    # mesmo CSV misturaria regimes sem aviso.
    regime = a.ruido if a.ruido == "limpo" else _rotulo_regime(degraus, sigmas)
    if a.ruido == "sorteio":
        if len(escada) == 1:
            d, s = escada[0]
            print(f"  regime FIXO: degrau {d} ({NOME_DEGRAU[d]}) · σ={s:g} — "
                  f"todos os {a.runs * a.episodios:,} episódios", flush=True)
        else:
            print(f"  escada sorteável: {len(escada)} condições "
                  f"(degraus {degraus} × σ {[f'{s:g}' for s in sigmas]})", flush=True)
    arq = a.dir_saida / f"{a.rotulo}-a{a.na}s{a.ns}-{regime}.csv"
    arq_ep = a.dir_saida / f"{a.rotulo}-a{a.na}s{a.ns}-{regime}-episodios.csv"
    feitas = ler_cache(arq, a.episodios, COLUNAS)
    faltam = [r for r in range(a.runs) if r not in feitas]
    print(f"{a.rotulo} · Na={a.na} Ns={a.ns} · regime {regime} · "
          f"{len(faltam)} de {a.runs} runs a rodar ({a.episodios} ep cada)", flush=True)
    if not faltam:
        return 0

    t_ini = time.perf_counter()
    for j, r in enumerate(faltam, 1):
        s0 = r * a.episodios
        t0 = time.perf_counter()
        cts, oks, cols, eps = [], [], [], []
        for s in range(s0, s0 + a.episodios):
            degrau, sigma = ((0, 0.0) if a.ruido == "limpo"
                             else condicao_do_episodio(s, escada))
            amb = ambientes.obter(degrau, sigma)
            ct, ok, col = rodar_episodio(amb, cfg, politica, s, handlers)
            cts.append(ct); oks.append(ok); cols.append(col)
            eps.append({"run": r, "modelo": a.rotulo, "num_drones": a.na,
                        "num_sensores": a.ns, "seed": s, "degrau": degrau,
                        "sigma": sigma, "completion_time": round(ct, 4),
                        "sucesso": int(ok), "coleta_media": round(col, 4)
                        if col == col else ""})
        ct = np.array(cts, float); ok = np.array(oks, bool)
        d0, s0d = ((0, 0.0) if a.ruido == "limpo"
                   else condicao_do_episodio(s0, escada))
        T = trilhas(ambientes.obter(d0, s0d), cfg, politica, s0, handlers)
        lat_med, lat_p99 = latencia()
        linha = {
            "run": r, "modelo": a.rotulo, "num_drones": a.na, "num_sensores": a.ns,
            "treinado_em_ns": treinado_ns, "transferencia": int(transf),
            "regime": regime, "episodios": a.episodios, "seed_inicial": r * a.episodios,
            "completion_time_medio": round(float(ct.mean()), 4),
            "completion_time_std_episodios": round(float(ct.std(ddof=1)), 4),
            "taxa_sucesso": round(float(ok.mean()), 6),
            "coleta_media": round(float(np.nanmean(cols)), 4),
            "separacao_media_m": round(separacao(T, a.na), 3),
            "params_ator": custo["params_ator"],
            "bytes_ator_fp32": custo["bytes_ator_fp32"],
            "mult_fp": ops["mult_fp"],
            "latencia_mediana_us": round(lat_med, 3),
            "latencia_p99_us": round(lat_p99, 3),
            "wall_s": round(time.perf_counter() - t0, 2)}
        anexar(arq, COLUNAS, [linha])
        anexar(arq_ep, COL_EP, eps)
        decorrido = time.perf_counter() - t_ini
        resta = decorrido / j * (len(faltam) - j)
        print(f"  [{j:>3}/{len(faltam)}] run {r:>3} · CT {linha['completion_time_medio']:7.2f} s "
              f"· SR {linha['taxa_sucesso']:.1%} · {linha['wall_s']:6.1f} s "
              f"· resta {resta/3600:4.1f} h", flush=True)
    print(f"\n-> {arq}\n-> {arq_ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
