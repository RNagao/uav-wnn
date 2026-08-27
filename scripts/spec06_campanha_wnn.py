"""Campanha das políticas WNN — figuras de mérito e trajetórias, como no MADDPG.

Espelha `spec02_maddpg_campanha.py`: `R` runs por célula, CSV como cache, média ±
desvio **sobre as runs**, e as **mesmas colunas**, para as tabelas do relatório
serem comparadas linha a linha sem conversão.

    uv run python scripts/spec06_campanha_wnn.py --aproximador ntupla_dissertacao \
        --checkpoint reports/ckpt/ntupla_dissertacao-a4s24 --runs 1000

TRÊS APROXIMADORES, UM LAÇO

    mlp                MLP densa `D→32→32→3`, NumPy puro — controle do MÉTODO
    ntupla_dissertacao a nossa: gradiente funcional (§4.3.1) + NTRN com
                       esquecimento (§4.2.1); convenção de média
    ntupla_anbubus   portada do `NTNModel` de `anbubus/ntn_neurocomputing_v0`;
                       gradiente funcional nos dois lados, convenção de soma

O baseline do **problema** é outro e não sai daqui: o MADDPG de Lamenza, medido dos
pesos publicados pela SPEC-02. Estes três são variações do **método**, e cada um é
um braço da comparação contra ele.

O AVISO QUE ESTE SCRIPT EXISTE PARA NÃO DEIXAR PASSAR

**Uma política não treinada produz números que se parecem com resultados.** O
MADDPG tem pesos publicados; as nossas redes começam vazias, e uma rede n-tuple
vazia prediz `f = 0` em todo estado — o que a normal projetada mapeia num rumo
constante. Isso rende tempo de conclusão, taxa de sucesso e separação média: uma
linha de CSV completa, plausível e **sem significado**.

Por isso `--checkpoint` é obrigatório, e rodar sem ele exige
`--sem-treino-proposital`, que marca cada linha do CSV com `treinado=False`. O
treino no ambiente real é a **SPEC-06**; este script é o arnês de avaliação dela.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec00_demo import salvar_grafico_lamenza  # noqa: E402
from spec02_maddpg_campanha import (  # noqa: E402
    CENARIOS_LAMENZA,
    EPISODIOS_LAMENZA,
    analisar_celulas,
    separacao,
)
from spec02_maddpg_rollout import trilhas_de_um_episodio  # noqa: E402
from uavwnn.agents import APROXIMADORES, AtorCriticoContinuo  # noqa: E402
from uavwnn.agents.politica_continua import NormalProjetada  # noqa: E402
from uavwnn.agents.base import Acao, ContextoEpisodio, Observacao  # noqa: E402
from uavwnn.encoding import TermometroQuantil  # noqa: E402
from uavwnn.refenv import LAMENZA_003, run_episode  # noqa: E402

COLUNAS = [
    "run", "aproximador", "treinado", "num_drones", "num_sensores", "episodios",
    "seed_inicial",
    # figuras de mérito de desempenho — MESMOS nomes do CSV do MADDPG
    "completion_time_medio", "completion_time_std_episodios", "taxa_sucesso",
    "coleta_media", "separacao_media_m",
    # figuras de mérito de custo
    "params_ator", "bytes_ator_fp32", "mult_fp",
    "latencia_mediana_us", "latencia_p99_us",
    "wall_s",
]
"""`params_ator`, `bytes_ator_fp32` e `mult_fp` mantêm os nomes do MADDPG **de
propósito**, ainda que para a n-tupla "parâmetros" signifique posições ocupadas e
`mult_fp` seja zero. Renomear quebraria o `join` das tabelas; a diferença de
semântica está no relatório, onde ela pode ser explicada, e não no cabeçalho."""

BITS_POR_ESCALAR = 32
"""Resolução do termômetro. `T = 32` é o ponto que a SPEC-01 escolheu."""


def caminho_csv(dir_saida: Path, aprox: str, na: int, ns: int) -> Path:
    return dir_saida / f"campanha-{aprox}-a{na}s{ns}.csv"


def ler_cache(arquivo: Path, episodios: int) -> dict[int, dict]:
    """Runs já feitas, indexadas por `run`. Ignora as de outro `episodios`."""
    if not arquivo.exists():
        return {}
    with open(arquivo, newline="") as f:
        return {int(l["run"]): l for l in csv.DictReader(f)
                if int(l["episodios"]) == episodios}


def escrever_linha(arquivo: Path, linha: dict) -> None:
    novo = not arquivo.exists()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with open(arquivo, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        if novo:
            w.writeheader()
        w.writerow(linha)


# ---------------------------------------------------------------------------


def _termometro(cfg, checkpoint: Path | None, seed: int) -> TermometroQuantil:
    """O termômetro do checkpoint, ou ajustado num corpus de política aleatória.

    **Os limiares são estado treinado e viajam com o modelo.** Reajustá-los num
    corpus diferente daria outra binarização para a mesma observação, e as
    posições de memória passariam a ser endereçadas por outros bits — erro
    silencioso, que apareceria como "a política não aprendeu".

    Sem checkpoint, ajusta num corpus colhido com caminhada aleatória. É o que a
    SPEC-01 fez para escolher o E2, e serve para exercitar o arnês; **não** é o
    termômetro de um modelo treinado.
    """
    if checkpoint is not None:
        return TermometroQuantil.carregar(Path(checkpoint) / "termometro.npz")

    from uavwnn.agents import CaminhadaAleatoria
    corpus: list[np.ndarray] = []
    aleat = CaminhadaAleatoria()
    ctx = ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval)
    aleat.reset(ctx, np.random.default_rng(seed))

    def coletar(observacao):
        corpus.append(np.asarray(observacao, float).ravel())
        return aleat.act(Observacao.decodificar(observacao, ctx)).como_array()

    for s in range(3):
        run_episode(cfg, coletar, seed=10_000 + s)
    return TermometroQuantil(BITS_POR_ESCALAR).ajustar(np.array(corpus))


def _procedencia(checkpoint: Path | None) -> dict:
    """Lê o `procedencia.json` do checkpoint, se houver."""
    if checkpoint is None:
        return {}
    arq = Path(checkpoint) / "procedencia.json"
    return json.loads(arq.read_text()) if arq.exists() else {}


def montar(aprox: str, cfg, checkpoint: Path | None, seed: int = 0,
           plana: bool = False):
    """Devolve `(callable no protocolo do refenv, aproximador)`.

    O codificador vive **fora** do aproximador, como a §4.8 exige: os três recebem
    a mesma observação e cada um a prepara à sua maneira. É a única assimetria
    permitida entre os braços.
    """
    d_obs = cfg.observation_size
    ctx = ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval)

    proc = _procedencia(checkpoint)
    # `d_saida` VEM DO CHECKPOINT quando há um. Assumir 3 sobre um modelo treinado
    # pelo PPO do RLlib montaria a rede com a forma errada — e, pior, a
    # `NormalProjetada` interpretaria as 4 saídas dele como `(u₁, u₂, μ_s)`,
    # produzindo ações plausíveis e sem relação com o que foi treinado.
    d_saida = int(proc.get("ator", {}).get("n_saidas", 3))

    if aprox == "mlp":
        ap = APROXIMADORES[aprox](n_bits_ou_dim=d_obs, seed=seed, d_saida=d_saida)
        codificar = lambda v: np.asarray(v, float).ravel()      # noqa: E731
    else:
        termo = _termometro(cfg, checkpoint, seed)
        ap = APROXIMADORES[aprox](n_bits=termo.n_bits, seed=seed,
                                  d_saida=d_saida,
                                  n_dims_codificador=d_obs,
                                  limiares_por_dim=proc.get("resolucao",
                                                            BITS_POR_ESCALAR))
        codificar = lambda v: termo.codificar(np.asarray(v, float)[None, :])[0]  # noqa: E731

    if checkpoint is not None:
        # `plana` só vale para a família da dissertação; as outras ignoram, e
        # passar a bandeira adiante sem checar daria TypeError num braço só.
        if plana and aprox == "ntupla_dissertacao":
            ap.carregar(checkpoint, plana=True)
        else:
            ap.carregar(checkpoint)

    if d_saida == 4:
        # CHECKPOINT DO PPO/RLLIB. As 4 saídas são `[μ_θ, μ_s, logσ_θ, logσ_s]` da
        # `DiagGaussian`, que é uma gaussiana diagonal **direta sobre a ação** com
        # recorte — isto é, o **C0** da SPEC-05 §4.3.
        #
        # Avaliá-lo com a `NormalProjetada` seria erro grave e silencioso: ela
        # leria `(u₁, u₂, μ_s)` das três primeiras componentes e derivaria o rumo
        # por `atan2`, produzindo ações no domínio certo e **sem relação** com o
        # que o PPO treinou.
        #
        # Vale registrar o que isso implica: o braço PPO herda o modo de falha que
        # o portão da SPEC-05 mediu — o C0 perdeu do C2 por 13,45°, com IC a 12
        # MDEs de zero. Não é defeito da nossa ponte; é o que a `DiagGaussian` do
        # RLlib faz com uma ação circular, e é resultado a reportar.
        def acao_de(f):
            return Acao(theta_norm=float(np.clip(f[0], 0.0, 1.0)),
                        s_norm=float(np.clip(f[1], 0.0, 1.0)))
    else:
        politica_np = AtorCriticoContinuo(
            aproximador=ap, codificar=codificar, id=f"wnn_{aprox}",
            treinando=False)
        politica_np.reset(ctx, np.random.default_rng(seed))
        acao_de = None

    # VIA RÁPIDA para a LUT plana em avaliação: consulta fundida (termômetro +
    # endereçamento + busca numa closure) e projeção sem intermediários. Bit a
    # bit igual à via lenta — pinado em tests/test_caminho_rapido.py — e ~3×
    # mais barata: a decisão saía a ~30 despachos NumPy, e para vetores deste
    # tamanho o despacho domina o trabalho. Só existe fora do treino: o treino
    # precisa do `z` bruto e da trilha que o `act` guarda.
    consulta = None
    if (aprox == "ntupla_dissertacao" and acao_de is None
            and not politica_np.treinando
            and type(ap.ator).__name__ == "LUTPlana"):
        consulta = ap.ator.consulta_do_vetor(
            np.repeat(np.arange(d_obs), termo.n_niveis),
            termo.limiares.ravel())

    def callable_refenv(observacao):
        """Adapta ao protocolo do `refenv`: `obs -> ação em [0,1]²`.

        A ponte é `Observacao.decodificar`, a mesma que o harness usa — então a
        política vê exatamente o que veria lá. (A via rápida pula a ponte porque
        o ator só lê `.vetor`; a igualdade das ações é pinada em teste.)
        """
        if acao_de is not None:
            return acao_de(ap.f(codificar(np.asarray(observacao, float)))).como_array()
        if consulta is not None:
            f = consulta(np.asarray(observacao, float).ravel())
            politica_np._concentracoes.append(NormalProjetada.concentracao(f))
            return NormalProjetada.media_rapida(f)
        obs = Observacao.decodificar(observacao, ctx)
        return politica_np.act(obs).como_array()

    return callable_refenv, ap


def medir_latencia(politica, obs_dim: int, repeticoes: int = 2000) -> tuple[float, float]:
    """Mediana e p99 da decisão completa, na mesma run.

    Difere do `medir_latencia` do MADDPG por medir **o callable inteiro** —
    decodificação da observação, codificador, aproximador e amostrador — e não só
    o `forward` da rede. Não é descuido: a nossa alegação de custo é sobre o
    caminho de decisão, e a SPEC-05 §4.8 já mediu que **o codificador domina** a
    contagem de operações da WNN. Medir só o aproximador infaria a vantagem.

    Aquele mede `rede(x)` porque do lado do MADDPG o `forward` da MLP *é* o
    caminho todo — não há binarização. A assimetria de escopo é, portanto, a favor
    do baseline, e a comparação continua honesta.
    """
    rng = np.random.default_rng(0)
    obs = rng.uniform(0.3, 0.7, obs_dim)
    for _ in range(200):
        politica(obs)
    t = np.empty(repeticoes)
    for i in range(repeticoes):
        t0 = time.perf_counter_ns()
        politica(obs)
        t[i] = time.perf_counter_ns() - t0
    us = t / 1000.0
    return float(np.median(us)), float(np.percentile(us, 99))


def figuras_de_custo(ap) -> dict:
    """Traduz o `custo()` de qualquer família para as colunas do MADDPG."""
    c = ap.custo()
    if c["familia"] == "mlp":
        return {"params_ator": c["params_ator"],
                "bytes_ator_fp32": c["bytes_persistente"],
                "mult_fp": c["mult_fp_aproximador"]}
    # Para a n-tupla, "parâmetros" é posições ocupadas — ver a nota em COLUNAS.
    return {"params_ator": c["ocupadas_ator"],
            "bytes_ator_fp32": c["bytes_persistente"],
            "mult_fp": c["mult_fp_aproximador"]}


def rodar_uma(cfg, aprox: str, checkpoint, treinado: bool, r: int,
              episodios: int) -> dict:
    """Uma run: `E` episódios com sementes `[r·E, (r+1)·E)`. Igual ao MADDPG."""
    s0 = r * episodios
    t0 = time.perf_counter()
    politica, ap = montar(aprox, cfg, checkpoint, seed=r)

    resumos = [run_episode(cfg, politica, seed=s) for s in range(s0, s0 + episodios)]
    ct = np.array([x.completion_time for x in resumos], float)
    ok = np.array([x.all_collected for x in resumos], bool)
    coleta = np.array([np.mean(x.collection_times) if x.collection_times else np.nan
                       for x in resumos], float)

    T, _, _, _ = trilhas_de_um_episodio(cfg, politica, seed=s0)
    lat_med, lat_p99 = medir_latencia(politica, cfg.observation_size)

    return {
        "run": r, "aproximador": aprox, "treinado": treinado,
        "num_drones": cfg.num_drones, "num_sensores": cfg.num_sensors,
        "episodios": episodios, "seed_inicial": s0,
        "completion_time_medio": round(float(ct.mean()), 4),
        "completion_time_std_episodios": round(float(ct.std(ddof=1)), 4),
        "taxa_sucesso": round(float(ok.mean()), 6),
        "coleta_media": round(float(np.nanmean(coleta)), 4),
        "separacao_media_m": round(separacao(T, cfg.num_drones), 3),
        **figuras_de_custo(ap),
        "latencia_mediana_us": round(lat_med, 3),
        "latencia_p99_us": round(lat_p99, 3),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


NUMERICAS = ["completion_time_medio", "taxa_sucesso", "coleta_media",
             "separacao_media_m", "latencia_mediana_us", "latencia_p99_us"]


def resumir(linhas: list[dict]) -> dict[str, tuple[float, float]]:
    """Média e desvio **sobre as runs** — não sobre os episódios.

    A distinção importa: o desvio entre episódios mede variabilidade do cenário; o
    desvio entre runs mede variabilidade do **método**, que é o que uma barra de
    erro comparativa deve mostrar.
    """
    out = {}
    for col in NUMERICAS:
        v = np.array([float(l[col]) for l in linhas], float)
        out[col] = (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0)
    return out


def campanha(aprox: str, cenarios, runs: int, episodios: int, checkpoint,
             treinado: bool, dir_saida: Path, dir_figuras: Path | None,
             so_resumo: bool, verboso: bool = True) -> dict:
    """Roda (ou lê do cache) e devolve o resumo por célula."""
    resultado: dict = {}
    for na, ns in cenarios:
        cfg = LAMENZA_003(num_drones=na, num_sensors=ns)
        arquivo = caminho_csv(dir_saida, aprox, na, ns)
        cache = ler_cache(arquivo, episodios)
        if verboso:
            print(f"\n[{aprox}] Na={na} Ns={ns} — {len(cache)}/{runs} em cache")

        for r in range(runs):
            if r in cache or so_resumo:
                continue
            linha = rodar_uma(cfg, aprox, checkpoint, treinado, r, episodios)
            escrever_linha(arquivo, linha)
            cache[r] = linha
            if verboso and (r + 1) % 10 == 0:
                print(f"  run {r + 1}/{runs}  "
                      f"t_conc {linha['completion_time_medio']:.1f} s  "
                      f"sucesso {float(linha['taxa_sucesso']):.1%}  "
                      f"({linha['wall_s']:.0f} s)")

        linhas = [cache[r] for r in sorted(cache) if r < runs]
        if not linhas:
            continue
        resultado[f"a_{na}-s_{ns}"] = {
            "runs": len(linhas), "treinado": treinado, **{
                k: {"media": m, "desvio": d} for k, (m, d) in resumir(linhas).items()}
        }

        if dir_figuras is not None:
            politica, _ = montar(aprox, cfg, checkpoint, seed=0)
            T, sensores, _, _ = trilhas_de_um_episodio(cfg, politica, seed=0)
            dir_figuras.mkdir(parents=True, exist_ok=True)
            arq = dir_figuras / f"trajetoria-{aprox}-{na}a{ns}s.png"
            classe = type("Args", (), {"agents": na, "sensors": ns, "seed": 0})
            salvar_grafico_lamenza(str(arq), T, sensores, cfg, classe)
            resultado[f"a_{na}-s_{ns}"]["figura"] = str(arq)
            if verboso:
                print(f"  figura: {arq}")
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aproximador", choices=sorted(APROXIMADORES), required=True)
    ap.add_argument("--runs", type=int, default=100,
                    help="runs por célula. 1000 é viável com o cache do CSV")
    ap.add_argument("--episodios", type=int, default=EPISODIOS_LAMENZA,
                    help="episódios por run (padrão: o `total_runs` de Lamenza)")
    ap.add_argument("--celulas", default=None,
                    help="ex. '4:24,8:36'. Padrão: as 9 células de Lamenza")
    ap.add_argument("--checkpoint", default=None,
                    help="diretório com o modelo treinado (obrigatório, salvo "
                         "--sem-treino-proposital)")
    ap.add_argument("--sem-treino-proposital", action="store_true",
                    help="roda com rede VAZIA. As linhas saem com treinado=False; "
                         "serve para exercitar o arnês, não para reportar")
    ap.add_argument("--dir-saida", default=None)
    ap.add_argument("--dir-figuras", default=None)
    ap.add_argument("--sem-figuras", action="store_true")
    ap.add_argument("--so-resumo", action="store_true")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()

    if a.checkpoint is None and not a.sem_treino_proposital:
        sys.exit(
            "ERRO: --checkpoint é obrigatório.\n\n"
            "Uma política não treinada produz uma linha de CSV completa,\n"
            "plausível e sem significado: a rede vazia prediz f = 0 em todo\n"
            "estado, o que a normal projetada mapeia num rumo constante — e\n"
            "isso rende tempo de conclusão, taxa de sucesso e separação média.\n\n"
            "O treino no ambiente real é a SPEC-06; este script é o arnês de\n"
            "avaliação dela. Para exercitar o arnês mesmo assim, passe\n"
            "--sem-treino-proposital (as linhas saem com treinado=False)."
        )
    treinado = a.checkpoint is not None
    ckpt = Path(a.checkpoint) if treinado else None
    if treinado and not ckpt.exists():
        sys.exit(f"ERRO: checkpoint não encontrado: {ckpt}")

    cenarios = analisar_celulas(a.celulas) if a.celulas else CENARIOS_LAMENZA
    dir_saida = Path(a.dir_saida or f"reports/wnn/{a.aproximador}")
    dir_fig = None if a.sem_figuras else Path(
        a.dir_figuras or f"reports/img/wnn/{a.aproximador}")
    saida = Path(a.saida or f"reports/spec06-campanha-{a.aproximador}.json")

    print(f"campanha · {a.aproximador} · {len(cenarios)} células × {a.runs} runs "
          f"× {a.episodios} episódios")
    print(f"modelo: {'treinado — ' + str(ckpt) if treinado else 'VAZIO (treinado=False)'}")

    res = campanha(a.aproximador, cenarios, a.runs, a.episodios, ckpt, treinado,
                   dir_saida, dir_fig, a.so_resumo)

    print(f"\n{'célula':>10} {'runs':>5} {'t_conc (s)':>18} {'sucesso':>16} "
          f"{'separação (m)':>16}")
    for cel, d in res.items():
        t, s, sep = (d["completion_time_medio"], d["taxa_sucesso"],
                     d["separacao_media_m"])
        print(f"{cel:>10} {d['runs']:>5} {t['media']:>9.2f} ± {t['desvio']:<6.2f} "
              f"{s['media']:>7.1%} ± {s['desvio']:<6.1%} "
              f"{sep['media']:>7.1f} ± {sep['desvio']:<6.1f}")

    artefato = {"aproximador": a.aproximador, "config": vars(a),
                "treinado": treinado, "celulas": res}
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(artefato, indent=2, ensure_ascii=False))
    print(f"\nartefato: {saida}")
    if not treinado:
        print("\nAVISO: treinado=False. Estes números NÃO são resultado —\n"
              "são o arnês funcionando com uma rede vazia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
