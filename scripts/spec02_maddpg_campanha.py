"""Campanha do baseline MADDPG — `R` runs por cenário, com CSV como cache.

ESTRUTURA, e ela é o ponto do script

    1 episódio  →  um tempo de conclusão
    1 RUN       →  `E` episódios, e o que se guarda é a MÉDIA deles
    1 campanha  →  `R` runs, e o que se reporta é média ± desvio SOBRE AS RUNS

`E = 200` por omissão, que é o `total_runs` do `test.py` de Lamenza. Cada run é
portanto uma **réplica da campanha de avaliação dele**, e o desvio entre runs
mede a variabilidade do número que ele publica — não a variabilidade entre
episódios, que é muito maior e não é o que a figura reporta.

Confundir os dois é fácil e muda a conclusão: o desvio entre episódios diz
"quanto um episódio varia"; o desvio entre runs diz "quanto o valor publicado
variaria se ele tivesse rodado outra vez". Para comparar com a nossa política é
o segundo que importa.

SEMEADURA SEM SOBREPOSIÇÃO

    run `r` usa as sementes `[r·E, (r+1)·E)`

Então runs diferentes veem cenários diferentes, e a média de cada uma é uma
estimativa independente. Reusar o mesmo bloco de sementes daria `R` cópias do
mesmo número e um desvio artificialmente nulo.

CACHE

Um CSV por cenário, `reports/maddpg/campanha-a{Na}s{Ns}.csv`, uma linha por run.
Ao reiniciar, as runs já presentes são lidas e puladas; só as que faltam rodam. O
CSV guarda `episodios`, e o script **recusa** reaproveitar um cache com `E`
diferente do pedido — média sobre runs de tamanhos distintos não significa nada.

    uv run --group baseline python scripts/spec02_maddpg_campanha.py
    uv run --group baseline python scripts/spec02_maddpg_campanha.py --runs 10
    uv run --group baseline python scripts/spec02_maddpg_campanha.py --celulas 4:24
    uv run --group baseline python scripts/spec02_maddpg_campanha.py --so-resumo

CUSTO: ~1 h 50 para 100 runs × 200 episódios × 9 células num núcleo. Rode em
segundo plano; o cache torna a interrupção barata.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from spec02_baseline_maddpg import checkpoints, operacoes_ator, tamanho  # noqa: E402
from spec02_maddpg_rollout import carregar_ator, trilhas_de_um_episodio  # noqa: E402
from uavwnn.refenv import LAMENZA_003, run_episode  # noqa: E402

# `total_runs` do `test.py` do submodule.
EPISODIOS_LAMENZA = 200

COLUNAS = [
    "run", "num_drones", "num_sensores", "episodios", "seed_inicial",
    # figuras de mérito de desempenho
    "completion_time_medio", "completion_time_std_episodios", "taxa_sucesso",
    "coleta_media", "separacao_media_m",
    # figuras de mérito de custo
    "params_ator", "bytes_ator_fp32", "mult_fp",
    "latencia_mediana_us", "latencia_p99_us",
    # instrumentação
    "wall_s",
]


def caminho_csv(dir_saida: Path, na: int, ns: int) -> Path:
    return dir_saida / f"campanha-a{na}s{ns}.csv"


def ler_cache(arquivo: Path, episodios: int) -> dict[int, dict]:
    """Runs já feitas, por índice. Recusa cache com `E` diferente.

    Média sobre runs de tamanhos distintos não significa nada — melhor falhar
    alto que produzir um número silenciosamente incomparável.
    """
    if not arquivo.exists():
        return {}
    feitas: dict[int, dict] = {}
    with open(arquivo, newline="") as f:
        for linha in csv.DictReader(f):
            e = int(linha["episodios"])
            if e != episodios:
                sys.exit(
                    f"{arquivo} tem runs de {e} episódios, mas foram pedidos "
                    f"{episodios}.\nApague o arquivo ou use --episodios {e}."
                )
            feitas[int(linha["run"])] = linha
    return feitas


def escrever_linha(arquivo: Path, linha: dict) -> None:
    """Append com cabeçalho na criação. Uma linha por run, gravada na hora —
    é o que torna a interrupção barata."""
    novo = not arquivo.exists()
    with open(arquivo, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        if novo:
            w.writeheader()
        w.writerow(linha)


def medir_latencia(politica, obs_dim: int, repeticoes: int = 2000) -> tuple[float, float]:
    """Mediana e p99 do forward, na mesma run. Medir por run em vez de uma vez
    dá a distribuição do número, que é o que a campanha existe para produzir."""
    import torch

    rede = politica.rede
    x = torch.randn(1, obs_dim)
    with torch.no_grad():
        for _ in range(200):
            rede(x)
        t = np.empty(repeticoes)
        for i in range(repeticoes):
            t0 = time.perf_counter_ns()
            rede(x)
            t[i] = time.perf_counter_ns() - t0
    us = t / 1000.0
    return float(np.median(us)), float(np.percentile(us, 99))


def separacao(T: np.ndarray, num_drones: int) -> float:
    """Separação média entre pares de UAVs — o número que mede coordenação.

    O `greedy` mediu ~0,1 m contra 20 m de alcance (SPEC-00): voa em formação.
    """
    if num_drones < 2:
        return float("nan")
    passos = min(len(t) for t in T)
    pares = [np.linalg.norm(T[i][:passos] - T[j][:passos], axis=1)
             for i in range(num_drones) for j in range(i + 1, num_drones)]
    return float(np.mean(pares))


def rodar_uma(cfg, politica, custo, ops, r: int, episodios: int) -> dict:
    """Uma run: `E` episódios com sementes `[r·E, (r+1)·E)`."""
    s0 = r * episodios
    t0 = time.perf_counter()

    resumos = [run_episode(cfg, politica, seed=s)
               for s in range(s0, s0 + episodios)]
    ct = np.array([x.completion_time for x in resumos], float)
    ok = np.array([x.all_collected for x in resumos], bool)
    coleta = np.array([np.mean(x.collection_times) if x.collection_times else np.nan
                       for x in resumos], float)

    # Um episódio a mais, com trilhas, só para a separação. ~0,5 % de custo.
    T, _, _, _ = trilhas_de_um_episodio(cfg, politica, seed=s0)
    lat_med, lat_p99 = medir_latencia(politica, cfg.observation_size)

    return {
        "run": r,
        "num_drones": cfg.num_drones,
        "num_sensores": cfg.num_sensors,
        "episodios": episodios,
        "seed_inicial": s0,
        "completion_time_medio": round(float(ct.mean()), 4),
        "completion_time_std_episodios": round(float(ct.std(ddof=1)), 4),
        "taxa_sucesso": round(float(ok.mean()), 6),
        "coleta_media": round(float(np.nanmean(coleta)), 4),
        "separacao_media_m": round(separacao(T, cfg.num_drones), 3),
        "params_ator": custo["params_ator"],
        "bytes_ator_fp32": custo["bytes_ator_fp32"],
        "mult_fp": ops["mult_fp"],
        "latencia_mediana_us": round(lat_med, 3),
        "latencia_p99_us": round(lat_p99, 3),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


NUMERICAS = ["completion_time_medio", "taxa_sucesso", "coleta_media",
             "separacao_media_m", "latencia_mediana_us", "latencia_p99_us"]


def resumir(linhas: list[dict]) -> dict[str, tuple[float, float]]:
    """Média e desvio **sobre as runs**, por figura de mérito."""
    out = {}
    for col in NUMERICAS:
        v = np.array([float(x[col]) for x in linhas], float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        out[col] = (float(v.mean()),
                    float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--episodios", type=int, default=EPISODIOS_LAMENZA,
                    help=f"por run; {EPISODIOS_LAMENZA} é o do test.py de Lamenza")
    ap.add_argument("--celulas", default="", help="ex. '4:24,8:36'; vazio = as 9")
    ap.add_argument("--dir-saida", default="reports/maddpg")
    ap.add_argument("--so-resumo", action="store_true",
                    help="não roda nada; só resume os CSVs existentes")
    a = ap.parse_args()

    disponiveis = {(na, ns): p for na, ns, p in checkpoints()}
    if not disponiveis and not a.so_resumo:
        sys.exit("nenhum checkpoint; git submodule update --init --recursive")

    if a.celulas:
        pedidas = []
        for item in a.celulas.split(","):
            na, ns = (int(x) for x in item.split(":"))
            pedidas.append((na, ns))
    else:
        pedidas = sorted(disponiveis) or sorted(
            (int(p.stem.split("-")[0][2:]), int(p.stem.split("-")[1][1:]))
            for p in Path(a.dir_saida).glob("campanha-a*s*.csv")
        )

    dir_saida = Path(a.dir_saida)
    dir_saida.mkdir(parents=True, exist_ok=True)
    resumos: dict[tuple[int, int], dict] = {}

    for na, ns in pedidas:
        arquivo = caminho_csv(dir_saida, na, ns)
        feitas = ler_cache(arquivo, a.episodios)
        faltam = [r for r in range(a.runs) if r not in feitas]

        print(f"\n=== a_{na}-s_{ns} · {a.runs} runs × {a.episodios} episódios ===")
        print(f"  cache: {len(feitas)} runs em {arquivo.name}"
              f"{'' if not faltam else f'; faltam {len(faltam)}'}")

        if faltam and not a.so_resumo:
            if (na, ns) not in disponiveis:
                print(f"  SEM CHECKPOINT para a_{na}-s_{ns}; pulando")
                continue
            cfg = LAMENZA_003(na, ns, phase="eval")
            politica = carregar_ator(disponiveis[(na, ns)], cfg.observation_size)
            custo = tamanho(disponiveis[(na, ns)])
            ops = operacoes_ator(custo["obs_dim"])

            t0 = time.perf_counter()
            for i, r in enumerate(faltam, 1):
                linha = rodar_uma(cfg, politica, custo, ops, r, a.episodios)
                escrever_linha(arquivo, linha)
                feitas[r] = {k: str(v) for k, v in linha.items()}
                if i == 1 or i % 10 == 0 or i == len(faltam):
                    decorrido = time.perf_counter() - t0
                    resta = decorrido / i * (len(faltam) - i)
                    print(f"    run {r:>3} · {linha['completion_time_medio']:>6.2f}s "
                          f"· {linha['wall_s']:>5.1f}s "
                          f"· {i}/{len(faltam)}, resta ~{resta/60:.0f} min")

        if feitas:
            linhas = [feitas[r] for r in sorted(feitas) if r < a.runs]
            resumos[(na, ns)] = resumir(linhas)

    if not resumos:
        print("\nnada a resumir")
        return 1

    # ------------------------------------------------------------------ resumo
    print("\n" + "=" * 96)
    nota_ep = ("o número do test.py de Lamenza" if a.episodios == EPISODIOS_LAMENZA
               else f"ATENÇÃO: Lamenza usa {EPISODIOS_LAMENZA}")
    print(f"RESUMO — média ± desvio SOBRE AS RUNS (cada run = {a.episodios} "
          f"episódios; {nota_ep})")
    print("=" * 96)
    print(f"{'célula':>9} {'runs':>5} {'conclusão (s)':>19} {'sucesso':>16} "
          f"{'separação (m)':>18} {'latência (µs)':>17}")
    for (na, ns), r in resumos.items():
        n = len([1 for x in Path(caminho_csv(Path(a.dir_saida), na, ns))
                 .read_text().splitlines()[1:] if x.strip()])
        def par(c, casas=2):
            if c not in r:
                return "—"
            m, s = r[c]
            return f"{m:.{casas}f} ± {s:.{casas}f}"
        print(f"{f'a_{na}-s_{ns}':>9} {min(n, a.runs):>5} "
              f"{par('completion_time_medio'):>19} "
              f"{par('taxa_sucesso', 4):>16} "
              f"{par('separacao_media_m', 1):>18} "
              f"{par('latencia_mediana_us', 1):>17}")

    print(f"\n{'célula':>9} {'params ator':>12} {'ator fp32':>12} {'mult. fp':>10}")
    for (na, ns) in resumos:
        with open(caminho_csv(Path(a.dir_saida), na, ns), newline="") as f:
            primeira = next(csv.DictReader(f))
        print(f"{f'a_{na}-s_{ns}':>9} {int(primeira['params_ator']):>12} "
              f"{int(primeira['bytes_ator_fp32'])/1024:>9.1f} KB "
              f"{int(primeira['mult_fp']):>10}")

    print("\nO desvio acima é ENTRE RUNS, não entre episódios: mede quanto o valor")
    print("publicado variaria numa nova campanha. O desvio entre episódios está na")
    print("coluna `completion_time_std_episodios` do CSV, e é muito maior.")
    print(f"\nCSVs: {a.dir_saida}/campanha-a*s*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
