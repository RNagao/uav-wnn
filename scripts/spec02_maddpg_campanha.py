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

CENÁRIOS

`CENARIOS_LAMENZA` é o padrão: as 9 células `(num_drones, num_sensores)` da
tab. 7.1, as mesmas da fig. 7.3 e as mesmas para as quais o submodule publica
checkpoints. Dá para pedir um subconjunto ou outra ordem.

Da linha de comando:

    uv run --group baseline python scripts/spec02_maddpg_campanha.py
    uv run --group baseline python scripts/spec02_maddpg_campanha.py --runs 10
    uv run --group baseline python scripts/spec02_maddpg_campanha.py --celulas 4:24,8:36
    uv run --group baseline python scripts/spec02_maddpg_campanha.py --so-resumo

Como biblioteca, recebendo o array direto:

    import sys; sys.path.insert(0, "scripts")
    from spec02_maddpg_campanha import campanha, CENARIOS_LAMENZA

    campanha()                                    # as 9 de Lamenza, 100 runs
    campanha([(4, 24), (8, 36)], runs=30)         # subconjunto
    campanha(CENARIOS_LAMENZA, runs=100)          # explícito, mesmo do padrão

`campanha()` devolve o resumo por cenário — `{(Na, Ns): {figura: (média, desvio)}}`
— então dá para encadear com outra análise sem passar pelo CSV.

**Pedir uma célula sem checkpoint recusa em voz alta** em vez de cair para a
vizinha. Não há checkpoint para `Na = 1`, que é a armadilha mais provável aqui.

CUSTO: ~2–3 h para 100 runs × 200 episódios × 9 células num núcleo. Rode em
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

CENARIOS_LAMENZA: list[tuple[int, int]] = [
    (2, 12), (2, 24), (2, 36),
    (4, 12), (4, 24), (4, 36),
    (8, 12), (8, 24), (8, 36),
]
"""A grade do trabalho de referência: `(num_drones, num_sensores)`.

São as 9 células da tab. 7.1 de Lamenza, as mesmas da fig. 7.3 e as mesmas para
as quais o submodule publica checkpoints. É o padrão de `--celulas`, e o valor
que `campanha()` usa quando não recebe `cenarios`.

**Não há checkpoint para `Na = 1`**, então a Etapa 1 do nosso plano não tem
baseline MADDPG. Pedir `1:12` falha com mensagem, e é o comportamento certo:
melhor recusar que devolver silenciosamente uma célula vizinha.
"""

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


def analisar_celulas(texto: str) -> list[tuple[int, int]]:
    """`'4:24,8:36'` -> `[(4, 24), (8, 36)]`. Vazio devolve `CENARIOS_LAMENZA`.

    Valida a forma e recusa duplicatas — pedir a mesma célula duas vezes faria
    o resumo contar as runs dela em dobro.
    """
    if not texto.strip():
        return list(CENARIOS_LAMENZA)
    fora: list[tuple[int, int]] = []
    for item in texto.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            na_txt, ns_txt = item.split(":")
            na, ns = int(na_txt), int(ns_txt)
        except ValueError:
            sys.exit(f"célula inválida: {item!r}; esperado 'Na:Ns', ex. '4:24'")
        if na < 1 or ns < 1:
            sys.exit(f"célula inválida: {item!r}; Na e Ns devem ser >= 1")
        if (na, ns) in fora:
            sys.exit(f"célula repetida: {item!r}")
        fora.append((na, ns))
    return fora


def campanha(
    cenarios: list[tuple[int, int]] | None = None,
    runs: int = 100,
    episodios: int = EPISODIOS_LAMENZA,
    dir_saida: str | Path = "reports/maddpg",
    so_resumo: bool = False,
    verboso: bool = True,
) -> dict[tuple[int, int], dict]:
    """Roda a campanha e devolve o resumo por cenário. Reusável de fora.

        from spec02_maddpg_campanha import campanha, CENARIOS_LAMENZA
        campanha([(4, 24), (8, 36)], runs=30)
        campanha()                       # as 9 células de Lamenza, 100 runs

    O cache vale igual: chamar de novo com mais `runs` só roda o que falta.
    """
    cenarios = list(CENARIOS_LAMENZA) if cenarios is None else list(cenarios)
    return _executar(cenarios, runs, episodios, Path(dir_saida), so_resumo, verboso)


def _executar(cenarios, runs, episodios, dir_saida: Path, so_resumo, verboso):
    """Miolo da campanha. `main()` e `campanha()` só preparam os argumentos."""
    fala = print if verboso else (lambda *_a, **_k: None)

    disponiveis = {(na, ns): c for na, ns, c in checkpoints()}
    if not disponiveis and not so_resumo:
        sys.exit("nenhum checkpoint; git submodule update --init --recursive")

    dir_saida.mkdir(parents=True, exist_ok=True)
    resumos: dict[tuple[int, int], dict] = {}
    contagens: dict[tuple[int, int], int] = {}

    for na, ns in cenarios:
        arquivo = caminho_csv(dir_saida, na, ns)
        feitas = ler_cache(arquivo, episodios)
        faltam = [r for r in range(runs) if r not in feitas]

        fala(f"\n=== a_{na}-s_{ns} · {runs} runs × {episodios} episódios ===")
        fala(f"  cache: {len(feitas)} runs em {arquivo.name}"
             f"{'' if not faltam else f'; faltam {len(faltam)}'}")

        if faltam and not so_resumo:
            if (na, ns) not in disponiveis:
                # Recusar em voz alta, não cair para uma célula vizinha. Não há
                # checkpoint para `Na = 1`, e é a armadilha mais provável aqui.
                fala(f"  SEM CHECKPOINT para a_{na}-s_{ns}; pulando "
                     f"(publicadas: {sorted(disponiveis)})")
                continue
            cfg = LAMENZA_003(na, ns, phase="eval")
            politica = carregar_ator(disponiveis[(na, ns)], cfg.observation_size)
            custo = tamanho(disponiveis[(na, ns)])
            ops = operacoes_ator(custo["obs_dim"])

            t0 = time.perf_counter()
            for i, r in enumerate(faltam, 1):
                linha = rodar_uma(cfg, politica, custo, ops, r, episodios)
                escrever_linha(arquivo, linha)
                feitas[r] = {k: str(v) for k, v in linha.items()}
                if i == 1 or i % 10 == 0 or i == len(faltam):
                    resta = (time.perf_counter() - t0) / i * (len(faltam) - i)
                    fala(f"    run {r:>3} · {linha['completion_time_medio']:>6.2f}s "
                         f"· {linha['wall_s']:>5.1f}s "
                         f"· {i}/{len(faltam)}, resta ~{resta/60:.0f} min")

        usaveis = [feitas[r] for r in sorted(feitas) if r < runs]
        if usaveis:
            resumos[(na, ns)] = resumir(usaveis)
            contagens[(na, ns)] = len(usaveis)

    if not resumos:
        fala("\nnada a resumir")
        return resumos

    # ------------------------------------------------------------------ resumo
    nota = ("o número do test.py de Lamenza" if episodios == EPISODIOS_LAMENZA
            else f"ATENÇÃO: Lamenza usa {EPISODIOS_LAMENZA}")
    fala("\n" + "=" * 96)
    fala(f"RESUMO — média ± desvio SOBRE AS RUNS "
         f"(cada run = {episodios} episódios; {nota})")
    fala("=" * 96)
    fala(f"{'célula':>9} {'runs':>5} {'conclusão (s)':>19} {'sucesso':>16} "
         f"{'separação (m)':>18} {'latência (µs)':>17}")

    def par(r, c, casas=2):
        if c not in r:
            return "—"
        m, s = r[c]
        return f"{m:.{casas}f} ± {s:.{casas}f}"

    for (na, ns), r in resumos.items():
        fala(f"{f'a_{na}-s_{ns}':>9} {contagens[(na, ns)]:>5} "
             f"{par(r, 'completion_time_medio'):>19} "
             f"{par(r, 'taxa_sucesso', 4):>16} "
             f"{par(r, 'separacao_media_m', 1):>18} "
             f"{par(r, 'latencia_mediana_us', 1):>17}")

    fala(f"\n{'célula':>9} {'params ator':>12} {'ator fp32':>12} {'mult. fp':>10}")
    for na, ns in resumos:
        with open(caminho_csv(dir_saida, na, ns), newline="") as f:
            primeira = next(csv.DictReader(f))
        fala(f"{f'a_{na}-s_{ns}':>9} {int(primeira['params_ator']):>12} "
             f"{int(primeira['bytes_ator_fp32'])/1024:>9.1f} KB "
             f"{int(primeira['mult_fp']):>10}")

    fala("\nO desvio acima é ENTRE RUNS, não entre episódios: mede quanto o valor")
    fala("publicado variaria numa nova campanha. O desvio entre episódios está na")
    fala("coluna `completion_time_std_episodios` do CSV, e é muito maior.")
    fala(f"\nCSVs: {dir_saida}/campanha-a*s*.csv")
    return resumos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--episodios", type=int, default=EPISODIOS_LAMENZA,
                    help=f"por run; {EPISODIOS_LAMENZA} é o do test.py de Lamenza")
    ap.add_argument(
        "--celulas", default="",
        help="lista 'Na:Ns' separada por vírgula, ex. '4:24,8:36'. "
             "Vazio = CENARIOS_LAMENZA (as 9 células da tab. 7.1)")
    ap.add_argument("--dir-saida", default="reports/maddpg")
    ap.add_argument("--so-resumo", action="store_true",
                    help="não roda nada; só resume os CSVs existentes")
    a = ap.parse_args()

    resumos = _executar(analisar_celulas(a.celulas), a.runs, a.episodios,
                        Path(a.dir_saida), a.so_resumo, verboso=True)
    return 0 if resumos else 1


if __name__ == "__main__":
    sys.exit(main())
