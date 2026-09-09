"""Média e desvio-padrão da taxa de sucesso e do completion time de um CSV da campanha.

Escolha um dos CSVs de `reports/campanha/` — pelo número no menu, ou passando o nome
(ou caminho) como argumento — e o script devolve, para a taxa de sucesso e para o
completion time, a média, o desvio-padrão amostral (ddof=1), o erro-padrão, o n, o
mínimo e o máximo.

A campanha grava dois formatos de CSV, e o script reconhece os dois sozinho pelo
cabeçalho:

    episódios   uma linha por episódio — colunas `sucesso` (0/1) e `completion_time`.
                A média da taxa de sucesso é a fração de episódios com sucesso=1; o
                desvio é sobre os episódios. É o formato dos arquivos `*-episodios.csv`.

    agregado    uma linha por corrida (bloco de episódios) — colunas `taxa_sucesso` e
                `completion_time_medio`, já resumidas por corrida. Aqui a média e o
                desvio são ENTRE corridas/sementes, que é a forma que vai para o texto
                da dissertação ("média ± desvio sobre as sementes").

Exemplos:

    # menu interativo com todos os CSVs da pasta
    uv run python scripts/campanha_resumo.py

    # direto num arquivo (nome basta; a pasta padrão é reports/campanha/)
    uv run python scripts/campanha_resumo.py wnn-T16-a2s24-limpo.csv

    # agrupando por degrau da escada (só faz sentido nos *-episodios.csv)
    uv run python scripts/campanha_resumo.py wnn-T16-a2s24-limpo-episodios.csv --por degrau

    # completion time só dos episódios que completaram
    uv run python scripts/campanha_resumo.py wnn-T16-a2s24-limpo-episodios.csv --apenas-sucesso
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PASTA_PADRAO = Path(__file__).resolve().parent.parent / "reports" / "campanha"

# Nos dois esquemas, qual coluna carrega cada grandeza. A primeira que existir vence.
COL_SUCESSO = ("taxa_sucesso", "sucesso")
COL_TEMPO = ("completion_time_medio", "completion_time")


def _primeira_presente(campos: list[str], candidatas: tuple[str, ...]) -> str | None:
    for c in candidatas:
        if c in campos:
            return c
    return None


def listar_csvs(pasta: Path) -> list[Path]:
    return sorted(pasta.glob("*.csv"))


def escolher_interativo(pasta: Path) -> Path:
    arquivos = listar_csvs(pasta)
    if not arquivos:
        sys.exit(f"Nenhum CSV em {pasta}")
    print(f"CSVs em {pasta}:\n")
    for i, f in enumerate(arquivos):
        print(f"  [{i:2d}] {f.name}")
    escolha = input("\nNúmero do CSV: ").strip()
    try:
        return arquivos[int(escolha)]
    except (ValueError, IndexError):
        sys.exit(f"Escolha inválida: {escolha!r}")


def resolver_caminho(alvo: str, pasta: Path) -> Path:
    """Aceita índice do menu, nome dentro da pasta padrão, ou caminho completo."""
    if alvo.isdigit():
        arquivos = listar_csvs(pasta)
        idx = int(alvo)
        if not 0 <= idx < len(arquivos):
            sys.exit(f"Índice fora do intervalo: {idx}")
        return arquivos[idx]
    p = Path(alvo)
    if p.exists():
        return p
    candidato = pasta / alvo
    if candidato.exists():
        return candidato
    if not alvo.endswith(".csv") and (pasta / f"{alvo}.csv").exists():
        return pasta / f"{alvo}.csv"
    sys.exit(f"CSV não encontrado: {alvo!r} (nem em {pasta})")


def carregar(caminho: Path) -> tuple[list[str], list[dict[str, str]]]:
    with caminho.open(newline="") as fh:
        leitor = csv.DictReader(fh)
        campos = leitor.fieldnames or []
        linhas = list(leitor)
    if not linhas:
        sys.exit(f"CSV vazio: {caminho}")
    return campos, linhas


def _floats(linhas: list[dict[str, str]], coluna: str) -> np.ndarray:
    vals = []
    for ln in linhas:
        bruto = (ln.get(coluna) or "").strip()
        if bruto == "":
            continue
        try:
            vals.append(float(bruto))
        except ValueError:
            continue
    return np.asarray(vals, dtype=float)


def resumo(valores: np.ndarray) -> dict[str, float]:
    n = valores.size
    media = float(np.mean(valores)) if n else float("nan")
    # desvio amostral (ddof=1) precisa de n>=2; com n=1 não há dispersão a estimar.
    desvio = float(np.std(valores, ddof=1)) if n >= 2 else 0.0
    ep = desvio / np.sqrt(n) if n >= 2 else 0.0
    return {
        "n": n,
        "media": media,
        "desvio": desvio,
        "erro_padrao": ep,
        "min": float(np.min(valores)) if n else float("nan"),
        "max": float(np.max(valores)) if n else float("nan"),
    }


def _imprimir_bloco(titulo: str, r: dict[str, float], percentual: bool, unidade: str = "") -> None:
    if percentual:
        print(f"  {titulo}")
        print(
            f"    média ± desvio : {r['media'] * 100:.2f}% ± {r['desvio'] * 100:.2f} pp"
            f"   (EP {r['erro_padrao'] * 100:.2f} pp)"
        )
        print(
            f"    min / max      : {r['min'] * 100:.2f}% / {r['max'] * 100:.2f}%"
            f"   (n={r['n']})"
        )
    else:
        u = f" {unidade}" if unidade else ""
        print(f"  {titulo}")
        print(
            f"    média ± desvio : {r['media']:.3f} ± {r['desvio']:.3f}{u}"
            f"   (EP {r['erro_padrao']:.3f}{u})"
        )
        print(f"    min / max      : {r['min']:.3f}{u} / {r['max']:.3f}{u}   (n={r['n']})")


def relatar(
    caminho: Path,
    campos: list[str],
    linhas: list[dict[str, str]],
    por: str | None,
    apenas_sucesso: bool,
) -> None:
    col_suc = _primeira_presente(campos, COL_SUCESSO)
    col_tmp = _primeira_presente(campos, COL_TEMPO)
    if col_suc is None and col_tmp is None:
        sys.exit(
            f"{caminho.name} não tem coluna de sucesso nem de completion time "
            f"reconhecível (colunas: {', '.join(campos)})"
        )

    esquema = "episódios" if col_suc == "sucesso" else "agregado (por corrida)"
    print(f"\nArquivo : {caminho}")
    print(f"Esquema : {esquema}")
    print(f"Sucesso : coluna {col_suc!r}   |   Tempo : coluna {col_tmp!r}")

    # completion time só dos episódios que completaram — só existe no esquema de episódios,
    # onde há uma linha por episódio com o seu sucesso ao lado.
    if apenas_sucesso and col_suc == "sucesso":
        antes = len(linhas)
        linhas = [ln for ln in linhas if (ln.get("sucesso") or "").strip() == "1"]
        print(f"Filtro  : apenas episódios com sucesso=1 ({len(linhas)}/{antes})")
    elif apenas_sucesso:
        print("Filtro  : --apenas-sucesso ignorado (não é CSV de episódios)")

    if por and por not in campos:
        sys.exit(f"Coluna de agrupamento inexistente: {por!r} (colunas: {', '.join(campos)})")

    grupos: list[tuple[str, list[dict[str, str]]]]
    if por:
        chaves = sorted({(ln.get(por) or "").strip() for ln in linhas}, key=_ordem_chave)
        grupos = [(f"{por}={k}", [ln for ln in linhas if (ln.get(por) or "").strip() == k]) for k in chaves]
    else:
        grupos = [("todas as linhas", linhas)]

    for rotulo, subset in grupos:
        print(f"\n── {rotulo}  ({len(subset)} linhas) ──")
        if col_suc is not None:
            _imprimir_bloco("Taxa de sucesso", resumo(_floats(subset, col_suc)), percentual=True)
        if col_tmp is not None:
            _imprimir_bloco("Completion time", resumo(_floats(subset, col_tmp)), percentual=False, unidade="s")
    print()


def _ordem_chave(k: str):
    try:
        return (0, float(k))
    except ValueError:
        return (1, k)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="nome, índice do menu ou caminho do CSV. Sem isso, abre o menu.")
    ap.add_argument("--dir", type=Path, default=PASTA_PADRAO, help=f"pasta dos CSVs (padrão: {PASTA_PADRAO})")
    ap.add_argument("--por", metavar="COL", help="agrupa por uma coluna (ex.: degrau, sigma, seed) antes de resumir")
    ap.add_argument("--apenas-sucesso", action="store_true", help="completion time só dos episódios com sucesso=1 (CSV de episódios)")
    args = ap.parse_args(argv)

    caminho = resolver_caminho(args.csv, args.dir) if args.csv else escolher_interativo(args.dir)
    campos, linhas = carregar(caminho)
    relatar(caminho, campos, linhas, args.por, args.apenas_sucesso)


if __name__ == "__main__":
    main()
