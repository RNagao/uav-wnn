"""Campanha do baseline MADDPG — a lógica de cache e de agregação.

Estes testes **não rodam o ambiente**: exercitam o cache, a semeadura e a
agregação com dados sintéticos. É o que dá para verificar sem torch, sem
submodule e em milissegundos — e é onde os erros silenciosos moram.

O que os testes protegem, em ordem de gravidade se quebrasse:

1. **Runs com `E` diferente não podem ser misturadas.** Média sobre runs de 20 e
   de 200 episódios não significa nada, e o número sairia plausível.
2. **Sementes não podem se sobrepor entre runs.** Se `run 0` e `run 1` vissem os
   mesmos cenários, o desvio entre runs colapsaria para ~0 e reportaríamos
   precisão inexistente.
3. **O desvio reportado é entre RUNS, não entre episódios.** São grandezas
   diferentes por um fator de `√E`, e confundi-las muda a conclusão.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from spec02_maddpg_campanha import (
    COLUNAS,
    EPISODIOS_LAMENZA,
    caminho_csv,
    escrever_linha,
    ler_cache,
    resumir,
    separacao,
)


def _linha(run: int, episodios: int = 200, ct: float = 13.5, sep: float = 90.0):
    return {
        "run": run, "num_drones": 4, "num_sensores": 24,
        "episodios": episodios, "seed_inicial": run * episodios,
        "completion_time_medio": ct, "completion_time_std_episodios": 3.2,
        "taxa_sucesso": 1.0, "coleta_media": 6.5, "separacao_media_m": sep,
        "params_ator": 74498, "bytes_ator_fp32": 297992, "mult_fp": 73984,
        "latencia_mediana_us": 35.0, "latencia_p99_us": 110.0, "wall_s": 7.1,
    }


# --- o número de episódios por run --------------------------------------


def test_episodios_por_run_e_o_de_lamenza():
    """`total_runs = 200` no `test.py` do submodule. Cada run é uma réplica da
    campanha de avaliação dele."""
    assert EPISODIOS_LAMENZA == 200


def test_cache_recusa_runs_com_episodios_diferentes(tmp_path):
    """A proteção mais importante: média sobre runs de tamanhos distintos
    produziria um número plausível e sem significado."""
    arq = tmp_path / "c.csv"
    escrever_linha(arq, _linha(0, episodios=20))
    with pytest.raises(SystemExit) as e:
        ler_cache(arq, episodios=200)
    assert "20" in str(e.value) and "200" in str(e.value)


def test_cache_aceita_quando_bate(tmp_path):
    arq = tmp_path / "c.csv"
    for r in (0, 1, 2):
        escrever_linha(arq, _linha(r))
    feitas = ler_cache(arq, episodios=200)
    assert set(feitas) == {0, 1, 2}


def test_cache_vazio_quando_nao_existe(tmp_path):
    assert ler_cache(tmp_path / "nao-existe.csv", 200) == {}


# --- ida e volta pelo CSV ------------------------------------------------


def test_csv_preserva_todas_as_colunas(tmp_path):
    """Se uma figura de mérito sumisse do CSV, o resumo a omitiria em silêncio."""
    arq = tmp_path / "c.csv"
    escrever_linha(arq, _linha(0))
    with open(arq, newline="") as f:
        lido = next(csv.DictReader(f))
    assert list(lido.keys()) == COLUNAS


def test_append_nao_reescreve_cabecalho(tmp_path):
    arq = tmp_path / "c.csv"
    for r in range(5):
        escrever_linha(arq, _linha(r))
    linhas = arq.read_text().strip().splitlines()
    assert len(linhas) == 6, "1 cabeçalho + 5 runs"
    assert linhas[0].startswith("run,")
    assert sum(1 for x in linhas if x.startswith("run,")) == 1


def test_nome_do_csv_identifica_o_cenario(tmp_path):
    """O cache é por cenário — misturar células no mesmo arquivo daria média
    entre configurações diferentes."""
    a = caminho_csv(tmp_path, 4, 24)
    b = caminho_csv(tmp_path, 8, 24)
    assert a != b
    assert a.name == "campanha-a4s24.csv"


# --- semeadura sem sobreposição -----------------------------------------


def test_sementes_das_runs_nao_se_sobrepoem():
    """`run r` usa `[r·E, (r+1)·E)`. Reusar o bloco daria `R` cópias do mesmo
    número e um desvio artificialmente nulo."""
    E, R = 200, 100
    blocos = [set(range(r * E, (r + 1) * E)) for r in range(R)]
    assert all(len(b) == E for b in blocos)
    assert len(set().union(*blocos)) == E * R, "houve sobreposição"
    for r in range(R):
        assert _linha(r, E)["seed_inicial"] == r * E


# --- agregação ----------------------------------------------------------


def test_resumo_e_media_e_desvio_sobre_as_runs():
    linhas = [_linha(r, ct=c) for r, c in enumerate([13.0, 13.5, 14.0])]
    r = resumir([{k: str(v) for k, v in x.items()} for x in linhas])
    m, s = r["completion_time_medio"]
    assert m == pytest.approx(13.5)
    assert s == pytest.approx(np.std([13.0, 13.5, 14.0], ddof=1))


def test_desvio_entre_runs_e_menor_que_entre_episodios():
    """São grandezas diferentes por ~`√E`, e o resumo reporta a **primeira**.

    Confundi-las é o erro fácil: o desvio entre episódios diz "quanto um
    episódio varia"; o entre runs diz "quanto o valor publicado variaria numa
    nova campanha". Para comparar com a nossa política importa o segundo.
    """
    rng = np.random.default_rng(0)
    E = 200
    por_episodio = rng.normal(13.5, 3.2, size=(20, E))
    medias = por_episodio.mean(axis=1)
    entre_runs = medias.std(ddof=1)
    entre_episodios = por_episodio.std(ddof=1)
    assert entre_runs < entre_episodios / 5
    assert entre_runs == pytest.approx(entre_episodios / np.sqrt(E), rel=0.5)


def test_resumo_de_uma_run_unica_tem_desvio_zero():
    r = resumir([{k: str(v) for k, v in _linha(0).items()}])
    assert r["completion_time_medio"][1] == 0.0


def test_resumo_ignora_nan():
    """`separacao_media_m` é `nan` com um drone só; a coluna não pode envenenar
    a média das outras células."""
    linhas = [{k: str(v) for k, v in _linha(0).items()},
              {k: str(v) for k, v in _linha(1, sep=float("nan")).items()}]
    r = resumir(linhas)
    assert r["separacao_media_m"][0] == pytest.approx(90.0)


# --- separação -----------------------------------------------------------


def test_separacao_de_dois_uavs_parados_e_a_distancia():
    T = np.array([[[0.0, 0.0]] * 5, [[3.0, 4.0]] * 5])
    assert separacao(T, 2) == pytest.approx(5.0)


def test_separacao_de_um_uav_e_nan():
    """Com um drone não há par, e `nan` é o valor honesto — zero sugeriria
    formação perfeita."""
    assert np.isnan(separacao(np.array([[[0.0, 0.0]] * 3]), 1))


def test_separacao_usa_o_menor_comprimento_comum():
    """Trilhas de comprimentos diferentes acontecem quando um UAV para antes.
    Truncar no menor evita comparar passos desalinhados."""
    T = np.array([np.zeros((5, 2)), np.ones((3, 2))], dtype=object)
    T = [np.zeros((5, 2)), np.ones((3, 2))]
    assert separacao(T, 2) == pytest.approx(np.sqrt(2))


# --- o array de cenários -------------------------------------------------


def test_cenarios_lamenza_sao_as_nove_celulas():
    """A grade da tab. 7.1: `Na ∈ {2,4,8} × Ns ∈ {12,24,36}`.

    São as mesmas células da fig. 7.3 e as mesmas para as quais o submodule
    publica checkpoints — não uma escolha nossa.
    """
    from spec02_maddpg_campanha import CENARIOS_LAMENZA as C

    assert len(C) == 9
    assert {na for na, _ in C} == {2, 4, 8}
    assert {ns for _, ns in C} == {12, 24, 36}
    assert len(set(C)) == 9, "sem duplicatas"
    assert 1 not in {na for na, _ in C}, "não há checkpoint para Na = 1"


def test_celulas_vazio_devolve_o_padrao_de_lamenza():
    from spec02_maddpg_campanha import CENARIOS_LAMENZA, analisar_celulas

    assert analisar_celulas("") == CENARIOS_LAMENZA
    assert analisar_celulas("   ") == CENARIOS_LAMENZA


def test_celulas_aceita_subconjunto_e_preserva_a_ordem():
    """A ordem importa: quem pede `8:36,2:12` quer rodar a célula cara primeiro."""
    from spec02_maddpg_campanha import analisar_celulas

    assert analisar_celulas("4:24,8:36") == [(4, 24), (8, 36)]
    assert analisar_celulas("8:36,2:12") == [(8, 36), (2, 12)]
    assert analisar_celulas(" 4:24 , 8:36 ") == [(4, 24), (8, 36)]


@pytest.mark.parametrize("entrada,pedaco", [
    ("abc", "esperado 'Na:Ns'"),
    ("4-24", "esperado 'Na:Ns'"),
    ("0:12", ">= 1"),
    ("4:0", ">= 1"),
    ("4:24,4:24", "repetida"),
])
def test_celulas_invalidas_falham_com_mensagem(entrada, pedaco):
    """Recusar alto em vez de adivinhar. Uma célula repetida faria o resumo
    contar as runs dela em dobro."""
    from spec02_maddpg_campanha import analisar_celulas

    with pytest.raises(SystemExit) as e:
        analisar_celulas(entrada)
    assert pedaco in str(e.value)


def test_campanha_e_chamavel_como_biblioteca():
    """`campanha()` existe para encadear com outra análise sem passar pelo CSV.

    Aqui só verificamos a assinatura e o padrão do array — rodar de verdade
    exige torch e submodule, e isso está nos testes de nível 2.
    """
    import inspect

    from spec02_maddpg_campanha import CENARIOS_LAMENZA, campanha

    sig = inspect.signature(campanha)
    assert sig.parameters["cenarios"].default is None, "None -> CENARIOS_LAMENZA"
    assert sig.parameters["runs"].default == 100
    assert sig.parameters["episodios"].default == EPISODIOS_LAMENZA
    assert list(CENARIOS_LAMENZA)  # o padrão não é vazio
