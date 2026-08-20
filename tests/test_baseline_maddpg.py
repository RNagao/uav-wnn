"""Baseline MADDPG dos pesos publicados — P4 fechada por medição.

Dois níveis, e a separação é deliberada:

- **Sem torch e sem submodule:** confere o artefato congelado
  `reports/spec02-baseline-maddpg.json` contra as fórmulas do PRD. É o que roda
  na suíte padrão, e é o que protege a razão de tamanho da §5.5 de mudar sem
  ninguém notar.
- **Com torch e submodule:** relê os checkpoints e confirma que o artefato
  corresponde aos pesos de fato. Pulado por omissão, porque torch é um grupo
  opcional de ~200 MB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTEFATO = Path(__file__).parents[1] / "reports" / "spec02-baseline-maddpg.json"
LARGURA_ATOR, N_ACOES = 256, 2


@pytest.fixture(scope="module")
def medido():
    if not ARTEFATO.exists():
        pytest.skip("artefato ausente; rode scripts/spec02_baseline_maddpg.py")
    return json.loads(ARTEFATO.read_text())


# --- o artefato bate com as fórmulas do PRD -------------------------------


def test_nove_celulas_publicadas(medido):
    """`Na ∈ {2,4,8} × Ns ∈ {12,24,36}` — as células da fig. 7.3 de Lamenza."""
    celulas = medido["celulas"]
    assert len(celulas) == 9
    assert {c["num_drones"] for c in celulas.values()} == {2, 4, 8}
    assert {c["num_sensores"] for c in celulas.values()} == {12, 24, 36}


def test_nao_existe_checkpoint_para_um_drone(medido):
    """Ressalva que precisa ir para o relatório: a Etapa 1 começa em `Na = 1`,
    e ali **não há baseline MADDPG**. As células publicadas começam em 2."""
    assert 1 not in {c["num_drones"] for c in medido["celulas"].values()}


def test_observacao_segue_a_formula_do_prd(medido):
    """`D = 2(Na − 1) + 25` (PRD §5.2). Confirmado nas formas dos checkpoints.

    Vale como validação cruzada da nossa leitura da observação de Lamenza: se a
    fórmula estivesse errada, `fc1.weight` teria outra largura de entrada.
    """
    for nome, c in medido["celulas"].items():
        na = c["num_drones"]
        assert c["tamanho"]["obs_dim"] == 2 * (na - 1) + 25, nome


def test_observacao_nao_depende_do_numero_de_sensores(medido):
    """`N*_s = 12` é fixo: a observação sempre traz os 12 sensores mais
    próximos, então `Ns ∈ {12,24,36}` não muda `obs_dim`. É o que sustenta o
    zero-padding do PRD §5.2."""
    por_na: dict[int, set[int]] = {}
    for c in medido["celulas"].values():
        por_na.setdefault(c["num_drones"], set()).add(c["tamanho"]["obs_dim"])
    assert all(len(v) == 1 for v in por_na.values()), por_na


def test_contagem_de_parametros_do_ator(medido):
    """Ator `obs → 256 → 256 → 2`, pesos **e** vieses. Fecha a **P4**: era
    estimativa marcada ("padrão CleanRL, 256×256"), agora é medição."""
    for nome, c in medido["celulas"].items():
        obs = c["tamanho"]["obs_dim"]
        esperado = ((obs * LARGURA_ATOR + LARGURA_ATOR)
                    + (LARGURA_ATOR * LARGURA_ATOR + LARGURA_ATOR)
                    + (LARGURA_ATOR * N_ACOES + N_ACOES))
        assert c["tamanho"]["params_ator"] == esperado, nome
        assert c["tamanho"]["bytes_ator_fp32"] == esperado * 4


def test_critico_e_maior_que_o_ator(medido):
    """Crítico `512×512` e centralizado — recebe observações e ações de **todos**
    os drones, que é o que faz o MADDPG ser MADDPG. Não entra na razão de
    tamanho da §5.5, que compara **atores**, mas precisa estar medido para a
    coluna ator+crítico."""
    for nome, c in medido["celulas"].items():
        t = c["tamanho"]
        assert t["params_critico"] > 3 * t["params_ator"], nome


def test_razao_contra_a_nossa_wnn_esta_na_faixa_reportada(medido):
    """A SPEC-03 reporta `0,47× o ator do MADDPG` para a WNN de 133,9 KB.

    Aquele número era **estimativa marcada** porque supunha 256×256. Agora que a
    arquitetura é medida, o teste trava a razão: se alguém mexer no artefato ou
    na configuração da WNN, a razão da §5.5 deixa de fechar e isto acusa.
    """
    wnn_kb = 133.9                      # reports/spec03-custo.md, variante ordenada
    for nome, c in medido["celulas"].items():
        kb = c["tamanho"]["bytes_ator_fp32"] / 1024
        assert 0.44 < wnn_kb / kb < 0.48, f"{nome}: razão {wnn_kb / kb:.3f}"


def test_operacoes_do_ator_sao_dominadas_por_multiplicacoes(medido):
    """O contraste central de T3: ~73 000 multiplicações fp por decisão, contra
    **zero** no nosso aproximador."""
    for nome, c in medido["celulas"].items():
        ops = c["operacoes"]
        assert ops["mult_fp"] > 70_000, nome
        assert ops["mult_fp"] > 0.45 * ops["total"], nome


def test_latencia_medida_com_uma_thread_e_batch_um(medido):
    """A decisão de um agente é sequencial. Paralelismo de biblioteca inflaria a
    comparação com um modelo que não tem o que paralelizar."""
    cfg = medido["config"]
    assert cfg["threads"] == 1 and cfg["batch"] == 1 and cfg["dtype"] == "fp32"
    for nome, c in medido["celulas"].items():
        lat = c["latencia"]
        assert 0 < lat["mediana_us"] < 500, nome
        assert lat["p99_us"] >= lat["mediana_us"], nome


# --- nível 2: relê os pesos de verdade -------------------------------------


def test_artefato_corresponde_aos_checkpoints(medido):
    """Só com torch e submodule presentes. Fecha o laço: o artefato congelado
    descreve os pesos publicados, não uma tabela digitada à mão."""
    pytest.importorskip("torch", reason="grupo opcional: uv sync --group baseline")
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from spec02_baseline_maddpg import checkpoints, tamanho

    achados = checkpoints()
    if not achados:
        pytest.skip("submodule ausente")
    for na, ns, caminho in achados:
        esperado = medido["celulas"][f"a_{na}-s_{ns}"]["tamanho"]
        obtido = tamanho(caminho)
        assert obtido["params_ator"] == esperado["params_ator"]
        assert obtido["obs_dim"] == esperado["obs_dim"]
