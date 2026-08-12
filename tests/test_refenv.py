"""Testes do acesso ao ambiente de referência (SPEC-00).

Estes testes NÃO testam o MDP — ele é código de terceiros, verbatim, e testá-lo
seria testar o trabalho de Lamenza. Testam o que é nosso: que carregamos o
arquivo certo, na versão certa, com a configuração certa, de forma
reprodutível e sem vazar recursos.

Também documentam, com número, as características do ambiente que importam
para o projeto — em particular as diferenças entre as fases de treino e de
avaliação usadas no trabalho de referência.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from uavwnn.refenv import (
    LAMENZA_003,
    REFERENCE_COMMIT,
    EnvConfig,
    ReferenceNotAvailable,
    load_environment,
    load_heuristics,
    reference_root,
    run_episode,
)

try:
    reference_root()
    _AVAILABLE = True
except ReferenceNotAvailable:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="submodule ausente: rode `git submodule update --init --recursive`",
)


def greedy_for(cfg: EnvConfig):
    h = load_heuristics()
    n_a = cfg.state_num_closest_drones
    if n_a is None:
        n_a = cfg.num_drones - 1
    return h.create_greedy_heuristics(n_a, cfg.state_num_closest_sensors)


# --- proveniência ----------------------------------------------------------


def test_submodule_fixado_na_tag_dissertation():
    """O commit precisa ser o da tag `dissertation`, não o do branch default.

    `dissertation` é TAG; o branch default (`main`) é um estado posterior. Nos
    arquivos que usamos eles coincidem hoje, mas depender disso seria depender
    de um acidente.
    """
    import subprocess

    head = subprocess.run(
        ["git", "-C", str(reference_root()), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == REFERENCE_COMMIT


def test_ambiente_carrega_sem_patch():
    """O `environment.py` de referência importa `VisualizationController`, que
    existe até o gradysim 0.6.11 e sumiu na série 0.7. Se este teste falhar
    após um bump de versão, o código intacto passou a rodar contra outro
    simulador — o que é, por si só, uma divergência do baseline."""
    mod = load_environment()
    assert hasattr(mod, "GrADySEnvironment")
    assert hasattr(mod, "SensorProtocol") and hasattr(mod, "DroneProtocol")


def test_versao_do_gradysim_fixada():
    from importlib.metadata import version

    assert version("gradysim") == "0.6.11"


# --- configuração publicada ------------------------------------------------


def test_config_publicada():
    """Valores do `benchmark.py`, não os defaults de teste do `main.py`."""
    cfg = LAMENZA_003(4, 24)
    assert cfg.communication_range == 20.0
    assert cfg.algorithm_iteration_interval == 0.5
    assert cfg.state_num_closest_sensors == 12
    assert cfg.reward == "punish"
    assert cfg.state_mode == "relative"
    assert cfg.speed_action is True
    assert cfg.min_sensor_priority == 1.0


@pytest.mark.parametrize("n_a,expected", [(1, 25), (2, 27), (4, 31), (8, 39)])
def test_tamanho_da_observacao(n_a, expected):
    cfg = LAMENZA_003(n_a, 24)
    assert cfg.observation_size == expected

    import random

    from uavwnn.refenv import make_env

    env = make_env(cfg)
    random.seed(0)
    obs, _ = env.reset()
    assert obs["drone0"].shape == (expected,)


def test_fases_de_treino_e_avaliacao_diferem():
    """No trabalho de referência, treino e avaliação NÃO usam a mesma config."""
    assert LAMENZA_003(4, 24, "train").end_when_all_collected is False
    assert LAMENZA_003(4, 24, "eval").end_when_all_collected is True
    # E a heurística é avaliada com decisão 5x mais frequente que a política.
    assert LAMENZA_003(4, 24, "eval_heuristic").algorithm_iteration_interval == 0.1
    assert LAMENZA_003(4, 24, "eval").algorithm_iteration_interval == 0.5


# --- comportamento ---------------------------------------------------------


def test_guloso_completa_a_coleta():
    cfg = LAMENZA_003(4, 24)
    rows = [run_episode(cfg, greedy_for(cfg), seed=s) for s in range(1, 6)]
    assert all(r.success for r in rows)
    assert all(5.0 < r.completion_time < 200.0 for r in rows)


def test_padding_do_original_e_menos_um():
    """O padding da observação é -1, fora da faixa dos valores válidos.

    É por isso que "slot vazio" e "sensor sobre o UAV" são distinguíveis por
    construção, e o bit de validade que a SPEC-01 projetava é desnecessário.
    """
    import random

    from uavwnn.refenv import make_env

    cfg = LAMENZA_003(1, 2)  # menos sensores que N*_s = 12 -> slots vazios
    env = make_env(cfg)
    random.seed(3)
    obs, _ = env.reset()
    v = obs["drone0"]
    assert np.isclose(v, -1.0).any(), "esperava slots preenchidos com -1"
    validos = v[~np.isclose(v, -1.0)]
    assert validos.min() >= -0.6 and validos.max() <= 1.6


def test_avaliar_guloso_a_0_1_e_mais_rapido():
    """Quantifica a assimetria da fig. 7.3: lá o MADDPG é avaliado a 0,5 s por
    decisão e o guloso a 0,1 s. O guloso mais reativo é mais rápido, logo a
    vantagem publicada do MADDPG sobre ele é conservadora."""
    seeds = range(1, 9)
    a = LAMENZA_003(4, 24, "eval")
    b = LAMENZA_003(4, 24, "eval_heuristic")
    ct_a = np.mean([run_episode(a, greedy_for(a), seed=s).completion_time for s in seeds])
    ct_b = np.mean([run_episode(b, greedy_for(b), seed=s).completion_time for s in seeds])
    assert ct_b < ct_a, f"esperava guloso mais rápido a 0.1 s: {ct_b:.2f} vs {ct_a:.2f}"


# --- reprodutibilidade e higiene -------------------------------------------


def test_mesma_semente_mesmo_resultado():
    cfg = LAMENZA_003(2, 12)
    g = greedy_for(cfg)
    a = run_episode(cfg, g, seed=42)
    b = run_episode(cfg, g, seed=42)
    assert a.completion_time == b.completion_time
    assert a.steps == b.steps
    assert a.collected == b.collected
    assert a.collection_times == b.collection_times


def test_sementes_diferentes_dao_cenarios_diferentes():
    cfg = LAMENZA_003(2, 12)
    g = greedy_for(cfg)
    ts = {run_episode(cfg, g, seed=s).completion_time for s in range(1, 8)}
    assert len(ts) > 1


def test_sem_vazamento_de_handlers_de_logging():
    """O `Simulator` do gradysim acrescenta um StreamHandler ao logger raiz por
    instância e nunca o remove. Sem poda, 10 000 episódios acumulariam 10 000
    handlers."""
    cfg = LAMENZA_003(2, 12)
    g = greedy_for(cfg)
    run_episode(cfg, g, seed=0)
    antes = len(logging.getLogger().handlers)
    for s in range(1, 16):
        run_episode(cfg, g, seed=s)
    assert len(logging.getLogger().handlers) == antes


# --- validação contra os resultados publicados -----------------------------

PUBLICADO_FIG_7_3 = {
    (2, 12): 29.6, (2, 24): 38.2, (2, 36): 43.2,
    (4, 12): 28.2, (4, 24): 35.4, (4, 36): 39.7,
    (8, 12): 27.6, (8, 24): 33.2, (8, 36): 36.7,
}
"""Barras "Heuristic" da fig. 7.3, lidas dos PNGs em
`vendor/.../statistics/completion_times/`. Precisão de leitura ~0,3 s."""


@pytest.mark.parametrize("cell", [(2, 12), (4, 24), (8, 36)])
def test_guloso_reproduz_a_figura_7_3(cell):
    """Prova de que a montagem está correta.

    Se o guloso do próprio trabalho de referência, rodando no ambiente de
    referência, reproduz os tempos publicados, então submodule, versão do
    simulador, configuração e laço de avaliação estão todos certos -- sem
    precisar de nenhuma política aprendida.

    Três células aqui (a diagonal) para manter o teste rápido. A grade completa
    está em `scripts/spec00_validar_guloso.py`: 9/9 células dentro de 10 %,
    erro mediano 1,7 %, viés médio +0,5 %.
    """
    n_a, n_s = cell
    publicado = PUBLICADO_FIG_7_3[cell]

    # dt = 0,1 s: é assim que `visualize_performance.py` avalia a heurística.
    cfg = LAMENZA_003(n_a, n_s, phase="eval_heuristic")
    greedy = greedy_for(cfg)

    n_ep = 60
    ct = np.array([run_episode(cfg, greedy, seed=i).completion_time for i in range(n_ep)])
    media = float(ct.mean())
    erro = (media - publicado) / publicado

    assert abs(erro) <= 0.12, (
        f"Na={n_a} Ns={n_s}: medido {media:.2f} s vs publicado {publicado:.1f} s "
        f"({erro:+.1%})"
    )
