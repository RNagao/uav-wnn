"""Subclasse de ruído do ambiente — SPEC-07 T-07.12, guardas da ARQUITETURA §3.8.

Exige o `gradysim` (submodule): estes testes constroem o ambiente de referência.
As guardas centrais: com `σ = 0` a observação é **bit-idêntica** à do pai (a
subclasse não introduz diferença por si), o ruído de fato entra (a observação
muda com a semente de ruído mas não com a verdade), e o degrau 4 filtra.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from uavwnn.refenv import LAMENZA_003, load_heuristics, make_env
from uavwnn.fusion import criar_ambiente_ruidoso
from uavwnn.fusion.ruido import ConfigRuido

pytest.importorskip("gradysim")


def test_degrau1_sigma_zero_e_bit_identico_ao_pai():
    """ARQUITETURA §3.8: `σ = 0` ⇒ observação idêntica à do ambiente puro.

    Valida a reimplementação da observação — ranking, normalização, padding e a
    ordem das partes — contra o método de referência, no mesmo cenário.
    """
    cfg = LAMENZA_003(2, 12, phase="eval")
    random.seed(7); np.random.seed(7)
    o_ref, _ = make_env(cfg).reset()
    random.seed(7); np.random.seed(7)
    amb = criar_ambiente_ruidoso(cfg, degrau=1, config_ruido=ConfigRuido(sigma_gps=0.0))
    o_amb, _ = amb.reset()
    dif = max(float(np.abs(o_ref[a] - o_amb[a]).max()) for a in o_ref)
    assert dif == 0.0, f"σ=0 deveria reproduzir o pai; max|Δ| = {dif}"


def test_degrau0_delega_ao_pai():
    cfg = LAMENZA_003(2, 12, phase="eval")
    random.seed(3); np.random.seed(3)
    o_ref, _ = make_env(cfg).reset()
    random.seed(3); np.random.seed(3)
    amb = criar_ambiente_ruidoso(cfg, degrau=0, config_ruido=ConfigRuido(sigma_gps=10.0))
    o_amb, _ = amb.reset()
    dif = max(float(np.abs(o_ref[a] - o_amb[a]).max()) for a in o_ref)
    assert dif == 0.0, "degrau 0 não deve corromper nada"


def test_observacao_muda_com_o_ruido_nao_com_a_verdade():
    """RF-04, em espírito: a observação depende da MEDIDA, não da verdade.

    Mesmo cenário (mesma semente stdlib), sementes de ruído diferentes → a
    observação do degrau 1 muda. No degrau 0 (sem ruído) ela não muda.
    """
    cfg = LAMENZA_003(2, 12, phase="eval")

    def obs_com_ruido(indice, degrau):
        random.seed(5); np.random.seed(5)
        amb = criar_ambiente_ruidoso(cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=10.0))
        amb.indice_ruido = indice
        o, _ = amb.reset()
        return o["drone0"]

    d1_a, d1_b = obs_com_ruido(0, 1), obs_com_ruido(1, 1)
    assert not np.allclose(d1_a, d1_b), "ruído diferente deveria mudar a observação"
    d0_a, d0_b = obs_com_ruido(0, 0), obs_com_ruido(1, 0)
    assert np.allclose(d0_a, d0_b), "sem ruído, a semente de ruído é irrelevante"


def test_degrau4_filtra_e_so_ele_registra():
    """O degrau 4 reduz o erro de estimativa e popula o log; o degrau 1 não loga."""
    cfg = LAMENZA_003(1, 12, phase="eval")
    sigma = 10.0
    # o guloso de referência fica na arena; uma política de rumo fixo voaria na
    # parede, onde o comando deixa de bater com o movimento e o KF diverge — um
    # achado real (a borda quebra a hipótese "u conhecido"), mas ruim como guarda.
    greedy = load_heuristics().create_greedy_heuristics(0, 12)

    def rodar(degrau):
        random.seed(11); np.random.seed(11)
        amb = criar_ambiente_ruidoso(cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=sigma))
        obs, _ = amb.reset()
        for _ in range(200):
            obs, _, term, trunc, infos = amb.step({a: greedy(obs[a]) for a in amb.agents})
            if (infos and "avg_reward" in next(iter(infos.values()))) or any(term.values()) or any(trunc.values()):
                break
        return amb

    amb4 = rodar(4)
    amb1 = rodar(1)
    assert len(amb1.log_estimacao) == 0, "degrau 1 não tem filtro, não loga"
    assert len(amb4.log_estimacao) > 0, "degrau 4 registra a estimativa"
    erro = np.array([e["erro"] for e in amb4.log_estimacao])
    rmse = float(np.sqrt(np.mean(np.sum(erro**2, axis=1))))
    # o filtro tem de bater a medida crua (σ·√2 ≈ 14,1 m)
    assert rmse < sigma * np.sqrt(2), f"KF ({rmse:.1f} m) deveria filtrar abaixo do cru ({sigma*np.sqrt(2):.1f} m)"


@pytest.mark.parametrize("degrau", [1, 4])
def test_vizinho_recebe_a_posicao_divulgada_nao_a_verdadeira(degrau):
    """O drone `j` transmite a `i` a MESMA posição que usa para si (medida crua
    no degrau 1, estimativa no 4), não a verdade do simulador.

    Com Na = 2 a entrada de vizinho de `i` é `(p_i − p_j + S)/(2S)` e a de `j` é
    `(p_j − p_i + S)/(2S)`; se ambos usam as posições divulgadas, a soma é
    exatamente 1 em cada coordenada. Se `i` visse a verdade de `j` (o
    comportamento antigo), a soma carregaria `(z_j − p_j) + (z_i − p_i)` e não
    fecharia. O teste corre alguns passos para cobrir o filtro já iniciado.
    """
    cfg = LAMENZA_003(2, 12, phase="eval")
    random.seed(13); np.random.seed(13)
    amb = criar_ambiente_ruidoso(cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=10.0))
    obs, _ = amb.reset()
    n_viz = cfg.num_drones - 1  # state_num_closest_drones
    for _ in range(5):
        viz0, viz1 = obs["drone0"][: 2 * n_viz], obs["drone1"][: 2 * n_viz]
        assert np.allclose(viz0 + viz1, 1.0), (
            f"degrau {degrau}: vizinhos deveriam ver posições divulgadas "
            f"simétricas; soma = {viz0 + viz1}")
        obs, *_ = amb.step({a: np.array([0.25, 1.0]) for a in amb.agents})
