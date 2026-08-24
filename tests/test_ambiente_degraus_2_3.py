"""Degraus 2 (passa-baixa) e 3 (velocidade constante) no ambiente ruidoso.

Completam a escada de ablação da SPEC-08 no ambiente: a SPEC-07 ligou 0/1/4;
estes são os estimadores intermediários que o degrau 4 precisa bater. Guardas:
o degrau 2 (EMA) **não** tem `P` honesta e não loga `cov`; o degrau 3 (KF-VC)
loga `cov` 2×2 e filtra abaixo da medida crua. Exige o `gradysim` (submodule).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from uavwnn.refenv import LAMENZA_003, load_heuristics
from uavwnn.fusion import criar_ambiente_ruidoso
from uavwnn.fusion.ruido import ConfigRuido

pytest.importorskip("gradysim")


def _rodar(cfg, degrau, sigma, **kw):
    """Roda o guloso de referência na arena por um episódio e devolve o ambiente."""
    random.seed(11); np.random.seed(11)
    amb = criar_ambiente_ruidoso(
        cfg, degrau=degrau, config_ruido=ConfigRuido(sigma_gps=sigma), **kw)
    greedy = load_heuristics().create_greedy_heuristics(0, 12)
    obs, _ = amb.reset()
    for _ in range(200):
        obs, _, term, trunc, infos = amb.step({a: greedy(obs[a]) for a in amb.agents})
        if (infos and "avg_reward" in next(iter(infos.values()))) \
                or any(term.values()) or any(trunc.values()):
            break
    return amb


def test_degrau2_passabaixa_loga_erro_sem_cov():
    """Degrau 2 estima e registra o erro, mas não tem covariância (EMA)."""
    cfg = LAMENZA_003(1, 12, phase="eval")
    amb = _rodar(cfg, 2, 10.0, alpha=0.5)
    assert len(amb.log_estimacao) > 0, "degrau 2 deveria registrar a estimativa"
    assert all("erro" in e for e in amb.log_estimacao)
    assert all("cov" not in e for e in amb.log_estimacao), \
        "o passa-baixa não mantém P honesta — não deve logar cov"


def test_degrau3_velconstante_loga_cov_2x2_e_filtra():
    """Degrau 3 loga `cov` 2×2 em todo passo e filtra abaixo da medida crua."""
    cfg = LAMENZA_003(1, 12, phase="eval")
    sigma = 10.0
    amb = _rodar(cfg, 3, sigma, q_a=5.0)
    assert len(amb.log_estimacao) > 0
    covs = [np.array(e["cov"]) for e in amb.log_estimacao if "cov" in e]
    assert len(covs) == len(amb.log_estimacao), "degrau 3 loga cov em todo passo"
    assert covs[0].shape == (2, 2)
    erro = np.array([e["erro"] for e in amb.log_estimacao])
    rmse = float(np.sqrt(np.mean(np.sum(erro**2, axis=1))))
    assert rmse < sigma * np.sqrt(2), \
        f"KF-VC ({rmse:.1f} m) deveria filtrar abaixo do cru ({sigma*np.sqrt(2):.1f} m)"


def test_degrau2_variante_com_comando_difere():
    """A variante com termo de comando muda a estimativa (ablação honesta)."""
    cfg = LAMENZA_003(1, 12, phase="eval")
    a_sem = _rodar(cfg, 2, 10.0, alpha=0.5, usar_comando_pb=False)
    a_com = _rodar(cfg, 2, 10.0, alpha=0.5, usar_comando_pb=True)
    n = min(len(a_sem.log_estimacao), len(a_com.log_estimacao))
    assert n > 0
    e_sem = np.array([e["erro"] for e in a_sem.log_estimacao[:n]])
    e_com = np.array([e["erro"] for e in a_com.log_estimacao[:n]])
    assert not np.allclose(e_sem, e_com), \
        "o termo de comando deveria mudar a estimativa do passa-baixa"
