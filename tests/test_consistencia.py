"""NEES/NIS, faixa qui-quadrado e o detector — SPEC-07 T-07.6/07.

Um detector que nunca acusa não é evidência de nada (T-07.7): estes testes
constroem dados **consistentes por construção** (erro amostrado da própria `P`) e
verificam que o veredito é "consistente", depois injetam desintonias conhecidas
(`P` inflada/encolhida) e exigem que o detector as pegue.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from uavwnn.fusion.consistencia import (
    autocorrelacao,
    avaliar,
    faixa_chi2,
    limiar_brancura,
    nees_por_passo,
)


def test_faixa_chi2_bate_com_scipy():
    M, n, alpha = 100, 2, 0.05
    lo, hi = faixa_chi2(M, n, alpha)
    assert abs(lo - chi2.ppf(alpha / 2, M * n) / M) < 1e-12
    assert abs(hi - chi2.ppf(1 - alpha / 2, M * n) / M) < 1e-12
    assert lo < 2.0 < hi  # o valor esperado n=2 está na faixa


def test_nees_de_dados_consistentes_e_chi2():
    """Erro `e ~ N(0, P)` dá `ε = eᵀP⁻¹e ~ χ²(n)`, média ≈ n."""
    rng = np.random.default_rng(0)
    n = 2
    P = np.array([[4.0, 1.0], [1.0, 9.0]])
    L = np.linalg.cholesky(P)
    e = (L @ rng.standard_normal((n, 50000))).T
    covs = np.broadcast_to(P, (50000, 2, 2))
    eps = nees_por_passo(e, covs)
    assert abs(eps.mean() - n) < 0.05


def test_avaliar_declara_consistente_conservador_otimista():
    """A tabela de sintomas de T-07.6, com `P` deliberadamente errada."""
    rng = np.random.default_rng(1)
    M, K, n = 100, 30, 2
    P = np.eye(2) * 4.0
    L = np.linalg.cholesky(P)
    erros = np.einsum("ij,mkj->mki", L, rng.standard_normal((M, K, n)))

    # P correta → consistente
    covs = np.broadcast_to(P, (M, K, n, n))
    assert avaliar(erros, covs).veredito == "consistente"

    # P inflada 4× → filtro conservador (NEES abaixo da faixa)
    assert avaliar(erros, np.broadcast_to(4 * P, (M, K, n, n))).veredito == "conservador"

    # P encolhida 4× → filtro otimista (NEES acima da faixa) — o caso PERIGOSO
    assert avaliar(erros, np.broadcast_to(P / 4, (M, K, n, n))).veredito == "otimista"


def test_brancura_pega_correlacao():
    """Ruído branco fica dentro de `1,96/√K`; um AR(1) estoura o lag 1."""
    rng = np.random.default_rng(2)
    K = 4000
    branco = rng.standard_normal(K)
    rho = autocorrelacao(branco, max_lag=20)
    lim = limiar_brancura(K)
    assert np.mean(np.abs(rho) > lim) < 0.15, "branco majoritariamente dentro do envelope"

    ar = np.zeros(K)
    for k in range(1, K):
        ar[k] = 0.7 * ar[k - 1] + rng.standard_normal()
    rho_ar = autocorrelacao(ar, max_lag=20)
    assert rho_ar[0] > 5 * lim, "AR(1) tem ρ(1) muito acima do envelope"


def test_nees_no_filtro_real_fica_na_faixa():
    """O KF de controle bem sintonizado é consistente na bancada B1.

    É o critério de saída 1 de SPEC-07 em miniatura: NEES médio na faixa a 95 %
    para `σ = 10 m`. Mobilidade sem massa, `Q` pequeno — o filtro é quase exato.
    """
    from uavwnn.fusion.bancada import rodar_filtro, trajetoria_b1
    from uavwnn.fusion.kalman import FiltroKalmanControle
    from uavwnn.fusion.ruido import CanalGPS, ConfigRuido, fluxo_ruido

    sigma = 10.0
    M = 100
    erros, covs = [], []
    for ep in range(M):
        traj = trajetoria_b1(np.random.default_rng(500 + ep), n_passos=60, sigma_salto=np.pi)
        canal = CanalGPS(ConfigRuido(sigma_gps=sigma), fluxo_ruido(9, "gps", 0, ep))
        # Q = 0: na bancada B1 o movimento é exato (mobilidade sem massa, u
        # conhecido), então qualquer Q > 0 infla P e o filtro fica conservador —
        # exatamente a ressalva de T-07.5 sobre `Q` degenerado. Com Q = 0 o
        # filtro é a média recursiva perfeita e o NEES cai na faixa.
        f = FiltroKalmanControle(sigma_gps=sigma, q_w=0.0, sigma_v0=0.0)
        r = rodar_filtro(traj, canal, f)
        erros.append(r.erros)
        covs.append(r.covs)
    c = avaliar(np.array(erros), np.array(covs))
    assert c.passou(), f"NEES {c.nees_medio:.2f} fora da faixa {c.faixa}"
