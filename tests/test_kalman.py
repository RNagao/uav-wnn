"""KF com entrada de controle e os degraus da escada — SPEC-07 T-07.2/09/10.

Testar filtro contra si mesmo é o modo clássico de não descobrir nada, então o
que decide aqui são **formas fechadas**: com `Q = 0` o filtro é a média amostral
(`P_k = σ²/k`), e o ganho estacionário tem raiz de Riccati explícita. Os testes
de sincronia e de taxa de `Q` cobrem os dois erros que a spec marca como os mais
prováveis.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.fusion.kalman import (
    ErroDeSincronia,
    FiltroKalmanControle,
    FiltroVelocidadeConstante,
    PassaBaixa,
    ganho_estacionario_escalar,
    _inv2x2,
)


def test_inversa_2x2_bate_com_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.normal(size=(2, 2))
        if abs(np.linalg.det(m)) < 1e-6:
            continue
        assert np.allclose(_inv2x2(m), np.linalg.inv(m), atol=1e-12)


def test_com_Q_zero_o_filtro_e_a_media_amostral():
    """`Q = 0`, `H = I`, `R = σ²I`, alvo estático, medida por passo → `P_k = σ²/k`.

    Forma fechada da T-07.2. `iniciar` já conta como a 1ª medida (`P = σ²I`), então
    após `N` atualizações `P = σ²/(N+1)`.
    """
    sigma = 5.0
    f = FiltroKalmanControle(sigma_gps=sigma, q_w=0.0, sigma_v0=0.0)
    f.iniciar(np.zeros(2), t0=0.0)
    for k in range(1, 30):
        f.atualizar(np.zeros(2), t=float(k))
        esperado = sigma**2 / (k + 1)
        assert np.allclose(f.covariancia, np.eye(2) * esperado, rtol=1e-9)


def test_media_amostral_converge_para_a_media_das_medidas():
    """Com `Q = 0` e alvo estático, `x̂` é a média das medidas, exatamente."""
    sigma = 3.0
    rng = np.random.default_rng(1)
    f = FiltroKalmanControle(sigma_gps=sigma, q_w=0.0, sigma_v0=0.0)
    zs = [rng.normal(scale=sigma, size=2) for _ in range(50)]
    f.iniciar(zs[0], t0=0.0)
    for k, z in enumerate(zs[1:], start=1):
        f.atualizar(z, t=float(k))
    assert np.allclose(f.media, np.mean(zs, axis=0), atol=1e-9)


def test_ganho_estacionario_bate_com_a_recursao():
    """`K∞` da forma fechada == ganho para onde a recursão de Riccati converge."""
    q, sigma2 = 0.5, 100.0
    p_inf, K_inf = ganho_estacionario_escalar(q, sigma2)
    # roda a recursão escalar até convergir
    p = sigma2
    for _ in range(10_000):
        p_pred = p + q
        K = p_pred / (p_pred + sigma2)
        p = (1 - K) * p_pred
    assert abs(p - p_inf) < 1e-9
    assert abs(K - K_inf) < 1e-9
    # e a raiz satisfaz p∞² + q·p∞ − q·σ² = 0
    assert abs(p_inf**2 + q * p_inf - q * sigma2) < 1e-9


def test_regra_de_taxa_x_igual_P_diferente():
    """A armadilha de taxa (T-07.2): 1 passo de 2 s vs 20 de 0,1 s.

    Com `F = I` e `u` constante, `x̂⁻` é idêntico nos dois. `P` não: o termo
    `q_w·dt` é invariante à taxa, mas `σ_v0²·dt²` não é — e é isso que obriga um
    `predizer_ate` por evento, não por tique.
    """
    u = np.array([15.0, 0.0])

    def roda(passos, dt):
        f = FiltroKalmanControle(sigma_gps=10.0, q_w=0.3, sigma_v0=0.5)
        f.iniciar(np.zeros(2), t0=0.0)
        f.comando(u)
        for k in range(1, passos + 1):
            f.predizer_ate(k * dt)
        return f.media.copy(), f.covariancia.copy()

    x_grosso, P_grosso = roda(1, 2.0)
    x_fino, P_fino = roda(20, 0.1)

    assert np.allclose(x_grosso, x_fino, atol=1e-9), "x̂ deve ser idêntico"
    # termo dt²: 1×(2²)=4 contra 20×(0.1²)=0.2 → fator 20 no termo de atuação
    dif = float(P_grosso[0, 0] - P_fino[0, 0])
    assert dif > 0, "o passo grosso acumula mais Q pelo termo dt²"
    # o termo linear q_w·dt é igual nos dois (0.3·2 = 20·0.3·0.1); a diferença
    # vem só do dt²: 0.5²·(4 − 0.2) = 0.95
    assert abs(dif - 0.5**2 * (4.0 - 0.2)) < 1e-9


def test_predizer_para_o_passado_levanta():
    f = FiltroKalmanControle(sigma_gps=1.0)
    f.iniciar(np.zeros(2), t0=5.0)
    with pytest.raises(ErroDeSincronia):
        f.predizer_ate(4.0)


def test_joseph_mantem_P_simetrica_e_positiva():
    """Mesmo com `Q` mal sintonizado (o regime das ablações), `P` fica válida."""
    rng = np.random.default_rng(2)
    f = FiltroKalmanControle(sigma_gps=10.0, q_w=50.0, sigma_v0=5.0)
    f.iniciar(np.zeros(2), t0=0.0)
    for k in range(1, 40):
        f.comando(rng.normal(scale=15, size=2))
        f.atualizar(rng.normal(scale=10, size=2), t=float(k))
        P = f.covariancia
        assert np.allclose(P, P.T, atol=1e-10), "simétrica"
        assert np.all(np.linalg.eigvalsh(P) > 0), "positiva definida"


def test_controle_bate_o_velocidade_constante_sob_virada_brusca():
    """Degrau 4 > degrau 3 quando o rumo muda a cada passo — o achado A3.

    O KF de velocidade constante trata cada virada como surpresa; o de controle
    sabe o comando. Sob comando i.i.d. o de controle tem de errar menos.
    """
    from uavwnn.fusion.bancada import rodar_filtro, trajetoria_b1
    from uavwnn.fusion.ruido import CanalGPS, ConfigRuido, fluxo_ruido

    sigma = 10.0
    erros_c, erros_vc = [], []
    for ep in range(20):
        traj = trajetoria_b1(np.random.default_rng(100 + ep), n_passos=120, sigma_salto=np.pi)
        canal_c = CanalGPS(ConfigRuido(sigma_gps=sigma), fluxo_ruido(7, "gps", 0, ep))
        canal_vc = CanalGPS(ConfigRuido(sigma_gps=sigma), fluxo_ruido(7, "gps", 0, ep))
        fc = FiltroKalmanControle(sigma_gps=sigma, q_w=0.5, sigma_v0=0.5)
        fvc = FiltroVelocidadeConstante(sigma_gps=sigma, q_a=5.0)
        erros_c.append(rodar_filtro(traj, canal_c, fc).rmse_filtrado)
        erros_vc.append(rodar_filtro(traj, canal_vc, fvc).rmse_filtrado)
    assert np.mean(erros_c) < np.mean(erros_vc), (
        f"controle {np.mean(erros_c):.2f} deveria bater VC {np.mean(erros_vc):.2f}"
    )
    # e os dois filtram: ambos abaixo do erro cru (σ·√2 ≈ 14,1)
    assert np.mean(erros_c) < 10.0


def test_passa_baixa_atrasa_alvo_movel():
    """Degrau 2 é enviesado: rastreando velocidade `v`, atraso `v·dt·(1−α)/α`.

    Sem termo de comando, o EMA fica permanentemente atrás de um alvo que anda.
    """
    v, dt, alpha = 15.0, 2.0, 0.5
    pb = PassaBaixa(alpha=alpha, usar_comando=False)
    pb.iniciar(np.zeros(2), t0=0.0)
    p = np.zeros(2)
    u = np.array([v, 0.0])
    for k in range(1, 200):
        p = p + u * dt          # verdade anda
        pb.comando(u)
        pb.atualizar(p, t=float(k * dt))  # medida limpa (sem ruído)
    atraso = float(p[0] - pb.media[0])
    esperado = v * dt * (1 - alpha) / alpha  # 30 m
    assert abs(atraso - esperado) < 1.0, f"atraso {atraso:.2f} vs esperado {esperado:.2f}"


def test_passa_baixa_com_comando_nao_atrasa():
    """A variante *com* comando remove o viés — isola 'usar o comando' de 'usar KF'."""
    v, dt, alpha = 15.0, 2.0, 0.5
    pb = PassaBaixa(alpha=alpha, usar_comando=True)
    pb.iniciar(np.zeros(2), t0=0.0)
    p = np.zeros(2)
    u = np.array([v, 0.0])
    for k in range(1, 100):
        p = p + u * dt
        pb.comando(u)
        pb.atualizar(p, t=float(k * dt))
    assert abs(float(p[0] - pb.media[0])) < 1e-6, "com comando e medida limpa, sem atraso"
