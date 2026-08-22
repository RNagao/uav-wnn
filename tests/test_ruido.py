"""Camada de ruído e CRN — SPEC-07 T-07.1.

O que faz a varredura `σ × T` caber no orçamento é o pareamento por números
aleatórios comuns: as células de `σ` diferente veem a **mesma** realização
normalizada. Isso só vale se o `ε` for independente de `σ`, e é o primeiro teste.
A independência entre canais é o que permite ligar um degrau sem mexer no ruído
dos outros.
"""

from __future__ import annotations

import numpy as np

from uavwnn.fusion.ruido import (
    CanalGPS,
    ConfigRuido,
    FilaMedidas,
    fluxo_ruido,
)


def test_crn_erro_escala_linearmente_com_sigma():
    """Mesma semente, `σ` diferente → `(z − p)` escala exatamente com `σ`.

    É a propriedade que pareia a varredura de `σ`: o `ε` não depende de `σ`.
    """
    p = np.array([10.0, -20.0])
    c1 = CanalGPS(ConfigRuido(sigma_gps=1.0), fluxo_ruido(42, "gps", 0, 0))
    c10 = CanalGPS(ConfigRuido(sigma_gps=10.0), fluxo_ruido(42, "gps", 0, 0))
    for k in range(20):
        e1 = c1.medir(p, t=float(k)) - p
        e10 = c10.medir(p, t=float(k)) - p
        assert np.allclose(e10, 10.0 * e1, atol=1e-12)


def test_fluxos_de_canais_sao_independentes():
    """Consumir o fluxo de GPS não altera o fluxo de broadcast (T-07.1)."""
    g = fluxo_ruido(1, "gps", 0, 0)
    b_antes = fluxo_ruido(1, "broadcast", 0, 0).standard_normal(5)
    _ = g.standard_normal(1000)  # consome o GPS à vontade
    b_depois = fluxo_ruido(1, "broadcast", 0, 0).standard_normal(5)
    assert np.array_equal(b_antes, b_depois)


def test_determinismo_mesma_semente_mesmo_fluxo():
    p = np.array([1.0, 2.0])
    a = CanalGPS(ConfigRuido(sigma_gps=5.0), fluxo_ruido(7, "gps", 3, 9))
    b = CanalGPS(ConfigRuido(sigma_gps=5.0), fluxo_ruido(7, "gps", 3, 9))
    za = [a.medir(p, float(k)) for k in range(10)]
    zb = [b.medir(p, float(k)) for k in range(10)]
    assert np.allclose(za, zb, atol=0.0)


def test_nos_e_episodios_diferentes_dao_fluxos_diferentes():
    p = np.zeros(2)
    z00 = CanalGPS(ConfigRuido(sigma_gps=5.0), fluxo_ruido(7, "gps", 0, 0)).medir(p, 0.0)
    z10 = CanalGPS(ConfigRuido(sigma_gps=5.0), fluxo_ruido(7, "gps", 1, 0)).medir(p, 0.0)
    z01 = CanalGPS(ConfigRuido(sigma_gps=5.0), fluxo_ruido(7, "gps", 0, 1)).medir(p, 0.0)
    assert not np.allclose(z00, z10)
    assert not np.allclose(z00, z01)


def test_branco_tem_variancia_marginal_sigma2():
    p = np.zeros(2)
    sigma = 4.0
    c = CanalGPS(ConfigRuido(sigma_gps=sigma), fluxo_ruido(11, "gps", 0, 0))
    erros = np.array([c.medir(p, float(k)) - p for k in range(20000)])
    assert abs(erros.std() - sigma) < 0.1


def test_gauss_markov_preserva_variancia_marginal_mas_correlaciona():
    """A variância marginal é `σ_gps²` (isola correlação, não magnitude), mas a
    autocorrelação em lag 1 é positiva — o que o modo branco não tem e o que o
    teste de brancura da inovação deve acusar."""
    p = np.zeros(2)
    sigma = 5.0
    cfg = ConfigRuido(sigma_gps=sigma, modelo="gauss_markov", tau=20.0, fracao_markov=0.7)
    c = CanalGPS(cfg, fluxo_ruido(3, "gps", 0, 0))
    erros = np.array([c.medir(p, float(k)) for k in range(40000)])
    assert abs(erros.std() - sigma) < 0.15, "variância marginal preservada"
    x = erros[:, 0] - erros[:, 0].mean()
    rho1 = float((x[:-1] * x[1:]).sum() / (x * x).sum())
    assert rho1 > 0.2, "componente lenta correlaciona a leitura"


def test_fila_entrega_em_ordem_de_emissao():
    fila = FilaMedidas(janela=3.0)
    cfg = ConfigRuido(sigma_gps=1.0, atraso_base=1.0)
    rng = np.random.default_rng(0)
    # emitidas fora de ordem de entrega; ambas dentro da janela em t=2.0
    fila.enviar("b", t_emissao=1.0, config=cfg, rng=rng)  # entrega 2.0
    fila.enviar("a", t_emissao=0.0, config=cfg, rng=rng)  # entrega 1.0
    assert fila.entregar_ate(2.0) == ["a", "b"], "ordena por carimbo de emissão"


def test_fila_descarta_o_que_e_velho_demais():
    """O filtro não reprocessa passado arbitrário: mais velho que `janela` cai."""
    fila = FilaMedidas(janela=1.0)
    fila.enviar("velha", t_emissao=0.0, config=ConfigRuido(sigma_gps=1.0),
                rng=np.random.default_rng(0))
    assert fila.entregar_ate(5.0) == []  # 5.0 − 0.0 = 5 > janela
    assert fila.descartadas == 1


def test_fila_perde_com_probabilidade():
    cfg = ConfigRuido(sigma_gps=1.0, p_perda=1.0)  # perde tudo
    fila = FilaMedidas()
    ok = fila.enviar("x", t_emissao=0.0, config=cfg, rng=np.random.default_rng(0))
    assert ok is False
    assert fila.entregar_ate(10.0) == []
