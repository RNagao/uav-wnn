"""TOST (equivalência por bootstrap pareado) — SPEC-08 T-08.2.

Guardas: amostras (quase) idênticas são equivalentes dentro de ±δ; uma diferença
grande não é; e uma amostra ruidosa demais é marcada **não informativa**
(meia-largura > δ) — o modo de falha que a spec exige distinguir de "equivalente".
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.eval.stats import tost


def test_amostras_quase_identicas_sao_equivalentes():
    rng = np.random.default_rng(0)
    a = rng.normal(50.0, 5.0, 500)
    b = a + rng.normal(0.0, 0.01, 500)      # pareado, praticamente igual
    eq = tost(a, b, delta=1.0, rng=np.random.default_rng(1))
    assert eq.equivalente
    assert eq.informativa
    assert abs(eq.delta_media) < 1.0
    assert -1.0 <= eq.ic_lo and eq.ic_hi <= 1.0


def test_diferenca_grande_nao_e_equivalente():
    rng = np.random.default_rng(2)
    a = rng.normal(50.0, 3.0, 500)
    b = a + 5.0                              # dif = −5, |dif| ≫ δ
    eq = tost(a, b, delta=1.0, rng=np.random.default_rng(3))
    assert not eq.equivalente
    assert eq.p_inferior > 0.9              # quase toda réplica cai ≤ −δ


def test_amostra_ruidosa_e_nao_informativa():
    """Dif média ~0, mas variância alta e n pequeno → IC mais largo que ±δ."""
    a = np.random.default_rng(4).normal(50.0, 20.0, 8)
    b = np.random.default_rng(40).normal(50.0, 20.0, 8)
    eq = tost(a, b, delta=1.0, rng=np.random.default_rng(5))
    assert not eq.informativa               # meia_largura > δ = falta de poder
    assert eq.meia_largura > eq.delta


def test_delta_nao_positivo_levanta():
    rng = np.random.default_rng(6)
    a = np.arange(10.0)
    with pytest.raises(ValueError):
        tost(a, a, delta=0.0, rng=rng)


def test_pareamento_exige_mesmo_tamanho():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        tost(np.arange(10.0), np.arange(9.0), delta=1.0, rng=rng)


def test_como_linha_serializa_os_campos():
    rng = np.random.default_rng(8)
    a = rng.normal(0.0, 1.0, 100)
    b = rng.normal(0.0, 1.0, 100)
    linha = tost(a, b, delta=0.5, rng=np.random.default_rng(9)).como_linha()
    for k in ("delta", "delta_media", "ic_lo", "ic_hi", "meia_largura",
              "equivalente", "informativa", "p_inferior", "p_superior"):
        assert k in linha
