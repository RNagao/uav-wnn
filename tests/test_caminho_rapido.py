"""A via rápida de inferência é BIT A BIT igual à via lenta.

Três atalhos, três testes de identidade exata (não `allclose` — a via rápida só
tem direito de existir se for indistinguível):

  * `LUTPlana.consulta_do_vetor` funde termômetro + endereçamento + busca numa
    closure, sem materializar os bits;
  * `NormalProjetada.media_rapida` produz `media(f).como_array()` sem as duas
    dataclasses — com `np.arctan2`, não `math.atan2`: as duas libm divergem em
    1–2 ulp em ~7 % dos ângulos, medido, e a igualdade exata manda;
  * `AtorCriticoContinuo.act_vetor` pula a `Observacao` (o ator só lê `.vetor`).

O motivo de tudo: a decisão gastava ~30 despachos NumPy, e para vetores deste
tamanho o despacho domina — a MESMA rede do MADDPG em NumPy puro roda 2× mais
rápido que em torch pelo mesmo motivo. Medido pareado: 33,9 µs contra 70,7 µs da
via lenta e 76,4 µs do MADDPG.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uavwnn.agents.politica_continua import NormalProjetada

RAIZ = Path(__file__).resolve().parent.parent
DEPLOY = RAIZ / "models" / "deploy" / "T16" / "a1s12"

precisa_deploy = pytest.mark.skipif(
    not DEPLOY.exists(), reason="modelo de deploy não publicado neste checkout")


def test_media_rapida_bit_exata():
    rng = np.random.default_rng(11)
    casos = [rng.normal(0, 3, 3) for _ in range(50_000)]
    casos += [np.array([0.0, 0.0, 0.5]),      # rumo indefinido
              np.array([-1e-300, -1e-300, 1.5]),   # subnormal, s recortado
              np.array([1e300, -1e300, -0.5]),     # magnitudes extremas
              np.array([0.0, -1e-17, 0.0]),        # θ ≈ −0 → mod perto de 2π
              np.array([-0.0, 0.0, 1.0])]          # zeros com sinal
    pol = NormalProjetada(0.35, None)
    for f in casos:
        assert np.array_equal(pol.media(f).como_array(),
                              NormalProjetada.media_rapida(f)), f"divergiu em f={f}"


@precisa_deploy
def test_consulta_fundida_bit_exata():
    from uavwnn.encoding import TermometroQuantil
    from uavwnn.wnn.redes import LUTPlana, RedeGradienteFuncional

    termo = TermometroQuantil.carregar(DEPLOY / "termometro.npz")
    rede = RedeGradienteFuncional.carregar(
        next(DEPLOY.glob("*ator.npz")))
    plana = LUTPlana(rede)
    d = termo.n_bits // termo.n_niveis
    consulta = plana.consulta_do_vetor(
        np.repeat(np.arange(d), termo.n_niveis), termo.limiares.ravel())

    rng = np.random.default_rng(13)
    for _ in range(2_000):
        v = rng.uniform(-0.2, 1.2, d)
        # slots de preenchimento de verdade, aos pares como no ambiente
        if rng.random() < 0.5:
            k = int(rng.integers(0, d // 2)) * 2
            v[k:k + 2] = -1.0
        esperado = plana.prever(termo.codificar(v[None, :]))[0]
        assert np.array_equal(esperado, consulta(v))

    with pytest.raises(ValueError, match="entradas"):
        consulta(np.zeros(d + 1))  # a validação de tamanho não se perdeu


@precisa_deploy
def test_via_rapida_ponta_a_ponta():
    """O callable da campanha (via rápida) == decodificar + act (via lenta)."""
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    from spec06_campanha_wnn import montar
    from uavwnn.agents.base import ContextoEpisodio, Observacao
    from uavwnn.encoding import TermometroQuantil
    from uavwnn.refenv import LAMENZA_003

    cfg = LAMENZA_003(num_drones=1, num_sensors=12)
    pol, ap = montar("ntupla_dissertacao", cfg, DEPLOY, plana=True)
    ctx = ContextoEpisodio(
        num_drones=1, num_sensores=12, scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval)
    termo = TermometroQuantil.carregar(DEPLOY / "termometro.npz")
    npj = NormalProjetada(0.35, None)

    def lenta(observacao):
        o = Observacao.decodificar(observacao, ctx)
        f = ap.f(termo.codificar(np.asarray(o.vetor, float)[None, :])[0])
        return npj.media(f).como_array()

    rng = np.random.default_rng(17)
    for _ in range(400):
        v = rng.uniform(0.0, 1.0, cfg.observation_size)
        if rng.random() < 0.4:
            k = 2 * int(rng.integers(0, 12))
            v[k:k + 2] = -1.0
        assert np.array_equal(pol(v), lenta(v))
