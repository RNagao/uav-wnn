"""O sentinela de preenchimento é reconhecido — e chega exato, hoje.

`Observacao.decodificar` e o `_valido` do termômetro comparam com `PADDING` por
tolerância explícita de `TOL_PADDING = 1e-12`, não por `np.isclose` (7× mais
caro, com maquinaria de `rtol·|b|` que aqui não serve) nem por `==`.

DOIS TESTES, COM FORÇAS DIFERENTES DE PROPÓSITO:

`test_padding_reconhecido` é o CONTRATO. É o que o código precisa para estar
correto, e continua valendo se o ambiente um dia perturbar o sentinela em 1e-16.

`test_padding_chega_bit_exato` é um CANÁRIO. Documenta que hoje o desvio é
exatamente zero — `fill(-1)` no `environment.py:348` e `np.full(-1.0)` no nosso
`ambiente_ruidoso.py`, com a normalização tocando só a fatia dos válidos. Se ele
falhar sozinho, o código continua correto: alguém passou a fazer aritmética sobre
os slots vazios, e vale saber antes que a perturbação cresça além da tolerância.

O modo de falha que os dois previnem é silencioso: `−1` decodifica para a posição
válida `3S`, então um preenchimento não reconhecido vira um sensor fantasma e a
política persegue algo que não existe — sem erro, só resultado pior.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from uavwnn.agents.base import PADDING, TOL_PADDING
from uavwnn.refenv import LAMENZA_003, make_env

# `Na=1, Ns=8`: menos sensores que os 12 slots, então 4 ficam vazios desde o
# reset. Com `Ns=24` NENHUM slot fica vazio no início e o teste passaria vazio.
CASOS = [
    ("limpo", None, None, None),
    ("d1 σ=10", 1, 10.0, {}),
    ("d4 σ=10 (Kalman)", 4, 10.0, {"q_w": 4.0}),
]


def _ambiente(cfg, degrau, sigma, extra):
    if degrau is None:
        return make_env(cfg)
    from uavwnn.fusion import criar_ambiente_ruidoso
    from uavwnn.fusion.ruido import ConfigRuido
    return criar_ambiente_ruidoso(cfg, degrau=degrau,
                                  config_ruido=ConfigRuido(sigma_gps=sigma), **extra)


def _varrer(degrau, sigma, extra):
    """`(desvios dos slots vazios, nº de slots vistos)` num punhado de episódios."""
    cfg = LAMENZA_003(num_drones=1, num_sensors=8)
    amb = _ambiente(cfg, degrau, sigma, extra)
    n_s = cfg.state_num_closest_sensors
    desvios, vistos = [], 0
    try:
        for s in range(3):
            random.seed(4_000_000 + s); np.random.seed(4_000_000 + s)
            obs, _ = amb.reset()
            for _ in range(40):
                for a in list(obs):
                    v = np.asarray(obs[a], float).ravel()
                    bs = v[:2 * n_s].reshape(n_s, 2)
                    # a referência independente do que o código faz
                    vazio = np.isclose(bs, PADDING).all(axis=1)
                    vistos += int(vazio.sum())
                    if vazio.any():
                        desvios.append(np.abs(bs[vazio] - PADDING).max())
                obs, _, term, trunc, _ = amb.step(
                    {a: amb.action_space(a).sample() for a in obs})
                if any(term.values()) or any(trunc.values()):
                    break
    finally:
        try:
            amb.close()
        except Exception:  # pragma: no cover
            pass
    return desvios, vistos


@pytest.mark.parametrize("rotulo,degrau,sigma,extra", CASOS,
                         ids=[c[0] for c in CASOS])
def test_padding_reconhecido(rotulo, degrau, sigma, extra):
    """CONTRATO: o desvio cabe na tolerância que o código usa."""
    desvios, vistos = _varrer(degrau, sigma, extra)
    # sem esta guarda o teste passaria sem ter visto um único slot vazio — foi
    # exatamente o que aconteceu na primeira versão desta verificação, com Ns=24
    assert vistos >= 100, f"{rotulo}: só {vistos} slots de preenchimento vistos"
    pior = max(desvios)
    assert pior < TOL_PADDING, (
        f"{rotulo}: preenchimento desviou {pior:.2e}, acima de TOL_PADDING="
        f"{TOL_PADDING:.0e} — slots vazios viram sensores fantasma")


@pytest.mark.parametrize("rotulo,degrau,sigma,extra", CASOS,
                         ids=[c[0] for c in CASOS])
def test_padding_chega_bit_exato(rotulo, degrau, sigma, extra):
    """CANÁRIO: hoje o desvio é zero. Se cair, o código ainda está correto."""
    desvios, vistos = _varrer(degrau, sigma, extra)
    assert vistos >= 100
    pior = max(desvios)
    assert pior == 0.0, (
        f"{rotulo}: o sentinela passou a chegar com desvio {pior:.2e}. O código "
        f"segue correto (TOL_PADDING={TOL_PADDING:.0e} cobre), mas alguém passou "
        f"a fazer aritmética sobre os slots vazios — investigar antes que cresça")
