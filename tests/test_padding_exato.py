"""O sentinela de preenchimento chega EXATO — é o que autoriza `==`.

`Observacao.decodificar` e o `_valido` do termômetro comparam com `PADDING` por
igualdade, não por `np.isclose`. Isso é 17× mais barato (medido: 10,87 µs contra
0,64 µs num (24,2)) e mais estrito — a tolerância de 1e-5 do `isclose` aceitaria
um valor VÁLIDO próximo de −1.

Mas a correção não é uma propriedade do nosso código: depende de o ambiente
escrever o sentinela por atribuição e nunca fazer aritmética sobre ele. No
`environment.py` de referência isso é `fill(-1)` seguido de normalização aplicada
SÓ à fatia dos válidos (`[:len(sorted_sensor_indices)]`); no nosso
`ambiente_ruidoso.py`, `np.full(..., -1.0)` com o mesmo cuidado.

Se alguém normalizar o vetor inteiro, ou o ruído passar a somar sobre os slots
vazios, `==` deixa de casar e a política passa a tratar preenchimento como
sensor real — silenciosamente, porque `-1` decodifica para uma posição válida.
Este teste é o que faz esse dia doer aqui em vez de na campanha.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from uavwnn.agents.base import PADDING
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


@pytest.mark.parametrize("rotulo,degrau,sigma,extra", CASOS,
                         ids=[c[0] for c in CASOS])
def test_padding_chega_bit_exato(rotulo, degrau, sigma, extra):
    cfg = LAMENZA_003(num_drones=1, num_sensors=8)
    amb = _ambiente(cfg, degrau, sigma, extra)
    n_s = cfg.state_num_closest_sensors
    vistos = 0
    try:
        for s in range(3):
            random.seed(4_000_000 + s); np.random.seed(4_000_000 + s)
            obs, _ = amb.reset()
            for _ in range(40):
                for a in list(obs):
                    v = np.asarray(obs[a], float).ravel()
                    bs = v[:2 * n_s].reshape(n_s, 2)
                    vazio = np.isclose(bs, PADDING).all(axis=1)
                    vistos += int(vazio.sum())
                    if vazio.any():
                        # o que o `==` precisa: desvio EXATAMENTE zero, não ε
                        assert np.abs(bs[vazio] - PADDING).max() == 0.0, (
                            f"{rotulo}: preenchimento chegou perturbado — `==` "
                            f"deixaria de casar e o slot viraria sensor real")
                    assert np.array_equal(vazio, (bs == PADDING).all(axis=1)), (
                        f"{rotulo}: `==` divergiu de `np.isclose`")
                obs, _, term, trunc, _ = amb.step(
                    {a: amb.action_space(a).sample() for a in obs})
                if any(term.values()) or any(trunc.values()):
                    break
    finally:
        try:
            amb.close()
        except Exception:  # pragma: no cover
            pass
    # sem esta guarda o teste passaria sem ter visto um único slot vazio — foi
    # exatamente o que aconteceu na primeira versão desta verificação, com Ns=24
    assert vistos >= 100, f"{rotulo}: só {vistos} slots de preenchimento vistos"
