"""Teste diferencial do núcleo n-tuple contra o código Julia do Katopodis.

Este é o único teste do repositório que valida o **aproximador** contra um
oráculo externo. Os demais testes do `NTupleRegressor` verificam propriedades
que eu mesmo escolhi — RMSE numa regressão sintética, a identidade
`Δf = −η(f−y)` — e passariam mesmo se a semântica do modelo do Katopodis
tivesse sido entendida errado. Aqui não: os números do lado direito saíram do
`FunctionalDiscriminator` dele.

As fixtures em `tests/fixtures/diferencial/` são as predições produzidas por
`scripts/diferencial.jl` rodando `Ramnet.jl@faster_hashing`. Ficam congeladas
para que a suíte não exija Julia instalado; para regerá-las, veja o cabeçalho
de `scripts/spec04_diferencial.py`.

Tolerância de `1e-12`: os dois lados fazem as mesmas operações em `float64` na
mesma ordem, então o desvio admissível é o de reassociação em ponto flutuante.
O medido é ~5,6e-17. Uma divergência de semântica — soma em vez de média, sinal
do gradiente trocado, potências do endereço invertidas — apareceria em ordens
de grandeza acima disso, não perto do limiar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spec04_diferencial import CONFIGS, ETA, M_TREINO, _monta

FIXTURES = Path(__file__).parent / "fixtures" / "diferencial"


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c["nome"] for c in CONFIGS])
def test_equivale_ao_functional_discriminator(cfg):
    esperado = np.loadtxt(FIXTURES / f"{cfg['nome']}.txt", ndmin=2)

    net, bits, alvo, _ = _monta(cfg)
    net.treinar(bits[:M_TREINO], alvo[:M_TREINO], eta=ETA, epocas=1,
                rng=np.random.default_rng(3))
    obtido = net.prever(bits[M_TREINO:])

    assert obtido.shape == esperado.shape
    desvio = float(np.abs(obtido - esperado).max())
    assert desvio < 1e-12, f"divergência {desvio:.3e} do modelo do Katopodis"
    # Rede que só devolvesse zeros passaria no teste acima se a fixture também
    # fosse zero. Não é: o oráculo tem sinal de verdade.
    assert np.abs(esperado).max() > 0.1


def test_fixtures_cobrem_mais_de_uma_forma():
    """Uma configuração só poderia esconder erro dependente de `N` ou de `n`.

    As três variam tamanho de tupla, número de RAMs e número de saídas — é o
    que separa "concorda por acaso" de "implementa a mesma coisa".
    """
    formas = {(c["tamanho_tupla"], c["n_ram"], c["n_saidas"]) for c in CONFIGS}
    assert len(formas) == len(CONFIGS)
    assert len({c["n_saidas"] for c in CONFIGS}) > 1
