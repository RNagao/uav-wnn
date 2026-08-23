"""Guarda a fronteira entre os dois ambientes — SPEC-06, treino por PPO.

`scripts/spec06_ppo_anbubus.py` roda num ambiente efêmero que **não pode importar
`uavwnn`**, porque `ray[rllib]` fixa `gymnasium==1.2.2` no máximo e este projeto
exige `>=1.3.0`. São incompatíveis, medido.

A consequência é uma **duplicação inevitável**: os parâmetros do ambiente estão
escritos nos dois lados. Se eles divergirem, o treino e a avaliação deixam de ser o
mesmo problema — e o sintoma seria uma política que "não transfere", sem nenhum
erro. É o tipo de divergência que só um teste pega.

`torch` e `ray` **não** são necessários aqui: o que se compara são dicionários.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from uavwnn.refenv import LAMENZA_003

RAIZ = Path(__file__).resolve().parents[1]


def _script():
    """Importa o treinador sem executá-lo, e sem exigir ray/torch."""
    caminho = RAIZ / "scripts" / "spec06_ppo_anbubus.py"
    spec = importlib.util.spec_from_file_location("_ppo_anbubus", caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ppo_anbubus"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_env_treino_bate_com_o_refenv():
    """Os parâmetros duplicados têm de ser **idênticos** aos do `refenv`.

    Compara contra `phase="train"`, que é a fase do `benchmark.py`
    (`--no-end-when-all-collected`) e a que o treino deve usar — treinar na
    configuração de avaliação daria um sinal de recompensa diferente do que Lamenza
    §5 descreve.
    """
    mod = _script()
    for na, ns in ((2, 12), (8, 36)):
        esperado = LAMENZA_003(num_drones=na, num_sensors=ns, phase="train").kwargs()
        obtido = dict(mod.ENV_TREINO, num_drones=na, num_sensors=ns,
                      state_num_closest_drones=na - 1)
        assert obtido == esperado, (
            f"divergência em Na={na} Ns={ns}: "
            f"{ {k: (obtido.get(k), esperado.get(k))
                 for k in set(obtido) | set(esperado)
                 if obtido.get(k) != esperado.get(k)} }")


def test_treina_na_fase_certa():
    """`end_when_all_collected=False` — o episódio não acaba na coleta completa.

    É o `--no-end-when-all-collected` do `benchmark.py`, e é o que dá o *"longer
    duration positive reward signal"* da §5. Treinar com `True` mudaria a
    recompensa efetiva sem mudar a eq. 5-6, o que é a pior forma de mudar o
    problema: invisível na fórmula.
    """
    assert _script().ENV_TREINO["end_when_all_collected"] is False


def test_faixa_da_observacao_e_a_medida_nao_a_declarada():
    """O ambiente declara `Box(-1, 1)` e emite até **1,50**. Medido.

    A normalização é `(d + S)/(2S)` com `S = 100`, o que dá 1,5 quando `d = 200` —
    os drones **saem** da caixa de 100 × 100 e a normalização supõe que não saem.

    Um termômetro de faixa fixa em `[-1, 1]` saturaria tudo acima de 1,0, perdendo a
    distinção entre "longe" e "muito longe" justamente onde ela importa para
    reaproximar o enxame. O `FAIXA_OBS` do script existe por isso.
    """
    lo, hi = _script().FAIXA_OBS
    assert lo == -1.0
    assert hi >= 1.5, f"faixa superior {hi} não cobre o 1,50 medido"


@pytest.mark.parametrize("celula", [(2, 12), (4, 24), (8, 36)])
def test_hiperparametros_sao_os_do_anbubus(celula):
    """Os valores vêm das configurações deles, e o teste os fixa como citação.

    Se alguém "melhorar" um deles, o braço PPO deixa de ser a reprodução do artigo
    Neurocomputing e passa a ser a nossa sintonia — o que é legítimo, mas tem de
    ser decisão registrada, não deriva.
    """
    mod = _script()
    # Fixa as DUAS tarefas contínuas, não uma: é a citação completa de
    # `wnn/{hopper,reacher}/thermo/individual_run/experiment.py`, e é o que a grade
    # varre. Pinar só a base deixaria a outra livre para derivar.
    assert mod.REFERENCIAS["hopper"] == dict(
        lr=4e-4, gamma=0.98, lambda_=1.0, train_batch_size=1000,
        minibatch_size=512, tuple_size=8, resolucao=32, vf_clip_param=3000.0)
    assert mod.REFERENCIAS["reacher"] == dict(
        lr=7e-5, gamma=0.99, lambda_=0.9, train_batch_size=1000,
        minibatch_size=128, tuple_size=10, resolucao=300, vf_clip_param=10.0)

    # A BASE é o reacher, e a escolha está registrada: é a tarefa contínua com
    # γ = 0,99, o desconto que Lamenza fixa, e a única com λ declarado.
    assert mod.ANBUBUS == mod.REFERENCIAS["reacher"]
    assert mod.TUPLA_ANBUBUS == 10
    assert mod.RESOLUCAO_ANBUBUS == 300

    # `num_sgd_iter = 1` é INVARIANTE nas oito configurações publicadas, então não
    # é eixo nem escolha: fica embutido em `construir`.
    assert "num_epochs" not in mod.ANBUBUS


def test_o_treinador_nao_importa_uavwnn():
    """A fronteira dos dois ambientes, verificada no texto do script.

    Um `import uavwnn` reintroduziria a restrição `gymnasium>=1.3.0` e o `ray`
    deixaria de resolver — falha de instalação, não de execução, o que é pior de
    diagnosticar.
    """
    texto = (RAIZ / "scripts" / "spec06_ppo_anbubus.py").read_text()
    linhas = [l.strip() for l in texto.splitlines()
              if l.strip().startswith(("import ", "from "))]
    culpados = [l for l in linhas if "uavwnn" in l]
    assert not culpados, f"o treinador importa uavwnn: {culpados}"
