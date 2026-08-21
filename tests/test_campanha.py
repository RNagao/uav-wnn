"""Infraestrutura da campanha da SPEC-06 — §4, pré-registrada.

Estes testes guardam invariantes de **protocolo**, não de aritmética. Uma violação
aqui não produz erro: produz um resultado que parece válido e não é.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.campanha import (
    Banco,
    BancoPar,
    PapeisDeSemente,
    RECOMPENSAS,
    carregar_bancos,
    gerar_bancos,
    horizonte_permissivo,
)


# --- §4.2 bancos ----------------------------------------------------------


@pytest.mark.parametrize("na,ns", [(1, 12), (2, 12), (4, 24), (8, 36)])
def test_test_e_val_sao_disjuntos(na, ns):
    """A disjunção é o que impede a seleção de checkpoint de virar seleção sobre o
    conjunto de reporte — o problema de protocolo que a §4.2 separa de propósito."""
    par = gerar_bancos(na, ns)
    assert not (set(par.test.sementes) & set(par.val.sementes))
    assert len(par.test) == 500
    assert len(par.val) == 100


def test_bancopar_recusa_intersecao():
    """A verificação vive no construtor, não numa checagem opcional.

    Se ela fosse opcional, alguém montaria um par sem chamá-la, e o vazamento
    passaria — sem erro, com números plausíveis.
    """
    b = Banco("test", 1, 12, (1, 2, 3))
    with pytest.raises(ValueError, match="disjunção"):
        BancoPar(test=b, val=Banco("val", 1, 12, (3, 4, 5)))


def test_sementes_de_cenario_cabem_em_uint32():
    """O `reset()` de referência usa o gerador LEGADO do NumPy, que exige `< 2³²`.

    A SPeC-02 pagou este bug uma vez: sementes `uint64` estouravam
    `np.random.seed` e o cenário deixava de ser reprodutível.
    """
    par = gerar_bancos(8, 36)
    for banco in (par.test, par.val):
        assert all(0 <= s < 2 ** 32 for s in banco.sementes)


def test_bancos_sao_deterministicos_e_o_hash_detecta_mudanca():
    a, b = gerar_bancos(2, 12), gerar_bancos(2, 12)
    assert a.test.sementes == b.test.sementes
    assert a.test.hash == b.test.hash
    # a ordem faz parte do pareamento, então o hash tem de mudar com ela
    trocado = Banco("test", 2, 12, tuple(reversed(a.test.sementes)))
    assert trocado.hash != a.test.hash


def test_celulas_diferentes_dao_bancos_diferentes():
    """Cada célula tem cenário próprio; reaproveitar sementes entre células faria
    `Na = 2` e `Na = 4` compartilharem posições de sensores por acidente."""
    assert gerar_bancos(1, 12).test.sementes != gerar_bancos(2, 12).test.sementes
    assert gerar_bancos(1, 12).test.sementes != gerar_bancos(1, 24).test.sementes


def test_carregar_detecta_arquivo_adulterado(tmp_path):
    """Um banco editado à mão invalida toda comparação já registrada."""
    import json
    carregar_bancos(1, 12, raiz=tmp_path)
    arq = tmp_path / "bancos-a1s12.json"
    d = json.loads(arq.read_text())
    d["test"]["sementes"][0] += 1
    arq.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        carregar_bancos(1, 12, raiz=tmp_path)


# --- §4.1 sementes --------------------------------------------------------


def test_seed_eval_e_fixa_entre_repeticoes():
    """§4.1: idêntica para WNN, MLP, guloso, aleatório e k-means+TSP.

    Se variasse com `seed_train`, cada repetição veria cenários diferentes e o
    pareamento por episódio deixaria de existir.
    """
    assert len({PapeisDeSemente(seed_train=s).seed_eval for s in range(5)}) == 1


def test_seed_tuple_varia_com_seed_train_por_padrao():
    """D-06.1: o ladrilhamento é parte do modelo, não do ambiente.

    Fixá-lo reportaria como propriedade do método algo que pode ser propriedade de
    um sorteio.
    """
    tuplas = [PapeisDeSemente(seed_train=s).seed_tuple for s in range(5)]
    assert len(set(tuplas)) == 5


def test_corrida_de_controle_fixa_o_ladrilhamento():
    """A corrida da D-06.1 que quantifica quanto da variância vem das tuplas."""
    fixos = {PapeisDeSemente(seed_train=s, ladrilhamento_fixo=True).seed_tuple
             for s in range(5)}
    assert len(fixos) == 1


def test_seed_tuple_nunca_coincide_com_seed_train():
    """Usar o mesmo inteiro para o ladrilhamento e o ruído criaria correlação entre
    duas coisas que a §4.1 quer independentes dentro da corrida."""
    for s in range(20):
        p = PapeisDeSemente(seed_train=s)
        assert p.seed_tuple != p.seed_train


def test_rngs_de_politica_e_cenario_sao_independentes():
    """Se os cenários de treino saíssem do gerador do ruído, mudar o número de
    passos mudaria os cenários — e duas corridas com orçamentos diferentes
    deixariam de ser comparáveis."""
    p = PapeisDeSemente(seed_train=0)
    a = p.rng_politica().random(10)
    b = p.rng_cenarios_treino().random(10)
    assert not np.allclose(a, b)


# --- recompensas ----------------------------------------------------------


def test_eq_5_6_tem_o_bonus_de_dez():
    """O PRD §5.2 registra `Δvisited·10` na coleta; a fração restante é o `else`.

    Ler a equação só pelo `else` daria uma recompensa sem bônus, que é outra.
    """
    assert RECOMPENSAS["eq_5_6"](3, 2, 12) == 10.0
    assert RECOMPENSAS["eq_5_6"](5, 2, 12) == 30.0
    assert RECOMPENSAS["eq_5_6"](2, 2, 12) == pytest.approx(-10 / 12)


def test_eq_7_2_e_nao_negativa_e_nao_penaliza_demora():
    """É a diferença material com a 5-6, e é o que o §7.1 descreve.

    Com a 7-2 um episódio mais longo acumula MAIS retorno — *"reduces the urgency
    to complete the scenario as quickly as possible"*. Com a 5-6, menos.
    """
    f = RECOMPENSAS["eq_7_2"]
    assert f(0, 0, 12) == 0.0
    assert f(12, 12, 12) == 1.0
    assert all(f(k, k, 12) >= 0 for k in range(13))
    # e o retorno acumulado cresce com o número de passos
    assert sum(f(6, 6, 12) for _ in range(10)) > sum(f(6, 6, 12) for _ in range(5))


def test_as_duas_recompensas_nao_diferem_por_constante():
    """Um deslocamento constante seria absorvido pelo crítico; isto não é.

    Nos passos de coleta a 5-6 dá +10 e a 7-2 dá uma fração ≤ 1 — a diferença muda
    de sinal e de escala conforme o estado, então trocar uma pela outra troca o
    objetivo, não o zero da escala.
    """
    d_coleta = RECOMPENSAS["eq_5_6"](3, 2, 12) - RECOMPENSAS["eq_7_2"](3, 2, 12)
    d_idle = RECOMPENSAS["eq_5_6"](2, 2, 12) - RECOMPENSAS["eq_7_2"](2, 2, 12)
    assert abs(d_coleta - d_idle) > 1.0


# --- P5 -------------------------------------------------------------------


def test_horizonte_respeita_piso_e_teto():
    assert horizonte_permissivo(np.full(50, 10.0)) == 500.0     # piso
    assert horizonte_permissivo(np.full(50, 9999.0)) == 2000.0  # teto


def test_horizonte_arredonda_para_cima_em_centenas():
    """Arredondar para baixo poderia cair abaixo de `3 × p99` e deixar o horizonte
    de ser permissivo pela margem do arredondamento."""
    h = horizonte_permissivo(np.full(50, 250.0))     # 3 × 250 = 750
    assert h == 800.0
    assert h >= 3 * 250.0
