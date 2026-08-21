"""T-05.5 item 3 e critério de saída 7 — determinismo com `Na > 1`.

> Determinismo: duas execuções com a mesma semente produzem tabelas idênticas bit
> a bit, **inclusive com `Na > 1` e ordens de evento diferentes**.

ESTE MÓDULO CORRIGE A REDAÇÃO DESSE CRITÉRIO, com medição.

A barreira da §4.7 garante que todos os UAVs leiam `f` do **mesmo instantâneo** da
tabela antes de qualquer escrita. Isso é necessário, e é o que impede que o rumo do
agente 2 dependa de o agente 1 já ter atualizado. Mas **não é suficiente** para
invariância à ordem, e a razão é aritmética:

    ator      `v ← v + passo`             soma, COMUTA
    crítico   `v ← φ·v + alvo`            NÃO comuta

Com dois agentes escrevendo na mesma posição no mesmo tique:

    (φ·v + a₁)·φ + a₂  ≠  (φ·v + a₂)·φ + a₁      salvo a₁ = a₂

Medido: o ator sai idêntico bit a bit em qualquer ordem, sempre; o crítico sai
idêntico **só quando não há colisão de endereço**.

E A COLISÃO É O CASO COMUM, NÃO O DE CANTO

O canal de identidade são `R·Na` bits num padrão de ~800, então a **maioria** das
RAMs não lê bit de índice nenhum (medido em T-05.5: com `Na = 4`, `R = 1`, apenas
18 % das RAMs leem o canal). Nessas RAMs, dois agentes com a mesma observação
posicional endereçam exatamente a mesma posição. Colisão é o regime normal.

CONCLUSÃO REGISTRADA

Invariância à ordem é **inatingível** para o crítico com esquecimento, a menos que
se acumulem os alvos por endereço dentro do tique e se aplique `φ` uma vez só — o
que muda a semântica da eq. (4.8) de "por observação" para "por tique", e é desvio
do método, não detalhe de implementação.

O que este projeto entrega é **determinismo**, que é o que `make repro` precisa: a
ordem de aplicação é canônica (índice do agente crescente), então a mesma semente
dá a mesma tabela. A distinção entre *determinístico* e *invariante à ordem* fica
escrita, em vez de o critério ser declarado satisfeito por leitura generosa.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.wnn.redes import RedeGradienteFuncional, RedeRegressaoEsquecimento

N_BITS, TUPLA, PHI = 64, 8, 0.85


def _tabelas_iguais(x, y) -> bool:
    """Igualdade **bit a bit**, não por tolerância: `array_equal`, não `allclose`."""
    if len(x.memoria) != len(y.memoria):
        return False
    for ma, mb in zip(x.memoria, y.memoria):
        if ma.keys() != mb.keys():
            return False
        for chave in ma:
            va, vb = ma[chave], mb[chave]
            if isinstance(va, tuple):                 # crítico: (valor, contador)
                if not all(np.array_equal(p, q) for p, q in zip(va, vb)):
                    return False
            elif not np.array_equal(va, vb):
                return False
    return True


def _tique(ordem: list[int], colisao: bool, tiques: int = 5):
    """Simula `tiques` decisões de 2 agentes com barreira, aplicando em `ordem`.

    A barreira está nas duas linhas de `lidos`: **todas** as leituras acontecem
    antes de **qualquer** escrita. Sem isso, o `f` do segundo agente já veria a
    atualização do primeiro e a ordem mudaria até o comportamento, não só a tabela.
    """
    ator = RedeGradienteFuncional(N_BITS, TUPLA, n_saidas=3, seed=0)
    critico = RedeRegressaoEsquecimento(N_BITS, TUPLA, n_saidas=1, phi=PHI, seed=0)
    base = np.random.default_rng(0).random(N_BITS) > 0.5

    for _ in range(tiques):
        entradas = [base.copy(), base.copy()] if colisao else [base.copy(), ~base]
        lidos = [(e, ator.prever(e[None, :])[0]) for e in entradas]   # BARREIRA
        for i in ordem:
            e, _ = lidos[i]
            ator.atualizar(e, np.full(3, 0.01 * (i + 1)))
            critico.treinar(e[None, :], np.array([float(i + 1)]))
    return ator, critico


def test_o_ator_e_invariante_a_ordem_porque_a_soma_comuta():
    """`v ← v + passo` comuta, então a ordem não pode importar — nem com colisão."""
    for colisao in (False, True):
        a1, _ = _tique([0, 1], colisao)
        a2, _ = _tique([1, 0], colisao)
        assert _tabelas_iguais(a1, a2), f"ator dependeu da ordem (colisão={colisao})"


def test_o_critico_e_invariante_a_ordem_apenas_sem_colisao():
    """Sem endereço compartilhado, `φ` age em posições disjuntas e a ordem não
    importa."""
    _, c1 = _tique([0, 1], colisao=False)
    _, c2 = _tique([1, 0], colisao=False)
    assert _tabelas_iguais(c1, c2)


def test_o_critico_NAO_e_invariante_a_ordem_sob_colisao():
    """**O achado.** `(φv + a₁)φ + a₂ ≠ (φv + a₂)φ + a₁`.

    Este teste afirma a desigualdade de propósito. Se algum dia ele passar a
    falhar, alguém mudou a semântica do esquecimento — e isso precisa ser uma
    decisão, não um efeito colateral.
    """
    _, c1 = _tique([0, 1], colisao=True)
    _, c2 = _tique([1, 0], colisao=True)
    assert not _tabelas_iguais(c1, c2), (
        "o crítico ficou invariante à ordem sob colisão; a eq. (4.8) deixou de "
        "ser aplicada por observação")


def test_a_nao_comutatividade_e_exatamente_a_do_esquecimento():
    """Fecha a conta na aritmética, para o achado não depender do simulador.

    Duas escritas na mesma posição, alvos `a₁` e `a₂`, partindo de virgem:
    a ordem `(1,2)` dá `φ·a₁ + a₂` e a ordem `(2,1)` dá `φ·a₂ + a₁`. A diferença
    é `(φ−1)(a₁−a₂)`, que é zero se `φ = 1` (sem esquecimento) ou `a₁ = a₂`.
    """
    bits = np.zeros(N_BITS, bool)
    bits[:TUPLA] = True
    a1, a2 = 3.0, 7.0

    def valor(primeiro, segundo, phi):
        c = RedeRegressaoEsquecimento(N_BITS, TUPLA, n_saidas=1, phi=phi, seed=0)
        c.treinar(bits[None, :], np.array([primeiro]))
        c.treinar(bits[None, :], np.array([segundo]))
        return c.memoria[0][next(iter(c.memoria[0]))][0][0]

    v12, v21 = valor(a1, a2, PHI), valor(a2, a1, PHI)
    assert v12 == pytest.approx(PHI * a1 + a2)
    assert v21 == pytest.approx(PHI * a2 + a1)
    assert v12 - v21 == pytest.approx((PHI - 1) * (a1 - a2))

    # `φ = 1` recupera a comutatividade — e recupera também a eq. (3.16).
    assert valor(a1, a2, 1.0) == pytest.approx(valor(a2, a1, 1.0))


def test_determinismo_e_o_que_o_projeto_entrega():
    """Critério 7 na leitura defensável: **mesma ordem canônica, mesma tabela.**

    Ordem canônica = índice do agente crescente. É o que `make repro` precisa, e é
    atingível; invariância à ordem não é, pelo teste acima.
    """
    for colisao in (False, True):
        a1, c1 = _tique([0, 1], colisao)
        a2, c2 = _tique([0, 1], colisao)
        assert _tabelas_iguais(a1, a2) and _tabelas_iguais(c1, c2)


def test_a_barreira_e_necessaria_para_o_comportamento_nao_so_para_a_tabela():
    """Sem barreira, o `f` do segundo agente já vê a escrita do primeiro.

    Aí a ordem muda a **ação emitida**, não apenas a tabela final — o que é uma
    falha qualitativamente pior, porque contamina a trajetória e nenhuma
    verificação de tabela a pegaria.
    """
    bits = np.zeros(N_BITS, bool)
    bits[:TUPLA] = True

    rede = RedeGradienteFuncional(N_BITS, TUPLA, n_saidas=3, seed=0)
    com_barreira = [rede.prever(bits[None, :])[0].copy() for _ in range(2)]
    rede.atualizar(bits, np.full(3, 0.5))
    np.testing.assert_array_equal(com_barreira[0], com_barreira[1])

    rede2 = RedeGradienteFuncional(N_BITS, TUPLA, n_saidas=3, seed=0)
    primeiro = rede2.prever(bits[None, :])[0].copy()
    rede2.atualizar(bits, np.full(3, 0.5))              # escrita ANTES da 2ª leitura
    segundo = rede2.prever(bits[None, :])[0]
    assert not np.array_equal(primeiro, segundo), (
        "sem barreira as duas leituras coincidiram; o teste perdeu o seu objeto")
