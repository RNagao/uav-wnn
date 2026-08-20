"""Algoritmos 4.1 e 4.2 — SARSA de `m` passos com esquecimento.

Não é o algoritmo do nosso problema (ação contínua usa o 4.4). Está aqui porque
é com ele que o capítulo 5 da dissertação produziu resultados publicados, e sem
ele não há como replicar nada.

Como no `test_actor_critic.py`, o laço não tem oráculo diferencial: os
invariantes do pseudocódigo são verificados um a um, e um teste de aprendizado
em CartPole fecha a ponta.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.encoding import CircularArtigo, TermometroArtigo
from uavwnn.rl import SarsaNTupla
from uavwnn.wnn import RedeRegressaoEsquecimento

gym = pytest.importorskip("gymnasium")

B, TUPLA, N_ACOES = 64, 8, 2


class EnvFalso:
    """Ambiente determinístico de `n` passos, recompensa 1 por passo.

    Simples de propósito: com γ = 1 e recompensa constante, o retorno de cada
    transição é conhecido em fórmula fechada, então o teste do retorno de `m`
    passos não depende de nenhuma dinâmica.
    """

    def __init__(self, n_passos: int) -> None:
        self.n_passos = n_passos
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(4, dtype=np.float64), {}

    def step(self, acao):
        self.t += 1
        return (np.full(4, self.t / 10.0), 1.0, self.t >= self.n_passos, False, {})


def _agente(phi=1.0, m=2, gamma=1.0, eps=0.0, seed=0, n_niveis=16):
    enc = TermometroArtigo(0.0, 1.0, n_niveis)
    n_bits = enc.n_bits_para(4)
    rede = RedeRegressaoEsquecimento(n_bits, TUPLA, n_saidas=N_ACOES,
                                     phi=phi, seed=seed)
    return SarsaNTupla(
        rede=rede,
        codificar=lambda o: enc.codificar(np.clip(o, 0.0, 0.999)[None, :]),
        n_acoes=N_ACOES, passos=m, desconto=gamma, exploracao=eps,
        rng=np.random.default_rng(seed),
    )


# --- Algoritmo 4.2: seleção de ação ---------------------------------------


def test_desempate_entre_maximos_e_uniforme():
    """Detalhe fácil de perder: "Sample **uniformly** from arg max".

    Com a rede zerada todas as ações valem 0. Desempatar pelo índice daria uma
    política inicial degenerada, sempre a ação 0 — e o agente nunca exploraria
    o suficiente para sair disso com ε = 0,005.
    """
    ag = _agente(eps=0.0)
    bits = np.zeros((1, ag.rede.n_bits), dtype=bool)
    assert np.array_equal(ag.valores(bits), np.zeros(N_ACOES))
    escolhas = [ag.escolher(bits) for _ in range(400)]
    assert set(escolhas) == {0, 1}
    assert 0.35 < np.mean(escolhas) < 0.65, "desempate deveria ser equilibrado"


def test_greedy_segue_o_maior_valor():
    ag = _agente(eps=0.0)
    bits = np.zeros((1, ag.rede.n_bits), dtype=bool)
    ag.rede.treinar_componente(bits, 1, 5.0)
    assert ag.valores(bits)[1] > ag.valores(bits)[0]
    assert all(ag.escolher(bits) == 1 for _ in range(50))


def test_exploracao_um_e_uniforme_e_zero_e_ganancioso():
    bits = np.zeros((1, _agente().rede.n_bits), dtype=bool)

    total = _agente(eps=1.0)
    total.rede.treinar_componente(bits, 0, 99.0)
    escolhas = [total.escolher(bits) for _ in range(400)]
    assert 0.35 < np.mean(escolhas) < 0.65, "ε = 1 deveria ignorar os valores"

    nenhuma = _agente(eps=0.0)
    nenhuma.rede.treinar_componente(bits, 0, 99.0)
    assert all(nenhuma.escolher(bits) == 0 for _ in range(50))


def test_parametros_invalidos_falham_cedo():
    with pytest.raises(ValueError, match="exploracao"):
        _agente(eps=1.5)
    with pytest.raises(ValueError, match="passos"):
        _agente(m=0)
    with pytest.raises(ValueError, match="ações"):
        enc = TermometroArtigo(0.0, 1.0, 16)
        SarsaNTupla(
            rede=RedeRegressaoEsquecimento(enc.n_bits_para(4), TUPLA, n_saidas=2),
            codificar=lambda o: enc.codificar(o[None, :]),
            n_acoes=3, passos=1, desconto=1.0, exploracao=0.0)


# --- Algoritmo 4.1: o laço ------------------------------------------------


def test_retorno_de_m_passos_com_bootstrap_on_policy():
    """`G = Σ γ^{i−τ−1} R_i`, mais `γ^m f(S_{τ+m}, A_{τ+m})` se `τ + m < T`.

    O bootstrap usa **a ação tomada**, não o máximo — é SARSA, não Q-learning.
    Com `φ = 1`, recompensa 1 por passo e `m = 2`, as duas últimas transições do
    episódio não têm bootstrap, então valem 2 e 1.
    """
    ag = _agente(phi=1.0, m=2, gamma=1.0, eps=0.0)
    env = EnvFalso(n_passos=6)
    assert ag.episodio(env, max_passos=10) == 6.0

    # A última transição (τ = T−1) tem G = R_T = 1; a penúltima, R + R = 2.
    S_ultimo = ag.codificar(np.full(4, 0.5))
    valores_finais = ag.valores(S_ultimo)
    assert valores_finais.max() > 0, "alguma componente deveria ter sido escrita"


def test_toda_transicao_do_episodio_e_atualizada():
    """O laço vai até `τ = T − 1`: a cauda não pode ficar sem atualização."""
    ag = _agente(phi=1.0, m=3, gamma=1.0, eps=0.0)
    env = EnvFalso(n_passos=5)
    assert ag.rede.entradas_ocupadas() == 0
    ag.episodio(env, max_passos=10)
    # 5 estados distintos visitados, cada um escrevendo em N RAMs
    assert ag.rede.entradas_ocupadas() >= ag.rede.n_ram


def test_so_a_componente_da_acao_e_escrita():
    """Alg. 4.1 atualiza `v[i](S_code, A)` — só o discriminador da ação tomada.

    É o que justifica o contador por componente: cada ação acumula as suas
    próprias visitas.
    """
    rede = RedeRegressaoEsquecimento(B, TUPLA, n_saidas=3, phi=0.9, seed=0)
    bits = np.random.default_rng(0).random((1, B)) > 0.5
    rede.treinar_componente(bits, 1, 7.0)
    v, c = next(iter(rede.memoria[0].values()))
    assert list(v) == [0.0, 7.0, 0.0]
    assert list(c) == [0.0, 1.0, 0.0]

    rede.treinar_componente(bits, 1, 3.0)
    v, c = next(iter(rede.memoria[0].values()))
    assert v[1] == pytest.approx(0.9 * 7.0 + 3.0)
    assert c[1] == 2.0
    assert v[0] == v[2] == 0.0 and c[0] == c[2] == 0.0


def test_indice_de_componente_invalido_falha_cedo():
    rede = RedeRegressaoEsquecimento(B, TUPLA, n_saidas=2, seed=0)
    bits = np.zeros((1, B), dtype=bool)
    with pytest.raises(ValueError, match="indice"):
        rede.treinar_componente(bits, 2, 1.0)


# --- aprende no CartPole --------------------------------------------------


def _cartpole(esquema, seed, episodios=100, res=128, tupla=64, m=500, eps=0.005):
    """Configuração do quadro de §5.2.3 (a): R=128, n=64, m=500, ε=0,005, φ=0,85.

    O recorte da observação em `[l, u]` é nosso: a eq. (3.24) pressupõe `x` no
    intervalo de definição, e fora dele o circular **dá a volta**, apelidando
    extremos em valores centrais. Sem recorte a comparação mediria essa
    patologia em vez do esquema de codificação.
    """
    faixa = (-1.5, 1.5)
    cls = CircularArtigo if esquema == "circular" else TermometroArtigo
    enc = cls(faixa[0], faixa[1], res)
    rede = RedeRegressaoEsquecimento(enc.n_bits_para(4), tupla, n_saidas=2,
                                     phi=0.85, seed=seed)
    ag = SarsaNTupla(
        rede=rede,
        codificar=lambda o: enc.codificar(
            np.clip(o, faixa[0], faixa[1] - 1e-6)[None, :]),
        n_acoes=2, passos=m, desconto=1.0, exploracao=eps,
        rng=np.random.default_rng(seed),
    )
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    retornos = [ag.episodio(env, max_passos=m) for _ in range(episodios)]
    env.close()
    return float(np.mean(retornos[:20])), float(np.mean(retornos[-20:]))


@pytest.mark.parametrize("esquema", ["simples", "circular"])
def test_aprende_cartpole(esquema):
    """Fecha a ponta: o Algoritmo 4.1 tem de aprender a tarefa do capítulo 5.

    Orçamento pequeno — o objetivo é detectar laço quebrado, não reproduzir as
    curvas do artigo. A replicação de §5.2.2 está em
    `scripts/replica_katopodis_52.py`.
    """
    inicio, fim = _cartpole(esquema, seed=0)
    assert fim > inicio * 1.5, f"{esquema}: não aprendeu ({inicio:.0f} -> {fim:.0f})"
    assert fim > 100, f"{esquema}: retorno final baixo ({fim:.0f})"


def test_circular_vence_termometro_em_baixa_resolucao():
    """T-B2 — replicação de §5.2.2: "the circular thermometer reaped better
    results than the simple one".

    Medido em `R = 32, n = 16`, onde o efeito é grande e consistente: ~65
    contra ~139 de retorno final, em todas as sementes, e a diferença **cresce**
    com mais episódios (não é falta de orçamento).

    Escolhemos a baixa resolução de propósito. Na configuração do artigo
    (`R = 128`) o efeito existe mas encolhe — com resolução alta o termômetro
    já tem discriminação de sobra, e o ganho do wrap-around importa menos.
    Isso refina a afirmação do texto em vez de contradizê-la, e está registrado
    em `reports/replica-katopodis-52.md`.
    """
    finais = {}
    for esquema in ("simples", "circular"):
        finais[esquema] = [
            _cartpole(esquema, seed=s, episodios=120, res=32, tupla=16,
                      m=200, eps=0.01)[1]
            for s in range(3)
        ]
    circ, simp = np.mean(finais["circular"]), np.mean(finais["simples"])
    assert circ > 1.4 * simp, f"circular {circ:.0f} vs simples {simp:.0f}"
    # e vence em TODAS as sementes, não só na média
    assert all(c > s for c, s in zip(finais["circular"], finais["simples"])), finais
