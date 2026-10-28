"""CTDE — crítico centralizado n-tuple (`uavwnn.agents.ctde`): guard rails.

O que pode quebrar em silêncio, e o teste que pega cada caso:

1. **O crítico central não vê todos os drones.** Guarda: a entrada tem exatamente
   `Na·(bits_obs + bits_ação)` bits, na ordem dos drones, e trocar a ação do
   drone `i` muda só a fatia dele.
2. **O ator mudou.** Guarda: o ator continua lendo `bits_obs` do próprio drone e
   o checkpoint de deploy continua só com `ator.npz` + `termometro.npz` — o
   crítico central tem nome próprio e NÃO casa com o glob `*critico.npz` que a
   retomada do crítico descentralizado usa.
3. **O contrafactual não atribui crédito.** Guarda: num time de dois drones em que
   a recompensa depende SÓ da ação do drone 0, a vantagem do drone 1 tem de ser
   pequena em relação à do drone 0 com `K > 0`, e igual à dele com `K = 0`. O
   ator fica CONGELADO no brinquedo — mede-se o crítico, não a política.
4. **Ida e volta do checkpoint.** Guarda: salvar e carregar devolve o mesmo `Q`.
5. **A base depende da própria ação.** Guarda: `K = 0` só com crítico de estado,
   `K > 0` só com crítico de ação — `G − Q_c(s, a)` não é vantagem.
6. **Extremos da velocidade.** Guarda: velocidade 0 e 1 têm bits distintos no
   crítico (termômetro monotônico).
"""
from __future__ import annotations

import numpy as np
import pytest

from uavwnn.agents import NTuplaDissertacaoAprox
from uavwnn.agents.ctde import D_ACAO, CriticoCentral, TimeAtorCritico


def _aprox(n_bits: int, passo: float = 2e-3) -> NTuplaDissertacaoAprox:
    return NTuplaDissertacaoAprox(n_bits=n_bits, tamanho_tupla=4, phi=0.85, taxa_ator=passo,
                                  seed=0, c_max=None)


def test_entrada_do_critico_central_e_a_concatenacao_de_todos_os_drones():
    na, b, R = 3, 24, 8
    c = CriticoCentral(n_bits_obs=b, na=na, tamanho_tupla=4, resolucao_acao=R)
    assert c.n_bits_acao == D_ACAO * R
    assert c.n_bits == na * (b + D_ACAO * R)
    rng = np.random.default_rng(0)
    xs = [rng.random(b) > 0.5 for _ in range(na)]
    acoes = [np.array([0.1, 0.9]), np.array([0.5, 0.5]), np.array([0.9, 0.1])]
    bits = c.bits(xs, acoes)
    assert bits.shape == (c.n_bits,)
    # as fatias de observação são as observações, na ordem
    for i in range(na):
        assert np.array_equal(bits[i * b:(i + 1) * b], xs[i])
    # trocar a ação do drone 1 muda SÓ a fatia de ação do drone 1
    novo = c.bits_substituindo(bits, 1, np.array([0.0, 1.0]))
    dif = np.flatnonzero(bits != novo)
    ini = na * b + 1 * c.n_bits_acao
    assert dif.size > 0 and dif.min() >= ini and dif.max() < ini + c.n_bits_acao
    with pytest.raises(ValueError):
        c.bits(xs[:2], acoes[:2])
    # crítico de ESTADO: só observações
    v = CriticoCentral(n_bits_obs=b, na=na, tamanho_tupla=4, com_acoes=False)
    assert v.n_bits == na * b and v.n_bits_acao == 0
    assert np.array_equal(v.bits(xs), np.concatenate(xs))
    with pytest.raises(RuntimeError):
        v.bits_substituindo(v.bits(xs), 0, acoes[0])


def test_velocidade_maxima_nao_vira_velocidade_zero():
    c = CriticoCentral(n_bits_obs=4, na=1, tamanho_tupla=4, resolucao_acao=8)
    x = [np.zeros(4, bool)]
    b0 = c.bits(x, [np.array([0.3, 0.0])])
    b1 = c.bits(x, [np.array([0.3, 1.0])])
    assert not np.array_equal(b0, b1)


def test_o_ator_nao_muda_e_o_deploy_nao_leva_o_critico_central(tmp_path):
    b = 24
    ap = _aprox(b)
    c = CriticoCentral(n_bits_obs=b, na=2, tamanho_tupla=4)
    # o ator lê os bits do PRÓPRIO drone, não os conjuntos
    x = np.zeros(b, bool); x[::3] = True
    assert ap.f(x).shape == (3,)
    with pytest.raises(Exception):
        ap.f(np.zeros(c.n_bits, bool))
    # checkpoint: o crítico central tem nome próprio e não casa com `*critico.npz`
    ap.salvar(tmp_path); c.salvar(tmp_path)
    nomes = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert "critico-central.npz" in nomes
    assert [p.name for p in tmp_path.glob("*critico.npz")] == ["ntupla_dissertacao-critico.npz"]
    assert [p.name for p in tmp_path.glob("*ator.npz")] == ["ntupla_dissertacao-ator.npz"]


def test_salvar_e_carregar_devolve_o_mesmo_q(tmp_path):
    b, na = 16, 2
    c = CriticoCentral(n_bits_obs=b, na=na, tamanho_tupla=4, phi=0.85, resolucao_acao=8)
    rng = np.random.default_rng(1)
    amostras = [(c.bits([rng.random(b) > 0.5 for _ in range(na)], rng.random((na, 2))), rng.normal())
                for _ in range(50)]
    for bits, g in amostras:
        c.treinar(bits, g)
    c.salvar(tmp_path)
    c2 = CriticoCentral.carregar(tmp_path, n_bits_obs=b, na=na, resolucao_acao=8)
    for bits, _ in amostras[:10]:
        assert c2.q(bits) == pytest.approx(c.q(bits))
    with pytest.raises(ValueError):
        CriticoCentral.carregar(tmp_path, n_bits_obs=b, na=na + 1, resolucao_acao=8)


class _AtorCongelado:
    """Ator fixo (`f = (0, 0, ½)`): o brinquedo mede o CRÍTICO, não a política."""

    def __init__(self):
        self.direcoes = []

    def f(self, x):
        return np.array([0.0, 0.0, 0.5])

    def ascender_ator(self, x, direcao):
        self.direcoes.append(np.asarray(direcao, float))


def _time(na, b, K, sigma=0.35, phi=1.0, n_tupla=8):
    ap = _AtorCongelado()
    c = CriticoCentral(n_bits_obs=b, na=na, tamanho_tupla=n_tupla, phi=phi, resolucao_acao=4,
                       com_acoes=K > 0)
    t = TimeAtorCritico(ap, lambda v: np.asarray(v, bool), c, na=na, sigma=sigma, passos=1,
                        desconto=0.99, contrafactual=K)
    t.reset([np.random.default_rng(10 + i) for i in range(na)])
    return ap, c, t


def _bandido_de_time(t, b, n):
    """Episódios de UM passo: recompensa = velocidade executada pelo drone 0; a
    ação do drone 1 é irrelevante. Observações fixas por drone. Devolve a média
    de |A_i| na segunda metade (crítico já aquecido) e a dispersão do crédito."""
    xs = [np.zeros(b, bool), np.zeros(b, bool)]
    xs[0][::2] = True; xs[1][1::2] = True
    adv = []
    for k in range(n):
        acoes = t.agir(xs)
        t.registrar(float(acoes[0][1]))
        t.fim_de_episodio()
        if k == n // 2:
            t.estatisticas()                  # descarta o aquecimento
        adv.append(np.abs(t.ultimas_vantagens))
    adv = np.array(adv[n // 2:])
    return float(adv[:, 0].mean()), float(adv[:, 1].mean()), t.estatisticas()


def test_critico_de_estado_da_a_mesma_vantagem_a_todos():
    b = 8
    _, _, t = _time(na=2, b=b, K=0)
    a0, a1, est = _bandido_de_time(t, b, 400)
    assert a0 == pytest.approx(a1)
    assert est["dispersao_credito"] == pytest.approx(0.0)


def test_contrafactual_atribui_credito_ao_drone_que_decide():
    """Com K > 0 o drone irrelevante recebe vantagem menor que o drone cuja
    ação define a recompensa — é o análogo do `∂Q/∂a_i` do MADDPG. A razão
    |A_1|/|A_0| = 1 − c, onde c é a fração do efeito da ação que o crítico
    explica. Medido nesta configuração (32 bits, n = 12, 3 RAMs): razão 0,55 e
    dispersão 0,35, estável de 1 200 a 6 000 episódios (viés da média entre RAMs,
    não ruído); com n = 16, 0,40. O limiar 0,7 pega uma regressão que deixe de
    isolar crédito sem exigir o que a rede n-tuple não dá."""
    b = 8
    _, _, t = _time(na=2, b=b, K=8, n_tupla=12)
    a0, a1, est = _bandido_de_time(t, b, 1200)
    assert a0 > 0.05
    assert a1 < 0.7 * a0, (a0, a1)
    assert est["dispersao_credito"] > 0.2, est


def test_k_e_tipo_de_critico_tem_de_casar():
    b = 8
    ap = _AtorCongelado()
    cod = lambda v: np.asarray(v, bool)  # noqa: E731
    q = CriticoCentral(n_bits_obs=b, na=2, tamanho_tupla=4, com_acoes=True)
    v = CriticoCentral(n_bits_obs=b, na=2, tamanho_tupla=4, com_acoes=False)
    with pytest.raises(ValueError):
        TimeAtorCritico(ap, cod, q, na=2, sigma=0.3, contrafactual=0)
    with pytest.raises(ValueError):
        TimeAtorCritico(ap, cod, v, na=2, sigma=0.3, contrafactual=4)


def test_time_segue_a_disciplina_de_chamada():
    b = 16
    _, _, t = _time(na=2, b=b, K=0)
    with pytest.raises(RuntimeError):
        t.registrar(0.0)
    xs = [np.zeros(b, bool), np.ones(b, bool)]
    acoes = t.agir(xs)
    assert len(acoes) == 2 and all(a.shape == (2,) for a in acoes)
    t.registrar(0.5)
    t.fim_de_episodio()
    assert t.n_atualizacoes == 1
