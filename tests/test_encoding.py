"""Testes do codificador binário e do núcleo n-tuple (SPEC-01)."""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.encoding import (
    PADDING,
    TermometroGlobal,
    TermometroPorSlot,
    TermometroQuantil,
    coletar_corpus,
)
from uavwnn.refenv import ReferenceNotAvailable, reference_root
from uavwnn.wnn import NTupleRegressor

ESQUEMAS = (TermometroGlobal, TermometroPorSlot, TermometroQuantil)

try:
    reference_root()
    _SUBMODULE = True
except ReferenceNotAvailable:
    _SUBMODULE = False


@pytest.fixture(scope="module")
def obs():
    """Amostra sintética com a mesma estrutura da observação real:
    valores válidos na faixa medida e slots de padding em -1."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.4, 1.4, size=(400, 25)).astype(np.float32)
    x[:, 10:14] = PADDING  # dois slots vazios
    x[:200, 18:20] = PADDING
    return x


# --- o padding se separa por construção ------------------------------------


@pytest.mark.parametrize("cls", ESQUEMAS)
def test_padding_vira_todo_zero(cls, obs):
    """A propriedade que dispensa o bit de validade.

    Todo limiar fica acima de PADDING, então um slot vazio produz o padrão
    todo-zero. E o primeiro limiar é o mínimo empírico dos válidos, então
    nenhum valor válido produz todo-zero — o mínimo acende exatamente um bit.
    """
    T = 8
    enc = cls(T).ajustar(obs)
    bits = enc.codificar(obs).reshape(len(obs), obs.shape[1], T)
    vazio = np.isclose(obs, PADDING)

    assert bits[vazio].sum() == 0, "padding acendeu algum bit"
    assert (bits[~vazio].sum(axis=1) > 0).all(), "valor válido colidiu com o padding"


@pytest.mark.parametrize("cls", ESQUEMAS)
def test_monotonico_no_valor(cls, obs):
    """Termômetro: mais valor, nunca menos bits. É o que preserva similaridade."""
    T = 8
    enc = cls(T).ajustar(obs)
    v = np.linspace(-0.4, 1.4, 50, dtype=np.float32)
    entrada = np.tile(v[:, None], (1, obs.shape[1]))
    contagem = enc.codificar(entrada).reshape(len(v), obs.shape[1], T).sum(axis=2)
    assert (np.diff(contagem, axis=0) >= 0).all()


@pytest.mark.parametrize("cls", ESQUEMAS)
def test_determinismo_e_forma(cls, obs):
    enc = cls(16).ajustar(obs)
    a, b = enc.codificar(obs), enc.codificar(obs)
    assert a.shape == (len(obs), obs.shape[1] * 16) == (len(obs), enc.n_bits)
    assert np.array_equal(a, b)
    assert a.dtype == np.bool_


def test_quantil_distribui_melhor_que_global(obs):
    """O motivo de E2 vencer: bins equiprováveis não desperdiçam resolução.

    Com entrada concentrada — como o slot do sensor mais próximo, cuja
    amplitude é 3x menor que a do 12º — o binning global gasta quase todos os
    bits em regiões vazias.
    """
    rng = np.random.default_rng(1)
    x = np.concatenate([
        rng.normal(0.5, 0.03, size=(2000, 1)),      # concentrada
        rng.uniform(-0.4, 1.4, size=(2000, 1)),     # espalhada
    ], axis=1).astype(np.float32)

    def bits_distintos(cls):
        enc = cls(16).ajustar(x)
        b = enc.codificar(x)[:, :16]                 # só a dimensão concentrada
        return len(np.unique(b, axis=0))

    assert bits_distintos(TermometroQuantil) > 3 * bits_distintos(TermometroGlobal)


# --- núcleo n-tuple --------------------------------------------------------


def test_ntuple_aprende_regressao_sintetica():
    """`sin(3x)cos(2y)`, o alvo do protótipo do PLANO.md."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(3000, 2))
    y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1])
    lim = np.linspace(0, 1, 32)
    B = (X[:, :, None] >= lim[None, None, :]).reshape(len(X), -1)

    net = NTupleRegressor(n_bits=B.shape[1], n_ram=48, tamanho_tupla=12, seed=1)
    net.treinar(B[:2400], y[:2400], eta=0.3, epocas=5, rng=np.random.default_rng(2))
    rmse = float(np.sqrt(((net.prever(B[2400:])[:, 0] - y[2400:]) ** 2).mean()))
    assert rmse < 0.05, f"RMSE {rmse:.4f} alto demais"


def test_passo_efetivo_independe_de_N():
    """A convenção de MÉDIA desacopla a taxa de aprendizado de `N`.

    Com `f = (1/N) Σ v_i` e `v_i -= η(f − y)`, a variação da saída num passo é
    exatamente `−η(f − y)`, seja qual for `N`. Com a convenção de soma seria
    `−Nη(f − y)`, e varrer `N` com `η` fixo mediria taxa de aprendizado em vez
    de capacidade.
    """
    bits = np.zeros((1, 64), dtype=bool)
    bits[0, ::2] = True
    for n_ram in (8, 64):
        net = NTupleRegressor(n_bits=64, n_ram=n_ram, tamanho_tupla=8, seed=0)
        antes = float(net.prever(bits)[0, 0])
        net.treinar(bits, np.array([1.0]), eta=0.5, epocas=1)
        depois = float(net.prever(bits)[0, 0])
        assert depois - antes == pytest.approx(0.5, abs=1e-9)


def test_cobertura_de_bits_detecta_subamostragem():
    """Achado C4: `N·n < B` significa que a rede ignora parte da entrada."""
    baixa = NTupleRegressor(n_bits=800, n_ram=16, tamanho_tupla=12, seed=0)
    alta = NTupleRegressor(n_bits=200, n_ram=128, tamanho_tupla=12, seed=0)
    assert baixa.cobertura < 0.35
    assert alta.cobertura == 1.0


def test_tupla_maior_que_entrada_falha_cedo():
    with pytest.raises(ValueError, match="maior que n_bits"):
        NTupleRegressor(n_bits=8, n_ram=4, tamanho_tupla=12)


# --- integração com o ambiente ---------------------------------------------


@pytest.mark.skipif(not _SUBMODULE, reason="submodule ausente")
def test_corpus_do_ambiente_real():
    corpus = coletar_corpus([(1, 12)], n_episodios=2, politicas=("greedy",))
    assert len(corpus) > 50
    assert corpus.obs.shape[1] == 25
    validos = corpus.obs[~np.isclose(corpus.obs, PADDING)]
    # Faixa REAL, não a declarada `Box(-1, 1)`: `p_k - p_i` vive em [-2S, 2S].
    assert -0.6 < validos.min() and validos.max() < 1.6
    assert np.isclose(corpus.obs, PADDING).any(), "esperava slots de padding"


