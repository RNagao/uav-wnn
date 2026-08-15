"""Testes do harness de custo (SPEC-03)."""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.custo import custo_mlp, custo_ntuple, ocupacao_por_ram
from uavwnn.custo.realizado import LUTHash, medir_mlp, medir_realizado
from uavwnn.custo.operacoes import (
    CICLOS,
    ciclos,
    operacoes_amostragem,
    operacoes_codificador,
    operacoes_mlp,
    operacoes_ntuple,
)
from uavwnn.wnn import NTupleRegressor


@pytest.fixture
def rede_treinada():
    rng = np.random.default_rng(0)
    bits = rng.random((300, 200)) > 0.5
    net = NTupleRegressor(n_bits=200, n_ram=48, tamanho_tupla=12, n_saidas=2, seed=1)
    net.treinar(bits, rng.normal(size=(300, 2)), eta=0.5, epocas=1)
    return net


# --- T1 --------------------------------------------------------------------


def test_alocado_ocupado_realizado_sao_diferentes(rede_treinada):
    """A distinção é o ponto da spec: o alocado é cota superior, o ocupado é
    conteúdo de informação, e só o realizado responde "cabe no microcontrolador"."""
    net = rede_treinada
    occ = int(ocupacao_por_ram(net.pesos).sum())
    alocado = 2 * 48 * (1 << 12) * 4
    ordenado = custo_ntuple(net.pesos, 200, 12, 48, variante="ordenado").persistente
    denso = custo_ntuple(net.pesos, 200, 12, 48, variante="denso").persistente
    hash_ = custo_ntuple(net.pesos, 200, 12, 48, variante="hash").persistente

    assert occ * 4 < ordenado < hash_ < denso
    assert denso >= alocado, "a variante densa é o alocado, por definição"


def test_hash_paga_folga_de_carga(rede_treinada):
    """V-a paga fator de carga E arredondamento para potência de 2 — a spec
    previu que V-b pode ganhar nas duas dimensões que interessam."""
    net = rede_treinada
    h = custo_ntuple(net.pesos, 200, 12, 48, variante="hash").persistente
    o = custo_ntuple(net.pesos, 200, 12, 48, variante="ordenado").persistente
    assert h > 1.5 * o


def test_detalhe_soma_com_o_total(rede_treinada):
    """Fórmula que não bate com as partes é bug — a validação está no
    __post_init__ e este teste garante que ela é exercida."""
    c = custo_ntuple(rede_treinada.pesos, 200, 12, 48, variante="ordenado",
                     limiares_por_dim=8, n_dims=25)
    soma = sum(v for k, v in c.detalhe.items() if not k.startswith("buffer"))
    assert soma == c.persistente


def test_mapeamento_de_tuplas_nao_e_desprezivel(rede_treinada):
    """Omitir o mapeamento infla a vantagem.

    A spec ilustrava ~1,1 KB supondo índices de 2 bytes (`B < 65 536`). Com o
    `B = 200` bits que a SPEC-01 escolheu, um índice cabe em **1 byte** e o
    mapeamento custa 576 B — metade do ilustrado, mas ainda não desprezível
    diante das LUTs.
    """
    c = custo_ntuple(rede_treinada.pesos, 200, 12, 48, variante="ordenado")
    assert c.detalhe["mapeamento_tuplas"] == 48 * 12 * 1

    # Com entrada maior o índice passa a exigir 2 bytes.
    grande = custo_ntuple(rede_treinada.pesos, 200, 12, 48, variante="ordenado")
    assert grande.detalhe["mapeamento_tuplas"] == 576


def test_seed_nao_pode_encolher_o_residente(rede_treinada):
    """Proibição da §4.2: a seed reduz o persistente, nunca o residente — é o
    residente que responde "cabe no RAM"."""
    a = custo_ntuple(rede_treinada.pesos, 200, 12, 48, mapeamento_por_seed=False)
    b = custo_ntuple(rede_treinada.pesos, 200, 12, 48, mapeamento_por_seed=True)
    assert b.persistente < a.persistente
    assert b.residente == a.residente


def test_mlp_conta_pesos_e_vieses():
    c = custo_mlp([25, 32, 32, 2])
    esperado = ((25 * 32 + 32) + (32 * 32 + 32) + (32 * 2 + 2)) * 4
    assert c.persistente == esperado


def test_mlp_muda_de_tamanho_com_a_celula():
    """`D = 2(Na−1) + 25`, então a MLP pareada é diferente por célula e a razão
    precisa vir sempre com a configuração ao lado (RNF-06)."""
    assert custo_mlp([25, 32, 32, 2]).persistente < custo_mlp([39, 32, 32, 2]).persistente


# --- T3 --------------------------------------------------------------------


def test_zero_multiplicacoes_fp_no_aproximador():
    """A alegação central do PRD §5.5, como zero literal."""
    w = operacoes_ntuple(48, 12, 2, n_dims=25, n_niveis=8, incluir_amostragem=False)
    assert w.mult_fp == 0
    m = operacoes_mlp([25, 32, 32, 2], incluir_amostragem=False)
    assert m.mult_fp == 25 * 32 + 32 * 32 + 32 * 2


def test_amostragem_e_comum_aos_dois():
    """Ela não entra na diferença, mas precisa aparecer — senão a alegação de
    'sem transcendentais' ficaria falsa."""
    a = operacoes_amostragem()
    assert a.transcendental_fp > 0 and a.mult_fp > 0
    w = operacoes_ntuple(48, 12, 2, n_dims=25, n_niveis=8)
    assert w.mult_fp == a.mult_fp  # tudo que a WNN multiplica vem daí


def test_codificador_contabilizado_a_parte():
    """Parte da vantagem é do termômetro, não da rede — o relatório precisa
    poder separar."""
    cod = operacoes_codificador(25, 8)
    assert cod.comparacoes == 200
    com = operacoes_ntuple(48, 12, 2, n_dims=25, n_niveis=8)
    sem = operacoes_ntuple(48, 12, 2, incluir_codificador=False)
    assert com.comparacoes - sem.comparacoes == 200


def test_vantagem_cresce_sem_fpu():
    """A vantagem da WNN é CONDICIONAL à classe de alvo, e o gradiente é
    íngreme: marginal com FPU, quase 20x sem."""
    w = operacoes_ntuple(48, 12, 2, n_dims=25, n_niveis=8)
    m = operacoes_mlp([25, 32, 32, 2])
    razoes = {a: ciclos(m, a) / ciclos(w, a) for a in CICLOS}
    com_fpu = razoes["x86-64 (com FPU e SIMD)"]
    sem_fpu = razoes["Cortex-M0 (SEM FPU)"]
    assert 1.0 < com_fpu < 3.0
    assert sem_fpu > 10 * com_fpu


# --- T1: validação cruzada, fórmula contra medição -------------------------


@pytest.mark.parametrize("variante", ["ordenado", "hash", "denso"])
def test_formula_bate_com_a_estrutura_materializada(rede_treinada, variante):
    """A validação cruzada que faltava: `memoria.py` calcula, `realizado.py` mede.

    O veredito de T1 — WNN 17x maior que a MLP pareada — descansava numa
    fórmula que ninguém tinha conferido contra a realidade. Aqui as chaves,
    valores, bitmaps e cabeçalhos viram bytes de verdade e o tamanho é
    `len(blob)`.
    """
    f = custo_ntuple(rede_treinada.pesos, 200, 12, 48, variante=variante,
                     limiares_por_dim=8, n_dims=25)
    m = medir_realizado(rede_treinada, variante, limiares_por_dim=8, n_dims=25)
    assert m.persistente == f.persistente
    assert m.residente == f.residente


@pytest.mark.parametrize("variante", ["ordenado", "hash", "denso"])
def test_estrutura_esparsa_responde_como_a_densa(rede_treinada, variante):
    """O que dá valor à medição: a estrutura FUNCIONA.

    Uma LUT que ocupasse poucos bytes por ter perdido entradas mediria bonito e
    estaria errada. `medir_realizado` levanta se a predição divergir do denso
    além do arredondamento `float32` — este teste garante que essa verificação
    é de fato exercida, com bits reais.
    """
    rng = np.random.default_rng(11)
    bits = rng.random((120, 200)) > 0.5
    m = medir_realizado(rede_treinada, variante, bits_verificacao=bits)
    assert m.erro_max_vs_denso < 1e-6


def test_hash_acomoda_toda_a_ocupacao(rede_treinada):
    """`cap = 2^ceil(log2(occ/alpha))` com `alpha < 1` precisa caber sempre.

    Se a capacidade ficasse abaixo da ocupação, a sondagem linear entraria em
    laço infinito na inserção — o risco real da variante V-a, e a razão de a
    SPEC-03 exigir a mitigação do RNF-10.
    """
    lut = LUTHash(rede_treinada.pesos)
    occ = ocupacao_por_ram(rede_treinada.pesos)
    for i, (cap, oc) in enumerate(zip(lut.caps, occ)):
        assert cap >= oc, f"RAM {i}: capacidade {cap} < ocupação {oc}"
        assert cap == 0 or cap & (cap - 1) == 0, "capacidade não é potência de 2"


def test_mlp_medida_em_float32_como_a_wnn(rede_treinada):
    """A comparação só é honesta com a mesma régua nos dois lados."""
    assert medir_mlp([25, 32, 32, 2]).persistente == custo_mlp([25, 32, 32, 2]).persistente
