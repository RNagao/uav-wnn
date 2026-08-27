"""Teste diferencial contra o `NTNModel` de `anbubus/ntn_neurocomputing_v0`.

Prova que a nossa `NTuplaAnbubusAprox` é uma **portagem fiel** daquele modelo, e
não uma reimplementação com o mesmo nome. É o mesmo padrão dos testes contra o
`Ramnet.jl`: um oráculo executável, comparação numérica, tolerância declarada.

O QUE ESTE TESTE REPLICA, E O QUE ELE DELIBERADAMENTE NÃO USA

O `NTNModel` original é um `TorchModelV2` do Ray RLlib. Importá-lo exigiria `ray`
inteiro no caminho de teste, e o que interessa não é a casca de RLlib — é o
**núcleo numérico**, que são três coisas:

    endereçamento   `conv1d(x[:, mapping], exp2(arange(n)), stride=n)`
    memória         `EmbeddingBag(mode='sum')` com pesos iniciados em zero
    codificação     `Thermometer`: `arange(R) < (x−min)/(max−min)·R`

As três são replicadas aqui em PyTorch puro, transcritas do código-fonte deles, e
comparadas com a nossa implementação em NumPy.

**`torch` é dependência só de teste** (grupo `baseline`), e o teste é pulado se ela
não estiver presente — a RNF-11 exige NumPy puro no caminho de medição, não no de
verificação.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch é opcional (grupo `baseline`)")

from uavwnn.agents.ac_continuo import NTuplaAnbubusAprox  # noqa: E402
from uavwnn.encoding.artigo import TermometroArtigo  # noqa: E402
from uavwnn.wnn.redes import RedeGradienteFuncional  # noqa: E402

CONFIGS = [
    pytest.param(64, 8, 3, id="B64_n8_d3"),
    pytest.param(128, 8, 1, id="B128_n8_d1"),
    pytest.param(96, 12, 3, id="B96_n12_d3"),
]


# --------------------------------------------------------------------------
# o núcleo deles, transcrito do ntn_model.py / ntn_encodings.py
# --------------------------------------------------------------------------


class NucleoAnbubus(torch.nn.Module):
    """`NTNModel` sem a casca do RLlib. Transcrição literal do núcleo."""

    def __init__(self, input_dim: int, tuple_size: int, n_saidas: int,
                 mapping: torch.Tensor) -> None:
        super().__init__()
        self.tuple_size = tuple_size
        self.n_nodes = int(np.ceil(input_dim / tuple_size))
        self.mapping = mapping
        # `exp2(arange(n))`: o k-ésimo bit da tupla pesa 2^k — mesma convenção do
        # `RawBitsEncoder` do Ramnet.jl, e a mesma que usamos.
        self.key_weights = torch.special.exp2(
            torch.arange(tuple_size).reshape(1, 1, -1)).to(torch.long)
        self.mem_offsets = (2 ** tuple_size) * torch.arange(self.n_nodes).reshape((1, -1))
        self.mem = torch.nn.EmbeddingBag(
            num_embeddings=self.n_nodes * 2 ** tuple_size,
            embedding_dim=n_saidas, mode="sum")           # <-- SOMA, não média
        torch.nn.init.zeros_(self.mem.weight)             # v_0 = 0

    def keys(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv1d(
            x[:, self.mapping].unsqueeze(1).to(torch.long).float(),
            self.key_weights.float(), stride=self.tuple_size,
        ).squeeze(1).to(torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mem(self.keys(x) + self.mem_offsets)


def termometro_anbubus(x, minimo, maximo, resolucao):
    """`Thermometer.encode` deles: `arange(R) < (x−min)/(max−min)·R`."""
    X = torch.as_tensor(np.asarray(x, dtype=np.float64))
    enc = torch.arange(resolucao).expand(*X.shape, resolucao)
    limite = (X.reshape((*X.shape, 1)) - minimo) / (maximo - minimo) * resolucao
    return (enc < limite).to(torch.long)


def _nossa_com_mapeamento_deles(n_bits, tupla, n_saidas, mapping):
    """A nossa rede, forçada a usar o `randperm` deles.

    Sem isto o teste compararia mapeamentos diferentes e falharia por uma razão
    sem interesse. O que está sob teste é a **aritmética**, não o sorteio.
    """
    r = RedeGradienteFuncional(n_bits, tupla, n_saidas=n_saidas, seed=0,
                               c_max=None, convencao="soma")
    idx = mapping.numpy()
    # `remapear` e não atribuição direta: a rede pré-computa matrizes de
    # endereçamento, e trocar `mapeamento` sem remontá-las usaria o mapeamento
    # antigo em silêncio. Este teste foi o que pegou isso.
    r.remapear([idx[i:i + tupla] for i in range(0, len(idx), tupla)])
    r.memoria = [{} for _ in range(r.n_ram)]
    return r


# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_bits,tupla,n_saidas", CONFIGS)
def test_enderecamento_identico(n_bits, tupla, n_saidas):
    """As chaves têm de coincidir **exatamente** — são inteiros, não floats."""
    g = torch.Generator().manual_seed(7)
    mapping = torch.randperm(n_bits, generator=g)
    deles = NucleoAnbubus(n_bits, tupla, n_saidas, mapping)
    nossa = _nossa_com_mapeamento_deles(n_bits, tupla, n_saidas, mapping)

    rng = np.random.default_rng(0)
    bits = rng.random((16, n_bits)) > 0.5
    chaves_deles = deles.keys(torch.as_tensor(bits)).numpy()
    chaves_nossas = nossa.enderecos(bits).astype(np.int64)
    np.testing.assert_array_equal(chaves_nossas, chaves_deles)


@pytest.mark.parametrize("n_bits,tupla,n_saidas", CONFIGS)
def test_saida_identica_na_convencao_de_soma(n_bits, tupla, n_saidas):
    """`EmbeddingBag(mode='sum')` **é** a nossa `convencao='soma'`."""
    g = torch.Generator().manual_seed(11)
    mapping = torch.randperm(n_bits, generator=g)
    deles = NucleoAnbubus(n_bits, tupla, n_saidas, mapping)
    nossa = _nossa_com_mapeamento_deles(n_bits, tupla, n_saidas, mapping)

    rng = np.random.default_rng(1)
    # escreve os MESMOS valores nos dois, pelas chaves
    for _ in range(30):
        b = rng.random(n_bits) > 0.5
        passo = rng.normal(size=n_saidas)
        nossa.atualizar(b, passo)
        with torch.no_grad():
            chaves = deles.keys(torch.as_tensor(b[None, :]))[0] + deles.mem_offsets[0]
            for k in chaves:
                deles.mem.weight[k] += torch.as_tensor(passo, dtype=torch.float32)

    b = rng.random((8, n_bits)) > 0.5
    with torch.no_grad():
        y_deles = deles(torch.as_tensor(b)).numpy()
    y_nossa = nossa.prever(b)
    np.testing.assert_allclose(y_nossa, y_deles, rtol=0, atol=2e-5)


@pytest.mark.parametrize("n_bits,tupla,n_saidas", CONFIGS)
def test_autograd_deles_e_o_nosso_gradiente_funcional(n_bits, tupla, n_saidas):
    """O gradiente de `EmbeddingBag` **é** a atualização da eq. (4.11).

    Confirma o achado que já valia para a rede da dissertação: autograd sobre uma
    tabela endereçada por `EmbeddingBag` produz exatamente "some `passo` nas `N`
    posições endereçadas, deixe o resto intacto" — o conteúdo da eq. (4.17).
    """
    g = torch.Generator().manual_seed(13)
    mapping = torch.randperm(n_bits, generator=g)
    deles = NucleoAnbubus(n_bits, tupla, n_saidas, mapping)
    deles.mem.weight.data.double()

    rng = np.random.default_rng(2)
    b = rng.random(n_bits) > 0.5
    direcao = rng.normal(size=n_saidas)

    y = deles(torch.as_tensor(b[None, :]))
    y.backward(torch.as_tensor(direcao, dtype=torch.float32)[None, :])
    grad = deles.mem.weight.grad

    chaves = (deles.keys(torch.as_tensor(b[None, :]))[0]
              + deles.mem_offsets[0]).numpy()
    # exatamente `direcao` nas posições endereçadas...
    for k in chaves:
        np.testing.assert_allclose(grad[k].numpy(), direcao, rtol=0, atol=1e-6)
    # ...e zero em todo o resto
    mascara = np.ones(grad.shape[0], bool)
    mascara[chaves] = False
    assert not grad[torch.as_tensor(mascara)].abs().sum().item()


def test_termometro_deles_equivale_ao_do_artigo():
    """O `Thermometer` deles e o nosso `TermometroArtigo` são a mesma função.

    Reordenando o deles: `k < (x−min)/(max−min)·R` ⟺ `x > min + k·(max−min)/R`.
    É o termômetro da eq. (3.23) com `>` estrito, que é o que implementamos.
    """
    lo, hi, res = -1.2, 0.6, 16
    nosso = TermometroArtigo(lo, hi, res)
    rng = np.random.default_rng(3)
    xs = rng.uniform(lo, hi, 200)
    a = nosso.codificar(xs).reshape(len(xs), res).astype(int)
    b = termometro_anbubus(xs, lo, hi, res).numpy()
    np.testing.assert_array_equal(a, b)


def test_as_duas_familias_diferem_onde_deve_e_concordam_onde_deve():
    """Contraste explícito entre as duas opções WNN — o objeto da comparação.

    Não é teste de igualdade: é de que as diferenças são **as declaradas**, e só
    elas. Se alguém alinhar as duas por acidente, a comparação perde o objeto.
    """
    from uavwnn.agents.ac_continuo import NTuplaDissertacaoAprox

    nossa = NTuplaDissertacaoAprox(n_bits=128, tamanho_tupla=8)
    deles = NTuplaAnbubusAprox(n_bits=128, tamanho_tupla=8)

    # DIFEREM: convenção, e a família do crítico
    assert nossa.ator.convencao == "media"
    assert deles.ator.convencao == "soma"
    assert type(nossa.critico) is not type(nossa.ator)     # duas famílias (C2)
    assert type(deles.critico) is type(deles.ator)         # uma família (PPO)
    assert deles.custo()["critico_mesma_familia"] is True

    # CONCORDAM: a alegação de H1, e a dimensão da política
    for c in (nossa.custo(), deles.custo()):
        assert c["mult_fp_aproximador"] == 0
    assert nossa.ator.n_saidas == deles.ator.n_saidas == 3


# --- os invariantes de ARQUITETURA do `NTNModel`, lidos do código publicado ------
#
# Os testes acima provam equivalência NUMÉRICA — endereçamento, saída, gradiente e
# termômetro. Estes provam que a ESTRUTURA é a do artigo, que é o que a equivalência
# numérica não captura: dois modelos podem concordar num vetor e diferir em quantas
# memórias têm, na dimensão do crítico ou no estado inicial.


def test_duas_memorias_separadas_ambas_de_gradiente_funcional():
    """`policy_mem` e `vf_mem`, as duas `EmbeddingBag`, as duas por gradiente.

    É a diferença que resolve a pendência C2 e o motivo de o `anbubus` ser um braço
    separado. O PPO deles treina ator e crítico pelo MESMO mecanismo — descida —,
    enquanto o Algoritmo 4.4 usa gradiente funcional no ator e regressão com
    esquecimento no crítico, que são modelos diferentes.

    Confundir os dois braços aqui apagaria justamente o que a comparação mede.
    """
    ap = NTuplaAnbubusAprox(n_bits=200, tamanho_tupla=8)
    assert type(ap.ator) is RedeGradienteFuncional
    assert type(ap.critico) is RedeGradienteFuncional, (
        "o crítico do anbubus é gradiente funcional, não regressão com esquecimento")
    assert ap.ator.convencao == ap.critico.convencao == "soma", (
        "`EmbeddingBag(mode='sum')` é a convenção de soma nas DUAS memórias")


def test_critico_tem_uma_saida_como_vf_output_dim():
    """`self.vf_output_dim = torch.Size([1])` no `NTNModel`.

    Não é detalhe de forma: o crítico estima `V(s)`, um escalar. Se ele tivesse a
    dimensão do ator, `regredir_critico` treinaria três valores e o `v()` leria só
    um — erro silencioso, com dois terços da memória do crítico nunca consultados.
    """
    ap = NTuplaAnbubusAprox(n_bits=200, tamanho_tupla=8)
    assert ap.critico.n_saidas == 1
    assert isinstance(ap.v(np.zeros(200, dtype=np.uint8)), float)


@pytest.mark.parametrize("n_bits,tupla", [(200, 8), (800, 12), (7500, 10)])
def test_n_ram_e_o_teto_de_bits_sobre_tupla(n_bits, tupla):
    """`n_nodes = ceil(input_dim / tuple_size)` — eq. do `NTNModel`, linha 24.

    `N` é DERIVADO, não escolhido, e é o que faz a taxa deles não ser transferível:
    somar `passo` às `N` posições muda `f` em `passo·N` na convenção de soma. Um `N`
    errado aqui propaga silenciosamente para toda conversão de taxa.

    O caso `(7500, 10)` é a configuração `reacher` sobre a nossa observação de 25
    escalares com `T = 300`: 750 RAMs.
    """
    ap = NTuplaAnbubusAprox(n_bits=n_bits, tamanho_tupla=tupla)
    assert ap.ator.n_ram == -(-n_bits // tupla)
    assert ap.critico.n_ram == ap.ator.n_ram


def test_memorias_comecam_em_zero():
    """`nn.init.zeros_` nas duas memórias do `NTNModel`.

    A consequência é comportamental, não estética: com tudo em zero a política
    inicial é `N(0, σ²)` **independentemente da arquitetura**, então duas redes de
    tamanhos diferentes com a mesma semente produzem as mesmas primeiras ações. Foi
    o que explicou dois retornos idênticos em corridas de `T = 32` e `T = 300` que
    eu quase tratei como defeito.

    No nosso caso o zero vem da memória esparsa vazia, não de uma inicialização —
    o efeito observável é o mesmo, e é ele que o teste fixa.
    """
    ap = NTuplaAnbubusAprox(n_bits=200, tamanho_tupla=8)
    bits = np.zeros(200, dtype=np.uint8)
    bits[::3] = 1
    assert np.allclose(ap.f(bits), 0.0)
    assert ap.v(bits) == 0.0


def test_mapeamento_e_permutacao_cada_bit_uma_vez():
    """`torch.randperm(input_dim)` — cada bit de entrada em exatamente uma tupla.

    Amostrar com reposição daria bits ignorados e bits repetidos, o que muda a
    capacidade do modelo sem mudar `N`. O teste não compara com o `randperm` deles
    (isso o `test_enderecamento_identico` já faz forçando o mapa); fixa a
    PROPRIEDADE, que é o que sobrevive a trocar de gerador.
    """
    ap = NTuplaAnbubusAprox(n_bits=800, tamanho_tupla=12)
    usados = np.concatenate([np.asarray(t) for t in ap.ator.mapeamento])
    assert len(usados) == len(set(usados.tolist())), "algum bit aparece duas vezes"
    assert set(usados.tolist()) <= set(range(800))
    # `ceil` implica sobra: 800/12 dá 66 tuplas cheias e uma de 8.
    assert sum(len(t) for t in ap.ator.mapeamento) == 800


def test_sigma_fixa_e_desvio_DECLARADO_do_artigo():
    """O artigo aprende `log σ`; nós não. É desvio, e tem de ser visível.

    Sob a `DiagGaussian` do RLlib, `num_outputs = 2 × dim_ação`, e a `policy_mem`
    emite média E `log σ` — verificado executando: `policy_output_dim = 4` para a
    ação de 2 componentes. O bloco comentado deles (`chunk(2, dim=1)[1]` em
    `log(.5)`) confirma o layout.

    Nós usamos `d_saida = 3` com σ FIXA, pelas quatro razões da §4.4, e a terceira
    é a que importa aqui: uma cabeça `log σ` no Algoritmo 4.4 diverge, porque onde a
    vantagem é consistentemente positiva `σ → 0` e o fator `σ⁻²` explode. O PPO
    neutraliza isso com o recorte do objetivo; o Algoritmo 4.4 não tem recorte.

    O teste fixa duas coisas: que o padrão é 3 (o nosso), e que `d_saida` é
    PARAMETRIZÁVEL para o layout deles — sem isso o braço PPO não teria como usar a
    mesma classe.
    """
    from uavwnn.agents.politica_continua import D_ATOR

    assert NTuplaAnbubusAprox(n_bits=200).d_saida == D_ATOR == 3
    deles = NTuplaAnbubusAprox(n_bits=200, d_saida=4)
    assert deles.ator.n_saidas == 4, (
        "`d_saida` tem de aceitar 2×dim_ação, que é o layout da DiagGaussian")
    assert deles.critico.n_saidas == 1, "o crítico não acompanha o ator"
