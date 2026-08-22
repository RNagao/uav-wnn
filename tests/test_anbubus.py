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
