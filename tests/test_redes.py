"""As duas redes da dissertação: equivalência com o oráculo e propriedades.

O teste que decide é `test_equivale_ao_ramnet`: as predições do lado direito
saíram do `Ramnet.jl` do próprio Katopodis. Os demais cobrem o que o oráculo
não distingue — casos-limite que as configurações do teste diferencial não
exercitam, como tupla curta no fim da partição e `φ = 1`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spec04_diferencial_redes import CONFIGS, M_TREINO, _monta, _treina
from uavwnn.wnn import (
    MAX_TUPLA,
    NTupleRegressor,
    RedeGradienteFuncional,
    RedeRegressaoEsquecimento,
    mapeamento_particionado,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diferencial_redes"


# --- o oráculo -------------------------------------------------------------


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c["nome"] for c in CONFIGS])
def test_equivale_ao_ramnet(cfg):
    """Contra `Ramnet.jl@faster_hashing`, Julia 1.10.5.

    Cobre `FunctionalDiscriminator` (§4.3.1) e `RegressionDiscriminator`
    (§4.2.1) com φ ∈ {0,5; 0,85; 1,0}. A média ponderada da eq. (4.9) tem
    várias formas plausíveis de errar — somar contadores em vez de pesos,
    aplicar φ na ordem trocada — e todas produziriam números com cara de
    certos. Só o oráculo separa.
    """
    esperado = np.loadtxt(FIXTURES / f"{cfg['nome']}.txt", ndmin=2)
    rede, bits, alvo = _monta(cfg)
    _treina(cfg, rede, bits, alvo)
    obtido = rede.prever(bits[M_TREINO:])

    assert obtido.shape == esperado.shape
    desvio = float(np.abs(obtido - esperado).max())
    assert desvio < 1e-12, f"divergência {desvio:.3e} do Ramnet.jl"
    assert np.abs(esperado).max() > 0.1, "fixture sem sinal"


def test_fixtures_cobrem_os_dois_papeis_e_varios_phi():
    papeis = {c["papel"] for c in CONFIGS}
    phis = {c["phi"] for c in CONFIGS if c["papel"] == "critico"}
    assert papeis == {"ator", "critico"}
    assert len(phis) >= 3 and 1.0 in phis, "φ = 1 é o ramo especial da eq. (4.10)"


# --- mapeamento por partição ----------------------------------------------


@pytest.mark.parametrize("n_bits,tupla", [(64, 8), (96, 12), (200, 12), (100, 7)])
def test_particao_le_cada_bit_exatamente_uma_vez(n_bits, tupla):
    """`:uniform_random` do Ramnet: cobertura 1,0 por construção.

    É o achado C4 resolvido pela estrutura. A amostragem por RAM do
    `NTupleRegressor` pode deixar bits sem leitura; a partição não pode.
    """
    m = mapeamento_particionado(n_bits, tupla, seed=3)
    todos = np.concatenate(m)
    assert len(todos) == n_bits
    assert np.array_equal(np.sort(todos), np.arange(n_bits))
    assert len(m) == -(-n_bits // tupla)


def test_tupla_curta_no_fim_quando_nao_divide():
    """`Iterators.partition` do Julia devolve resto; nós também.

    É exatamente o caso que quebra o `NTNModel` em PyTorch, onde `n_nodes` usa
    `ceil` mas a `conv1d` produz `floor`.
    """
    m = mapeamento_particionado(200, 12, seed=0)
    assert len(m) == 17
    assert [len(t) for t in m[:-1]] == [12] * 16
    assert len(m[-1]) == 8

    rede = RedeGradienteFuncional(200, 12, n_saidas=2, seed=0)
    bits = np.random.default_rng(1).random((5, 200)) > 0.5
    assert rede.enderecos(bits).shape == (5, 17)
    assert rede.prever(bits).shape == (5, 2)


def test_tupla_grande_que_o_denso_nao_alcanca():
    """As tabelas do apêndice A usam n = 16, 32 e 64. O denso alocaria 2^64."""
    rede = RedeGradienteFuncional(128, 64, n_saidas=3, seed=1)
    bits = np.random.default_rng(0).random((4, 128)) > 0.5
    end = rede.enderecos(bits)
    assert end.dtype == np.uint64
    assert end.max() > 2**32, "endereços deveriam usar a faixa alta de 64 bits"
    rede.atualizar(bits[:1], np.array([1.0, 2.0, 3.0]))
    assert rede.entradas_ocupadas() == rede.n_ram


def test_tupla_acima_de_64_falha_cedo():
    with pytest.raises(ValueError, match=f"excede {MAX_TUPLA}"):
        RedeGradienteFuncional(256, 65)


# --- ator: eqs. (4.11) e (4.12) -------------------------------------------


def test_ator_usa_media_e_o_passo_independe_de_N():
    """A eq. (4.12) traz o `1/N` explícito.

    Com `Δ = −η(f − y)`, a variação da saída é exatamente `−η(f − y)`, seja
    qual for `N`. Com a convenção de soma — a do `EmbeddingBag` do repositório
    em PyTorch — seria `−Nη(f − y)`.
    """
    for tupla in (8, 16):
        rede = RedeGradienteFuncional(64, tupla, n_saidas=1, seed=0)
        bits = np.zeros((1, 64), dtype=bool)
        bits[0, ::2] = True
        antes = float(rede.prever(bits)[0, 0])
        f = rede.prever(bits)[0]
        rede.atualizar(bits, -0.5 * (f - np.array([1.0])))
        depois = float(rede.prever(bits)[0, 0])
        assert depois - antes == pytest.approx(0.5, abs=1e-12)


def test_ator_esparso_concorda_com_o_regressor_denso():
    """Mesma matemática, armazenamentos diferentes.

    O `NTupleRegressor` denso continua servindo a SPEC-01 e a SPEC-03; este
    teste garante que a troca para dicionário não mudou o modelo. O mapeamento
    é forçado a ser o mesmo porque as duas classes o sorteiam de formas
    distintas (partição contra amostragem por RAM).
    """
    B, tupla, d = 96, 12, 2
    esparsa = RedeGradienteFuncional(B, tupla, n_saidas=d, seed=5)
    densa = NTupleRegressor(n_bits=B, n_ram=esparsa.n_ram, tamanho_tupla=tupla,
                            n_saidas=d, seed=5)
    densa.mapeamento = np.stack(esparsa.mapeamento)

    rng = np.random.default_rng(0)
    bits = rng.random((150, B)) > 0.5
    alvo = rng.normal(size=(150, d))
    for m in range(120):
        x = bits[m:m + 1]
        esparsa.atualizar(x, -0.4 * (esparsa.prever(x)[0] - alvo[m]))
        f = np.stack([densa.pesos[k][np.arange(densa.n_ram),
                                     densa.enderecos(x)[0]].mean()
                      for k in range(d)])
        for k in range(d):
            densa.pesos[k][np.arange(densa.n_ram), densa.enderecos(x)[0]] -= \
                0.4 * (f[k] - alvo[m, k])

    assert np.abs(esparsa.prever(bits[120:]) - densa.prever(bits[120:])).max() < 1e-12


# --- crítico: eqs. (4.8), (4.9) e (4.10) ----------------------------------


def test_critico_sem_esquecimento_e_media_simples():
    """`φ = 1` cai no ramo `c[i](x)` da eq. (4.10): o denominador vira o
    contador, e a rede degenera na regressão n-tuple original."""
    rede = RedeRegressaoEsquecimento(64, 8, phi=1.0, seed=0)
    bits = np.zeros((1, 64), dtype=bool)
    bits[0, ::3] = True
    for y in (2.0, 4.0, 6.0):
        rede.treinar(bits, np.array([y]))
    assert float(rede.prever(bits)[0, 0]) == pytest.approx(4.0, abs=1e-12)


def test_critico_com_esquecimento_pesa_o_recente():
    """Com φ < 1 a média é ponderada e a última observação domina.

    Denominador da eq. (4.10) com c = 3, φ = 0,5: (1 − 0,5³)/(1 − 0,5) = 1,75.
    Numerador: 0,5²·2 + 0,5·4 + 6 = 8,5. Logo f = 8,5/1,75.
    """
    rede = RedeRegressaoEsquecimento(64, 8, phi=0.5, seed=0)
    bits = np.zeros((1, 64), dtype=bool)
    bits[0, ::3] = True
    for y in (2.0, 4.0, 6.0):
        rede.treinar(bits, np.array([y]))
    assert float(rede.prever(bits)[0, 0]) == pytest.approx(8.5 / 1.75, abs=1e-12)


def test_critico_devolve_zero_onde_nada_foi_escrito():
    """A eq. (4.9) define explicitamente o ramo `Σ s = 0 -> 0`."""
    rede = RedeRegressaoEsquecimento(64, 8, phi=0.85, seed=0)
    bits = np.random.default_rng(0).random((3, 64)) > 0.5
    assert np.array_equal(rede.prever(bits), np.zeros((3, 1)))


def test_phi_fora_do_intervalo_falha_cedo():
    with pytest.raises(ValueError, match="fora de"):
        RedeRegressaoEsquecimento(64, 8, phi=1.5)


def test_critico_guarda_o_dobro_do_ator():
    """Valor E contador por posição — é o que a contabilidade de T1 precisa."""
    B, tupla = 64, 8
    bits = np.random.default_rng(0).random((20, B)) > 0.5
    ator = RedeGradienteFuncional(B, tupla, n_saidas=1, seed=0)
    critico = RedeRegressaoEsquecimento(B, tupla, n_saidas=1, phi=0.85, seed=0)
    for m in range(20):
        ator.atualizar(bits[m:m + 1], np.array([1.0]))
        critico.treinar(bits[m:m + 1], np.array([1.0]))
    assert ator.entradas_ocupadas() == critico.entradas_ocupadas()
    v_ator = next(iter(ator.memoria[0].values()))
    v_critico = next(iter(critico.memoria[0].values()))
    assert isinstance(v_ator, np.ndarray)
    assert len(v_critico) == 2, "crítico guarda (valor, contador)"


def test_carregar_remonta_o_enderecamento_vetorizado():
    """Regressão: `carregar` tem de passar por `remapear`, não atribuir direto.

    O defeito que isto guarda era silencioso e total. `carregar` atribuía
    `.mapeamento` sem remontar `_tuplas`/`_pot_mat` — as matrizes que o `enderecos()`
    vetorizado de fato usa —, então a rede carregada lia os endereços do mapa
    ALEATÓRIO do construtor, com os valores certos na memória. Nada levantava:
    `n_ram` coincide e as formas coincidem, só o conteúdo está trocado.

    Medido antes da correção: original `[1,004; 2,008; 3,012]`, carregada
    `[0,012; 0,024; 0,036]`. Todo checkpoint carregado estava errado, e o único
    motivo pelo qual a grade não foi afetada é que ela avalia a rede VIVA.

    Verifica os dois: predição idêntica E `_tuplas` coerente com `mapeamento`.
    """
    import tempfile

    g = np.random.default_rng(0)
    rede = RedeGradienteFuncional(200, 8, n_saidas=3, seed=1)
    bits = (g.random(200) < 0.5).astype(np.uint8)
    for _ in range(30):
        rede.atualizar((g.random(200) < 0.5).astype(np.uint8),
                       np.array([0.1, 0.2, 0.3]))
    rede.atualizar(bits, np.array([1.0, 2.0, 3.0]))
    esperado = rede.prever(bits).copy()

    with tempfile.TemporaryDirectory() as t:
        alvo = Path(t) / "rede.npz"
        rede.salvar(alvo)
        lida = RedeGradienteFuncional.carregar(alvo)

    assert np.allclose(esperado, lida.prever(bits)), (
        "a rede carregada prediz outro valor: o endereçamento não foi remontado")
    # E o invariante que causa o defeito, verificado direto: a matriz de tuplas
    # tem de ser o mapeamento, não o sorteio do construtor.
    for r, tupla in enumerate(lida.mapeamento):
        assert list(lida._tuplas[r][:len(tupla)]) == list(tupla), (
            f"_tuplas[{r}] não corresponde a mapeamento[{r}]")


def test_contador_e_inteiro_e_sobrevive_ao_ciclo(tmp_path):
    """O contador é contagem de visitas — inteiro por construção (`c += 1`).

    Guardá-lo em `float64` custava 8 bytes onde 4 bastam, e o crítico guarda DOIS
    números por entrada: era metade do custo dele. `uint32` e não `uint16` porque
    o máximo medido num modelo real é 2 353 751 visitas na posição mais quente.

    O ciclo salvar→carregar tem de preservar o tipo: um `.npz` antigo traz
    `float64`, e recarregá-lo sem converter reintroduziria o custo em silêncio.
    """
    import numpy as np

    from uavwnn.wnn.redes import RedeRegressaoEsquecimento

    rede = RedeRegressaoEsquecimento(n_bits=32, tamanho_tupla=4, n_saidas=2, phi=0.85)
    bits = np.zeros((1, 32), bool)
    bits[0, ::3] = True
    for _ in range(5):
        rede.treinar(bits, np.array([1.5, -0.5]))

    _, c = next(iter(rede.memoria[0].items()))[1]
    assert c.dtype == np.uint32, f"contador em {c.dtype}, esperado uint32"
    assert (c == 5).all()

    antes = rede.prever(bits)
    rede.salvar(tmp_path / "c.npz")
    lida = RedeRegressaoEsquecimento.carregar(tmp_path / "c.npz")
    _, c2 = next(iter(lida.memoria[0].items()))[1]
    assert c2.dtype == np.uint32, "carregar perdeu o tipo do contador"
    np.testing.assert_allclose(lida.prever(bits), antes, rtol=0, atol=0)


def test_custo_do_critico_conta_o_contador():
    """`_bytes_luts` contava só `n_saidas` valores por entrada — mas a rede com
    esquecimento guarda valor E contador, e saía ~40 % subestimada.

    Não afetou o T1 publicado (que compara ATORES), mas qualquer conta de
    "ator + crítico" pelo harness estava errada.
    """
    import numpy as np

    from uavwnn.custo.memoria import custo_ntuple_de_rede
    from uavwnn.wnn.redes import RedeGradienteFuncional, RedeRegressaoEsquecimento

    bits = np.zeros((1, 32), bool)
    bits[0, ::3] = True
    ator = RedeGradienteFuncional(n_bits=32, tamanho_tupla=4, n_saidas=1)
    ator.atualizar(bits, np.array([0.5]))
    critico = RedeRegressaoEsquecimento(n_bits=32, tamanho_tupla=4, n_saidas=1, phi=0.85)
    critico.treinar(bits, np.array([0.5]))

    ka = custo_ntuple_de_rede(ator, variante="ordenado", rotulo="a").detalhe["luts"]
    kc = custo_ntuple_de_rede(critico, variante="ordenado", rotulo="c").detalhe["luts"]
    # mesma ocupação e mesma n_saidas: a diferença é exatamente o contador
    assert kc > ka, "crítico não pode custar o mesmo que o ator"
    occ = sum(len(m) for m in critico.memoria)
    assert kc - ka == occ * 4, "o contador deve pesar 4 bytes (uint32) por entrada"


def test_carrega_checkpoint_antigo_com_contador_float(tmp_path):
    """Checkpoints gravados ANTES da troca para `uint32` têm contadores `float64`.

    Sem conversão no `carregar`, o `treinar` seguinte faria `c += 1` sobre um
    `float64` e reintroduziria o custo em silêncio — o modelo funcionaria, só
    gastaria o dobro. Este teste fixa a conversão.

    A predição tem de ser IDÊNTICA bit a bit: o contador sempre foi inteiro, e
    `1 − φ^c` não distingue `5.0` de `5`.
    """
    import numpy as np

    from uavwnn.wnn.redes import RedeRegressaoEsquecimento

    rede = RedeRegressaoEsquecimento(n_bits=32, tamanho_tupla=4, n_saidas=1, phi=0.85)
    bits = np.zeros((1, 32), bool)
    bits[0, ::3] = True
    for _ in range(7):
        rede.treinar(bits, np.array([2.0]))
    esperado = rede.prever(bits)

    # regrava no formato ANTIGO: contadores em float64
    rede.salvar(tmp_path / "c.npz")
    d = dict(np.load(tmp_path / "c.npz", allow_pickle=False))
    d["contadores"] = d["contadores"].astype(np.float64)
    np.savez_compressed(tmp_path / "antigo.npz", **d)

    lida = RedeRegressaoEsquecimento.carregar(tmp_path / "antigo.npz")
    _, c = next(iter(lida.memoria[0].values()))
    assert c.dtype == np.uint32, f"carregar não converteu: {c.dtype}"
    np.testing.assert_allclose(lida.prever(bits), esperado, rtol=0, atol=0)

    # e continuar treinando no checkpoint antigo tem de funcionar
    end = int(lida.enderecos(bits)[0][0])
    antes = int(lida.memoria[0][end][1][0])
    lida.treinar(bits, np.array([2.0]))
    assert int(lida.memoria[0][end][1][0]) == antes + 1


def test_lut_plana_e_equivalente_e_recusa_escrita():
    """A `LUTPlana` é o ator de inferência: mesma predição, memória contígua.

    A equivalência tem de ser EXATA quando os valores ficam em `float64` — a
    estrutura não introduz erro nenhum por si, só a escolha de precisão. Com
    `float32` o erro é o arredondamento do tipo, não da busca.

    E ela tem de RECUSAR escrita: as chaves estão ordenadas e a busca binária
    depende disso, então inserir exigiria reordenar. Degradar em silêncio seria
    pior que falhar.
    """
    import numpy as np
    import pytest

    from uavwnn.wnn.redes import LUTPlana, RedeGradienteFuncional

    rede = RedeGradienteFuncional(n_bits=256, tamanho_tupla=8, n_saidas=3, seed=3)
    rng = np.random.default_rng(0)
    for _ in range(40):
        bits = rng.random((1, 256)) > 0.5
        rede.atualizar(bits, rng.normal(0, 0.3, 3))

    prova = rng.random((5, 256)) > 0.5
    esperado = rede.prever(prova)

    exata = LUTPlana(rede, dtype_valor=np.float64)
    np.testing.assert_allclose(exata.prever(prova), esperado, rtol=0, atol=0)

    f32 = LUTPlana(rede)                      # padrão: float32
    np.testing.assert_allclose(f32.prever(prova), esperado, rtol=1e-6, atol=0)

    # a estrutura interna: chaves globais, ordenadas
    assert exata.chaves.dtype == np.int32
    assert (np.diff(exata.chaves) > 0).all(), "chaves têm de estar ordenadas"
    assert exata.chaves.size == sum(len(m) for m in rede.memoria)

    for metodo in ("atualizar", "treinar", "salvar"):
        with pytest.raises(TypeError, match="imutável"):
            getattr(exata, metodo)(prova, np.zeros(3))


def test_lut_plana_recusa_chave_grande_demais():
    """`n_ram · 2ⁿ` precisa caber em `int32`. Estourar em silêncio daria
    endereços negativos e uma rede que lê lixo sem levantar nada."""
    import numpy as np
    import pytest

    from uavwnn.wnn.redes import LUTPlana, RedeGradienteFuncional

    # n = 28 → 2^28 por RAM; com ~9 RAMs já passa de 2^31
    rede = RedeGradienteFuncional(n_bits=256, tamanho_tupla=28, n_saidas=1)
    with pytest.raises(ValueError, match="int32"):
        LUTPlana(rede)


def test_montar_com_plana_da_a_mesma_acao(tmp_path):
    """A bandeira do `montar` não pode mudar o comportamento da política."""
    import sys

    import numpy as np

    sys.path.insert(0, "scripts")
    from spec06_campanha_wnn import montar

    from uavwnn.refenv import LAMENZA_003

    cfg = LAMENZA_003(1, 12, phase="eval")
    ck = Path("models/deploy/T16/a1s12")
    if not ck.exists():
        import pytest
        pytest.skip("modelo de deploy ausente")

    p_dict, ap_d = montar("ntupla_dissertacao", cfg, ck)
    p_plana, ap_p = montar("ntupla_dissertacao", cfg, ck, plana=True)
    assert type(ap_p.ator).__name__ == "LUTPlana"

    rng = np.random.default_rng(7)
    for _ in range(20):
        obs = rng.random(cfg.observation_size)
        np.testing.assert_allclose(np.asarray(p_plana(obs), float),
                                   np.asarray(p_dict(obs), float),
                                   rtol=1e-6, atol=1e-9)
