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


# --- §4.4 parada antecipada -----------------------------------------------


def test_parada_nao_dispara_com_metrica_achatada_no_zero():
    """A correção ao critério da §4.4, guardada por teste.

    O critério como escrito disparava com `SR = 0` no início do treino, declarando
    "convergiu" numa corrida que não havia começado a aprender — medido numa corrida
    de fumaça: 1950 de 6000 passos. Confundir "platô antes" com "platô depois"
    libera orçamento de quem mais precisa dele.
    """
    from uavwnn.campanha.treino import MINIMO_AVALIACOES, ParadaAntecipada

    p = ParadaAntecipada(maior_e_melhor=True)
    for i in range(MINIMO_AVALIACOES - 1):
        assert not p.registrar(0.0), f"parou na avaliação {i + 1} com métrica zero"


def test_parada_dispara_depois_de_platô_real():
    """Sobe, estabiliza, e aí sim para."""
    from uavwnn.campanha.treino import MINIMO_AVALIACOES, ParadaAntecipada

    p = ParadaAntecipada(maior_e_melhor=True)
    subida = [0.1 * i for i in range(1, 11)]
    parou_em = None
    for i, v in enumerate(subida + [1.0] * 40):
        if p.registrar(v):
            parou_em = i + 1
            break
    assert parou_em is not None, "nunca parou, mesmo com platô longo"
    assert parou_em >= MINIMO_AVALIACOES


def test_parada_respeita_a_direcao_da_metrica():
    """`SR` é maior-é-melhor e `CT_cens` é menor-é-melhor.

    Deixar a direção implícita faria a parada disparar de imediato numa das duas —
    uma métrica que cai seria lida como "sem melhora".
    """
    from uavwnn.campanha.treino import ParadaAntecipada

    caindo = [50.0 - i for i in range(30)]
    p_menor = ParadaAntecipada(maior_e_melhor=False)
    assert not any(p_menor.registrar(v) for v in caindo), \
        "menor-é-melhor parou enquanto a métrica ainda caía"
    p_maior = ParadaAntecipada(maior_e_melhor=True)
    assert any(p_maior.registrar(v) for v in caindo), \
        "maior-é-melhor não parou com a métrica piorando"


# --- a grade de hiperparâmetros (T-06.4/T-06.3) ---------------------------


def _grid():
    import importlib.util
    import pathlib
    import sys

    caminho = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
               / "spec06_grid.py")
    spec = importlib.util.spec_from_file_location("_grid", caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_grid"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_os_tres_modelos_recebem_o_mesmo_numero_de_pontos():
    """T-06.3: *"um pareamento em que a WNN foi varrida em 15 pontos e a MLP em 2
    não é pareamento; é vantagem de esforço disfarçada de vantagem de método."*

    O `φ` do crítico com esquecimento fica FORA da grade justamente por isso —
    existe em um dos três e daria 12 pontos a ele contra 6 aos outros.
    """
    g = _grid()
    tamanhos = {m: len(t) * len(g.SIGMAS) for m, t in g.TAXAS.items()}
    assert len(set(tamanhos.values())) == 1, tamanhos
    assert set(g.TAXAS) == {"ntupla_dissertacao", "ntupla_anbubus", "mlp"}


def test_a_taxa_do_anbubus_e_a_equivalente_na_convencao_de_soma():
    """A afirmação de justiça da grade, verificada em aritmética.

    Somar `passo` às `N` posições endereçadas muda `f` em `passo` na convenção de
    **média** (eq. 4.12) e em `passo·N` na de **soma**. Então a mesma taxa nominal dá
    um passo `N` vezes maior no braço do `anbubus`, e comparar sem converter mediria
    a **convenção**, não o aproximador.

    Se alguém editar uma das duas faixas sem a outra, a comparação passa a ser
    injusta **sem nenhum erro aparecer** — e é isso que este teste impede.
    """
    from uavwnn.wnn.redes import RedeGradienteFuncional

    g = _grid()
    n_bits = 25 * g.BITS_POR_ESCALAR
    n_ram = RedeGradienteFuncional(n_bits, g.TAMANHO_TUPLA, seed=0).n_ram

    for media, soma in zip(g.TAXAS["ntupla_dissertacao"], g.TAXAS["ntupla_anbubus"]):
        esperado = media / n_ram
        assert soma == pytest.approx(esperado, rel=0.02), (
            f"taxa de soma {soma:.3e} não equivale a {media:.3e} da média "
            f"(esperado {esperado:.3e} com N={n_ram}); o passo efetivo em `f` "
            f"difere e a grade deixa de medir o aproximador")


def test_as_faixas_cobrem_a_mesma_razao():
    """Mesma razão em cada família é o que torna as faixas comparáveis apesar de
    valores diferentes — a MLP usa Adam e o `anbubus` usa soma, então valores iguais
    não significam esforço igual."""
    g = _grid()
    razoes = {m: max(t) / min(t) for m, t in g.TAXAS.items()}
    assert max(razoes.values()) / min(razoes.values()) < 1.05, razoes


# --- normalização móvel da observação (o MeanStdFilter deles) --------------


def test_normalizador_estima_media_e_desvio():
    """Welford, não soma dos quadrados: com milhões de passos a forma acumulada
    perde precisão quando a média é grande em relação ao desvio."""
    from uavwnn.campanha import NormalizadorMovel

    g = np.random.default_rng(0)
    nz = NormalizadorMovel(3)
    for _ in range(4000):
        nz.atualizar(g.normal(loc=[5.0, -2.0, 100.0], scale=[3.0, 0.5, 0.1], size=3))
    assert np.allclose(nz.media, [5.0, -2.0, 100.0], atol=0.15)
    assert np.allclose(nz.desvio, [3.0, 0.5, 0.1], rtol=0.1)


def test_congelar_impede_vazamento_na_avaliacao():
    """A estatística é estado TREINADO. Se ela seguisse acumulando durante a
    avaliação, o agente se calibraria nos episódios de reporte — vazamento pela
    porta de trás, sem tocar em pesos."""
    from uavwnn.campanha import NormalizadorMovel

    nz = NormalizadorMovel(2)
    g = np.random.default_rng(1)
    for _ in range(50):
        nz.atualizar(g.normal(size=2))
    n_antes, media_antes = nz.n, nz.media.copy()

    nz.congelar()
    for _ in range(500):
        nz.aplicar(g.normal(loc=99.0, size=2))     # distribuição MUITO diferente
    assert nz.n == n_antes
    assert np.array_equal(nz.media, media_antes)

    nz.descongelar()
    nz.aplicar(g.normal(size=2))
    assert nz.n == n_antes + 1


def test_normalizador_persiste_bit_a_bit(tmp_path):
    """Vai no checkpoint: um modelo carregado sem a estatística veria a observação
    numa escala diferente da que treinou."""
    from uavwnn.campanha import NormalizadorMovel

    g = np.random.default_rng(2)
    nz = NormalizadorMovel(5)
    for _ in range(300):
        nz.atualizar(g.normal(loc=3.0, scale=2.0, size=5))
    x = g.normal(size=5)
    antes = nz.aplicar(x, atualizar=False)

    nz.salvar(tmp_path / "nz.npz")
    outro = NormalizadorMovel.carregar(tmp_path / "nz.npz")
    assert np.array_equal(outro.aplicar(x, atualizar=False), antes)
    assert outro.congelado, "carregar deve devolver congelado — é avaliação"


def _busca():
    """Carrega `spec06_busca.py` — onde vive `GRADE`.

    Distinto de `_grid()`, que carrega `spec06_grid.py` (a varredura de dois eixos).
    Confundir os dois foi o que quebrou este teste na primeira escrita.
    """
    import importlib.util
    import pathlib
    import sys

    caminho = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
               / "spec06_busca.py")
    spec = importlib.util.spec_from_file_location("_busca", caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_busca"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_o_eixo_de_normalizacao_esta_nas_duas_grades():
    """O eixo existe porque o braço do `anbubus` estagnou com o NOSSO codificador
    enquanto todas as configurações dos autores usam `MeanStdFilter`. Sem ele não há
    como distinguir "saturou" de "codificador errado"."""
    grade = _busca().GRADE
    for nivel in ("reduzido", "completo"):
        assert "normalizar" in grade[nivel]
        assert set(grade[nivel]["normalizar"]) == {False, True}


def test_a_faixa_padronizada_e_a_dos_autores():
    """`min: -1.5, max: 1.5` em TODAS as configurações do `anbubus`, e a faixa só
    faz sentido junto com a normalização: são ±1,5 DESVIOS."""
    from uavwnn.campanha import FAIXA_PADRONIZADA

    assert FAIXA_PADRONIZADA == (-1.5, 1.5)
