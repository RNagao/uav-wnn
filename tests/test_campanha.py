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


# O `_grid()` e os três testes que dependiam dele saíram junto com
# `scripts/spec06_grid.py`, a varredura de dois eixos que o `spec06_busca.py`
# substituiu. Eles guardavam o pareamento do esforço de sintonia da T-06.3 —
# *"um pareamento em que a WNN foi varrida em 15 pontos e a MLP em 2 não é
# pareamento; é vantagem de esforço disfarçada de vantagem de método"* —, e o
# princípio migrou para `test_a_conversao_de_taxa_pareia_as_convencoes`, abaixo,
# que verifica o mecanismo que sobreviveu.


def test_a_conversao_de_taxa_pareia_as_convencoes():
    """A afirmação de justiça da grade, verificada em aritmética.

    Somar `passo` às `N` posições endereçadas muda `f` em `passo` na convenção de
    **média** (eq. 4.12) e em `passo·N` na de **soma**. A mesma taxa nominal dá um
    passo `N` vezes maior no braço de soma, e comparar sem converter mediria a
    CONVENÇÃO, não o aproximador.

    O `spec06_busca.py` converte por AMOSTRA, e é o eixo `n` que obriga: `N =
    ⌈25·T/n⌉` varia de 12 a 200 no espaço varrido, então uma tabela fixa de taxas
    equivalentes valeria para um único `(T, n)`.

    Substitui `test_a_taxa_do_anbubus_e_a_equivalente_na_convencao_de_soma`, que
    verificava o mesmo sobre o `spec06_grid.py`, removido.
    """
    import math

    from uavwnn.agents.ac_continuo import (
        NTuplaAnbubusAprox,
        NTuplaDissertacaoAprox,
    )

    passo = 5e-4
    for n_bits, tupla in ((400, 12), (800, 12), (7500, 10)):
        n_ram = math.ceil(n_bits / tupla)
        d = NTuplaDissertacaoAprox(n_bits=n_bits, tamanho_tupla=tupla,
                                   taxa_ator=passo)
        a = NTuplaAnbubusAprox(n_bits=n_bits, tamanho_tupla=tupla,
                               taxa_ator=passo / n_ram)
        # `ganho_do_passo` é 1 na média e `N` na soma: é ele que faz `c_max`
        # significar a mesma coisa nas duas, e é ele que o pareamento usa.
        df_d = d.ator.taxa_ator_efetiva if hasattr(d.ator, "taxa_ator_efetiva") \
            else passo * d.ator.ganho_do_passo
        df_a = (passo / n_ram) * a.ator.ganho_do_passo
        assert a.ator.n_ram == n_ram
        assert math.isclose(df_d, df_a, rel_tol=1e-9), (
            f"Δf difere entre as convenções em (n_bits={n_bits}, n={tupla}): "
            f"{df_d} contra {df_a}")


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



# --- revezamento de Na na busca (spec06_busca.py --na 1,2,4) -----------------


def test_celulas_revezadas_tem_a_mesma_largura_de_observacao(tmp_path):
    """Uma rede só vê vários `Na` se a observação tiver a MESMA largura em todos:
    `state_num_closest_drones = max(Na) − 1` em cada célula. Com um `Na` só, a
    largura é a de Lamenza (`2·(Na−1) + 2·12 + 1`), para não mudar a grade antiga."""
    b = _busca()
    cel = b._celulas("1b", (1, 2, 4), 12, raiz_bancos=tmp_path)
    larguras = {na: c["cfg_tr"].observation_size for na, c in cel.items()}
    assert larguras == {1: 31, 2: 31, 4: 31}          # 2·3 + 2·12 + 1
    assert all(c["cfg_tr"].state_num_closest_drones == 3 for c in cel.values())
    so_um = b._celulas("1b", (1,), 12, raiz_bancos=tmp_path)
    assert so_um[1]["cfg_tr"].observation_size == 25
    assert so_um[1]["cfg_tr"].state_num_closest_drones == 0


def test_cfgs_recusa_menos_vizinhos_que_na_menos_um():
    """O `environment.py` escreve `Na − 1` vizinhos nos slots; menos slots que isso
    estoura a atribuição lá dentro. Melhor falhar aqui, com mensagem."""
    import sys
    sys.path.insert(0, "scripts")
    from spec06_treinar import _cfgs

    with pytest.raises(ValueError, match="n_vizinhos"):
        _cfgs("1b", 4, 12, n_vizinhos=1)


def test_registro_de_na_preserva_o_hash_da_grade_antiga():
    """`Na = 1` grava o ESCALAR `1`, como as 720 corridas existentes; só a lista
    aparece com revezamento. E `_nas_de` lê os dois, mais o registro antigo sem
    campo."""
    b = _busca()
    assert b._na_registro((1,)) == 1
    assert b._na_registro((1, 2, 4)) == [1, 2, 4]
    assert b._nas_de({"na": 1}) == (1,)
    assert b._nas_de({"na": [1, 2, 4]}) == (1, 2, 4)
    assert b._nas_de({}) == (1,)


def test_revezamento_alterna_na_por_episodio_e_treina_cada_drone(tmp_path):
    """Round-robin sobre `(1, 2)`: o episódio `k` roda com `nas[k mod 2]` drones, e
    cada drone fecha a sua própria transição — o laço single-agente registrava uma
    só por passo e treinava apenas o último drone."""
    import sys
    sys.path.insert(0, "scripts")
    from spec06_treinar import RECOMPENSAS, REPLICAS
    from uavwnn.agents import AtorCriticoContinuo
    from uavwnn.agents.politica_continua import NormalProjetada
    from uavwnn.campanha import PapeisDeSemente

    b = _busca()
    nas = (1, 2)
    cel = b._celulas("1b", nas, 12, raiz_bancos=tmp_path)
    amostra = {"passo_efetivo": 5e-4, "sigma": 0.35, "resolucao": 8, "tupla": 8,
               "passos_m": 4, "idx_critico": 1, "normalizar": False}
    sem = PapeisDeSemente(seed_train=0)
    corpus = np.concatenate([b.corpus_de_observacoes(c["cfg_tr"], n_episodios=1)
                             for c in cel.values()])
    ap, _termo, pol, _det = b._montar("ntupla_dissertacao", cel[1]["cfg_tr"], sem,
                                      amostra, corpus=corpus)
    acs = [pol, AtorCriticoContinuo(aproximador=ap, codificar=pol.codificar,
                                    politica=NormalProjetada(0.35, None), passos=4,
                                    id="a1", treinando=True)]
    registrados = {0: 0, 1: 0}
    for i, ac in enumerate(acs):
        orig = ac.registrar

        def _reg(r, _o=orig, _i=i):
            registrados[_i] += 1
            return _o(r)
        ac.registrar = _reg

    r = REPLICAS["1b"]
    info = b.rodar_episodios_revezando(
        cel, nas, acs, RECOMPENSAS[r["recompensa"]], sem.rng_cenarios_treino(),
        orcamento=120, escala=r["escala_recompensa"])
    assert info["passos"] >= 120
    por_na = info["episodios_por_na"]
    assert por_na[1] >= 1 and por_na[2] >= 1
    assert abs(por_na[1] - por_na[2]) <= 1          # round-robin
    # o segundo drone só existe nos episódios de Na=2, e registra em todos eles
    assert 0 < registrados[1] < registrados[0]
    assert ap.ator.entradas_ocupadas() > 0
