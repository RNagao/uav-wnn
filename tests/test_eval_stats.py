"""Harness estatístico — SPEC-02 T-02.5 a T-02.7 e T-02.9.

O teste que de fato calibra o bootstrap é o **A/A**, e ele vive em
`scripts/spec02_aa.py` porque leva minutos. Aqui ficam as propriedades que dão
para verificar em milissegundos, e uma versão reduzida do A/A que roda na suíte.

O que estes testes protegem, em ordem de gravidade:

1. **O pareamento não pode ser destruído.** Reamostrar os dois métodos com
   índices independentes infla o IC e faz diferenças reais desaparecerem. É um
   erro de duas linhas que passa em todo teste óbvio.
2. **`t_conc_pen` não pode premiar quem trava cedo.** Imputar o instante da poda
   daria tempo baixo a uma política que falha rápido.
3. **LB-1 tem de ser um limite de verdade.** Resultado abaixo dele é bug, e o
   harness assere; se LB-1 estiver errado, a asserção mente.
4. **As sementes de treino e avaliação têm de ser disjuntas.** Sobreposição é
   vazamento silencioso.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.eval import (
    MAX_EPISODE_LENGTH,
    ArvoreSementes,
    MetricasEpisodio,
    bootstrap_media,
    bootstrap_pareado,
    disjuntos,
    gini,
    holm,
    lb1,
    mcnemar,
    mde_95,
    resumir_episodios,
    sucesso_comum,
    wilcoxon_pareado,
)


def _ep(idx=0, sucesso=True, t=30.0, visitas=(3, 3), lb=5.0, ns=12):
    return MetricasEpisodio(
        idx_episodio=idx, seed_cenario=idx, sucesso=sucesso, n_sensores=ns,
        n_visitados=ns if sucesso else 1,
        t_conc_s=t if sucesso else float("nan"),
        t_fim_s=t, motivo_fim="sucesso" if sucesso else "stalled",
        n_decisoes=10, visitas_por_agente=list(visitas), lb1_s=lb,
    )


# --- árvore de sementes (T-02.6) ------------------------------------------


def test_treino_e_avaliacao_sao_disjuntos():
    """Avaliar num cenário visto no treino é vazamento silencioso, e é o que
    `seed = i` nos dois ramos produziria."""
    arv = ArvoreSementes(20260809)
    assert disjuntos(arv.cenarios("treino", 500), arv.cenarios("avaliacao", 500))


def test_sementes_sao_reprodutiveis_entre_processos():
    """A raiz vai para o manifesto; com ela o resultado tem de voltar igual.

    `hash()` embutido é aleatorizado por processo, então derivar de
    `hash(policy.id)` daria sementes diferentes a cada execução. Este teste
    quebraria se alguém trocasse o sha256 por `hash()`.
    """
    a = ArvoreSementes(42).cenarios("avaliacao", 20)
    b = ArvoreSementes(42).cenarios("avaliacao", 20)
    assert a == b
    assert ArvoreSementes(43).cenarios("avaliacao", 20) != a


def test_semente_de_politica_independe_entre_metodos():
    """Parear o fluxo da política é ilusão: métodos diferentes consomem
    quantidades diferentes de números aleatórios. O pareamento é no CENÁRIO."""
    arv = ArvoreSementes(7)
    a = arv.politica("guloso", 3).normal(size=5)
    b = arv.politica("aleatorio", 3).normal(size=5)
    assert not np.allclose(a, b)
    # ...mas cada uma é reprodutível
    assert np.allclose(a, ArvoreSementes(7).politica("guloso", 3).normal(size=5))


def test_ramo_desconhecido_falha_cedo():
    with pytest.raises(ValueError, match="ramo desconhecido"):
        ArvoreSementes(1)._ramo("inventado")


# --- métrica primária (T-02.5, D-02.2) -----------------------------------


def test_falha_e_imputada_em_max_episode_length():
    assert _ep(sucesso=True, t=42.0).t_conc_pen_s == 42.0
    assert _ep(sucesso=False, t=8.0).t_conc_pen_s == MAX_EPISODE_LENGTH


def test_travar_cedo_nao_e_premiado():
    """O erro que D-02.2 existe para evitar: imputar o instante da poda daria
    tempo BAIXO a quem falha rápido, premiando o pior comportamento."""
    trava_cedo = _ep(sucesso=False, t=5.0)
    conclui_tarde = _ep(sucesso=True, t=480.0)
    assert trava_cedo.t_conc_pen_s > conclui_tarde.t_conc_pen_s


def test_a_metrica_e_monotona_nas_duas_dimensoes():
    """Falhar é sempre pior que concluir; concluir mais cedo é sempre melhor."""
    assert _ep(sucesso=True, t=10).t_conc_pen_s < _ep(sucesso=True, t=20).t_conc_pen_s
    for t in (1.0, 100.0, 499.0):
        assert _ep(sucesso=True, t=t).t_conc_pen_s < _ep(sucesso=False).t_conc_pen_s


def test_resumo_traz_o_conjunto_nao_so_a_primaria():
    """A primária sozinha esconde o mecanismo — a distribuição é assimétrica e
    censurada, então mediana e p90 vêm junto."""
    eps = [_ep(0, True, 20), _ep(1, True, 30), _ep(2, False), _ep(3, True, 40)]
    r = resumir_episodios(eps)
    assert r.n_ep == 4 and r.n_sucesso == 3
    assert r.taxa_sucesso == pytest.approx(0.75)
    assert r.media_t_conc_pen == pytest.approx((20 + 30 + 500 + 40) / 4)
    assert r.media_t_conc_sucesso == pytest.approx(30.0)
    assert r.p90_t_conc_pen > r.mediana_t_conc_pen


def test_media_sobre_sucessos_tem_vies_de_sobrevivencia():
    """Por isso ela carrega aviso e não serve para comparar métodos com taxas de
    sucesso diferentes: é média sobre os cenários que cada um CONSEGUIU."""
    facil = [_ep(i, True, 20) for i in range(10)]
    dificil = [_ep(i, i < 2, 20) for i in range(10)]     # só 2 sucessos, fáceis
    ra, rb = resumir_episodios(facil), resumir_episodios(dificil)
    assert ra.media_t_conc_sucesso == pytest.approx(rb.media_t_conc_sucesso)
    assert rb.media_t_conc_pen > ra.media_t_conc_pen     # a primária não engana


def test_sucesso_comum_degenera_com_politica_ruim():
    """Com a caminhada aleatória na comparação o conjunto fica vazio, e a T-02.5
    manda não reportar — daí devolver máscara em vez de já agregar."""
    bom = [_ep(i, True) for i in range(10)]
    ruim = [_ep(i, False) for i in range(10)]
    assert sucesso_comum(bom, bom).sum() == 10
    assert sucesso_comum(bom, ruim).sum() == 0


# --- LB-1 (T-02.4d) ------------------------------------------------------


def test_lb1_desconta_o_alcance_de_comunicacao():
    """Não se chega ao sensor — chega-se ao disco de `communication_range`."""
    d = 100.0
    t = lb1(np.array([[d, 0.0]]), np.array([[0.0, 0.0]]), 20.0, 15.0)
    assert t == pytest.approx((d - 20.0) / 15.0)


def test_lb1_e_zero_quando_o_sensor_ja_esta_no_alcance():
    assert lb1(np.array([[5.0, 0.0]]), np.array([[0.0, 0.0]]), 20.0, 15.0) == 0.0


def test_lb1_usa_o_agente_mais_proximo_e_o_sensor_mais_distante():
    """`max_i min_k` — o sensor pior servido decide, e cada um é servido pelo
    agente mais próximo."""
    sensores = np.array([[100.0, 0.0], [0.0, 300.0]])
    agentes = np.array([[100.0, 0.0], [0.0, 0.0]])
    t = lb1(sensores, agentes, 20.0, 15.0)
    assert t == pytest.approx((300.0 - 20.0) / 15.0)


def test_lb1_e_limite_valido_no_caso_de_um_agente_um_sensor():
    """Sanidade: com um agente e um sensor, LB-1 é exato — o agente voa reto."""
    d, alc, v = 200.0, 20.0, 15.0
    assert lb1(np.array([[d, 0.0]]), np.array([[0.0, 0.0]]), alc, v) == \
        pytest.approx((d - alc) / v)


# --- Gini (T-02.5) -------------------------------------------------------


def test_gini_distingue_divisao_de_trabalho():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert gini([20, 0, 0, 0]) == pytest.approx(0.75)
    assert gini([10, 5, 3, 2]) > gini([6, 5, 5, 4])


def test_gini_de_amostra_degenerada_e_nan_ou_zero():
    assert np.isnan(gini([]))
    assert np.isnan(gini([0, 0]))
    assert gini([7]) == 0.0


# --- bootstrap pareado (T-02.7) ------------------------------------------


def test_pareamento_da_ic_muito_mais_estreito_que_independencia():
    """**O ponto do módulo.** Com efeito de cenário compartilhado, o pareamento
    cancela a variância entre cenários. Reamostrar independentemente a
    reintroduz e infla o IC — a diferença real desaparece.
    """
    rng = np.random.default_rng(0)
    n = 500
    cenario = rng.lognormal(np.log(30), 0.6, n)     # variância grande, compartilhada
    a = cenario + rng.normal(0, 2, n)
    b = cenario + rng.normal(2.0, 2, n)             # B é 2 s pior, de verdade

    pareado = bootstrap_pareado(a, b, rng)
    largura_par = pareado.delta_ic_hi - pareado.delta_ic_lo

    idx_a = rng.integers(0, n, size=(1000, n))
    idx_b = rng.integers(0, n, size=(1000, n))
    deltas = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    assert largura_par < (hi - lo) / 3, "o pareamento deveria estreitar muito"
    assert pareado.conclusivo, "com pareamento, a diferença de 2 s é detectável"
    assert lo <= 0 <= hi, "sem pareamento, ela desaparece"


def test_pareamento_exige_mesmo_tamanho():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        bootstrap_pareado([1.0, 2.0], [1.0], rng)


def test_regra_de_decisao_e_zero_fora_do_ic():
    rng = np.random.default_rng(1)
    igual = bootstrap_pareado(rng.normal(30, 5, 300), rng.normal(30, 5, 300), rng)
    assert igual.contem_zero and not igual.conclusivo


def test_taxa_de_vitoria_respeita_menor_e_melhor():
    """Em `t_conc_pen`, A vence quando conclui MAIS CEDO."""
    rng = np.random.default_rng(2)
    a, b = np.full(100, 10.0), np.full(100, 20.0)
    assert bootstrap_pareado(a, b, rng).taxa_vitoria_a == pytest.approx(1.0)
    assert bootstrap_pareado(a, b, rng, menor_e_melhor=False).taxa_vitoria_a == 0.0


def test_mde_cresce_com_a_variancia_e_cai_com_n():
    """Sem MDE, um resultado nulo é ininterpretável."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 2000)
    pequeno = mde_95(base[:100], base[:100] + rng.normal(0, 1, 100))
    grande = mde_95(base, base + rng.normal(0, 1, 2000))
    assert grande < pequeno
    ruidoso = mde_95(base[:100], base[:100] + rng.normal(0, 10, 100))
    assert ruidoso > pequeno


def test_ic_marginal_sobreposto_nao_implica_ausencia_de_diferenca():
    """O aviso metodológico da T-02.7, como teste.

    Constrói o caso em que os ICs marginais se sobrepõem largamente e a
    diferença pareada é conclusiva. É por isso que nenhuma afirmação
    comparativa do projeto usa sobreposição de barras.
    """
    rng = np.random.default_rng(4)
    n = 400
    cenario = rng.lognormal(np.log(40), 0.7, n)
    a = cenario
    b = cenario + 1.5

    ma = bootstrap_media(a, rng)
    mb = bootstrap_media(b, rng)
    sobrepoem = ma.ic_lo <= mb.ic_hi and mb.ic_lo <= ma.ic_hi
    assert sobrepoem, "o caso construído precisa ter ICs marginais sobrepostos"
    assert bootstrap_pareado(a, b, rng).conclusivo, "e diferença pareada real"


# --- A/A reduzido, na suíte ----------------------------------------------


def test_aa_reduzido_tem_cobertura_perto_do_nominal():
    """Versão barata do A/A de `scripts/spec02_aa.py`.

    60 execuções em vez de 200 — o erro padrão é ~2,8 pp, então a faixa é larga
    de propósito. Não substitui o script; existe para que uma regressão no
    pareamento apareça na suíte, e não só quando alguém lembrar de rodar o A/A.
    """
    rng = np.random.default_rng(11)
    n, execucoes, contidos = 300, 60, 0
    for _ in range(execucoes):
        cenario = rng.lognormal(np.log(30), 0.55, n)
        a = cenario + rng.normal(0, 4, n)
        b = cenario + rng.normal(0, 4, n)          # mesma distribuição
        d = bootstrap_pareado(a, b, rng, B=400)
        contidos += int(d.contem_zero)
    cobertura = contidos / execucoes
    assert 0.85 < cobertura < 1.0, f"cobertura {cobertura:.1%} longe de 95 %"


# --- apoios --------------------------------------------------------------


def test_wilcoxon_e_nan_quando_nao_ha_diferenca():
    """Indefinido com todas as diferenças nulas; `nan` em vez de exceção."""
    x = np.arange(10.0)
    assert np.isnan(wilcoxon_pareado(x, x))
    assert wilcoxon_pareado(x, x + 1.0) < 0.05


def test_mcnemar_conta_discordantes():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    r = mcnemar(a, b)
    assert r["b"] == 2 and r["c"] == 1
    assert r["delta_taxa_sucesso"] == pytest.approx(0.2)


def test_mcnemar_sem_discordantes_e_nan():
    a = [True, False, True]
    assert np.isnan(mcnemar(a, a)["p_mcnemar"])


def test_holm_e_monotono_e_nunca_reduz_p():
    p = [0.001, 0.02, 0.04, 0.5]
    aj = holm(p)
    assert np.all(aj >= np.array(p))
    assert np.all(np.diff(aj[np.argsort(p)]) >= -1e-12), "não pode decrescer"
    assert aj.max() <= 1.0
