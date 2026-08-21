"""Interface `Policy` e os três baselines — SPEC-02 T-02.1 a T-02.4, T-02.9.

Os quatro testes de contrato da T-02.1 são os que a SPEC-05 vai herdar, então
falham alto se alguém mudar a moeda da ação ou o regime de privilégio.

O teste que mais importa aqui é o **de sinal** (T-02.9 §3): com o sinal da
observação invertido o guloso *foge* do sensor, o episódio falha por parada, e
isso passaria por "heurística ruim" em vez de bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.agents import (
    PADDING,
    V_MAX,
    Acao,
    AdaptadorRunner,
    CaminhadaAleatoria,
    ContextoEpisodio,
    EstadoVerdadeiro,
    Guloso,
    KMeansTSP,
    Observacao,
    Policy,
    PoliticaPrivilegiada,
    held_karp,
    vizinho_2opt,
)

S = 100.0


def _ctx(na=2, ns=12):
    return ContextoEpisodio(num_drones=na, num_sensores=ns, scenario_size=S,
                           communication_range=20.0, dt=0.5)


def _vetor(ctx, sensores_rel=(), agentes_rel=(), k=0):
    """Monta a observação canônica a partir de posições RELATIVAS em metros.

    Inverte a normalização de referência: `v = (p_k − p_outro + S)/(2S)`, e
    `p_k − p_outro = −rel`.
    """
    n_a = ctx.num_drones - 1
    n_s = ctx.state_num_closest_sensors
    v = np.full(2 * n_a + 2 * n_s + 1, PADDING, dtype=float)
    for i, rel in enumerate(agentes_rel):
        v[2 * i:2 * i + 2] = (-np.asarray(rel, float) + S) / (2 * S)
    base = 2 * n_a
    for i, rel in enumerate(sensores_rel):
        v[base + 2 * i:base + 2 * i + 2] = (-np.asarray(rel, float) + S) / (2 * S)
    v[-1] = k / ctx.num_drones
    return v


# --- T-02.1 regra 1: a ação é normalizada -------------------------------


def test_acao_fora_de_zero_um_falha():
    """A desnormalização é do ambiente (`θ = a₀·2π`, `v = a₁·15`). Deixar a
    política emitir fora de faixa quebraria a moeda comum."""
    Acao(0.0, 0.0), Acao(1.0, 1.0)
    for t, s in ((-0.01, 0.5), (1.01, 0.5), (0.5, -0.1), (0.5, 1.5)):
        with pytest.raises(ValueError, match="fora de"):
            Acao(t, s)


def test_de_rumo_envolve_o_angulo():
    """`θ` é circular. Aqui o wrap é trivial porque a ação é escolhida; na
    política gaussiana não é, e essa é a pendência C1."""
    assert Acao.de_rumo(0.0).theta_norm == pytest.approx(0.0)
    assert Acao.de_rumo(np.pi).theta_norm == pytest.approx(0.5)
    assert Acao.de_rumo(2 * np.pi).theta_norm == pytest.approx(0.0)
    assert Acao.de_rumo(-np.pi / 2).theta_norm == pytest.approx(0.75)


def test_de_rumo_satura_a_velocidade():
    assert Acao.de_rumo(0.0, V_MAX).s_norm == pytest.approx(1.0)
    assert Acao.de_rumo(0.0, 2 * V_MAX).s_norm == pytest.approx(1.0)
    assert Acao.de_rumo(0.0, 0.0).s_norm == 0.0


# --- T-02.1 regra 2: `vetor` é canônico, o resto é vista ----------------


def test_decodificacao_e_ida_e_volta_exata():
    ctx = _ctx(na=3)
    sensores = [(30.0, 0.0), (0.0, -45.0)]
    agentes = [(50.0, 50.0), (-20.0, 10.0)]
    o = Observacao.decodificar(_vetor(ctx, sensores, agentes, k=1), ctx)
    assert np.allclose(o.p_s[:2], sensores)
    assert np.allclose(o.p_a[:2], agentes)
    assert o.k == 1


def test_padding_vira_invalido_e_nao_zero():
    """Padding é `-1` e fica **fora** da faixa dos válidos — é o que dispensa o
    bit de validade. Confundi-lo com um sensor na origem seria um alvo falso."""
    ctx = _ctx()
    o = Observacao.decodificar(_vetor(ctx, [(10.0, 10.0)]), ctx)
    assert o.valido_s.sum() == 1
    assert np.isnan(o.p_s[1:]).all(), "slots vazios não podem virar coordenada"


def test_observacao_de_tamanho_errado_falha_cedo():
    ctx = _ctx()
    with pytest.raises(ValueError, match="esperado"):
        Observacao.decodificar(np.zeros(10), ctx)


def test_um_unico_drone_nao_tem_slot_de_agente():
    """`Na = 1` dá `n_a = 0`: a observação é 0 + 24 + 1 = 25."""
    ctx = _ctx(na=1)
    o = Observacao.decodificar(_vetor(ctx, [(5.0, 0.0)]), ctx)
    assert o.vetor.size == 25
    assert o.p_a.shape == (0, 2)


# --- T-02.9 §3: o teste de sinal ----------------------------------------


@pytest.mark.parametrize("rel,graus", [
    ((10.0, 0.0), 0.0), ((0.0, 10.0), 90.0),
    ((-10.0, 0.0), 180.0), ((0.0, -10.0), 270.0),
])
def test_guloso_aponta_PARA_o_sensor(rel, graus):
    """**O modo de falha desta política é a inversão de sinal.**

    Com o sinal trocado o UAV foge do sensor, o episódio falha por parada, e o
    resultado passaria por "heurística ruim" em vez de bug — que é exatamente
    por que a T-02.9 §3 pede este teste.
    """
    ctx = _ctx()
    o = Observacao.decodificar(_vetor(ctx, [rel]), ctx)
    a = Guloso().act(o)
    assert a.theta_norm * 360.0 == pytest.approx(graus, abs=1e-6)
    assert a.s_norm == pytest.approx(1.0)


def test_guloso_usa_o_primeiro_slot_que_ja_vem_ordenado():
    """A observação de referência ordena por distância crescente, então não é
    preciso recalcular — e recalcular com o sinal errado seria outro caminho
    para o mesmo bug."""
    ctx = _ctx()
    o = Observacao.decodificar(_vetor(ctx, [(5.0, 0.0), (0.0, 80.0)]), ctx)
    assert Guloso().act(o).theta_norm == pytest.approx(0.0)


def test_guloso_para_quando_nao_ha_sensor():
    """Todos coletados, ou padding total: parar em vez de vaguear."""
    ctx = _ctx()
    o = Observacao.decodificar(_vetor(ctx, []), ctx)
    assert Guloso().act(o).s_norm == 0.0


def test_distancia_ao_sensor_decresce_sob_guloso():
    """Versão dinâmica do teste de sinal: simula o avanço e confere que a
    distância cai monotonicamente."""
    ctx = _ctx(na=1)
    pos = np.array([0.0, 0.0])
    alvo = np.array([90.0, 40.0])
    g = Guloso()
    g.reset(ctx, np.random.default_rng(0))
    anterior = np.linalg.norm(alvo - pos)
    for _ in range(12):
        o = Observacao.decodificar(_vetor(ctx, [tuple(alvo - pos)]), ctx)
        a = g.act(o)
        theta = a.theta_norm * 2 * np.pi
        pos = pos + np.array([np.cos(theta), np.sin(theta)]) * a.s_norm * V_MAX * ctx.dt
        d = np.linalg.norm(alvo - pos)
        assert d < anterior + 1e-9, "a distância deveria decrescer"
        anterior = d


# --- T-02.9 §4: privilégio é declarado ----------------------------------


def test_flags_de_privilegio_estao_corretas():
    assert CaminhadaAleatoria().usa_estado_verdadeiro is False
    assert Guloso().usa_estado_verdadeiro is False
    assert KMeansTSP().usa_estado_verdadeiro is True


def test_baselines_sem_privilegio_nao_expoem_observar_verdade():
    """O runner só chama `observar_verdade` se a flag for `True`. Antecipa a
    RF-04: a verdade não chega à política candidata a implantação."""
    for pol in (CaminhadaAleatoria(), Guloso()):
        assert not isinstance(pol, PoliticaPrivilegiada)
        assert not hasattr(pol, "observar_verdade")
    assert isinstance(KMeansTSP(), PoliticaPrivilegiada)


def test_todos_os_baselines_satisfazem_o_protocolo():
    for pol in (CaminhadaAleatoria(), Guloso(), KMeansTSP()):
        assert isinstance(pol, Policy)
        assert isinstance(pol.metadados(), dict)
        assert "deterministica" in pol.metadados()


# --- T-02.1 regra 5: a política nunca cria o seu rng --------------------


def test_caminhada_aleatoria_usa_o_rng_recebido():
    """Fonte única de aleatoriedade. Se a política criasse o seu, o resultado
    deixaria de ser reprodutível pela raiz do manifesto."""
    ctx = _ctx()
    o = Observacao.decodificar(_vetor(ctx, [(1.0, 1.0)]), ctx)
    a = CaminhadaAleatoria()
    a.reset(ctx, np.random.default_rng(7))
    x = [a.act(o).theta_norm for _ in range(5)]
    b = CaminhadaAleatoria()
    b.reset(ctx, np.random.default_rng(7))
    assert x == [b.act(o).theta_norm for _ in range(5)]


def test_act_sem_reset_falha():
    ctx = _ctx()
    o = Observacao.decodificar(_vetor(ctx, [(1.0, 1.0)]), ctx)
    with pytest.raises(AssertionError, match="reset"):
        CaminhadaAleatoria().act(o)


# --- T-02.4b: as rotas --------------------------------------------------


def test_held_karp_acha_o_caminho_aberto_otimo():
    """Três pontos colineares em 0-1-2: o ótimo é a ordem natural."""
    d = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    custo, rota = held_karp(d)
    assert custo == pytest.approx(2.0)
    assert rota == [0, 1, 2]


def test_held_karp_prefere_desviar_quando_compensa():
    """Caminho aberto não volta à origem, então a ordem ótima não é
    necessariamente a mais próxima primeiro."""
    pontos = np.array([[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [11.0, 0.0]])
    d = np.linalg.norm(pontos[:, None] - pontos[None], axis=2)
    custo, rota = held_karp(d)
    assert rota == [0, 2, 1, 3]
    assert custo == pytest.approx(11.0)


def test_held_karp_degenera_com_zero_e_um_ponto():
    assert held_karp(np.zeros((1, 1))) == (0.0, [0])
    c, r = held_karp(np.array([[0.0, 5.0], [5.0, 0.0]]))
    assert c == pytest.approx(5.0) and r == [0, 1]


def test_heuristico_bate_o_exato_em_instancia_pequena():
    """O gap tem de ser medido, senão "TSP" na tabela é uma palavra."""
    rng = np.random.default_rng(3)
    pontos = rng.uniform(-100, 100, size=(9, 2))
    d = np.linalg.norm(pontos[:, None] - pontos[None], axis=2)
    exato, _ = held_karp(d)
    heur, _ = vizinho_2opt(d, rng, reinicios=12)
    assert heur >= exato - 1e-9
    assert (heur - exato) / exato < 0.05, "2-opt deveria ficar a <5 % do exato"


# --- T-02.4a: k-means e o cluster vazio --------------------------------


def test_kmeans_tsp_planeja_e_aponta_para_o_waypoint():
    ctx = _ctx(na=1, ns=3)
    pol = KMeansTSP()
    pol.reset(ctx, np.random.default_rng(0))
    estado = EstadoVerdadeiro(
        pos_sensores=np.array([[50.0, 0.0], [0.0, 50.0], [-50.0, 0.0]]),
        pos_agentes=np.array([[0.0, 0.0]]),
        coletado=np.zeros(3, bool),
    )
    pol.observar_verdade(estado)
    o = Observacao.decodificar(_vetor(ctx, [(50.0, 0.0)]), ctx)
    a = pol.act(o)
    # Com um agente e três sensores a rota é o caminho aberto ótimo saindo da
    # origem; o rumo aponta para o primeiro waypoint, que é um dos três sensores.
    assert a.s_norm == pytest.approx(1.0)
    theta = a.theta_norm * 2 * np.pi
    direcao = np.array([np.cos(theta), np.sin(theta)])
    rumos = estado.pos_sensores / np.linalg.norm(
        estado.pos_sensores, axis=1, keepdims=True)
    assert np.max(rumos @ direcao) == pytest.approx(1.0, abs=1e-6), \
        "o rumo tem de coincidir com a direção de ALGUM sensor"


def test_kmeans_tsp_pula_waypoint_coletado_por_outro():
    """Pular, não replanejar — a variante que replaneja está na linha de corte."""
    ctx = _ctx(na=1, ns=2)
    pol = KMeansTSP()
    pol.reset(ctx, np.random.default_rng(0))
    sensores = np.array([[30.0, 0.0], [0.0, 60.0]])
    pol.observar_verdade(EstadoVerdadeiro(sensores, np.array([[0.0, 0.0]]),
                                          np.zeros(2, bool)))
    rota = pol._rotas[0]
    # marca o primeiro da rota como coletado por outro agente
    coletado = np.zeros(2, bool)
    coletado[rota[0]] = True
    pol.observar_verdade(EstadoVerdadeiro(sensores, np.array([[0.0, 0.0]]), coletado))
    o = Observacao.decodificar(_vetor(ctx, [tuple(sensores[rota[1]])]), ctx)
    pol.act(o)
    assert pol._idx[0] == 1, "deveria ter avançado o índice do waypoint"


def test_kmeans_tsp_para_quando_a_rota_acaba():
    ctx = _ctx(na=1, ns=1)
    pol = KMeansTSP()
    pol.reset(ctx, np.random.default_rng(0))
    s = np.array([[10.0, 0.0]])
    pol.observar_verdade(EstadoVerdadeiro(s, np.array([[0.0, 0.0]]),
                                          np.ones(1, bool)))
    o = Observacao.decodificar(_vetor(ctx, []), ctx)
    assert pol.act(o).s_norm == 0.0


def test_kmeans_com_mais_agentes_que_sensores_nao_deixa_uav_sem_rota():
    """Degenerescência da T-02.9 §6: `Na > Ns` gera cluster vazio."""
    ctx = _ctx(na=4, ns=2)
    pol = KMeansTSP()
    pol.reset(ctx, np.random.default_rng(1))
    pol.observar_verdade(EstadoVerdadeiro(
        pos_sensores=np.array([[40.0, 0.0], [0.0, 40.0]]),
        pos_agentes=np.zeros((4, 2)),
        coletado=np.zeros(2, bool),
    ))
    assert len(pol._rotas) == 4, "todo agente precisa de entrada, mesmo vazia"
    assert sum(len(r) for r in pol._rotas.values()) == 2, "cada sensor uma vez"


def test_kmeans_tsp_reporta_o_tempo_planejado():
    """`T_plan` é diagnóstico: a razão contra `t_conc` diz se o gargalo é o
    planejador ou o executor."""
    ctx = _ctx(na=1, ns=2)
    pol = KMeansTSP()
    pol.reset(ctx, np.random.default_rng(0))
    pol.observar_verdade(EstadoVerdadeiro(
        np.array([[80.0, 0.0], [160.0, 0.0]]), np.array([[0.0, 0.0]]),
        np.zeros(2, bool)))
    t = pol.metadados()["t_plan_por_agente"][0]
    # duas pernas de 80 m, cada uma descontando 20 m de alcance, a 15 m/s
    assert t == pytest.approx(2 * (80.0 - 20.0) / 15.0)


# --- adaptador ----------------------------------------------------------


def test_adaptador_devolve_array_e_conta_chamadas():
    """O runner da SPEC-00 recebe um callable e não sabe do contrato."""
    ctx = _ctx()
    g = Guloso()
    g.reset(ctx, np.random.default_rng(0))
    ad = AdaptadorRunner(g, ctx)
    a = ad(_vetor(ctx, [(0.0, 10.0)]))
    assert a.shape == (2,) and 0.0 <= a[0] <= 1.0
    assert ad.chamadas == 1
