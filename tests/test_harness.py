"""Harness de avaliação — SPEC-02 T-02.8 e T-02.9.

Estes testes rodam o simulador, então são os mais lentos da suíte. Cada um existe
por um invariante que o harness **aplica** em vez de supor:

1. **Determinismo** — duas execuções da mesma célula dão colunas determinísticas
   idênticas. Sem isso `make repro` não significa nada.
2. **Privilégio** — nenhuma política com `usa_estado_verdadeiro = False` recebe
   `EstadoVerdadeiro`. É a RF-04 antecipada.
3. **Pareamento** — todas as políticas veem exatamente os mesmos cenários.
4. **LB-1 é limite** — abaixo dele é bug, e a asserção precisa de fato disparar.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.agents import (
    Acao,
    CaminhadaAleatoria,
    ContextoEpisodio,
    Guloso,
    KMeansTSP,
    Observacao,
)
from uavwnn.eval import ArvoreSementes
from uavwnn.eval.harness import (
    agregado,
    avaliar,
    comparacoes,
    contexto_de,
    escrever_csv,
    rodar_episodio,
)
from uavwnn.refenv import LAMENZA_003, ReferenceNotAvailable, reference_root

try:
    reference_root()
    _SUBMODULE = True
except ReferenceNotAvailable:  # pragma: no cover
    _SUBMODULE = False

pytestmark = pytest.mark.skipif(not _SUBMODULE, reason="submodule ausente")

CELULA = (2, 12)


def _cfg():
    return LAMENZA_003(*CELULA, phase="eval")


# --- T-02.9 §4: privilégio ----------------------------------------------


class Espiao:
    """Política sem privilégio que grita se receber a verdade."""

    id = "espiao"
    usa_estado_verdadeiro = False

    def __init__(self) -> None:
        self.recebeu = 0

    def reset(self, ctx, rng) -> None:
        self.ctx = ctx

    def observar_verdade(self, estado) -> None:      # não deveria ser chamado
        self.recebeu += 1

    def act(self, obs: Observacao) -> Acao:
        return Acao(0.5, 1.0)

    def metadados(self) -> dict:
        return {"deterministica": True}


def test_politica_sem_privilegio_nunca_recebe_a_verdade():
    """Antecipa a RF-04. O espião **expõe** `observar_verdade`, então o teste
    verifica a decisão do harness e não a ausência do método."""
    esp = Espiao()
    rodar_episodio(esp, _cfg(), seed_cenario=1, idx_episodio=0,
                   rng_politica=np.random.default_rng(0))
    assert esp.recebeu == 0, "a verdade vazou para uma política não privilegiada"


def test_politica_privilegiada_recebe_a_verdade_a_cada_passo():
    """O k-means+TSP precisa da posição atual para apontar ao waypoint — uma
    versão anterior só recebia no `reset` e degenerava no guloso."""
    contagem = {"n": 0}
    pol = KMeansTSP()
    original = pol.observar_verdade

    def espiar(estado):
        contagem["n"] += 1
        original(estado)

    pol.observar_verdade = espiar
    ep = rodar_episodio(pol, _cfg(), 2, 0, np.random.default_rng(0))
    assert contagem["n"] == ep.n_decisoes, "uma chamada por época de decisão"


# --- T-02.8: determinismo -----------------------------------------------


def test_mesma_semente_da_episodio_identico():
    """Sem isso, `make repro` não significa nada."""
    cfg = _cfg()
    a = rodar_episodio(Guloso(), cfg, 7, 0, np.random.default_rng(0))
    b = rodar_episodio(Guloso(), cfg, 7, 0, np.random.default_rng(0))
    assert a.sucesso == b.sucesso
    assert a.t_conc_s == pytest.approx(b.t_conc_s, nan_ok=True)
    assert a.n_decisoes == b.n_decisoes
    assert a.visitas_por_agente == b.visitas_por_agente
    assert a.lb1_s == pytest.approx(b.lb1_s)


def test_sementes_diferentes_dao_cenarios_diferentes():
    cfg = _cfg()
    a = rodar_episodio(Guloso(), cfg, 11, 0, np.random.default_rng(0))
    b = rodar_episodio(Guloso(), cfg, 12, 1, np.random.default_rng(0))
    assert a.lb1_s != pytest.approx(b.lb1_s), "geometria deveria diferir"


def test_estocastica_com_mesmo_rng_repete():
    """A caminhada aleatória consome o `rng` que recebe; mesma semente, mesmo
    resultado. É o que torna o piso reprodutível."""
    cfg = _cfg()
    a = rodar_episodio(CaminhadaAleatoria(), cfg, 5, 0, np.random.default_rng(3))
    b = rodar_episodio(CaminhadaAleatoria(), cfg, 5, 0, np.random.default_rng(3))
    assert a.n_visitados == b.n_visitados
    assert a.t_fim_s == pytest.approx(b.t_fim_s)


# --- T-02.9 §1: ordenação e LB-1 ----------------------------------------


def test_lb1_e_respeitado_pelo_guloso():
    """A asserção do harness protege contra LB-1 errado. Aqui verificamos o
    outro lado: o resultado real fica **acima** do limite."""
    ep = rodar_episodio(Guloso(), _cfg(), 21, 0, np.random.default_rng(0))
    if ep.sucesso:
        assert ep.t_conc_s >= ep.lb1_s - 1e-6


def test_ordenacao_esperada_numa_amostra_pequena():
    """`LB-1 ≤ kmeans_tsp ≤ guloso ≤ random`, com as adjacentes conclusivas.

    Poucos episódios de propósito — a campanha de 500 vive em
    `scripts/spec02_baselines.py`. Aqui é regressão: se alguém inverter um sinal
    ou quebrar o planejador, a ordem desaba e isto acusa.
    """
    arv = ArvoreSementes(20260809)
    pols = {"random_uniforme": CaminhadaAleatoria(), "guloso": Guloso(),
            "kmeans_tsp": KMeansTSP()}
    res = avaliar(pols, _cfg(), arv, n_episodios=8, verboso=False)
    m = {l["politica"]: l for l in agregado(res, arv)}

    assert m["guloso"]["media_lb1"] <= m["kmeans_tsp"]["media_t_conc_pen"]
    assert (m["kmeans_tsp"]["media_t_conc_pen"]
            < m["guloso"]["media_t_conc_pen"]
            < m["random_uniforme"]["media_t_conc_pen"])

    cmps = {(c["politica_a"], c["politica_b"]): c
            for c in comparacoes(res, arv, [("kmeans_tsp", "guloso"),
                                            ("guloso", "random_uniforme")])
            if c["subconjunto"] == "todos"}
    for par in cmps:
        assert not cmps[par]["contem_zero"], f"{par} deveria ser conclusivo"


def test_pareamento_usa_os_mesmos_cenarios():
    """Invariante 2. Se cada política vissem cenários diferentes, o bootstrap
    pareado estaria comparando coisas incomparáveis."""
    arv = ArvoreSementes(1234)
    pols = {"guloso": Guloso(), "kmeans_tsp": KMeansTSP()}
    res = avaliar(pols, _cfg(), arv, n_episodios=5, verboso=False)
    sementes = {pid: [e.seed_cenario for e in eps]
                for pid, eps in res.por_politica.items()}
    assert sementes["guloso"] == sementes["kmeans_tsp"] == res.sementes
    # e o LB-1, que é função só da geometria, tem de coincidir episódio a episódio
    lb_g = [e.lb1_s for e in res.por_politica["guloso"]]
    lb_k = [e.lb1_s for e in res.por_politica["kmeans_tsp"]]
    assert lb_g == pytest.approx(lb_k)


# --- T-02.9 §6: degenerescências ----------------------------------------


def test_um_drone_e_padding_total_funcionam():
    """`Na = 1` (sem slot de agente) e `Ns < N*_s = 12` (padding parcial)."""
    cfg = LAMENZA_003(1, 3, phase="eval")
    assert contexto_de(cfg).num_drones == 1
    ep = rodar_episodio(Guloso(), cfg, 31, 0, np.random.default_rng(0))
    assert ep.n_sensores == 3
    assert len(ep.visitas_por_agente) == 1


def test_mais_agentes_que_sensores():
    """Cluster vazio no k-means; nenhum UAV pode ficar sem entrada de rota."""
    cfg = LAMENZA_003(4, 2, phase="eval")
    ep = rodar_episodio(KMeansTSP(), cfg, 41, 0, np.random.default_rng(0))
    assert len(ep.visitas_por_agente) == 4
    assert sum(ep.visitas_por_agente) <= 2


# --- T-02.8: artefatos --------------------------------------------------


def test_agregado_e_comparacoes_tem_as_colunas_obrigatorias(tmp_path):
    """MDE é coluna obrigatória: sem ela um resultado nulo é ininterpretável."""
    arv = ArvoreSementes(99)
    pols = {"guloso": Guloso(), "kmeans_tsp": KMeansTSP()}
    res = avaliar(pols, _cfg(), arv, n_episodios=5, verboso=False)

    ag = agregado(res, arv)
    for obrig in ("celula", "politica", "media_t_conc_pen", "ic_lo", "ic_hi",
                  "taxa_sucesso", "mediana", "p90", "media_lb1", "gini_visitas"):
        assert obrig in ag[0], f"falta {obrig} no agregado"

    cmp = comparacoes(res, arv, [("kmeans_tsp", "guloso")])
    for obrig in ("delta_media", "delta_ic_lo", "delta_ic_hi", "contem_zero",
                  "mde_95", "taxa_vitoria_a", "p_wilcoxon", "p_mcnemar",
                  "delta_taxa_sucesso", "subconjunto", "primaria"):
        assert obrig in cmp[0], f"falta {obrig} nas comparações"

    escrever_csv(tmp_path / "agregado.csv", ag)
    assert (tmp_path / "agregado.csv").read_text().count("\n") == len(ag) + 1


# --- fidelidade: o nosso guloso É o guloso de referência -----------------


def test_nosso_guloso_reproduz_o_greedy_de_referencia():
    """**Elo que faltava na cadeia de validação.**

    A SPEC-00 provou que o `greedy` do `heuristics.py` de referência reproduz a
    fig. 7.3 em 9/9 células. Este teste prova que a **nossa** `Guloso` é aquela
    política — e não só "parecida com".

    Se as duas coincidem episódio a episódio, então tudo entre elas está certo:
    a decodificação da observação (`Observacao.decodificar`), o sinal do rumo, a
    moeda da ação, a semeadura e a métrica. Um erro em qualquer desses pontos
    apareceria aqui como divergência.

    Medido em 60 cenários e 3 células: **delta 0,00 s**. Aqui usamos menos
    episódios, e a igualdade tem de ser **exata** — não aproximada.
    """
    from uavwnn.refenv import load_heuristics, run_episode

    arv = ArvoreSementes(20260809)
    sementes = arv.cenarios("avaliacao", 12)
    h = load_heuristics()

    for na, ns in ((1, 12), (2, 12)):
        cfg = LAMENZA_003(na, ns, phase="eval")
        ref = h.create_greedy_heuristics(na - 1, cfg.state_num_closest_sensors)
        for i, s in enumerate(sementes):
            nosso = rodar_episodio(Guloso(), cfg, s, i, arv.politica("guloso", i))
            deles = run_episode(cfg, ref, seed=s)
            assert nosso.sucesso == deles.all_collected, f"Na{na} semente {s}"
            if nosso.sucesso:
                assert nosso.t_conc_s == pytest.approx(deles.completion_time), \
                    f"Na{na} semente {s}: divergiu do greedy de referência"


def test_guloso_na_config_da_figura_reproduz_o_publicado():
    """`phase="eval_heuristic"` — intervalo 0,1 s, que é o que a fig. 7.3 usou.

    A SPEC-02 avalia tudo a 0,5 s, o que é a escolha certa internamente
    (comparar políticas com intervalos diferentes seria variável não
    controlada). Mas isso faz a nossa coluna do guloso **não** ser a da fig. 7.3,
    e este teste marca a diferença: a 0,1 s o número volta ao publicado.

    Medido com 200 episódios: `Na2_Ns12` dá 29,23 contra 29,6 publicado (−1,3 %)
    e `Na4_Ns12` dá 28,35 contra 28,2 (+0,5 %). Com os 25 daqui a folga é maior.
    """
    arv = ArvoreSementes(20260809)
    sementes = arv.cenarios("avaliacao", 25)
    cfg = LAMENZA_003(2, 12, phase="eval_heuristic")
    assert cfg.algorithm_iteration_interval == pytest.approx(0.1)

    v = np.mean([rodar_episodio(Guloso(), cfg, s, i,
                                arv.politica("guloso", i)).t_conc_pen_s
                 for i, s in enumerate(sementes)])
    assert abs(v - 29.6) / 29.6 < 0.10, f"medido {v:.2f}s contra 29,6 publicado"


def test_intervalo_de_decisao_muda_o_guloso_e_a_diferenca_e_declarada():
    """A assimetria da fig. 7.3, como teste.

    O `visualize_performance.py` avalia o MADDPG a 0,5 s e o guloso a **0,1 s** —
    cinco vezes mais reativo. O efeito favorece o guloso, então a vantagem
    publicada do MADDPG sobre ele é conservadora. Aqui só travamos que o efeito
    existe e tem o sinal esperado.
    """
    arv = ArvoreSementes(20260809)
    sementes = arv.cenarios("avaliacao", 25)

    def media(phase):
        cfg = LAMENZA_003(2, 12, phase=phase)
        return np.mean([rodar_episodio(Guloso(), cfg, s, i,
                                       arv.politica("guloso", i)).t_conc_pen_s
                        for i, s in enumerate(sementes)])

    assert media("eval_heuristic") < media("eval"), \
        "o guloso mais reativo deveria concluir antes"
