"""RNF-07 — com `fusao/` desligado, o sistema executa o problema de Lamenza puro.

> `fusion/` é um pacote separado e opcional. Com ele desligado, o sistema executa o
> problema de Lamenza puro. **Isso é o que torna o princípio da §0 verificável por
> construção, e não por promessa.** — PRD RNF-07

É o critério de saída 10 da SPEC-06.

POR QUE O TESTE EXISTE ANTES DO PACOTE

`src/uavwnn/fusao/` só nasce na SPEC-07. Hoje a RNF-07 é satisfeita trivialmente —
não há o que desligar —, e um teste que apenas constatasse isso não valeria nada.

O que estes testes fazem é **guardar o futuro**: eles continuam passando quando o
pacote aparecer e **falham** se a fusão vazar para o caminho central. Escrevê-los
depois seria escrevê-los sobre um acoplamento já feito, quando desfazê-lo custa caro.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

RAIZ = pathlib.Path(__file__).resolve().parents[1]
NUCLEO = RAIZ / "src" / "uavwnn"
NOME_FUSAO = "fusao"


def _importa(caminho: pathlib.Path) -> set[str]:
    """Módulos de topo importados por um arquivo, por AST.

    AST e não `grep`: `# import fusao` num comentário não é import, e
    `from .fusao import x` dentro de uma docstring também não. Um teste de
    arquitetura que dá falso positivo é abandonado, e aí não guarda nada.
    """
    arvore = ast.parse(caminho.read_text())
    fora: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            fora.update(a.name.split(".")[0] for a in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            fora.add(no.module.split(".")[0])
        elif isinstance(no, ast.ImportFrom) and no.level and not no.module:
            fora.update(a.name for a in no.names)
    return fora


def _modulos_do_nucleo() -> list[pathlib.Path]:
    """Todo `.py` de `src/uavwnn/` **fora** de `fusao/`."""
    return [p for p in NUCLEO.rglob("*.py")
            if NOME_FUSAO not in p.relative_to(NUCLEO).parts
            and "__pycache__" not in p.parts]


def test_o_nucleo_nao_importa_fusao():
    """A verificação estrutural da RNF-07.

    Se qualquer módulo do núcleo importar `fusao`, desligar o pacote deixa de ser
    possível — e a alegação de que o problema de Lamenza roda puro passa a depender
    de leitura de código em vez de teste.
    """
    culpados = [
        str(p.relative_to(RAIZ)) for p in _modulos_do_nucleo()
        if NOME_FUSAO in _importa(p)
    ]
    assert not culpados, (
        f"módulos do núcleo importando `{NOME_FUSAO}`: {culpados}. A RNF-07 exige "
        f"que o pacote de fusão seja opcional — com ele desligado, o sistema tem de "
        f"executar o problema de Lamenza puro.")


def test_o_pacote_de_fusao_e_opcional_de_verdade():
    """Importar `uavwnn` e o caminho de avaliação **sem** tocar em `fusao`.

    Cobre o caso que o teste de AST não cobre: um import indireto, via um terceiro
    módulo que o AST veria como inocente.
    """
    import sys

    antes = {m for m in sys.modules if NOME_FUSAO in m}
    import uavwnn  # noqa: F401
    import uavwnn.agents  # noqa: F401
    import uavwnn.campanha  # noqa: F401
    import uavwnn.eval  # noqa: F401
    import uavwnn.refenv  # noqa: F401
    import uavwnn.wnn  # noqa: F401
    novos = {m for m in sys.modules if NOME_FUSAO in m} - antes
    assert not novos, f"importar o núcleo carregou {novos}"


@pytest.mark.parametrize("na,ns", [(1, 12), (2, 12)])
def test_o_problema_de_lamenza_roda_puro(na, ns):
    """A parte positiva: sem fusão, o problema **executa** e dá métrica válida.

    Não basta não importar — o sistema tem de funcionar. Roda a heurística gulosa,
    que é o baseline do problema, e verifica que as três métricas da §4.5 saem
    coerentes: `SR` em [0,1], `CT` positivo e limitado pelo horizonte, e `CT_succ`
    definido quando há sucesso.
    """
    from uavwnn.agents import BASELINES
    from uavwnn.agents.base import ContextoEpisodio, Observacao
    from uavwnn.refenv import LAMENZA_003, run_episode

    cfg = LAMENZA_003(num_drones=na, num_sensors=ns)
    ctx = ContextoEpisodio(
        num_drones=cfg.num_drones, num_sensores=cfg.num_sensors,
        scenario_size=cfg.scenario_size,
        communication_range=cfg.communication_range,
        dt=cfg.algorithm_iteration_interval,
        max_episode_length=cfg.max_episode_length)
    pol = BASELINES["guloso"]()

    resumos = []
    for s in range(4):
        pol.reset(ctx, np.random.default_rng(s))
        resumos.append(run_episode(
            cfg, lambda o: pol.act(Observacao.decodificar(o, ctx)).como_array(),
            seed=s))

    ct = np.array([r.completion_time for r in resumos], float)
    ok = np.array([r.all_collected for r in resumos], bool)
    assert 0.0 <= ok.mean() <= 1.0
    assert np.all(ct > 0) and np.all(ct <= cfg.max_episode_length)
    if ok.any():
        assert np.isfinite(ct[ok].mean())


def test_a_verdade_nao_vaza_para_a_politica():
    """RF-04, o invariante irmão — e ele é pré-requisito da RNF-07.

    Uma política sem `usa_estado_verdadeiro` **não** recebe `EstadoVerdadeiro`. Se
    recebesse, "o problema de Lamenza puro" deixaria de ser puro por outra via: o
    agente veria posições absolutas que a eq. 5-1 não dá.
    """
    from uavwnn.agents import BASELINES

    privilegiadas = {n: c for n, c in BASELINES.items()
                     if getattr(c(), "usa_estado_verdadeiro", False)}
    nao = {n: c for n, c in BASELINES.items() if n not in privilegiadas}
    # O k-means+TSP é privilegiado POR DECLARAÇÃO — é o ponto dele.
    assert "kmeans_tsp" in privilegiadas
    # Guloso e aleatório não podem ser.
    for nome in ("guloso", "random_uniforme"):
        assert nome in nao, f"{nome} declarou privilégio que não deveria ter"
