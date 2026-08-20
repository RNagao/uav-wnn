"""Harness de avaliação — SPEC-02.

O módulo existe para que **nenhuma afirmação comparativa deste projeto** dependa
de olhar sobreposição de barras de erro. O que decide é o IC da diferença
**pareada**, e `stats.bootstrap_pareado` é onde ele é calculado.

    sementes.py   árvore de sementes; o pareamento vive aqui (T-02.6)
    metricas.py   t_conc_pen, censura, LB-1, Gini (T-02.5)
    stats.py      bootstrap pareado, MDE, Wilcoxon, McNemar, Holm (T-02.7)

A calibração do bootstrap é verificada por `scripts/spec02_aa.py` — o teste A/A.
Sem ele, nenhum IC deste repositório é confiável, e isso inclui os que já foram
reportados.
"""

from .metricas import (
    MAX_EPISODE_LENGTH,
    MetricasEpisodio,
    Resumo,
    gini,
    lb1,
    resumir_episodios,
    sucesso_comum,
)
from .sementes import ArvoreSementes, disjuntos
from .stats import (
    B_CANONICO,
    Diferenca,
    Marginal,
    bootstrap_media,
    bootstrap_pareado,
    holm,
    mcnemar,
    mde_95,
    wilcoxon_pareado,
)

__all__ = [
    "ArvoreSementes", "B_CANONICO", "Diferenca", "MAX_EPISODE_LENGTH",
    "Marginal", "MetricasEpisodio", "Resumo", "bootstrap_media",
    "bootstrap_pareado", "disjuntos", "gini", "holm", "lb1", "mcnemar",
    "mde_95", "resumir_episodios", "sucesso_comum", "wilcoxon_pareado",
]
