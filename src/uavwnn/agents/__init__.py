"""Políticas — a interface congelada da SPEC-02 e os três baselines.

`base.py` define o contrato que o harness usa para não distinguir baseline de
política aprendida: mesma censura, mesmo bootstrap, mesma tabela. A SPEC-05
implementa a mesma interface.
"""

from .base import (
    PADDING,
    V_MAX,
    Acao,
    AdaptadorRunner,
    ContextoEpisodio,
    EstadoVerdadeiro,
    Observacao,
    Policy,
    PoliticaPrivilegiada,
)
from .baselines import (
    BASELINES,
    CaminhadaAleatoria,
    Guloso,
    KMeansTSP,
    held_karp,
    vizinho_2opt,
)

__all__ = [
    "Acao", "AdaptadorRunner", "BASELINES", "CaminhadaAleatoria",
    "ContextoEpisodio", "EstadoVerdadeiro", "Guloso", "KMeansTSP",
    "Observacao", "PADDING", "Policy", "PoliticaPrivilegiada", "V_MAX",
    "held_karp", "vizinho_2opt",
]
