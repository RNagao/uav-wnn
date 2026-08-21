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
from .ac_continuo import (
    APROXIMADORES,
    AtorCriticoContinuo,
    Aproximador,
    MlpAprox,
    NTuplaAprox,
    NTuplaFuncionalAprox,
)
from .baselines import (
    BASELINES,
    CaminhadaAleatoria,
    Guloso,
    KMeansTSP,
    held_karp,
    vizinho_2opt,
)
from .mlp_ac import MLP, Adam
from .politica_continua import D_ATOR, NormalProjetada

__all__ = [
    "Acao", "Adam", "AdaptadorRunner", "Aproximador", "AtorCriticoContinuo",
    "APROXIMADORES", "BASELINES", "CaminhadaAleatoria", "D_ATOR", "MLP",
    "MlpAprox", "NTuplaAprox", "NTuplaFuncionalAprox", "NormalProjetada",
    "ContextoEpisodio", "EstadoVerdadeiro", "Guloso", "KMeansTSP",
    "Observacao", "PADDING", "Policy", "PoliticaPrivilegiada", "V_MAX",
    "held_karp", "vizinho_2opt",
]
