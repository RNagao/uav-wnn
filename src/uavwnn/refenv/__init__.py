"""Acesso ao ambiente de referência, sem modificá-lo.

O ambiente do trabalho **é** o `GrADySEnvironment` de Lamenza, carregado
verbatim do submodule `vendor/gradys-data-collection`, fixado na tag
`dissertation`. Nada aqui edita aquele arquivo: este pacote só o localiza,
carrega e acrescenta o que o simulador não oferece (semeadura determinística e
poda do vazamento de logging).

Por que o ambiente não é reimplementado: qualquer diferença de código, por
menor que seja, é uma ameaça à comparação com os números publicados. Medimos
isso — uma "correção" aparentemente inócua na posição inicial do drone
(`(0,0,0)` em vez da posição de spawn) fazia a divergência de observação saltar
de 7,9e-08 para 8,5e-01 em 60 passos, por efeito caótico. Ver
`reports/spec00-parametros.md`.

Versão do simulador: `gradysim==0.6.11`. O `environment.py` de referência
importa `VisualizationController`, que existe até a 0.6.11 e desapareceu na
série 0.7. Rodar o código intacto contra outra versão do simulador seria, por
si só, uma divergência do baseline.
"""

from .loader import (
    REFERENCE_COMMIT,
    ReferenceNotAvailable,
    load_environment,
    load_heuristics,
    reference_root,
)
from .runner import (
    LAMENZA_003,
    EnvConfig,
    EpisodeSummary,
    make_env,
    prune_logging_handlers,
    run_episode,
)

__all__ = [
    "EnvConfig",
    "EpisodeSummary",
    "LAMENZA_003",
    "REFERENCE_COMMIT",
    "ReferenceNotAvailable",
    "load_environment",
    "load_heuristics",
    "make_env",
    "prune_logging_handlers",
    "reference_root",
    "run_episode",
]
