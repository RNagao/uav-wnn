"""Extensão de Fusão de Dados (Etapas 2–3) — pacote separado e opcional.

RNF-07: com este pacote desligado, o sistema executa o problema de Lamenza puro.
É o que torna o princípio da §0 do PRD (o problema não é modificado) verificável
por construção, e não por promessa — nada em `refenv/` ou `agents/` importa daqui.

Escopo pós-D13: o ruído entra **apenas no GPS do drone**. O canal do sensor, o
filtro de informação, a H4, a T3 e a P3 saíram. O que fica: KF com entrada de
controle (Etapa 2) e interseção de covariância entre UAVs (Etapa 3, ainda a
implementar na SPEC-09), mais a validação NEES/NIS.
"""

from .bancada import (
    ResultadoFiltragem,
    Trajetoria,
    rodar_filtro,
    trajetoria_b1,
    trajetoria_b2,
)
from .consistencia import (
    Consistencia,
    autocorrelacao,
    avaliar,
    faixa_chi2,
    faixa_chi2_individual,
    limiar_brancura,
    nees_por_passo,
    nis_por_passo,
)
from .kalman import (
    ErroDeSincronia,
    FiltroGanhoEstacionario,
    FiltroKalmanControle,
    FiltroVelocidadeConstante,
    PassaBaixa,
    ganho_estacionario_escalar,
)
from .ruido import CanalGPS, ConfigRuido, FilaMedidas, fluxo_ruido
from .verdade import (
    ArrayMarcado,
    GravadorRuido,
    PosicaoMedida,
    PosicaoVerdadeira,
    RegistroVerdade,
    ReprodutorRuido,
    TokenMetricas,
    esta_marcado,
)

__all__ = [
    # kalman
    "FiltroKalmanControle",
    "PassaBaixa",
    "FiltroVelocidadeConstante",
    "FiltroGanhoEstacionario",
    "ganho_estacionario_escalar",
    "ErroDeSincronia",
    # ruido
    "ConfigRuido",
    "CanalGPS",
    "FilaMedidas",
    "fluxo_ruido",
    # consistencia
    "nees_por_passo",
    "nis_por_passo",
    "faixa_chi2",
    "faixa_chi2_individual",
    "autocorrelacao",
    "limiar_brancura",
    "avaliar",
    "Consistencia",
    # verdade
    "RegistroVerdade",
    "TokenMetricas",
    "PosicaoVerdadeira",
    "PosicaoMedida",
    "ArrayMarcado",
    "esta_marcado",
    "GravadorRuido",
    "ReprodutorRuido",
    # bancada
    "Trajetoria",
    "ResultadoFiltragem",
    "rodar_filtro",
    "trajetoria_b1",
    "trajetoria_b2",
]
