"""Harness de custo — T1 tamanho, T2 latência, T3 operações (SPEC-03).

Figuras de mérito de primeira classe do PRD §5.5, não notas de rodapé: são o
conteúdo de H1 e a única contribuição que sobrevive mesmo se o desempenho for
apenas comparável.
"""

from .memoria import (
    CustoMemoria,
    Variante,
    custo_mlp,
    custo_ntuple,
    ocupacao_por_ram,
)
from .operacoes import CustoOperacoes, operacoes_mlp, operacoes_ntuple

__all__ = [
    "CustoMemoria",
    "CustoOperacoes",
    "Variante",
    "custo_mlp",
    "custo_ntuple",
    "ocupacao_por_ram",
    "operacoes_mlp",
    "operacoes_ntuple",
]
