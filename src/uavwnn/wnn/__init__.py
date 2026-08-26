"""Redes n-tuple sem peso.

`NTupleRegressor` é o regressor denso da SPEC-01 — rápido, vetorizado, mas
preso a tuplas pequenas porque aloca `2^n` posições por RAM.

`redes.py` traz as **duas** redes da dissertação do Katopodis com armazenamento
esparso, que é o que permite as tuplas de 16 a 64 bits do apêndice A dela.
"""

from .ntuple import NTupleRegressor
from .redes import (
    MAX_TUPLA,
    LUTPlana,
    RedeGradienteFuncional,
    RedeRegressaoEsquecimento,
    mapeamento_particionado,
)

__all__ = [
    "MAX_TUPLA",
    "LUTPlana",
    "NTupleRegressor",
    "LUTPlana",
    "RedeGradienteFuncional",
    "RedeRegressaoEsquecimento",
    "mapeamento_particionado",
]
