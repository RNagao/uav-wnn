"""Infraestrutura da campanha da SPEC-06 — §4, pré-registrado.

Tudo aqui é **pré-registrado**: escrito e versionado antes da primeira corrida.
Mudança posterior exige commit próprio com justificativa, e as corridas anteriores
continuam no registro.
"""

from .bancos import (
    Banco,
    BancoPar,
    carregar_bancos,
    gerar_bancos,
    horizonte_permissivo,
)
from .recompensas import RECOMPENSAS, eq_5_6, eq_7_2
from .sementes import PapeisDeSemente

__all__ = ["Banco", "BancoPar", "PapeisDeSemente", "RECOMPENSAS", "carregar_bancos",
           "eq_5_6", "eq_7_2", "gerar_bancos", "horizonte_permissivo"]
