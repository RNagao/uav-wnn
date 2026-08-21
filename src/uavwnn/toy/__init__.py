"""Brinquedos independentes, sem simulador e sem rede.

`nav.py` é o único ambiente do projeto com ação **angular**, e existe porque
nenhum benchmark clássico tem — sem ele a decisão da normal projetada (SPEC-05
§4.3) seria indefensável.
"""

from .nav import ALCANCE, DT, NavAngular, TAMANHO, V_MAX

__all__ = ["ALCANCE", "DT", "NavAngular", "TAMANHO", "V_MAX"]
