"""Aprendizado por reforço com redes n-tuple, conforme a dissertação do Katopodis.

O Algoritmo 4.4 e a parametrização gaussiana da política. A binarização fica
fora — é o termômetro quantil da SPEC-01, escolha nossa e não da dissertação.
"""

from .actor_critic import AtorCriticoNTupla, Transicao
from .politica import PoliticaGaussiana

__all__ = ["AtorCriticoNTupla", "PoliticaGaussiana", "Transicao"]
