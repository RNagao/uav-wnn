"""Aprendizado por reforço com redes n-tuple, conforme a dissertação do Katopodis.

`actor_critic.py` traz o **Algoritmo 4.4** (ação contínua), que é o do nosso
problema. `sarsa.py` traz o **Algoritmo 4.1** (ação discreta), que é o algoritmo
com que o capítulo 5 produziu resultados publicados — está aqui para permitir
replicá-los.

A binarização fica fora dos dois: é o termômetro quantil da SPEC-01, escolha
nossa e não da dissertação.
"""

from .actor_critic import AtorCriticoNTupla, Transicao
from .politica import PoliticaGaussiana
from .sarsa import SarsaNTupla

__all__ = ["AtorCriticoNTupla", "PoliticaGaussiana", "SarsaNTupla", "Transicao"]
