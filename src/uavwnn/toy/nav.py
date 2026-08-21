"""Brinquedo de navegação angular — SPEC-05 T-05.4(b).

**Nenhum benchmark clássico tem ação angular**, e é justamente a circularidade que
a §4.3 decide como tratar. Sem um ambiente onde `θ` importe, a decisão da normal
projetada é indefensável — e é o único ponto do projeto em que ela é falsificável
de forma barata.

    1 UAV, 1 sensor, área 200 × 200, `dt = 0,5 s`
    ação `⟨θ, s⟩` normalizada, `v_max = 15 m/s`
    recompensa `+10` na visita, `−1` por passo

Sem simulador e sem rede: episódios de **milissegundos**. Não usa o gerador
analítico da SPEC-01, que foi descartado — é brinquedo independente.

O QUE ELE EXISTE PARA DECIDIR

    C0  gaussiana direta sobre `(θ_norm, s_norm)`, com recorte  ← o ingênuo
    C2  normal projetada em R³                                  ← a proposta

O modo de falha do C0 é específico e é o que o brinquedo expõe: uma política que
quer rumo perto de `θ_norm = 0` (ou 1) tem **metade da massa de probabilidade
recortada** na fronteira. Ela não consegue expressar "aponte para leste" sem
perder metade das amostras, e o gradiente na fronteira fica viesado para dentro.

Com alvos espalhados uniformemente no círculo, uma fração dos cenários cai
exatamente nessa região — e é lá que a diferença aparece.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

TAMANHO = 200.0
DT = 0.5
V_MAX = 15.0
ALCANCE = 20.0
"""`communication_range` do cenário de referência — a coleta é por proximidade."""

RECOMPENSA_VISITA = 10.0
RECOMPENSA_PASSO = -1.0

# RECOMPENSA DENSA NO PROGRESSO — e por que ela é obrigatória aqui.
#
# A primeira versão do brinquedo dava só `-1` por passo e `+10` na visita. Medido:
# 0-8 % de sucesso, e `δ ≈ -19` constante até o fim do treino. Com quase nenhum
# episódio bem-sucedido o retorno vale ≈ -60 **qualquer que seja o rumo**, então o
# gradiente não carrega informação direcional alguma. As duas políticas paravam em
# ~150° de erro — pior que rumo aleatório.
#
# Isso mediria EXPLORAÇÃO sob recompensa esparsa, não parametrização de ação. São
# perguntas diferentes, e T-05.4(b) é sobre a segunda. O termo de progresso, em
# unidades de deslocamento máximo por passo, torna o sinal direcional explícito e
# devolve o teste ao seu objeto:
#
#     r_prog = (d_antes − d_depois) / (v_max · dt)     ∈ [−1, +1]
#
# `+1` é apontar exatamente para o alvo na velocidade máxima. Não é atalho para a
# política: ela ainda tem de inferir o rumo a partir da observação, e ainda tem de
# atravessar a fronteira `θ ≈ 0` — que é o ponto todo. O brinquedo é independente
# do problema de Lamenza, então moldar a recompensa aqui não toca no baseline.
PESO_PROGRESSO = 1.0


@dataclass
class NavAngular:
    """Um episódio: sair da origem e alcançar um sensor sorteado.

    A observação é o vetor relativo **normalizado como o ambiente de referência**
    faz — `(p_uav − p_sensor + S)/(2S)` —, para o brinquedo exercitar o mesmo
    caminho de decodificação que o problema real.
    """

    max_passos: int = 40

    def reset(self, rng: np.random.Generator) -> NDArray[np.float64]:
        # Alvo em anel: longe o bastante para exigir vários passos, e com ângulo
        # UNIFORME no círculo — é o que garante que a fronteira `θ ≈ 0` seja
        # visitada com frequência.
        ang = rng.uniform(0.0, 2 * np.pi)
        raio = rng.uniform(60.0, 90.0)
        self.sensor = np.array([raio * np.cos(ang), raio * np.sin(ang)])
        self.pos = np.zeros(2)
        self.passos = 0
        self.coletado = False
        return self.observar()

    def observar(self) -> NDArray[np.float64]:
        """Mesma normalização do `observe_simulation_relative_positions`."""
        rel = (self.pos - self.sensor + TAMANHO) / (2 * TAMANHO)
        return rel

    @property
    def rumo_otimo(self) -> float:
        """O que uma política perfeita escolheria agora. Alvo do erro angular."""
        d = self.sensor - self.pos
        return float(np.arctan2(d[1], d[0]))

    def passo(self, theta_norm: float, s_norm: float):
        """Devolve `(obs, recompensa, terminou)`."""
        d_ant = float(np.linalg.norm(self.sensor - self.pos))
        theta = float(theta_norm) * 2 * np.pi
        v = float(np.clip(s_norm, 0.0, 1.0)) * V_MAX
        self.pos = self.pos + np.array([np.cos(theta), np.sin(theta)]) * v * DT
        self.passos += 1

        d_dep = float(np.linalg.norm(self.sensor - self.pos))
        r = RECOMPENSA_PASSO + PESO_PROGRESSO * (d_ant - d_dep) / (V_MAX * DT)
        if d_dep <= ALCANCE:
            self.coletado = True
            r += RECOMPENSA_VISITA
        terminou = self.coletado or self.passos >= self.max_passos
        return self.observar(), r, terminou
