"""Geração de cenário e semeadura reprodutível (SPEC-00, T-00.2).

Geração pura: nada de simulador aqui. Este módulo é compartilhado com o gerador
analítico da SPEC-01, o que é o que torna o invariante V-1 uma medida de
diferença de *dinâmica* e não de diferença de *código*.

Proveniência dos valores: `reports/spec00-parametros.md`. Os defaults abaixo
vêm do código de referência (`scenario/003-collaboration/environment.py` e
`benchmark.py` do repositório `Project-GrADyS/gradys-data-collection@dissertation`),
não do texto da dissertação, sempre que os dois divergem — ver o relatório.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScenarioConfig:
    """Parâmetros do cenário de Lamenza.

    Os campos seguem os nomes do código de referência para que a comparação
    linha a linha seja possível.
    """

    n_agents: int
    """Na — número de UAVs."""

    n_sensors: int
    """Ns — número de sensores."""

    scenario_size: float = 100.0
    """Semi-lado da área: o quadrado é [-scenario_size, scenario_size]^2."""

    communication_range: float = 20.0
    """P1, RESOLVIDA: `communication_range: float = 20` em environment.py."""

    algorithm_iteration_interval: float = 0.5
    """Intervalo entre decisões, em segundos.

    DIVERGÊNCIA: a tab. 7.1 da dissertação diz 2 s; o `benchmark.py` que gerou os
    resultados passa `--algorithm-iteration-interval=0.5`. Adotamos 0.5 porque
    2 s é fisicamente inconsistente com `communication_range = 20`: a 15 m/s o
    UAV percorreria 30 m entre heartbeats e atravessaria a cobertura de um sensor
    sem coletá-lo. Ver `reports/spec00-parametros.md`.
    """

    max_episode_length: float = 500.0
    max_seconds_stalled: float = 30.0
    max_speed: float = 15.0

    spawn_radius: float = 2.0
    """P6, RESOLVIDA: `random.uniform(-2, 2)` em environment.py (reset)."""

    n_star_s: int = 12
    """N*_s — sensores não visitados mais próximos na observação (tab. 7.1)."""

    n_star_a: int | None = None
    """N*_a. `None` significa `n_agents - 1` (tab. 7.1 e benchmark.py)."""

    end_when_all_collected: bool = False
    """P/RF-02, RESOLVIDA: `--no-end-when-all-collected` no benchmark.py.

    O episódio NÃO termina ao coletar tudo, mas a poda por parada continua
    valendo — logo ele acaba `max_seconds_stalled` após a última visita.
    É a leitura (B) da SPEC-00 §9.2, não a (A) que a spec havia proposto.
    """

    update_rate: float | None = None
    """Passo do integrador de mobilidade. `None` = `algorithm_iteration_interval / 16`.

    P2, RESOLVIDA POR MEDIÇÃO — e o resultado contraria o código de referência.

    O `environment.py` de referência usa `update_rate = interval / 2` (= 0,25 s).
    Medimos 200 episódios PAREADOS na célula 4x24 com política determinística,
    contra um integrador fino, usando apenas valores exatos em binário
    (`interval / 2^k`) para que a comparação não seja contaminada por deriva
    de ponto flutuante:

        update_rate   /2^k   viés (s)   IC95 da diferença   KS p    s/ep
          0,250000      /2     +2,97    [+1,77, +4,17]     0,000   0,088  <- referência
          0,125000      /4     +1,93    [+1,15, +2,70]     0,088   0,107
          0,062500      /8     +1,02    [+0,32, +1,73]     0,628   0,146
          0,031250     /16     +0,46    [-0,24, +1,16]     0,965   0,227  <- adotado
          0,015625     /32     +0,27    [-0,08, +0,62]     1,000   0,383

    A escolha do próprio trabalho de referência introduz viés SISTEMÁTICO de
    +2,97 s (+7,9 %) no tempo de conclusão. A causa é mecânica: o heartbeat é
    emitido em `0,99 * interval` após a ação, mas com integrador grosseiro a
    posição do nó nesse instante está um tique atrasada — o broadcast sai de um
    ponto que o UAV já deixou, e coletas são perdidas.

    Adotamos `interval / 16` = 0,031250 s: é o valor mais barato cujo IC95 da
    diferença pareada CONTÉM ZERO. Custa 2,6x mais que a referência, o que cabe
    no orçamento medido (T-00.10).

    Isto NÃO altera o MDP — observação, ação, recompensa e terminação são
    idênticos. É resolução numérica. Mas altera valores absolutos em ~8 % em
    relação aos publicados, e por isso a comparação rigorosa da SPEC-06 é por
    RAZÃO contra o guloso reimplementado.

    Use `update_rate=0.25` para reproduzir o comportamento exato da referência.
    """

    id_on_state: bool = True
    """Índice do agente na observação, codificado como `k / Na`."""

    edge_behavior: str = "reference"
    """Como o waypoint é limitado na borda.

    - "reference": reproduz `DroneProtocol.act` verbatim, inclusive a quirk de
      que o alcance é medido contra a borda MAIS PRÓXIMA em cada eixo
      (`scenario_size - |x|`), independentemente do sentido do movimento. Perto
      de uma borda o UAV quase para, mesmo apontando para dentro.
    - "corrected": mede a distância até a borda no sentido do movimento.
      NÃO é fiel ao original; existe só para quantificar o efeito da quirk.
    """

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("n_agents deve ser >= 1")
        if self.n_sensors < 1:
            raise ValueError("n_sensors deve ser >= 1")
        if self.edge_behavior not in ("reference", "corrected"):
            raise ValueError(f"edge_behavior inválido: {self.edge_behavior!r}")

        # SPEC-00 T-00.6 / teste 23. O handler de mobilidade move
        # `speed * update_rate` por tique; se `dt` não for múltiplo inteiro de
        # `update_rate`, o UAV chega ao waypoint no meio de um tique e a
        # distância efetiva por intervalo fica silenciosamente abaixo da
        # comandada.
        ur = self.effective_update_rate
        ratio = self.algorithm_iteration_interval / ur
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                "algorithm_iteration_interval deve ser múltiplo inteiro de "
                f"update_rate (razão={ratio})"
            )

        # O handler de mobilidade reagenda por SOMA REPETIDA
        # (`current_time + update_rate`), enquanto a barreira de época soma
        # `interval`. Se `update_rate` não for exato em binário, os dois
        # relógios DERIVAM um em relação ao outro e o número de tiques por
        # época passa a oscilar. Medido com update_rate=0.1: a sequência de
        # deslocamentos por época foi 4,5 / 9,0 / 6,0 / 7,5 m em vez de 7,5
        # constante. A média fica certa, a dinâmica não.
        acc = 0.0
        n_ticks = int(round(self.max_episode_length / ur))
        for _ in range(min(n_ticks, 20000)):
            acc += ur
        exact = ur * min(n_ticks, 20000)
        if abs(acc - exact) > 1e-12:
            raise ValueError(
                f"update_rate={ur!r} não é exato em binário: a soma repetida "
                f"deriva {abs(acc - exact):.3e} s ao longo do episódio, e o "
                "número de tiques de mobilidade por época passa a oscilar. "
                "Use uma fração diádica, por exemplo interval/2^k."
            )

        # SPEC-00 T-00.3 / teste 24. Entre dois heartbeats o UAV percorre até
        # `max_speed * heartbeat_interval`. Se isso exceder o alcance, ele pode
        # ATRAVESSAR a cobertura de um sensor sem detectá-lo, e o ambiente fica
        # mais difícil que o original sem aparecer na tabela de desvios.
        travel = self.max_speed * self.heartbeat_interval
        if travel > self.communication_range:
            raise ValueError(
                f"heartbeat esparso demais: o UAV percorre {travel:.1f} m entre "
                f"heartbeats, contra alcance de {self.communication_range:.1f} m. "
                "Sensores seriam atravessados sem coleta."
            )

    @property
    def effective_update_rate(self) -> float:
        if self.update_rate is not None:
            return self.update_rate
        # /16 e não /2: ver a docstring de `update_rate`. Fração diádica, logo
        # exata em binário e sem deriva na soma repetida.
        return self.algorithm_iteration_interval / 16.0

    @property
    def effective_n_star_a(self) -> int:
        if self.n_star_a is not None:
            return self.n_star_a
        return self.n_agents - 1

    @property
    def heartbeat_interval(self) -> float:
        """O broadcast é agendado em `act()` para `now + interval * 0.99`."""
        return self.algorithm_iteration_interval * 0.99

    @property
    def observation_size(self) -> int:
        """2*N*_a + 2*N*_s + (1 se id_on_state)."""
        return 2 * self.effective_n_star_a + 2 * self.n_star_s + int(self.id_on_state)

    @property
    def max_decision_steps(self) -> int:
        return int(math.ceil(self.max_episode_length / self.algorithm_iteration_interval))


@dataclass(frozen=True)
class Scenario:
    config: ScenarioConfig
    agent_positions: NDArray[np.float64]
    """(Na, 2) — posições iniciais dos UAVs."""
    sensor_positions: NDArray[np.float64]
    """(Ns, 2) — posições dos sensores (estáticas)."""
    seed: int


def episode_seed(root_seed: int, episode_index: int) -> tuple[np.random.Generator, int]:
    """Deriva o RNG do episódio `episode_index` a partir de uma semente raiz.

    Usa `SeedSequence.spawn`, de modo que o episódio `i` é reproduzível
    ISOLADAMENTE, sem depender de rodar os `i-1` anteriores. É isso que faz o
    `make repro` funcionar sob paralelismo (SPEC-10).
    """
    root = np.random.SeedSequence(root_seed)
    child = root.spawn(episode_index + 1)[episode_index]
    derived = int(child.generate_state(1, dtype=np.uint32)[0])
    return np.random.default_rng(child), derived


def generate_scenario(config: ScenarioConfig, rng: np.random.Generator, seed: int = 0) -> Scenario:
    """Sorteia posições de sensores e UAVs.

    Sensores: uniformes em [-scenario_size, scenario_size]^2 (Lamenza §3).
    UAVs: uniformes em [-spawn_radius, spawn_radius]^2 — o código de referência
    usa `random.uniform(-2, 2)`, que é um QUADRADO e não um disco. Reproduzido
    como está.
    """
    s = config.scenario_size
    sensor_positions = rng.uniform(-s, s, size=(config.n_sensors, 2))
    r = config.spawn_radius
    agent_positions = rng.uniform(-r, r, size=(config.n_agents, 2))
    return Scenario(
        config=config,
        agent_positions=agent_positions,
        sensor_positions=sensor_positions,
        seed=seed,
    )
