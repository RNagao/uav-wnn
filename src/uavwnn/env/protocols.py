"""Protocolos do gradysim (SPEC-00, T-00.3).

Referência de mecanismo: `SensorProtocol` e `DroneProtocol` em
`scenario/003-collaboration/environment.py` (linhas 28-127).

RF-03: coletar NÃO é ação do agente. O UAV emite um broadcast por época de
decisão; o sensor que o recebe passa a ser considerado visitado. O alcance é
avaliado pelo `CommunicationHandler` do simulador.

Injeção de contexto: o gradysim instancia protocolos por `cls()` sem
argumentos (`IProtocol.instantiate` faz `protocol = cls()` e só depois atribui
`provider`). O código de referência resolve isso escrevendo em atributos de
classe globais (`DroneProtocol.speed_action = ...`), o que é estado global
mutável e quebra sob paralelismo. Aqui usamos subclasse por episódio, que é
equivalente em efeito e seguro por construção.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import (
    GotoCoordsMobilityCommand,
    SetSpeedMobilityCommand,
)
from gradysim.protocol.messages.telemetry import Telemetry

if TYPE_CHECKING:
    from .bridge import EpisodeContext


class SensorProtocol(IProtocol):
    """Nó estático. Não emite nada por conta própria; só reage ao broadcast."""

    _ctx: "EpisodeContext"

    def initialize(self) -> None:
        self.provider.tracked_variables["collected"] = False

    def handle_packet(self, message: str) -> None:
        # Qualquer pacote recebido significa que um UAV está no alcance.
        # O `CommunicationHandler` só entrega dentro de `transmission_range`,
        # então a checagem geométrica é do simulador, não nossa.
        newly = self._ctx.state.mark_visited(
            self.provider.get_id(), self.provider.current_time()
        )
        if newly:
            self.provider.tracked_variables["collected"] = True

    def handle_timer(self, timer: str) -> None:  # pragma: no cover - nunca dispara
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        # Chega mesmo assim: a mobilidade emite telemetria para TODOS os nós a
        # cada tique, inclusive os estáticos. É a origem do termo (1 + Na + Ns)
        # no modelo de custo.
        pass

    def finish(self) -> None:
        pass


class DroneProtocol(IProtocol):
    """UAV. A decisão é aplicada pelo driver via `act()`; ver `bridge.py`."""

    _ctx: "EpisodeContext"

    def initialize(self) -> None:
        # (0,0,0) e não a posição de spawn — verbatim do original.
        #
        # É tecnicamente incorreto: enquanto a primeira telemetria não chega, o
        # protocolo acredita estar na origem, e o spawn é sorteado em [-2,2]^2.
        # O efeito é pequeno e dura uma decisão: entra apenas no cálculo da
        # ESCALA do waypoint, e o waypoint é recomputado a cada época.
        #
        # Ainda assim, "corrigir" isto QUEBRA a equivalência com a referência.
        # Medido: com a correção, `max|Δobs|` contra a implementação de
        # referência vai de 7,9e-08 (precisão de float32) para 8,5e-01 em 60
        # passos. A causa é caos, não magnitude: a diferença de escala altera
        # levemente o rumo da primeira ação e as trajetórias divergem.
        #
        # PRD §0 manda reproduzir, não melhorar. Ver tests/test_referencia.py.
        self.current_position = (0.0, 0.0, 0.0)
        self._broadcast()

    def handle_timer(self, timer: str) -> None:
        # Único timer do UAV: o heartbeat, agendado por `act()`.
        self._broadcast()

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position

    def finish(self) -> None:
        pass

    # -- ação ---------------------------------------------------------------

    def act(self, u0: float, u1: float) -> None:
        """Aplica a ação normalizada `u = (u0, u1)` em [0, 1]^2.

        Lamenza eq. 3-2 e §5.2: `theta = u0 * 2*pi`, `speed = u1 * 15`.

        Mecanismo verbatim de `DroneProtocol.act` (environment.py, 59-102):
        o destino é colocado na BORDA do cenário ao longo da direção escolhida,
        não a um passo de distância. Combinado com `SetSpeedMobilityCommand`,
        isso faz o UAV percorrer exatamente `speed * dt` por intervalo, sem
        depender de o waypoint ser alcançado.

        ATENÇÃO ao fato 4 da SPEC-00 §4.1, verificado por execução:
        `SetVelocityMobilityCommand(vx, vy, vz)` compartilha o `command_type`
        SET_SPEED e, sob o handler massless, seria silenciosamente
        interpretado como `speed = vx`, SEM ERRO. Por isso a ação se aplica
        como SetSpeed + GotoCoords, e nunca como SetVelocity.
        """
        cfg = self._ctx.state.scenario.config
        direction = u0 * 2.0 * math.pi
        speed = u1 * cfg.max_speed

        self.provider.send_mobility_command(SetSpeedMobilityCommand(speed))

        ux, uy = math.cos(direction), math.sin(direction)
        x, y = self.current_position[0], self.current_position[1]
        limit = cfg.scenario_size

        dest = self._waypoint(x, y, ux, uy, limit, cfg.edge_behavior)
        self.provider.send_mobility_command(GotoCoordsMobilityCommand(dest[0], dest[1], 0.0))

        self.provider.schedule_timer(
            "", self.provider.current_time() + cfg.heartbeat_interval
        )

    @staticmethod
    def _waypoint(
        x: float, y: float, ux: float, uy: float, limit: float, mode: str
    ) -> tuple[float, float]:
        if mode == "reference":
            # Verbatim. Note que a distância é medida até a borda MAIS PRÓXIMA
            # em cada eixo (`limit - |x|`), independentemente do SENTIDO do
            # movimento. Perto de uma borda, `scale` colapsa e o UAV quase
            # para, mesmo apontando para dentro. É quirk do original, e é
            # reproduzida de propósito.
            dx = limit - abs(x)
            dy = limit - abs(y)
            if dx > 0 and dy > 0:
                scale = min(dx / (abs(ux) + 1e-10), dy / (abs(uy) + 1e-10))
                return (x + ux * scale, y + uy * scale)
            dest_x = max(-limit, min(limit, x + ux * 1e5))
            dest_y = max(-limit, min(limit, y + uy * 1e5))
            return (dest_x, dest_y)

        # "corrected": distância até a borda NO SENTIDO do movimento.
        t_x = ((limit if ux > 0 else -limit) - x) / ux if abs(ux) > 1e-10 else math.inf
        t_y = ((limit if uy > 0 else -limit) - y) / uy if abs(uy) > 1e-10 else math.inf
        t = max(0.0, min(t_x, t_y))
        return (
            max(-limit, min(limit, x + ux * t)),
            max(-limit, min(limit, y + uy * t)),
        )

    def _broadcast(self) -> None:
        self.provider.send_communication_command(BroadcastMessageCommand(""))


def bind_protocols(ctx: "EpisodeContext") -> tuple[type[IProtocol], type[IProtocol]]:
    """Cria subclasses ligadas ao contexto deste episódio."""
    sensor_cls = type("BoundSensorProtocol", (SensorProtocol,), {"_ctx": ctx})
    drone_cls = type("BoundDroneProtocol", (DroneProtocol,), {"_ctx": ctx})
    return sensor_cls, drone_cls
