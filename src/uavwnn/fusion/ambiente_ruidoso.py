"""Ambiente com ruído no GPS do drone — a costura da Etapa 2 (T-07.12).

Subclasse do `GrADySEnvironment` de Lamenza que **não edita uma linha** do código
de referência: sobrescreve apenas `observe_simulation_relative_positions`, o
único método por onde a política enxerga o mundo. A física — movimento, coleta,
recompensa, terminação — continua exatamente a de Lamenza (RNF-07 permanece
verdadeiro: nada em `refenv/` importa isto).

O que muda, e só isto (D13): a **posição própria** do drone é corrompida por GPS
e filtrada. A posição dos sensores e dos vizinhos chega exata — o canal
correlacionado entre UAVs é da Etapa 3 (SPEC-09). Como toda entrada da observação
tem a forma `(p_k − p_i + S)/(2S)`, o erro de posição própria é um **deslocamento
comum** a todas as entradas: filtrar esse único canal melhora a observação
inteira (ARQUITETURA §3.7).

Três degraus da escada:

- **0** — sem ruído: `passthrough` para o método do pai, bit a bit (o teto).
- **1** — ruído bruto: a observação é montada sobre a **medida** `z`, sem filtro.
- **4** — KF+controle: montada sobre a **estimativa** `x̂`, com a ação da política
  como entrada de controle.

**A verdade não vaza (RF-04):** o método só devolve observação derivada de `z`
ou `x̂`; a posição verdadeira do drone entra apenas no registro de métricas
(`self.log_estimacao`), nunca no vetor que a política recebe. As posições
verdadeiras no `Simulator` ficam intocadas — as métricas de tempo de conclusão
saem delas, como no ambiente puro.

**O ranking sai da estimativa** (ARQUITETURA §3.4): os sensores e vizinhos são
ordenados por distância da posição **estimada**, nunca da verdadeira — herdar o
ranking verdadeiro devolveria parte do canal global que a Etapa 2 fecha.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from uavwnn.refenv.loader import load_environment

from .kalman import FiltroKalmanControle
from .ruido import CanalGPS, ConfigRuido, fluxo_ruido

V_MAX = 15.0


def _classe_base():
    """A classe `GrADySEnvironment` de referência, carregada do submodule."""
    return load_environment().GrADySEnvironment


def criar_ambiente_ruidoso(
    config,
    degrau: int,
    config_ruido: ConfigRuido,
    raiz_ruido: int = 20260815,
    q_w: float = 0.0,
    sigma_v0: float = 0.0,
    kappa: float = 0.0,
):
    """Instancia a subclasse com os kwargs do `EnvConfig` (mesma API de `make_env`).

    A subclasse é criada dinamicamente sobre a classe de referência carregada do
    submodule — não há herança em tempo de importação, então o módulo importa sem
    o `gradysim` presente (só `criar_ambiente_ruidoso` o exige).
    """
    Base = _classe_base()

    class AmbienteRuidoso(Base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.degrau = degrau
            self.config_ruido = config_ruido
            self.raiz_ruido = raiz_ruido
            self._q_w, self._sigma_v0, self._kappa = q_w, sigma_v0, kappa
            self.indice_ruido = 0          # o runner sobrescreve por episódio (CRN)
            self.log_estimacao: list[dict] = []
            self._preparar_fusao()

        # -- estado da fusão, por episódio ---------------------------------

        def _preparar_fusao(self) -> None:
            n = self.num_drones
            self._kf = {
                i: FiltroKalmanControle(
                    sigma_gps=self.config_ruido.sigma_gps,
                    q_w=self._q_w, sigma_v0=self._sigma_v0, kappa=self._kappa,
                )
                for i in range(n)
            }
            self._canais = {
                i: CanalGPS(self.config_ruido,
                            fluxo_ruido(self.raiz_ruido, "gps", i, self.indice_ruido))
                for i in range(n)
            }
            self._acao = {i: np.zeros(2) for i in range(n)}
            self._iniciado = {i: False for i in range(n)}

        def reset(self, seed=None, options=None):
            # recria a fusão ANTES do observe inicial que o reset do pai dispara
            self._preparar_fusao()
            self.log_estimacao = []
            return super().reset(seed, options)

        def step(self, actions):
            # captura a ação comandada por drone → entrada de controle u
            for indice, acao in enumerate(actions.values()):
                direcao = float(acao[0]) * 2 * math.pi
                velocidade = float(acao[1]) * V_MAX
                self._acao[indice] = velocidade * np.array([math.cos(direcao), math.sin(direcao)])
            return super().step(actions)

        # -- a observação, com a posição própria substituída ----------------

        def _posicao_usada(self, i: int, p_true: NDArray[np.float64], t: float) -> NDArray[np.float64]:
            """Posição própria que alimenta a observação do drone `i`.

            Degrau 1: a medida crua. Degrau 4: a estimativa do KF, que usa a ação
            como entrada de controle. A verdade só vai para o log de métricas.
            """
            z = self._canais[i].medir(p_true, t)
            if self.degrau == 1:
                return z
            # degrau 4
            kf = self._kf[i]
            if not self._iniciado[i]:
                kf.iniciar(z, t)
                self._iniciado[i] = True
            else:
                kf.comando(self._acao[i])
                kf.atualizar(z, t)
            x_hat = kf.media
            # RF-04: a verdade entra SÓ no log, nunca na observação
            self.log_estimacao.append({
                "t": t, "drone": i,
                "erro": (p_true - x_hat).tolist(),
                "cov": kf.covariancia.tolist(),
            })
            return x_hat

        def observe_simulation_relative_positions(self):
            if self.degrau == 0:
                return super().observe_simulation_relative_positions()

            t = self.episode_duration * self.algorithm_iteration_interval
            S = self.scenario_size

            sensor_nodes = np.array(
                [self.simulator.get_node(sid).position[:2] for sid in self.sensor_node_ids]
            )
            nao_visitados = np.array([
                not self.simulator.get_node(sid).protocol_encapsulator.protocol.has_collected
                for sid in self.sensor_node_ids
            ])
            sensores_livres = sensor_nodes[nao_visitados]
            drones = np.array(
                [self.simulator.get_node(aid).position[:2] for aid in self.agent_node_ids]
            )

            estado = {}
            for i in range(self.num_drones):
                p_own = self._posicao_usada(i, drones[i], t)  # medida ou estimativa

                # sensores não visitados mais próximos, ranqueados pela ESTIMATIVA
                d_sensores = np.linalg.norm(sensores_livres - p_own, axis=1)
                ordem_s = np.argsort(d_sensores)
                sensores_obs = np.full((self.state_num_closest_sensors, 2), -1.0)
                escolhidos = sensores_livres[ordem_s[: self.state_num_closest_sensors]]
                sensores_obs[: len(escolhidos)] = (p_own - escolhidos + S) / (2 * S)

                # vizinhos mais próximos, EXCLUINDO o próprio (por índice, não por
                # "distância zero" — sob ruído o próprio não está a distância zero)
                outros = [j for j in range(self.num_drones) if j != i]
                vizinhos_obs = np.full((self.state_num_closest_drones, 2), -1.0)
                if outros:
                    pos_outros = drones[outros]
                    d_viz = np.linalg.norm(pos_outros - p_own, axis=1)
                    ordem_v = np.argsort(d_viz)
                    escolhidos_v = pos_outros[ordem_v[: self.state_num_closest_drones]]
                    vizinhos_obs[: len(escolhidos_v)] = (p_own - escolhidos_v + S) / (2 * S)

                partes = [vizinhos_obs.flatten(), sensores_obs.flatten()]
                if self.id_on_state:
                    partes.append(np.array([i / self.num_drones]))
                estado[f"drone{i}"] = np.concatenate(partes)
            return estado

    return AmbienteRuidoso(render_mode=None, **config.kwargs())
