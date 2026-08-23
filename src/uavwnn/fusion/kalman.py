"""Filtros de estado da Etapa 2 — SPEC-07 T-07.2, T-07.9, T-07.10.

O canal é **um só** (D13): o GPS do drone. O estado é a posição própria, o
comando é a velocidade que a política acabou de escolher, e é usá-lo como
entrada de controle que separa a proposta (degrau 4) do KF de velocidade
constante (degrau 3), que trata cada virada de rumo como surpresa e fica
permanentemente atrasado.

Quatro objetos, um por degrau da escada mais a variante barata:

- `FiltroKalmanControle` — degrau 4, a proposta. 2 estados `[x, y]`, `F = I`,
  entrada de controle `u = s·[cos θ, sin θ]`, forma de Joseph.
- `PassaBaixa` — degrau 2, o filtro barato que o KF precisa bater. Enviesado
  para um alvo em movimento; a variante *com* comando isola "usar o comando"
  de "usar Kalman".
- `FiltroVelocidadeConstante` — degrau 3, o "óbvio que falha". 4 estados, sem
  entrada de controle, `Q` de aceleração branca.
- `ganho_estacionario_escalar` / `FiltroGanhoEstacionario` — a ablação de
  custo: para `dt`, `q` e `R` fixos a recursão de Riccati converge e o filtro
  degenera em `x̂ = x̂⁻ + K∞(z − x̂⁻)`, ~10 flops sem matriz e sem divisão.

**Unidade canônica é o metro, sempre** (T-07.1). A corrupção e a filtragem
acontecem em metros, sobre a posição absoluta, antes de qualquer subtração
relativa e antes da normalização do codificador.

**Regra de taxa (a armadilha da §T-07.2).** Como `F = I` e `u` é constante
entre decisões, propagar a média em 20 passos de 0,1 s ou em 1 passo de 2 s dá
o mesmo `x̂⁻` — mas o termo `dt²` de `Q` composto 20 vezes difere por fator 20.
Logo há **um `predizer_ate` por intervalo entre eventos**, nunca um por tique de
mobilidade, e a taxa é parte da definição de `Q`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uavwnn.custo import CustoOperacoes


def _inv2x2(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inversa 2×2 em forma fechada.

    Nunca `np.linalg.inv` (T-07.2): a matriz de inovação é sempre 2×2, e a
    forma fechada é mais barata, exata e sem o despacho do LAPACK. O único ponto
    do ciclo com divisão — a categoria que a contagem de custo (T-07.9) trata à
    parte, porque num MCU sem FPU custa mais que um MAC.
    """
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]], dtype=np.float64) / det


class ErroDeSincronia(RuntimeError):
    """Tentou-se propagar o filtro para um instante ANTERIOR ao estado atual.

    É o erro mais provável da spec (T-07.2): aplicar uma medida antiga no
    instante corrente introduz um viés `‖u‖·Δt` sistemático — 3 m a 15 m/s com
    0,2 s de latência, a mesma ordem do `σ` que se quer remover, e invisível na
    variância. Levantar é melhor que enviesar em silêncio.
    """


@dataclass
class FiltroKalmanControle:
    """Degrau 4 — KF de 2 estados com a ação da política como entrada de controle.

    ```
    predição:  x̂⁻ = x̂ + dt·u          P⁻ = P + Q(dt)
    inovação:  ỹ  = z − x̂⁻            S  = P⁻ + R
    ganho:     K  = P⁻ S⁻¹
    correção:  x̂  = x̂⁻ + K ỹ
               P  = (I − K)P⁻(I − K)ᵀ + K R Kᵀ        (forma de Joseph)
    ```

    A forma de Joseph custa ~30 flops a mais (T-07.9) e garante `P` simétrica e
    positiva definida mesmo com `K` subótimo — que é exatamente o regime das
    ablações de `Q` mal sintonizado, onde a forma simples `(I−K)P⁻` pode produzir
    `P` sem sentido e um NEES sem significado.

    `Q(dt) = q_w·dt·I + σ_v0²·dt²·I` decompõe processo em duas fontes com
    composições diferentes na taxa: turbulência de banda larga (`q_w·dt`,
    invariante à taxa) e erro de atuação constante dentro do passo (`σ_v²·dt²`,
    não invariante). Com `kappa > 0` o termo de atuação cresce com o salto de
    comando `‖u_k − u_{k-1}‖²` — o erro de atuação é grande logo após uma virada
    e pequeno em rumo constante.
    """

    sigma_gps: float
    """Desvio-padrão do ruído de GPS, em metros. `R = σ_gps²·I`."""
    q_w: float = 0.0
    """Densidade de ruído de velocidade branco, m²/s. Termo `q_w·dt` de `Q`."""
    sigma_v0: float = 0.0
    """Erro de velocidade base (atuação/discretização), m/s. Termo `σ_v0²·dt²`."""
    kappa: float = 0.0
    """Sensibilidade de `Q` ao salto de comando. `0` = `Q` estacionário."""

    def __post_init__(self) -> None:
        self._x = np.zeros(2, dtype=np.float64)
        self._P = np.eye(2, dtype=np.float64) * self.sigma_gps**2
        self._R = np.eye(2, dtype=np.float64) * self.sigma_gps**2
        self._t = 0.0
        self._u = np.zeros(2, dtype=np.float64)
        self._u_ant = np.zeros(2, dtype=np.float64)
        self._iniciado = False

    # -- ciclo de vida ------------------------------------------------------

    def iniciar(self, z0: NDArray[np.float64], t0: float = 0.0) -> None:
        """`x̂_0 = z_0`, `P_0 = R`, no instante `t0`.

        Não `x̂_0 = 0` com `P_0` grande: essa escolha produz NEES explodindo nos
        primeiros passos, obrigando a descartar o transiente — que é justamente
        onde a política toma as decisões mais consequentes. Com `x̂_0 = z_0` e
        `P_0 = R` a consistência vale desde `k = 0`, exatamente.
        """
        self._x = np.asarray(z0, dtype=np.float64).copy()
        self._P = self._R.copy()
        self._t = float(t0)
        self._u[:] = 0.0
        self._u_ant[:] = 0.0
        self._iniciado = True

    def comando(self, u: NDArray[np.float64]) -> None:
        """Registra a velocidade comandada que valerá na próxima predição.

        `u = s·[cos θ, sin θ]`, em m/s. Guarda o comando anterior para o `Q`
        adaptativo ao salto de rumo.
        """
        self._u_ant = self._u.copy()
        self._u = np.asarray(u, dtype=np.float64).copy()

    # -- predição -----------------------------------------------------------

    def _Q(self, dt: float) -> NDArray[np.float64]:
        salto = float(np.dot(self._u - self._u_ant, self._u - self._u_ant))
        var = self.q_w * dt + (self.sigma_v0**2 + self.kappa * salto) * dt**2
        return np.eye(2, dtype=np.float64) * var

    def predizer_ate(self, t: float) -> None:
        """Propaga a estimativa até `t`, uma única vez, com o comando corrente.

        `Q` é composto sobre `dt = t − t_state` de uma vez só (regra de taxa).
        Retroceder no tempo levanta `ErroDeSincronia`.
        """
        if not self._iniciado:
            raise RuntimeError("iniciar() antes de predizer")
        dt = t - self._t
        if dt < -1e-12:
            raise ErroDeSincronia(
                f"predizer_ate({t}) < t_state={self._t}: medida fora de ordem"
            )
        if dt <= 1e-12:
            return
        self._x = self._x + dt * self._u
        self._P = self._P + self._Q(dt)
        self._t = t

    # -- correção -----------------------------------------------------------

    def atualizar(self, z: NDArray[np.float64], t: float) -> None:
        """Propaga até o instante do fix `t` e aplica a medida `z`.

        A medida refere-se ao instante do fix, não ao da decisão. A ordem
        correta — `predizer_ate(t_z) → correção` — está embutida aqui; ler `z`
        no instante errado é o viés sistemático da T-07.2.
        """
        self.predizer_ate(t)
        z = np.asarray(z, dtype=np.float64)
        S = self._P + self._R
        S_inv = _inv2x2(S)
        K = self._P @ S_inv
        y = z - self._x
        self._x = self._x + K @ y
        I = np.eye(2, dtype=np.float64)
        IKH = I - K
        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T
        # Simetriza contra deriva numérica; a forma de Joseph já a controla, mas
        # o custo é desprezível e o NEES é sensível a assimetria residual.
        self._P = 0.5 * (self._P + self._P.T)

    # -- leitura ------------------------------------------------------------

    def estimativa(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """`(x̂, P)` propagados até `t`. Não há como ler `x̂` sem dar o instante."""
        self.predizer_ate(t)
        return self._x.copy(), self._P.copy()

    @property
    def media(self) -> NDArray[np.float64]:
        return self._x.copy()

    @property
    def covariancia(self) -> NDArray[np.float64]:
        return self._P.copy()

    @property
    def instante(self) -> float:
        return self._t

    def inovacao(self, z: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Inovação `ỹ = z − x̂⁻` e sua covariância `S = P⁻ + R` no estado atual.

        Para o NIS (T-07.6), que não precisa da verdade — é o que existiria num
        voo real. Chamar depois de `predizer_ate(t_z)` e antes de `atualizar`.
        """
        z = np.asarray(z, dtype=np.float64)
        return z - self._x, self._P + self._R

    @staticmethod
    def custo_operacoes() -> CustoOperacoes:
        """Flops por ciclo (predição + correção), forma de Joseph, `P` simétrica.

        Deriva a faixa de 50–80 flops/passo do PRD (RNF-05) em vez de citá-la.
        Contra os 1 888 MACs fp da MLP pareada (SPEC-03), é ~4 %. A divisão da
        inversa 2×2 é a linha que um MCU sem FPU sente.
        """
        return CustoOperacoes(
            mult_fp=2 + 2 + 3 + 8 + 4 + 8,   # dt·u, S⁻¹ (2+3), K, x̂, Joseph parcial
            soma_fp=2 + 3 + 3 + 4 + 2 + 6,   # x̂⁻, P⁻, S, K, ỹ, x̂ e P
            div_fp=1,                        # a inversa 2×2
        )


@dataclass
class PassaBaixa:
    """Degrau 2 — média móvel exponencial. O filtro barato que o KF precisa bater.

    `x̂_k = (1−α)·(x̂_{k-1} + comando) + α·z_k`. Para um veículo em movimento é
    **enviesado**, não só menos eficiente: rastreando velocidade `v`, o atraso em
    regime permanente é `v·dt·(1−α)/α` — 30 m a 15 m/s, `dt = 2 s`, `α = 0,5`,
    uma ordem de grandeza acima de `σ_gps`. A variante *com* comando (`usar_comando`)
    é a ablação honesta: isola "usar o comando" de "usar Kalman". Se ela já
    captura a maior parte do ganho, a tese vira "conhecer o comando é o que
    importa, e Kalman é a forma principiada de usá-lo".
    """

    alpha: float
    usar_comando: bool = False

    def __post_init__(self) -> None:
        self._x = np.zeros(2, dtype=np.float64)
        self._t = 0.0
        self._u = np.zeros(2, dtype=np.float64)
        self._iniciado = False

    def iniciar(self, z0: NDArray[np.float64], t0: float = 0.0) -> None:
        self._x = np.asarray(z0, dtype=np.float64).copy()
        self._t = float(t0)
        self._u[:] = 0.0
        self._iniciado = True

    def comando(self, u: NDArray[np.float64]) -> None:
        self._u = np.asarray(u, dtype=np.float64).copy()

    def atualizar(self, z: NDArray[np.float64], t: float) -> None:
        z = np.asarray(z, dtype=np.float64)
        dt = t - self._t
        if dt < -1e-12:
            raise ErroDeSincronia(f"atualizar em t={t} < t_state={self._t}")
        predicao = self._x + (dt * self._u if self.usar_comando else 0.0)
        self._x = (1.0 - self.alpha) * predicao + self.alpha * z
        self._t = t

    def estimativa(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        dt = t - self._t
        pred = self._x + (dt * self._u if self.usar_comando else 0.0)
        # O passa-baixa não mantém covariância; devolve a identidade só para a
        # API casar com o KF. NEES não se aplica a ele (não tem `P` honesta).
        return pred, np.eye(2, dtype=np.float64)

    @property
    def media(self) -> NDArray[np.float64]:
        return self._x.copy()


@dataclass
class FiltroVelocidadeConstante:
    """Degrau 3 — KF de velocidade constante, 4 estados. O "óbvio que falha".

    `x = [x, y, vx, vy]`, `F = [[I, dt·I], [0, I]]`, `H = [I, 0]`, `Q` de
    aceleração branca:

    ```
    Q = q_a · [[ dt³/3·I ,  dt²/2·I ],
               [ dt²/2·I ,  dt·I    ]]
    ```

    A previsão de T-07.10: ele é **inconsistente** sob comando de rumo i.i.d.
    para todo `q_a`, porque nenhum `Q` de aceleração branca representa um salto de
    rumo determinístico. Não está mal sintonizado — está **mal especificado**. É
    o que transforma "o KF ingênuo é pior" de anedota em diagnóstico.
    """

    sigma_gps: float
    q_a: float = 1.0

    def __post_init__(self) -> None:
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64)
        self._R = np.eye(2, dtype=np.float64) * self.sigma_gps**2
        self._H = np.zeros((2, 4), dtype=np.float64)
        self._H[0, 0] = self._H[1, 1] = 1.0
        self._t = 0.0
        self._iniciado = False

    def iniciar(self, z0: NDArray[np.float64], t0: float = 0.0) -> None:
        self._x = np.zeros(4, dtype=np.float64)
        self._x[:2] = np.asarray(z0, dtype=np.float64)
        # Posição conhecida a `σ`, velocidade totalmente incerta.
        self._P = np.diag([self.sigma_gps**2, self.sigma_gps**2, 15.0**2, 15.0**2])
        self._t = float(t0)
        self._iniciado = True

    def comando(self, u: NDArray[np.float64]) -> None:  # noqa: ARG002
        """Sem entrada de controle — é o ponto do degrau 3. Aceito por simetria
        de API e ignorado."""

    def _F_Q(self, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = F[1, 3] = dt
        i2 = np.eye(2)
        Q = self.q_a * np.block([
            [dt**3 / 3 * i2, dt**2 / 2 * i2],
            [dt**2 / 2 * i2, dt * i2],
        ])
        return F, Q

    def predizer_ate(self, t: float) -> None:
        dt = t - self._t
        if dt < -1e-12:
            raise ErroDeSincronia(f"predizer_ate({t}) < t_state={self._t}")
        if dt <= 1e-12:
            return
        F, Q = self._F_Q(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._t = t

    def atualizar(self, z: NDArray[np.float64], t: float) -> None:
        self.predizer_ate(t)
        z = np.asarray(z, dtype=np.float64)
        S = self._H @ self._P @ self._H.T + self._R
        S_inv = _inv2x2(S)
        K = self._P @ self._H.T @ S_inv
        y = z - self._H @ self._x
        self._x = self._x + K @ y
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self._H
        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    def estimativa(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.predizer_ate(t)
        return self._x[:2].copy(), self._P[:2, :2].copy()

    @property
    def media(self) -> NDArray[np.float64]:
        return self._x[:2].copy()

    @property
    def covariancia(self) -> NDArray[np.float64]:
        return self._P[:2, :2].copy()

    def inovacao(self, z: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        z = np.asarray(z, dtype=np.float64)
        S = self._H @ self._P @ self._H.T + self._R
        return z - self._H @ self._x, S


def ganho_estacionario_escalar(q: float, sigma2: float) -> tuple[float, float]:
    """`(p∞, K∞)` do KF escalar com `Q = q`, `R = σ²`, `F = 1`.

    Da recursão `p = (p+q)σ²/(p+q+σ²)`, o ponto fixo satisfaz
    `p∞² + q·p∞ − q·σ² = 0`, logo `p∞ = (−q + √(q² + 4qσ²))/2` e
    `K∞ = (p∞ + q)/(p∞ + q + σ²)`. Vira asserção em `test_kalman.py`.
    """
    p_inf = (-q + math.sqrt(q * q + 4.0 * q * sigma2)) / 2.0
    p_pred = p_inf + q
    K_inf = p_pred / (p_pred + sigma2)
    return p_inf, K_inf


@dataclass
class FiltroGanhoEstacionario:
    """Ablação de custo (T-07.9) — KF de ganho constante `K∞`.

    Para `dt`, `q` e `R` fixos o ganho converge e o filtro degenera em
    `x̂ = x̂⁻ + K∞(z − x̂⁻)`: ~10 flops, sem matriz, sem divisão, implementável em
    inteiro. Preserva a narrativa de embarcabilidade quase intacta, ao custo do
    transiente inicial — que precisa ser reportado, não escondido.
    """

    sigma_gps: float
    q_w: float

    def __post_init__(self) -> None:
        _, self._K = ganho_estacionario_escalar(self.q_w, self.sigma_gps**2)
        self._x = np.zeros(2, dtype=np.float64)
        self._t = 0.0
        self._u = np.zeros(2, dtype=np.float64)

    @property
    def ganho(self) -> float:
        return self._K

    def iniciar(self, z0: NDArray[np.float64], t0: float = 0.0) -> None:
        self._x = np.asarray(z0, dtype=np.float64).copy()
        self._t = float(t0)
        self._u[:] = 0.0

    def comando(self, u: NDArray[np.float64]) -> None:
        self._u = np.asarray(u, dtype=np.float64).copy()

    def atualizar(self, z: NDArray[np.float64], t: float) -> None:
        z = np.asarray(z, dtype=np.float64)
        dt = t - self._t
        if dt < -1e-12:
            raise ErroDeSincronia(f"atualizar em t={t} < t_state={self._t}")
        x_pred = self._x + dt * self._u
        self._x = x_pred + self._K * (z - x_pred)
        self._t = t

    def estimativa(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        dt = t - self._t
        _, K = self._K, self._K
        p_inf, _ = ganho_estacionario_escalar(self.q_w, self.sigma_gps**2)
        return self._x + dt * self._u, np.eye(2) * p_inf

    @property
    def media(self) -> NDArray[np.float64]:
        return self._x.copy()

    @staticmethod
    def custo_operacoes() -> CustoOperacoes:
        return CustoOperacoes(mult_fp=2 + 2, soma_fp=2 + 2 + 2)
