"""Interface `Policy` congelada — SPEC-02 T-02.1.

**O harness não pode distinguir baseline de política aprendida.** É esse contrato
que permite rodar caminhada aleatória, guloso, k-means+TSP, n-tuple Actor-Critic e
MLP pareada pelo mesmo cano — mesma censura, mesmo bootstrap, mesma tabela.

CINCO REGRAS, TODAS COM TESTE

1. **A ação é sempre normalizada em `[0,1]²`**, como Lamenza §5.2. A
   desnormalização é do ambiente: `θ = a₀·2π` e `v = a₁·15`, verificado no
   `environment.py` de referência. Manter a mesma moeda entre baselines e
   política gaussiana é o que torna a comparação possível.
2. **`vetor` é canônico**; `p_a`, `p_s` e `valido_s` são **vistas** do mesmo
   conteúdo, não informação nova. `valido_s` é a convenção de padding tornada
   explícita, e seu consumo é regulado: o codificador da WNN pode usá-la, a
   **MLP pareada não pode** — senão deixa de ser fiel à entrada de Lamenza.
3. **Privilégio é declarado, não inferido.** O runner só chama
   `observar_verdade` se `usa_estado_verdadeiro` for `True`. É o invariante da
   RF-04 antecipado aqui e reaproveitado na Etapa 2.
4. **Avaliação é determinística por padrão.** Lamenza §7: *"During an evaluation
   the training is temporarily halted and action noise is removed"*. Políticas
   estocásticas por natureza declaram `deterministica = False` nos metadados.
5. **`reset` recebe o `rng`; a política nunca cria o seu.** Fonte única de
   aleatoriedade, derivada pelo harness (T-02.6).

O LAYOUT DA OBSERVAÇÃO, LIDO DO CÓDIGO DE REFERÊNCIA

`observe_simulation_relative_positions` concatena, nesta ordem:

    [ agentes  (2·n_a) | sensores não visitados (2·12) | índice do agente (1) ]

com cada par normalizado por `(p_k − p_outro + S) / (2S)` e **padding `-1`**.
Logo a inversão é `p_k − p_outro = v·2S − S`, e a direção para o outro é
`−(v·2S − S)`. `Observacao.decodificar` faz exatamente isso, e há teste de ida e
volta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

PADDING = -1.0
"""Sentinela de slot vazio. Fora da faixa dos válidos, o que torna o bit de
validade desnecessário (PRD §5.4)."""

V_MAX = 15.0
"""m/s. `speed: float = action[1] * 15` no `environment.py`."""


@dataclass(frozen=True)
class ContextoEpisodio:
    """O que a política pode saber **antes** do episódio, sem privilégio.

    São parâmetros do cenário (tab. 3.1 e 7.1), não estado. O guloso precisa de
    `scenario_size` para desnormalizar a observação, e isso não é vazamento — é
    a constante de normalização, publicada.
    """

    num_drones: int
    num_sensores: int
    scenario_size: float
    communication_range: float
    dt: float
    v_max: float = V_MAX
    max_episode_length: float = 500.0
    state_num_closest_sensors: int = 12
    state_num_closest_drones: int | None = None
    """`None` = `num_drones − 1`, o default do `benchmark.py`.

    Existe explícito porque o experimento exploratório do §7.1 usa **`N*_a = 0`** —
    agentes cegos aos demais —, e aí a observação tem 25 escalares
    **independentemente de `Na`**. Derivar de `num_drones` faria a decodificação
    esperar duas entradas que não existem, e o erro só apareceria em `Na > 1`."""


@dataclass(frozen=True)
class EstadoVerdadeiro:
    """Posições reais. **Só chega a políticas com `usa_estado_verdadeiro`.**

    Existe para o k-means+TSP, que é régua e não candidato a implantação. O
    runner assere que nenhuma outra política o recebe.
    """

    pos_sensores: NDArray[np.float64]      # (Ns, 2)
    pos_agentes: NDArray[np.float64]       # (Na, 2)
    coletado: NDArray[np.bool_]            # (Ns,)


@dataclass(frozen=True)
class Observacao:
    """Eq. 5-1 de Lamenza, com vistas convenientes do mesmo conteúdo."""

    vetor: NDArray[np.float64]
    p_a: NDArray[np.float64]               # (n_a, 2) relativas, em METROS
    p_s: NDArray[np.float64]               # (12, 2)  relativas, em METROS
    valido_s: NDArray[np.bool_]            # (12,) False onde há padding
    valido_a: NDArray[np.bool_]            # (n_a,)
    k: int
    t: float

    @classmethod
    def decodificar(cls, vetor, ctx: ContextoEpisodio, t: float = 0.0) -> "Observacao":
        """Reconstrói as vistas a partir do vetor canônico.

        `p_a` e `p_s` saem em **metros e apontando do agente para o outro**:
        `p_outro − p_k = S − v·2S`. É essa a convenção que o guloso usa para
        calcular rumo, e inverter o sinal aqui daria uma política que **foge**
        dos sensores — o modo de falha que o teste de sinal da T-02.9 §3 pega.
        """
        v = np.asarray(vetor, dtype=float).ravel()
        n_a = (ctx.num_drones - 1 if ctx.state_num_closest_drones is None
               else ctx.state_num_closest_drones)
        n_s = ctx.state_num_closest_sensors
        esperado = 2 * n_a + 2 * n_s + 1
        if v.size != esperado:
            raise ValueError(
                f"observação com {v.size} entradas, esperado {esperado} "
                f"(= 2·{n_a} + 2·{n_s} + 1)")

        bruto_a = v[:2 * n_a].reshape(n_a, 2) if n_a else np.zeros((0, 2))
        bruto_s = v[2 * n_a:2 * n_a + 2 * n_s].reshape(n_s, 2)
        S = ctx.scenario_size

        val_a = ~np.isclose(bruto_a, PADDING).all(axis=1) if n_a else np.zeros(0, bool)
        val_s = ~np.isclose(bruto_s, PADDING).all(axis=1)

        # p_outro − p_k = −(p_k − p_outro) = −(bruto·2S − S) = S − bruto·2S
        rel_a = S - bruto_a * 2 * S
        rel_s = S - bruto_s * 2 * S
        rel_a[~val_a] = np.nan
        rel_s[~val_s] = np.nan

        return cls(vetor=v, p_a=rel_a, p_s=rel_s, valido_s=val_s,
                   valido_a=val_a, k=int(round(v[-1] * ctx.num_drones)), t=t)

    @property
    def sensor_mais_proximo(self) -> NDArray[np.float64] | None:
        """Vetor relativo ao sensor não visitado mais próximo, ou `None`.

        A observação **já vem ordenada** por distância crescente (o
        `argsort` do `observe_simulation_relative_positions`), então o primeiro
        slot válido é o mais próximo — não é preciso recalcular.
        """
        idx = np.flatnonzero(self.valido_s)
        return None if idx.size == 0 else self.p_s[idx[0]]


@dataclass(frozen=True)
class Acao:
    """Normalizada em `[0,1]²`. A desnormalização é do ambiente."""

    theta_norm: float
    s_norm: float

    def __post_init__(self) -> None:
        for nome, v in (("theta_norm", self.theta_norm), ("s_norm", self.s_norm)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{nome}={v} fora de [0,1]")

    def como_array(self) -> NDArray[np.float64]:
        return np.array([self.theta_norm, self.s_norm], dtype=float)

    @staticmethod
    def de_rumo(theta_rad: float, velocidade_ms: float = V_MAX) -> "Acao":
        """De rumo em radianos e velocidade em m/s para a moeda normalizada.

        `θ` é envolvido em `[0, 2π)` **antes** de dividir — é a topologia
        circular da ação, e é a pendência C1 do projeto. Aqui o wrap é trivial
        porque a ação é escolhida, não amostrada; na política gaussiana não é.
        """
        theta = float(np.mod(theta_rad, 2 * np.pi))
        return Acao(theta_norm=theta / (2 * np.pi),
                    s_norm=float(np.clip(velocidade_ms / V_MAX, 0.0, 1.0)))


@runtime_checkable
class Policy(Protocol):
    """Contrato mínimo. O harness não sabe mais que isto sobre a política."""

    id: str
    usa_estado_verdadeiro: bool

    def reset(self, ctx: ContextoEpisodio, rng: np.random.Generator) -> None: ...
    def act(self, obs: Observacao) -> Acao: ...
    def metadados(self) -> dict[str, Any]: ...


@runtime_checkable
class PoliticaPrivilegiada(Policy, Protocol):
    """Recebe o estado verdadeiro. **Não é candidata a implantação.**"""

    def observar_verdade(self, estado: EstadoVerdadeiro) -> None: ...


# --------------------------------------------------------------------------
# adaptador para o runner da SPEC-00
# --------------------------------------------------------------------------


@dataclass
class AdaptadorRunner:
    """Faz uma `Policy` parecer o callable que `refenv.run_episode` espera.

    O runner da SPEC-00 recebe `(observacao_crua) -> acao_array` e não sabe de
    `Observacao` nem de `reset`. Este adaptador decodifica, chama a política e
    devolve o array — mantendo o runner ignorante do contrato, como o PRD §7
    exige.
    """

    politica: Policy
    ctx: ContextoEpisodio
    t: float = 0.0
    chamadas: int = field(default=0, init=False)

    def __call__(self, observacao) -> NDArray[np.float64]:
        obs = Observacao.decodificar(observacao, self.ctx, self.t)
        self.chamadas += 1
        return self.politica.act(obs).como_array()

    def avancar_tempo(self) -> None:
        """O runner chama uma vez por época de decisão; o `dt` vem do cenário."""
        self.t += self.ctx.dt
