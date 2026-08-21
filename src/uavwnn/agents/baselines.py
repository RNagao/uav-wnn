"""Os três baselines — SPEC-02 T-02.2, T-02.3 e T-02.4.

    random_uniforme   piso              — rumo e velocidade sorteados
    guloso            o de Lamenza      — sensor não visitado mais próximo
    kmeans_tsp        régua             — planejador centralizado offline

O guloso é **o baseline da fig. 7.3**, e é contra ele que a H6 se mede. O
k-means+TSP é acréscimo nosso: não participa do aprendizado, não entra na
observação e não é candidato a implantação. É régua, e a coluna
`usa_estado_verdadeiro` carrega isso na própria tabela.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import Acao, ContextoEpisodio, EstadoVerdadeiro, Observacao, V_MAX


# --------------------------------------------------------------------------
# T-02.2 — piso
# --------------------------------------------------------------------------


@dataclass
class CaminhadaAleatoria:
    """Rumo e velocidade uniformes a cada decisão. **O piso.**

    Vai falhar na esmagadora maioria dos episódios com `max_seconds_stalled =
    30 s`, e é justamente por isso que a imputação D-02.2 existe: a média só
    sobre os que ela concluiu seria média sobre os cenários mais fáceis.

    Declara `deterministica = False` — é estocástica por natureza, e continua
    consumindo o `rng` que recebe no `reset` mesmo em avaliação.
    """

    id: str = "random_uniforme"
    usa_estado_verdadeiro: bool = False
    _rng: np.random.Generator | None = field(default=None, init=False)

    def reset(self, ctx: ContextoEpisodio, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, obs: Observacao) -> Acao:
        assert self._rng is not None, "reset() antes de act()"
        return Acao(theta_norm=float(self._rng.random()),
                    s_norm=float(self._rng.random()))

    def metadados(self) -> dict[str, Any]:
        return {"deterministica": False, "familia": "piso"}


# --------------------------------------------------------------------------
# T-02.3 — o baseline de Lamenza
# --------------------------------------------------------------------------


@dataclass
class Guloso:
    """Rumo para o sensor não visitado mais próximo, a `v_max`.

    **Lê apenas a observação** — nenhuma informação privilegiada. A observação já
    vem ordenada por distância crescente, então o primeiro slot válido é o alvo.

    **A inversão de sinal é o modo de falha desta política**, e o teste de sinal
    da T-02.9 §3 existe para pegá-la: com o sinal trocado o UAV foge do sensor e
    o episódio falha por parada, o que passaria por "heurística ruim" em vez de
    bug. `Observacao.decodificar` devolve `p_outro − p_k`, então o rumo é
    `atan2` direto, sem negação.

    Quando não há sensor visível — todos coletados, ou padding total — para
    (`s = 0`) em vez de vaguear.
    """

    id: str = "guloso"
    usa_estado_verdadeiro: bool = False
    velocidade: float = V_MAX

    def reset(self, ctx: ContextoEpisodio, rng: np.random.Generator) -> None:
        self._ctx = ctx

    def act(self, obs: Observacao) -> Acao:
        alvo = obs.sensor_mais_proximo
        if alvo is None or not np.all(np.isfinite(alvo)):
            return Acao(theta_norm=0.0, s_norm=0.0)
        return Acao.de_rumo(float(np.arctan2(alvo[1], alvo[0])), self.velocidade)

    def metadados(self) -> dict[str, Any]:
        return {"deterministica": True, "familia": "heuristica",
                "velocidade_ms": self.velocidade}


# --------------------------------------------------------------------------
# T-02.4 — planejador centralizado offline
# --------------------------------------------------------------------------


def held_karp(dist: NDArray[np.float64]) -> tuple[float, list[int]]:
    """Caminho aberto mínimo saindo do nó 0, exato por programação dinâmica.

    `dist` é `(m+1, m+1)` com o nó 0 = origem. Custo `2^m · m²`, que para
    `m ≤ 12` é ~5,9 × 10⁵ operações — desprezível.

    Caminho **aberto**: sem retorno à origem, porque o UAV não precisa voltar.
    """
    n = dist.shape[0]
    m = n - 1
    if m == 0:
        return 0.0, [0]
    INF = float("inf")
    # custo[mascara][j] = menor custo saindo de 0, visitando `mascara`, terminando em j
    custo = np.full((1 << m, m), INF)
    pai = np.full((1 << m, m), -1, dtype=int)
    for j in range(m):
        custo[1 << j, j] = dist[0, j + 1]
    for mascara in range(1 << m):
        for j in range(m):
            if not (mascara >> j) & 1 or custo[mascara, j] == INF:
                continue
            for k in range(m):
                if (mascara >> k) & 1:
                    continue
                nova = mascara | (1 << k)
                c = custo[mascara, j] + dist[j + 1, k + 1]
                if c < custo[nova, k]:
                    custo[nova, k] = c
                    pai[nova, k] = j
    cheia = (1 << m) - 1
    fim = int(np.argmin(custo[cheia]))
    total = float(custo[cheia, fim])
    rota, mascara, j = [], cheia, fim
    while j >= 0:
        rota.append(j + 1)
        pj = pai[mascara, j]
        mascara ^= 1 << j
        j = int(pj)
    return total, [0] + rota[::-1]


def vizinho_2opt(dist: NDArray[np.float64], rng: np.random.Generator,
                 reinicios: int = 8) -> tuple[float, list[int]]:
    """Vizinho mais próximo + 2-opt, para `m > 12`.

    O gap contra o exato é medido nas células em que os dois são viáveis — sem
    isso, "TSP" na tabela é uma palavra e não um número.
    """
    n = dist.shape[0]
    if n <= 1:
        return 0.0, [0]

    def comprimento(rota: list[int]) -> float:
        return float(sum(dist[rota[i], rota[i + 1]] for i in range(len(rota) - 1)))

    melhor, melhor_rota = float("inf"), None
    for r in range(reinicios):
        naovisit = set(range(1, n))
        rota = [0]
        # O primeiro reinício é vizinho mais próximo puro; os demais sorteiam o
        # primeiro salto, o que dá diversidade sem perder a estrutura gulosa.
        while naovisit:
            atual = rota[-1]
            if r > 0 and len(rota) == 1:
                prox = int(rng.choice(sorted(naovisit)))
            else:
                prox = min(naovisit, key=lambda j: dist[atual, j])
            rota.append(prox)
            naovisit.discard(prox)

        melhorou = True
        while melhorou:
            melhorou = False
            for i in range(1, len(rota) - 1):
                for j in range(i + 1, len(rota)):
                    candidata = rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]
                    if comprimento(candidata) < comprimento(rota) - 1e-12:
                        rota, melhorou = candidata, True
        c = comprimento(rota)
        if c < melhor:
            melhor, melhor_rota = c, rota
    return melhor, melhor_rota or [0]


def _kmeans(pontos: NDArray, k: int, rng: np.random.Generator,
            reinicios: int = 10) -> NDArray[np.int64]:
    """k-means++ com `reinicios`, escolha pela menor inércia.

    **Cluster vazio recebe o sensor mais distante do seu centroide** — acontece
    com `Na` grande e `Ns` pequeno, e sem esse tratamento um UAV ficaria sem
    rota.
    """
    n = len(pontos)
    if k >= n:
        return np.arange(n) % k
    melhor_inercia, melhor_rotulo = float("inf"), None
    for _ in range(reinicios):
        # k-means++
        centros = [pontos[rng.integers(n)]]
        for _ in range(k - 1):
            d2 = np.min(((pontos[:, None, :] - np.array(centros)[None]) ** 2).sum(-1),
                        axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            centros.append(pontos[rng.choice(n, p=p)])
        C = np.array(centros)
        for _ in range(100):
            d = np.linalg.norm(pontos[:, None, :] - C[None], axis=2)
            rot = d.argmin(axis=1)
            novos = np.array([pontos[rot == j].mean(axis=0) if (rot == j).any()
                              else C[j] for j in range(k)])
            if np.allclose(novos, C):
                break
            C = novos
        # cluster vazio: recebe o ponto mais distante do próprio centroide
        for j in range(k):
            if not (rot == j).any():
                d = np.linalg.norm(pontos - C[rot], axis=1)
                doador = int(np.argmax(d))
                rot[doador] = j
        inercia = float(((pontos - C[rot]) ** 2).sum())
        if inercia < melhor_inercia:
            melhor_inercia, melhor_rotulo = inercia, rot.copy()
    return melhor_rotulo if melhor_rotulo is not None else np.zeros(n, dtype=int)


def _designar(centroides: NDArray, pos_agentes: NDArray) -> list[int]:
    """Cluster → UAV pelo problema de designação, minimizando a soma das
    distâncias. Como os UAVs partem todos de uma área pequena no centro
    (Lamenza §3), a designação quase não importa — mas precisa ser
    **determinística**."""
    from scipy.optimize import linear_sum_assignment

    custo = np.linalg.norm(pos_agentes[:, None, :] - centroides[None], axis=2)
    _, col = linear_sum_assignment(custo)
    return list(col)


@dataclass
class KMeansTSP:
    """Planejador centralizado offline com posições verdadeiras. **D-02.1.**

    Planeja uma vez em `observar_verdade` e depois **executa waypoints** no mesmo
    simulador, com o mesmo `dt`, a mesma coleta por proximidade e as mesmas
    regras de término. Executar em vez de calcular analiticamente é decisão
    registrada: um tempo analítico teria de modelar a discretização de `dt` (o
    rumo só muda a cada passo, logo há *overshoot*), o corte por
    `communication_range` e o `max_episode_length` — cada uma uma aproximação
    **diferente** das que o simulador aplica às outras políticas.

    **NÃO é limite inferior e NÃO é ótimo.** O k-means minimiza inércia, não
    *makespan*; o problema alvo (mTSP min-max com alcance de coleta) é NP-difícil
    e não está sendo resolvido. A rotulação correta no relatório é *"planejador
    centralizado offline com posições verdadeiras"*. Para um limite honesto,
    `metricas.lb1`.
    """

    id: str = "kmeans_tsp"
    usa_estado_verdadeiro: bool = True
    reinicios_kmeans: int = 10
    limite_exato: int = 12
    """`m ≤ 12` usa Held–Karp exato; acima disso, vizinho mais próximo + 2-opt."""

    def reset(self, ctx: ContextoEpisodio, rng: np.random.Generator) -> None:
        self._ctx = ctx
        self._rng = rng
        self._rotas: dict[int, list[NDArray]] = {}
        self._idx: dict[int, int] = {}
        self._t_plan: dict[int, float] = {}
        self._gap_exato: float | None = None
        self._estado: EstadoVerdadeiro | None = None

    def observar_verdade(self, estado: EstadoVerdadeiro) -> None:
        """Recebe o estado verdadeiro **a cada decisão**.

        Planeja na primeira chamada e guarda o estado corrente nas seguintes. Isto
        é dentro do contrato: a política declara `usa_estado_verdadeiro = True`,
        e é o que a torna régua e não candidata a implantação.

        **Por que precisa a cada passo, e não só no `reset`:** o executor de
        waypoints precisa da posição **atual** do agente para calcular o rumo. Uma
        primeira versão deste código tentou derivar o rumo da observação
        relativa, e degenerava no guloso — a observação não carrega a posição
        absoluta, então "aponte para o waypoint 3" é inexprimível nela. Receber a
        verdade resolve, e é honesto porque o privilégio já estava declarado.
        """
        self._estado = estado
        if self._rotas:
            return                          # já planejou
        ctx = self._ctx
        sensores = np.asarray(estado.pos_sensores, float)
        agentes = np.asarray(estado.pos_agentes, float)
        k = ctx.num_drones

        rotulos = _kmeans(sensores, k, self._rng, self.reinicios_kmeans)
        centroides = np.array([
            sensores[rotulos == j].mean(axis=0) if (rotulos == j).any()
            else agentes[min(j, len(agentes) - 1)]
            for j in range(k)
        ])
        designacao = _designar(centroides, agentes)

        for agente in range(k):
            cluster = designacao[agente]
            indices = np.flatnonzero(rotulos == cluster)
            if indices.size == 0:
                self._rotas[agente] = []
                self._t_plan[agente] = 0.0
                self._idx[agente] = 0
                continue
            pontos = sensores[indices]
            nos = np.vstack([agentes[agente], pontos])
            d = np.linalg.norm(nos[:, None, :] - nos[None], axis=2)
            m = len(pontos)
            if m <= self.limite_exato:
                _, ordem = held_karp(d)
            else:
                _, ordem = vizinho_2opt(d, self._rng)
                if m <= self.limite_exato + 4:      # mede o gap onde é viável
                    exato, _ = held_karp(d)
                    heur = sum(d[ordem[i], ordem[i + 1]] for i in range(len(ordem) - 1))
                    self._gap_exato = float((heur - exato) / exato) if exato > 0 else 0.0
            # Guarda os ÍNDICES globais dos sensores, na ordem da rota: é o que
            # permite consultar `estado.coletado` para pular waypoints.
            self._rotas[agente] = [int(indices[i - 1]) for i in ordem[1:]]
            self._idx[agente] = 0
            self._t_plan[agente] = self._tempo_planejado(nos, ordem)

    def _tempo_planejado(self, nos: NDArray, ordem: list[int]) -> float:
        """T-02.4c, só para diagnóstico.

            T_plan = Σ_j max(0, d(w_{j-1}, w_j) − communication_range) / v_max

        **Ressalva registrada:** aproxima o ponto de partida da perna seguinte
        pelo waypoint, e não pelo ponto do disco onde a coleta de fato ocorreu.
        É aproximação, é diagnóstico, e o número autoritativo é sempre o
        simulado. A razão `T_plan / t_conc` diz se o gargalo é o planejador ou o
        executor — se for grande, o planejador está bom e o executor é o
        problema.
        """
        alc, v = self._ctx.communication_range, self._ctx.v_max
        total = 0.0
        for i in range(len(ordem) - 1):
            d = float(np.linalg.norm(nos[ordem[i + 1]] - nos[ordem[i]]))
            total += max(0.0, d - alc) / v
        return total

    def act(self, obs: Observacao) -> Acao:
        """Executor de waypoints — T-02.4c.

        Aponta para o waypoint corrente a `v_max`, **pulando** os já coletados —
        inclusive os que outro UAV coletou incidentalmente. Isso é pular, não
        replanejar; a variante que replaneja (`kmeans_tsp_replan`) está na linha
        de corte.

        Termina com `s = 0` quando a rota acaba, em vez de vaguear.
        """
        k = obs.k
        rota = self._rotas.get(k, [])
        if self._estado is None:
            return Acao(theta_norm=0.0, s_norm=0.0)

        coletado = np.asarray(self._estado.coletado, bool)
        pos = np.asarray(self._estado.pos_agentes, float)[k]

        # Pula waypoints já cobertos, por qualquer agente.
        while self._idx.get(k, 0) < len(rota) and coletado[rota[self._idx[k]]]:
            self._idx[k] += 1

        if self._idx.get(k, 0) >= len(rota):
            return Acao(theta_norm=0.0, s_norm=0.0)

        alvo = np.asarray(self._estado.pos_sensores, float)[rota[self._idx[k]]]
        delta = alvo - pos
        if not np.any(np.abs(delta) > 1e-9):
            return Acao(theta_norm=0.0, s_norm=0.0)
        return Acao.de_rumo(float(np.arctan2(delta[1], delta[0])), self._ctx.v_max)

    def metadados(self) -> dict[str, Any]:
        """Seguro **antes** do `reset`.

        O harness chama isto para montar o `manifest.json`, e a ordem em que o
        faz não é garantida. Uma versão anterior acessava `self._t_plan`
        diretamente e quebrava com `AttributeError` numa política recém
        construída.
        """
        return {"deterministica": True, "familia": "planejador",
                "reinicios_kmeans": self.reinicios_kmeans,
                "limite_exato": self.limite_exato,
                "gap_heuristico_vs_exato": getattr(self, "_gap_exato", None),
                "t_plan_por_agente": dict(getattr(self, "_t_plan", {}))}


BASELINES = {
    "random_uniforme": CaminhadaAleatoria,
    "guloso": Guloso,
    "kmeans_tsp": KMeansTSP,
}
