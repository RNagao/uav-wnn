"""Corpus de observações, colhido do ambiente de referência (SPEC-01 §4).

O gerador analítico que a spec original previa foi descartado: o ambiente roda
a 0,02–0,15 s por episódio, então o corpus sai direto dele e o invariante V-1
deixa de existir — com ele, o maior risco daquela spec.

A estratificação por fração de sensores restantes é obrigatória: no fim do
episódio a maioria dos slots é padding, e uma métrica agregada mistura dois
regimes muito diferentes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..refenv import LAMENZA_003, load_heuristics, make_env


@dataclass(frozen=True)
class Corpus:
    """Observações e o contexto de onde cada uma veio."""

    obs: NDArray[np.float32]
    """(N, D) — o vetor de observação, exatamente como o ambiente entrega."""

    n_restantes: NDArray[np.int16]
    """(N,) — sensores não visitados no instante da observação."""

    n_sensores: NDArray[np.int16]
    """(N,) — Ns da célula."""

    n_agentes: NDArray[np.int16]
    """(N,) — Na da célula."""

    politica: NDArray[np.int8]
    """(N,) — 0 = guloso (on-policy), 1 = aleatório (cobertura)."""

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def filtrar(self, **cond) -> "Corpus":
        m = np.ones(len(self), dtype=bool)
        for chave, valor in cond.items():
            m &= getattr(self, chave) == valor
        return Corpus(
            obs=self.obs[m], n_restantes=self.n_restantes[m],
            n_sensores=self.n_sensores[m], n_agentes=self.n_agentes[m],
            politica=self.politica[m],
        )

    @property
    def fracao_restante(self) -> NDArray[np.float32]:
        return (self.n_restantes / self.n_sensores).astype(np.float32)


def coletar_corpus(
    celulas: list[tuple[int, int]],
    n_episodios: int = 30,
    seed0: int = 0,
    politicas: tuple[str, ...] = ("greedy", "random"),
) -> Corpus:
    """Roda episódios e devolve todas as observações de todos os agentes.

    `greedy` dá a distribuição **on-policy** — a que a política vai encontrar.
    `random` dá cobertura, para o codificador não ser ajustado ao guloso.
    """
    h = load_heuristics()
    fabricas = {
        "greedy": h.create_greedy_heuristics,
        "random": h.create_random_heuristics,
    }
    obs_l, rest_l, ns_l, na_l, pol_l = [], [], [], [], []

    for na, ns in celulas:
        cfg = LAMENZA_003(na, ns)
        for idx_pol, nome in enumerate(politicas):
            pol = fabricas[nome](na - 1, cfg.state_num_closest_sensors)
            for ep in range(n_episodios):
                env = make_env(cfg)
                random.seed(seed0 + ep)
                np.random.seed(seed0 + ep)
                obs, _ = env.reset()
                while True:
                    restantes = ns - int(env.sensors_collected)
                    for a in env.agents:
                        obs_l.append(np.asarray(obs[a], dtype=np.float32))
                        rest_l.append(restantes)
                        ns_l.append(ns)
                        na_l.append(na)
                        pol_l.append(idx_pol)
                    obs, _, term, trunc, infos = env.step(
                        {a: pol(obs[a]) for a in env.agents}
                    )
                    if infos and "avg_reward" in next(iter(infos.values())):
                        break
                    if any(term.values()) or any(trunc.values()):
                        break
                env.close()

    return Corpus(
        obs=np.stack(obs_l),
        n_restantes=np.array(rest_l, dtype=np.int16),
        n_sensores=np.array(ns_l, dtype=np.int16),
        n_agentes=np.array(na_l, dtype=np.int16),
        politica=np.array(pol_l, dtype=np.int8),
    )
