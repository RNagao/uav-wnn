"""Bancada sintética — SPEC-07 §4.

Nenhuma tarefa da spec, exceto a integração final com o `gradysim` (T-07.12,
cortável), precisa do simulador: ruído e filtro são funções de uma trajetória e
de uma sequência de comandos. É a spec que mais se beneficia de rodar em paralelo
com a SPEC-06.

Dois níveis, do mais controlado ao mais fiel:

- **B1** — comando sintético paramétrico: `u` constante por partes, com a
  distribuição do salto de rumo `Δθ` controlável de "rumo quase constante" a
  "rumo i.i.d. uniforme". Isola o efeito que matou o KF de velocidade constante.
- **B2** — rollout guloso analítico: 1 UAV rumo ao sensor não visitado mais
  próximo, `dt = 2 s`, `v = 15 m/s`. É a estatística **on-policy** aproximada.
  Autocontido — o `env/analytic.py` da SPEC-01 foi descartado, então não há
  dependência do `gradysim`.

O que a bancada precisa reproduzir não é a trajetória, é o **resíduo de modelo**
`e_k = Δp_k − u_k·dt` (T-07.5): é ele que define `Q`. Com mobilidade sem massa
(o modelo de Lamenza) esse resíduo é zero e `Q` degenera — o filtro vira mediador
recursivo perfeito, e é provavelmente daí que vêm os 18,8× do achado A3. A
inércia, que o tornaria mensurável, é ablação declarada e não a configuração
nominal (T-07.5), porque trocar a mobilidade alteraria o cenário de Lamenza.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

V_MAX = 15.0
DT_DEC = 2.0
CENARIO = 100.0  # semi-lado, em metros; a arena é [-100, 100]²
ALCANCE = 20.0   # communication_range


@dataclass
class Trajetoria:
    """Trajetória verdadeira com comando constante por intervalo de decisão.

    A posição em qualquer `t` dentro de `[t_k, t_{k+1}]` é `p_k + u_k·(t − t_k)`,
    porque o comando é constante no intervalo — o que permite corromper o GPS em
    qualquer instante de fix, não só nos de decisão.
    """

    posicoes: NDArray[np.float64]   # (K+1, 2), nos instantes de decisão
    comandos: NDArray[np.float64]   # (K, 2), u aplicado em cada intervalo
    instantes: NDArray[np.float64]  # (K+1,)

    @property
    def n_passos(self) -> int:
        return self.comandos.shape[0]

    def posicao_em(self, t: float) -> NDArray[np.float64]:
        """Posição verdadeira interpolada no instante `t`."""
        k = int(np.searchsorted(self.instantes, t, side="right")) - 1
        k = max(0, min(k, self.n_passos - 1))
        return self.posicoes[k] + self.comandos[k] * (t - self.instantes[k])


def _integra(p0: NDArray[np.float64], comandos: NDArray[np.float64], dt: float) -> Trajetoria:
    K = comandos.shape[0]
    pos = np.zeros((K + 1, 2), dtype=np.float64)
    pos[0] = p0
    for k in range(K):
        pos[k + 1] = pos[k] + comandos[k] * dt
    inst = np.arange(K + 1, dtype=np.float64) * dt
    return Trajetoria(pos, comandos, inst)


def trajetoria_b1(
    rng: np.random.Generator,
    n_passos: int = 250,
    dt: float = DT_DEC,
    v: float = V_MAX,
    sigma_salto: float = np.pi,
) -> Trajetoria:
    """B1 — comando constante por partes com salto de rumo controlável.

    `sigma_salto` governa a distribuição de `Δθ` entre decisões: `→ 0` é rumo
    quase constante (fácil para qualquer filtro), `π` é i.i.d. aproximadamente
    uniforme (o pior caso para o KF de velocidade constante, que trata cada
    virada como surpresa). A velocidade é fixa em `v`.
    """
    theta = float(rng.uniform(0, 2 * np.pi))
    comandos = np.zeros((n_passos, 2), dtype=np.float64)
    for k in range(n_passos):
        if sigma_salto >= np.pi:
            theta = float(rng.uniform(0, 2 * np.pi))
        else:
            theta = float(theta + rng.normal(0, sigma_salto))
        comandos[k] = v * np.array([np.cos(theta), np.sin(theta)])
    p0 = rng.uniform(-2.0, 2.0, size=2)  # área de largada de Lamenza
    return _integra(p0, comandos, dt)


def trajetoria_b2(
    rng: np.random.Generator,
    n_sensores: int = 12,
    dt: float = DT_DEC,
    v: float = V_MAX,
    max_passos: int = 250,
) -> Trajetoria:
    """B2 — rollout guloso analítico, autocontido (sem `gradysim`).

    1 UAV parte do centro, ruma ao sensor não visitado mais próximo a `v`, coleta
    quando entra no `communication_range`. Gera trajetórias com viradas bruscas de
    rumo — a estatística que importa para `Q` — sem depender do simulador. Não
    reproduz o ambiente de referência; reproduz o **padrão de comando** dele.
    """
    sensores = rng.uniform(-CENARIO, CENARIO, size=(n_sensores, 2))
    visitado = np.zeros(n_sensores, dtype=bool)
    p = rng.uniform(-2.0, 2.0, size=2)
    comandos = []
    pos = [p.copy()]
    for _ in range(max_passos):
        if visitado.all():
            break
        livres = np.where(~visitado)[0]
        d = np.linalg.norm(sensores[livres] - p, axis=1)
        alvo = sensores[livres[int(np.argmin(d))]]
        direcao = alvo - p
        norma = float(np.linalg.norm(direcao))
        u = v * direcao / norma if norma > 1e-9 else np.zeros(2)
        comandos.append(u)
        p = p + u * dt
        pos.append(p.copy())
        # coleta por proximidade, como no ambiente de referência
        d_pos = np.linalg.norm(sensores - p, axis=1)
        visitado |= d_pos <= ALCANCE
    comandos_arr = np.array(comandos, dtype=np.float64) if comandos else np.zeros((0, 2))
    pos_arr = np.array(pos, dtype=np.float64)
    inst = np.arange(pos_arr.shape[0], dtype=np.float64) * dt
    return Trajetoria(pos_arr, comandos_arr, inst)


@dataclass
class ResultadoFiltragem:
    """Saída de uma execução: erros filtrados, covariâncias e inovações.

    `erros`/`covs` alimentam o NEES; `inovacoes`/`esses` o NIS; `erro_cru` é a
    medida sem filtro (o piso da escada) e `erro_filtrado` o degrau avaliado.
    """

    erros: NDArray[np.float64]        # (K, 2) verdade − estimativa, nos instantes de decisão
    covs: NDArray[np.float64]         # (K, 2, 2)
    inovacoes: NDArray[np.float64]    # (Nfix, 2)
    esses: NDArray[np.float64]        # (Nfix, 2, 2)
    erro_cru: NDArray[np.float64]     # (Nfix,) |z − p| na medida
    erro_filtrado: NDArray[np.float64]  # (K,) |x̂ − p| no instante de decisão

    @property
    def rmse_cru(self) -> float:
        return float(np.sqrt(np.mean(self.erro_cru**2)))

    @property
    def rmse_filtrado(self) -> float:
        return float(np.sqrt(np.mean(self.erro_filtrado**2)))


def rodar_filtro(
    traj: Trajetoria,
    canal,
    filtro,
    dt_dec: float = DT_DEC,
    dt_gps: float = DT_DEC,
) -> ResultadoFiltragem:
    """Roda um filtro sobre uma trajetória, com GPS a cada `dt_gps`.

    O filtro é iniciado no primeiro fix (`x̂_0 = z_0`, `P_0 = R`). Em cada
    intervalo de decisão, o comando é registrado, os fixes daquele intervalo são
    aplicados na ordem correta (`predizer_ate(t_z) → atualizar`), e a estimativa
    é lida no instante de decisão — que é o que a política enxergaria.

    A regra de taxa (T-07.2) está aqui: um `predizer_ate` por evento (fix ou
    decisão), nunca por tique de mobilidade.
    """
    t0 = float(traj.instantes[0])
    z0 = canal.medir(traj.posicao_em(t0), t0)
    filtro.iniciar(z0, t0)

    erros, covs = [], []
    inovacoes, esses, erro_cru, erro_filtrado = [], [], [], []

    K = traj.n_passos
    for k in range(K):
        t_ini = float(traj.instantes[k])
        t_fim = float(traj.instantes[k + 1])
        filtro.comando(traj.comandos[k])

        # fixes dentro do intervalo (t_ini, t_fim], evitando repetir t_ini
        n_fix = max(1, int(round((t_fim - t_ini) / dt_gps)))
        for j in range(1, n_fix + 1):
            t_fix = t_ini + j * dt_gps
            if t_fix > t_fim + 1e-9:
                break
            p_true = traj.posicao_em(t_fix)
            z = canal.medir(p_true, t_fix)
            # NIS: inovação ANTES da correção
            if hasattr(filtro, "inovacao"):
                filtro.predizer_ate(t_fix)
                y, S = filtro.inovacao(z)
                inovacoes.append(y)
                esses.append(S)
            filtro.atualizar(z, t_fix)
            erro_cru.append(float(np.linalg.norm(z - p_true)))

        x_hat, P = filtro.estimativa(t_fim)
        p_true = traj.posicao_em(t_fim)
        erros.append(p_true - x_hat)
        covs.append(P)
        erro_filtrado.append(float(np.linalg.norm(x_hat - p_true)))

    return ResultadoFiltragem(
        erros=np.array(erros, dtype=np.float64),
        covs=np.array(covs, dtype=np.float64),
        inovacoes=np.array(inovacoes, dtype=np.float64) if inovacoes else np.zeros((0, 2)),
        esses=np.array(esses, dtype=np.float64) if esses else np.zeros((0, 2, 2)),
        erro_cru=np.array(erro_cru, dtype=np.float64),
        erro_filtrado=np.array(erro_filtrado, dtype=np.float64),
    )
