"""Métricas por episódio, censura e imputação — SPEC-02 T-02.5.

O PROBLEMA QUE NÃO PODE FICAR IMPLÍCITO

Qual é o tempo de conclusão de um episódio que **falhou**?

Com `max_seconds_stalled = 30 s`, a caminhada aleatória falha na esmagadora
maioria dos episódios. Duas saídas ingênuas, e as duas quebram:

- **Média só sobre os que concluíram** é média sobre os cenários mais fáceis —
  viés de sobrevivência que faz o piso parecer bom.
- **Imputar o instante da poda** *premia* uma política que trava cedo: falhar
  rápido daria tempo baixo.

DECISÃO D-02.2 — imputação em `max_episode_length`

    t_conc_pen = t_conc                se sucesso
                 max_episode_length    caso contrário   (= 500 s, tab. 7.1)

É monótona nas duas dimensões que importam: falhar é sempre pior que concluir, e
concluir mais cedo é sempre melhor. Não é manipulável por travar cedo, e é um
escalar único sobre o qual o bootstrap opera sem tratamento especial. É o análogo
do escore **PAR-1** de competições com limite de tempo; a variante PAR-2
(`2 × max_episode_length`) fica registrada e não usada.

**E NÃO É INVENÇÃO NOSSA — é fidelidade ao original.** O `environment.py` de
referência já faz exatamente isto, em duas linhas:

    # linha 674
    "completion_time": self.max_episode_length if not all_sensors_collected
                       else self.simulator._current_timestamp

    # linha 551 — o vetor NASCE penalizado, e cada coleta sobrescreve a sua posição
    self.collection_times = [self.max_episode_length for _ in range(self.num_sensors)]

Ou seja, **Lamenza usa PAR-1 nas duas métricas de tempo**: por episódio em
`completion_time` e por sensor em `avg_collection_time`. Chegamos à mesma escolha
por raciocínio antes de ler essas linhas, e a coincidência tem consequência
prática: **os nossos números são diretamente comparáveis com os publicados, sem
conversão**.

Por isso `t_conc_s` guarda o valor que o ambiente devolve — já penalizado — e
`t_conc_pen_s` é reaplicação defensiva. Ver a nota em cada um.

**A primária sozinha esconde o mecanismo**, então `resumir_episodios` devolve
sempre o conjunto: `t_conc_pen`, `taxa_sucesso`, `t_conc_sucesso` (com `n`
explícito e o aviso preso), mediana e p90.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

MAX_EPISODE_LENGTH = 500.0
"""Tab. 7.1 de Lamenza. É o valor da imputação de D-02.2."""


@dataclass(frozen=True)
class MetricasEpisodio:
    """Um episódio, um método, uma célula. Grão de `episodios.parquet`."""

    idx_episodio: int
    seed_cenario: int
    sucesso: bool
    n_visitados: int
    n_sensores: int
    t_conc_s: float
    """O `completion_time` do ambiente de referência, **já penalizado**.

    Cuidado ao ler `episodios.csv`: esta coluna **não** é `NaN` nas falhas — ela
    traz `max_episode_length`, porque é o que a linha 674 do `environment.py`
    devolve. `NaN` aparece só quando o episódio termina **sem** `info` populada,
    o que acontece se o laço quebra por `termination`/`truncation` antes de o
    ambiente montar o dicionário.

    Guardamos o valor do ambiente em vez de reconstruir o nosso para manter a
    métrica **idêntica à do artigo**. Quem quer o tempo de conclusão sem
    penalização usa `t_conc_s` filtrado por `sucesso`, que é o que
    `resumir_episodios` faz.
    """
    t_fim_s: float
    motivo_fim: str           # sucesso | stalled | timeout
    n_decisoes: int
    coleta_media_s: float = float("nan")
    """`avg_collection_time` do ambiente — a **quarta métrica do artigo**.

    Média dos instantes de coleta, com os sensores não coletados entrando como
    `max_episode_length` (linha 551 do `environment.py`). Ver o aviso em
    `Resumo.media_coleta`.
    """
    dist_por_agente_m: list[float] = field(default_factory=list)
    visitas_por_agente: list[int] = field(default_factory=list)
    coletas_redundantes: int = 0
    lb1_s: float = float("nan")
    t_plan_s: float = float("nan")

    @property
    def t_conc_pen_s(self) -> float:
        """D-02.2. **A métrica primária.**

        Reaplicação defensiva: como o ambiente já penaliza (linha 674), isto é
        no-op quando `info` veio populada. Existe pelo caso em que ela não veio —
        aí `t_conc_s` é `NaN` e propagar `NaN` para o bootstrap envenenaria a
        média inteira em silêncio.
        """
        return float(self.t_conc_s) if self.sucesso else MAX_EPISODE_LENGTH

    @property
    def gini_visitas(self) -> float:
        """Desigualdade na divisão de trabalho entre agentes.

        0 = perfeitamente dividido, 1 = um agente fez tudo. É o número que
        sustenta a leitura de coordenação de H6 — o `greedy` voa em formação, e
        isso aparece aqui além de aparecer na separação média.
        """
        return gini(self.visitas_por_agente)


def gini(x: Sequence[float]) -> float:
    """Coeficiente de Gini. `nan` para amostra vazia ou soma nula."""
    v = np.sort(np.asarray(x, dtype=float))
    n = v.size
    if n == 0 or v.sum() == 0:
        return float("nan")
    if n == 1:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2 * (indices * v).sum()) / (n * v.sum()) - (n + 1) / n)


def lb1(
    pos_sensores: np.ndarray,
    pos_agentes: np.ndarray,
    communication_range: float,
    v_max: float,
) -> float:
    """LB-1 — limite inferior **válido**, T-02.4d.

        T ≥ max_i  min_k  (‖p_i − p_k⁰‖ − communication_range)₊ / v_max

    Todo sensor precisa ficar a menos de `communication_range` de algum agente, e
    nenhum agente anda mais rápido que `v_max`. É frouxo — ignora que um agente
    não pode estar em dois lugares — mas é **válido**, custa microssegundos e vai
    para a tabela como coluna própria.

    **Um resultado abaixo de LB-1 é bug, não descoberta**, e o harness assere
    isso.
    """
    s = np.atleast_2d(np.asarray(pos_sensores, float))
    a = np.atleast_2d(np.asarray(pos_agentes, float))
    if s.size == 0 or a.size == 0:
        return 0.0
    # (n_sensores, n_agentes)
    d = np.linalg.norm(s[:, None, :] - a[None, :, :], axis=2)
    alcance = np.maximum(0.0, d - communication_range)
    return float(alcance.min(axis=1).max() / v_max)


@dataclass(frozen=True)
class Resumo:
    """Agregado de um método numa célula. Grão de `agregado.csv`."""

    n_ep: int
    taxa_sucesso: float
    n_sucesso: int
    media_t_conc_pen: float
    mediana_t_conc_pen: float
    p90_t_conc_pen: float
    media_t_conc_sucesso: float
    """Média só sobre os bem-sucedidos.

    **AVISO PRESO À MÉTRICA:** inválida para comparar métodos com taxas de
    sucesso diferentes — é média sobre os cenários que cada um conseguiu, e não
    sobre os mesmos cenários. Existe porque é a leitura mais provável da fig. 7.3
    de Lamenza, não porque seja comparável.
    """
    media_lb1: float
    media_coleta: float = float("nan")
    """`avg_collection_time` do ambiente, média sobre os episódios.

    **MESMO AVISO de `media_t_conc_sucesso`, e pelo mesmo motivo.** A linha 551
    do `environment.py` inicializa `collection_times` em `max_episode_length` e
    divide por `num_sensors` sempre, então num episódio com 11 de 12 coletados o
    sensor faltante entra com 500 s e domina a média. É consistente com PAR-1 —
    mas torna a métrica **incomparável entre métodos com taxas de sucesso
    diferentes**, exatamente como a média sobre sucessos.
    """

    def como_linha(self) -> dict:
        return {
            "n_ep": self.n_ep, "taxa_sucesso": self.taxa_sucesso,
            "n_sucesso": self.n_sucesso,
            "media_t_conc_pen": self.media_t_conc_pen,
            "mediana": self.mediana_t_conc_pen, "p90": self.p90_t_conc_pen,
            "media_t_conc_sucesso": self.media_t_conc_sucesso,
            "media_coleta": self.media_coleta,
            "media_lb1": self.media_lb1,
        }


def _media_ignorando_nan(x: Sequence[float]) -> float:
    """`nanmean` sem o `RuntimeWarning` de fatia vazia.

    Acontece de verdade: `coleta_media_s` é `NaN` quando o `info` do ambiente não
    veio, e num conjunto onde isso ocorre em todos os episódios o `np.nanmean`
    avisa e devolve `nan`. O aviso é ruído — `nan` já é a resposta certa —, mas
    poluir a saída de teste esconde avisos que importam.
    """
    v = np.asarray(x, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def resumir_episodios(eps: Iterable[MetricasEpisodio]) -> Resumo:
    """Agrega, devolvendo o conjunto inteiro — a primária nunca sozinha."""
    lista = list(eps)
    if not lista:
        raise ValueError("nenhum episódio")
    pen = np.array([e.t_conc_pen_s for e in lista], float)
    ok = np.array([e.sucesso for e in lista], bool)
    conc = np.array([e.t_conc_s for e in lista], float)[ok]
    lb = np.array([e.lb1_s for e in lista], float)
    return Resumo(
        n_ep=len(lista),
        taxa_sucesso=float(ok.mean()),
        n_sucesso=int(ok.sum()),
        media_t_conc_pen=float(pen.mean()),
        mediana_t_conc_pen=float(np.median(pen)),
        p90_t_conc_pen=float(np.percentile(pen, 90)),
        media_t_conc_sucesso=float(conc.mean()) if conc.size else float("nan"),
        media_lb1=_media_ignorando_nan(lb),
        media_coleta=_media_ignorando_nan([e.coleta_media_s for e in lista]),
    )


def sucesso_comum(*conjuntos: Sequence[MetricasEpisodio]) -> np.ndarray:
    """Máscara dos episódios em que **todos** os métodos tiveram sucesso.

    Base do subconjunto `sucesso_comum` das comparações: média sem viés de
    sobrevivência, porque restringe aos mesmos cenários.

    **Degenera para vazio** quando a caminhada aleatória entra na comparação, e
    nesse caso a T-02.5 manda não reportar — daí devolver a máscara em vez de já
    agregar, para o chamador poder checar `mask.sum()`.
    """
    if not conjuntos:
        return np.zeros(0, bool)
    n = len(conjuntos[0])
    if any(len(c) != n for c in conjuntos):
        raise ValueError("pareamento exige mesmo número de episódios")
    m = np.ones(n, bool)
    for c in conjuntos:
        m &= np.array([e.sucesso for e in c], bool)
    return m
