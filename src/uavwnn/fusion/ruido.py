"""Camada de corrupção — SPEC-07 T-07.1.

Um canal só, por decisão de escopo (D13): o **GPS do drone**. O canal do sensor
saiu, e com ele o filtro de informação, a H4, a T3 e a P3. O que permanece é o
canal cujo modelo estava fechado — ruído branco, estado móvel, comando conhecido
— e onde vive o achado medido A3 (`12,54 m → 0,67 m`, 18,8×).

O atraso de broadcast fica como **provisão da SPEC-09**: a fila ordenada por
carimbo de tempo é propriedade do filtro, não do multiagente, e retrofitá-la
depois obrigaria a re-sintonizar tudo.

**Números aleatórios comuns (CRN), e por que importam.** A camada gera o desvio
padronizado `ε ~ N(0, I₂)` a partir de um fluxo que depende de
`(semente, canal, nó, episódio)` e **não** de `σ`; a corrupção é `z = p + σ·ε`.
Consequência: as quatro condições de `σ` da varredura veem exatamente a mesma
realização normalizada, e a comparação entre elas é pareada. O mesmo vale para
`T`: mudar a resolução do termômetro não pode mudar o ruído. Isso reduz a
variância da varredura `σ × T` — o experimento mais caro do projeto — de graça.

**Unidade canônica é o metro** (T-07.1). Nenhum `σ` normalizado em código; a
corrupção é em metros, sobre a posição absoluta, antes de qualquer subtração
relativa e antes da normalização do codificador.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

CANAIS = ("gps", "broadcast")


def _chave_estavel(texto: str) -> int:
    """Hash determinístico entre execuções (o `hash()` embutido é aleatorizado
    por `PYTHONHASHSEED`). Mesmo mecanismo de `eval.sementes`."""
    return int.from_bytes(hashlib.sha256(texto.encode()).digest()[:8], "big")


def fluxo_ruido(raiz: int, canal: str, no: int, episodio: int) -> np.random.Generator:
    """Gerador de um canal de ruído, independente dos demais canais.

    A independência entre canais é o que garante que ligar um degrau da escada
    não muda o ruído dos outros (T-07.1): consumir o fluxo de GPS não pode
    alterar o fluxo do broadcast. `spawn_key` nomeado por canal isola os fluxos.
    """
    if canal not in CANAIS:
        raise ValueError(f"canal desconhecido: {canal!r}; use um de {CANAIS}")
    seq = np.random.SeedSequence(
        raiz,
        spawn_key=(_chave_estavel(canal), int(no), int(episodio)),
    )
    return np.random.default_rng(seq)


@dataclass(frozen=True)
class ConfigRuido:
    """Parâmetros da corrupção, serializável e gravada junto com o resultado.

    Nenhum parâmetro de ruído com default espalhado pelo código: uma configuração
    é este objeto, e ela vai para o `manifest.json`.
    """

    sigma_gps: float
    """Desvio-padrão marginal do ruído de posição própria, em metros."""

    modelo: str = "branco"
    """`"branco"` (nominal) ou `"gauss_markov"` (ablação de robustez, T-07.1a)."""

    # -- Gauss-Markov (só quando modelo == "gauss_markov") ------------------
    tau: float = 120.0
    """Constante de tempo da componente correlacionada, em segundos."""
    fracao_markov: float = 0.5
    """Fração da variância marginal na componente lentamente variável.

    `σ_b² = f·σ_gps²`, `σ_w² = (1−f)·σ_gps²`: a variância marginal continua
    `σ_gps²`, então a comparação com o modo branco isola a **correlação**, não a
    magnitude. É o que o NEES tem de detectar (o KF nominal assume branco e fica
    otimista sob correlação).
    """

    # -- broadcast (provisão da SPEC-09) -----------------------------------
    atraso_base: float = 0.0
    """Atraso determinístico de entrega, em segundos."""
    jitter: float = 0.0
    """Faixa do atraso adicional `U[0, jitter]`, em segundos."""
    p_perda: float = 0.0
    """Probabilidade de perda independente por mensagem."""

    def __post_init__(self) -> None:
        if self.modelo not in ("branco", "gauss_markov"):
            raise ValueError(f"modelo inválido: {self.modelo!r}")
        if not 0.0 <= self.fracao_markov <= 1.0:
            raise ValueError("fracao_markov fora de [0,1]")

    @property
    def sigma_b(self) -> float:
        return self.sigma_gps * float(np.sqrt(self.fracao_markov))

    @property
    def sigma_w(self) -> float:
        return self.sigma_gps * float(np.sqrt(1.0 - self.fracao_markov))

    def como_dict(self) -> dict[str, Any]:
        return asdict(self)


class CanalGPS:
    """Corrompe a posição própria do drone. `p̂ = p + e`, `e` por leitura.

    No modo `branco`, `e = σ·ε` com `ε ~ N(0, I₂)` i.i.d. — o modelo nominal, e o
    único em que o KF é ótimo. No modo `gauss_markov`, `e` tem uma componente
    lentamente variável (multicaminho, geometria de constelação) que o KF nominal
    não modela e sob a qual ele fica **otimista** — é a ablação que o teste de
    brancura da inovação (T-07.6) tem de acusar.
    """

    def __init__(self, config: ConfigRuido, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng
        self._b = np.zeros(2, dtype=np.float64)  # estado Gauss-Markov
        self._t = None  # instante da última leitura, para o dt do Gauss-Markov

    def medir(self, p: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Devolve a posição corrompida `p̂` no instante `t`.

        O `ε` é sacado do fluxo próprio do canal, independente de `σ`: dois canais
        com `σ` diferente e a mesma semente veem o mesmo `ε` (CRN).
        """
        p = np.asarray(p, dtype=np.float64)
        eps = self._rng.standard_normal(2)
        if self.config.modelo == "branco":
            return p + self.config.sigma_gps * eps
        # Gauss-Markov de 1ª ordem + branco, variância marginal σ_gps².
        dt = 0.0 if self._t is None else (t - self._t)
        a = float(np.exp(-dt / self.config.tau)) if dt > 0 else 1.0
        eta = self._rng.standard_normal(2)
        self._b = a * self._b + np.sqrt(max(0.0, 1.0 - a * a)) * self.config.sigma_b * eta
        self._t = t
        return p + self._b + self.config.sigma_w * eps


@dataclass(order=True)
class _MsgAgendada:
    entrega: float
    ordem: int
    emissao: float = field(compare=False)
    conteudo: Any = field(compare=False)


class FilaMedidas:
    """Fila de mensagens ordenada por carimbo de tempo — provisão da SPEC-09.

    O tratamento de medida fora de ordem é propriedade do filtro (T-07.2), não do
    multiagente: uma mensagem emitida em `t_e` chega em `t_e + Δ`, e as entregas
    de emissores diferentes podem se cruzar. A fila reordena por instante de
    entrega antes que qualquer coisa toque o KF, e descarta o que é mais velho
    que `janela` (por omissão um intervalo de decisão), expondo o contador.
    """

    def __init__(self, janela: float = 2.0) -> None:
        self.janela = janela
        self._fila: list[_MsgAgendada] = []
        self._ordem = 0
        self.descartadas = 0

    def enviar(
        self, conteudo: Any, t_emissao: float, config: ConfigRuido,
        rng: np.random.Generator,
    ) -> bool:
        """Agenda a entrega, aplicando atraso e perda. Devolve `False` se perdida."""
        if config.p_perda > 0 and rng.random() < config.p_perda:
            return False
        atraso = config.atraso_base + (
            rng.uniform(0.0, config.jitter) if config.jitter > 0 else 0.0
        )
        self._fila.append(_MsgAgendada(t_emissao + atraso, self._ordem, t_emissao, conteudo))
        self._ordem += 1
        return True

    def entregar_ate(self, t: float) -> list[Any]:
        """Mensagens entregues até `t`, **em ordem de carimbo de emissão**.

        Mensagens cuja emissão é mais velha que `janela` em relação a `t` são
        descartadas (o filtro não reprocessa passado arbitrário) e contadas.
        """
        prontas = [m for m in self._fila if m.entrega <= t]
        self._fila = [m for m in self._fila if m.entrega > t]
        saida = []
        for m in sorted(prontas, key=lambda x: x.emissao):
            if t - m.emissao > self.janela:
                self.descartadas += 1
                continue
            saida.append(m.conteudo)
        return saida
