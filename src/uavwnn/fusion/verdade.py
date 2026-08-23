"""Invariante de vazamento RF-04 — SPEC-07 T-07.8.

O que se afirma: o caminho `observação → política` **não é função da posição
verdadeira**. A verdade é corrompida na entrada e a cópia exata fica guardada só
para métricas — inclusive para o NEES, que precisa dela por definição.

Provar isso é menos trivial do que parece: o vazamento pode ocorrer por valor
derivado, por conversão para escalar, por variável global ou por dentro do
simulador. Quatro camadas, com papéis distintos:

1. **Ponto único de custódia** (`RegistroVerdade`) — a verdade existe num só
   objeto, que só se escreve; ler exige um `TokenMetricas` construível apenas
   nos módulos de métrica. Torna o vazamento acidental impossível de escrever
   distraidamente.
2. **Tipos distintos** (`PosicaoVerdadeira` vs `PosicaoMedida`) — a política só
   aceita a medida. Pega o erro na hora de escrever. Não sobrevive à conversão
   para `ndarray` cru.
3. **Marcação em tempo de execução** (`ArrayMarcado`) — propaga uma marca por
   arrays derivados. Pega quem calcula `dx = p_true − x̂` e põe na observação.
   **Não sobrevive a `float()`/`.item()`/`.tolist()`** — por isso é a rede de
   segurança, não a prova.
4. **Teste diferencial** (`GravadorRuido`/`ReprodutorRuido`, exercido em
   `tests/test_vazamento.py`) — **esta é a prova**. Grava-e-repete o fluxo de
   ruído, desloca a verdade por `δ` só no caminho de telemetria, e exige hash
   idêntico da sequência de ações. Agnóstico à implementação: qualquer
   dependência da política na verdade, por qualquer canal, quebra a igualdade.

**Escopo honesto do invariante** (T-07.8): o `δ` incide só sobre os canais que a
Etapa 2 declara corromper — a posição própria. O **status de visitação** continua
oracular; o canal global de Lamenza é fechado pela metade, e o relatório precisa
dizê-lo.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Camada 1 — os únicos módulos autorizados a construir um token de leitura da
# verdade. `bancada` calcula NEES na bancada; `consistencia` e a futura
# `eval.metricas` são os consumidores legítimos.
_MODULOS_AUTORIZADOS = frozenset({
    "uavwnn.fusion.bancada",
    "uavwnn.fusion.consistencia",
    "uavwnn.eval.metricas",
})


@dataclass(frozen=True)
class PosicaoVerdadeira:
    """Posição real, `[x, y]`. Nunca alimenta a política — só métricas."""

    xy: tuple[float, float]

    def como_array(self) -> NDArray[np.float64]:
        return np.array(self.xy, dtype=np.float64)


@dataclass(frozen=True)
class PosicaoMedida:
    """Posição corrompida, `[x, y]`. O único tipo que a política aceita."""

    xy: tuple[float, float]

    def como_array(self) -> NDArray[np.float64]:
        return np.array(self.xy, dtype=np.float64)


class TokenMetricas:
    """Camada 1 — chave de leitura da verdade, construível só nos módulos de métrica.

    A verificação por pilha de chamadas é ceremonial de propósito: o valor real
    é que `RegistroVerdade.ler` **exige** um token, então uma leitura acidental
    não compila. A restrição de módulo documenta, por construção, quem está
    autorizado a ver a verdade.
    """

    def __init__(self) -> None:
        quadro = inspect.stack()[1]
        modulo = inspect.getmodule(quadro[0])
        nome = modulo.__name__ if modulo else ""
        if nome not in _MODULOS_AUTORIZADOS:
            raise PermissionError(
                f"TokenMetricas só pode ser construído em {sorted(_MODULOS_AUTORIZADOS)}, "
                f"não em {nome!r}"
            )


class RegistroVerdade:
    """Camada 1 — ponto único de custódia. Escreve-se sem token; lê-se com token.

    O protocolo (ou o subclasse do ambiente) só chama `registrar`. Qualquer
    código que precise da verdade — o cálculo de NEES — tem de apresentar um
    `TokenMetricas`, que só os módulos de métrica constroem.
    """

    def __init__(self) -> None:
        self._verdade: dict[tuple[int, float], NDArray[np.float64]] = {}

    def registrar(self, t: float, no: int, p: NDArray[np.float64]) -> None:
        """Só escreve. Não devolve nada — não há como ler por aqui."""
        self._verdade[(int(no), float(t))] = np.asarray(p, dtype=np.float64).copy()

    def ler(self, token: TokenMetricas, t: float, no: int) -> NDArray[np.float64]:
        if not isinstance(token, TokenMetricas):
            raise PermissionError("ler() exige um TokenMetricas")
        return self._verdade[(int(no), float(t))].copy()

    def historico(self, token: TokenMetricas, no: int) -> list[tuple[float, NDArray[np.float64]]]:
        if not isinstance(token, TokenMetricas):
            raise PermissionError("historico() exige um TokenMetricas")
        return [
            (t, p.copy()) for (n, t), p in sorted(self._verdade.items()) if n == no
        ]


class ArrayMarcado(np.ndarray):
    """Camada 3 — `ndarray` com marca de verdade propagada por operações.

    Fatias, somas e diferenças herdam a marca via `__array_finalize__`, de modo
    que `p_true − x̂` chegue marcado à observação. Limite conhecido e
    documentado: a marca **não** sobrevive a `float(x)`, `.item()` nem
    `.tolist()` — por isso esta camada é rede de segurança, não prova.
    """

    def __new__(cls, dados: NDArray[np.float64]) -> "ArrayMarcado":
        obj = np.asarray(dados, dtype=np.float64).view(cls)
        obj._verdade = True
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self._verdade = getattr(obj, "_verdade", False)


def esta_marcado(x) -> bool:
    """`True` se `x` carrega a marca de verdade da camada 3."""
    return bool(getattr(x, "_verdade", False))


# Camada 4 — grava-e-repete o fluxo de ruído. A prova de RF-04.


class GravadorRuido:
    """Embrulha um canal de ruído e grava toda saída da camada de corrupção.

    Na corrida A do teste diferencial, registra a sequência exata de posições
    corrompidas para que a corrida B possa reproduzi-la bit a bit.
    """

    def __init__(self, canal) -> None:
        self._canal = canal
        self.gravado: list[NDArray[np.float64]] = []

    def medir(self, p: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        z = self._canal.medir(p, t)
        self.gravado.append(np.asarray(z, dtype=np.float64).copy())
        return z


class ReprodutorRuido:
    """Devolve exatamente os valores gravados por um `GravadorRuido`.

    Na corrida B, substitui o canal e ignora a posição verdadeira que recebe —
    é isso que permite deslocar a verdade por `δ` sem tocar no que a política vê.
    """

    def __init__(self, gravado: list[NDArray[np.float64]]) -> None:
        self._gravado = list(gravado)
        self._i = 0

    def medir(self, p: NDArray[np.float64], t: float) -> NDArray[np.float64]:  # noqa: ARG002
        z = self._gravado[self._i]
        self._i += 1
        return z.copy()
