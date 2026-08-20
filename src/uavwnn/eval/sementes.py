"""Árvore de sementes — SPEC-02 T-02.6.

O pareamento é o que faz 500 episódios renderem muito mais que 500 episódios não
pareados: **todos os métodos são avaliados exatamente nos mesmos cenários**. Como
a variância entre episódios é dominada pela geometria do cenário e não pela
política, parear remove a maior parte da variância da *diferença*.

Para isso, a aleatoriedade tem de vir de uma árvore explícita, e não de
`seed = i`:

    raiz
     ├── treino        ramos de cenário para treinar
     ├── avaliacao     ramos de cenário para avaliar  ← DISJUNTO de treino
     ├── bootstrap     reamostragem
     └── politica      fluxo interno das políticas estocásticas

**Por que `seed = i` é armadilha.** Avaliar num cenário visto no treino é
vazamento silencioso, e com `seed = i` em treino e `seed = i` em avaliação é
exatamente o que acontece. Aqui os dois ramos saem de `SeedSequence.spawn`, que
garante disjunção — e há teste.

**Por que não pareamos o fluxo da política.** Números aleatórios comuns só fazem
sentido no fluxo do *cenário*. Métodos diferentes consomem quantidades diferentes
de números aleatórios, e depois do primeiro descasamento não há correspondência
nenhuma. Tentar parear isso é ilusão; a semente de política é derivada de forma
determinística mas **independente** entre métodos.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

RAMOS = ("treino", "avaliacao", "bootstrap", "politica")


def _chave_estavel(texto: str) -> int:
    """Hash determinístico entre execuções e entre versões do Python.

    `hash()` embutido é aleatorizado por processo (`PYTHONHASHSEED`), então usar
    `hash(policy.id)` daria sementes diferentes a cada execução — o oposto de
    reprodutível. Este é estável.
    """
    return int.from_bytes(hashlib.sha256(texto.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class ArvoreSementes:
    """Sementes derivadas de uma raiz única, por ramo nomeado.

    A raiz vai para o `manifest.json`; com ela e este módulo, qualquer resultado
    do projeto é reproduzível.
    """

    raiz: int

    def _ramo(self, nome: str) -> np.random.SeedSequence:
        if nome not in RAMOS:
            raise ValueError(f"ramo desconhecido: {nome!r}; use um de {RAMOS}")
        # `spawn_key` nomeado em vez de posicional: acrescentar um ramo novo no
        # futuro não desloca as sementes dos existentes, e resultados antigos
        # continuam reproduzíveis.
        return np.random.SeedSequence(
            self.raiz, spawn_key=(_chave_estavel(nome),)
        )

    # -- cenários -----------------------------------------------------------

    def cenarios(self, ramo: str, n: int) -> list[int]:
        """`n` sementes de cenário do ramo pedido.

        São as sementes que o `refenv.run_episode` recebe, e é sobre elas que o
        pareamento acontece: **a mesma lista é usada por todos os métodos**.
        """
        if ramo not in ("treino", "avaliacao"):
            raise ValueError(f"cenários vêm de 'treino' ou 'avaliacao', não {ramo!r}")
        seq = self._ramo(ramo)
        return [int(s) for s in seq.generate_state(n, dtype=np.uint64)]

    # -- política -----------------------------------------------------------

    def politica(self, policy_id: str, idx_episodio: int) -> np.random.Generator:
        """Gerador da política, determinístico e independente entre métodos.

        Derivado de `(raiz_politica, hash(policy_id), idx_episodio)`, como a
        T-02.6 §2 prescreve. Duas políticas no mesmo episódio recebem fluxos
        diferentes — de propósito.
        """
        seq = np.random.SeedSequence(
            self.raiz,
            spawn_key=(_chave_estavel("politica"),
                       _chave_estavel(policy_id),
                       int(idx_episodio)),
        )
        return np.random.default_rng(seq)

    # -- bootstrap ----------------------------------------------------------

    def bootstrap(self, etiqueta: str = "") -> np.random.Generator:
        """Gerador da reamostragem. A etiqueta separa usos distintos sem
        precisar de outra raiz."""
        seq = np.random.SeedSequence(
            self.raiz,
            spawn_key=(_chave_estavel("bootstrap"), _chave_estavel(etiqueta)),
        )
        return np.random.default_rng(seq)


def disjuntos(a: list[int], b: list[int]) -> bool:
    """Verificação explícita, para virar asserção no harness em vez de fé."""
    return not (set(a) & set(b))
