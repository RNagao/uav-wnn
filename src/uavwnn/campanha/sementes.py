"""Os três papéis de semente da §4.1 — e por que separá-los importa.

| semente | governa | regra |
|---|---|---|
| `seed_eval` | cenários do banco de avaliação | **fixa e única**, idêntica para todos os métodos |
| `seed_train` | init das LUTs/pesos, ruído de exploração, cenários de treino | varia em `{0..R−1}`, **única** fonte de variância entre repetições |
| `seed_tuple` | mapeamento dos `N × n` bits das RAMs | **amarrada a `seed_train`** (D-06.1) |

**D-06.1 — o ladrilhamento é parte do modelo, não do ambiente.** `seed_tuple` varia
junto com `seed_train`. Fixá-lo condicionaria todos os resultados a um único
ladrilhamento e reportaria como propriedade do método algo que pode ser propriedade
de um sorteio. O preço é variância maior entre corridas; o benefício é que o IC
passa a cobrir a escolha das tuplas — que é exatamente a incerteza que o leitor
precisa ver.

A corrida de controle com `seed_tuple` **fixo** e `seed_train` variável roda **uma
vez**, no ponto base, para quantificar quanto da variância vem do ladrilhamento.
`PapeisDeSemente(..., ladrilhamento_fixo=True)` é essa corrida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PapeisDeSemente:
    """Deriva as três sementes de um par `(raiz_eval, seed_train)`.

    A raiz de avaliação é **constante de projeto**, não parâmetro: mudá-la troca o
    banco de teste e invalida toda comparação já registrada. Fica aqui, visível,
    em vez de num argumento de linha de comando onde alguém a mudaria sem perceber.
    """

    seed_train: int
    raiz_eval: int = 20260901
    """Fixa. Governa os bancos `test` e `val` de **todas** as células e métodos."""
    ladrilhamento_fixo: bool = False
    """`True` só na corrida de controle da D-06.1."""

    @property
    def seed_eval(self) -> int:
        return self.raiz_eval

    @property
    def seed_tuple(self) -> int:
        """Amarrada a `seed_train`, salvo na corrida de controle.

        O deslocamento constante evita que `seed_tuple == seed_train` — usar o mesmo
        inteiro para inicializar o ladrilhamento e o ruído de exploração criaria
        correlação entre duas coisas que a §4.1 quer independentes dentro da corrida.
        """
        return 7919 if self.ladrilhamento_fixo else 7919 + self.seed_train

    def rng_politica(self) -> np.random.Generator:
        """Ruído de exploração e init dos pesos."""
        return np.random.default_rng([self.seed_train, 1])

    def rng_cenarios_treino(self) -> np.random.Generator:
        """Sorteio dos cenários de TREINO — disjunto dos bancos por construção.

        Usa uma sub-sequência própria: se os cenários de treino saíssem do mesmo
        gerador do ruído, mudar o número de passos mudaria os cenários, e duas
        corridas com orçamentos diferentes deixariam de ser comparáveis.
        """
        return np.random.default_rng([self.seed_train, 2])

    def como_dict(self) -> dict:
        return {"seed_train": self.seed_train, "seed_eval": self.seed_eval,
                "seed_tuple": self.seed_tuple,
                "ladrilhamento_fixo": self.ladrilhamento_fixo}
