"""As duas recompensas de Lamenza, calculadas **fora** do ambiente.

O ambiente de referência implementa `punish` (eq. 5-6) e `reward` (o incremento),
mas **não** a eq. 7-2 do experimento exploratório do §7.1 — e a T-06.1 a exige.

CALCULAR AQUI NÃO É MODIFICAR O PROBLEMA

A recompensa usada para **aprender** é escolha do agente; o ambiente expõe o estado
(`env.sensors_collected`). Recomputá-la de fora é o que permite reproduzir o §7.1
**sem editar o ambiente**, que é a regra do projeto. E é o que a eq. 7-2 verbatim
significa: o §7.2 usa a do ambiente, o §7.1 usa outra.

    eq. 5-6   R_t = Δvisited·10                    se Δvisited > 0
                    (Σ visited − Ns)/Ns            caso contrário
    eq. 7-2   R_t = (Σ visited)/Ns                 sempre

A DIFERENÇA ENTRE ELAS NÃO É UMA CONSTANTE, E ISSO IMPORTA

A eq. 7-2 é **não negativa**, então com episódio de duração variável um episódio mais
longo acumula **mais** retorno: ela não penaliza demora. A eq. 5-6 é ≤ 0 fora das
coletas, então penaliza. São objetivos materialmente diferentes, e é exatamente o que
o texto do §7.1 descreve — *"allows agents plenty of time... reduces the urgency to
complete the scenario as quickly as possible"*.

Confundi-las faria a réplica do §7.1 medir o problema do §7.2 com horizonte generoso,
o que é outra coisa.
"""

from __future__ import annotations


def eq_5_6(colhidos: int, colhidos_antes: int, n_sensores: int) -> float:
    """A recompensa do §7.2, idêntica ao `reward='punish'` do ambiente.

    Inclui o **bônus de +10 por sensor novo**, que o PRD §5.2 registra e que uma
    leitura apressada da equação omitiria — o `else` sozinho seria só a fração
    restante.
    """
    d = colhidos - colhidos_antes
    if d > 0:
        return float(d * 10)
    return -float(n_sensores - colhidos) / n_sensores


def eq_7_2(colhidos: int, colhidos_antes: int, n_sensores: int) -> float:
    """A recompensa do §7.1: fração visitada acumulada, **sempre** aplicada.

    `colhidos_antes` entra na assinatura por simetria com `eq_5_6`, e não é usada —
    a 7-2 depende só do estado corrente. Manter a assinatura igual é o que permite
    trocar a recompensa por chave sem o chamador saber qual é.
    """
    return float(colhidos) / n_sensores


RECOMPENSAS = {"eq_5_6": eq_5_6, "eq_7_2": eq_7_2}
"""Selecionável por nome. `eq_5_6` é a do ambiente e a do §7.2; `eq_7_2` é a do
experimento exploratório do §7.1 e **só** existe aqui."""
