# Modelos treinados (checkpoints `.npz`)

Pesos da rede sem peso da dissertação (`ntupla_dissertacao`) treinada neste
projeto. Cada pasta é o checkpoint **`final`** — o estado ao fim do treino.

> **Mudou em 25/08, e a razão importa.** Antes as pastas traziam o `melhor`
> (melhor validação). Duas medições derrubaram essa escolha: (1) o `melhor`
> congela cedo, porque o banco de validação satura em 100 % de cobertura e o
> critério é `>` estrito — o `final` ganha em SR em TODAS as células medidas;
> (2) as pastas estavam MISTAS — as do `T = 32` vieram com `melhor` e as do
> `T = 16` com `final`, então a figura T1 comparava redes com orçamentos de
> treino diferentes (2,5 M contra 6,75 M passos). O `T1` do `T = 32` subiu de
> 975 KB para 1073 KB ao corrigir isso, e é o número certo.
de uma célula. `runs/` é gitignored (GBs de artefatos por corrida); estes são os
modelos versionados de propósito.

> **Provisórios/parciais.** Config eleita sobre a **parcial** do grid (não o
> vencedor definitivo), **1 semente**, **PPO não testado**. Ver
> [`../reports/modelo-e-hiperparametros.md`](../reports/modelo-e-hiperparametros.md).

## Configuração (a mesma em todos)

`ntupla_dissertacao` · passo **8e-4** · σ **0,35** · T **32** · n **12** · φ **0,85**
· m **8** · γ **0,99** · d_saída **3** · réplica **1b** · c_max **0,5**.
Multiagente: **parameter sharing** + **recompensa/Na**. É o ponto #1 de
`PONTOS_LONGA` (melhor da parcial do grid).

## Modelos

| pasta | célula | N (RAMs) | desempenho (test, 100 sementes) | estado |
|---|---|---|---|---|
| `ntupla_dissertacao-a1s12` | Na=1, Ns=12 | 67 | SR 91% · cob 95,4% | completo |
| `ntupla_dissertacao-a2s24` | Na=2, Ns=24 | 72 | SR 91% · cob 96,7% | completo |
| `ntupla_dissertacao-a4s24` | Na=4, Ns=24 | 83 | **SR 100% · cob 100%** | completo |

Cada pasta: `ntupla_dissertacao-ator.npz` (a política), `-critico.npz` (auxiliar
de treino), `termometro.npz` (limiares — **estado treinado**, viaja com o modelo)
e `familia.txt`.

> **Na=8 não fechou** (limite provável do crítico descentralizado). O parcial fica
> em [`em-treino/`](em-treino/README.md) com o comando `--retomar` para **concluir
> noutra máquina** — retomando de onde parou ou recomeçando do zero.

## Como carregar

```python
import sys; sys.path.insert(0, "scripts")
from spec06_campanha_wnn import montar
from uavwnn.refenv import LAMENZA_003
cfg = LAMENZA_003(4, 24, phase="eval")
politica, ap = montar("ntupla_dissertacao", cfg, "models/ntupla_dissertacao-a4s24")
# politica(obs) -> ação; aplicar por drone (política compartilhada)
```

Ressalva de método: comparação WNN×MADDPG é cross-método (o MADDPG é CTDE, crítico
centralizado; este AC é descentralizado on-policy). Ver o relatório geral.
