# SPEC-06/07 — Figuras de mérito: WNN treinada × MADDPG

Custo do modelo (T1 memória, T3 operações) da rede sem peso **realmente treinada**
na Etapa 1 (célula Na=1, Ns=12), contra o ator MADDPG (256×256) das células
publicadas do cenário 003 de Lamenza e contra o MLP pareado da mesma célula (o
controle de método da SPEC-05). Números do harness de custo da SPEC-03
(`uavwnn.custo`), não estimados.

## Ressalva de cobertura de cenário

**Não há célula em comum** entre o WNN treinado (Na=1) e os checkpoints MADDPG
(Na∈{2,4,8}). Custo é propriedade do modelo, pouco sensível ao cenário, então a
comparação vale como **ordem de grandeza** com a diferença de célula anotada; o
MLP pareado é medido na **mesma** célula do WNN. A comparação de **desempenho por
cenário** WNN×MADDPG exigiria treinar o WNN nas células do artigo (escalada
T-06.5) e fica adiada.

## A tabela

| modelo | mult. fp (MAC) / decisão | T1 memória |
|---|---|---|
| **WNN dissertação (1/12)** | **4** | 975 KB (ordenado) – 1845 KB (hash) |
| WNN anbubus (1/12, abort) | 4 | 1725 KB (hash) |
| MLP pareado (1/12, 25-32-32-2) | 1 892 | 7,6 KB |
| MADDPG (Na=2, D=27) | 72 964 | 287 KB |
| MADDPG (Na=4, D=31) | 73 988 | 291 KB |
| MADDPG (Na=8, D=39) | 76 036 | 299 KB |

## O achado: trade-off de dois eixos

Não há vencedor único — há um **trade-off memória ↔ computação**, e é essa a
figura de mérito honesta.

- **T3 (computação): a WNN elimina os MACs.** 4 multiplicações fp por decisão
  (só a amostragem da política) contra ~74 000 do MADDPG — **0,005 %**. O resto do
  custo da WNN é *lookup* + soma inteira (201 lookups, 804 somas int), as
  operações que um microcontrolador **sem FPU** executa barato. É o lado de custo
  que sustenta a proposta (RNF-05, "sem MACs").
- **T1 (memória): a WNN paga.** As LUTs treinadas ocupam **975 KB** (variante mais
  compacta, `ordenado`) a **1,8 MB** (`hash`) — contra **287–299 KB** dos pesos
  densos do MADDPG. Mesmo no melhor caso, **3,3× o MADDPG**. A rede sem peso troca
  memória por computação: 70 950 endereços distintos ocupados após o treino.

O filtro de Kalman da fusão (SPEC-07) acrescenta **48 flops** por decisão — ~0,07 %
do ator MADDPG e desprezível ao lado de qualquer dos dois; não muda o quadro.

## Leitura para o alvo embarcado

O perfil da WNN é o de um alvo **com pouca capacidade de multiplicação e memória
sobrando** (MCU sem FPU, com RAM/flash externa): troca ~74 k MACs por lookups e
somas inteiras, ao custo de alguns MB de tabela. O MADDPG é o perfil oposto —
compacto em memória, caro em MAC. A escolha deixa de ser "quem é menor" e passa a
ser "qual recurso é o gargalo no hardware alvo", que é a figura de mérito certa a
reportar.

## Adiado

- Desempenho por cenário WNN×MADDPG — exige a escalada de treino do WNN para
  Na∈{2,4,8} (a observação cresce com Na, `N=⌈B/n⌉` cresce; a config pode precisar
  de reconfirmação, T-06.5).
- T2 (latência) por estágio com o filtro embutido.

## Artefatos e reprodução

- `reports/spec06-merito.json` — números por modelo.
- `reports/img/spec06-merito/` — `t3_operacoes.png`, `t1_memoria.png`,
  `mapa_merito.png` (o mapa de mérito memória×computação).

```bash
uv run python scripts/spec06_figuras_merito.py
```
