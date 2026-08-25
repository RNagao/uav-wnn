# Modelos de IMPLANTAÇÃO

O que voa. Contém **só o ator** — a política — em `uint16`/`float32`.

Os checkpoints de treino ficam em [`../T32/`](../T32/) e [`../T16/`](../T16/), e é
de lá que se retoma um treinamento. Estes aqui **não servem para retomar**: sem o
crítico não há como continuar o Algoritmo 4.4.

## O que foi tirado, e por quê

| | no treino | no deploy | motivo |
|---|---|---|---|
| crítico | presente | **removido** | estima "quão boa é esta situação"; nunca escolhe ação. Só existe para treinar |
| endereços | `uint64` | **`uint16`** | o endereço tem `tamanho_tupla` bits (12 → 4096 valores) |
| valores | `float64` | **`float32`** | é a precisão que o modelo de custo da SPEC-03 já assume no T1 |

## Redução medida

| modelo | treino | deploy | | dif. máx. na ação |
|---|---|---|---|---|
| T16/a1s12 | 1333 KB | **472 KB** | 2,8× | 2,70e-08 |
| T16/a2s24 | 2620 KB | **918 KB** | 2,9× | 3,56e-08 |
| T16/a4s24 | 2697 KB | **943 KB** | 2,9× | 4,54e-08 |
| T16/a8s24 | 3082 KB | **1079 KB** | 2,9× | 4,88e-07 |
| T32/a1s12 | 2569 KB | **913 KB** | 2,8× | 2,01e-08 |
| T32/a2s24 | 4722 KB | **1661 KB** | 2,8× | 1,12e-07 |
| T32/a4s24 | 5406 KB | **1896 KB** | 2,9× | 2,64e-08 |

## A verificação

O `float32` **muda as predições nos últimos bits**, então o export não é
publicado sem prova de que a política não muda de comportamento. Duas checagens:

1. **Por ação** — `spec06_exportar_deploy.py` compara a ação dos dois formatos
   sobre observações REAIS do ambiente (não vetores aleatórios, que tocariam
   endereços vazios e mediriam o caso trivial). Recusa exportar acima de `1e-6`.
2. **Por comportamento** — avaliação no banco de teste, 100 sementes:

```
T16 a8s24 (a maior diferença por ação):
  float64  SR 88,0 %  CT 97,545 s
  float32  SR 88,0 %  CT 97,545 s   → idênticos

T32 a2s24:
  float64  SR 91,0 %  CT 80,860 s
  float32  SR 91,0 %  CT 80,860 s   → idênticos
```

## Poda (padrão pós-treino desde 25/08)

Os treinadores (`spec06_treino_multiagente.py`, `spec06_busca.py` no modo
`longa`) gravam `checkpoints/deploy` **podado a 70 %** e verificado no banco de
VAL, com fallback automático para sem-poda se o SR cair mais de 2 pp. O
checkpoint de treino fica intacto — de rede podada não se retoma treino.

Estado destes artefatos (verificação com 200 sementes de teste):

| modelo | deploy | poda | SR cheio → podado |
|---|---|---|---|
| T16/a1s12 | **144 KB** | 70 % | 97,5 % → 96,0 % |
| T16/a2s24 | **276 KB** | 70 % | 88,0 % → 93,0 % |
| T16/a4s24 | 943 KB | **sem poda** (caiu 4,0 pp) | 95,5 % → 91,5 % |
| T16/a8s24 | **325 KB** | 70 % | 87,5 % → 90,5 % |
| T32/a1s12 | **276 KB** | 70 % | 80,5 % → 83,0 % |
| T32/a2s24 | **500 KB** | 70 % | 94,0 % → 93,0 % |
| T32/a4s24 | 1896 KB | **sem poda** (caiu 3,5 pp) | 98,0 % → 94,5 % |

A fração tolerável é POR CÉLULA — as duas `a4s24` perdem mais de 2 pp a 70 % e
ficam sem poda. Curioso e ainda sem explicação: três células MELHORARAM com a
poda (+3 a +5 pp); com 200 sementes o ruído é ±2,3 pp, então o efeito no
`T16/a2s24` (+5 pp) parece real — hipótese: a cauda removida continha resíduos
de exploração que atrapalhavam, o mesmo mecanismo do dithering.

## Reproduzir

```bash
uv run python scripts/spec06_exportar_deploy.py --todos --poda 0.7
```

## O que isto NÃO resolve

Mesmo assim a WNN gasta mais memória que o ator do MADDPG (287–299 KB), e a razão
é estrutural: o ator de 2 drones guarda **129 664 endereços × 3 saídas = 388 992
números**, contra **73 474 pesos** do MLP — 5,3× mais valores. Codificação e
variante de armazenamento tiram o que é artefato; o resto é o modelo.
