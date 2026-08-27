# Latência e memória — 2d/24s, medidos

Gerado por `scripts/spec03_latencia_memoria.py` em 2026-08-27 · x86_64 · mesma observação para os três · um processo por modelo · menor mediana de 8×1000.

## Tempo de inferência (decisão completa: obs → ação)

| modelo | mediana | p99 |
| --- | --- | --- |
| wnn-T16 | 15.7 µs | 33.0 µs |
| wnn-T32 | 18.3 µs | 40.3 µs |
| maddpg | 45.6 µs | 88.8 µs |

- wnn-T16 é **2.9×** mais rápida que o MADDPG

- wnn-T32 é **2.5×** mais rápida que o MADDPG

## Memória, nas quatro leituras (KB)

| modelo | T1 artefato | residente (RAM) | disco | ator + crítico |
| --- | --- | --- | --- | --- |
| wnn-T16 | 310 | 362 | 289 | 1,385 |
| wnn-T32 | 572 | 670 | 533 | 2,561 |
| maddpg † | 287 | 289 | 1,438 | 1,433 |

- **T1 artefato**: chave `uint16` + valores `float32` — a definição da SPEC-03, comparável com as demais tabelas do projeto;
- **residente**: o que ocupa RAM executando — na WNN, a `LUTPlana` (chave global `int32`, mapeamento, limiares) + buffers de trabalho; no MADDPG, pesos + entrada e ativações (2×256) que o forward usa. Sem os buffers do MADDPG a comparação seria assimétrica: o T1 da WNN embute os dela;
- **disco**: o arquivo como está; † o `.cleanrl_model` traz ator E crítico no mesmo arquivo, a barra dele já é o total;
- **ator + crítico**: o custo de treino — o crítico não voa.

## Arcabouço em RAM (antes de qualquer peso)

| modelo | imports |
| --- | --- |
| wnn-T16 | 17 MB |
| wnn-T32 | 17 MB |
| maddpg | 481 MB |

![tempos](spec03-latencia-a2s24.png)
![memória](spec03-memoria-a2s24.png)
