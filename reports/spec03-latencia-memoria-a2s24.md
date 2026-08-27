# Latência e memória — 2d/24s, medidos

Gerado por `scripts/spec03_latencia_memoria.py` em 2026-08-27 · x86_64 · mesma observação para os três · um processo por modelo · menor mediana de 8×1000.

## Tempo de inferência (decisão completa: obs → ação)

| modelo | mediana | p99 |
| --- | --- | --- |
| wnn-T16 | 16.9 µs | 81.8 µs |
| wnn-T32 | 20.0 µs | 59.8 µs |
| maddpg | 45.4 µs | 107.3 µs |

- wnn-T16 é **2.7×** mais rápida que o MADDPG

- wnn-T32 é **2.3×** mais rápida que o MADDPG

## Memória, nas quatro leituras (KB)

| modelo | T1 artefato | residente (RAM) | disco | ator + crítico |
| --- | --- | --- | --- | --- |
| wnn-T16 | 296 | 346 | 276 | 2,027 |
| wnn-T32 | 537 | 628 | 500 | 3,675 |
| maddpg † | 287 | 287 | 1,438 | 1,433 |

- **T1 artefato**: chave `uint16` + valores `float32` — a definição da SPEC-03, comparável com as demais tabelas do projeto;
- **residente**: o que a `LUTPlana` ocupa executando — chave global `int32` (não cabe em 16 bits), mapeamento das tuplas e limiares;
- **disco**: o arquivo como está; † o `.cleanrl_model` traz ator E crítico no mesmo arquivo, a barra dele já é o total;
- **ator + crítico**: o custo de treino — o crítico não voa.

## Arcabouço em RAM (antes de qualquer peso)

| modelo | imports |
| --- | --- |
| wnn-T16 | 17 MB |
| wnn-T32 | 17 MB |
| maddpg | 479 MB |

![tempos](spec03-latencia-a2s24.png)
![memória](spec03-memoria-a2s24.png)
