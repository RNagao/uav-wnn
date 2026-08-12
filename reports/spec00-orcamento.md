# SPEC-00 — Orçamento de simulação no ambiente de referência (P2)

Máquina: WSL2, um núcleo. Ambiente de referência (submodule, tag `dissertation`), guloso do `heuristics.py`, `gradysim==0.6.11`, 20 episódios por célula.

O `update_rate` **não é mais decisão nossa**: o `environment.py` de referência o fixa em `algorithm_iteration_interval / 2`. Esta medição responde apenas *quanto custa*.

| Na | Ns | passos/ep | s/ep | sucesso | t_conc (s) | h/núcleo por 50 M passos |
|---|---|---|---|---|---|---|
| 1 | 12 | 68.5 | 0.0174 | 1.00 | 34.3 | 3.5 |
| 1 | 24 | 91.3 | 0.0254 | 1.00 | 45.6 | 3.9 |
| 1 | 36 | 100.0 | 0.0374 | 1.00 | 50.0 | 5.2 |
| 2 | 12 | 61.5 | 0.0185 | 1.00 | 30.7 | 2.1 |
| 2 | 24 | 83.5 | 0.0340 | 1.00 | 41.8 | 2.8 |
| 2 | 36 | 90.1 | 0.0471 | 1.00 | 45.0 | 3.6 |
| 4 | 12 | 57.1 | 0.0295 | 1.00 | 28.6 | 1.8 |
| 4 | 24 | 86.5 | 0.0582 | 1.00 | 43.2 | 2.3 |
| 4 | 36 | 84.8 | 0.0724 | 1.00 | 42.4 | 3.0 |
| 8 | 12 | 59.4 | 0.0654 | 1.00 | 29.7 | 1.9 |
| 8 | 24 | 84.9 | 0.1105 | 1.00 | 42.5 | 2.3 |
| 8 | 36 | 91.4 | 0.1451 | 1.00 | 45.7 | 2.8 |

**P2 fechada: 1.8 a 5.2 h por núcleo** para o orçamento de 50 M passos de Lamenza. O PRD §6.1 estimava 8–30 h — o custo real é **menor** que a estimativa, com folga.

Duas razões para ser mais barato que a medição da versão anterior desta spec (3,0–21,8 h):

1. Na fase de avaliação o episódio termina na conclusão (`end_when_all_collected=True`), em vez de rodar mais 30 s até a poda por parada.
2. O `update_rate` do ambiente de referência (`interval/2`) é bem mais grosseiro que o `interval/16` que havíamos adotado quando o integrador ainda era escolha nossa.

## Correção ao PRD

O PRD §8 e a SPEC-00 supunham que a célula cara seria `Na = 8`. Medido: a célula cara em horas de treino é **`Na = 1, Ns = 36`** (5.2 h), porque o contador de passos é `passos × Na` — menos agentes significam menos passos por segundo simulado. `Na = 8, Ns = 36` custa 2.8 h.

Consequência: **a Etapa 1, que roda `Na = 1`, é a mais cara em horas de treino de todo o projeto** — o oposto do que a linha de corte do PRD supõe ao listar `Na = 8` entre os primeiros cortes.

## Critério de saída 5 da SPEC-00

Um episódio ponta a ponta com o guloso de referência em **todas as 12 células** da grade `Na ∈ {1,2,4,8} × Ns ∈ {12,24,36}`: satisfeito, com taxa de sucesso 1,00 em todas.
