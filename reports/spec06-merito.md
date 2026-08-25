# WNN treinada × MADDPG — tamanho, inferência e as figuras de Lamenza

Comparação de custo entre a rede sem peso (WNN) **realmente treinada** na Etapa 1
(célula Na=1, Ns=12) e o ator MADDPG (256×256) das células publicadas do cenário
003 de Lamenza. Números do harness de custo da SPEC-03 (`uavwnn.custo`) e da
medição da SPEC-02, não estimados.

**As figuras de mérito de Lamenza são duas** (PRD §5.5): **tempo de conclusão** e
**taxa de sucesso**. O projeto acrescenta três de custo — **T1 tamanho**, **T2
tempo de inferência**, **T3 operações** —, que são o núcleo de H1. Este relatório
cobre as três de custo aqui e situa as duas de Lamenza no fim.

## Ressalva de cobertura (célula)

**Não há célula em comum** entre o WNN treinado (Na=1) e os checkpoints MADDPG
(Na∈{2,4,8}) — Lamenza não treinou Na=1. Tamanho, inferência e operações são
propriedades do modelo, pouco sensíveis ao cenário: a comparação vale como ordem
de grandeza, com a célula anotada. **Tempo de conclusão e taxa de sucesso**, ao
contrário, dependem do cenário — e essas **não** têm par (ver §4).

## T1 — Tamanho do modelo

| modelo | tamanho (ator, fp32) |
|---|---|
| **WNN T=32 (1/12), RL real** | **975 KB** (ordenado) — 70 950 entradas · 1 845 KB na variante `hash` |
| **WNN T=16 (1/12), RL real** | **502 KB** (ordenado) — 36 547 entradas · 942 KB na variante `hash` |
| WNN clone sintético (SPEC-03) | 133,9 KB |
| MADDPG (Na=2/4/8) | 287 / 291 / 299 KB |
| MLP pareado (1/12, 25-32-32-2) | 7,6 KB |

**A WNN perde em tamanho — e por mais do que a SPEC-03 estimava.** O clone
sintético (133,9 KB) veio de uma tarefa pequena; a política de RL de verdade
enche **70 950 endereços** e ocupa **975 KB a 1,8 MB** conforme a variante de
armazenamento — **3,3× a 6,3×** o ator MADDPG. H1 cai no eixo de tamanho, o que já
era o veredito da SPEC-03, agora reforçado pelo modelo treinado.

## T2 — Tempo de inferência

**Por ciclos (independente de hardware, RNF-12) — a figura defensável:**

| hardware | WNN | MADDPG (256×256) | ganho WNN |
|---|---|---|---|
| x86-64 (FPU+SIMD) | 4 465 | 149 200 | **33×** |
| Cortex-M4F (FPU) | 4 777 | 446 476 | **93×** |
| **Cortex-M0 (sem FPU)** | 19 809 | 9 281 172 | **469×** |

**A WNN ganha em inferência, e o ganho explode sem FPU.** Ela troca ~74 000 MACs
por lookups e somas inteiras; num MCU sem unidade de ponto flutuante, cada MAC do
MADDPG custa ~70 ciclos e a diferença vira **três ordens de grandeza**. É o lado
de inferência de H1.

**Por wall-clock (protótipos) — inverte, e é por quê:**

| | mediana | p99 |
|---|---|---|
| WNN (NumPy, nosso protótipo) | 173 µs | 322 µs |
| MADDPG (PyTorch, SPEC-02) | 35 µs | 115 µs |

No relógio, o protótipo da WNN é **mais lento** — mas isso é **artefato de
implementação, não do algoritmo** (RNF-11): o NumPy paga overhead de Python por
lookup, o PyTorch despacha os MACs para BLAS vetorizado. A figura honesta é a de
ciclos; o wall-clock de dois arcabouços diferentes não é comparável, e uma
implementação compilada da WNN realizaria a vantagem de ciclos. Vale lembrar
(PRD §5.5): com intervalo de decisão de **2 s**, os dois cabem com >4 ordens de
grandeza de folga — "cumprir o prazo" não distingue nada; a estrutura de operação
sim.

## T3 — Operações por decisão

| modelo | mult. fp (MAC) | resto |
|---|---|---|
| **WNN dissertação** | **4** (só a amostragem) | 201 lookups + 804 somas int + 203 somas fp |
| MLP pareado (1/12) | 1 892 | — |
| MADDPG (Na=2/4/8) | 72 964 / 73 988 / 76 036 | — |

**A WNN elimina os MACs** — 4 contra ~74 000 (**0,005 %**). É a alegação central
da RNF-05, agora medida no modelo treinado. O filtro de Kalman da fusão (SPEC-07)
soma 48 flops/decisão — desprezível ao lado de qualquer dos dois.

## 4. As figuras de Lamenza (tempo de conclusão e taxa de sucesso)

Estas dependem do cenário e **não têm par** — o WNN treinado é Na=1/Ns=12, os
números publicados de Lamenza são Na∈{2,4,8}. O que temos:

| | tempo de conclusão | taxa de sucesso |
|---|---|---|
| **WNN dissertação (1/12)** | ~40 s (sucesso) · CT censurado ~130 s | 81 % (500 sem.) / 91 % (100 sem.) |
| MADDPG publicado (Na=2 / 4 / 8) | 29,6 / 13,8 / 9,7 s | ~100 % |

Não são comparáveis: mais drones concluem mais rápido (mais cobertura por passo),
então CT cai com Na — o Na=1 do WNN é a tarefa **mais difícil** da grade. A
comparação head-to-head nas figuras de Lamenza exige treinar o WNN nas células do
artigo (**escalada T-06.5**), que fica adiada (treino pesado; a observação cresce
com Na).

## Veredito de custo (H1)

| eixo | vencedor | margem |
|---|---|---|
| **Tamanho (T1)** | MADDPG | WNN 3,3–6,3× maior |
| **Inferência (T2, ciclos)** | **WNN** | 33× (x86) a **469× (Cortex-M0)** |
| **Operações (T3)** | **WNN** | 4 vs ~74 000 MACs (sem FPU) |

O perfil da WNN é o de um alvo **sem FPU e com memória sobrando** (MCU + flash
externa): paga em tabela, ganha em aritmética. O MADDPG é o oposto. A figura de
mérito certa não é "quem é menor", e sim **qual recurso é o gargalo no hardware
alvo** — e é onde a rede sem peso tem uma vantagem estrutural real.

## Artefatos e reprodução

- `reports/spec06-merito.json` — números por modelo (T1, T2 ciclos + wall-clock, T3).
- `reports/img/spec06-merito/` — `t1_memoria.png`, `t2_ciclos.png`,
  `t3_operacoes.png`, `mapa_merito.png`.

```bash
uv run python scripts/spec06_figuras_merito.py
```

**Adiado:** desempenho por cenário (tempo de conclusão / SR) WNN×MADDPG — exige a
escalada T-06.5; T2 wall-clock com os dois lados no mesmo arcabouço (RNF-11).
