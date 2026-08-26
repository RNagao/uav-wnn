# Custo comparado — memória, computação e tempo de inferência

Os três modelos medidos **na mesma execução e com o mesmo protocolo**. Juntar
números de relatórios diferentes foi como a razão de tamanho acabou comparando
checkpoints com orçamentos de treino distintos.

Reprodução: `uv run --group baseline python scripts/spec06_custo_comparado.py`

---

## Os três eixos não concordam — e é esse o resultado

| eixo | quem ganha | por quanto |
|---|---|---|
| **T1 memória** | MADDPG | 1,9× (287 KB contra 537 KB) |
| **T3 operações** | **WNN** | **18 241×** (4 contra 72 964 mult. fp) |
| **T2 ciclos** (Cortex-M0) | **WNN** | **1 913×** |
| T2 wall-clock | MADDPG | 2,6× — mas ver §4 |

---

## 1. T1 — memória

| célula | modelo | params | T1 (KB) | disco (KB) | RSS (MB) |
|---|---|---|---|---|---|
| 1d/12s | wnn-T16 | 10 965 | **152** | 144 | 2,2 |
| 1d/12s | wnn-T32 | 23 440 | 325 | 305 | 3,0 |
| 2d/24s | **maddpg** | 73 474 | **287** | 1 438 ¹ | 494 ² |
| 2d/24s | wnn-T16 | 21 472 | 296 | 276 | 2,2 |
| 2d/24s | wnn-T32 | 38 900 | 537 | 500 | 2,4 |

**¹ O arquivo do MADDPG traz ator E crítico.** 287 KB + 1 146 KB. A coluna T1
conta só o ator, dos dois lados; o disco não, porque o `.cleanrl_model` é uma
tupla `(ator, crítico)` e não se separa sem reescrevê-lo.

**² Os 494 MB do MADDPG são o PyTorch, não o modelo.** Medi: importar `torch`
custa 494 MB de RSS antes de qualquer peso ser carregado. É custo real de quem
implanta com PyTorch — mas é arcabouço, não rede, e comparar com os 2,2 MB da
WNN (NumPy) mediria bibliotecas.

**Na célula comum, o T=16 empata com o MADDPG** — 296 KB contra 287 KB. Chegou
lá com a poda de 70 % e o export `float32`/`uint16`; sem eles seriam ~2 MB.

---

## 2. T3 — operações por decisão

| célula | modelo | mult. fp | soma fp | consultas | soma int |
|---|---|---|---|---|---|
| 1d/12s | wnn-T16 | **4** | 104 | 102 | 408 |
| 1d/12s | wnn-T32 | **4** | 203 | 201 | 804 |
| 2d/24s | maddpg | **72 964** | 73 476 | 0 | 0 |
| 2d/24s | wnn-T16 | **4** | 110 | 108 | 432 |
| 2d/24s | wnn-T32 | **4** | 218 | 216 | 864 |

**As 4 multiplicações da WNN são do amostrador Box-Muller, não do aproximador.**
Com `incluir_amostragem=False` o aproximador dá **zero** — é a alegação central
de H1. E o MADDPG paga as mesmas 4 (72 964 = 72 960 MACs + 4), então a
comparação é entre iguais.

A WNN troca multiplicação por **consulta e soma inteira**: 108 acessos à LUT e
432 somas de endereço, contra 73 mil MACs. É outro perfil de custo, não um custo
menor em tudo.

---

## 3. T2 — ciclos, independente de hardware

| célula | modelo | x86-64 | Cortex-M4F | Cortex-M0 |
|---|---|---|---|---|
| 1d/12s | wnn-T16 | **2 382** | **2 694** | **12 578** |
| 1d/12s | wnn-T32 | 4 465 | 4 777 | 19 809 |
| 2d/24s | maddpg | 147 152 | 440 332 | **9 153 172** |
| 2d/24s | wnn-T16 | **2 516** | **2 828** | **13 024** |
| 2d/24s | wnn-T32 | 4 784 | 5 096 | 20 908 |

**A vantagem cresce à medida que o hardware encolhe:**

| | x86-64 | Cortex-M4F | Cortex-M0 |
|---|---|---|---|
| WNN T=16 é mais rápida | **58×** | **156×** | **703×** |
| WNN T=32 é mais rápida | 31× | 86× | 438× |

No **Cortex-M0 não há FPU**: cada multiplicação em ponto flutuante vira rotina de
software. É onde a alegação "sem MACs" vale mais — e é o alvo de quem embarca num
drone pequeno.

---

## 4. T2 — wall-clock, e por que ele não decide

| célula | modelo | mediana (µs) | p99 (µs) |
|---|---|---|---|
| 1d/12s | wnn-T16 | 70,2 | 118,9 |
| 2d/24s | maddpg | **33,2** | **56,1** |
| 2d/24s | wnn-T16 | 83,7 | 122,5 |
| 2d/24s | wnn-T32 | 87,6 | 134,4 |

**O wall-clock inverte o veredito dos ciclos, e a razão é implementação.** A WNN
roda em NumPy interpretado; o MADDPG, em PyTorch com kernels BLAS compilados. A
diferença de 2,6× mede arcabouços, não algoritmos.

Registro os dois porque omitir o wall-clock seria escolher a métrica conveniente
— e porque ele diz uma coisa verdadeira: **do jeito que está implementada hoje, a
WNN é mais lenta em x86.** A vantagem de 58–703× é o que uma implementação em C
entregaria, e o modelo de ciclos é a estimativa dela.

---

## 5. Leitura

**A WNN não é "mais barata"; ela troca eixos.** Ganha em computação por 4 ordens
de grandeza, perde ou empata em memória, e o wall-clock depende de quem
implementou.

**A poda mudou o quadro do T1.** Antes dela a WNN era 3,4–6,3× o MADDPG; com ela
o `T=16` chega a 296 KB contra 287 KB — **empate**. O que era a fraqueza clara do
método virou paridade.

**O alvo importa.** Em x86 com PyTorch o MADDPG é melhor nos dois eixos práticos.
Num Cortex-M0 sem FPU a conta inverte: 13 mil ciclos contra 9,2 milhões, e 296 KB
contra 287 KB de memória — a WNN passa a ser a única das duas que roda.

## Ressalvas

- O `T1` conta **só o ator** dos dois lados; o crítico do MADDPG (1 146 KB) e o
  da WNN não entram, porque não voam.
- O `RSS` do MADDPG está dominado pelo import do PyTorch (494 MB) e não deve ser
  comparado com o da WNN.
- Os modelos WNN medidos são os de `models/deploy/` — podados a 70 % e em
  `float32`/`uint16`. É o artefato que a campanha avaliou.
