# Custo comparado — memória, computação e tempo de inferência

Os três modelos medidos **na mesma execução e com o mesmo protocolo**. Juntar
números de relatórios diferentes foi como a razão de tamanho acabou comparando
checkpoints com orçamentos de treino distintos.

Reprodução: `uv run --group baseline python scripts/spec03_custo_comparado.py`

---

## Os três eixos não concordam — e é esse o resultado

| eixo | quem ganha | por quanto |
|---|---|---|
| **T2 ciclos** (Cortex-M0) | **WNN** | **1 913×** |
| **T2 latência medida** (após via rápida) | **WNN** | **2,3×** (ver §4) |

*A antiga linha **T3 (multiplicações fp)** foi retirada: contar multiplicação
favorece a WNN por construção — ela é* weightless *e quase não multiplica. O custo
de computação fica nos **ciclos** (§3, que somam todas as operações) e na **latência
empírica medida** (§4). A tabela de operações do §2 permanece só como perfil.*

---

## 1. T1 — memória: medida à parte, como pico de RAM

A comparação de **memória persistente** (bytes de artefato por fórmula) que esta
seção trazia foi **retirada do projeto**. A figura de memória é o **pico de RSS
medido** (`VmHWM`) do processo que carrega o modelo e decide — um processo por
modelo, porque o pico é monotônico e o laço intercalado deste script não consegue
medi-lo. Ver `scripts/spec03_latencia_memoria.py` e `reports/spec06-merito.md`.

---

## 2. T3 — operações por decisão

| célula | modelo | mult. fp | soma fp | consultas | soma int |
|---|---|---|---|---|---|
| 1d/12s | wnn-T16 | **4** | 104 | 102 | 408 |
| 1d/12s | wnn-T32 | **4** | 203 | 201 | 804 |
| 2d/24s | maddpg | **72 964** | 73 476 | 0 | 0 |
| 2d/24s | wnn-T16 | **4** | 110 | 108 | 432 |
| 2d/24s | wnn-T32 | **4** | 218 | 216 | 864 |

**Esta tabela é o PERFIL de operações — não é mais a figura de mérito de custo.**
A contagem de multiplicações fp foi retirada como veredito: a WNN é *weightless* e
quase não multiplica, então "4 contra 72 964" favorece o método por construção e
não captura o custo real (consultas à LUT, acesso a memória). As 4 mult. da WNN são
do amostrador Box-Muller (comum aos dois; `incluir_amostragem=False` dá **zero** no
aproximador), e o MADDPG paga as mesmas 4.

O que a tabela mostra, e continua válido: a WNN troca multiplicação por **consulta
e soma inteira** — 108 acessos à LUT e 432 somas de endereço, contra 73 mil MACs.
É **outro perfil de custo**, não um custo menor em tudo. O veredito de computação
vem dos **ciclos** (§3, que somam todas essas operações) e da **latência medida**
(§4), não desta contagem de multiplicação.

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
software. É onde a estrutura sem multiplicação fp mais pesa nos ciclos — e é o alvo de quem embarca num
drone pequeno.

---

## 4. T2 — wall-clock, e por que ele não decide

**Os números publicados na 1ª versão desta tabela comparavam coisas diferentes**,
e a explicação que dei para eles estava errada. As duas correções, medidas:

**A medida era assimétrica.** `_latencia_torch` cronometrava `rede(x)` — o
forward puro sobre um tensor já pronto. `_latencia_wnn` cronometrava
`politica(x)` — decodificar, termometrizar, consultar, amostrar, projetar. Uma
media uma multiplicação de matrizes; a outra, uma decisão inteira. Corrigido no
código; medindo os dois de observação bruta a ação:

| célula | modelo | decisão completa |
|---|---|---|
| 2d/24s | maddpg | **50,6 µs** |
| 2d/24s | wnn-T16 | 91,5 µs — **1,8×**, não 2,6× |

**E não é o PyTorch.** Eu havia escrito que o MADDPG ganha por ter "kernels BLAS
compilados" contra NumPy interpretado. É falso, e a medida é direta: a **mesma
rede** do MADDPG, reimplementada em NumPy puro, roda em **18,2 µs** contra os
**36,7 µs** do `forward` em torch. Para uma rede 27→256→256→2, o despacho do
PyTorch custa o dobro do trabalho aritmético — torch é o arcabouço *mais lento*
aqui, não o mais rápido.

**A causa real é o nosso próprio decodificador.** Perfilando `politica(x)` da
WNN com `cProfile`, 3 000 chamadas:

| etapa | fração |
|---|---|
| `Observacao.decodificar` | **52 %** |
| consulta às LUTs (`LUTPlana.prever`) | 22 % |
| `para_acao` (projeção) | 14 % |

`decodificar` faz duas chamadas a `np.isclose` sobre matrizes `(n_a,2)` e
`(n_s,2)` para achar o preenchimento, mascara com `NaN` e monta uma dataclass —
27 mil reduções de ufunc e 12 mil `.all()` para 3 000 decisões. **O MADDPG não
tem camada equivalente**: alimenta o vetor de 27 floats direto na multiplicação.

Ou seja: a parte que implementa a alegação da tese — a consulta sem multiplicação
— é 22 % do tempo. Os outros 78 % são o cano em volta, e o cano da WNN é mais
caro que o do baseline por decisão nossa de implementação, não por propriedade do
método. Comparando só os núcleos, termômetro + `prever` dá **19,3 µs** contra os
**18,2 µs** da matriz em NumPy — empate, com quatro ordens de grandeza menos
multiplicações.

### Uma otimização feita a partir disto

`np.isclose` era 62 % do `decodificar`, e o trabalho dele ali é reconhecer um
sentinela `-1.0`. Substituído por tolerância explícita `|v − (−1)| < 1e-12`:

| forma | µs | |
|---|---|---|
| `np.isclose` | 38,91 | maquinaria de `rtol·\|b\| + atol` com NaN/inf |
| `==` | 15,69 | 2,48×, mas exige o sentinela bit-exato |
| **tolerância** | **16,70** | **2,33×**, e não depende disso |

O `==` é 1,01 µs mais rápido e foi descartado de propósito: ele transformaria uma
propriedade do AMBIENTE (`fill(-1)` sem aritmética depois) em requisito silencioso
do nosso código. Se um dia alguém normalizar o vetor inteiro, `-1` decodifica para
a posição válida `3S` e o slot vazio vira **sensor fantasma** — sem erro, só
resultado pior.

Verificado em 5 242 ações contra a implementação original, atravessando 30 363
slots de preenchimento, em limpo e sob Kalman com σ=10: diferença máxima na ação
de **0,00e+00**.

### E depois da via rápida, o wall-clock virou

O diagnóstico acima apontava o remédio: a decisão da WNN gastava ~30 despachos
NumPy, e nesse tamanho de vetor o despacho domina a conta. A via rápida funde o
pipeline — a `Observacao` sai do caminho (o ator só lê `.vetor`), o termômetro,
o endereçamento e a busca viram uma closure sem materializar os bits, e a
projeção da ação perde as dataclasses. Bit a bit igual, pinado em
`tests/test_caminho_rapido.py`. Pareado, no mesmo laço:

| decisão completa | mediana | p99 |
|---|---|---|
| **WNN T=16 via rápida** | **33,9 µs** | 78,6 µs |
| WNN T=32 via rápida | 25,2 µs ¹ | 62,1 µs |
| WNN T=16 via lenta | 70,7 µs | 153,4 µs |
| MADDPG (torch) | 76,4 µs | 165,2 µs |

**A WNN passou a ganhar também no relógio de parede: 2,26×.** Nenhuma lib nova —
a lição é que trocar de arcabouço não era o remédio (o torch perde do NumPy puro
aqui); reduzir despachos era.

¹ A inversão T32 < T16 no laço intercalado NÃO é ruído — é vizinhança de cache,
e foi isolada trocando a ordem: quem roda logo depois do forward do torch paga os
caches que as matrizes 256×256 dele despejam (~2× no seguinte). Medido cada um
SOZINHO em bloco — que é como um modelo roda num drone — a ordem arquitetural
reaparece: T16 17,7 µs · T32 42,4 µs · MADDPG 98,1 µs (sob a mesma carga de
fundo). As razões honestas são T16 5,5× e T32 2,3× mais rápidas que o MADDPG.

O modelo de ciclos continua sendo o argumento para o embarcado: 31–703× num
Cortex-M0 sem FPU é o que uma implementação em C entregaria.

---

## 5. Leitura

**A WNN troca eixos.** Ganha em computação — em **ciclos** (31–703× conforme o
hardware, §3) e na **latência medida** após a via rápida (2,3×, §4). O relógio
dependeu de quem implementou: a versão NumPy inicial parecia perder, mas o gargalo
era o nosso decodificador, não o método. A memória é medida à parte, como pico de
RAM do processo (§1).

**O alvo importa — e depois da via rápida, menos ainda.** Em x86 a WNN faz 33,9 µs
contra 76,4 µs. Num Cortex-M0 sem FPU a conta é 13 mil ciclos contra 9,2 milhões —
a WNN passa a ser a única das duas que roda.

## Ressalvas

- Os modelos WNN medidos são os de `models/deploy/` — podados a 70 % e em
  `float32`/`uint16`. É o artefato que a campanha avaliou.
- Os `latencia_mediana_us` das 33 corridas de campanha já gravadas usam a medida
  ANTIGA e assimétrica do MADDPG. Só são comparáveis entre corridas do mesmo
  modelo, não entre modelos. As próximas campanhas usam a medida corrigida.
