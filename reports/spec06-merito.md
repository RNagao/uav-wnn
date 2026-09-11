# WNN treinada × MADDPG — pico de RAM, inferência e as figuras de Lamenza

Comparação de custo entre a rede sem peso (WNN) **realmente treinada** e o ator
MADDPG (256×256), **ambos na mesma célula a2s24 (Na=2, Ns=24)** do cenário 003 de
Lamenza. Números **medidos** pelo protocolo da SPEC-03 (`scripts/spec03_latencia_memoria.py`:
um processo por modelo, mesma observação) e ciclos do harness `uavwnn.custo.operacoes`.

**As figuras de mérito de Lamenza são duas** (PRD §5.5): **tempo de conclusão** e
**taxa de sucesso**. O projeto acrescenta duas de custo — **T1 pico de RAM** e **T2
tempo de inferência** (medido por dois caminhos: ciclos modelados, independentes de
hardware, e latência empírica no relógio) —, que são o núcleo de H1. Este relatório
cobre as duas de custo aqui e situa as duas de Lamenza no fim.

**Nota sobre a figura de memória.** Versões anteriores comparavam a **memória
persistente** dos modelos — bytes do artefato (chave + valores da LUT; pesos `float32`
da MLP), calculados por fórmula. Essa comparação foi **retirada do projeto**: ela
responde "quanto ocupa em flash", não "quanta RAM o dispositivo precisa ter para a
rede rodar", e o segundo é o que decide um deploy. A figura de memória passou a ser
o **pico de RSS medido** (`VmHWM`) do processo que carrega o modelo e decide.

**Nota sobre operações em ponto flutuante.** Versões anteriores incluíam uma
figura T3 que contava multiplicações fp (MACs) por decisão. Ela foi **retirada**:
contar multiplicação favorece a WNN por construção — ela é *weightless* e quase não
multiplica, então o número dizia "≈0" sem capturar o custo real (lookups, acesso a
memória). O custo de computação passou a ser reportado por **ciclos** (que somam
todas as operações) e, sobretudo, pela **latência empírica medida** (T2).

## Comparação na mesma célula (a2s24)

WNN e MADDPG são **ambos de a2s24 (Na=2, Ns=24)** — mesma tarefa, mesma observação
(27 dims), mesma resolução. Pico de RAM, inferência, operações **e** a latência
empírica casam todos no mesmo modelo e na mesma célula, sem a ressalva de "célula
diferente" que versões anteriores (WNN em Na=1) carregavam. **Tempo de conclusão e
taxa de sucesso** (§4) são a parte que ainda depende do checkpoint de performance
avaliado.

## T1 — Pico de RAM do processo

Modelos da WNN: os artefatos de **deploy** (`models/deploy/T{16,32}/a2s24`) — o
ator + termômetro que de fato voam, a MESMA fonte da latência medida (SPEC-03).
Pico = `VmHWM` do processo ao fim do laço de inferência; **arcabouço** = pico depois
dos imports e antes de qualquer peso; **acréscimo do modelo** = pico total − arcabouço.

A figura usa **só o pico total**, como média ± desvio-padrão amostral de **100
processos novos por modelo** (`--repeticoes 100`, `reports/spec03-pico-ram-rep100-a2s24.md`;
mínimo–máximo entre parênteses):

| modelo (arcabouço) | **pico total** (n=100) | latência mediana (n=100) |
|---|---|---|
| **WNN T=16 (2/24), deploy** (NumPy) | **45,93 ± 0,11 MB** (45,57–46,14) | 22,6 ± 0,3 µs |
| **WNN T=32 (2/24), deploy** (NumPy) | **53,26 ± 0,15 MB** (52,97–53,60) | 25,2 ± 0,5 µs |
| MADDPG (Na=2, a2s24) (PyTorch) | **505,98 ± 0,06 MB** (505,84–506,16) | 49,1 ± 0,8 µs |

**O processo da WNN tem pico ~0,09–0,11× o do MADDPG, e é estável**: desvio abaixo
de 0,2 MB nos três modelos entre processos. O motivo da diferença é o arcabouço — o
MADDPG paga ~496 MB só para importar o PyTorch, contra ~31 MB de NumPy na WNN; a
decomposição em pico do arcabouço e acréscimo do modelo (a WNN acresce mais, 15–22 MB
contra 10 MB, porque o load constrói dicionários antes da `LUTPlana`) fica no
relatório da SPEC-03 (`reports/spec03-latencia-memoria-a2s24.md` e o agregado das
100). Nenhum desses números é bytes de artefato.

![box plot das 100 medições por modelo: pico total (log) e latência mediana](spec03-pico-ram-rep100-a2s24.png)

## T2 — Tempo de inferência

**Por ciclos (independente de hardware, RNF-12) — a figura defensável:**

| hardware | WNN (T=32) | MADDPG (256×256, Na=2) | ganho WNN |
|---|---|---|---|
| x86-64 (FPU+SIMD) | 4 784 | 147 152 | **31×** |
| Cortex-M4F (FPU) | 5 096 | 440 332 | **86×** |
| **Cortex-M0 (sem FPU)** | 20 908 | 9 153 172 | **438×** |

**A WNN ganha em inferência, e o ganho explode sem FPU.** Ela troca ~74 000 MACs
por lookups e somas inteiras; num MCU sem unidade de ponto flutuante, cada MAC do
MADDPG custa ~70 ciclos e a diferença vira **três ordens de grandeza**. É o lado
de inferência de H1.

**Por latência empírica MEDIDA (SPEC-03, célula a2s24):**

| modelo | mediana | p99 |
|---|---|---|
| WNN T=16 (NumPy) | 22,1 µs | 78,3 µs |
| WNN T=32 (NumPy) | 24,7 µs | 65,5 µs |
| MADDPG (PyTorch) | 48,6 µs | 213,7 µs |

Medida com **um processo por modelo**, a **mesma observação** para os três
(calculada no processo pai, lida de um `.npy`) e a **menor mediana de 8×1000**
repetições (protocolo da SPEC-03). No relógio, a WNN é **~2,2× mais rápida** que o
ator MADDPG — mesmo com a WNN em NumPy (overhead de Python por lookup) contra o
MADDPG em PyTorch (MACs em BLAS vetorizado). A concordância com os ciclos é **de
sinal, não de magnitude**: o fator empírico (~2×) é muito menor que o modelado (31×
no x86) porque o overhead de arcabouço domina os dois lados e achata a diferença —
uma implementação compilada da WNN realizaria mais da vantagem de ciclos. (Os
valores absolutos são ~5× menores que os do artefato anterior — 118/121/236 µs —,
medido com treinos ao fundo; esta medição foi feita em máquina ociosa. A razão
WNN/MADDPG, que é o que a figura afirma, ficou a mesma.) Vale lembrar (PRD §5.5):
com intervalo de decisão de **2 s**, os dois cabem com >4 ordens de grandeza de
folga — "cumprir o prazo" não distingue nada; a estrutura de operação sim.

## Perfil de operações por decisão (apoio, não veredito)

Para transparência do *perfil* de custo — **não** como figura de mérito. As colunas
estão **separadas por tipo de operação** para não pré-fixar uma convenção de FLOP:
some as colunas de ponto flutuante para o total (mult+add), ou olhe só `mult fp`
para a convenção "só multiplicação". As três últimas (`lookups` + `soma int` +
`bits`) **não são FLOP** — é o trabalho real da WNN, invisível a qualquer FLOP.

| modelo | mult fp | soma fp | div fp | lookups | soma int | bits |
|---|---|---|---|---|---|---|
| WNN T=32 (2/24) | 4 | 218 | 3 | 216 | 864 | 1 728 |
| WNN T=16 (2/24) | 4 | 110 | 3 | 108 | 432 | 864 |
| MADDPG Na=2 | 72 964 | 73 476 | 0 | 0 | 0 | 0 |

(Amostrador Box-Muller: +5 transcendentais por decisão, comum aos dois, omitido acima.)

**Por que separar `mult` e `add` importa.** Olhe a WNN: `mult fp = 4`, mas `soma fp
= 218`. Ela agrega os lookups **somando**, não multiplicando. Então a razão contra o
MADDPG depende inteiramente da convenção:
- **só multiplicação** (coluna `mult fp`): WNN 4 × MADDPG 72 964 → ~**18 000×** —
  enviesado, conta justamente a operação que a WNN *evita*;
- **mult + add** (soma das colunas fp): WNN 230 × MADDPG 146 445 → ~**640×** — mais
  honesto, conta também as somas fp da WNN.

E mesmo o mult+add ignora as **2 808** operações não-FLOP (lookups + inteiros +
bits) que a WNN de fato executa. Essa é a distância para o **~2× medido no
relógio** — por isso o **veredito de computação é a latência empírica**, com este
perfil só como transparência. Números de `reports/spec06-merito.json` (campo
`operacoes`).

## 4. Desempenho (tempo de conclusão e taxa de sucesso)

> **Nota de checkpoint.** As tabelas de **custo** acima usam a WNN de deploy
> **a2s24 (Na=2)**, a mesma célula do MADDPG. As figuras de **desempenho** abaixo
> são do checkpoint da **dissertação (a1s12, Na=1)** — a tarefa mais difícil da
> grade — avaliado isoladamente; o desempenho da WNN a2s24 fica na campanha
> multiagente (escalada T-06.5).

| | tempo de conclusão | taxa de sucesso |
|---|---|---|
| **WNN dissertação (a1s12, Na=1)** | ~40 s (sucesso) · CT censurado ~130 s | 81 % (500 sem.) / 91 % (100 sem.) |
| MADDPG publicado (Na=2 / 4 / 8) | 29,6 / 13,8 / 9,7 s | ~100 % |

O desempenho depende do cenário: mais drones concluem mais rápido (mais cobertura
por passo), então CT cai com Na — o Na=1 da dissertação é a tarefa **mais difícil**
da grade. A comparação head-to-head de desempenho nas células do artigo é a
**escalada T-06.5** (em curso; Na=2 já fechou).

## Veredito de custo (H1)

| eixo | vencedor | margem (mesma célula a2s24) |
|---|---|---|
| **Pico de RAM (T1, processo, n=100)** | **WNN** | 45,9 / 53,3 MB × 506,0 MB (~0,1×) — quase todo o pico do MADDPG é PyTorch |
| **Inferência (T2, ciclos)** | **WNN** | 31× (x86) a **438× (Cortex-M0)** |
| **Inferência (T2, latência medida)** | **WNN** | **~2,2×** no relógio (NumPy × PyTorch) |

O perfil da WNN é o de um alvo **sem FPU** (MCU): ganha em aritmética — nos ciclos
modelados e na latência empírica medida. Em RAM, o **processo** da WNN cabe em
~46–53 MB contra ~506 MB do MADDPG em PyTorch, com desvio abaixo de 0,2 MB em 100
processos — mas isso é sobretudo o preço do arcabouço (RNF-11: arcabouços
diferentes); descontado o arcabouço, a rede sem peso acresce mais ao pico (15–22 MB
contra 10 MB), porque o load em Python constrói dicionários antes da LUT plana. A figura de
mérito certa não é "quem é menor", e sim **qual recurso é o gargalo no hardware
alvo** — e num MCU sem FPU, onde o MADDPG estoura em ciclos, a rede sem peso tem
uma vantagem estrutural real.

## Metodologia — como cada número foi calculado

Todos os números saem de `scripts/spec06_figuras_merito.py` (nada transcrito à
mão), que **consome** o artefato de medição da SPEC-03. Modelos: os artefatos de
**deploy** (`models/deploy/T{16,32}/a2s24`), na mesma célula do MADDPG.

### Pico de RAM (T1) — MEDIDO por processo

Do artefato da SPEC-03 (`scripts/spec03_latencia_memoria.py`), na célula a2s24:
- **um processo por modelo** — o pico de RSS (`VmHWM`) é monotônico dentro de um
  processo: depois de importar o torch, nenhum modelo medido em seguida mostraria
  um pico próprio; e um drone carrega UM modelo;
- **três leituras do `VmHWM`**: base (interpretador), **arcabouço** (após os
  imports: NumPy; ou NumPy + PyTorch), **total** (ao fim do laço de inferência, com
  o modelo carregado). A figura de mérito usa **só o total**; a decomposição fica
  no relatório da SPEC-03;
- **100 repetições** (`--repeticoes 100`): um processo novo por medição; média,
  desvio-padrão amostral, mínimo e máximo por modelo, e o box plot;
- **RSS pós-carga**: `VmRSS` estável após `malloc_trim` — o que fica, não o pico;
- **o simulador fica fora**: a observação é calculada no processo pai e gravada em
  `.npy`; construir o ambiente (~15 MB) entraria no pico do filho.
- **Não há mais contagem de bytes de artefato** (`persistente`/`residente` por
  fórmula): retirada do relatório, das figuras e do JSON.

### Latência (T2) — MEDIDA no relógio, não modelada

Do mesmo artefato e da mesma execução. Protocolo:
- **um processo por modelo** (isola o cache — o forward do torch despeja os caches
  e penalizaria o modelo seguinte);
- a **mesma observação real** (do `reset`, semente fixa) para os três — não `randn`,
  porque o custo da WNN depende de *quais* endereços são tocados;
- aquecimento + **8 blocos × 1 000** repetições cronometradas com
  `time.perf_counter_ns()`; reporta a **menor mediana de bloco** (estimador sem
  contenção) e o **p99 global**.
- Wall-clock em x86-64; **WNN em NumPy, MADDPG em PyTorch** — arcabouços diferentes,
  ambos dominados por overhead, o que achata a diferença (ressalva RNF-11).

### Operações / FLOPs (perfil de apoio) — contagem analítica

Por decisão, via `uavwnn.custo.operacoes`:
- **MADDPG (MLP densa):** por camada `entrada→saída`, `mult_fp = entrada×saída`,
  `soma_fp = entrada×saída + saída` (acumulação + viés), `comparação = saída`
  (ReLU). É a contagem de MACs do produto matriz-vetor.
- **WNN (n-tupla):** `n_ram = ⌈n_dims×n_níveis / tupla⌉` nós de RAM; cada nó monta
  um endereço (`op_bits`, `soma_int` — **inteiro**) e faz um `lookup` (**memória**);
  os valores lidos são somados (`soma_fp = n_ram×n_cabeças`) e há uma `div_fp` por
  cabeça (a média). **Não há multiplicação no aproximador.**
- **Amostrador Box-Muller** (4 mult, 2 soma, 5 transcendentais) é **comum aos dois**.
- **FLOP** = operação de ponto flutuante. `mult_fp`+`soma_fp`+`div_fp` são FLOP;
  `lookups`+`soma_int`+`op_bits` **não são** (memória e inteiro). Daí as duas
  convenções: *só-mult* (coluna `mult fp`) e *mult+add* (soma das colunas fp).

### Ciclos (T2 modelado) — operações × custo por hardware

`ciclos = Σ (nº de cada operação × custo em ciclos daquela operação no alvo)`. Os
pesos por operação (`CICLOS` em `operacoes.py`) são **ordens de grandeza de manuais
de arquitetura** — *Agner Fog instruction tables* (x86) e *ARM Technical Reference
Manuals* (Cortex-M0/M4) —, **não medições nossas**. Servem para mostrar que a
vantagem é condicional ao hardware (sobretudo sem FPU), não um benchmark
ciclo-a-ciclo.

## Artefatos e reprodução

- `reports/spec03-latencia-memoria-a2s24.json` — a MEDIÇÃO (latência e pico de RAM
  por modelo), com `spec03-latencia-memoria-a2s24.md` e os PNGs
  `spec03-latencia-a2s24.png` e `spec03-pico-ram-a2s24.png`.
- `reports/spec06-merito.json` — números por modelo (T1 pico de RAM no campo `ram`,
  T2 ciclos, latência medida, e o perfil de operações no campo `operacoes`).
- `reports/img/spec06-merito/` — `t1_pico_ram.png`,
  `latencia_empirica.png`, `mapa_merito.png`.
- Modelos: `models/deploy/T{16,32}/a2s24` (o artefato que voa, mesma célula do
  MADDPG).

```bash
uv run --group baseline python scripts/spec03_latencia_memoria.py                   # mede (a2s24)
uv run --group baseline python scripts/spec03_latencia_memoria.py --repeticoes 100  # 100 processos/modelo
uv run python scripts/spec06_figuras_merito.py                      # célula padrão a2s24
uv run python scripts/spec06_figuras_merito.py --na 4 --ns 24       # outra célula → sufixo -a4s24
```

O script compara **só modelos da mesma célula** `(Na, Ns)`: os atores WNN de deploy em
`models/deploy/T{16,32}/a{Na}s{Ns}` e o ator MADDPG com a observação dessa célula. Uma
célula sem deploy de WNN não gera figura; uma célula sem o artefato de medição sai sem
T1 e sem latência. Células com WNN treinada abaixo de 50 M passos (hoje a4s24 e a8s24,
6,75 M) saem com aviso: a LUT cresce com o treino.

**Adiado:** desempenho por cenário (tempo de conclusão / SR) WNN×MADDPG — exige a
escalada T-06.5; latência e pico de RAM com os dois lados no MESMO arcabouço (RNF-11),
que separaria o custo da rede do custo da biblioteca.
