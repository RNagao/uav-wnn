# WNN treinada × MADDPG — tamanho, inferência e as figuras de Lamenza

Comparação de custo entre a rede sem peso (WNN) **realmente treinada** e o ator
MADDPG (256×256), **ambos na mesma célula a2s24 (Na=2, Ns=24)** do cenário 003 de
Lamenza. Números do harness de custo da SPEC-03 (`uavwnn.custo`) e da medição da
SPEC-02, não estimados.

**As figuras de mérito de Lamenza são duas** (PRD §5.5): **tempo de conclusão** e
**taxa de sucesso**. O projeto acrescenta duas de custo — **T1 tamanho** e **T2
tempo de inferência** (medido por dois caminhos: ciclos modelados, independentes de
hardware, e latência empírica no relógio) —, que são o núcleo de H1. Este relatório
cobre as duas de custo aqui e situa as duas de Lamenza no fim.

**Nota sobre operações em ponto flutuante.** Versões anteriores incluíam uma
figura T3 que contava multiplicações fp (MACs) por decisão. Ela foi **retirada**:
contar multiplicação favorece a WNN por construção — ela é *weightless* e quase não
multiplica, então o número dizia "≈0" sem capturar o custo real (lookups, acesso a
memória). O custo de computação passou a ser reportado por **ciclos** (que somam
todas as operações) e, sobretudo, pela **latência empírica medida** (T2).

## Comparação na mesma célula (a2s24)

WNN e MADDPG são **ambos de a2s24 (Na=2, Ns=24)** — mesma tarefa, mesma observação
(27 dims), mesma resolução. Tamanho, inferência, operações **e** a latência empírica
casam todos no mesmo modelo e na mesma célula, sem a ressalva de "célula diferente"
que versões anteriores (WNN em Na=1) carregavam. **Tempo de conclusão e taxa de
sucesso** (§4) são a parte que ainda depende do checkpoint de performance avaliado.

## T1 — Tamanho do modelo

Modelos da WNN: os artefatos de **deploy** (`models/deploy/T{16,32}/a2s24`) — o
ator + termômetro que de fato voam, a MESMA fonte da latência medida (SPEC-03).

| modelo | tamanho (ator, fp32) |
|---|---|
| **WNN T=32 (2/24), deploy** | **572 KB** (ordenado) — 41 476 entradas |
| **WNN T=16 (2/24), deploy** | **310 KB** (ordenado) — 22 453 entradas |
| MADDPG (Na=2, a2s24) | 287 KB |

**A WNN paga em memória, mas por margem pequena na mesma célula.** A alta resolução
(T=32) enche 41 476 endereços e ocupa **572 KB (ordenado) a 3 461 KB (denso)** —
**2,0× a 12×** o ator MADDPG. Já o **T=16** (22 453 endereços, **310 KB**) fica
**quase empatado** com o MADDPG (287 KB — só 1,08×). O tamanho é sensível à
resolução do termômetro (mais níveis → mais endereços distintos → mais LUT), mas em
nenhuma das duas a WNN fica *abaixo* do MADDPG: ela **perde** (T=32) ou **empata**
(T=16) em memória — o preço do trade-off memória↔computação que ela ganha em
inferência.

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
| WNN T=16 (NumPy) | 118 µs | 323 µs |
| WNN T=32 (NumPy) | 121 µs | 197 µs |
| MADDPG (PyTorch) | 236 µs | 344 µs |

Medida com **um processo por modelo**, a **mesma observação** para os três e a
**menor mediana de 8×1000** repetições (protocolo da SPEC-03; a2s24 é a célula desta
comparação inteira — WNN e MADDPG). No relógio, a WNN é **~2× mais
rápida** que o ator MADDPG — mesmo com a WNN em NumPy (overhead de Python por
lookup) contra o MADDPG em PyTorch (MACs em BLAS vetorizado). A concordância com os
ciclos é **de sinal, não de magnitude**: o fator empírico (~2×) é muito menor que o
modelado (31× no x86) porque o overhead de arcabouço domina os dois lados e achata
a diferença — uma implementação compilada da WNN realizaria mais da vantagem de
ciclos. (Isto **corrige** uma versão anterior deste relatório, que citava números
hardcoded — WNN 173 µs × MADDPG 35 µs — dizendo o wall-clock *invertido*; a medição
sob o protocolo da SPEC-03 mostra a WNN mais rápida.) Vale lembrar (PRD §5.5): com
intervalo de decisão de **2 s**, os dois cabem com >4 ordens de grandeza de folga —
"cumprir o prazo" não distingue nada; a estrutura de operação sim.

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
| **Tamanho (T1)** | MADDPG | T=32 perde 2,0× · T=16 ~empata (310 vs 287 KB) |
| **Inferência (T2, ciclos)** | **WNN** | 31× (x86) a **438× (Cortex-M0)** |
| **Inferência (T2, latência medida)** | **WNN** | **~2×** no relógio (NumPy × PyTorch) |

O perfil da WNN é o de um alvo **sem FPU** (MCU): ganha em aritmética — nos ciclos
modelados e na latência empírica medida. Em memória, ela **paga**: o T=32 ocupa 2×
o ator MADDPG (o trade-off clássico memória↔computação) e o T=16 apenas empata. A
figura de mérito certa não é "quem é menor", e sim **qual recurso é o gargalo no
hardware alvo** — e num MCU sem FPU, onde o MADDPG estoura em ciclos, a rede sem
peso tem uma vantagem estrutural real ao custo de mais memória (flash externa).

## Metodologia — como cada número foi calculado

Todos os números saem de `scripts/spec06_figuras_merito.py` (nada transcrito à
mão). Modelos: os artefatos de **deploy** (`models/deploy/T{16,32}/a2s24`), na
mesma célula do MADDPG.

### Memória (T1) — contagem de bytes

- **WNN:** `entradas ocupadas × (chave + valores)`, variante **ordenada** (chave
  `uint16` + `d_saída` valores `float32`, vetor ordenado sem folga), via
  `uavwnn.custo.memoria.custo_ntuple_de_rede`. Ex.: T=32 tem 41 476 entradas →
  572 KB. É `persistente` — distinto de `residente` (RAM em execução) e `disco`.
- **MADDPG:** `nº de parâmetros × 4 bytes` (`float32`). Ex.: `27→256→256→2` =
  72 960 pesos + 514 vieses = 73 474 números → **287 KB**.

### Latência (T2) — MEDIDA no relógio, não modelada

Do artefato da SPEC-03 (`scripts/spec03_latencia_memoria.py`), na célula a2s24 (a
única com WNN de deploy e checkpoint MADDPG). Protocolo:
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

- `reports/spec06-merito.json` — números por modelo (T1, T2 ciclos, latência
  medida, e o perfil de operações no campo `operacoes` — FLOPs + lookups/inteiros).
- `reports/img/spec06-merito/` — `t1_memoria.png`, `t2_ciclos.png`,
  `latencia_empirica.png`, `mapa_merito.png`.
- Modelos: `models/deploy/T{16,32}/a2s24` (o artefato que voa, mesma célula do
  MADDPG). Latência: consome `reports/spec03-latencia-memoria-a2s24.json` (medida
  sob o protocolo da SPEC-03).

```bash
uv run python scripts/spec06_figuras_merito.py
```

**Adiado:** desempenho por cenário (tempo de conclusão / SR) WNN×MADDPG — exige a
escalada T-06.5; latência empírica com os dois lados no MESMO arcabouço (RNF-11),
que realizaria mais da vantagem de ciclos que o overhead de NumPy×PyTorch hoje achata.
