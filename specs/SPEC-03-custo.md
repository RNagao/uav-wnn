# SPEC-03 — Harness de custo: T1 tamanho, T2 latência, T3 operações

**Etapa:** todas · **Depende de:** nada · **Estimativa:** 2 dias
**Portão de saída:** nenhum próprio — alimenta `G1`, `G2` e `G3` · **Referência:** PRD §5.5 (protocolo), §6.4 (métricas), D11, D12, RNF-02/03/05/08/09/10/11/12/13

---

## 1. Objetivo

Entregar a **instrumentação** que produz, para qualquer aproximador do projeto (n-tuple, MLP pareada, heurísticas, e depois o KF embutido), três figuras de mérito com protocolo fixo e artefato de saída versionado:

- **T1 — tamanho do modelo**, em bytes **realizados**, separando ator de ator+crítico e armazenamento persistente de memória residente;
- **T2 — tempo de inferência**, decomposto por estágio, em duas fronteiras (modelo isolado e ponta a ponta), com mediana **e p99**;
- **T3 — contagem de operações** por decisão e por tipo, independente de hardware.

O produto desta spec não é um número: é um **medidor** com fronteiras, unidades e regras de atribuição fixadas antes de qualquer resultado existir, mais o formato de artefato que as tabelas do relatório consomem sem digitação manual.

## 2. Por que agora, e por que transversal

Três razões, em ordem de força.

**(a) O protocolo tem que anteceder o resultado.** Tamanho e latência são o conteúdo de H1 (PRD D11). Toda decisão de medição — o que conta como "o modelo", qual fronteira, qual denominador — desloca o número em fatores de 2 a 5. Fixá-las depois de ver o resultado é escolher o número. A conta preliminar do PRD §5.5 já mostra a margem apertada: **~5,6 KB da WNN contra ~8,6 KB da MLP pareada é 1,5×, não 10×**. Numa margem dessas, qualquer arbitrariedade de contabilidade decide a conclusão.

**(b) É a contribuição que sobrevive a resultados medianos.** Se a política ficar apenas comparável, o par (tamanho, aritmética) continua sendo resultado. O PRD §8 classifica a tabela de custo como não-cortável exatamente por isso.

**(c) O que se mede não é prazo.** O intervalo de decisão é 2 s; 15 µs são 0,0008 % do orçamento, e uma MLP 2×32 também não chega perto de estourá-lo. Dizer "roda a tempo" não distingue nada e não é a métrica. As perguntas reais são três, e cada tarefa desta spec responde a uma:

| Pergunta | Figura |
|---|---|
| Que **classe de hardware** o modelo exige (há FPU? há MMU? cabe em flash?) | T3 + T1 |
| Que **aritmética** o caminho de inferência usa (multiplicação fp? só lookup e soma?) | T3 |
| Quanta memória precisa estar **residente** | T1 |

## 3. Escopo

**Dentro:** modelo de custo de memória e medidor T1; curva de ocupação e teste de saturação; ablação de quantização; harness de latência com decomposição por estágio e regimes de cache; paridade de arcabouço entre WNN e MLP; contador de operações analítico e seu espelho instrumentado; modelo de ciclos por classe de alvo; captura e declaração de hardware; esquema do artefato e gerador das tabelas do relatório.

**Fora:** treinar qualquer política (as medições rodam sobre modelos salvos e sobre corpora de observação); escolher `T`, `n`, `N` (SPEC-01); implementar o armazenamento esparso (SPEC-04 — esta spec **especifica o que ele precisa expor** e mede as variantes); portar para hardware restrito (Raspberry Pi é *stretch* do PRD §5.5, aqui só entra como modelo de ciclos).

**Interfaces com outras specs — obrigações que esta spec impõe:**

| Spec | O que precisa expor para o harness |
|---|---|
| SPEC-01 | tempo de codificação e tamanho dos limiares no **esquema desta spec** (T-03.1); as duas rodam em paralelo, então o esquema é a primeira entrega |
| SPEC-04 | `n_entries_occupied()`, `probe_count` do último acesso, e as **três variantes de armazenamento** da T-03.2 atrás da mesma interface |
| SPEC-05 | ator e crítico serializáveis; a MLP com pesos exportáveis para NumPy puro (T-03.6) |
| SPEC-06/08/09 | chamam `measure_all(model, corpus, config)` e nada mais; não reimplementam medição |

---

## 4. Definições que fixam a medição

Esta seção é o contrato. Nenhum número reportado no projeto pode usar definição diferente destas sem decisão registrada.

### 4.1 O que é "o modelo"

**Ator** = tudo que precisa embarcar para produzir uma ação a partir de uma observação já montada: LUTs das cabeças de política, mapeamento das tuplas, limiares do codificador, e — quando aplicável — parâmetros da distribuição de ação. **Ator + crítico** = o anterior mais as LUTs do crítico. O crítico é artefato de treino e fica em terra, o que vale igualmente para o crítico centralizado do MADDPG.

**Regra de atribuição de partes compartilhadas.** Mapeamento de tuplas e limiares do codificador são compartilhados entre ator e crítico. Contam **integralmente na coluna do ator** (é o que embarca) e **uma única vez** na coluna ator+crítico. Reportar o custo marginal do crítico explicitamente, para que a soma seja auditável.

### 4.2 Persistente × residente

Duas colunas, sempre:

- **Persistente (flash/disco):** o que precisa ser gravado no dispositivo.
- **Residente (RAM):** o que precisa estar em memória durante a inferência, incluindo buffers de trabalho (vetor de bits `B`, vetor de `N` endereços, acumuladores).

A distinção não é cosmética. O mapeamento das tuplas é uma permutação pseudoaleatória: se for gerado por PRNG a partir de uma seed, o **persistente** cai para 4 bytes e o **residente** continua ~1,1 KB. **É proibido usar a seed para reduzir a coluna residente** — é a coluna que responde à pergunta "cabe no RAM do microcontrolador".

### 4.3 Fronteiras de tempo

| Fronteira | Começa em | Termina em | Papel |
|---|---|---|---|
| **Modelo isolado** | vetor de observação já montado (`float64[D]`) | ação `⟨θ, s⟩` amostrada e normalizada | **é a comparação de H1** |
| **Ponta a ponta** | estado bruto do simulador (posições, máscara de visitados) | idem | orçamento real por decisão |

A **montagem da observação** (distâncias, ordenação por proximidade, zero-padding — PRD §5.2) é comum a WNN, MLP e ao guloso. Ela vive só na fronteira ponta a ponta. Misturar as duas fronteiras dilui exatamente a diferença que H1 quer medir; a heurística gulosa é medida na fronteira ponta a ponta para dar o **piso comum** — a fração do orçamento que nenhum método evita.

### 4.4 Regra da razão dupla (D12)

Toda razão de tamanho ou de operações é reportada como **par ordenado**, na mesma linha, mesma unidade e mesma precisão:

```
razão = ( X / MLP_pareada , X / ator_MADDPG )
```

O gerador de tabelas (T-03.9) **falha** se uma das duas colunas estiver ausente, e o teste automatizado verifica isso. Vale para relatório, slides e resumo. Enquanto **P4** (arquitetura do ator do MADDPG) estiver aberta, a segunda coluna é reportada como estimativa marcada, com a hipótese declarada na própria célula (padrão CleanRL 256×256), nunca como número liso.

A dimensão da observação depende da célula (`2(Na−1) + 25`): `Na = 1 → 25`, `Na = 4 → 31`. Portanto **a MLP pareada tem tamanho diferente por célula** e a razão precisa vir sempre acompanhada de `(T, n, N, Na, Ns)` — RNF-06.

---

## 5. Tarefas

### T-03.1 — Esquema do artefato de custo

**Vem primeiro: todo o resto escreve nele, e a SPEC-01 já produz dados que precisam caber aqui.**

Um arquivo JSON por `(modelo, célula, configuração, variante)`, mais um Parquet agregado que o gerador de tabelas consome.

```
{
  "meta": { git_sha, timestamp, seeds, spec_version,
            config: {T, n, N, Na, Ns, encoding, forgetting, dtype},
            hw: <bloco da T-03.8>, sw: {python, numpy, torch, blas} },
  "t1":   { actor: {...}, actor_critic: {...}, breakdown: {...},
            occupancy_curve: [[step, occupied, distinct_ram_hits], ...],
            quantization: {fp32: {...}, int16: {...}} },
  "t2":   { boundary: "isolated"|"end_to_end", regime: "seq"|"shuffled"|"cold",
            framework: "numpy"|"torch",
            stages: { assembly, encoding, addressing, accumulate_sample },
            samples_ns: <ou histograma>, median_ns, p99_ns, p99_ci, min_ns,
            block_medians_ns: [...] },
  "t3":   { fp_mul, fp_add, fp_transcendental, int_add, lookups, bit_ops,
            comparisons, bytes_touched, hash_probes: {median, p99, max},
            method: "analytic"|"instrumented" }
}
```

Regras: unidades fixas (bytes e nanossegundos, sempre inteiros); nenhum número derivado é gravado — razões e KB são calculados no gerador de tabelas; toda medição carrega o `git_sha` e a configuração que a produziu.

### T-03.2 — T1: modelo de custo de memória e as três variantes de armazenamento

**Alocado** (cota superior teórica, o número que hoje é reportado e é uma mentira a favor da hipótese):
`cabeças × N × 2^n × bytes_valor`.

**Ocupado** (conteúdo de informação): entradas distintas efetivamente endereçadas × `bytes_valor`. Medido: 2 268 entradas após 80k passos, 1,2 %. **Atenção ao modelo usado:** a rede de gradiente funcional (Katopodis §4.3.1) guarda **só valores**; a NTRN com esquecimento (§4.2.1) guarda **valor e contador** por posição, o que **dobra** o ocupado. O medidor lê `bytes_por_entrada` do modelo, não assume.

**Realizado** (o número honesto para "embarcável"): footprint da estrutura de dados de fato usada. Não é `ocupado`. Avaliar **três variantes** atrás da mesma interface, e reportar a fronteira de Pareto bytes × p99 junto com a T-03.5:

| Variante | Bytes persistentes | Observação |
|---|---|---|
| **V-a** hash aberto por RAM | `Σ_i cap_i × (b_chave + b_valor) + cap_i/8 (bitmap de ocupação) + N × b_cabeçalho`, com `cap_i = 2^⌈log₂(occ_i/α)⌉` | O(1) esperado; paga fator de carga **e** arredondamento para potência de 2 |
| **V-b** chaves ordenadas + busca binária | `occ × (b_chave + b_valor) + N × b_offset` | sem folga de carga; O(log occ) com cauda determinística |
| **V-c** densa por RAM | `= alocado` | previsibilidade máxima, memória máxima — é a mitigação do RNF-10 |

Chave por RAM = o endereço de `n` bits, então `b_chave = 2` para `n ≤ 16` e o índice da RAM não precisa ser guardado.

> **Conta ilustrativa, a refazer com a ocupação medida.** Com `occ = 2 268`, `N = 48` (≈ 47 entradas/RAM), `α = 0,75`, `b_chave = b_valor = 2`: V-a dá `cap_i = 64` → `48 × 64 × 4 ≈ 12,3 KB`; V-b dá `2 268 × 4 ≈ 9,1 KB`. Ambas **muito acima** dos 4,4 KB de "ocupado × bytes". Se esse fator se confirmar, o **realizado** — não o ocupado — é que decide a comparação com a MLP pareada, e a WNN pode perfeitamente sair **maior**. Isso é resultado, e o argumento migra para T3, como o PRD §5.5 já antecipa.

**Componentes obrigatórios de T1, todos individualizados no `breakdown`:**

| Componente | Fórmula |
|---|---|
| LUTs (alocado/ocupado/realizado, por variante) | acima |
| **Mapeamento das tuplas** | `N × n × b_índice`, `b_índice = ⌈log₂ B⌉` arredondado para byte. Com `N=48, n=12, B<65 536` → 576 × 2 B ≈ 1,1 KB. **Mais de 20 % do total; omitir infla a vantagem** |
| Limiares do codificador | `0` se o binning for uniforme (E0); `D × T × b_limiar` para E1/E2/E3 da SPEC-01. **Este item pode dominar**: com `D=25, T=32`, 800 limiares em fp32 são 3,2 KB — mais que as LUTs ocupadas |
| Parâmetros da distribuição de ação | `σ²` é hiperparâmetro (PRD §5.3): 4–8 bytes |
| Buffers de trabalho (só residente) | vetor de bits `B/8`, `N` endereços, acumuladores por cabeça |

**MLP:** `Σ numel(p)` sobre pesos **e vieses**, contados por camada. Para `D→32→32→2`: `(D·32+32) + (32·32+32) + (32·2+2)`. Reportar por célula, porque `D` varia com `Na`.

**Verificação cruzada obrigatória.** O modelo de custo é uma fórmula, e fórmula erra. Confrontar com (i) o tamanho do artefato serializado sem compressão (`.npz` cru) e (ii) uma medição de heap (`tracemalloc` sobre a construção da estrutura, ou `pympler.asizeof`). A medição de heap é **diagnóstico, não é o número reportado** — ela mede o overhead do CPython, não o do alvo embarcado. Divergência maior que 10 % entre fórmula e artefato serializado é bug e bloqueia o reporte.

### T-03.3 — T1: curva de ocupação e teste de saturação (RNF-08)

Ocupação cresce com o treino; valor final sem curva é afirmação com prazo de validade.

- Instrumentar o contador de entradas distintas e amostrá-lo em passos **log-espaçados** (≈ 40 pontos por ordem de grandeza inicial, rarefazendo depois), gravando também a ocupação por RAM (para expor desbalanceamento).
- Ajustar dois modelos: lei de potência `occ(t) = a·t^b` e saturante `occ(t) = occ_∞(1 − e^{−t/τ})`. Reportar qual ajusta melhor por AIC e o parâmetro relevante (`b` ou `occ_∞`).
- **Critério de saturação:** elasticidade `d log occ / d log t < 0,05` sustentada no último terço dos passos medidos. Se não saturar, a alegação de "kilobytes" sai **com o horizonte declarado** e com a extrapolação para o orçamento de 50 M passos de Lamenza, marcada como extrapolação.
- **Curva de ocupação efetiva (poda):** com esquecimento `φ < 1`, entradas tocadas uma vez há muito tempo têm valor decaído. Medir também `occ_ε` = entradas com `|v| > ε` e reportar a curva podada como ablação. Se `occ_ε` saturar e `occ` não, a estrutura embarcável é a podada e isso precisa estar dito com a perda de desempenho medida junto.

### T-03.4 — T1: quantização como ablação avaliada (RNF-09)

Reportar int16 só é legítimo se a política quantizada for avaliada.

- Pipeline: treinar em fp32 → quantizar valores das LUTs para int16 com escala **declarada** (global vs por cabeça vs por RAM — a escolha muda o erro e é ablação) → **reavaliar tempo de conclusão nos mesmos 500 episódios com as mesmas seeds**.
- Saída obrigatória: o **par (tamanho, desempenho)** para fp32 e para int16, lado a lado. É proibido publicar o tamanho int16 ao lado do desempenho fp32.
- Critério: se a diferença de tempo de conclusão ficar dentro do IC bootstrap 95 %, int16 vira o número principal com a nota "quantização avaliada, sem degradação detectável"; senão os dois pares são reportados e a discussão fica.
- Medir também o **erro de quantização** das saídas do ator sobre o corpus (mediana e p99 do desvio angular em graus), que diagnostica antes de gastar 500 episódios.

### T-03.5 — T2: harness de latência

**Relógio.** `time.perf_counter_ns()` (monotônico, resolução de ns) para as amostras individuais; `time.process_time_ns()` numa passada paralela como verificação cruzada de que não estamos medindo tempo roubado por outros processos. **Não usar `timeit` com autorange** — ele agrega e destrói o p99.

**Piso do relógio.** Calibrar o overhead do par de chamadas medindo 10⁵ pares vazios; reportar a mediana como piso. **Regra:** estágio cuja mediana seja menor que 10× o piso não recebe p99 — é medido em lote e reportado só como mediana, marcado como `below_clock_floor`. Isso evita p99 fabricado pelo próprio instrumento.

**Decomposição por estágio.** Método principal: **prefixos cumulativos**. Cronometrar quatro pipelines — (1) montagem; (2) montagem+codificação; (3) montagem+codificação+endereçamento; (4) completo — e diferenciar. Assim o custo do relógio não entra entre os estágios. Método de verificação: cronometragem aninhada numa execução **separada**, jamais a mesma que produz o total. As duas precisam bater dentro de 10 %.

**Regimes de acesso à memória.** É aqui que o p99 se decide, e usar um só regime é escolher a resposta:

| Regime | Como | Papel |
|---|---|---|
| **seq** | observações consecutivas de rollouts on-policy reais, na ordem | localidade temporal como no uso real |
| **shuffled** | as mesmas observações, ordem embaralhada por seed | pior caso de localidade |
| **cold** | seq + varredura de um buffer maior que o L3 entre decisões | **o mais fiel ao cenário embarcado**: uma decisão a cada 2 s significa cache frio de verdade |

Reportar `seq` como linha principal do modelo isolado e `cold` como a linha de referência para a alegação de embarcabilidade. **Nunca comparar WNN e MLP em regimes diferentes** — o gerador de tabelas bloqueia isso.

**Entradas.** Mínimo de 10 000 observações **distintas** de rollouts. Repetir uma observação deixaria a LUT esparsa integralmente quente em cache e falsearia o p99 justamente na dimensão que se quer medir.

**Repetições e significância.** `R = 5` blocos em **processos separados** × `M = 20 000` decisões, após aquecimento de 2 000 decisões descartadas por bloco. `n = 10⁵` amostras garante `n(1−p) = 1 000` acima do p99. IC do p99 por **estatística de ordem binomial** (mais barato e exato que bootstrap para quantis); bootstrap de 1 000 reamostragens fica para a mediana, por coerência com o resto do PRD. **Validade do ambiente:** se as medianas dos 5 blocos variarem mais de 5 % entre si, a medição é descartada e o ambiente é reinvestigado.

**Estimadores reportados:** `min` (melhor estimativa do custo intrínseco, robusto a interferência), `mediana`, `p99` com IC, e `block_medians` para auditoria.

**Antifalseamento.** Consumir o resultado (acumular a ação numa soma impressa ao final) para impedir eliminação silenciosa; variar a entrada a cada chamada; verificar que nenhum objeto do caminho medido está memoizado; `gc.disable()` durante o bloco com coleta entre blocos; `sys.setswitchinterval` alto; conferir que nada do caminho medido foi içado para fora do laço (revisão de código explícita, item de checklist).

**Separar cauda por hash de cauda por SO.** Instrumentar o **número de sondas do hash por decisão** (T-03.7) e correlacioná-lo com a latência da mesma decisão. Se o p99 de latência não correlacionar com o p99 de sondas, a cauda é do escalonador e não da estrutura — e a conclusão sobre "inferência previsível" muda completamente. Sem essa separação o p99 é ininterpretável.

### T-03.6 — T2: paridade de arcabouço (RNF-11)

Uma MLP `31→32→32→2` executada via PyTorch mede sobretudo o despacho do PyTorch.

- Implementar o caminho de inferência da MLP em **NumPy puro** (três `matmul`, ReLU, sem autograd), importando os pesos treinados.
- **Teste de equivalência:** `max|y_numpy − y_torch| < 1e-5` sobre 10 000 observações. É teste automatizado, não conferência manual.
- Reportar **duas colunas** — NumPy e PyTorch (`torch.no_grad()`, `torch.set_num_threads(1)`). A linha comparativa de H1 usa a coluna NumPy dos dois lados.
- Registrar no artefato quanto do tempo PyTorch é despacho (`torch − numpy`): é um número interessante por si e evita que alguém o cite como custo do modelo.

### T-03.7 — T3: contagem de operações (RNF-12)

A figura independente de hardware, e onde a diferença é estrutural.

**Método duplo, obrigatório:**

1. **Analítico** — fórmula fechada parametrizada por `(D, T, N, n, Ns, Na, cabeças)`. É o que vai na tabela, porque não depende de implementação.
2. **Instrumentado** — implementação-espelho com um contador que envolve as operações elementares, executada **uma vez, fora do caminho de tempo**. Serve de teste.

Discrepância maior que 1 % entre os dois é bug e bloqueia o reporte.

**Categorias, reportadas separadamente (agregar em "flops" apaga o argumento):**

`fp_mul` · `fp_add` · `fp_transcendental` (exp, log, sqrt, sin/cos) · `int_add` · `lookups` (acessos indexados) · `bit_ops` (shift/or/and da montagem de endereço) · `comparisons` · `bytes_touched` (working set por decisão).

**Formas esperadas das fórmulas — a instanciar, não a assumir:**

- *Montagem da observação (comum a todos, inclusive ao guloso):* `Ns` distâncias (2 subtrações, 2 multiplicações, 1 soma cada; `sqrt` evitável usando `d²`) + ordenação parcial `O(Ns log N*_s)` comparações.
- *Codificação:* termômetro por limiar custa `D·T` comparações; por índice custa `D` divisões + `D` floors + escrita de `D·T` bits. **As duas implementações são contadas**, porque a escolha muda a categoria de operação (comparação vs divisão fp) e é exatamente o que a alegação "sem FPU" depende.
- *Endereçamento:* `N·n` `bit_ops` + `N` `lookups` por cabeça.
- *Acumulação:* `cabeças × (N−1)` somas + 1 multiplicação por `1/N` por cabeça.
- *Amostragem gaussiana:* 2 normais padrão (Box–Muller: `log`, `sqrt`, `cos/sin`) + escala e deslocamento.
- *MLP:* `D·32 + 32·32 + 32·2` MACs fp + 66 somas de viés + 64 comparações de ReLU.

**Honestidade exigida.** O caminho da WNN **não** é livre de ponto flutuante: a acumulação é soma fp (a menos que quantizada) e a amostragem da gaussiana usa transcendentais. A alegação defensável e verificável é **"zero multiplicações em ponto flutuante no aproximador"**, e o resíduo fp (amostragem; nas Etapas 2–3 também o KF, ~50–80 flops/passo pelo RNF-05) aparece **na mesma tabela**. O texto do relatório só pode afirmar sobre o que a tabela conta.

**Sondas de hash.** `hash_probes` por decisão (mediana, p99, máximo) é figura de T3 e insumo da T-03.5. É a medida estrutural da cauda — independe de hardware.

### T-03.8 — Classe de hardware: modelo de ciclos e declaração de ambiente (RNF-13)

**Modelo de ciclos por classe de alvo.** Converte T3 em afirmação de classe de hardware sem comprar hardware: uma tabela de custo em ciclos por operação para 2–3 perfis declarados (por exemplo: núcleo sem FPU com fp em software; núcleo com FPU de precisão simples; o x86 do experimento), e o produto `Σ ops × ciclos/op`. **É estimativa declarada, não medição** — rotulada como tal em toda tabela. O ponto que ela fecha: numa classe sem FPU, cada multiplicação fp custa dezenas de ciclos em software, e é aí que "2 080 MACs" e "0 multiplicações fp" deixam de ser a mesma ordem de grandeza. O *stretch* do Raspberry Pi, se houver tempo, valida um ponto desse modelo.

**Declaração de ambiente, capturada automaticamente para o bloco `hw`:** modelo e frequência nominal do CPU, `governor` e estado de turbo, tamanhos de L1/L2/L3, RAM, kernel e distribuição, versões de Python/NumPy/PyTorch e BLAS em uso.

**Controle de execução, aplicado e registrado:** `taskset` para um núcleo fixo; `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; `CUDA_VISIBLE_DEVICES=""` para os dois lados; prioridade elevada (`nice`/`chrt`) se disponível; nenhuma outra carga na máquina, verificada por *load average* antes e depois.

> **Ameaça de validade específica deste ambiente.** O desenvolvimento está em **WSL2**, onde controle de frequência, isolamento de núcleo e latência de escalonamento passam pelo hipervisor do host. Isso ataca o p99 justamente onde ele importa. Mitigação, em ordem: (i) rodar a bateria final em Linux nativo, se houver máquina; (ii) se não houver, declarar WSL2 no relatório, apoiar-se em `min` e mediana para o custo intrínseco, e usar a correlação latência × sondas da T-03.5 para dizer o que na cauda é da estrutura e o que é do ambiente.

### T-03.9 — Tabelas do relatório e trava do denominador (D12)

Um gerador que lê o Parquet agregado e emite as três tabelas em Markdown, prontas para E2. **Nenhuma tabela do relatório é digitada à mão** — isso é o que faz `make repro` significar alguma coisa.

- **Tabela T1:** linhas = {WNN ator, WNN ator+crítico, MLP ator, MLP ator+crítico, ator MADDPG (referência)}; colunas = {alocado, ocupado, realizado-V*, persistente, residente, razão vs MLP pareada, razão vs ator MADDPG, config `(T,n,N,Na,Ns)`}.
- **Tabela T2:** linhas = {WNN, MLP, guloso} × {isolado, ponta a ponta} × regime; colunas = {por estágio, min, mediana, p99, IC do p99, arcabouço}.
- **Tabela T3:** linhas = métodos; colunas = as categorias de operação + `bytes_touched` + ciclos estimados por perfil de alvo.

**Travas automatizadas:** falha se faltar uma das duas razões; falha se WNN e MLP forem comparadas em regimes ou arcabouços diferentes; falha se houver número de tamanho ou latência sem a configuração que o gerou (RNF-06); falha se um tamanho int16 aparecer sem o desempenho int16 correspondente (RNF-09).

### T-03.10 — Ganchos para as Etapas 2 e 3 (RNF-05)

O KF é código que voa no UAV e conta no orçamento. O harness precisa medir o **pipeline completo** — filtro + codificador + aproximador — sem reescrita, quando `fusion/` estiver ligado:

- T1 acrescido do estado do filtro (`x̂`, `P`, matrizes) e, na Etapa 3, dos buffers de interseção de covariância;
- T2 com um estágio novo, `filtering`, antes da codificação;
- T3 com as flops do KF na mesma tabela — a comparação relevante é `~50–80 flops/passo` contra `~2 080 MACs` da MLP, ~2–3 %, que é o número que impede a acusação de "escondeu o custo do filtro".

Com `fusion/` desligado, o harness produz exatamente a tabela da Etapa 1 (RNF-07): mesmo esquema, estágio `filtering` ausente.

---

## 6. Critério de saída

1. **Esquema do artefato congelado** (T-03.1) e adotado pela SPEC-01 retroativamente.
2. **T1 medido de ponta a ponta em um modelo real** (o protótipo de 80k passos serve): alocado, ocupado e realizado nas **três variantes**, com mapeamento de tuplas e limiares individualizados, persistente e residente separados, e a verificação cruzada fórmula × artefato serializado dentro de 10 %.
3. **Curva de ocupação produzida** com veredito de saturação escrito (satura / não satura, com o horizonte declarado).
4. **T2 medido** para WNN e MLP nas duas fronteiras, nos três regimes, com decomposição por estágio pelos dois métodos batendo dentro de 10 %, e estabilidade entre blocos abaixo de 5 %.
5. **Paridade de arcabouço verificada** por teste de equivalência numérica automatizado.
6. **T3 fechado**: analítico e instrumentado batendo dentro de 1 %, categorias separadas, resíduo fp explícito.
7. **Gerador de tabelas rodando** com as travas ativas, e as três tabelas emitidas — ainda que com dados do protótipo.
8. **Testes automatizados:** determinismo do medidor (mesma entrada, mesmo T1/T3); T1 e T3 invariantes à ordem das observações; trava da razão dupla falhando quando deve; equivalência NumPy × PyTorch; medidor de latência recusando p99 abaixo do piso do relógio.
9. **Relatório curto** (1–2 páginas) com o protocolo e as decisões de contabilidade, pronto para virar a seção de método de custo de E2.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **O "realizado" desmonta a vantagem de tamanho** — hash com fator de carga pode dar 2–3× o "ocupado", e a WNN sair maior que a MLP pareada | É o resultado, e o PRD §5.5 já o antecipa. A tarefa da spec é que ele apareça cedo e por fórmula auditável, não na véspera. O argumento migra para T3 (aritmética e ausência de FPU), que é a alegação estrutural |
| **Limiares do codificador dominam T1** se a SPEC-01 escolher binning por slot (E1/E2/E3) | Contabilizar desde já como item de primeira classe; alimentar de volta a SPEC-01 com o custo, para que a escolha do codificador seja feita com o preço à vista (T-01.7) |
| **p99 medido é o do escalonador, não o do hash** — agravado por WSL2 | Correlação latência × sondas de hash (T-03.5); `min` e mediana como estimadores do custo intrínseco; Linux nativo para a bateria final se possível; ambiente declarado |
| **Instrumentação altera o que mede** (efeito observador na decomposição por estágio) | Prefixos cumulativos como método principal; cronometragem aninhada só como verificação, em execução separada; piso do relógio calibrado e respeitado |
| **Comparação injusta por arcabouço** | NumPy puro dos dois lados com equivalência numérica testada; duas colunas quando PyTorch entrar |
| **Escolha conveniente de denominador** entra pelo texto, mesmo com a tabela correta | Trava no gerador + revisão explícita do texto do relatório e dos slides contra a regra da §4.4 |
| **Curva de ocupação não satura** e "kilobytes" vira alegação com prazo | RNF-08: horizonte declarado, extrapolação marcada como extrapolação, e a curva podada por `|v| > ε` como alternativa embarcável com a perda medida junto |
| **P4 aberta** deixa metade da razão dupla como estimativa | Célula marcada, hipótese declarada (CleanRL 256×256); escalar P4 se a comparação virar argumento central do relatório |
| Harness vira dependência crítica de todas as specs e atrasa tudo | Escopo fechado em 2 dias; a interface `measure_all(...)` é mínima e estável; variantes de armazenamento entram atrás da mesma interface, pela SPEC-04 |
| Medição de heap do CPython citada como se fosse o footprint embarcado | Regra explícita: heap é diagnóstico, o número reportado é o do modelo de custo do formato de implantação |

---

## 8. Entregáveis

- `src/uavwnn/eval/cost/schema.py` — esquema do artefato, validação e escrita
- `src/uavwnn/eval/cost/size.py` — T1: modelo de custo, três variantes, persistente × residente, curva de ocupação
- `src/uavwnn/eval/cost/latency.py` — T2: fronteiras, prefixos cumulativos, regimes de cache, estatística de ordem para o p99
- `src/uavwnn/eval/cost/ops.py` — T3: contagem analítica e espelho instrumentado
- `src/uavwnn/eval/cost/hwinfo.py` — captura de hardware/software e controle de execução
- `src/uavwnn/eval/cost/tables.py` — gerador das tabelas T1/T2/T3 com as travas
- `src/uavwnn/agents/mlp_numpy.py` — caminho de inferência da MLP em NumPy puro (paridade de arcabouço)
- `configs/spec03/` — perfis de benchmark (repetições, regimes, classes de alvo)
- `reports/spec03-custo.md` — protocolo e decisões de contabilidade
- `tests/test_cost.py` — determinismo, invariâncias, travas, equivalência NumPy × PyTorch, piso do relógio
- alvo `make cost` no Makefile, encadeado em `make repro`

---

## 9. Perguntas em aberto

1. **Qual variante de armazenamento é "a de implantação"?** V-a (hash) é o que o PRD assume, mas a conta ilustrativa sugere que V-b (chaves ordenadas) pode ser menor **e** de cauda mais previsível, ao custo de `O(log occ)`. Se V-b vencer nas duas dimensões, a decisão precisa ser registrada e a SPEC-04 muda de alvo. Decidir com a fronteira de Pareto medida, não antes.
2. **O modelo de gradiente funcional guarda contador?** Katopodis §4.3.1 diz que não; a NTRN com esquecimento (§4.2.1) guarda valor e contador. Se a implementação do projeto misturar os dois (gradiente funcional **com** esquecimento multiplicativo, que é o que o PRD RNF-04 pede), é preciso decidir se o esquecimento exige contador por posição — a resposta **dobra ou não** o ocupado, e é a maior incerteza isolada de T1.
3. **A seed do mapeamento pode reduzir a coluna persistente?** A regra da §4.2 diz que sim para flash e não para RAM. Falta decidir se o relatório reporta a coluna persistente com a seed (número menor, defensável) ou sem (mais conservador). Recomendação: reportar as duas, com a residente como número principal.
4. **Qual perfil de alvo declarar no modelo de ciclos?** A alegação do trabalho é "microcontrolador sem FPU". Falta escolher um alvo nomeado com números de ciclos citáveis, ou assumir o risco de o revisor considerar o perfil arbitrário.
5. **Onde roda a bateria final de T2?** Se WSL2 for o único ambiente, o p99 sai com ressalva permanente. Vale procurar uma máquina Linux nativa antes da semana 2.
6. **A SPEC-01 roda em paralelo e já mede custo de codificação** (T-01.7). Se o esquema desta spec não estiver pronto no dia 1, ou a SPEC-01 grava em formato próprio e converte depois, ou espera. Precisa de decisão de sequenciamento no primeiro dia.
7. **O guloso entra na tabela T3?** Ele não tem modelo, mas tem custo por decisão (distâncias e ordenação) e é o baseline de H6. Reportá-lo dá o piso comum e é honesto; falta decidir se ele ocupa linha própria na tabela ou aparece só como fração da montagem da observação.
8. **Precisão do valor nas LUTs:** fp32 é o padrão de treino, int16 é a ablação. Falta decidir se existe um terceiro ponto (int8) que valha medir, dado que a faixa de valores de um crítico e de um ator são muito diferentes e a escala por cabeça pode não bastar.
