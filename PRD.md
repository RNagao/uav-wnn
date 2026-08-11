# PRD — Coleta de dados por enxame de UAVs: rede sem peso e, depois, filtro de Kalman

**Projeto:** `uav-wnn`
**Disciplina:** Fusão de Dados — PESC/COPPE-UFRJ
**Autor:** Raphael Nagao Ferreira · rnagao@cos.ufrj.br
**Data:** 09/08/2026 · **Prazo:** 4 semanas (entrega em ~06/09/2026)
**Status:** rascunho para aprovação · **Versão:** 2.1 — *problema de Lamenza preservado; tamanho de modelo e tempo de inferência como figuras de mérito de primeira classe (§5.5)*

---

## 0. Princípio que rege este documento

> **O problema formulado por Lamenza (2025) não é modificado.** Cenário, observação, ação, recompensa, métricas e parâmetros são reproduzidos como estão na dissertação. A única coisa que trocamos é o **aproximador de função**: rede n-tuple sem peso no lugar das redes profundas.

Só depois de a rede sem peso ser avaliada sobre o problema original é que a **extensão de Fusão de Dados** entra: medição ruidosa e filtro de Kalman. A extensão é declarada como extensão, em capítulo próprio, e nunca se mistura com a reprodução.

Toda alteração ao problema original precisa aparecer explicitamente na tabela da §5.4. Hoje ela tem **uma única linha**: a binarização da entrada, que é inevitável porque redes n-tuple endereçam RAMs com bits.

---

## 1. Contexto

Sensores IoT em regiões remotas não têm infraestrutura de rede. Um enxame de UAVs voa até eles, coleta por enlace local e transporta os dados. Lamenza (2025, PUC-Rio) resolveu a coordenação desse enxame com **MADDPG** sobre redes profundas no simulador GrADyS-SIM NextGen, com resultado forte — comportamento colaborativo emergente, superando consistentemente a heurística gulosa — e um custo declarado no próprio texto: **32 dias de treino em GTX 3090** para 9 células da grade, ~7 dias por célula isolada, um modelo por configuração.

Katopodis (2022, COPPE/UFRJ) mostrou que **redes n-tuple sem peso** atingem desempenho comparável ao de MLPs em SARSA, REINFORCE e Actor-Critic — inclusive em tarefas de **ação contínua** (Hopper: n-tuple Actor-Critic 748,8 contra 643,6 do Actor-Critic com MLP; Mountain Car: 93,3 contra 0,02) — com inferência em tempo constante, sem MACs e sem GPU.

Este projeto testa se o aproximador barato resolve o problema caro. E, num segundo momento, ataca a hipótese mais frágil da formulação original: em Lamenza, a política enxerga o **estado verdadeiro**, via um "canal global" (satélite) que entrega posições exatas de todos os UAVs e o status de visitação de todos os sensores. O próprio autor lista isso como limitação a endereçar no futuro. É aí que a disciplina tem algo a dizer — e é aí que o filtro de Kalman entra.

Experimentos preliminares já executados (documentados em `tmp/PLANO.md`) produziram três achados **medidos** que sustentam o projeto:

| # | Achado | Número |
|---|---|---|
| A1 | Ocupação real das LUTs é minúscula | 2 268 de 196 608 entradas (**1,2 %**) após 80k passos → ~4,4 KB em int16 |
| A2 | WNN não é mais frágil a ruído que MLP | degradação estatisticamente indistinguível de σ=0 a σ=0,2 |
| A3 | KF ingênuo piora; KF **com entrada de controle** ganha muito | erro de posição 12,54 m → 0,67 m com σ_GPS = 10 m (**18,8×**) |

E uma regra de projeto que amarra as duas metades do trabalho:

> **Largura do bin do termômetro ≈ σ da estimativa filtrada.** O KF não melhora a política diretamente — ele habilita um codificador mais fino, que é o que recupera a acurácia.

---

## 2. Objetivo e etapas

### Estrutura

| Etapa | Configuração | O que muda em relação a Lamenza | Responde |
|---|---|---|---|
| **1** | 1 UAV, sem ruído | só o aproximador | H1 — a rede sem peso aprende o problema? |
| **2** | 1 UAV, com ruído | + medição ruidosa e KF | H2, H3, H4 |
| **3** | `Na` UAVs, com ruído | + coordenação e estimativas correlacionadas | H5, H6 |

A Etapa 1 é a **reprodução**. As Etapas 2 e 3 são a **extensão de Fusão de Dados**.

`Na = 1` não é uma modificação do problema: `Na` é parâmetro do cenário (tab. 3.1 de Lamenza), e o próprio autor roda experimentos exploratórios com `N*_a = 0`, isto é, agentes cegos aos demais. A Etapa 1 percorre `Na = 1 → 2 → 4 → 8` dentro da mesma formulação.

Cada etapa tem um **portão** (`G1`, `G2`, `G3`): critério objetivo a ser atingido — ou explicitamente dispensado por decisão registrada — antes de a seguinte começar. Portão não atingido é resultado publicável, não motivo para parar.

### Hipóteses

| ID | Etapa | Hipótese | Como é falsificada |
|---|---|---|---|
| **H1** | 1 | Sobre o problema **inalterado**, a política n-tuple atinge desempenho comparável ao de uma rede densa, a custo de **modelo** e **inferência** materialmente menor | Comparação pareada: mesmo ambiente, mesma observação, mesma recompensa, **mesmo algoritmo**, trocando só o aproximador. Critério detalhado em **§5.5** — H1 cai se o tempo de conclusão da WNN for pior além do IC, ou se o ator não couber em poucas dezenas de KB **realizados**, ou se a contagem de operações não mostrar o aproximador livre de **multiplicações em ponto flutuante**. **Atenção:** três specs independentes (03, 04, 05) estimam que o ator WNN realizado fica entre ~12 e ~15 KB contra ~8,6 KB da MLP pareada — isto é, provavelmente **maior**. Ver §5.5 e a nota de revisão |
| **H2** | 2 | A binarização por termômetro **não** torna a WNN mais frágil a ruído de medição que uma rede densa | Curvas de degradação WNN vs MLP sobre σ crescente. H2 cai se os ICs separarem |
| **H3** | 2 | O KF com a **ação da política como entrada de controle** devolve a rede ao regime praticamente sem ruído | Escada de ablação 1→4. H3 cai se a degradação com filtro for estatisticamente distinguível do oráculo |
| **H4** | 2 | Uma política que enxerga a **própria covariância** aprende busca ativa | Ablação com/sem canais de `P`; inspeção de trajetórias. H4 cai se o ganho for nulo |
| **H5** | 3 | Fundir estimativas entre UAVs por **interseção de covariância** supera o KF local isolado e não diverge, ao contrário do KF ingênuo | Erro de estimativa e NEES/NIS. H5 cai se CI não bater o KF local, ou se o KF ingênuo não divergir |
| **H6** | 3 | A coordenação emerge também com aproximador sem peso: o enxame supera a heurística gulosa, como em Lamenza | Tempo de conclusão vs heurística gulosa. H6 cai se a política não bater o guloso |

---

## 3. Fora de escopo

- **Alterar a formulação do problema.** Nenhuma mudança em observação, ação, recompensa, cenário ou métricas. Ideias descartadas por esse motivo estão registradas na §12 como trabalho futuro, não aplicadas.
- Reproduzir ou retreinar MADDPG. Entra como **referência de literatura** ("teto conhecido"), nunca como baseline pareado — comparar "n-tuple" com "MADDPG do artigo" confundiria algoritmo, arquitetura e implementação numa variável só.
- Modelo de propagação de rádio realista, restrições de energia, Age of Information, treino assíncrono.
- Generalização para número variável de agentes. Lamenza tentou (zero-padding e *phantom agents*, §7.4.2) e **falhou** em cenários grandes; não é uma batalha para 4 semanas.

---

## 4. Entregáveis

| # | Entregável | Aceite |
|---|---|---|
| E1 | **Repositório reprodutível** `uav-wnn` | `make repro` regenera todas as tabelas e figuras a partir de seeds fixas |
| E2 | **Relatório/artigo** | Reprodução e extensão em capítulos separados; resultados com IC; limitações honestas |
| E3 | **Apresentação de slides** | Mesmo padrão visual da proposta já apresentada; 12–15 slides |

---

## 5. Especificação — o problema de Lamenza, verbatim

### 5.1 Cenário (Lamenza §3, tab. 3.1 e 7.1)

| Parâmetro | Valor | Origem |
|---|---|---|
| Área | 200 × 200 m (`scenario_size = 100`) | tab. 7.1 |
| Altitude | fixa; problema tratado em 2D | §3 |
| `algorithm_iteration_interval` | 2 s | tab. 7.1 |
| `max_episode_length` | 500 s | tab. 7.1 |
| `max_seconds_stalled` | 30 s | tab. 7.1 |
| Velocidade | 0–15 m/s, mudança de direção instantânea | §3, §5.2 |
| Posição inicial dos UAVs | aleatória em área pequena perto do centro | §3 |
| `Na` × `Ns` | {2, 4, 8} × {12, 24, 36} | tab. 7.1 |
| `communication_range` | **a confirmar** (P1) | não consta nas tabelas da dissertação |
| γ | 0,99 | tab. 7.1 |
| Episódios de avaliação | 500 | §7.2 |

**RF-01** — episódico: sensor exige visita única e sai do problema; posições sorteadas a cada episódio.
**RF-02** — poda de episódio parado (`max_seconds_stalled`), mas **não** podar quando todos os sensores são coletados. Lamenza mediu degradação forte em ambos os desvios (§5).
**RF-03** — coleta automática por proximidade: o UAV emite *heartbeats*; sensor dentro de `communication_range` responde e é marcado como visitado. Coletar **não** é ação do agente.

**Células:** Etapas 1 e 2 usam `Na = 1, Ns = 12`. Etapa 3 usa `Na = 4, Ns = 24`, com `Na ∈ {2, 8}` conforme o tempo permitir.

### 5.2 MDP — reproduzido sem alteração

**Observação (Lamenza eq. 5-1):**

```
o_k^t = ⟨ P^a_{k,t} , P^s_{k,t} , k ⟩
```

- `P^a_{k,t}` — coordenadas **relativas** dos `N*_a` agentes mais próximos, ordenados por distância crescente. `N*_a = Na − 1`.
- `P^s_{k,t}` — coordenadas **relativas** dos `N*_s` sensores **não visitados** mais próximos, ordenados por distância. `N*_s = 12`, com **zero-padding** quando restam menos.
- `k` — índice do agente.

Coordenadas relativas, e não absolutas, são escolha deliberada de Lamenza: colapsam cenários equivalentes numa mesma observação e dispensam incluir a posição do próprio agente. Sensores já visitados **não** aparecem — incluir `visited_i` mediu pior no trabalho original.

Dimensão: `2(Na − 1) + 24 + 1` escalares. Para `Na = 1` → **25**; `Na = 4` → **31**; `Na = 8` → **39**.

**Ação (Lamenza eq. 3-2 e §5.2):**

```
a_k^t = ⟨ θ ∈ [0, 2π] , s ∈ [0, 15] ⟩   →  normalizada para [0,1]²
```

**Contínua.** Lamenza dedica parte do capítulo 2 a criticar trabalhos que discretizam o espaço em grade, e a continuidade é uma das colunas da tabela comparativa dele. Discretizar seria modificar o problema — e é exatamente o desvio que o v1 deste PRD cometeu e que fica aqui revertido.

**Recompensa (Lamenza eq. 5-6):**

```
R_t =  Δvisited_t · 10                        , se Δvisited_t > 0
       ( Σ_i visited_i^t − Ns ) / Ns          , caso contrário
```

Sem *reward shaping*, sem penalidade de borda, sem termo de coordenação. A recompensa do experimento exploratório (eq. 7-2, `R = Σ visited_i / Ns`) é usada apenas na réplica do experimento exploratório, como no original.

**Sem sinalização entre agentes.** Em Lamenza a coordenação emerge só de cada agente conhecer a posição dos demais — não há mensagem de reivindicação nem canal aprendido. Mantido.

### 5.3 O que trocamos: o aproximador

**Algoritmo: n-tuple Actor-Critic** (Katopodis §4.3.4), com política gaussiana cuja média vem de uma rede n-tuple e crítico n-tuple para `V`. É a variante do trabalho de referência que suporta **ação contínua** e que, no benchmark Hopper, superou o Actor-Critic com MLP.

Duas saídas contínuas (`θ`, `s`), variância `σ²` como hiperparâmetro, como em Katopodis (0,5 em Mountain Car; 0,2 em Hopper).

**Compartilhamento de parâmetros** entre os UAVs homogêneos, como Lamenza faz (§5, citando Chu & Ye 2017), com o índice do agente na observação para permitir individualidade.

**Núcleo n-tuple.** Cada RAM lê `n` bits pseudo-aleatórios formando um endereço `a_i(x)`; com `φ(x)` = concatenação dos one-hots por RAM (exatamente `N` bits ligados):

```
f_k(x) = ⟨W_k, φ(x)⟩ = Σ_i W[k, i, a_i(x)]
```

Isso é **tile coding com ladrilhamento aleatório**, e portanto herda as garantias de convergência de TD com aproximação linear. Vai para a fundamentação do relatório.

- **RNF-01 (estabilidade).** `‖φ(x)‖² = N` ⇒ passo efetivo é `η·N`. Medido: `η = 0,05` com `N = 48` **divergiu para NaN**. Regra codificada com asserção: **`η = c/N`, `c ≲ 0,5`**.
- **RNF-02 (memória).** LUTs em armazenamento esparso/hash. Reportar **entradas ocupadas × bytes**, fp32 e int16 — nunca o array alocado.
- **RNF-03 (latência).** Mediana e p99 por decisão. Referência do protótipo NumPy: ~15 µs.
- **RNF-04.** Fator de esquecimento multiplicativo `φ ∈ [0,1]` (Katopodis §4.2.1), que trata a não-estacionariedade e é o que viabiliza o modelo em RL.

### 5.4 Desvios em relação ao problema original

Esta tabela é o contrato da §0. Ela tem hoje **uma linha**, e qualquer acréscimo exige decisão registrada na §10.

| # | Desvio | Por que é inevitável | Como é mitigado |
|---|---|---|---|
| **1** | **Binarização da observação** por termômetro (Katopodis §3.2.3) | Redes n-tuple endereçam RAMs com bits; não há como alimentá-las com reais | O **vetor de observação é idêntico** ao de Lamenza; o codificador é parte do aproximador, não do MDP. A resolução `T` é varrida e reportada, como Katopodis faz |

**Detalhe de codificação — `zero-padding`.** Slots vazios de `P^s` valem `(0,0)`, que sob termômetro é indistinguível de "sensor exatamente sobre o UAV". A MLP de Lamenza pode aprender a desambiguar pelo contexto; o endereçamento por RAM não. Solução: **um bit de validade por slot**, anexado ao padrão binário. É acréscimo ao codificador, não à observação.

---

## 5.5 Figuras de mérito de custo

Lamenza reporta **tempo de conclusão** e **taxa de sucesso**. Acrescentamos duas figuras que o trabalho original não mede e que são o núcleo de H1: **tamanho do modelo** e **tempo de inferência**. Não alteram o problema — são instrumentação sobre a solução, e por isso não entram na tabela de desvios da §5.4.

### Enquadramento: o que está realmente em jogo

O intervalo de decisão é **2 s**. Uma inferência de 15 µs consome 0,0008 % desse orçamento; uma MLP pequena também não chega perto de estourá-lo. Portanto a figura de mérito **não é cumprir o prazo** — dizer "roda a tempo" não distingue nada.

O que se mede, de fato, é:

1. **Que classe de hardware o modelo exige.** A alegação a defender é "roda num microcontrolador sem FPU nem GPU", não "roda a tempo".
2. **Que aritmética o modelo usa.** A WNN é estruturalmente diferente: *lookups* inteiros e somas, sem MAC em ponto flutuante. Essa diferença é robusta independentemente de quem tem menos bytes.
3. **Quanta memória precisa estar residente.**

### T1 — Tamanho do modelo

**Só conta o que é embarcado: o ator.** O crítico é artefato de treino e fica em terra — o que vale igualmente para o MADDPG de Lamenza, cujo crítico centralizado nunca voa. Reportar em duas colunas: **ator** (implantação) e **ator + crítico** (treino).

Componentes da WNN, todos obrigatórios:

| Componente | Como conta |
|---|---|
| LUTs — **alocado** | `cabeças × N × 2^n × bytes`; limite superior teórico |
| LUTs — **ocupado** | entradas efetivamente endereçadas × bytes (medido: **1,2 %** após 80k passos) |
| LUTs — **realizado** | footprint da estrutura de dados de fato usada, incluindo **chave e fator de carga do hash**. Uma tabela esparsa que guarda 2 268 valores não ocupa 2 268 × 2 bytes: guarda chave também e opera com folga. Este é o número honesto para "embarcável" |
| **Mapeamento das tuplas** | `N × n` índices de bit. **Não é desprezível:** com `N = 48`, `n = 12` são 576 índices ≈ 1,2 KB em int16, contra 4,4 KB de LUT ocupada — mais de 20 % do total. Omitir isso inflaria a vantagem |
| Codificador | limiares do termômetro, se não uniformes |

**RNF-08 — curva de ocupação.** A ocupação **cresce com o treino**. Reportar o valor final sem a curva é uma afirmação com prazo de validade. Obrigatório: gráfico de entradas ocupadas × passos de treino, com verificação de saturação. Se não saturar, a alegação de "kilobytes" precisa vir com a ressalva do horizonte medido.

**RNF-09 — quantização é ablação, não afirmação.** Reportar int16 só é legítimo se a política quantizada for avaliada. Medir tempo de conclusão em fp32 e int16; se degradar, reportar o par (tamanho, desempenho), nunca o tamanho int16 ao lado do desempenho fp32.

### T2 — Tempo de inferência

**Decomposição obrigatória**, porque o agregado esconde o interessante:

| Estágio | WNN | MLP |
|---|---|---|
| Montagem da observação (ordenar sensores por distância, `O(Ns log Ns)`) | comum | comum |
| Codificação (termômetro + bits de validade) | só WNN | — |
| Endereçamento (`N` lookups) / camadas (MACs) | lookups | MACs |
| Acumulação e amostragem da gaussiana | comum | comum |

A montagem da observação é **compartilhada por todos os métodos**, inclusive pela heurística gulosa. Reportar duas linhas: **modelo isolado** (a comparação de H1) e **ponta a ponta** (o orçamento real por decisão). Misturar as duas dilui exatamente a diferença que se quer medir.

**RNF-10 — mediana e p99.** A cauda importa mais aqui que a mediana: uma MLP densa é *branch-free* e de tempo praticamente constante; uma LUT esparsa com hash tem latência variável (falhas de cache, colisões). É perfeitamente possível que a WNN ganhe na mediana e **perca no p99** — e isso arranharia a alegação de "inferência previsível". Se acontecer, é resultado a reportar, e a mitigação é medir também a variante de LUT densa, que troca previsibilidade por memória.

**RNF-11 — comparação justa de arcabouço.** Uma MLP 31→32→32→2 executada via PyTorch mede sobretudo o *overhead* de despacho do PyTorch, não o modelo. Ou os dois lados rodam em NumPy puro, ou reporta-se **as duas colunas** (com arcabouço e sem). Caso contrário a medição não é do que se pensa que é.

**RNF-12 — contagem de operações.** Tempo de parede é dependente de implementação; contagem de operações não. Reportar, por decisão e por tipo: **MACs em ponto flutuante**, **lookups inteiros**, **somas**. É a figura de mérito independente de hardware e é onde a diferença estrutural aparece limpa.

**RNF-13 — hardware declarado.** Mesma máquina, mesmo processo, um núcleo, sem GPU para nenhum dos lados. Especificação no relatório.

### Referências de comparação — e uma ressalva que muda o critério de H1

Há **duas** referências, com ordens de grandeza diferentes, e confundi-las inflaria o resultado:

| Referência | Arquitetura | Parâmetros | fp32 |
|---|---|---|---|
| **MLP pareada** (nosso baseline, arquitetura de Katopodis §5.4) | 31→32→32→2 | ≈ 2 150 | ≈ 8,6 KB |
| **Ator do MADDPG de Lamenza** | a confirmar (P4); padrão CleanRL é 256×256 | ≈ 74 000 | ≈ 298 KB |
| **WNN (ator)** | `N=48`, `n=12`, ocupado + mapeamento | — | ≈ 5,6 KB (ocupado + tuplas), realizado a medir |

A conta preliminar diz algo desconfortável: contra a **MLP pareada**, a vantagem de tamanho é da ordem de **1,5×**, não de 10×. O fator grande só aparece contra o **ator do MADDPG**. O critério de falsificação de H1 na v2.0 deste PRD ("memória ocupada ≥10× menor") era, portanto, provavelmente inatingível contra o baseline pareado — e ficaria satisfeito trivialmente contra o MADDPG.

**Critério corrigido para H1:**

- reportar a razão de tamanho contra **ambas** as referências, sem escolher a conveniente;
- a alegação de embarcabilidade se sustenta se o ator WNN couber em **poucas dezenas de KB realizados** e for menor que o ator de referência de Lamenza;
- a alegação de custo computacional se sustenta pela **contagem de operações por tipo** — precisamente: **zero multiplicações em ponto flutuante no aproximador**. Não "zero MACs no caminho de inferência", que é falso como escrito: a acumulação usa somas em ponto flutuante e a amostragem gaussiana usa transcendentais. A alegação verificável é sobre multiplicações, e sobre o aproximador — o codificador e o amostrador entram na conta em linhas próprias.

Se a WNN acabar **maior** que a MLP pareada, isso é resultado — e o argumento do trabalho migra de "menor" para "aritmética mais barata e sem FPU", que é a alegação defensável.

**Opcional (não no caminho crítico):** medir num alvo restrito — Raspberry Pi ou perfil só-inteiro — para converter o argumento de classe de hardware em número. Fica como *stretch*.

---

## 6. Plano experimental

### 6.1 Etapa 1 — rede sem peso no problema original, sem ruído

Segue a mesma sequência iterativa que Lamenza usou, o que dá alvos numéricos exatos para comparação.

**1a — Réplica do experimento exploratório (Lamenza §7.1).** `N*_a = 0`, recompensa 7-2, `max_episode_length` permissivo, métrica = **taxa de sucesso** sobre 500 episódios. Alvo publicado: **100 %**. Serve para demonstrar que o n-tuple Actor-Critic e o laço de treino aprendem alguma coisa, antes de gastar tempo em cenário difícil. É a mesma função que teve no trabalho original.

**1b — Experimento de tempo de conclusão (Lamenza §7.2).** Recompensa 5-6, observação completa, métrica = **tempo de conclusão** médio sobre 500 episódios, comparado à heurística gulosa (vizinho não visitado mais próximo) — o baseline do próprio Lamenza. Percorrer `Na = 1 → 2 → 4 → 8` com `Ns = 12`, depois `Ns ∈ {24, 36}` conforme o tempo.

**Baselines:**

| # | Baseline | Papel |
|---|---|---|
| 1 | caminhada aleatória | piso |
| 2 | **guloso pelo sensor não visitado mais próximo** | o baseline de Lamenza — é contra este que H6 se mede |
| 3 | k-means + TSP por UAV, offline, com posições verdadeiras | teto quase-ótimo (acréscimo nosso, não altera o problema) |

**Comparação pareada (H1):** mesmo ambiente, mesma observação, mesma recompensa, **mesmo algoritmo** (Actor-Critic on-policy com parameter sharing), trocando só o aproximador — n-tuple vs MLP de 2 camadas × 32 unidades, a arquitetura que Katopodis usou nos baselines (§5.4).

**Varreduras:** resolução do termômetro `T ∈ {8, 32, 128}` × tamanho da tupla `n ∈ {8, 10, 12}` × número de RAMs `N ∈ {16, 48, 128}` × esquema de codificação (simples vs circular) × fator de esquecimento. Medição preliminar indica retorno decrescente em `N` (8× mais memória para 13 % menos erro): a varredura documenta o joelho da curva, não maximiza.

**Orçamento de simulação.** Medido: `update_rate = 0,1` dá 0,11 s por episódio de 120 s simulados com 4 UAVs / 12 sensores, idêntico ao default de 0,01. Extrapolando para episódios de até 500 s e 50 M passos (o orçamento de Lamenza), estima-se **8–30 h num único núcleo**, contra 7 dias de GPU. **Estimativa, não medição** — confirmar na semana 1 (P2). Se confirmada, é um resultado a reportar por si só.

> **Portão G1** — 1a atinge taxa de sucesso comparável aos 100 % publicados; 1b supera a caminhada aleatória decisivamente e fica dentro do IC da heurística gulosa; tabela pareada WNN vs MLP fechada com tempo, memória ocupada e latência.
> *Se G1 não for atingido:* é a refutação de H1 e o principal resultado da etapa. Registrar o veredito, diagnosticar (resolução? tamanho de tupla? variância da gaussiana?) e prosseguir — H2 a H4 medem degradação **relativa** ao próprio desempenho sem ruído e permanecem válidas mesmo com política medíocre.

### 6.2 Etapa 2 — extensão: medição ruidosa e filtro de Kalman, 1 UAV

Aqui, e só aqui, o problema é estendido. O capítulo do relatório é separado e a extensão é declarada.

**Onde o ruído entra.** Em Lamenza o "canal global" entrega o estado verdadeiro. Substituímos por medição corrompida em dois canais:

| Canal | Modelo | Natureza | Filtro |
|---|---|---|---|
| Posição própria | GPS, σ ≈ 1–10 m | ruído branco, **entrada de controle conhecida** | KF linear 2 estados `[x, y]`, predição `x̂ + u·dt` |
| Posição dos sensores | descoberta por enlace local | estática; linear se o sensor transmite sua posição estimada | **filtro de informação** (média ponderada recursiva pela precisão) |

**Escada de ablação:**

| # | Configuração | Papel |
|---|---|---|
| 0 | sem ruído — é a política da Etapa 1 | teto |
| 1 | ruído bruto, sem filtro | piso |
| 2 | média móvel / passa-baixa | filtro barato — o KF precisa bater isto |
| 3 | KF velocidade constante | o "óbvio", que já sabemos que falha |
| 4 | **KF + entrada de controle** | a proposta |
| 5 | 4 + covariância na observação | política consciente da incerteza |

O degrau 5 acrescenta canais à observação e portanto **entra na tabela de desvios (§5.4) quando executado** — é ablação explícita, com resultado reportado com e sem.

Cruzado com o regime de treino: **treino limpo → avaliação ruidosa** vs **treino ruidoso → avaliação ruidosa**. Se o degrau 4 tornar os dois equivalentes, é o argumento mais forte possível a favor do filtro.

**Varredura conjunta** (não uma de cada vez, pela regra da §1): `σ_GPS ∈ {1, 3, 5, 10} m` × `T ∈ {8, 32, 128}`. É o experimento mais caro do projeto e roda muito mais barato com 1 UAV — razão de ele vir antes do multiagente.

**RF-04 (invariante de vazamento).** A verdade **nunca** chega à política. `handle_telemetry` do gradysim entrega posição exata; a corrupção acontece na entrada do protocolo e a verdade fica guardada só para métricas. Teste automatizado obrigatório.

**RF-05 (consistência).** NEES/NIS contra o intervalo qui-quadrado **antes** de olhar qualquer resultado de RL. Um KF mal sintonizado produz números plausíveis que envenenam a análise a jusante — foi literalmente o que ocorreu no experimento preliminar.

**RNF-05 (ressalva de custo).** O KF faz MACs, o que arranha a narrativa "sem MACs, embarcável". Medir, não esconder: ~50–80 flops/passo do KF contra os **2 080 MACs** da MLP pareada `31→32→32→2` (992 + 1 024 + 64), ou seja ~3 %. Vai como número na tabela de custo. *(Corrigido: a versão anterior citava ~2 600 MACs de uma "MLP 31→32→2" — o número vinha da MLP 33→64→8 do `PLANO.md` §9.8 e a arquitetura citada não era a pareada. O KF acrescenta ainda **divisões**, que a contagem de operações da §5.5 precisa tratar como categoria própria: num MCU sem FPU, uma divisão custa mais que um MAC.)*

> **Portão G2** — NEES/NIS no intervalo; degrau 4 indistinguível do degrau 0; degrau 4 supera os degraus 2 e 3 com IC separado.

### 6.3 Etapa 3 — extensão: multiagente com ruído

Única etapa em que duas dificuldades entram juntas, por necessidade: só há erro correlacionado a fundir se houver mais de um estimador.

**Canal novo:** posição dos vizinhos, recebida por broadcast — ruidosa, **atrasada** e com erros **correlacionados** (dois UAVs que observaram o mesmo sensor, possivelmente com histórico compartilhado).

**Sub-passos:**

- **3a — multiagente sem ruído.** Degrau 0 da etapa e referência contra a qual toda degradação por ruído é medida. Sem ele, não se separa perda por coordenação de perda por incerteza. Não negociável.
- **3b — multiagente com ruído**, degraus 1, 4 e 6.

**Degrau 6:**

| # | Configuração | Papel |
|---|---|---|
| 6a | KF local isolado por UAV, sem troca de estimativas | referência |
| 6b | fusão por KF ingênuo entre UAVs | deve **divergir** com covariância otimista — é o espantalho que justifica o método |
| 6c | **interseção de covariância** | a proposta, conservadora sob correlação desconhecida |

> **Portão G3** — 6c supera 6a em erro de estimativa com NEES/NIS consistente; 6b demonstravelmente inconsistente; vereditos escritos para H5 e H6.

### 6.4 Métricas

**De Lamenza (desempenho):**

- **tempo de conclusão** do episódio (média sobre 500 episódios de avaliação, IC bootstrap 95 % com 1000 reamostragens, como em Katopodis §5.4)
- **taxa de sucesso**: fração de episódios com `Σ visited_i^T = Ns`

**Nossas (custo) — protocolo completo em §5.5:**

- **T1 tamanho do modelo**: ator e ator+crítico; alocado / ocupado / **realizado**; incluindo o mapeamento das tuplas; curva de ocupação × passos; fp32 e int16 com a quantização avaliada
- **T2 tempo de inferência**: modelo isolado e ponta a ponta; mediana e **p99**; decomposto por estágio; com e sem arcabouço
- **T3 contagem de operações** por decisão e por tipo: MACs em ponto flutuante, lookups inteiros, somas

**De fusão (Etapas 2–3):**

- **erro de estimativa** (m) e **consistência NEES/NIS** por canal
- T1/T2 **recalculados com o filtro embutido** — o KF é código que voa no UAV e conta no orçamento (RNF-05)

**De processo:**

- tempo de parede de treino, contra os 7 dias de GPU por célula do original

**RNF-06.** Nenhum resultado agregado sem IC. Nenhuma conclusão de RL antes do NEES/NIS passar. Nenhum número de tamanho ou latência sem a configuração (`T`, `n`, `N`, `Na`) que o gerou.

---

## 7. Arquitetura do código

```
uav-wnn/
├── pyproject.toml
├── Makefile                       # make repro
├── configs/
│   ├── etapa1/                    # problema original, varreduras T x n x N
│   ├── etapa2/                    # + ruído e KF, 1 UAV, grade sigma x T
│   └── etapa3/                    # + multiagente, degraus 6a/6b/6c
├── src/uavwnn/
│   ├── encoding/
│   │   ├── thermometer.py         # simples e circular (Katopodis §3.2.3)
│   │   └── observation.py         # obs de Lamenza (eq. 5-1) -> bits, com bit de validade
│   ├── wnn/
│   │   ├── ntuple.py              # PRONTO — gradiente funcional + esquecimento
│   │   └── sparse.py              # armazenamento hash/esparso das LUTs
│   ├── agents/
│   │   ├── ntuple_ac.py           # Actor-Critic n-tuple, ação contínua (Katopodis §4.3.4)
│   │   ├── mlp_ac.py              # ABLAÇÃO: mesmo algoritmo, aproximador denso 2x32
│   │   └── heuristics.py          # random, greedy, kmeans+TSP
│   ├── env/
│   │   ├── scenario.py            # geração de cenário, semeadura
│   │   ├── protocols.py           # SensorProtocol / UAVProtocol (IProtocol)
│   │   ├── bridge.py              # ponte simulador <-> treinador
│   │   └── rewards.py             # eq. 5-6 e eq. 7-2, verbatim
│   ├── fusion/                    # ETAPA 2+ — extensão, isolada do núcleo
│   │   ├── noise.py               # GPS, enlace local, atraso de broadcast
│   │   ├── kalman.py              # KF com entrada de controle; filtro de informação
│   │   ├── ci.py                  # interseção de covariância
│   │   └── consistency.py         # NEES / NIS
│   ├── train/runner.py
│   └── eval/                      # metrics.py, sweep.py, plots.py
└── tests/
```

**RNF-07.** `fusion/` é um pacote separado e opcional. Com ele desligado, o sistema executa o problema de Lamenza puro. Isso é o que torna o princípio da §0 verificável por construção, e não por promessa.

### A ponte simulador ↔ treinador

O `gradysim` é **orientado a eventos**: `step_simulation()` processa **um evento**, não um instante. O RL quer `obs = env.step(action)`. É o pedaço de engenharia não trivial do projeto.

**Injeção de política.** O `UAVProtocol` guarda referência a um objeto `Policy` e, no `handle_timer("decide")`, chama `policy.act(obs)` inline. Sem corrotinas, funciona para métodos **on-policy** — que é justamente o caso do n-tuple Actor-Critic. O treino inteiro cabe em `simulation.start_simulation()`.

A alternativa (estado compartilhado + inversão de controle, dando API PettingZoo de verdade) fica registrada e só é implementada se o baseline MLP exigir.

---

## 8. Cronograma — 4 semanas

A execução é decomposta em specs pequenas, com escopo fechado e critério de saída medido: ver **[`specs/README.md`](specs/README.md)**. A primeira é o **[estudo de binarização (SPEC-01)](specs/SPEC-01-binarizacao.md)**, que roda sem simulador e sem laço de RL, e portanto em paralelo com a montagem do ambiente.

| Semana | Etapa | Entrega | Portão / critério de pronto |
|---|---|---|---|
| **1** | 1 (infra) | **SPEC-01** (binarização, portão G0) em paralelo com **SPEC-00** (cenário, protocolos, ponte, `rewards.py` verbatim, baselines, harness de custo) | **G0**: codificador escolhido com evidência; P1 e P2 resolvidas; asserção `η = c/N` ativa; orçamento de simulação medido |
| **2** | 1 (aprendizado) | n-tuple Actor-Critic em 1b, `Na = 1 → 2 → 4`; ablação MLP pareada; varreduras `T × n × N`; tabela de memória e latência | **G1** |
| **3** | 2 | Ruído (GPS + enlace local); KF com entrada de controle; filtro de informação; NEES/NIS; degraus 0–5; varredura σ × T | **G2** |
| **4** | 3 + fechamento | Multiagente 3a e 3b; degraus 6a/6b/6c; interseção de covariância; relatório e slides | **G3**; `make repro` verde; E2 e E3 |

A redação **não** espera a semana 4: cada tabela entra no relatório assim que sai.

### Linha de corte (o que cai primeiro se atrasar)

1. Varreduras de `n × N` → fixar `n = 12`, `N = 48` (já testados no protótipo) e varrer só `T`
2. Células `Ns ∈ {24, 36}` e `Na = 8` na Etapa 1
3. Esquema de codificação circular (manter só o termômetro simples)
4. Varredura σ × T → reduzir a `σ ∈ {1, 10}` × `T ∈ {8, 128}`
5. Degraus 2 e 3 da escada — reter ao menos um como referência barata
6. Etapa 3 com `Na = 2` em vez de 4, menos sementes

**Não podem cair:** os três portões com veredito escrito; réplica 1a; baseline guloso; degraus 0, 1, 4 e 6c; NEES/NIS; sub-passo 3a; **a tabela de custo da §5.5 (T1/T2/T3)**; RF-04 (invariante de vazamento).

A tabela de custo é barata de produzir — instrumentação, não experimento — e é a única contribuição que sobrevive mesmo se todos os resultados de desempenho forem medianos. Cortá-la seria cortar o trabalho.

**Regra de escalonamento:** portão não atingido ao fim da semana aciona a linha de corte **da etapa seguinte** — nunca estende a etapa atual para dentro dela. Chegar às três etapas com resultados modestos vale mais, nesta disciplina, do que uma Etapa 1 exemplar e nenhuma fusão.

---

## 9. Riscos

| Risco | Mitigação |
|---|---|
| **Ação contínua com n-tuple é menos madura que SARSA discreto** | Katopodis validou n-tuple Actor-Critic em Mountain Car e Hopper, ambos contínuos, superando o AC com MLP. Sintonizar `σ²`, `T` e `n` antes de qualquer outra coisa. Discretizar seria **modificar o problema** e só é admissível como desvio declarado na §5.4, com aprovação |
| Zero-padding confundido com sensor sobre o UAV | bit de validade por slot no codificador |
| **A WNN não é menor que a MLP pareada** (conta preliminar: ~5,6 KB vs ~8,6 KB, só 1,5×) | §5.5: reportar contra as duas referências sem escolher a conveniente; migrar o argumento para contagem de operações e ausência de FPU, que é onde a diferença é estrutural |
| **A WNN perde no p99** por causa do hash | medir também a variante de LUT densa (troca previsibilidade por memória) e reportar o par |
| Medição de latência da MLP contaminada por overhead de arcabouço | RNF-11: NumPy puro dos dois lados, ou duas colunas |
| Ocupação das LUTs não satura e "kilobytes" vira alegação com prazo | RNF-08: curva de ocupação obrigatória, com o horizonte declarado |
| `2^n` explode com tuplas grandes | armazenamento por hash; varrer `n ∈ {8, 10, 12}` |
| Interferência catastrófica no aprendizado | fator de esquecimento (que também apaga — varrer) |
| Sem sinalização, UAVs convergem para o mesmo alvo | é comportamento do problema original (Lamenza fig. 7.1), não defeito nosso. Reportar, não corrigir |
| KF mal sintonizado passa despercebido | **NEES/NIS obrigatório antes de qualquer conclusão de RL** |
| `Q` do KF depende de o comando ser cumprido | mobilidade com inércia na Etapa 2: o descasamento vira `Q` mensurável |
| Fusão de estimativas correlacionadas diverge | interseção de covariância, não KF ingênuo |
| Orçamento de simulação estimado, não medido | medir na semana 1; se estourar, subir `update_rate` validando fidelidade contra 0,01 |
| 4 semanas é apertado | a linha de corte da §8 é para ser usada |

---

## 10. Decisões

| ID | Decisão | Razão |
|---|---|---|
| **D1** | **O problema de Lamenza não é modificado.** Observação, ação, recompensa, cenário e métricas verbatim | É o que torna a comparação com os números publicados interpretável, e o que a §0 protege |
| **D2** | **Ação contínua** `⟨θ, s⟩`, não discretizada | Reverte o v1 deste PRD. A continuidade é ponto central da crítica de Lamenza à literatura e coluna da tabela comparativa dele |
| **D3** | Recompensa **eq. 5-6 verbatim**: sem shaping potencial, sem penalidade de borda | Reverte o v1. Shaping alteraria o sinal de aprendizado e a comparação |
| **D4** | Observação **eq. 5-1 verbatim**: coordenadas relativas, `N*_s = 12`, zero-padding, índice do agente | Reverte o v1, que propunha mapa setorial egocêntrico. Fica como trabalho futuro (§12) |
| **D5** | **Sem mensagem de reivindicação** entre UAVs | Não existe no original; a coordenação de Lamenza emerge só de conhecer posições |
| **D6** | Algoritmo: **n-tuple Actor-Critic** com política gaussiana e parameter sharing | Única variante do trabalho de referência que suporta ação contínua; validada em Hopper |
| **D7** | **MADDPG não é reproduzido**; entra como referência de literatura | Reproduzi-lo exigiria replicar exatamente simulador, observação e recompensa; qualquer divergência invalidaria a comparação silenciosamente. Nosso pareamento é n-tuple AC vs MLP AC |
| **D8** | **Reprodução antes de extensão**; `fusion/` é pacote opcional e desligável | Sem uma linha de base fiel, nenhum efeito de filtro é atribuível |
| **D9** | **Fusão local (Etapa 2) antes do multiagente (Etapa 3)** | A varredura σ × T é o experimento mais caro e roda muito mais barato com 1 UAV. E a regra *bin ≈ σ* precisa estar fixada antes de o multiagente entrar |
| **D10** | Etapa 3 tem sub-passo **3a sem ruído** obrigatório | Sem ele, degradação por coordenação e por incerteza ficam confundidas |
| **D11** | **Tamanho do modelo e tempo de inferência são figuras de mérito de primeira classe**, medidas em toda etapa e reportadas junto com tempo de conclusão e taxa de sucesso | São o conteúdo de H1 e a única contribuição que independe de a política ser boa: mesmo com desempenho apenas comparável, o par (tamanho, aritmética) é resultado. Protocolo em §5.5 |
| **D12** | O tamanho reportado é o **realizado**, com mapeamento de tuplas contabilizado, e a razão é dada contra **ambas** as referências | Reportar só a LUT ocupada contra o ator do MADDPG seria escolher o numerador e o denominador convenientes |

### Pendências

- **P1** — Confirmar `communication_range` no repositório `Project-GrADyS/gradys-data-collection` (branch `dissertation`). Não consta nas tabelas 7.1/7.2. **Bloqueia a comparação numérica com os resultados publicados**, então é a primeira tarefa da semana 1.
- **P2** — Medir o orçamento real de simulação e validar `update_rate = 0,1` contra 0,01 com 8 UAVs.
- **P4** — Extrair do repositório público a **arquitetura do ator do MADDPG** (largura e profundidade das camadas). Sem ela, a razão de tamanho contra a referência de Lamenza não passa de estimativa (§5.5). O padrão do CleanRL, base declarada da implementação dele, é 256×256.
- **P3** — Definir se o sensor transmite sua posição estimada (caso linear → filtro de informação) ou apenas distância/RSSI (caso não linear → EKF/UKF e trilateração). **Recomendação:** caso linear; o não linear é um projeto inteiro e fica como trabalho futuro.

---

## 11. Definition of done

| Etapa | Pronta quando |
|---|---|
| **1** | Réplica 1a com taxa de sucesso reportada contra os 100 % publicados; 1b com tempo de conclusão vs guloso, em pelo menos `Na ∈ {1, 2, 4}`; **tabela de custo completa da §5.5** (T1 alocado/ocupado/realizado com mapeamento de tuplas, curva de ocupação, T2 mediana e p99 decomposto, T3 contagem de operações), contra as duas referências; veredito para H1 |
| **2** | NEES/NIS de cada canal no intervalo qui-quadrado; degraus 0–5 com IC nos dois regimes de treino; varredura σ × T reportada; vereditos para H2, H3, H4 |
| **3** | 3a medido como referência; degraus 1, 4, 6a/6b/6c medidos; vereditos para H5 e H6 |

**Global:**

1. `make repro` regenera todas as tabelas e figuras a partir de seeds fixas.
2. Testes passam, incluindo os invariantes: **a verdade não vaza para a política** (RF-04) e **com `fusion/` desligado, o sistema executa o problema de Lamenza puro** (RNF-07).
3. A tabela de desvios (§5.4) reflete exatamente o que foi executado.
4. A tabela de custo segue o protocolo da §5.5 integralmente: tamanho **realizado** (não só ocupado), mapeamento de tuplas contabilizado, p99 além da mediana, contagem de operações por tipo, e — nas Etapas 2–3 — T1/T2 recalculados com o filtro embutido.
5. As razões de tamanho e latência são reportadas contra **ambas** as referências (MLP pareada e ator do MADDPG), com a configuração `T, n, N, Na` que as gerou.
6. Cada hipótese H1–H6 tem veredito escrito — confirmada, refutada ou inconclusiva — **com o número que o sustenta**. Hipótese refutada é resultado, não fracasso.
7. E2 e E3 entregues.

---

## 12. Trabalho futuro (ideias descartadas por modificarem o problema)

Registradas para o capítulo de trabalhos futuros do relatório, **não aplicadas**:

- **Mapa setorial egocêntrico** como observação — daria tamanho fixo independente de `Na`, invariância a permutação e transferência entre células da grade. É atraente e é uma mudança de MDP.
- **Ação discretizada** em `K` rumos — casaria com n-tuple SARSA, a variante mais madura de Katopodis, mas contraria a premissa de espaço contínuo de Lamenza.
- **Shaping potencial** (Ng et al., 1999) sobre a distância ao sensor livre mais próximo — não altera a política ótima, mas altera o sinal e a comparação.
- **Mensagem de reivindicação de setor** entre UAVs — coordenação explícita barata; muda o problema de coordenação emergente para coordenação assistida.
- **Generalização a `Na` variável** — Lamenza tentou e falhou (§7.4.2); requer arquitetura própria.

---

## Referências

- LAMENZA, T. S. *A Deep Reinforcement Learning Approach for Sensor Data Collection by a Swarm of UAVs*. Dissertação de mestrado, PUC-Rio, 2025.
- KATOPODIS, R. F. *Reinforcement Learning with Weightless Neural Networks*. Dissertação de mestrado, COPPE/UFRJ, 2022.
- LOWE, R. et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), 2017.
- CHU, X.; YE, H. *Parameter Sharing Deep Deterministic Policy Gradient for Cooperative Multi-agent Reinforcement Learning*, 2017.
- JULIER, S.; UHLMANN, J. *A non-divergent estimation algorithm in the presence of unknown correlations* (interseção de covariância), 1997.
- LAMENZA, T. et al. *GrADyS-SIM NextGen*, 2024.
