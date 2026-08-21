# PRD — Coleta de dados por enxame de UAVs: rede sem peso e, depois, filtro de Kalman

**Projeto:** `uav-wnn`
**Disciplina:** Fusão de Dados — PESC/COPPE-UFRJ
**Autor:** Raphael Nagao Ferreira · rnagao@cos.ufrj.br
**Data:** 09/08/2026 · **Prazo:** 4 semanas (entrega em ~06/09/2026)
**Status:** em execução · **Versão:** 5.0 — *escopo do ruído reduzido ao **GPS do drone**; o canal do sensor sai, e com ele o bloco 2b, o filtro de informação, a H4, a T3 e a P3* · 4.0 — *a fonte primária passa a ser a **dissertação** de Katopodis, não código: RNF-01 corrigida, D6 corrigida, e a fundamentação de núcleo deixa de ser analogia e passa a ser identidade testada (ver [`specs/PLANO-TESTES.md`](specs/PLANO-TESTES.md))*

---

## 0. Princípio que rege este documento

> **O problema formulado por Lamenza (2025) não é modificado.** Cenário, observação, ação, recompensa, métricas e parâmetros são reproduzidos como estão. A única coisa que trocamos é o **aproximador de função**: rede n-tuple sem peso no lugar das redes profundas.

**Isto deixou de ser promessa e virou construção.** O ambiente **é** o `GrADySEnvironment` de Lamenza, carregado verbatim de um submodule fixado na tag `dissertation`. Não há reimplementação do MDP. Como as features do trabalho se encaixam sem tocar naquele código está em [`specs/ARQUITETURA.md`](specs/ARQUITETURA.md).

A decisão veio de uma medição: a reimplementação anterior era provadamente equivalente (`max|Δobs| = 7,9×10⁻⁸`), mas uma "correção" de **uma linha** — inicializar a posição do drone no spawn em vez de `(0,0,0)`, que é o que o original faz — levava a divergência a `8,5×10⁻¹` por efeito caótico. Ver `reports/spec00-parametros.md`.

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
| A1 | ~~Ocupação real das LUTs é minúscula~~ **REFUTADO (SPEC-03)** | O protótipo media 2 268 entradas → ~4,4 KB, mas tinha **2 variáveis de entrada**; a observação real tem **25**. Medido no problema real: **13 555 entradas → 133,9 KB**. Extrapolar de um brinquedo 2-D para 25-D foi a origem do erro. Ver `reports/spec03-custo.md` |
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
| **2** | 1 UAV, ruído no **GPS do drone** | + medição ruidosa e KF com entrada de controle | H2, H3 |
| **3** | `Na` UAVs, com ruído | + coordenação e estimativas correlacionadas | H5, H6 |

A Etapa 1 é a **reprodução**. As Etapas 2 e 3 são a **extensão de Fusão de Dados**.

`Na = 1` não é uma modificação do problema: `Na` é parâmetro do cenário (tab. 3.1 de Lamenza), e o próprio autor roda experimentos exploratórios com `N*_a = 0`, isto é, agentes cegos aos demais. A Etapa 1 percorre `Na = 1 → 2 → 4 → 8` dentro da mesma formulação.

Cada etapa tem um **portão** (`G1`, `G2`, `G3`): critério objetivo a ser atingido — ou explicitamente dispensado por decisão registrada — antes de a seguinte começar. Portão não atingido é resultado publicável, não motivo para parar.

### Hipóteses

| ID | Etapa | Hipótese | Como é falsificada |
|---|---|---|---|
| **H1** | 1 | Sobre o problema **inalterado**, a política n-tuple atinge desempenho comparável ao de uma rede densa, a custo de **modelo** e **inferência** materialmente menor | Comparação pareada: mesmo ambiente, mesma observação, mesma recompensa, **mesmo algoritmo**, trocando só o aproximador. Critério detalhado em **§5.5** — H1 cai se o tempo de conclusão da WNN for pior além do IC, ou se o ator não couber em poucas dezenas de KB **realizados**, ou se a contagem de operações não mostrar o aproximador livre de **multiplicações em ponto flutuante**. **MEDIDO (SPEC-03), e H1 cai na parte de tamanho:** o ator WNN realizado é **133,9 KB** contra **7,6 KB** da MLP pareada — **17,5×** —, e ainda 5× menos preciso na clonagem. A estimativa anterior de 12–15 KB errou uma ordem de grandeza porque vinha de um protótipo com 2 variáveis de entrada. Sobrevive, medido: **zero multiplicações em ponto flutuante no aproximador** e vantagem de ciclos condicional ao alvo (1,6× em x86, **19,9× em Cortex-M0 sem FPU**). Ver `reports/spec03-custo.md` |
| **H2** | 2a | A binarização por termômetro **não** torna a WNN mais frágil a ruído de medição que uma rede densa | Curvas de degradação WNN vs MLP sobre σ crescente. H2 cai se os ICs separarem |
| **H3** | 2a | O KF com a **ação da política como entrada de controle** devolve a rede ao regime praticamente sem ruído | Escada de ablação 1→4. H3 cai se a degradação com filtro for estatisticamente distinguível do oráculo |
| ~~**H4**~~ | — | ~~Uma política que enxerga a própria covariância aprende busca ativa~~ **RETIRADA pelo corte de escopo (D13)** | O mecanismo dela era o canal do sensor: relatórios ficam mais precisos de perto, logo aproximar-se **compra informação** e a busca ativa tem causa física. Com ruído só no GPS do drone, a covariância própria converge para regime estacionário e **não há incerteza por sensor a reduzir** — a hipótese perde a causa, não só o experimento. O degrau 5 da escada (política que enxerga o próprio `P`) continua como **ablação sem hipótese forte associada**. Retomar a H4 exigiria devolver o canal do sensor ao escopo |
| **H5** | 3 | Fundir estimativas entre UAVs por **interseção de covariância** supera o KF local isolado e não diverge, ao contrário do KF ingênuo | Erro de estimativa e NEES/NIS. H5 cai se CI não bater o KF local, ou se o KF ingênuo não divergir |
| **H6** | 3 | A coordenação emerge também com aproximador sem peso | **Barra revista.** Superar a `greedy` é barra baixa: medimos que ela não coordena — separação média entre UAVs de **0,1 m** contra alcance de 20 m, quatro drones em formação. A barra honesta é a **`smart`** do próprio `heuristics.py`, que coordena lendo só a observação e fica a **5–17 % do MADDPG publicado**, superando-o com 2 agentes. H6 divide-se: **H6a** supera `greedy` (provável), **H6b** supera `smart` (incerto). Ver `reports/spec00-parametros.md` §9 |

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

> **Correções que o código de referência impôs a esta seção.** O texto da eq. 5-1 descreve coordenadas relativas cruas; a implementação que gerou os resultados **normaliza** por `(p_k − p_i + S)/(2S)` e usa padding **`-1`**, não `(0,0)`. A faixa real dos valores é `[-0,5; 1,5]`, e não `[0,1]` — o `observation_space` de referência declara `Box(-1,1)`, o que também não bate. O índice do agente entra como `k/Na`. Nada disso é escolha nossa: é o que o ambiente entrega, e a SPEC-01 precisa binarizar a faixa **medida**. Ver `reports/spec00-parametros.md` §6.

### 5.3 O que trocamos: o aproximador

**Algoritmo: n-tuple Actor-Critic** — **Algoritmo 4.4** da dissertação, p. 48. É a variante do trabalho de referência que suporta **ação contínua** e que, no benchmark Hopper, superou o Actor-Critic com MLP.

**Hierarquia de fontes.** A fonte primária é a **dissertação** (`tmp/947841_compressed.pdf`), não código. Os algoritmos dela são 4.1 (SARSA com esquecimento), 4.3 (REINFORCE) e 4.4 (Actor-Critic); **PPO aparece uma única vez em 87 páginas**, no capítulo 6 de trabalhos futuros (p. 70), como sugestão para corrigir a instabilidade de passo fixo do SGD. Logo:

| repositório | o que é |
|---|---|
| `rkatopodis/RLWorktable.jl` + `Ramnet.jl` (`faster_hashing`) | **a dissertação implementada** — Algoritmo 4.4 literal |
| `anbubus/ntn_neurocomputing_v0` (PyTorch + Ray RLlib PPO) | **o capítulo 6** — o trabalho futuro, não o trabalho |

Duas saídas contínuas (`θ`, `s`), com **covariância `Σ` predefinida** — a eq. (4.34) é explícita: *"Σ is a **predefined** covariance matrix"*, e o apêndice A a lista como hiperparâmetro constante (0,5 em Mountain Car; **0,3** em Hopper). Nas tarefas multivariadas o artigo usa **identidade escalada** (§5.1.3). Ver **P13** sobre a divergência entre o artigo e o código do autor quanto a `Σ` ser covariância ou desvio.

**Compartilhamento de parâmetros** entre os UAVs homogêneos, como Lamenza faz (§5, citando Chu & Ye 2017), com o índice do agente na observação para permitir individualidade.

**Núcleo n-tuple — e agora com a fundamentação verificada, não presumida.** Cada RAM lê `n` bits formando um endereço `τ[i](x)`. A saída é a **média** sobre as RAMs, eq. (4.12):

```
f(x) = (1/N) Σ_i v[i](x)
```

A versão anterior deste PRD afirmava que isso "é tile coding com ladrilhamento aleatório, e portanto herda as garantias de convergência de TD com aproximação linear". A afirmação está certa, mas era **presumida por analogia**. A dissertação a **prova**, e agora nós a **testamos** (`tests/test_identidades_artigo.py`, 21 testes):

| eq. | o que prova | teste |
|---|---|---|
| (3.22) | a rede de regressão **é** um estimador de Nadaraya-Watson | `test_rede_de_regressao_e_estimador_de_nadaraya_watson` |
| (4.16) | a rede de gradiente funcional **é** máquina de núcleo: `f(x) = Σ_j α_j k(x,x_j)` | `test_rede_de_gradiente_funcional_e_maquina_de_nucleo` |
| (4.19) | a atualização **é** um passo de gradiente funcional: `Δf(x) = αΔ·k(x_t,x)` | `test_uma_atualizacao_propaga_pelo_nucleo` |
| (4.23) | o núcleo `k = 1 − δ/N` **é positivo-definido**, via mapa explícito `φ` | `test_matriz_de_gram_e_positiva_semidefinida` |

A (4.23) é a que sustenta as outras: o texto assinala que a passagem de (4.18) para (4.19) *"hinges on the kernel being positive-definite"*. Verificado no espectro — matriz de Gram 60×60, autovalor mínimo `> −1e−9`.

- **RNF-01 (estabilidade). CORRIGIDA — a regra anterior estava errada.** A versão anterior dizia `‖φ(x)‖² = N ⇒ passo efetivo η·N`, e prescrevia `η = c/N`. Isso vale para a convenção de **soma**; a eq. (4.12) do artigo usa a **média**, e a eq. (4.22) normaliza o mapa por `1/√N`, de modo que `‖φ(x)‖² = 1`. Com a média e `Δ = −η(f − y)`, a variação da saída num passo é **exatamente `−η(f − y)`, independente de `N`** — testado em `test_ator_usa_media_e_o_passo_independe_de_N`. **Não há regra `η = c/N`.** O NaN observado com `η = 0,05` e `N = 48` veio do protótipo em convenção de soma, não do modelo do artigo.
  Isto é o que **permite varrer `N` sem confundir capacidade com passo** — e é a razão de o `EmbeddingBag(mode='sum')` do repositório em PyTorch divergir do artigo nesse ponto.
- **RNF-01b (instabilidade de passo fixo). MEDIDO.** O que de fato limita `α` não é `N`, é a variância do gradiente de política. No bandido contextual, o pior erro entre 5 sementes cresce **monotonicamente** com `α` — 0,044 em `α = 0,002` contra 0,238 em `α = 0,05`, com uma semente divergindo em 0,05. É a limitação que a própria dissertação nomeia no capítulo 6 ao sugerir TRPO/PPO. **A busca de `α` da Etapa 1 precisa de faixa larga e múltiplas sementes.**
- **RNF-02 (memória).** LUTs em armazenamento esparso/hash. Reportar **entradas ocupadas × bytes**, fp32 e int16 — nunca o array alocado.
- **RNF-03 (latência).** Mediana e p99 por decisão. Referência do protótipo NumPy: ~15 µs.
- **RNF-04.** Fator de esquecimento `γ ∈ [0,1]` (Katopodis §4.2.1) **no crítico, não no ator**. O código do autor (`rkatopodis/Ramnet.jl`, branch `faster_hashing`) usa dois modelos distintos: `RegressionDiscriminator` com γ e contador para o crítico, `FunctionalDiscriminator` sem γ e sem contador para o ator. O esquecimento existe porque o crítico faz **regressão para o retorno**, cujo alvo se desloca com a política — no ator não há alvo se movendo assim. Consequência de custo: o crítico guarda **valor e contador**, dobrando os bytes por entrada.

### 5.4 Desvios em relação ao problema original

Esta tabela é o contrato da §0. Ela tem hoje **uma linha**, e qualquer acréscimo exige decisão registrada na §10.

| # | Desvio | Por que é inevitável | Como é mitigado |
|---|---|---|---|
| **1** | **Binarização da observação** por termômetro (Katopodis §3.2.3) | Redes n-tuple endereçam RAMs com bits; não há como alimentá-las com reais | O **vetor de observação é idêntico** ao de Lamenza; o codificador é parte do aproximador, não do MDP. A resolução `T` é varrida e reportada, como Katopodis faz |

**O nosso codificador também difere do de Katopodis, e isso não é desvio do problema de Lamenza.** A eq. (3.23) da dissertação usa limiares **uniformes** sobre `[l, u]` e comparação estrita; a SPEC-01 mediu que **quantis por slot** vencem por 4,3° contra 7,6°, porque a distribuição depende do slot (a ordenação por distância gera estatísticas de ordem). Os codificadores literais do artigo — termômetro (3.23) e circular (3.24) — estão em `src/uavwnn/encoding/artigo.py` como referência de fidelidade, com testes. Como o codificador é parte do aproximador e não do MDP, isto **não** acrescenta linha a esta tabela — mas precisa aparecer no relatório, porque é um ponto em que deliberadamente **não** somos fiéis à dissertação.

**Detalhe de codificação — `padding`. RESOLVIDO, e o desvio caiu.** A spec projetava um bit de validade porque supunha padding em `(0,0)`, indistinguível de "sensor exatamente sobre o UAV". O código de referência usa **`-1`**, que está **fora da faixa dos valores válidos** — logo slot vazio e sensor sobre o UAV (que normaliza para 0,5) são distinguíveis **por construção**. O bit de validade é **desnecessário** e não entra nesta tabela.

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

**RNF-11 — comparação justa de arcabouço.** Uma MLP 31→32→32→3 executada via PyTorch mede sobretudo o *overhead* de despacho do PyTorch, não o modelo. Ou os dois lados rodam em NumPy puro, ou reporta-se **as duas colunas** (com arcabouço e sem). Caso contrário a medição não é do que se pensa que é.

**RNF-12 — contagem de operações.** Tempo de parede é dependente de implementação; contagem de operações não. Reportar, por decisão e por tipo: **MACs em ponto flutuante**, **lookups inteiros**, **somas**. É a figura de mérito independente de hardware e é onde a diferença estrutural aparece limpa.

**RNF-13 — hardware declarado.** Mesma máquina, mesmo processo, um núcleo, sem GPU para nenhum dos lados. Especificação no relatório.

### Referências de comparação — e uma ressalva que muda o critério de H1

Há **duas** referências, com ordens de grandeza diferentes, e confundi-las inflaria o resultado:

| Referência | Arquitetura | Parâmetros | fp32 |
|---|---|---|---|
| **MLP pareada** (nosso baseline, arquitetura de Katopodis §5.4) | 31→32→32→**3** | ≈ 2 180 | ≈ 8,7 KB |
| **Ator do MADDPG de Lamenza** | **MEDIDO** dos pesos publicados: `obs → 256 → 256 → 2` | 73 474 / 74 498 / 76 546 | **287,0 / 291,0 / 299,0 KB** (`Na` = 2/4/8) |
| **WNN (ator)** | `N=48`, `n=12`, ocupado + mapeamento, **`d = 3`** | — | ≈ 14,8 KB estimado; **133,9 KB medido** (SPEC-03) |

**Duas correções na tabela, e a segunda desmonta a expectativa (SPEC-05, crit. 9).**

*A saída passa de 2 para 3.* A normal projetada da SPEC-05 §4.3 emite `f = (u₁, u₂, μ_s)`: amostra-se em R³ e a ação sai por `θ = atan2(z₂, z₁)`, `s = clip(z₃, 0, 1)`. O gradiente vive em R³, onde não há envoltório, e a topologia circular fica inteiramente no mapa `atan2`. Isto vale para **os dois** aproximadores, então a MLP pareada é `D→32→32→3`. São +33 parâmetros — desprezível no total, mas o número publicado tem de ser o certo. Consequência na contagem de MACs: **2 112** (992 + 1 024 + 96), e não 2 080.

*O `d = 3` triplica a carga útil por posição de memória da WNN.* A estimativa antiga de ≈ 5,6 KB pressupunha posição **escalar**:

```
2 268 × 3 × 2 B (int16) = 13,6 KB  +  1,2 KB (mapeamento)  ≈ 14,8 KB
```

Isto **inverte o sinal da comparação preliminar**: contra ≈ 8,7 KB da MLP pareada, o ator WNN estimado é ~1,7× **maior**, não 1,5× menor.

**E a estimativa corrigida ainda errou por 9×.** A SPEC-03 mediu **133,9 KB** realizados. O motivo é independente do fator `d` e vale registrar porque foi a mesma armadilha duas vezes: as 2 268 entradas vinham de um protótipo com **2 variáveis de entrada**, e a observação real tem **25**. Corrigir o `d` não salva uma estimativa cuja base de ocupação estava uma ordem de grandeza errada.

O critério de falsificação de H1 na v2.0 deste PRD ("memória ocupada ≥10× menor") era, portanto, inatingível contra o baseline pareado — e ficaria satisfeito trivialmente contra o MADDPG.

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

**Varreduras — revistas com o que foi medido.** O eixo mudou em três pontos:

| eixo | como era | como fica | por que |
|---|---|---|---|
| tupla | `n ∈ {8, 10, 12}` | **`n/B ∈ {0,04; 0,06; 0,09; 0,12}`** | O que governa a generalização é a razão, não `n`. Testado: a fração de consultas novas que só endereçam posições nunca escritas cresce monotonicamente com `n/B` (§3.2.4 da dissertação). A faixa cobre a do artigo, 0,042 a 0,125 |
| RAMs | `N ∈ {16, 48, 128}` livre | **`N = ⌈B/n⌉`, determinado** | O mapeamento é **partição** (`randperm` fatiado), como o `:uniform_random` do Ramnet. `N` deixa de ser eixo livre — e a cobertura de bits passa a ser 1,0 por construção, fechando **C4** |
| taxa `α` | não era eixo | **eixo obrigatório, faixa larga, múltiplas sementes** | Medido: o pior erro entre 5 sementes cresce monotonicamente com `α` (0,044 em 0,002 contra 0,238 em 0,05, com divergência em 0,05). Sem varrer `α` com sementes, uma curva ruim seria lida como limitação do modelo |

Mantidos: resolução `T ∈ {8, 32, 128}` e fator de esquecimento `φ`. Retirado o eixo **simples vs circular** — o circular acende sempre `⌊R/2⌋` bits, logo não tem padrão todo-zero e **quebra a convenção de padding** que dispensa o bit de validade. Fica como sub-experimento opcional para dimensões angulares, onde o wrap-around é a topologia correta (ver **P12** e a conexão com **C1**).

Medição preliminar indica retorno decrescente em capacidade (8× mais memória para 13 % menos erro): a varredura documenta o joelho da curva, não maximiza.

**Predição do artigo a checar em `φ` (§5.2.3).** O texto apresenta `φ` como compromisso: 0 descarta o histórico, 1 ignora a não-estacionariedade. Se a nossa varredura der efeito **monótono**, é sinal de que o problema não é suficientemente não-estacionário para o esquecimento importar — e isso é resultado a reportar, não bug.

**Orçamento de simulação — MEDIDO (P2 fechada).** No ambiente de referência, guloso, grade completa: **1,8 a 5,2 h por núcleo** para os 50 M passos de Lamenza, contra 7 dias de GPU. A estimativa anterior de 8–30 h se sustenta com folga. Tabela por célula em `reports/spec00-orcamento.md`.

**Correção que sai da medição:** a célula cara em horas de treino é `Na = 1`, não `Na = 8` — o contador de passos é `passos × Na`, então menos agentes significam menos passos por segundo simulado. **A Etapa 1 é a mais cara do projeto**, o oposto do que a linha de corte da §8 supunha.

> **Portão G1** — 1a atinge taxa de sucesso comparável aos 100 % publicados; 1b supera a caminhada aleatória decisivamente e fica dentro do IC da heurística gulosa; tabela pareada WNN vs MLP fechada com tempo, memória ocupada e latência.
> *Se G1 não for atingido:* é a refutação de H1 e o principal resultado da etapa. Registrar o veredito, diagnosticar (resolução? tamanho de tupla? variância da gaussiana?) e prosseguir — H2 a H4 medem degradação **relativa** ao próprio desempenho sem ruído e permanecem válidas mesmo com política medíocre.

### 6.2 Etapa 2 — extensão: medição ruidosa e filtro de Kalman, 1 UAV

Aqui, e só aqui, o problema é estendido. O capítulo do relatório é separado e a extensão é declarada.

**Um canal só — decisão D13.** A Etapa 2 introduz ruído **exclusivamente no GPS do drone**. O canal do sensor saiu do escopo, e com ele o antigo bloco 2b, o filtro de informação, a H4, a T3 e a P3.

| Bloco | Ruído | Filtro | Hipóteses | Varredura |
|---|---|---|---|---|
| **2** | só GPS do drone | KF + **entrada de controle** | H2, H3 | `σ_own ∈ {1,3,5,10}` × `T ∈ {8,32,128}` = 12 configurações |

Custo medido: **~5 h de um núcleo** a 2 M passos por corrida, paralelizando trivialmente.

**O que o corte ganha.** O canal do drone é o único cujo modelo estava **fechado** — ruído branco, estado móvel, comando conhecido — e é exatamente onde vive o achado medido A3 (`12,54 m → 0,67 m`, 18,8×). O bloco que sai era o que dependia de uma decisão de física não tomada (TS3) e cujo ganho era conjectural. O projeto fica mais estreito e inteiramente apoiado em medição.

**O que o corte custa, dito sem eufemismo.** A H4 perde a causa física, não só o experimento — ver a tabela de hipóteses. Some também a leitura mais bonita do trabalho, a de que *aproximar-se compra informação*.

**O conteúdo de Fusão de Dados não some.** Ele estava em três lugares e dois permanecem: **KF com entrada de controle** (esta etapa) e **interseção de covariância entre UAVs** (Etapa 3, H5), esta última fundindo as estimativas de posição própria que os vizinhos transmitem — construída sobre o degrau 4, e portanto indiferente ao canal do sensor. A validação de consistência por **NEES/NIS** também permanece.

**Bloco 2a — GPS do drone.** `p̂_k = p_k + e_k`. O erro é **comum** a todas as entradas válidas da observação, porque toda entrada tem a forma `(p_k − p_i + S)/(2S)`. Estado móvel, comando conhecido: é onde vive o achado A3 e onde o KF com entrada de controle faz sentido. Modelo fechado, nada pendente.

**A posição dos sensores continua exata**, como no problema original de Lamenza. O que o drone estima é a **própria** posição; e o dado derivado que isso afeta — o ranking dos sensores por distância — passa a ser computado sobre a estimativa filtrada da posição própria, nunca sobre a verdade. É a única costura que o ruído exige. Desenho em [`specs/ARQUITETURA.md`](specs/ARQUITETURA.md) §3.

> **~~TS3~~ — FECHADA pelo corte de escopo (D13), não resolvida.** A pergunta era a física do canal do sensor: GPS próprio do sensor (erro dominado por viés fixo, canal vira piso irredutível) ou localização por rádio (erro branco crescente com a distância, filtragem tem função). **Com o canal fora do escopo, a pergunta deixa de ter consequência.** Fica registrada porque é o primeiro item a reabrir se o escopo voltar a crescer — e porque a resposta determinaria se a H4 tem mecanismo.

**Escada de ablação:**

| # | Configuração | Papel |
|---|---|---|
| 0 | sem ruído — é a política da Etapa 1 | teto |
| 1 | ruído bruto, sem filtro | piso |
| 2 | média móvel / passa-baixa | filtro barato — o KF precisa bater isto |
| 3 | KF velocidade constante | o "óbvio", que já sabemos que falha |
| 4 | **KF + entrada de controle** | a proposta |
| 5 | 4 + covariância na observação | política consciente da incerteza |

A escada roda **uma vez**, sobre o canal do drone. A ressalva anterior — de que os degraus 2 e 3 mudam de sentido para alvo estático — deixa de se aplicar: só há alvo móvel, que é o próprio drone, e portanto o degrau 3 ("KF de velocidade constante") mantém o sentido literal de ser o *óbvio que falha*.

O degrau 5 acrescenta canais à observação e portanto **entra na tabela de desvios (§5.4) quando executado** — é ablação explícita, com resultado reportado com e sem.

Cruzado com o regime de treino: **treino limpo → avaliação ruidosa** vs **treino ruidoso → avaliação ruidosa**. Se o degrau 4 tornar os dois equivalentes, é o argumento mais forte possível a favor do filtro.

**Varredura conjunta** (não uma de cada vez, pela regra da §1): `σ_own × T`, **12 configurações**. Medido: ~5 h de um núcleo a 2 M passos com 3 sementes. Roda muito mais barato com 1 UAV — razão de a Etapa 2 vir antes do multiagente.

**RF-04 (invariante de vazamento).** A verdade **nunca** chega à política. `handle_telemetry` do gradysim entrega posição exata; a corrupção acontece na entrada do protocolo e a verdade fica guardada só para métricas. Teste automatizado obrigatório.

**RF-05 (consistência).** NEES/NIS contra o intervalo qui-quadrado **antes** de olhar qualquer resultado de RL. Um KF mal sintonizado produz números plausíveis que envenenam a análise a jusante — foi literalmente o que ocorreu no experimento preliminar.

**RNF-05 (ressalva de custo).** O KF faz MACs, o que arranha a narrativa "sem MACs, embarcável". Medir, não esconder: ~50–80 flops/passo do KF contra os **2 112 MACs** da MLP pareada `31→32→32→3` (992 + 1 024 + 96), ou seja ~3 %. Vai como número na tabela de custo. *(Corrigido: a versão anterior citava ~2 600 MACs de uma "MLP 31→32→2" — o número vinha da MLP 33→64→8 do `PLANO.md` §9.8 e a arquitetura citada não era a pareada. O KF acrescenta ainda **divisões**, que a contagem de operações da §5.5 precisa tratar como categoria própria: num MCU sem FPU, uma divisão custa mais que um MAC.)*

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
│   │   ├── kalman.py              # KF com entrada de controle (só isto, D13)
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

A execução é decomposta em specs pequenas, com escopo fechado e critério de saída medido: ver **[`specs/README.md`](specs/README.md)**.

| Semana | Bloco | Entrega | Portão / critério de pronto |
|---|---|---|---|
| ~~**1**~~ | ~~SPEC-00~~ | **Concluída.** Ambiente de referência via submodule, `gradysim==0.6.11`, 18 testes, P1 e P2 fechadas, guloso reproduzindo a fig. 7.3 em 9/9 células | ✔ |
| **1** | SPEC-01, 02, 03, 07 | Binarização (**G0**) · baselines e harness · harness de custo · ruído e KF. As quatro são independentes e rodam em paralelo | **G0**: codificador escolhido com evidência, `(T, N)` varridos juntos |
| **2** | SPEC-04, 05, 06 | Núcleo n-tuple · Actor-Critic e MLP pareado · Etapa 1 (réplicas 1a/1b, varreduras, tabela de custo da §5.5) | **G1** e veredito de H1 |
| **3** | SPEC-08 | Ruído no GPS do drone; KF com entrada de controle; NEES/NIS; escada 0–5; varredura `σ_own × T` | **G2**, H2 e H3 |
| **4** | SPEC-09 + fechamento | Multiagente 3a/3b; degraus 6a/6b/6c; interseção de covariância; relatório e slides | **G3**; `make repro` verde; E2 e E3 |

> **Estado real, 2026-08-15 — a numeração de semanas já não descreve a ordem em que as coisas saíram.** Concluídas: SPEC-00 (ambiente), SPEC-01 (**G0** fechado), SPEC-03 (T1 e T3 medidos, T2 pendente), SPEC-04 (as duas redes, desvio zero contra o `Ramnet.jl`) e a metade da SPEC-05 (Algoritmo 4.4 implementado; falta a MLP AC pareada). **Pendente e bloqueando:** a SPEC-02, cujo teste A/A é o último item estatístico de que todo IC já reportado depende.
>
> **O corte D13 devolveu folga à semana 4**, que perdeu o bloco 2b. Duas destinações honestas, em ordem de prioridade:
>
> 1. **SPEC-02 completa, com teste A/A e calibração do bootstrap pareado.** É dívida, não escopo novo: a validação 9/9 da SPEC-00, o `greedy` vs `smart` e a comparação de custo da SPEC-03 todos pressupõem um procedimento de intervalo nunca verificado.
> 2. **T2 (latência)** da SPEC-03, que ficou pendente da estrutura de armazenamento — e a estrutura agora existe, esparsa e validada.
>
> Não usar a folga para reabrir o canal do sensor pela porta de trás: o corte foi uma decisão, e revertê-lo informalmente seria pior que nunca tê-lo feito.

**A SPEC-00 saiu antes do prazo e mais barata que o previsto** — o ambiente veio pronto do repositório de referência em vez de ser escrito. Isso liberou a semana 1 para quatro specs em paralelo.

A redação **não** espera a semana 4: cada tabela entra no relatório assim que sai.

### Linha de corte (o que cai primeiro se atrasar)

O antigo item 1 — bloco 2b — **saiu do escopo por decisão (D13)**, não está mais na fila de corte. A lista encurtou e foi renumerada (havia dois itens rotulados "2"):

1. Varredura de `n/B` → fixar o ponto medido e varrer só `T` — com a ressalva de que `T` e `n/B` não são independentes
2. Células `Ns ∈ {24, 36}` e `Na = 8` na Etapa 1
3. Varredura `σ_own × T` → reduzir a `σ ∈ {1, 10}` × `T ∈ {8, 128}`
4. Degraus 2 e 3 da escada — reter ao menos um como referência barata
5. Etapa 3 com `Na = 2` em vez de 4, menos sementes

**O que não cai:** o degrau 4 (KF com entrada de controle) e o degrau 6c (interseção de covariância). São o conteúdo de Fusão de Dados do trabalho; cortá-los descaracterizaria a disciplina.

**Não podem cair:** os três portões com veredito escrito; réplica 1a; baseline guloso; degraus 0, 1, 4 e 6c; NEES/NIS; sub-passo 3a; **a tabela de custo da §5.5 (T1/T2/T3)**; RF-04 (invariante de vazamento).

A tabela de custo é barata de produzir — instrumentação, não experimento — e é a única contribuição que sobrevive mesmo se todos os resultados de desempenho forem medianos. Cortá-la seria cortar o trabalho.

**Regra de escalonamento:** portão não atingido ao fim da semana aciona a linha de corte **da etapa seguinte** — nunca estende a etapa atual para dentro dela. Chegar às três etapas com resultados modestos vale mais, nesta disciplina, do que uma Etapa 1 exemplar e nenhuma fusão.

---

## 9. Riscos

| Risco | Mitigação |
|---|---|
| **Ação contínua com n-tuple é menos madura que SARSA discreto** | Katopodis validou n-tuple Actor-Critic em Mountain Car e Hopper, ambos contínuos, superando o AC com MLP. Sintonizar `σ²`, `T` e `n` antes de qualquer outra coisa. Discretizar seria **modificar o problema** e só é admissível como desvio declarado na §5.4, com aprovação |
| Zero-padding confundido com sensor sobre o UAV | bit de validade por slot no codificador |
| **A WNN não é menor que a MLP pareada** — **CONFIRMADO, e por margem maior que a temida**: com `d = 3` a estimativa vira 14,8 KB vs 8,7 KB (1,7× maior), e o medido da SPEC-03 é 133,9 KB vs 7,6 KB (17,5× maior) | §5.5: reportar contra as duas referências sem escolher a conveniente; migrar o argumento para contagem de operações e ausência de FPU, que é onde a diferença é estrutural |
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
| **D6** | Algoritmo: **Algoritmo 4.4** — n-tuple Actor-Critic de `m` passos, vantagem `δ = G − f_value(S_τ)`, `Σ` e `α` **constantes**, e parameter sharing | Única variante do trabalho de referência que suporta ação contínua, validada em Hopper. **Correção:** a versão anterior desta decisão prescrevia covariância e taxa de aprendizado *recozidas*. O recozimento existe no `ContinousFunctionalAC.jl` mas **não na dissertação** — o apêndice A lista `Σ` e `α` como constantes. Seguimos o artigo; o recozimento fica disponível e desligado por padrão. Implementado em `src/uavwnn/rl/actor_critic.py`, validado contra o `Ramnet.jl` com desvio zero |
| **D7** | **O MADDPG não é retreinado — mas os pesos que Lamenza publicou são avaliados.** *Estreitada em 2026-08-20.* | A justificativa original era que *"reproduzi-lo exigiria replicar exatamente simulador, observação e recompensa; qualquer divergência invalidaria a comparação silenciosamente"*. Essa justificativa **não se aplica a carregar os pesos dele**: o submodule traz 9 checkpoints treinados e o próprio harness de avaliação, então não há nada a replicar e o risco de divergência não existe. O que continua valendo é não **retreinar**, que custaria ~7 dias de GPU por célula. Tamanho e latência **medidos** (`reports/spec02-baseline-maddpg.md`); tempo de conclusão sai do `test.py` do submodule, tarefa da SPEC-02. Ressalva: **não há checkpoint para `Na = 1`**. O pareamento n-tuple AC vs MLP AC continua sendo o principal |
| **D8** | **Reprodução antes de extensão**; `fusion/` é pacote opcional e desligável | Sem uma linha de base fiel, nenhum efeito de filtro é atribuível |
| **D9** | **Fusão local (Etapa 2) antes do multiagente (Etapa 3)** | A varredura σ × T é o experimento mais caro e roda muito mais barato com 1 UAV. E a regra *bin ≈ σ* precisa estar fixada antes de o multiagente entrar |
| **D10** | Etapa 3 tem sub-passo **3a sem ruído** obrigatório | Sem ele, degradação por coordenação e por incerteza ficam confundidas |
| **D11** | **Tamanho do modelo e tempo de inferência são figuras de mérito de primeira classe**, medidas em toda etapa e reportadas junto com tempo de conclusão e taxa de sucesso | São o conteúdo de H1 e a única contribuição que independe de a política ser boa: mesmo com desempenho apenas comparável, o par (tamanho, aritmética) é resultado. Protocolo em §5.5 |
| **D13** | **O ruído entra apenas no GPS do drone.** O canal do sensor sai do escopo, e com ele o bloco 2b, o filtro de informação, a H4, a T3 e a P3 | O canal do drone é o único com modelo fechado (ruído branco, estado móvel, comando conhecido) e é onde vive o achado medido A3 (18,8×). O canal do sensor dependia de uma decisão de física não tomada (TS3) e tinha ganho conjectural. **Custo aceito:** a H4 perde a causa física, não só o experimento. **O que se preserva:** KF com entrada de controle, interseção de covariância entre UAVs (H5) e validação NEES/NIS — o conteúdo de fusão permanece |
| **D12** | O tamanho reportado é o **realizado**, com mapeamento de tuplas contabilizado, e a razão é dada contra **ambas** as referências | Reportar só a LUT ocupada contra o ator do MADDPG seria escolher o numerador e o denominador convenientes |

### Pendências

- **P1** — Confirmar `communication_range` no repositório `Project-GrADyS/gradys-data-collection` (branch `dissertation`). Não consta nas tabelas 7.1/7.2. **Bloqueia a comparação numérica com os resultados publicados**, então é a primeira tarefa da semana 1.
- **P2** — Medir o orçamento real de simulação e validar `update_rate = 0,1` contra 0,01 com 8 UAVs.
- ~~**P4**~~ — **FECHADA POR MEDIÇÃO (2026-08-20).** O ator é `obs → 256 → 256 → 2`, exatamente o default do CleanRL que a estimativa supunha: 73 474 a 76 546 params, **287 a 299 KB** conforme `Na`. Lido dos 9 checkpoints que o submodule publica, sem treinar nada. A razão da §5.5 deixa de sair marcada como estimativa. Ver `reports/spec02-baseline-maddpg.md`. *Texto original:* Extrair do repositório público a **arquitetura do ator do MADDPG** (largura e profundidade das camadas). Sem ela, a razão de tamanho contra a referência de Lamenza não passa de estimativa (§5.5). O padrão do CleanRL, base declarada da implementação dele, é 256×256.
- ~~**P3**~~ — **FECHADA pelo corte de escopo (D13), não resolvida.** A pergunta era se o sensor transmite posição estimada (caso linear → filtro de informação) ou distância/RSSI (caso não linear → EKF/UKF e trilateração). Com o canal do sensor fora do escopo, a pergunta não tem consequência. Fica registrada junto com a **TS3** como o par a reabrir se o escopo crescer.

- **P13** — **`Σ` é covariância ou desvio-padrão?** A eq. (4.34) diz *"predefined **covariance** matrix"*, logo a amostragem correta é `√Σ·z`. O `ContinousActionPolicy.jl` do autor amostra `Σ·z` mas usa `inv(Σ)` no gradiente — trata o mesmo número como desvio num lugar e variância no outro. **Seguimos o artigo**; `PoliticaGaussiana(..., escala_como_desvio=True)` reproduz o Julia. Com `Σ = 0,25` a diferença é **fator 2 na exploração**, então a escolha tem de aparecer na tabela de hiperparâmetros do relatório. *Resolvida em recomendação; falta aval.*

- **P14** — **Revarrer o regime de tupla no eixo `n/B`.** *Rebaixada de bloqueio para refinamento.* Registrei-a supondo que o nosso `n = 12` fosse artefato do armazenamento denso diante dos `n ∈ {16, 32, 64}` do apêndice A. **Normalizando por `B` isso não se sustenta:** as dimensões de observação estão no §5.1 da dissertação, e `n/B` do artigo vai de 0,042 (Hopper) a 0,125 (CartPole), com `N` de 8 a 24. O nosso `n/B = 0,060` e `N = 17` ficam **dentro das duas faixas**, quase sobre o Lunar Lander. A diferença real é a decomposição: o artigo usa resolução alta com tupla grande; nós, resolução baixa com tupla pequena. A varredura vale como ajuste fino, no eixo `n/B`.

- **P15** — **Decidir se replicamos §5.2.2 com orçamento cheio.** Nossa replicação parcial dá **empate** na configuração que o artigo declara para CartPole (`R=128, n=64`: 0,96×, IC cruza zero) e **vitória do circular** em baixa resolução (`R=32, n=16`: 1,68×, IC95% [+39,6; +56,1], 20 de 20 sementes). O orçamento é ~1/50 do original. **Não é bloqueio para o nosso trabalho** — o codificador de produção é o quantil, por outras razões —, mas se o relatório citar §5.2.2 como fundamento, a ressalva precisa vir junto. Ver `reports/replica-katopodis-52.md`.

---

## 11. Definition of done

| Etapa | Pronta quando |
|---|---|
| **1** | Réplica 1a com taxa de sucesso reportada contra os 100 % publicados; 1b com tempo de conclusão vs guloso, em pelo menos `Na ∈ {1, 2, 4}`; **tabela de custo completa da §5.5** (T1 alocado/ocupado/realizado com mapeamento de tuplas, curva de ocupação, T2 mediana e p99 decomposto, T3 contagem de operações), contra as duas referências; veredito para H1 |
| **2** | NEES/NIS do canal do GPS próprio no intervalo qui-quadrado; degraus 0–5 com IC nos dois regimes de treino; varredura `σ_own` × T reportada; vereditos para H2, H3, H4 |
| **3** | 3a medido como referência; degraus 1, 4, 6a/6b/6c medidos; vereditos para H5 e H6 |

**Global:**

1. `make repro` regenera todas as tabelas e figuras a partir de seeds fixas.
2. Testes passam, incluindo os invariantes: **a verdade não vaza para a política** (RF-04) e **com `fusion/` desligado, o sistema executa o problema de Lamenza puro** (RNF-07).
3. A tabela de desvios (§5.4) reflete exatamente o que foi executado.
4. A tabela de custo segue o protocolo da §5.5 integralmente: tamanho **realizado** (não só ocupado), mapeamento de tuplas contabilizado, p99 além da mediana, contagem de operações por tipo, e — nas Etapas 2–3 — T1/T2 recalculados com o filtro embutido.
5. As razões de tamanho e latência são reportadas contra **ambas** as referências (MLP pareada e ator do MADDPG), com a configuração `T, n, N, Na` que as gerou.
6. Cada hipótese **em vigor** — H1, H2, H3, H5, H6 — tem veredito escrito, confirmada, refutada ou inconclusiva, **com o número que o sustenta**. A **H4 foi retirada** por D13 e entra no relatório como hipótese *abandonada com justificativa*, não como inconclusiva: registrar que o mecanismo saiu do escopo é diferente de tê-la testado sem resultado. Hipótese refutada é resultado, não fracasso.
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
