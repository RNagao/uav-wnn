# SPEC-06 — Etapa 1: réplicas 1a/1b, varreduras e tabela de custo → portão G1

**Etapa:** 1 · **Depende de:** SPEC-00 (ambiente), SPEC-02 (baselines e harness de avaliação), SPEC-03 (harness de custo), SPEC-05 (n-tuple AC e MLP AC pareado) · **Estimativa:** 4 dias
**Portão de saída:** `G1` · **Referência:** PRD §2 (H1), §5 inteira, §5.5 (custo), §6.1 (Etapa 1), §8 (cronograma e linha de corte) · **Fonte externa:** Lamenza (2025) §7.1, §7.2, §7.3, §7.4

---


## 0. Revisão pela dissertação — três eixos de varredura mudaram (2026-08-15)

| eixo | como era | como fica | por que |
|---|---|---|---|
| tupla | `n ∈ {8, 10, 12}` | **`n/B ∈ {0,04; 0,06; 0,09; 0,12}`** | O que governa a generalização é a razão, não `n` — testado contra §3.2.4. A faixa cobre a do artigo (0,042 a 0,125) |
| RAMs | `N ∈ {16, 48, 128}` livre | **`N = ⌈B/n⌉`, determinado** | O mapeamento é **partição**, como o `:uniform_random` do Ramnet. Deixa de ser eixo livre, e a cobertura de bits vira 1,0 por construção (**C4 fechada**) |
| taxa `α` | não era eixo | **obrigatório, faixa larga, múltiplas sementes** | Ver abaixo |

**A busca de `α` não é opcional, e é a mudança mais importante para esta spec.**
Medido no bandido contextual, 5 sementes, 20 000 passos:

| `α` | pior erro entre as sementes |
|---|---|
| 0,002 | **0,044** |
| 0,010 | 0,072 |
| 0,050 | 0,238 |

**Monótono**, com uma semente divergindo em `α = 0,05`. Meu primeiro palpite era
`α = 0,05` — **25× grande demais** —, e a curva resultante pareceria limitação do
modelo. É a instabilidade de passo fixo que a dissertação nomeia no capítulo 6 ao
sugerir TRPO/PPO.

**Se o portão G1 não for atingido, a instabilidade de `α` é a primeira suspeita**,
antes de resolução, tamanho de tupla ou variância da gaussiana. E a saída já está
nomeada pelo próprio autor.

**Predição do artigo a checar em `φ` (§5.2.3).** O texto apresenta `φ` como
compromisso: 0 descarta o histórico, 1 ignora a não-estacionariedade. Efeito
**monótono** na nossa varredura significaria que o problema não é
suficientemente não-estacionário para o esquecimento importar — resultado a
reportar, não bug.

**Eixo retirado:** simples vs circular. O circular quebra a convenção de padding
(sem padrão todo-zero). Fica como sub-experimento opcional para dimensões
angulares. Ver **P12** e `reports/replica-katopodis-52.md`.

**Ganho de escopo:** o Algoritmo 4.1 (SARSA) está implementado em
`src/uavwnn/rl/sarsa.py` e aprende CartPole. Não é o algoritmo desta spec — a
ação é contínua —, mas dá um segundo ponto de validação do núcleo n-tuple num
laço de RL real, contra uma tarefa do capítulo 5.

---

## 1. Objetivo

Produzir os **números** e o **veredito de H1**: a política n-tuple Actor-Critic, sobre o problema de Lamenza inalterado e sem ruído, aprende? Quanto custa em tempo de conclusão, tamanho de modelo e latência, comparada — em pareamento estrito — a um Actor-Critic com MLP densa, e comparada aos alvos publicados por Lamenza?

Concretamente, esta spec entrega:

1. a **réplica 1a** (Lamenza §7.1) com taxa de sucesso contra os **100 %** publicados;
2. a **réplica 1b** (Lamenza §7.2) com tempo de conclusão contra a heurística gulosa, percorrendo `Na = 1 → 2 → 4 → 8`;
3. a **comparação pareada** WNN vs MLP, mesma tudo exceto o aproximador;
4. as **varreduras** `T × n × N ×` esquema `×` fator de esquecimento **confirmadas no treino**;
5. a **medição** (não estimativa) do orçamento de simulação, fechando P2;
6. a **tabela de custo completa da §5.5** (T1/T2/T3);
7. o **veredito escrito de H1** e a decisão de portão.

## 2. Por que esta spec existe separada

**Ela não escreve infraestrutura.** Ambiente, protocolos, ponte, `rewards.py` verbatim, baselines, aproximadores e harness de custo são das SPEC-00/02/03/04/05. Se esta spec estiver implementando ambiente ou modelo, a divisão falhou e o cronograma da §8 do PRD já está comprometido: a Etapa 1 tem **uma semana** (semana 2) e é a etapa cujo atraso empurra as duas extensões de Fusão de Dados para fora do prazo.

O que ela produz é o que o PRD chama de **portão**: um critério objetivo, medido, com veredito escrito. E ela produz um segundo bem menos óbvio: **um protocolo experimental que sobrevive à variância de treino**. Lamenza é explícito sobre não ter conseguido repetir corridas — "*several repeated training runs would provide a larger sample space to draw a conclusion from, unfortunately the training time is too long for this to be feasible with our current resources*" (§7.4.1) — e relata **falha catastrófica de treino** na curva de 4 agentes do §7.3, em que o modelo não melhora a recompensa esperada em nada. Com corrida única por configuração, **método ruim e semente ruim são indistinguíveis**. Se o aproximador barato entrega o que promete, repetir corridas passa a caber no orçamento — e passar a repetir é, por si só, uma contribuição de método habilitada pelo custo, a ser reportada como tal.

## 3. Escopo

**Dentro:** execução das campanhas de treino e avaliação da Etapa 1; protocolo de semeadura, repetição e parada; varreduras confirmatórias; medição do orçamento de simulação; execução do harness de custo sobre as configurações finais; agregação estatística; veredito de H1; relatório da etapa.

**Fora:**
- qualquer código de ambiente, aproximador, codificador, baseline ou harness — pertence às specs de dependência;
- ruído, filtro de Kalman, NEES/NIS — SPEC-07/08. Esta spec roda com `fusion/` **desligado** (RNF-07), e isso é verificado por teste;
- multiagente **com** ruído — SPEC-09. A escalada `Na = 2 → 4 → 8` aqui é sem ruído e faz parte do problema original (PRD §2: `Na` é parâmetro de cenário, tab. 3.1 de Lamenza);
- generalização a `Na` ou `Ns` variáveis (Lamenza §7.4) — fora de escopo por decisão do PRD §3. O §7.4 é lido aqui apenas como **fonte de números de referência** e de evidência sobre variância de treino.

---

## 4. Protocolo experimental comum

Tudo que é fixado aqui vale para **todas** as tarefas T-06.N e é **pré-registrado**: escrito e versionado no repositório *antes* da primeira corrida de campanha. Mudança posterior exige commit próprio com justificativa, e as corridas anteriores continuam no registro.

### 4.1 Três semeaduras com papéis distintos

| Semente | Governa | Regra |
|---|---|---|
| `seed_eval` | posições dos sensores, posições iniciais dos UAVs e ordem dos episódios do banco de avaliação | **Fixa e única.** Idêntica para WNN, MLP, guloso, aleatório e k-means+TSP |
| `seed_train` | inicialização das LUTs/pesos, ruído de exploração da política, sorteio dos cenários de treino | Varia em `{0, …, R−1}`. É a única fonte de variância entre repetições |
| `seed_tuple` | mapeamento pseudo-aleatório dos `N × n` bits das RAMs | **Amarrada a `seed_train`** — ver decisão abaixo |

> **Decisão D-06.1 — o ladrilhamento é parte do modelo, não do ambiente.** `seed_tuple` varia junto com `seed_train`. Fixá-lo condicionaria todos os resultados a um único ladrilhamento aleatório e reportaria como propriedade do método algo que pode ser propriedade de um sorteio. O preço é variância maior entre corridas; o benefício é que o IC entre corridas passa a cobrir a escolha das tuplas, que é exatamente a incerteza que o leitor precisa ver. Uma corrida de controle com `seed_tuple` fixo e `seed_train` variável é executada **uma vez**, no ponto base, para quantificar quanto da variância vem do ladrilhamento.

### 4.2 Bancos de episódios: teste, validação, e pareamento

Gerados uma vez por célula `(Na, Ns)`, versionados em disco, com hash registrado em cada corrida:

| Banco | Tamanho | Uso | Regra |
|---|---|---|---|
| `test` | **500 episódios** (Lamenza §7.2) | métrica final reportada | **Nunca** usado para selecionar checkpoint, hiperparâmetro ou configuração |
| `val` | 100 episódios, **disjuntos** de `test` | avaliação intercalada durante o treino, parada antecipada, seleção de checkpoint, varreduras | disjunção verificada por teste automatizado |

Lamenza avalia **durante** o treino e reporta o modelo carregado dessas avaliações (§7, p36) — o que é, tecnicamente, seleção sobre o conjunto de reporte. Nós separamos e **dizemos que separamos**; é uma diferença de protocolo a favor do rigor, não do resultado, e vai declarada no relatório.

**Pareamento.** Todo método avalia exatamente os mesmos 500 episódios do banco `test`: as mesmas posições de sensores, as mesmas posições iniciais, na mesma ordem. É isso que torna legítimo o IC da **diferença** por episódio.

**Determinismo da avaliação.** A política é avaliada com a **média** da gaussiana, sem amostragem — o análogo do "*action noise is removed*" de Lamenza (§7, p36). A variante estocástica é medida **uma vez**, no ponto base, como sensibilidade.

### 4.3 Repetições de treino por configuração

| Classe de configuração | `R` (sementes de treino) |
|---|---|
| Caminho crítico: 1a (`Na ∈ {1,2,4}`), 1b (`Na ∈ {1,2,4}`), MLP pareada | **5** |
| Escalada `Na = 8`, células `Ns ∈ {24, 36}` | **3** |
| Pontos de varredura exploratória (T-06.4) | **3**, com re-execução a `R = 5` apenas dos 2 finalistas |

`R = 5` é o mínimo que permite um **teste de permutação de duas amostras** com 5 vs 5 (252 arranjos, `p` mínimo ≈ 0,008) — isto é, o menor `R` com que uma diferença entre métodos pode ser declarada significativa no nível da corrida, e não apenas no nível do episódio. Abaixo disso, a estatística de corrida é descritiva.

**Nenhuma corrida é descartada.** Uma corrida que falha catastroficamente é dado, não lixo. A única exceção admitida é divergência numérica por violação da asserção `η = c/N, c ≲ 0,5` (RNF-01) — que é **bug**, não resultado: corrigir e re-executar a configuração inteira, todas as sementes.

**Estabilidade de treino é métrica de primeira classe.** Para cada configuração, reportar `falhas/R`, com o critério de falha pré-registrado (§4.5). É a métrica que Lamenza não pôde medir e que separa "o método não funciona" de "essa semente não funcionou".

### 4.4 Critério de parada e definição de "convergiu"

- **Orçamento máximo por corrida:** `B_max` passos de ambiente, derivado em **T-06.0** a partir da vazão medida e do orçamento de parede alocado. Não é herdado dos 50 M de Lamenza — 50 M passos é o orçamento *dele*, com o custo por passo *dele*.
- **Avaliação intercalada** a cada `E = B_max/40` passos, sobre o banco `val`, com o treino congelado e sem ruído de exploração.
- **Convergiu** ⇔ a média móvel da métrica de validação sobre as últimas 5 avaliações não melhora mais que **1 % relativo** ao longo de **8 avaliações consecutivas**. Dispara parada antecipada e o restante do orçamento é liberado para outras corridas.
- **Checkpoint reportado** = o de melhor métrica de validação (parada antecipada), **não** o último. Regra fixada antes da campanha.
- **Falha catastrófica** ⇔ ao fim de `B_max` (ou da parada antecipada), a métrica de validação **não supera a caminhada aleatória** com IC bootstrap 95 % separado. É a definição operacional do fenômeno do §7.3 de Lamenza.

### 4.5 Métricas e tratamento de episódios não concluídos

**Tempo de conclusão é indefinido quando nem todos os sensores foram visitados** — e o PRD não trata esse caso. A tab. 7.3 de Lamenza mostra `500.00 s` em células degeneradas, o que revela a convenção dele: **censura em `max_episode_length`**. Reportamos as três, sempre juntas:

| Símbolo | Definição | Papel |
|---|---|---|
| `SR` | fração de episódios com `Σ visited_i^T = Ns` (Lamenza eq. 7-1) | métrica primária de 1a |
| `CT_cens` | tempo de conclusão, com episódios falhos contabilizados como `max_episode_length` | **comparável aos números publicados**; métrica primária de 1b |
| `CT_succ` | tempo de conclusão condicional aos episódios bem-sucedidos | mais informativo, mas **premia política que desiste** — nunca reportado sozinho |

**Nunca** reportar `CT_succ` sem `SR` ao lado. Uma política que fecha 20 % dos episódios rapidamente tem `CT_succ` excelente e é inútil.

### 4.6 Intervalos de confiança — dois níveis, e a regra de concordância

Dois níveis de variância, e reportar só um deles é o erro estatístico mais provável desta spec.

**Nível 1 — episódio (dentro de uma corrida).** Para o par de métodos `(A, B)` e cada episódio `i` do banco `test`, `d_i = m_A(i) − m_B(i)`. IC 95 % **bootstrap pareado** sobre `{d_i}`, 1000 reamostragens (PRD §6.4, Katopodis §5.4). Verificação não paramétrica secundária: Wilcoxon pareado de postos com sinal. Reportar mediana da diferença e IC — **nunca** concluir por ICs marginais sobrepostos, que é o erro que a §6.4 do PRD já proíbe implicitamente ao exigir IC em todo agregado.

**Nível 2 — corrida.** As `R` corridas de cada método dão `R` estimativas de `CT_cens` médio. Reportar mediana, amplitude `[min, max]` e **todas as corridas visíveis na figura**. Com `R = 5` de cada lado, teste de permutação exato entre os dois conjuntos.

> **Regra de concordância (D-06.2).** H1 só é declarada confirmada ou refutada se os **dois níveis concordarem em sinal**. Se o IC pareado por episódio da corrida mediana separa mas o teste de permutação entre corridas não, o veredito é **inconclusivo por variância de treino**, e essa discordância é o resultado a reportar. O IC de nível 1 **subestima sistematicamente** a incerteza total, porque condiciona a uma corrida; escrevê-lo sozinho seria afirmar mais do que se mediu.

### 4.7 O que fica registrado por corrida

Um diretório por corrida: `runs/<experimento>/<config_hash>/<seed_train>/`

| Artefato | Conteúdo |
|---|---|
| `config.yaml` | configuração completa e resolvida (não a de linha de comando), com `config_hash` |
| `env.json` | commit do repositório, versões de dependências, CPU, número de núcleos, hash dos bancos `test` e `val` |
| `metrics.parquet` | por avaliação intercalada: passo, recompensa média, `SR`, `CT_cens`, `CT_succ`, **entradas ocupadas nas LUTs** (RNF-08), tempo de parede acumulado, passos/s |
| `checkpoints/` | melhor por validação e final |
| `eval_test.parquet` | resultado **por episódio** dos 500 do banco `test` — é o insumo do bootstrap pareado, e sem ele o IC da diferença não é reconstruível |
| `stdout.log` | inclusive as asserções de RNF-01 disparadas |

Agregação em `results/spec06/*.csv`, regenerável por `make repro` a partir dos `eval_test.parquet` — sem re-treino. Re-treino completo é alvo separado (`make repro-full`), com o tempo declarado.

---

## 5. Tarefas

### T-06.0 — Orçamento de simulação medido, e `B_max` (fecha P2)

**É a primeira tarefa e ela bloqueia todas as outras**, porque `B_max` é derivado dela e todo o dimensionamento da campanha depende de `B_max`.

O PRD §6.1 é honesto quanto ao status do número: 0,11 s por episódio de 120 s simulados com 4 UAVs / 12 sensores a `update_rate = 0,1`, extrapolado para **8–30 h num núcleo** para 50 M passos. **Isso é estimativa.** Convertida em vazão, a faixa 8–30 h para 50 M passos equivale a **≈ 460–1 700 passos/s**, e é esse número que precisa ser confrontado com a medição.

**Medir:**

1. **Vazão** (passos de ambiente por segundo de parede) em `Na ∈ {1,2,4,8} × Ns ∈ {12,24,36}`, com `max_episode_length = 500 s` e `algorithm_iteration_interval = 2 s`.
2. **Decomposta** em quatro parcelas, porque o agregado esconde exatamente o que interessa:

| Parcela | WNN | MLP | Distingue os métodos? |
|---|---|---|---|
| simulação (`gradysim`, eventos, mobilidade, coleta) | comum | comum | **não** |
| montagem da observação (ordenar sensores, `O(Ns log Ns)`) | comum | comum | **não** |
| codificação (termômetro + bits de validade) | só WNN | — | sim |
| passo de aprendizado (forward + atualização) | lookups + somas | MACs | **sim** |

> **Ponto que muda a narrativa.** Se a simulação dominar a vazão, o tempo de parede de treino é **comum aos dois lados** e não sustenta a alegação "sem GPU". O que sustenta é a parcela de aprendizado, medida isolada. Reportar as duas: **tempo de parede total** (contra os 7 dias de GPU por célula de Lamenza) e **tempo de parede da parcela de aprendizado** (a comparação limpa). Misturá-las repete o erro que a §5.5 já proíbe para T2.

3. **Fidelidade de `update_rate = 0,1` contra `0,01`** (P2). Rodar a heurística gulosa sobre os **mesmos** cenários do banco `test` sob os dois valores. Aceite: diferença de `CT_cens` médio com IC pareado contendo 0 **e** `|Δ| < 1 %` da média, **e** KS de duas amostras não rejeitando a 5 % sobre a distribuição de `CT_cens`. Repetir com `Na = 8`, o caso denso em eventos. Se falhar, `update_rate = 0,01` e `B_max` cai proporcionalmente.

4. **Derivar `B_max`.** Com vazão medida `v`, orçamento de parede `W` (núcleos × horas alocadas à campanha), `C` configurações e `R` sementes: `B_max = v·W/(C·R)`. Publicar a conta e o plano de campanha resultante **antes** de rodar. Se `B_max` cair abaixo de ~10 M passos, acionar a linha de corte do PRD §8 na ordem publicada (varreduras `n × N` primeiro), não reduzir `R` — reduzir `R` destrói a única defesa contra o problema do §7.3.

**Saída:** tabela de vazão por célula e por parcela; veredito escrito sobre a estimativa de 8–30 h (confirmada, otimista ou pessimista, com o fator); `B_max` e o plano de campanha; P2 fechada.

---

### T-06.1 — Réplica 1a: experimento exploratório (Lamenza §7.1)

**Configuração**, verbatim exceto onde indicado:

| Parâmetro | Valor | Origem |
|---|---|---|
| `N*_a` | **0** — agentes cegos aos demais | §7.1 |
| Recompensa | **eq. 7-2**: `R_t = (Σ_i visited_i^t)/Ns` | §7.1 |
| `Ns` | 12 (célula base); 8 na réplica da fig. 7.1 | §7.1, tab. 7.1 |
| `Na` | **1, 2, 4** — Lamenza roda "todas as variações" e ilustra com 4 | §7.1 |
| `max_episode_length` | **permissivo** — ver abaixo | §7.1, sem valor publicado |
| `max_seconds_stalled` | 30 s | tab. 7.1 |
| demais | idênticos à tab. 7.1 | §7.1: "*Other than that, the parameters used are identical to 7.1*" |
| Métrica | **taxa de sucesso** sobre 500 episódios | §7.1 |
| **Alvo publicado** | **100 %** | §7.1 |

Com `N*_a = 0`, a observação tem `24 + 1 = 25` escalares **independentemente de `Na`** — o que torna 1a mais barato que 1b em todas as células e é a razão de ele vir primeiro.

**Papel.** Demonstrar que o n-tuple Actor-Critic e o laço de treino aprendem alguma coisa, antes de gastar orçamento em cenário difícil. É a mesma função que o experimento teve no trabalho original, e é a única tarefa desta spec que o PRD §8 marca como **não cortável**.

#### O que "permissivo" significa numericamente

Lamenza não publica o valor. Tratamos como **pendência declarada (P5)** e resolvemos por definição operacional, não por chute:

> **`max_episode_length_1a = 3 × CT_p99(guloso)`**, arredondado para cima em múltiplos de 100 s, com **piso de 500 s** (o valor da tab. 7.1) e **teto de 2 000 s** (limite de custo). `CT_p99(guloso)` é o percentil 99 do tempo de conclusão da heurística gulosa **medida por nós** sobre o mesmo banco `test`.

Justificativa: "permissivo" tem um significado funcional único no texto de Lamenza — *"allows agents plenty of time to complete the data harvesting process. This reduces the urgency to complete the scenario as quickly as possible"*. Traduzido: **o horizonte não é o gargalo**. A verificação disso é direta e mensurável: sob o horizonte escolhido, a **taxa de sucesso do guloso deve ser ≥ 99 %**. Se não for, o horizonte não é permissivo e sobe até ser.

**Sensibilidade obrigatória.** A comparação com os 100 % publicados depende de um parâmetro que o original não publica. Portanto a política final de 1a é avaliada em **três** horizontes — `500 s`, o valor escolhido, e `2 ×` o valor escolhido — e as três taxas de sucesso vão na tabela. Reportar uma só seria esconder que a réplica tem um grau de liberdade que o original não fixou.

#### Riscos técnicos específicos de 1a

- **Escala do alvo do crítico.** A recompensa 7-2 é não negativa e **monotonicamente não decrescente dentro do episódio**. Com `γ = 0,99` e horizonte permissivo, `V` se aproxima de `≈ 100 × fração_visitada`. O crítico n-tuple calcula `V = Σ_i W[i, a_i(x)]` sobre `N` RAMs: com `N = 48`, cada entrada precisa valer `≈ 2` para representar `V ≈ 100`, e os alvos de TD são grandes desde o começo. **Normalizar o alvo de valor por `(1−γ)`** é hiperparâmetro do aproximador, não alteração do MDP, e portanto **não** entra na tabela de desvios §5.4 — mas precisa ser decidido e registrado antes da campanha, não descoberto no meio dela.
- **Interação `max_seconds_stalled` × RF-02.** O PRD RF-02 manda **não** podar quando todos os sensores são coletados; mas `max_seconds_stalled = 30 s` poda 30 s após a última visita, já que nenhuma visita nova ocorre. O efeito líquido é que o episódio termina 30 s depois de completo, e o horizonte permissivo só atua sobre episódios **incompletos**. Isso precisa ser confirmado contra o código de referência antes de 1a rodar — é o que efetivamente define quanta "experiência extra pós-conclusão" o agente recebe, e Lamenza atribui a isso ganho grande de desempenho (§5).

#### Critério de aceite de 1a

| `SR` (mediana sobre as `R` corridas, banco `test`, `Na = 1`) | Veredito |
|---|---|
| ≥ 99 % | **reproduzido** — IC de Wilson 95 % sobre 500 episódios: 100/100 acertos dá limite inferior ≈ 99,3 % |
| 90 – 99 % | **parcialmente reproduzido** — reportar com o diagnóstico |
| < 90 % | **não reproduzido** — dispara a escada de diagnóstico |

**Escada de diagnóstico, em ordem fixa** (parar no primeiro degrau que explique):

1. `Ns = 4` e horizonte enorme — o problema ficou trivial e ainda assim não aprende? Então o defeito é do laço de treino ou do crítico, não do cenário.
2. Variância `σ²` da gaussiana em `{0,1; 0,2; 0,5}` (Katopodis usa 0,2 em Hopper, 0,5 em Mountain Car).
3. Crítico: substituir `V` n-tuple por retorno Monte Carlo tabular por episódio. Se resolver, o problema é do **crítico**, e a SPEC-01 T-01.6 (regressão de valor) previu errado.
4. Ator: reexecutar o proxy de clonagem do guloso (SPEC-01 T-01.5) na **configuração de treino real**. Se a clonagem falhar aqui e tiver passado lá, o corpus da SPEC-01 não representa a distribuição de treino e a spec precisa ser refeita. *(O invariante V-1 e o teste KS que o guardavam deixaram de existir: o corpus passou a sair do próprio ambiente de referência.)*
5. Trocar o codificador pela **segunda melhor** opção registrada em G0.

#### Evidência qualitativa: a fig. 7.1

Reproduzir a fig. 7.1 — 4 agentes, 8 sensores, trajetórias com o raio de comunicação como contorno semitransparente. Lamenza observa que as trajetórias **quase se sobrepõem perfeitamente**: sem incentivo à coordenação, todos voam ao sensor livre mais próximo e escolhem o mesmo. Para não depender de inspeção visual, medir:

> **Índice de sobreposição `Ω`** = média, sobre os passos do episódio, da fração de pares de UAVs a distância `≤ communication_range` um do outro.

Predição pré-registrada: `Ω` **alto** em 1a e **baixo** em 1b (T-06.2), com a mesma célula `(Na, Ns)`. Se isso se verificar, a coordenação emergente de Lamenza foi reproduzida **com aproximador sem peso** — e essa é uma das afirmações mais fortes que a Etapa 1 pode fazer, por um custo marginal próximo de zero.

---

### T-06.2 — Réplica 1b: tempo de conclusão (Lamenza §7.2), célula base `Na = 1, Ns = 12`

**Configuração:** recompensa **eq. 5-6** verbatim, observação completa (eq. 5-1), `max_episode_length = 500 s`, `max_seconds_stalled = 30 s`, `scenario_size = 100`, `γ = 0,99`, `N*_s = 12`, `N*_a = Na − 1`.

**Nota estrutural útil:** com `Na = 1`, `N*_a = 0`, e portanto a observação de 1b é **estruturalmente idêntica** à de 1a. A diferença entre as duas na célula base é **só a recompensa e o horizonte**. Isso é uma ablação limpa de graça: o efeito da eq. 5-6 contra a eq. 7-2, com todo o resto constante. Reportar.

**Baselines pareados** (PRD §6.1), todos sobre os mesmos 500 episódios:

| # | Baseline | Papel |
|---|---|---|
| 1 | caminhada aleatória | piso — define a resolução mínima de qualquer afirmação |
| 2 | **guloso pelo sensor não visitado mais próximo** | o baseline de Lamenza |
| 3 | k-means + TSP por UAV, offline, com posições verdadeiras | teto quase-ótimo |

**Métricas:** `SR`, `CT_cens`, `CT_succ`, `Ω`, mais os dois níveis de IC da §4.6.

#### Uma ressalva sobre comparabilidade com os números publicados

- Lamenza **não publica `Na = 1`**. Para essa célula não existe alvo externo; o referencial é o guloso **reimplementado por nós**.
- Os tempos de conclusão do §7.2 estão em **figura** (fig. 7.3), não em tabela. A única tabela numérica de tempo é a **tab. 7.3**, que pertence ao §7.4.2 e ao modelo de **sensores variáveis** — outra campanha, outro `α`. Usá-la como alvo do §7.2 seria comparar coisas diferentes.
- Portanto: **a comparação rigorosa é a razão `CT_cens(método)/CT_cens(guloso)` por célula**, não o valor absoluto em segundos. A razão cancela simulador, implementação e `communication_range`; o valor absoluto não cancela nada. Os valores absolutos vão na tabela como referência, com a ressalva escrita.

> **P1 fechada, e a verificação de sanidade foi executada.** `communication_range = 20 m` (`environment.py:157`), e `algorithm_iteration_interval = 0,5 s`, não os 2 s da tab. 7.1. A verificação que esta spec propunha foi feita de forma mais forte: o **guloso de referência reproduz a fig. 7.3 em 9/9 células**, erro mediano 1,7 % e viés médio +0,5 %. Ver `../reports/spec00-parametros.md` §8. A comparação rigorosa continua sendo por **razão contra o guloso**, não por segundos absolutos — porque o `visualize_performance.py` avalia o MADDPG a 0,5 s por decisão e o guloso a 0,1 s, uma assimetria que vale ~10 %.

---

### T-06.3 — Comparação pareada WNN vs MLP (o experimento de H1)

**Igual dos dois lados:** ambiente, cenário, banco `test`, banco `val`, observação de Lamenza (eq. 5-1), recompensa (eq. 5-6), algoritmo (Actor-Critic on-policy com *parameter sharing*), `γ`, `σ²` da política, `B_max`, protocolo de parada, `R = 5` sementes, protocolo de avaliação.

**Diferente:** só o aproximador. `n-tuple` (config vencedora de T-06.4) vs `MLP 2 × 32` (Katopodis §5.4).

**O confundimento que precisa ser nomeado.** A WNN recebe a observação **binarizada**; a MLP recebe os reais normalizados. Isso é o desvio único da §5.4 do PRD — mas significa que a diferença medida é `aproximador + binarização`, não `aproximador`. Duas atitudes possíveis, e a spec escolhe as duas:

1. **Declarar.** A tabela e o texto dizem "aproximador **e sua codificação**", que é a unidade tecnológica real e é o que seria embarcado.
2. **Separar, se couber.** Terceira coluna **MLP-bin**: a mesma MLP `2 × 32` alimentada com **os mesmos bits** da WNN. Custo marginal baixo (a infraestrutura das duas já existe), e é o que isola binarização de aproximador. Fica como **primeiro item da linha de corte interna** desta tarefa — corta antes das varreduras, mas depois de tudo o mais.

**Pareamento do orçamento de sintonia.** Um pareamento em que a WNN foi varrida em 15 pontos e a MLP em 2 não é pareamento; é vantagem de esforço disfarçada de vantagem de método. Regra: **os dois lados recebem o mesmo número de pontos de varredura e o mesmo `B_max`**, e o orçamento gasto por lado é reportado na tabela. Para a MLP, os pontos são `η ∈ {1e−4, 3e−4, 1e−3} × σ² ∈ {0,1; 0,2; 0,5}`, selecionados pelo banco `val` exatamente como os da WNN.

**Saída:** a tabela pareada de H1 — `CT_cens` com `Δ` e IC nos dois níveis, `SR`, `Ω`, `T1`, `T2`, `T3`, `falhas/R`, tempo de parede de treino (total e parcela de aprendizado), orçamento de sintonia gasto.

---

### T-06.4 — Varreduras confirmatórias no treino

A SPEC-01 (G0) já pré-selecionou esquema, faixa de `T` e o par `(n, N)` **fora do laço de RL**, por proxies supervisionados. Esta tarefa **confirma no treino** — porque o proxy é mais fácil que o RL e o próprio PRD §5.4 exige que `T` seja "varrida e reportada".

**A grade completa não cabe, e isso é aritmética, não preguiça:** `T(3) × n(3) × N(3) ×` esquema`(2) × φ(4)` = **216** pontos; a `R = 3` e ao `B_max` de T-06.0, excede o orçamento da semana 2 em mais de uma ordem de grandeza.

**Protocolo adotado: descida coordenada a coordenada**, partindo do ponto pré-selecionado em G0, em **ordem fixa e pré-registrada**, ordenada por (custo baixo, risco alto) primeiro:

| Ordem | Eixo | Valores | Por que nessa posição |
|---|---|---|---|
| 1 | fator de esquecimento `φ` | `{1,0; 0,999; 0,99; 0,9}` | RNF-04: é o que trata a não-estacionariedade e o que também **apaga**. É o eixo com maior chance de causar falha catastrófica, e é barato |
| 2 | `T` | `{8, 32, 128}` | exigido pela §5.4; é o eixo que a Etapa 2 vai reusar (regra *bin ≈ σ*) |
| 3 | `N` | `{16, 48, 128}` | o joelho da curva de memória — retorno decrescente já medido (8× memória, −13 % erro) |
| 4 | `n` | `{8, 10, 12}` | governa `2^n` e a ocupação; o mais caro de mover |
| 5 | esquema (simples vs circular) | 2 | só executado se G0 tiver deixado os dois vivos |

Total: `4 + 3 + 3 + 3 + 2 = 15` pontos, `−4` já visitados ⇒ ~**11 pontos novos** × `R = 3` = **33 corridas** a `B_sweep = B_max/4`. Os **2 finalistas** são re-executados a `R = 5` e `B_max` completo.

**Segunda passada.** Após a descida, repetir os **dois eixos de maior efeito medido**. Se o ótimo não se mover, declarar estável. Se se mover, reportar a **interação** — e não afirmar independência dos eixos. A limitação da busca coordenada (não enxerga interação de ordem alta) fica **declarada no relatório**, não escondida.

> **~~Confundimento `N` × `η`~~ — DISSOLVIDO, e por isso a varredura fica mais simples.** A versão anterior avisava que `‖φ(x)‖² = N` faz o passo efetivo ser `η·N`, logo variar `N` com `η` fixo mediria taxa de aprendizado. Isso vale para a convenção de **soma**; a eq. (4.12) do artigo usa a **média**, e a eq. (4.22) normaliza o mapa por `1/√N`, de modo que `‖φ(x)‖² = 1`. **O passo efetivo é exatamente `η`, independente de `N`** (testado em `test_ator_usa_media_e_o_passo_independe_de_N`). Não há reparametrização a fazer.
>
> **O confundimento real é outro, e foi medido.** O que limita `α` é a variância do gradiente de política: o pior erro entre 5 sementes cresce **monotonicamente** com `α` (0,044 em 0,002 contra 0,238 em 0,05, com divergência em 0,05). Logo `α` é **eixo de primeira classe com múltiplas sementes**, não um parâmetro derivado de `N`.

**Seleção** sempre pelo banco `val`, nunca pelo `test`.

**Saída:** por eixo, figura de `CT_cens` (banco `val`) com **todas as corridas visíveis**, não só a média; tabela de entradas ocupadas nas LUTs × `N` (o joelho da curva, PRD §6.1); tabela `falhas/R` por ponto — porque o eixo `φ` provavelmente aparece mais na estabilidade do que na média.

---

### T-06.5 — Escalada `Na = 1 → 2 → 4 → 8`, depois `Ns ∈ {24, 36}`

Com a configuração vencedora de T-06.4, `Ns = 12`, `R = 5` até `Na = 4` e `R = 3` em `Na = 8`.

**Regra de escalonamento (aplica o PRD §8):** cada célula só começa quando a anterior fecha. Se `Na = 4` falhar catastroficamente em `≥ 3` das 5 corridas, **registrar e não gastar orçamento em `Na = 8`** — é o item 2 da linha de corte. Idem para `Ns ∈ {24, 36}`, que são item 2 e caem antes de `Na = 8` se o orçamento apertar.

**Atenção técnica.** A observação cresce com `Na` (25 → 27 → 31 → 39 escalares), e portanto o número de bits `B` cresce também. `(n, N)` foi selecionado sobre `B` da célula base. Verificar por T-01.4 (utilização de endereços) se a seleção continua válida em `B` maior; se a ocupação por RAM cair muito, é capacidade morta e `(n, N)` precisa ser reconfirmado — o que é um custo de campanha a prever, não uma surpresa.

**Comparação com Lamenza.** O que se compara é a **razão `CT_cens(WNN)/CT_cens(guloso)` por `Na`** e o **sinal da tendência**: Lamenza mede que a vantagem sobre o guloso **cresce com `Na`**, com retorno decrescente entre 4 e 8 agentes (fig. 7.3, explicado pela fig. 7.4i, em que 2 dos 8 agentes não contribuem). Reproduzir a tendência é o resultado; reproduzir os segundos absolutos não é exigido nem interpretável (T-06.2).

**Evidência qualitativa:** a réplica da fig. 7.4 — trajetórias por célula — mais o índice `Ω` da T-06.1, agora esperado **baixo**. `Ω` alto em 1a e baixo em 1b, na mesma célula, é a medição da divisão de trabalho emergente **sem sinalização** (PRD D5).

---

### T-06.6 — Tabela de custo completa da §5.5 (T1/T2/T3)

**O harness é da SPEC-03.** Esta tarefa **executa** o harness sobre as configurações finais e produz a tabela. O PRD §8 é explícito: esta tabela **não pode cair**, e é a única contribuição que sobrevive mesmo se todos os resultados de desempenho forem medianos.

**Configurações medidas:** WNN vencedora por `Na ∈ {1,2,4,8}`; WNN segunda melhor (a opção reserva registrada em G0); MLP pareada; MLP-bin, se executada; guloso (para a parcela comum de T2); ator do MADDPG (**só T1**, derivado de P4 — se P4 não fechar, reportar como estimativa rotulada, com o padrão CleanRL 256×256 declarado como suposição).

**T1 — tamanho.** Ator e ator+crítico; **alocado / ocupado / realizado**; mapeamento das `N × n` tuplas contabilizado (não é desprezível: ~1,2 KB em int16 com `N=48, n=12`, mais de 20 % do total); limiares do codificador, se o binning não for uniforme (E1/E2/E3 pagam, E0 não); fp32 **e** int16 com a política int16 **avaliada** (RNF-09 — reportar o par `(tamanho, CT_cens)`, jamais o tamanho int16 ao lado do desempenho fp32).

**RNF-08 — curva de ocupação.** Já vem de graça: a ocupação é registrada em `metrics.parquet` a cada avaliação intercalada (§4.7). Ajustar e verificar saturação dentro de `B_max`. **Se não saturar, a alegação de "kilobytes" vai com a ressalva do horizonte medido**, explicitamente.

**T2 — latência.** Modelo isolado **e** ponta a ponta; mediana **e p99** (RNF-10); decomposta pelos quatro estágios da §5.5; com e sem arcabouço (RNF-11 — NumPy puro dos dois lados, ou duas colunas); hardware declarado, um núcleo, sem GPU para nenhum lado (RNF-13). **Se a WNN perder no p99** por causa do hash, medir também a variante de LUT densa e reportar o par — é risco previsto no PRD §9, não descoberta embaraçosa.

**T3 — contagem de operações** por decisão e por tipo: MACs em ponto flutuante, lookups inteiros, somas. Derivação analítica **e** contador instrumentado, com os dois números batendo — se não baterem, a derivação está errada e é ela que vai ao relatório corrigida.

**Razões contra ambas as referências** (D12): MLP pareada (~2 150 params, ~8,6 KB) **e** ator do MADDPG. Sem escolher a conveniente.

---

### T-06.7 — Veredito de H1, decisão de portão e relatório da etapa

Redigir o veredito com **o número que o sustenta** (PRD §11.6). Os três desfechos e suas consequências são **pré-registrados aqui**, antes de qualquer resultado:

| Desfecho | Condição | Consequência |
|---|---|---|
| **Confirmada** | `CT_cens` da WNN não é pior que o da MLP além do IC, **nos dois níveis** (§4.6); ator WNN em poucas dezenas de KB **realizados** e menor que o ator do MADDPG; T3 mostra caminho de inferência **sem MACs em ponto flutuante** | segue para SPEC-07/08 com a política congelada como degrau 0 |
| **Refutada** | qualquer uma das três condições falha com evidência clara | **é o principal resultado da etapa** (PRD §6.1). Registrar veredito, diagnosticar (resolução? tamanho de tupla? `σ²`? crítico?) e **prosseguir** |
| **Inconclusiva** | os dois níveis de IC discordam em sinal (D-06.2), ou `falhas/R ≥ 2/5` impede caracterizar o método | reportar como inconclusiva **por variância de treino**, com `falhas/R` como o número principal |

> **Se G1 não for atingido — a regra do PRD, escrita como procedimento.** Não é motivo para estender a Etapa 1 para dentro da semana 3: a regra de escalonamento do §8 aciona a linha de corte **da etapa seguinte**, nunca prolonga a atual. H2, H3 e H4 medem **degradação relativa** ao próprio desempenho sem ruído e permanecem válidas mesmo com política medíocre.
>
> **Piso de mensurabilidade (D-06.3).** Há um limite para isso: se a melhor política disponível for **indistinguível da caminhada aleatória** com IC 95 %, não há resolução para medir degradação relativa — degradar ruído sobre uma política aleatória mede zero contra zero. Nesse caso, e só nesse, a Etapa 2 **muda de eixo primário**: passa a ser avaliada por **erro de estimativa e consistência NEES/NIS** (que não dependem da política ser boa), H3 é avaliada sobre a qualidade do filtro, e H2 e H4 são declaradas **inconclusivas por piso de desempenho**, com a razão escrita. O degrau 0 da escada passa a ser a **heurística gulosa sob observação ruidosa**, que continua sendo uma pergunta de fusão legítima.

**Relatório da etapa** (regra 4 do `specs/README.md`): 4–6 páginas com todas as figuras e tabelas, pronto para entrar como o capítulo de reprodução de E2 — sem re-medição posterior.

---

## 6. Critério de saída — portão G1

Cada item **medido**, não implementado.

1. **P2 fechada.** Vazão medida por célula e decomposta em simulação / observação / codificação / aprendizado; fidelidade de `update_rate = 0,1` verificada contra `0,01`; veredito escrito sobre a estimativa de 8–30 h por 50 M passos, com o fator de erro; `B_max` derivado e publicado.
2. **1a reportada** para `Na ∈ {1,2,4}` com `N*_a = 0`: `SR` mediana sobre `R` corridas, com IC de Wilson, contra os **100 %** publicados; `max_episode_length` permissivo **declarado numericamente** com a justificativa e a sensibilidade nos três horizontes; réplica da fig. 7.1 com `Ω` medido.
3. **1b reportada** para `Na ∈ {1,2,4}` (`Ns = 12`): `CT_cens`, `CT_succ` e `SR` contra guloso, aleatório e k-means+TSP, com IC bootstrap **pareado da diferença** e teste de permutação entre corridas.
   - "supera a caminhada aleatória decisivamente" ⇔ IC 95 % da diferença **inteiramente** favorável **e** razão `CT_cens(aleatório)/CT_cens(WNN) ≥ 2`;
   - "fica dentro do IC da heurística gulosa" ⇔ IC 95 % da diferença `WNN − guloso` **contém 0 ou é inteiramente negativo**.
4. **Tabela pareada WNN vs MLP fechada**, com tempo de conclusão, memória e latência, orçamento de sintonia gasto por lado registrado, e `falhas/R` de cada lado.
5. **Varreduras executadas** conforme o protocolo coordenada a coordenada, com a segunda passada, os 2 finalistas re-executados a `R = 5`, e a limitação da busca declarada.
6. **Tabela de custo da §5.5 completa**: T1 alocado/ocupado/**realizado** com mapeamento de tuplas, curva de ocupação com verificação de saturação, T2 mediana **e p99** decomposto com e sem arcabouço, T3 por tipo de operação, tudo contra **ambas** as referências e com a configuração `(T, n, N, Na)` que gerou cada número.
7. **`R ≥ 3` corridas de treino** em toda configuração reportada, com **nenhuma descartada** e `falhas/R` na tabela.
8. **Veredito escrito de H1** — confirmada, refutada ou inconclusiva — com o número que o sustenta e a decisão de portão registrada.
9. **`make repro` verde** para as tabelas e figuras desta spec, a partir dos `eval_test.parquet` e das seeds fixas.
10. **Teste de RNF-07 passando:** com `fusion/` desligado, o sistema executa o problema de Lamenza puro.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **Corrida única não distingue método ruim de semente ruim** (Lamenza §7.3: falha catastrófica na curva de 4 agentes) | `R ≥ 3`, `R = 5` no caminho crítico; nenhuma corrida descartada; `falhas/R` como métrica de primeira classe; regra de concordância de dois níveis (D-06.2) |
| **`B_max` insuficiente** — a vazão medida fica muito abaixo dos 460–1 700 passos/s implicados pela estimativa do PRD | T-06.0 é a primeira tarefa e bloqueia as demais; se apertar, acionar a linha de corte do PRD §8 **na ordem publicada**, e **nunca** reduzir `R` |
| **"Permissivo" é grau de liberdade que Lamenza não publicou** | definição operacional (`3 × CT_p99` do guloso, piso 500 s), verificação (`SR` do guloso ≥ 99 %) e sensibilidade obrigatória nos três horizontes |
| ~~`communication_range` (P1) errado desloca toda a comparação~~ | **fechada**: 20 m, e o guloso de referência reproduz a fig. 7.3 em 9/9 células. Segue valendo comparar por **razão contra o guloso**, pela assimetria de `interval` na avaliação publicada |
| **Comparação com a MLP mede aproximador + binarização juntos** | declarar a unidade como "aproximador e sua codificação"; coluna MLP-bin como separação, primeira da linha de corte interna de T-06.3 |
| **Pareamento injusto de esforço de sintonia** | mesmo número de pontos e mesmo `B_max` dos dois lados, com o orçamento gasto reportado na tabela |
| **Seleção de checkpoint contamina a métrica final** | bancos `val` e `test` disjuntos, disjunção testada; seleção só por `val` |
| **IC pareado por episódio subestima a incerteza** e faz o resultado parecer mais forte do que é | dois níveis obrigatórios; regra de concordância; veredito inconclusivo quando discordam |
| **`CT_succ` premia política que desiste** | `SR` sempre ao lado; `CT_cens` como métrica primária, que é também a convenção de Lamenza |
| **Escala do alvo de valor sob a recompensa 7-2** (`V ≈ 100`) estoura o crítico n-tuple | normalização do alvo por `(1−γ)` decidida e registrada **antes** da campanha; é hiperparâmetro do aproximador, não desvio da §5.4 |
| **Busca coordenada não vê interações** | segunda passada nos dois eixos de maior efeito; limitação declarada no relatório |
| ~~`N` varrido com `η` fixo mede taxa de aprendizado~~ **não se aplica** | A convenção de média da eq. (4.12) torna o passo independente de `N`. O risco real é **subestimar a variância de `α`**: mitigado varrendo `α` em faixa larga com múltiplas sementes |
| **`(n, N)` selecionado em `B` da célula base não vale para `Na = 8`** | reverificar utilização de endereços (T-01.4) ao mudar de célula; prever o custo na campanha |
| **Ocupação das LUTs não satura dentro de `B_max`** | RNF-08: curva obrigatória; se não saturar, "kilobytes" sai com a ressalva do horizonte |
| **WNN perde no p99 por causa do hash** | medir a variante de LUT densa e reportar o par (RNF-10) |
| **Política final indistinguível do aleatório inviabiliza a Etapa 2** | piso de mensurabilidade D-06.3: a Etapa 2 muda de eixo primário para erro de estimativa e NEES/NIS |
| **A tabela de custo é adiada por resultados de desempenho ruins** | ela é instrumentação, não experimento, e o PRD §8 a marca como não cortável — T-06.6 roda mesmo com H1 refutada |

---

## 8. Entregáveis

- `configs/etapa1/` — configurações resolvidas de 1a, 1b, varreduras e escalada, com `config_hash`
- `data/eval_banks/` — bancos `test` (500) e `val` (100) por célula `(Na, Ns)`, versionados com hash
- `runs/etapa1/**` — um diretório por corrida, conforme §4.7, com `eval_test.parquet` por corrida
- `results/spec06/` — `custo_T1.csv`, `custo_T2.csv`, `custo_T3.csv`, `pareado_h1.csv`, `varreduras.csv`, `escalada_na.csv`, `orcamento_simulacao.csv`
- `figs/spec06/` — curvas de treino com todas as corridas visíveis; `CT_cens` por eixo de varredura; curva de ocupação × passos; réplicas das figs. 7.1 e 7.4 com `Ω`; barras `CT_cens` vs guloso por `Na`
- `reports/spec06-etapa1.md` — 4–6 páginas com o **veredito escrito de H1** e a decisão de portão G1
- Atualização do `specs/README.md` (estado da SPEC-06) e do PRD §10 (P2 fechada; P5 registrada; P1 e P4 com o status que tiverem)

---

## 9. Perguntas em aberto

1. **Qual o valor de `max_episode_length` "permissivo" do §7.1?** A regra `3 × CT_p99(guloso)` é nossa e é defensável, mas é **nossa**. Vale abrir *issue* no repositório `Project-GrADyS/gradys-data-collection` (branch `dissertation`) ou perguntar ao autor? O parâmetro está na origem da comparação com os 100 % publicados. **Registrar como P5 no PRD §10.**
2. **`max_seconds_stalled` poda o episódio 30 s após a última coleta, mesmo com todos os sensores visitados?** Se sim, RF-02 ("não podar quando todos coletados") é efetivamente limitado a 30 s de experiência extra, e é isso que precisa ser replicado. Precisa de leitura do código de referência antes de 1a rodar.
3. **`R = 5` é suficiente?** É o mínimo para permutação exata com `p < 0,01`. Se a variância entre corridas for tão alta quanto a fig. 7.6 de Lamenza sugere, 5 pode não resolver diferenças modestas — e a alternativa (mais sementes, menos configurações) é uma troca de projeto que precisa de decisão humana, não de default.
4. **A coluna MLP-bin entra ou não?** É o que separa "efeito do aproximador" de "efeito da binarização", e a pergunta científica de H1 é sobre o aproximador. Custa uma campanha de treino a mais. Cortá-la significa aceitar que a unidade comparada é "aproximador + codificação" — o que é defensável, mas precisa ser decisão consciente e não omissão.
5. **`B_max` é derivado de um orçamento de parede `W` que ainda não foi alocado.** Quantos núcleos-hora a semana 2 pode consumir de fato? Sem `W`, T-06.0 produz uma fórmula, não um número.
6. **Normalizar o alvo de valor por `(1−γ)`:** aplicar em 1a apenas (onde a recompensa 7-2 gera `V ≈ 100`) ou também em 1b (onde a eq. 5-6 tem sinal misto e magnitude diferente)? Aplicar em um e não no outro complica a comparação 1a vs 1b da T-06.2.
7. **`σ²` da política gaussiana é hiperparâmetro fixo** (Katopodis: 0,5 e 0,2). Ele é varrido dentro do orçamento pareado, ou fixado antes por decisão? Se varrido, entra na conta de pontos dos dois lados; se fixado, precisa de justificativa escrita.
8. **Se `Na = 8` couber no orçamento mas `Ns ∈ {24, 36}` não** — ou vice-versa — qual eixo tem prioridade? O PRD §8 lista os dois no item 2 da linha de corte, sem ordenar entre si. `Na` responde H6 e a tendência de coordenação; `Ns` responde escala. Decisão humana.
9. **O k-means+TSP é acréscimo nosso e não existe em Lamenza.** Ele entra na tabela principal como terceira coluna, ou em apêndice? Como teto quase-ótimo ele contextualiza o resultado, mas também expõe a distância até o ótimo, o que muda o tom do capítulo.
10. **Se as varreduras confirmarem exatamente a pré-seleção de G0**, o custo delas foi de confirmação. Vale reduzir o protocolo (só `T` e `φ`, item 1 da linha de corte) já na largada, aceitando o risco de a SPEC-01 ter previsto errado — e ganhar orçamento para mais sementes?
