# SPEC-01 — Estudo de binarização

**Etapa:** pré-Etapa 1 · **Depende de:** SPEC-00 (ambiente carregado) · **Estimativa:** 3 dias
**Revisada** após a SPEC-00: o gerador analítico foi **descartado** e três premissas caíram — ver §0.
**Portão de saída:** `G0` · **Referência:** PRD §5.2 (observação), §5.4 (desvios), §5.5 (custo)


## 0-ter. Revisão pela dissertação (2026-08-15)

**O nosso codificador não é o da dissertação, e isso é deliberado.** A eq. (3.23)
usa limiares **uniformes** sobre `[l, u]` com comparação **estrita**; nós usamos
quantis por slot com `>=`. A medição desta spec justifica o desvio (4,3° contra
7,6°), mas o relatório precisa dizer que aqui **não** somos fiéis ao artigo.

Os codificadores literais estão agora em `src/uavwnn/encoding/artigo.py`
(`TermometroArtigo`, `CircularArtigo`), com testes contra as eqs. (3.23) e
(3.24). Servem de referência de fidelidade, não de substitutos.

**Três achados que mudam esta spec:**

1. **`R+1` palavras-código.** Um termômetro de resolução `R` tem exatamente
   `R + 1` padrões distintos. É a medida exata do aliasing que `R` impõe, e o
   teto que a varredura de `T` deve usar como referência.
2. **O eixo da tupla é `n/B`, não `n`.** As dimensões de observação do artigo
   estão no §5.1; `n/B` dele vai de 0,042 a 0,125, e o nosso `n/B = 0,060` fica
   dentro. Ver **P11** no PRD — ela foi rebaixada de bloqueio para refinamento.
3. **O circular quebra a convenção de padding.** Ele acende sempre `⌊R/2⌋`
   bits, logo **não existe padrão todo-zero**. A propriedade que dispensa o bit
   de validade não sobrevive a ele. As tarefas do artigo não têm slot vazio; a
   nossa tem.

**Se esta spec for revisitada**, o candidato sério é o circular para as
dimensões **angulares**, onde o wrap-around é a topologia correta em vez de
apenas conveniente. Replicação de §5.2.2 em `reports/replica-katopodis-52.md`:
vence em baixa resolução, empata na configuração do artigo.

---

---

## 0. O que mudou depois da SPEC-00

A SPEC-00 leu o código de referência e mediu o ambiente. Quatro premissas desta spec caíram:

| # | A spec supunha | O ambiente entrega |
|---|---|---|
| 1 | Entrada em metros crus, faixa `[-200, 200]` | **Já normalizada**: `(p_k − p_i + S)/(2S)`. Faixa **real medida `[-0,5; 1,5]`** — nem `[0,1]`, nem o `Box(-1,1)` declarado |
| 2 | Padding `(0,0)`, ambíguo com "sensor sobre o UAV" — daí o **bit de validade** | Padding **`-1`**, fora da faixa dos válidos. Ambiguidade não existe; **o bit de validade foi removido do escopo** |
| 3 | Corpus vindo de um **gerador analítico** sem simulador, validado por KS (invariante V-1) | **Desnecessário.** O ambiente de referência roda a 0,02–0,15 s por episódio: gera-se o corpus direto dele. V-1 **deixa de existir**, e com ele o maior risco desta spec |
| 4 | `T` e `N` varridos independentemente | **Não são independentes** (achado C4): com `N·n = 576` contra `B ≈ 3200` em `T=128`, a rede lê ~18 % da entrada. Aumentar `T` sem aumentar `N` reduz a fração observada |

O índice do agente entra como `k/Na`, já em `[0,1)` — é categórica disfarçada de escalar, e o tratamento continua sendo pergunta desta spec.

**Consequência prática:** a spec fica mais curta e menos arriscada. Some o gerador analítico, some o teste KS, some o bit de validade; entra a cobertura de bits como métrica de primeira classe.

---

## 1. Objetivo

Escolher, **com evidência medida**, o codificador binário que alimenta a rede n-tuple: esquema, resolução `T`, tratamento dos casos degenerados, e uma pré-seleção de `(n, N)`. Antes de treinar qualquer política.

## 2. Por que antes do treino

A rede n-tuple só enxerga bits: toda a informação passa por um endereço de RAM. Katopodis §3.2.3 é explícito — o codificador precisa **preservar similaridade**, mapeando entradas próximas para padrões com pequena distância de tupla — e dedica seções inteiras (§5.2.2, §5.3.2) ao efeito do esquema de codificação sobre o aprendizado.

O argumento prático é mais direto: **um codificador ruim é indistinguível de "o RL não converge".** Diagnosticar isso dentro do laço de treino custa dias por iteração. Fora dele, custa horas, sem simulador e sem RL.

## 3. Escopo

**Dentro:** distribuição da observação de Lamenza; esquemas de codificação candidatos; resolução; geometria da representação binária; utilização de endereços; dois proxies supervisionados; custo do codificador; sensibilidade a ruído de entrada.

**Fora:** qualquer treino de política; escolha de algoritmo de RL; o filtro de Kalman (SPEC-07). A tarefa T-01.8 adianta a interação resolução × ruído, mas **sem** filtro — a varredura com estimativa filtrada é da SPEC-08.

---

## 4. Corpus — direto do ambiente de referência

O gerador analítico foi descartado (§0). O corpus vem do **ambiente de referência**, o que elimina de uma vez o maior risco desta spec: otimizar o codificador para a distribuição errada. O invariante V-1 e o teste KS deixam de existir.

Custo medido (SPEC-00): 0,02 a 0,15 s por episódio, ~60 a 100 passos de decisão por episódio, com 1 a 8 agentes. Um corpus de 10⁶ observações sai em minutos.

| Conjunto | Origem | Papel |
|---|---|---|
| `on-policy` | rollouts do **guloso de referência** (`heuristics.py`) | é a distribuição que a política vai encontrar |
| `exploratorio` | rollouts da política **aleatória** de referência | cobertura; evita ajustar o codificador ao guloso |

Células: `Na ∈ {1, 2, 4}` × `Ns ∈ {12, 24, 36}`. Sementes fixas via `run_episode(..., seed=i)`; corpus regenerável por script, não versionado.

**Estratificação obrigatória.** A fração de sensores restantes varia muito ao longo do episódio, e com ela a distribuição da observação — no fim do episódio a maioria dos slots é padding. Todas as métricas abaixo são reportadas **estratificadas por fração restante**, não só agregadas.

---

## 5. Tarefas

### T-01.1 — Caracterização da distribuição de entrada

Para cada componente escalar da observação — `(dx_j, dy_j)` de cada slot `j` de `P^s`, idem `P^a`, e o índice `k` — levantar histograma, quantis e suporte efetivo.

**Hipótese central a testar:** a distribuição depende fortemente do **índice do slot**. O sensor não visitado mais próximo (`j = 1`) concentra perto de zero; o 12º espalha por todo o quadrante. Se confirmado, **um termômetro global uniforme desperdiça a maior parte dos bits**: quase toda a massa de `j = 1` cai num punhado de bins, e a resolução onde ela importa é justamente a menor.

Métrica: entropia da distribuição quantizada, por slot, sob binning global uniforme vs. binning por slot. Saída: figura e tabela; decisão explícita sobre binning por slot.

### T-01.2 — Esquemas candidatos

| id | Esquema | Observação |
|---|---|---|
| **E0** | termômetro linear global | bins uniformes na faixa medida `[-0,5; 1,5]`, iguais para todos os slots — a linha de base ingênua |
| **E1** | termômetro linear **por slot** | limites do suporte empírico de cada slot |
| **E2** | termômetro **distributivo** (quantis) por slot | bins equiprováveis segundo o corpus |
| **E3** | E2 + **polar** `(distância, ângulo)`, ângulo com codificador **circular** | ver ressalva abaixo |

Varredura: `T ∈ {4, 8, 16, 32, 64}` bits por escalar **× `N ∈ {16, 48, 128}`** — os dois eixos **juntos**, nunca separados, pela razão de cobertura de bits (§0, item 4 e T-01.4).

**Tratamento do padding `-1`.** Ele fica fora da faixa dos válidos, então o termômetro o mapeia naturalmente para "tudo zero" (ou "tudo um", conforme o sentido). Isso é o comportamento desejado e é o que dispensa o bit de validade — mas precisa ser **verificado**, não presumido: teste de que o padrão binário de um slot de padding é constante e distinto do padrão de qualquer valor válido.

> **Ressalva sobre E3.** Converter `(dx, dy)` para `(d, θ)` é bijeção — a informação disponível ao agente é idêntica, e a conversão acontece **dentro do codificador**. Ainda assim, muda o vetor que a rede recebe e portanto **exige uma linha nova na tabela de desvios do PRD §5.4 se for adotado**. Há razão para testá-lo: o codificador circular de Katopodis existe exatamente para grandezas angulares, e sob distância de tupla a representação cartesiana degrada mal sob rotação. Há razão para desconfiar: Lamenza testou "ângulos e distâncias em vez de coordenadas" e mediu **pior** — para MLP. É uma pergunta empírica aberta e o mérito do estudo é respondê-la barato.

### T-01.3 — Geometria: preservação de similaridade

Amostrar pares de observações `(o, o′)`; medir `d_obs = ‖o − o′‖₂` normalizada e `d_tuple` = distância de tupla entre `φ(o)` e `φ(o′)` (Katopodis eq. 3.17).

Métricas:
- **ρ de Spearman** entre `d_obs` e `d_tuple` — monotonicidade
- curva `d_obs × d_tuple` com bandas de dispersão
- **fração de pares saturados** (`d_tuple = N`, isto é, sem nenhuma RAM em comum): saturação é morte da generalização, porque duas observações saturadas são "infinitamente distantes" independentemente de quão próximas estejam de fato

**Faixa que importa:** a de `d_obs` que ocorre entre **passos consecutivos** de um episódio. É essa a faixa em que o TD faz bootstrap de `V(s′)` a partir de `V(s)`; se a codificação satura aí, o crítico não tem como propagar valor. Reportar a distribuição de `d_obs` entre passos consecutivos e sobrepô-la à curva.

### T-01.4 — Utilização de endereços

Por RAM: quantos dos `2^n` endereços são efetivamente visitados no corpus, entropia da distribuição de endereços e índice de Gini.

Varrer `n ∈ {8, 10, 12, 16}` × `N ∈ {16, 48, 128}`.

Isto **prediz T1 (ocupação de memória, PRD §5.5) antes de qualquer treino**, e identifica os dois regimes ruins: subutilização (a maioria das RAMs vê poucas dezenas de endereços — capacidade morta) e explosão (ocupação cresce sem saturar — memorização, e a alegação de "kilobytes" cai).

### T-01.5 — Proxy supervisionado A: clonagem do guloso

**É o filtro decisivo desta spec.**

Alvo: a ação da heurística gulosa `a = (θ, s)` na observação `o`. Regredir `θ` como `(cos θ, sin θ)` para evitar a descontinuidade em `2π`.

Modelo: regressor n-tuple já existente (`ntuple.py`), com capacidade idêntica em todos os esquemas comparados.

Métricas: erro angular mediano e p90 (graus); acurácia de "escolheu o mesmo sensor-alvo que o guloso".

Racional: o guloso é determinístico, simples e é o baseline que a política precisa bater. **Se a codificação não sustenta nem clonar o guloso, ela não vai sustentar RL** — e descobrir isso agora custa horas em vez de uma semana de treino.

### T-01.6 — Proxy supervisionado B: regressão de valor

Alvo: retorno Monte Carlo sob a política gulosa, com a recompensa de Lamenza (eq. 5-6) **verbatim** e `γ = 0,99`.

Métricas: RMSE normalizado e — mais informativo — **erro relativo entre estados consecutivos**, que é a quantidade de que o TD depende.

Testa o que o **crítico** vai precisar, que é função diferente da do ator: valor é suave e de longo alcance, ação é local. Um codificador pode servir a um e não ao outro.

### T-01.7 — Custo do codificador

Alimenta diretamente PRD §5.5:

- tempo de codificação por observação (mediana e p99) × `T` × esquema
- número de bits `B` resultante
- tamanho dos limiares armazenados (parte de **T1**, se o binning não for uniforme — E1/E2/E3 pagam esse preço, E0 não)

**Ponto de atenção:** se a codificação dominar o tempo de inferência, a vantagem de latência da WNN é do codificador e não da rede, e o relatório precisa dizer isso.

### T-01.8 — Resolução × ruído (opcional; adianta a Etapa 2)

Corromper a observação com `σ ∈ {0, 1, 3, 5, 10} m` **antes** de codificar; medir a degradação dos proxies A e B por `T`.

Saída: a regra *largura do bin ≈ σ* (PRD §1) instanciada sobre a distribuição real da observação, e uma tabela `T*(σ)` recomendada. Reduz o custo da SPEC-08, que é o experimento mais caro do projeto.

---

## 6. Critério de saída — portão G0

1. **Um esquema e uma faixa de `T` escolhidos**, sustentados por T-01.3 (geometria) e T-01.5 (clonagem). Escolha justificada por número, não por preferência.
2. **`(n, N)` pré-selecionados** por T-01.4, com a ocupação prevista já estimada.
3. **As duas melhores opções registradas**, não só a vencedora — ver riscos.
4. **PRD §5.4 atualizado** se E3 for adotado.
5. **Relatório curto** (2–3 páginas, com as figuras) pronto para entrar no capítulo de método de E2.
6. **Testes automatizados:** codificador determinístico; monotônico na entrada; tamanho de saída constante para configuração fixa; bit de validade correto em slots com padding; invariante V-1 registrado como teste pendente da SPEC-00.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **Corpus da distribuição errada** (gerador analítico ≠ simulador) | invariante V-1: teste KS assim que a SPEC-00 entregar; refazer se divergir. É o risco mais caro desta spec |
| **P1 não resolvido** (`communication_range`) | rodar 2–3 valores candidatos e reportar sensibilidade; se as conclusões mudarem, escalar P1 como bloqueante |
| **O proxy não prediz o RL.** Clonar o guloso é tarefa mais fácil que aprender por reforço | registrar as **duas** melhores configurações; a SPEC-06 tem orçamento para a segunda se a primeira falhar no treino. Não tratar G0 como veredito final |
| **Sobreajuste do codificador ao guloso** | corpus uniforme além do on-policy; reportar as métricas nos dois |
| Binning por slot deixa de valer quando restam poucos sensores não visitados (slots viram padding) | avaliar as métricas estratificadas por fração de sensores restantes, não só agregadas |
| E3 (polar) vence no proxy mas é rejeitado por alterar o problema | decisão registrada antecipadamente: se vencer com margem, adota-se **com** a linha na §5.4 e a comparação cartesiana mantida como ablação no relatório |

---

## 8. Entregáveis

- `src/uavwnn/encoding/thermometer.py` — linear, distributivo, circular
- `src/uavwnn/encoding/observation.py` — observação de Lamenza (eq. 5-1) → bits, com bit de validade
- `src/uavwnn/env/analytic.py` — gerador de cenários e rollout guloso sem simulador
- `configs/spec01/` — varreduras
- `reports/spec01-binarizacao.md` + figuras
- `tests/test_encoding.py`
