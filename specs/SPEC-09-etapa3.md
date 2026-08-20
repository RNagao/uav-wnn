# SPEC-09 — Etapa 3: multiagente com ruído, interseção de covariância

**Etapa:** 3 · **Depende de:** SPEC-08 (G2), SPEC-06 (G1), SPEC-03 (harness de custo) · **Estimativa:** 4 dias
**Portão de saída:** `G3` · **Referência:** PRD §2 (H5, H6), §5.2 (MDP, parameter sharing, D5), §6.3 (sub-passos 3a/3b, degraus 6a/6b/6c), §10 (D5, D10) · `tmp/PLANO.md` §9.5, §9.7 · Lamenza §7.1, §7.2, §7.4.2 · Julier & Uhlmann (1997)

---

## 1. Objetivo

Fechar as duas hipóteses da Etapa 3 com número:

- **H5** — fundir estimativas entre UAVs por **interseção de covariância** supera o KF local isolado e permanece consistente, enquanto o **KF ingênuo diverge** sob correlação desconhecida.
- **H6** — a coordenação **emerge** com aproximador sem peso: o enxame supera a heurística gulosa, e a divisão de trabalho é **medida**, não descrita.

E fazer isso separando limpamente três perdas que, se medidas juntas, viram uma coisa só: perda por **coordenação** (vários agentes disputando o mesmo alvo), perda por **incerteza** (o estado que a política enxerga é estimado) e perda por **inconsistência** (o estimador acredita mais em si do que deveria).

## 2. Por que

Esta é a única etapa em que duas dificuldades entram juntas, e por necessidade: **só há erro correlacionado a fundir se houver mais de um estimador**. Antes dela, a Etapa 2 mediu tudo com um único UAV, onde não existe o problema de fusão distribuída — existe filtragem.

Três razões tornam a etapa não trivial e justificam o cuidado do protocolo:

1. **O canal novo é qualitativamente diferente dos da Etapa 2.** A posição do vizinho chega por broadcast: é ruidosa, **atrasada** e — o ponto que interessa à disciplina — vem de um estimador cujo erro é **correlacionado** com o meu. Dois UAVs que observaram o mesmo sensor compartilham o viés daquele sensor; e a minha trilha do vizinho é construída a partir dos broadcasts anteriores *dele*, então fundir a estimativa atual dele com a minha trilha é reutilizar informação que já contabilizei. Isso é *data incest*, e o KF ingênuo não tem defesa contra ele.

2. **A fusão não paga só em metros.** A coordenação de Lamenza emerge de cada agente **conhecer a posição dos demais** (§7.2, p40) — não há sinalização (PRD D5). Se essa posição chega degradada, o mecanismo que sustenta a coordenação é justamente o que se degrada. A predição central desta spec é que **a fusão compra coordenação, não apenas erro de estimativa** — e ela é falsificável.

3. **O sub-passo 3a é não negociável (PRD D10).** Sem multiagente sem ruído, a degradação medida em 3b é a soma de dois efeitos e não se sabe qual pagou. 3a é o degrau 0 da etapa.

Uma observação de honestidade que atravessa a spec: Lamenza descreve a coordenação **visualmente** (fig. 7.1 mostra trajetórias sobrepostas sem incentivo; fig. 7.4 mostra trajetórias separadas com a recompensa 5-6; fig. 7.4i mostra dois dos oito agentes que "não contribuíram significativamente"). Olhar trajetória não é medida. **Propor uma métrica quantitativa de divisão de trabalho é uma contribuição desta spec, não um detalhe de instrumentação** (T-09.6).

## 3. Escopo

**Dentro:** modelo de ruído do canal de broadcast (atraso e correlação explícitos e parametrizados); trilha local de vizinho; fusão 6a/6b/6c; interseção de covariância com escolha de ω e custo medido; NEES/NIS por canal; métricas de divisão de trabalho; célula base `Na = 4, Ns = 24` com `Na ∈ {2, 8}` conforme o tempo; transferência 4×24 → {2×12, 8×36} sem retreino; vereditos de H5 e H6; portão G3.

**Fora:** qualquer mudança na observação, ação ou recompensa de Lamenza (PRD D1–D4); mensagem de reivindicação ou canal aprendido entre agentes (**D5**); crítico centralizado / CTDE; reprodução de MADDPG (D7); trilateração não linear por RSSI (P3 — caso linear, PRD §10); Split-CI, ICI e demais variantes menos conservadoras da interseção de covariância (registradas como trabalho futuro em §10).

**Herdado e congelado** ao entrar nesta spec: codificador e `(T, n)` escolhidos no G0 — com `N = ⌈B/n⌉` determinado pela partição, não escolhido; `α` fixado com asserção de finitude no passo efetivo (RNF-01 **reformulada**: não há regra `η = c/N` na convenção de média do artigo); KF com entrada de controle e filtro de informação validados no G2; harness de custo T1/T2/T3 da SPEC-03; baselines (aleatório, guloso, k-means+TSP) da SPEC-02.

---

## 4. Montagem: canais, ruído e o que exatamente se funde

### 4.1 Desambiguação de dois canais que o PRD trata como um

O PRD §6.3 descreve o canal novo como "posição dos vizinhos, recebida por broadcast — ruidosa, atrasada e com erros correlacionados (dois UAVs que observaram o mesmo sensor, possivelmente com histórico compartilhado)". A frase mistura **dois** problemas de fusão com fontes de correlação distintas. Esta spec os separa, porque o experimento só é interpretável separado:

| id | Canal | O que se funde | Origem da correlação | Entra na observação como |
|---|---|---|---|---|
| **C1** | **Trilha de vizinho** | minha trilha do UAV `j` × a estimativa que `j` transmite de si mesmo | **incesto**: minha trilha foi construída com os broadcasts anteriores de `j` | `P^a` (os `N*_a = Na − 1` vizinhos mais próximos) |
| **C2** | **Posição de sensor** | minha estimativa do sensor `i` × a estimativa que outro UAV transmite do mesmo sensor `i` | **viés comum** por sensor (mesmo emissor, mesma anomalia de propagação) + incesto por retransmissão | `P^s` (os `N*_s = 12` sensores não visitados mais próximos) |

Ambos alimentam a observação de Lamenza (eq. 5-1) — logo, ambos afetam a política. **C1 é o canal de destaque** (é o que o PRD chama de canal novo e é o que sustenta a coordenação, §2 item 2); **C2 é onde a correlação é mais fácil de instrumentar de forma controlada**, e por isso é o canal onde a demonstração de divergência do 6b é feita primeiro (T-09.4).

### 4.2 Conteúdo do broadcast — e por que isso não viola D5

O broadcast carrega `⟨ k, t_emissão, x̂_k, P_k ⟩`: índice do emissor, carimbo de tempo, estimativa de posição e covariância. Nada mais.

**Não há reivindicação, intenção, alvo, rota ou canal aprendido.** D5 proíbe sinalização de coordenação; o que trafega aqui é a mesma grandeza que o "canal global" de Lamenza entregava de graça (posição), agora degradada e acompanhada da sua incerteza. A covariância é metadado do estimador de **localização**, não de intenção.

Ainda assim, isto **é uma decisão a registrar**: transmitir `P` é acréscimo em relação ao que o problema original transmitia, e a interseção de covariância é impossível sem ele (CI precisa de `P_a` e `P_b`). Fica como pergunta Q1 (§10) e, se aprovada, como decisão nova na §10 do PRD — não como linha na tabela de desvios §5.4, porque a **observação da política não muda**: o que muda é a qualidade do número que preenche `P^a` e `P^s`.

> **Ablação obrigatória de honestidade.** Rodar 6c também com `P` **não** transmitida, usando um `P` conservador fixo assumido para o vizinho. Se a diferença for pequena, a transmissão de covariância não se justifica e o resultado é mais limpo. Se for grande, é o preço declarado da proposta.

### 4.3 Modelo de ruído — com os dois fatores explícitos e separáveis

O ponto de projeto: **se a correlação não for construída de propósito e parametrizada, o experimento 6b vs 6c é vazio** — o KF ingênuo diverge ou não conforme o acaso do modelo de ruído, e não se sabe por quê.

| Fator | Parâmetro | Valores | Papel |
|---|---|---|---|
| Ruído de GPS próprio | `σ_GPS` | herdado da Etapa 2: `{1, 10} m` (grade completa `{1,3,5,10}` se o tempo permitir) | ruído branco, filtrado pelo KF com entrada de controle (degrau 4) |
| Ruído do enlace local (sensor) | `σ_link` | herdado do G2 | mede a posição do sensor (caso linear, P3) |
| **Atraso de broadcast** | `τ_d` | `{0, 1, 2}` intervalos de decisão (0, 2, 4 s) | a 15 m/s, `τ_d = 2 s` já vale **30 m** de erro se ignorado |
| **Correlação** | `ρ` | `{0, 0,5, 0,9}` | coeficiente do componente de erro comum entre os dois estimadores |
| Período de broadcast | `T_bc` | 1 intervalo de decisão | fixo; sensibilidade só se sobrar tempo |

**Construção da correlação.** O erro de cada estimador é decomposto em `e = √ρ · e_comum + √(1−ρ) · e_próprio`, com `e_comum` sorteado **por sensor e por episódio** (C2) ou herdado do histórico compartilhado (C1, que é incesto estrutural e acontece sozinho — não precisa ser injetado, basta não impedir). Assim `ρ = 0` é um caso de controle honesto: se o KF ingênuo **também** falhar em `ρ = 0`, o modelo de ruído está enviesado a favor da proposta e precisa ser corrigido antes de qualquer conclusão.

**Reprodutibilidade do ruído sob braços diferentes.** As realizações de ruído são sorteadas de um fluxo indexado por `(semente_cenário, canal, agente, índice_do_passo)` e **não** pela trajetória. Como a corrupção é aditiva, o mesmo sorteio se aplica no mesmo passo, mesmo que os braços 6a/6b/6c produzam trajetórias diferentes. É redução de variância barata e torna o pareamento entre braços quase exato. Requisito de implementação, com teste.

**RF-04 estendido.** O invariante de vazamento vale também para o broadcast: o que sai do UAV `j` é `x̂_j`, **nunca** `x_j` verdadeiro. Teste automatizado obrigatório, análogo ao da Etapa 2.

### 4.4 Definição operacional dos degraus

| # | Configuração | Posição própria | Vizinho (C1) | Sensor (C2) |
|---|---|---|---|---|
| **3a** | multiagente **sem ruído** | verdade | verdade | verdade |
| **1** | ruído bruto | medida crua | broadcast cru, sem compensar atraso | medida crua |
| **4** | KF + entrada de controle | **KF (Etapa 2)** | broadcast cru | medida crua |
| **6a** | KF local isolado, **sem troca de estimativas** | KF | trilha local de `j`, atualizada com o broadcast tratado como **medida** com `R₀` conservador **fixo** (o `P_j` transmitido é **ignorado**) | filtro de informação local |
| **6b** | **KF ingênuo entre UAVs** | KF | funde a estimativa de `j` usando o `P_j` transmitido como se fosse independente: `P⁻¹ ← P⁻¹ + P_j⁻¹` | idem entre UAVs |
| **6c** | **interseção de covariância** | KF | `CI(trilha, estimativa recebida)` | — *(a coluna do canal do sensor caiu com D13)* |

Notas que fecham a definição:

- **6a é a referência correta**, e não "sem vizinho": o problema original *sempre* deu ao agente a posição dos demais. 6a mantém o canal, apenas se recusa a tratar a estimativa alheia como informação nova quantificada.
- Todos os degraus 6x são construídos **sobre o degrau 4** (KF com entrada de controle para a posição própria), que foi validado no G2. **O degrau 5 (covariância na observação) fica desligado por padrão** — o PLANO §9.7 define "6 = 5 + CI", mas o PRD §6.3 lista apenas 1, 4 e 6. Esta spec resolve a divergência tratando o degrau 5 como **fator opcional cruzado**, ligado só se H4 tiver sido confirmada no G2 e só na célula base. Ligá-lo acrescenta canais à observação e portanto **exige a linha na tabela de desvios §5.4** (PRD §6.2). Ver Q4.
- **Compensação de atraso** é a mesma nos três degraus 6x (propagar a estimativa recebida por `τ_d` com a velocidade estimada do vizinho e inflar `P` por `Q·τ_d`), para que a diferença medida entre eles seja **só** a regra de fusão. Uma variante que ignora o atraso entra apenas como fator do fatorial de T-09.4.

---

## 5. Tarefas

### T-09.1 — Sub-passo 3a: multiagente sem ruído (degrau 0 da etapa)

**Não negociável (PRD D10 e linha de corte da §8).** É a referência contra a qual toda degradação por ruído é medida.

Célula base `Na = 4, Ns = 24`, com `Na ∈ {2, 8}` conforme o tempo. Política n-tuple Actor-Critic com **parameter sharing** e o índice do agente `k` na observação (Lamenza p31), treinada por célula.

Produz:

1. Tempo de conclusão e taxa de sucesso, contra os três baselines (aleatório, guloso, k-means+TSP), 500 episódios de avaliação com sementes de cenário fixas e compartilhadas.
2. **A curva de retorno de escala** `T(Na)` para `Na ∈ {2, 4, 8}` a `Ns = 24`, e a eficiência de escalonamento `S(Na) = T(1)/(Na · T(Na))`. Predição de Lamenza (§7.2, p39): a vantagem sobre o guloso **cresce** com `Na`, mas com **retornos decrescentes** — a diferença entre 2 e 4 agentes é muito maior que entre 4 e 8. Reproduzir ou refutar esse padrão é resultado.
3. **Contagem de agentes ociosos**: número de UAVs responsáveis por menos de `1/(2·Na)` das coletas do episódio. Lamenza observa exatamente isso na fig. 7.4i (dois de oito agentes sem contribuição significativa). É a explicação candidata do retorno decrescente e agora vira número.
4. Todas as métricas de divisão de trabalho de T-09.6, **sem ruído** — é aqui que a linha de base da coordenação é fixada.
5. Tabela de custo T1/T2/T3 na célula multiagente (SPEC-03). Com parameter sharing, o ator embarcado é **um só**, independentemente de `Na`: isso é um argumento de custo que merece a linha explícita.

**Instabilidade de treino é resultado, não ruído de bastidor.** Registrar, por semente, se houve falha catastrófica de aprendizado (curva de retorno que não sai do lugar). Lamenza reporta o fenômeno na própria implementação (§7.3). Reportar a fração de sementes que falhou, não só a média das que deram certo.

### T-09.2 — Interseção de covariância: matemática, ω e implementação

**Formulação (Julier & Uhlmann, 1997).** Dadas duas estimativas `(x̂_a, P_a)` e `(x̂_b, P_b)` do mesmo estado, com correlação cruzada **desconhecida**:

```
P_CI⁻¹ = ω · P_a⁻¹ + (1 − ω) · P_b⁻¹
x̂_CI   = P_CI · ( ω · P_a⁻¹ x̂_a + (1 − ω) · P_b⁻¹ x̂_b )
ω ∈ [0, 1]
```

A propriedade que importa: **para todo `ω ∈ [0,1]` e para qualquer correlação cruzada real, `P_CI` é um limite superior consistente** da covariância do erro de `x̂_CI`. Essa é a diferença estrutural para o KF ingênuo, que corresponde a `P⁻¹ = P_a⁻¹ + P_b⁻¹` — a mesma combinação **sem** o vínculo `ω + (1−ω) = 1`, e portanto uma covariância que só é válida sob independência.

**Escolha de ω.** Três variantes, todas implementadas e comparadas em custo e desempenho:

| id | Critério | Como |
|---|---|---|
| **ω-tr** | minimizar `tr(P_CI)` | busca em grade `ω ∈ {0, 0,1, …, 1}` (K = 11) + refino por seção áurea opcional |
| **ω-det** | minimizar `det(P_CI)` | idem; é o critério clássico do artigo original e o mais adequado quando os eixos de incerteza são muito anisotrópicos |
| **ω-rápido** | aproximação fechada `ω ≈ tr(P_b) / (tr(P_a) + tr(P_b))` | uma divisão; usada como variante barata |

Para estado 2D, `det(P_CI)` como função de `ω` admite solução analítica (raiz de polinômio de 2º grau). **Implementar a solução analítica e validá-la numericamente contra a busca em grade** — se coincidirem dentro de tolerância, ela vira o padrão e o custo desaba. `tr(P_CI)` é convexa em `ω` no domínio relevante, o que torna a busca unidimensional bem comportada; registrar isso, mas não depender dele: a grade é robusta.

**Fusão de mais de duas estimativas.** Com `Na = 8`, um sensor pode ter estimativa de vários UAVs. Duas opções: (i) **CI em lote** com `Σ ω_i = 1` (otimização em simplex de dimensão `m−1`, cara), (ii) **CI sequencial aos pares** (barata, mas **dependente da ordem** e progressivamente mais conservadora). Padrão: sequencial, com ordem canônica por índice de agente para reprodutibilidade, e a dependência de ordem **medida** (permutar a ordem e reportar a dispersão do resultado). Se a dispersão for material, escalar para CI em lote na célula base apenas.

**Custo — e a ressalva desconfortável.** Integra o harness da SPEC-03 (T3, contagem de operações por tipo). Conta preliminar a confirmar por medição, estado 2D, `ω-tr` com K = 11:

| Item | Estimativa |
|---|---|
| Inversão de `P_a` e `P_b` (2×2) | ~ 20 flops |
| Por valor de `ω`: combinação + inversão + traço | ~ 20 flops → **~220 flops** com K = 11 |
| Reconstrução de `x̂_CI` | ~ 20 flops |
| **Por par fundido** | **~ 260 flops** (grade) · **~ 60 flops** (analítico/rápido) |
| Fusões por decisão, pior caso `Na = 8` | `N*_a = 7` vizinhos + até 12 sensores = **19** |
| **Por decisão** | **~ 5 000 flops** (grade) · **~ 1 100 flops** (analítico) |

Compare-se com os ~2 100–2 600 MACs de uma MLP 31→32→32→2 (PRD §5.5). **Com busca em grade, a interseção de covariância pode custar mais que a rede densa contra a qual estamos argumentando.** Isso é exatamente o tipo de número que o PRD manda medir e não esconder (RNF-05). Mitigações, todas mensuráveis: ω analítico; fundir apenas os `N*_a` vizinhos e não os 12 sensores; fundir a taxa reduzida (a cada `m` decisões). **Reportar o par (custo, desempenho) para cada mitigação** — a alegação "embarcável" tem de sobreviver ao filtro, ou ser reescrita.

**Testes:** consistência algébrica (`ω = 0` e `ω = 1` reproduzem os extremos); `P_CI ⪰` a covariância verdadeira do erro em Monte Carlo sintético com correlação conhecida e varrida; monotonicidade do conservadorismo em `ρ`; simetria (fundir A com B = fundir B com A, para CI em par).

### T-09.3 — Consistência: NEES/NIS por canal, e o critério certo para um estimador conservador

Herda a maquinaria da SPEC-07 (`fusion/consistency.py`) e a aplica aos canais C1 e C2, por degrau.

**NEES** por passo e por execução: `ε_t = e_tᵀ P_t⁻¹ e_t`, com `e_t = x_t − x̂_t` contra a verdade guardada só para métricas. Média sobre `M` execuções: `ε̄_t`. Para estado de dimensão `n = 2` e `M` execuções, o intervalo de aceitação de 95 % é `[χ²_{Mn}(0,025)/M , χ²_{Mn}(0,975)/M]` — com `M = 500`, aproximadamente **[1,82 ; 2,18]** em torno do valor esperado 2. **NIS** análogo sobre a inovação, por canal de medição.

**O critério de aceitação é assimétrico e precisa ser declarado antes de olhar o resultado:**

| Degrau | Critério |
|---|---|
| **6a** | dentro do intervalo (é um KF local bem sintonizado; se sair, o problema é sintonia, não fusão) |
| **6b** | **espera-se acima do limite superior** — otimista, isto é, inconsistente. É a hipótese a demonstrar |
| **6c** | **abaixo do limite superior**; ficar **abaixo do inferior é aceitável e esperado** (conservadorismo é a propriedade do método, não defeito) |

Exigir de 6c "dentro do intervalo" seria aplicar a um estimador conservador o teste de um estimador ótimo. **A grandeza que quantifica o preço do conservadorismo é reportada à parte**: fator `κ = √(tr(P̂)/n) / RMSE` (≈ 1 para um filtro ótimo, > 1 para CI). Ver Q2.

**Razão de credibilidade** `τ_t = RMSE_t / √(tr(P̂_t)/n)`, plotada ao longo do episódio: para 6b espera-se crescimento sem limite (o erro sobe enquanto o desvio predito desce — é a assinatura visual de "divergir com confiança"), para 6c espera-se `τ_t ≤ 1` estável. Esta figura é a peça central do argumento de H5 no relatório.

**Nada de RL é interpretado antes disso passar (PRD RF-05, RNF-06).**

### T-09.4 — Demonstração de divergência do 6b: fatorial offline, com o espantalho reforçado

Um espantalho fraco não justifica método nenhum. Esta tarefa existe para tornar a divergência do KF ingênuo **convincente**, e não apenas conveniente.

**Roda offline, sem RL.** Banco de trajetórias gravadas: rollouts do guloso e da política de 3a, com posições verdadeiras registradas. Sobre elas, a camada de ruído e os três estimadores são executados repetidamente. É barato, é rápido, e desacopla a validação do filtro do custo do treino — a mesma lógica que a SPEC-07 aplica na Etapa 2.

**Fatorial 2 × 2 × 3, obrigatório**, porque sem ele a inconsistência do 6b pode ser atribuída ao fator errado:

```
{ atraso compensado, atraso ignorado } × { incesto bloqueado, incesto ativo } × ρ ∈ {0, 0,5, 0,9}
```

Resultados que precisam sair, e que se falharem invalidam a demonstração:

1. **Em `ρ = 0`, com atraso compensado e incesto bloqueado, o KF ingênuo é consistente.** Se não for, o cenário está montado contra ele e a conclusão não vale.
2. **A inconsistência do 6b cresce monotonicamente com `ρ`** e com o incesto ativo. É a atribuição causal.
3. **6c permanece consistente em toda a grade**, com `κ` crescente em `ρ` — o preço do conservadorismo.

**Reforço do espantalho (steel-man).** O 6b entra na comparação na sua **melhor versão**: varrer inflação heurística de covariância `P_j ← λ · P_j` com `λ ∈ {1, 2, 5, 10}` e período de fusão `m ∈ {1, 2, 5}`, e reportar a **melhor** configuração de 6b contra 6c. Se alguma inflação heurística tornar o 6b consistente **e** competitivo em erro, isso é um resultado — e a conclusão passa a ser "CI é a versão principiada de um ajuste que também funciona", não "só CI funciona". Registrar o veredito como saiu.

### T-09.5 — Sub-passo 3b no laço: degraus 1, 4, 6a, 6b, 6c com a política

Os degraus definidos em §4.4, rodados na célula base `Na = 4, Ns = 24`, com a política de 3a e com política retreinada sob ruído.

**Cruzamento com o regime de treino** (herdado da Etapa 2): *treino limpo → avaliação ruidosa* (transferência) vs *treino ruidoso → avaliação ruidosa* (randomização de domínio). Se o degrau 6c tornar os dois regimes equivalentes, é o argumento mais forte possível a favor da fusão — o mesmo desfecho que a Etapa 2 buscou para o degrau 4.

**Acoplamento com o codificador.** A regra de projeto do PRD §1 (*largura do bin ≈ σ da estimativa filtrada*) precisa ser reaplicada com o `σ` **do estimador fundido**, que para CI é **inflado** pelo conservadorismo. Consequência concreta e testável: 6c pode exigir termômetro mais **grosso** que 6a, e parte do ganho de 6c pode ser comida pela perda de resolução. Rodar a célula base com `T ∈ {8, 128}` (grade reduzida da linha de corte) e reportar o par (`T`, degrau). **Não reportar o melhor `T` de cada degrau sem dizer que foi escolhido por degrau.**

**Métricas por degrau:** tempo de conclusão, taxa de sucesso, erro de estimativa por canal, NEES/NIS, `κ`, `τ_t`, **e todas as métricas de divisão de trabalho de T-09.6**.

### T-09.6 — Métrica de divisão de trabalho (H6): tornar quantitativo o que Lamenza olhou

**Esta é a lacuna real da etapa.** Lamenza (§7.1, fig. 7.1) mostra trajetórias que "quase perfeitamente se sobrepõem" quando não há incentivo, e (§7.4, fig. 7.4) trajetórias que se separam quando há — e conclui, com a ressalva explícita de antropomorfismo, que os agentes coordenam por "confiança": cada um aprende que os demais compartilham objetivo e política, e deixa de visitar certos sensores confiando que outro, melhor posicionado, o fará. É uma leitura elegante e é **inspeção visual**. Superá-la é contribuição.

Quatro medidas, todas calculadas por episódio, e todas também para os baselines:

| id | Medida | Definição | Lê o quê |
|---|---|---|---|
| **M1** | **Disjunção de cobertura** | `η = \|∪_k A_k\| / Σ_k \|A_k\|`, com `A_k` = área varrida pelo disco de `communication_range` do UAV `k` ao longo do episódio, rasterizada em grade fina | é **exatamente** a fig. 7.1 virando número: `η → 1/Na` é sobreposição total, `η → 1` é disjunção total |
| **M2** | **Colisão de alvo implícito** | fração dos passos de decisão em que dois ou mais UAVs têm o mesmo **alvo implícito** (sensor não visitado de menor desvio angular em relação ao vetor de velocidade comandado, dentro de um cone de ±30°), normalizada por `C(Na,2)` | mede disputa **antes** da coleta, que é onde o desperdício acontece |
| **M3** | **Redundância no instante da coleta** | média, sobre sensores coletados, de (nº de UAVs dentro do alcance no instante da coleta − 1) | mede esforço duplicado consumado |
| **M4** | **Equilíbrio de carga** | entropia normalizada da distribuição de coletas por UAV; complementada pela contagem de agentes ociosos (T-09.1) | necessário, **não suficiente**: carga equilibrada com trajetórias sobrepostas ainda é coordenação ruim — por isso M4 nunca é reportada sozinha |

**A normalização é o que dá sentido aos números.** Cada medida `m` vira um **índice de divisão de trabalho** ancorado entre duas referências computadas **no mesmo cenário, com a mesma semente**:

```
I_m = ( m_política − m_guloso ) / ( m_kmeans+TSP − m_guloso )
```

- `m_guloso` = **zero por construção**: o guloso é, nas palavras de Lamenza (p38), a estratégia em que "os agentes não dividem o trabalho entre si". É o piso de coordenação.
- `m_kmeans+TSP` = partição centralizada offline com posições verdadeiras — o teto quase-ótimo de divisão de trabalho já previsto no PRD §6.1.
- `I_m ≈ 0` → nenhuma divisão emergiu. `I_m ≈ 1` → a política alcançou a partição centralizada. `I_m > 1` é possível e interessante (a política divide de forma diferente da partição geométrica).

Reportar `I_M1`, `I_M2`, `I_M3` e o `I` agregado (média), com IC pareado, por `Na` e por degrau.

**T-09.6b — o teste contrafactual do canal de vizinhos (o teste causal).**

As métricas acima mostram *que* houve divisão. Elas não mostram que a divisão veio de **conhecer a posição dos demais**, que é o mecanismo que Lamenza alega. O teste é uma ablação em **tempo de avaliação**, sobre a política já treinada, sem retreino:

| Variante | O que se faz com `P^a` | O que testa |
|---|---|---|
| **A0** | intacto | referência |
| **A1** | congelado no valor do primeiro passo | o agente usa a posição *atual* dos demais ou só a inicial? |
| **A2** | substituído por amostras da distribuição marginal de `P^a` (vizinhos fantasmas plausíveis, mas errados) | o canal carrega informação usada, ou é ruído decorativo? |
| **A3** | identidades dos vizinhos permutadas | a política usa a **geometria** ou a **identidade** (o índice `k` e a ordenação)? |
| **A4** | zerado (equivalente a `N*_a = 0`, o cenário exploratório de Lamenza) | limite inferior |

**Se A2 e A4 não degradarem o tempo de conclusão nem os índices `I_m`, a "coordenação" observada não é coordenação** — é apenas o efeito de agentes idênticos com posições iniciais ligeiramente distintas se espalhando. Esse é o resultado negativo que o teste existe para poder detectar, e reportá-lo seria mais valioso que confirmar H6 sem ele.

**T-09.6c — a predição que liga H5 e H6.**

> A fusão paga em **coordenação**, não apenas em metros.

Se a coordenação depende de conhecer a posição dos demais, então degradar essa posição degrada a coordenação, e melhorá-la por fusão a recupera. Predição falsificável: `I_m` cai monotonicamente com `σ` no degrau 1, e é **recuperado** por 6c mais do que por 6a. Medir `I_m` por degrau e por `σ` é barato (as execuções já existem) e é o resultado mais interessante que a Etapa 3 pode produzir — porque conecta a metade de fusão à metade de RL com um número, e não com uma frase.

### T-09.7 — Transferência sem retreino: o que transfere e o que não transfere

Enunciado do PRD: política treinada em **4×24**, aplicada a **2×12** e **8×36** sem retreino. As duas dimensões **não** são igualmente difíceis, e a spec trata isso com honestidade.

**Transferência em `Ns` — legítima e barata.** A observação já lida com número decrescente de sensores não visitados: `N*_s = 12` é fixo, com padding **`-1`** — que está fora da faixa dos valores válidos e portanto já distingue slot vazio de sensor sobre o UAV, sem precisar de bit de validade. Lamenza confirma (§7.4.1) que a generalização em `Ns` é direta e que o modelo com `Ns` variável fica competitivo com os modelos fixos. Logo, **4×24 → 4×12 e 4×36 é transferência de primeira classe** e deve ser medida sempre.

**Transferência em `Na` — não trivial, e Lamenza falhou.** A dimensão da observação depende de `Na`, porque `N*_a = Na − 1`: são 31 escalares em `Na = 4` e 39 em `Na = 8`. Para a rede n-tuple o problema é **pior** que para uma MLP: mudar o número de bits `B` muda o mapeamento das tuplas, e uma LUT indexada por um mapeamento diferente **não significa nada** — a transferência entre contagens de agentes não é apenas ruim, ela é **indefinida** sem uma providência explícita.

Três variantes, com honestidade sobre o que cada uma vale:

| id | Variante | Estado |
|---|---|---|
| **V1** | largura nativa (`N*_a = Na − 1`), transferência direta | **impossível**; registrar por quê, é conteúdo do relatório |
| **V2** | **codificador de largura fixa** `N*_a = 7` com padding `-1` (o mesmo sentinela que o ambiente já usa), treinado em 4×24, avaliado em 2×12 e 8×36 | única variante em que a transferência é sequer **definida**. É a mesma convenção já usada para sensores, aplicada aos agentes |
| **V3** | retreino por `Na` (o padrão da campanha de Lamenza) | **referência obrigatória**: o custo da transferência só é mensurável contra ela — `Δ = T_transferido − T_retreinado` |

**Ressalva que precisa aparecer no relatório, não em nota de rodapé.** V2 é a mesma ideia (zero-padding para largura fixa) que Lamenza tentou no crítico centralizado e que **funcionou em cenários pequenos e falhou em cenários grandes** (§7.4.2, fig. 7.10: 80 % de sucesso, mas desempenho muito aquém). A diferença estrutural do nosso caso é que **não temos crítico centralizado** — o crítico n-tuple estima `V` sobre a própria observação do agente, então o modo de falha específico que Lamenza diagnosticou (entrada do crítico de tamanho variável) **não se aplica**. O que resta é o problema do **ator**, que é real e que ninguém resolveu. **A expectativa registrada antes do experimento é de degradação material, sobretudo em 4→8.** Se for confirmada, é resultado; se for refutada, é resultado melhor.

**V2 é decisão de codificador, não de MDP** — a informação disponível ao agente é a mesma, os slots excedentes são marcados inválidos. Mas o vetor que a rede recebe muda, e por simetria com a ressalva de E3 na SPEC-01, **isso exige decisão registrada** antes de ser adotado (Q3). Se adotado, a variante de largura nativa permanece como ablação no relatório.

**Extra de baixo custo, alto valor (opcional).** Redes sem peso aprendem online, sem GPU e em segundos. Medir **adaptação barata**: `k ∈ {10, 100, 1000}` episódios de treino online a partir da política transferida, contra treino do zero. Se 100 episódios recuperarem a maior parte da distância para V3, o argumento do trabalho ganha uma frase que nenhuma rede profunda pode dar. Fora do caminho crítico; primeira coisa a cair.

### T-09.8 — Custo com o filtro embutido (T1/T2/T3 recalculados)

PRD §6.4 e RNF-05: o KF e a CI são código que voa no UAV. Recalcular, pelo harness da SPEC-03:

- **T1**: tamanho do ator + estado do filtro residente (`x̂`, `P` próprios; uma trilha `(x̂, P)` por vizinho; `(x̂, P)` por sensor não visitado rastreado). Contagem explícita: o estado do filtro **cresce com `Na` e `Ns`**, enquanto o ator (por parameter sharing) **não**. Essa assimetria é a observação de custo da etapa.
- **T2**: latência ponta a ponta com fusão ligada, mediana e p99, decomposta por estágio (montagem da observação, filtragem, CI, codificação, endereçamento, amostragem).
- **T3**: contagem de operações por tipo. **A CI introduz MACs em ponto flutuante no caminho de inferência**, e a alegação "zero MACs" da Etapa 1 precisa ser reescrita para "zero MACs no aproximador; `X` flops no estimador". Reportar `X` medido, não estimado, e a razão contra a MLP pareada — inclusive se for desfavorável (§T-09.2).

---

## 6. Protocolo experimental

**Sementes.**

- **Cenário/avaliação:** conjunto fixo `S_eval` de 500 sementes, **idêntico em todos os braços** (números aleatórios comuns). Posições de sensores, posições iniciais dos UAVs e realizações de ruído derivam dele.
- **Treino:** 5 sementes por braço treinado (`{101,…,105}`); **3 no mínimo**, e a redução é a primeira coisa a cair da linha de corte local (§7).
- **Ruído:** fluxo indexado por `(semente_cenário, canal, agente, índice_do_passo)`, independente da trajetória (§4.3).

**Repetições e unidade de análise — o ponto que mais se erra.**

| Comparação | Unidade de análise | Pareamento |
|---|---|---|
| Degraus de fusão (6a/6b/6c) sobre a **mesma** política | **episódio** | exato: mesma semente, mesmo sorteio de ruído |
| Política vs baselines não aprendidos | **episódio** | exato: mesmo cenário |
| Política vs política (regimes de treino, `Na`, transferência) | **semente de treino** | por cenário dentro de semente |

Comparar políticas tratando episódios como unidades independentes **subestima o IC**, porque todos os episódios de uma mesma semente de treino compartilham a mesma política. Para essas comparações: bootstrap agrupado (reamostrar sementes de treino, depois episódios dentro da semente), 1000 reamostragens. Reportar **os dois** ICs (por episódio e agrupado) quando divergirem materialmente — a diferença entre eles é informação.

**Inferência.** Sempre **IC da diferença pareada** `d_i = m_A(i) − m_B(i)`, bootstrap 95 % com 1000 reamostragens (padrão de Katopodis §5.4, herdado pelo PRD §6.4). **ICs sobrepostos de dois braços não constituem teste** e não são aceitos como evidência em nenhum item do portão.

**Múltiplas comparações.** A família do portão é declarada **antes** de rodar: {6c vs 6a}, {6c vs 6b}, {6b vs consistência}, {política vs guloso, por `Na`}, {degrau 6c vs 3a}. Correção de Holm–Bonferroni sobre essa família. Comparações exploratórias (varreduras de `ρ`, `τ_d`, `λ`, `T`) são reportadas **como exploratórias** e não sustentam veredito de hipótese.

**Censura — armadilha herdada.** Episódios que não completam (limite de 500 s ou poda por `max_seconds_stalled`) têm tempo de conclusão **censurado**. Lamenza tropeça exatamente nisso (§7.3: com 0 % de sucesso em 4 e 8 agentes, a métrica de tempo ficou inadequada e o relato migrou para recompensa média). Convenção fixada aqui, antes de qualquer resultado:

1. **Taxa de sucesso sempre reportada ao lado** do tempo, nunca separada dela.
2. Tempo primário = **mediana entre os episódios bem-sucedidos**.
3. Tempo secundário = **penalizado**, com substituição do valor limite (500 s) nos censurados — reportado explicitamente como tal.
4. Se a taxa de sucesso de um braço cair abaixo de 50 %, o tempo de conclusão **não** é usado para veredito daquele braço; o veredito passa a ser sobre a taxa de sucesso.

**RF-02 preservado:** poda por episódio parado ativa, e **nunca** poda ao coletar todos os sensores — Lamenza mediu degradação forte em ambos os desvios (p31).

**Orçamento.** Braços de treino: 3a com `Na ∈ {2,4,8}` (3) + 3b com treino ruidoso na célula base (1–2), × 5 sementes = **20–25 treinos**. Braços de avaliação: 3a (4 métodos × 3 `Na`) + 3b (5 degraus × 2 `σ` × 2 `T`) + transferência (3 células × 3 variantes) + contrafactual (5 variantes) ≈ **70 avaliações** de 500 episódios. A ~0,11 s/episódio medido (PLANO §1), a avaliação inteira é da ordem de **1–2 h de núcleo único** — desprezível. **O treino é o orçamento**; dimensioná-lo com o número medido no G1 é a primeira tarefa da spec, e é o que decide se `Na = 8` entra.

---

## 7. Critério de saída — portão G3

Todos medidos, com o protocolo da §6, e escritos:

1. **3a executado e reportado** — tempo de conclusão e taxa de sucesso multiagente sem ruído, contra os três baselines, em pelo menos `Na ∈ {2, 4}`; curva de escala e contagem de agentes ociosos. **Sem 3a, o portão não é avaliável** (D10).
2. **NEES/NIS por canal**, com o critério assimétrico da T-09.3 declarado antes: **6a consistente**, **6b demonstravelmente inconsistente** (acima do limite superior, com `τ_t` crescente), **6c não otimista** (abaixo do limite superior), com `κ` reportado.
3. **A demonstração do 6b é atribuível** — o fatorial da T-09.4 mostra consistência em `ρ = 0` sem incesto e inconsistência crescente em `ρ`; e a melhor variante inflada de 6b foi tentada e reportada.
4. **6c supera 6a em erro de estimativa**, com IC da diferença pareada separado de zero, em ao menos uma condição `(σ, ρ)` documentada — e o comportamento em `ρ = 0` reportado ainda que desfavorável.
5. **Veredito escrito de H5**, com o número que o sustenta.
6. **Veredito escrito de H6**, sustentado por: tempo de conclusão vs guloso com IC pareado, por `Na`; **os índices `I_m` de divisão de trabalho**; e o **teste contrafactual** (T-09.6b).
7. **Transferência reportada** para 4×24 → {4×12, 4×36} e, na variante V2, → {2×12, 8×36}, sempre contra a referência retreinada V3, com o veredito sobre o que transfere e o que não transfere.
8. **T1/T2/T3 recalculados com o filtro e a CI embutidos**, incluindo o custo de flops da CI e a comparação com a MLP pareada — mesmo se desfavorável.
9. **Testes automatizados:** RF-04 estendido (o broadcast carrega estimativa, nunca verdade); RNF-07 (com `fusion/` desligado, 3a é o problema de Lamenza puro); propriedades algébricas da CI (T-09.2); reprodutibilidade do fluxo de ruído independente de trajetória.
10. **Relatório curto** com figuras, pronto para o capítulo de extensão de E2.

### Se G3 falhar

Portão não atingido é resultado publicável (PRD §2). Diagnóstico prescrito por modo de falha:

| Falha | Leitura e ação |
|---|---|
| **6b não diverge** | O espantalho caiu — e é um resultado honesto e interessante. Verificar primeiro se `ρ` e o incesto estão realmente ativos (T-09.4 item 1 invertido). Se estiverem, a conclusão é que **neste** cenário a correlação é fraca demais para quebrar o KF ingênuo, e H5 é **refutada na primeira metade**. Reportar a magnitude de correlação necessária para quebrá-lo — isso vira o resultado |
| **6c não supera 6a** | Conservadorismo custou mais que a informação ganha. Reportar `κ` e o acoplamento com o termômetro (T-09.5): é possível que o ganho exista em erro e se perca na codificação. H5 fica **parcialmente confirmada** (consistência sim, desempenho não), que é um veredito legítimo |
| **A política não bate o guloso** | H6 refutada. Verificar se a falha é de coordenação (índices `I_m` ≈ 0) ou de competência individual (comparar com o resultado de `Na = 1` do G1). São diagnósticos diferentes e a distinção é o conteúdo |
| **`I_m` alto mas contrafactual A2/A4 sem degradação** | Não houve coordenação: houve dispersão. **Este é o resultado negativo mais valioso possível da etapa** e reinterpreta a leitura de "confiança" de Lamenza. Reportar em destaque |
| **Treino multiagente sob ruído instável** | Ver §8; escalonar para a linha de corte local abaixo antes de gastar tempo em sintonia |

### Linha de corte local (ordem de sacrifício)

1. `Na = 8` (fica `Na ∈ {2, 4}`)
2. Sementes de treino: 5 → 3
3. Grade de `ρ`: `{0, 0,9}`; grade de `τ_d`: `{0, 1}`
4. `T ∈ {8, 128}` → só o `T*` herdado do G2
5. Adaptação online da T-09.7 (opcional desde o início)
6. Célula base `Na = 4` → `Na = 2` (PRD §8, item 6 da linha de corte global)

**Não podem cair:** 3a; degraus 1, 4 e 6c; NEES/NIS; a demonstração de inconsistência do 6b; ao menos uma métrica de divisão de trabalho com o teste contrafactual; os vereditos escritos de H5 e H6.

---

## 8. Riscos

| Risco | Mitigação |
|---|---|
| **Ruído + aprendizado independente multiagente = instável.** Com parameter sharing, todos os agentes escrevem nas mesmas LUTs; o alvo de cada um é não estacionário porque os demais estão aprendendo; e o ruído acrescenta variância no topo disso. É o risco de cronograma número um da etapa | Revarrer o **fator de esquecimento** `φ` (RNF-04) já em 3a, que é barato porque não tem ruído, e revalidar em 3b no degrau 1 (pior caso). Manter a asserção `η = c/N`, `c ≲ 0,5` (RNF-01) e guarda de NaN. **Orçar a revarredura no cronograma**: PLANO §9.10 antecipa exatamente isto ("ruído pode exigir revarredura"); tratá-la como imprevisto seria planejar mal duas vezes |
| **Falha catastrófica de treino em algumas sementes** — Lamenza reporta o fenômeno na própria implementação (§7.3) | Reportar a fração de sementes que falhou, nunca só a média das bem-sucedidas. Definir "falha" antes de rodar (curva de retorno sem melhora em X % do orçamento) e aplicar o critério mecanicamente |
| **A correlação injetada é fraca e 6b não diverge** — o experimento fica vazio | `ρ` é parâmetro varrido, não constante; o incesto de C1 é estrutural e não depende de injeção; e T-09.4 reporta o `ρ` limiar em que 6b quebra, o que transforma o risco em resultado |
| **6b diverge pelo motivo errado** (atraso, não correlação) | Fatorial 2×2 de T-09.4 com atraso e incesto como fatores separados. Sem ele, a atribuição causal não existe |
| **CI custa mais que a MLP** e arranha a narrativa de embarcabilidade (~5 000 flops/decisão com ω em grade, §T-09.2) | Medir com o harness da SPEC-03; implementar ω analítico e a variante `ω-rápido`; reduzir o conjunto fundido e a taxa de fusão; reportar o par (custo, desempenho) por mitigação. Reescrever a alegação para "zero MACs no aproximador, `X` flops no estimador" se for o caso |
| **Conservadorismo de CI exige termômetro mais grosso** e come o próprio ganho | Varrer `T` conjuntamente com o degrau (T-09.5), nunca um de cada vez — é a regra do PRD §1 aplicada ao `σ` fundido |
| **CI sequencial é dependente da ordem** com `Na = 8` | Ordem canônica para reprodutibilidade + medição da dispersão sob permutação; escalar para CI em lote na célula base se for material |
| **A "coordenação" medida é só dispersão de agentes idênticos** | É exatamente o que T-09.6b existe para detectar; e é resultado, não fracasso |
| **Transferência em `Na` falha** (Lamenza falhou, §7.4.2) | Expectativa registrada **antes** do experimento; V3 (retreino) como referência obrigatória para quantificar o custo; V1 documentada como indefinida, não como omissão |
| **Sem sinalização, UAVs convergem para o mesmo alvo** | É comportamento do problema original (Lamenza fig. 7.1), não defeito nosso (PRD §9). Reportar, não corrigir. A mensagem de reivindicação permanece em trabalho futuro (PRD §12) |
| **`P` no broadcast é acréscimo em relação ao original** | Ablação obrigatória de 6c com `P` conservador fixo (§4.2); decisão registrada (Q1) |
| **Tempo:** a etapa cai na semana 4, junto com relatório e slides | A linha de corte local da §7 é para ser usada, e a regra de escalonamento do PRD §8 vale: portão não atingido aciona corte, não extensão do prazo |

---

## 9. Entregáveis

- `src/uavwnn/fusion/ci.py` — interseção de covariância: `ω-tr`, `ω-det`, `ω-rápido`, solução analítica 2×2, fusão sequencial e em lote
- `src/uavwnn/fusion/noise.py` — estendido: atraso de broadcast, componente de erro comum parametrizado por `ρ`, fluxo de ruído indexado e independente de trajetória
- `src/uavwnn/fusion/tracks.py` — trilha local de vizinho (C1) e registro de estimativas de sensor (C2), com carimbo de tempo e propagação pelo atraso
- `src/uavwnn/eval/coordination.py` — M1–M4, índices `I_m` normalizados entre guloso e k-means+TSP, contagem de agentes ociosos
- `src/uavwnn/eval/counterfactual.py` — variantes A0–A4 do teste contrafactual do canal de vizinhos
- `configs/etapa3/` — 3a; 3b degraus 1/4/6a/6b/6c; fatorial `ρ × τ_d × incesto`; transferência V2/V3
- `reports/spec09-etapa3.md` + figuras: `τ_t` por degrau (a figura central de H5); NEES com banda qui-quadrado; `I_m` por degrau e por `σ` (a figura central de T-09.6c); trajetórias amostrais por `Na` no estilo da fig. 7.4 de Lamenza, **acompanhadas** dos números de M1–M3
- `tests/test_ci.py`, `tests/test_broadcast_leak.py`, `tests/test_noise_stream.py`
- **Vereditos escritos de H5 e H6** no formato do PRD §11 (confirmada / refutada / inconclusiva, com o número que sustenta)

---

## 10. Perguntas em aberto

**Q1 — O broadcast pode carregar a covariância?** Sem `P_j`, a interseção de covariância é inexequível; com ela, o canal transmite mais do que o "canal global" de Lamenza transmitia. A posição desta spec é que isso é extensão de fusão e não sinalização de coordenação (§4.2), mas **precisa de decisão registrada na §10 do PRD** antes de rodar. Alternativa se recusada: `P` conservador fixo assumido para o vizinho, que já está prevista como ablação — e que, se empatar, torna a questão irrelevante.

**Q2 — Qual é o critério de consistência aceito para um estimador deliberadamente conservador?** O portão G3 do PRD diz "NEES/NIS consistente" para 6c. Um estimador conservador fica sistematicamente **abaixo** do intervalo, e isso é a propriedade, não o defeito. Esta spec propõe o critério assimétrico da T-09.3 (não otimista, com `κ` reportado). **Precisa de acordo explícito**, porque com o critério simétrico o 6c reprova por fazer exatamente o que deveria.

**Q3 — Adota-se o codificador de largura fixa `N*_a = 7` (V2)?** É a única forma de a transferência entre contagens de agentes ser sequer definida para uma rede n-tuple. É decisão de codificador e não de MDP — mesma natureza da ressalva sobre E3 na SPEC-01 — mas muda o vetor de bits em todas as células, inclusive nas que não precisam dele. Ou se adota para tudo (e a comparação com o G1 muda de base), ou se treina duas vezes na célula base (nativa e V2), o que custa sementes.

**Q4 — O degrau 5 (covariância na observação) entra na Etapa 3?** PLANO §9.7 define `6 = 5 + CI`; PRD §6.3 lista só 1, 4 e 6. Ligá-lo acrescenta canais à observação e exige linha na tabela de desvios §5.4. A proposta aqui é deixá-lo desligado por padrão e ligá-lo apenas se H4 tiver sido confirmada no G2, e só na célula base. **Depende do resultado da Etapa 2 e portanto não pode ser decidido agora** — mas o gatilho precisa estar acordado antes, para não virar decisão pós-hoc.

**Q5 — Qual definição de "alvo implícito" (M2) sobrevive a inspeção?** O cone de ±30° em torno do vetor de velocidade comandado é uma escolha arbitrária. Alternativas: sensor de menor tempo-para-alcance sob velocidade constante; atribuição por transporte ótimo entre UAVs e sensores não visitados. Propõe-se fixar uma definição **antes** de olhar os resultados e reportar a sensibilidade a ela como anexo. Sem isso, M2 é ajustável a posteriori e não vale nada.

**Q6 — ~~`communication_range` pendente~~. FECHADA:** `communication_range = 20 m`. M1 e M3 podem ser calibradas, e as áreas semitransparentes da fig. 7.4 de Lamenza correspondem a esse raio, o que torna a comparação visual utilizável.

**Q7 — Qual é o `Na` da célula base se o orçamento de treino medido no G1 não fechar?** O PRD fixa `Na = 4, Ns = 24`; a linha de corte global permite `Na = 2`. Com `Na = 2` há **um único vizinho**, e várias métricas desta spec (M4, equilíbrio de carga; a dependência de ordem da CI sequencial; a contagem de agentes ociosos) perdem quase todo o poder discriminante — e o achado de retornos decrescentes de Lamenza fica inacessível. Cortar para `Na = 2` deveria ser a última opção, e não a sexta.

**Q8 — Como se trata C2 quando o mesmo sensor é estimado por três ou mais UAVs?** CI sequencial é barata e dependente da ordem; CI em lote é correta e cara. A decisão depende da magnitude medida da dependência de ordem (T-09.2), que é justamente um resultado desta spec — ou seja, a decisão precisa ser tomada **durante** a execução, com o critério fixado antes: se a dispersão sob permutação exceder X % do erro médio, migra-se para lote. **Falta acordar X.**
