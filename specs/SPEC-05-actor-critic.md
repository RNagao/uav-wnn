# SPEC-05 — n-tuple Actor-Critic (ação contínua) e baseline MLP AC pareado

**Etapa:** 1 · **Depende de:** SPEC-04 (núcleo n-tuple esparso), SPEC-01 (codificador, G0) · **Estimativa:** 3 dias
**Alimenta:** SPEC-06 (Etapa 1, portão G1) · **Referência:** PRD §5.2 (MDP verbatim), §5.3 (aproximador), §5.5 (custo), §10 (D2, D6, D7)
**Fonte primária:** Katopodis (2022) §4.3.1–§4.3.4, §5.3, §5.4, apêndice A · Lamenza (2025) §5.1–§5.3, tab. 5.1

---

## 1. Objetivo

Entregar **duas políticas treináveis e intercambiáveis** sobre a mesma interface, com o mesmo algoritmo, a mesma observação e a mesma recompensa:

- `NTupleAC` — n-tuple Actor-Critic de Katopodis §4.3.4, política gaussiana com média vinda de uma rede n-tuple funcional e crítico n-tuple com esquecimento;
- `MlpAC` — **o mesmo algoritmo**, trocando apenas o aproximador por uma MLP de 2 camadas × 32 unidades ReLU (arquitetura de Katopodis §5.4).

E validá-las **fora do ambiente de UAVs**, em benchmarks com números publicados, antes de a SPEC-06 gastar tempo de simulação.

Esta spec **não** produz resultado sobre o problema de Lamenza. Ela produz o par de aproximadores que a SPEC-06 mede. Se o par não estiver isolado corretamente, H1 não é falsificável e o trabalho perde a contribuição central.

## 2. Por que esta spec existe separada da Etapa 1

Três razões, em ordem de custo evitado.

**(a) Um algoritmo com bug é indistinguível de "o problema é difícil".** É a mesma lógica da SPEC-01 aplicada ao laço de RL: diagnosticar dentro do `gradysim`, com episódios de 500 s e recompensa esparsa, custa dias por iteração. Katopodis publica valores exatos para n-tuple AC (Mountain Car 93,3 (92,9; 93,8); Hopper 748,8 (741,7; 758,0)) e para o AC com MLP (0,02 e 643,6). Reproduzir esses quatro números é um teste de aceitação forte, barato e **bilateral**: valida a implementação n-tuple *e* a do baseline.

**(b) O pareamento é a hipótese.** H1 afirma algo sobre o *aproximador*, não sobre o algoritmo. Qualquer assimetria entre os dois lados — um usa vantagem normalizada e o outro não, um avalia com ação média e o outro amostra, um roda em PyTorch e o outro em NumPy — transfere-se inteira para o veredito de H1 e o contamina. Isolar isso é trabalho de engenharia, e é melhor fazê-lo com contrato escrito do que por inspeção depois.

**(c) A formulação de Katopodis não cobre nosso espaço de ação.** Katopodis parametriza gaussianas multivariadas para Hopper (`R³`), mas nenhuma componente dele é **circular**. Nossa ação tem uma componente angular (`θ ∈ [0, 2π]`) e uma componente limitada (`s ∈ [0, 15]`). Tratar `θ` como escalar em `[0,1]` sob gaussiana é um erro estrutural, não um detalhe de sintonia — §4.3 desenvolve. A adaptação precisa estar decidida e testada antes de a Etapa 1 começar.

## 3. Escopo

**Dentro:** algoritmo AC on-policy com retornos de `m` passos; a rede funcional como ator e a rede com esquecimento como crítico; a parametrização da política para `⟨θ, s⟩`; o tratamento da circularidade de `θ`; a escolha de `σ²`; compartilhamento de parâmetros entre UAVs homogêneos; o contrato da interface `Policy` e sua integração com a ponte por injeção de política; o baseline MLP AC em NumPy puro; validação em benchmarks; faixas iniciais de hiperparâmetros.

**Fora:** o ambiente `gradysim` e a ponte em si (SPEC-00); as varreduras `T × n × N` sobre o problema real e as réplicas 1a/1b (SPEC-06); o harness de custo T1/T2/T3 (SPEC-03) — esta spec **consome** esse harness e declara os pontos de instrumentação, não o implementa; ruído e KF (SPEC-07+); MADDPG (fora do projeto, D7).

**Restrição que rege tudo (PRD §0, D1, D2):** a ação permanece contínua. Nenhuma tarefa aqui pode ser resolvida discretizando `θ`. Se ao fim da spec a conclusão for que a ação contínua é inviável com n-tuple, isso é uma **decisão a registrar na §10 do PRD e uma linha nova na §5.4**, não uma escolha de implementação.

---

## 4. Formalização e contratos

### 4.1 O algoritmo, escrito sem os erros de transcrição do original

O Algoritmo 4.4 de Katopodis (p. 59) tem três defeitos de impressão que, copiados literalmente, produzem código errado. Registramos a versão corrigida como o contrato:

| # | Como está impresso | Correção | Por quê |
|---|---|---|---|
| 1 | `v_policy ← v_policy + δ · ∂_f ln f(A\|S)` | `v_policy ← v_policy + η · δ · ∂_f ln f(A\|S)` | a taxa de aprendizado `η` é declarada na entrada do algoritmo e some da regra de atualização; sem ela, o passo é `δ`, que na nossa escala de recompensa chega a ~100 |
| 2 | `v_value(S,A)` e `c_value(S,A)` | `v_value(S)` e `c_value(S)` | o crítico do AC aproxima **V**, função só do estado. O argumento `A` é resíduo do Algoritmo 4.1 (SARSA multi-passo), copiado por engano |
| 3 | `v_value ← φ·v_policy + G` ; `c_value ← c_policy + 1` | `v_value ← φ·v_value + G` ; `c_value ← c_value + 1` | mistura das duas redes; obviamente um erro de digitação |

**Laço corrigido**, com `τ = t − m + 1` o índice do passo cuja atualização está madura:

```
G ← Σ_{i=τ+1}^{min(τ+m, T)} γ^{i−τ−1} R_i
se τ + m < T:  G ← G + γ^m · V(encode(S_{τ+m}))
δ ← G − V(encode(S_τ))                                  # vantagem de m passos
ator   (∀i ∈ 1..N_policy):  v_policy[i](φ_τ) ← v_policy[i](φ_τ) + η · δ · g_τ
crítico(∀j ∈ 1..N_value ):  v_value[j](φ_τ) ← ϕ · v_value[j](φ_τ) + G
                             c_value[j](φ_τ) ← ϕ · c_value[j](φ_τ) + 1
```

com `g_τ = ∇_f ln π(A_τ | S_τ)` dado por Katopodis eq. 4.35, e `φ_τ = encode(S_τ)`.

> **Notação.** Usamos `ϕ` para o fator de esquecimento (Katopodis) e `φ(x)` para o vetor binário esparso de ativação das RAMs (PRD §5.3). São coisas diferentes e a colisão de símbolo é do material de origem. No código: `forgetting` e `phi`.

### 4.2 A convenção de saída da rede — onde `η = c/N` realmente vale

Há **duas convenções incompatíveis** em circulação no projeto e elas mudam o significado de `η` por um fator de `N`:

| Convenção | Saída | Efeito de somar `η·g` às `N` posições endereçadas |
|---|---|---|
| **Soma** (protótipo `ntuple.py`, PRD §5.3: `f_k(x) = ⟨W_k, φ(x)⟩`) | `f(x) = Σ_i v[i](x)` | `f` muda `η·N·g` |
| **Média** (Katopodis eq. 4.12) | `f(x) = (1/N) Σ_i v[i](x)` | `f` muda `η·g` |

A RNF-01 do PRD (`η = c/N`, `c ≲ 0,5`, medida quando `η = 0,05` com `N = 48` divergiu para NaN) é **exatamente a convenção de média reexpressa na convenção de soma**. As taxas do apêndice A de Katopodis (`8·10⁻⁴` para Mountain Car com `N` implícito pelo par resolução/tupla, `2·10⁻⁵` para Hopper) estão na convenção de **média** e **não são transferíveis** para a implementação em soma sem dividir por `N`.

**Decisão:** adotar a convenção de **média** de Katopodis em `wnn/ntuple.py` (`f = (1/N)Σ v[i]`) e converter o protótipo. Razões: (i) torna `η` comparável com a literatura de referência sem conversão mental; (ii) desacopla a taxa de aprendizado de `N`, o que é essencial porque `N` é varrido — na convenção de soma, cada ponto da varredura de `N` mexe implicitamente na taxa e a varredura deixa de medir o que se pensa; (iii) mantém `k(x,x) = 1`, que é o que faz a leitura por máquina de kernel de §4.3.1 fechar.

**Asserção obrigatória** (herdada da RNF-01, reescrita para a nova convenção): `η · max|∂c/∂f| ≤ c_max` com `c_max` configurável, verificada em tempo de execução sobre o `δ·g` efetivo, com `NaN`/`inf` abortando o treino com mensagem que aponta o passo e o estado. O modo de falha medido no protótipo foi divergência para NaN; ele precisa de detecção, não de fé.

### 4.3 A ação: `θ` é circular e isso não é um detalhe

**O problema.** A ação de Lamenza é `a = ⟨θ ∈ [0,2π], s ∈ [0,15]⟩`, normalizada para `[0,1]²` (§5.2 dele). Uma gaussiana sobre `θ_n = θ/2π ∈ [0,1]` sofre de dois defeitos independentes:

1. **Descontinuidade no envoltório.** `θ_n = 0,01` e `θ_n = 0,99` são rumos separados por 3,6°, mas o resíduo `(a − f(s))` que aparece no gradiente de Katopodis (eq. 4.35) vale `+0,98`. O gradiente empurra a média na direção **oposta** à correta, com magnitude quase máxima. Não é ruído: é sinal errado, sistemático, e concentrado exatamente na fronteira onde o agente precisa decidir entre virar à esquerda e virar à direita.
2. **A média entre RAMs não respeita o círculo.** A saída da rede é `f(x) = (1/N) Σ_i v[i](x)`, uma média **linear**. Se metade das RAMs vota `0,02` e a outra metade vota `0,98` — duas votações no mesmo rumo, quase norte — a saída é `0,5`, rumo sul. Isso é pior que o item 1, porque acontece dentro do aproximador, a cada inferência, e não depende de a política estar perto da fronteira. **Qualquer** parametrização em que uma componente escalar da saída da rede n-tuple *é* um ângulo está estruturalmente quebrada.

O item 2 elimina de saída a solução "gaussiana envolvida (*wrapped*) sobre um escalar" e a von Mises com média angular escalar. A SPEC-01 já chegou à mesma conclusão por outro caminho: T-01.5 regride `θ` como `(cos θ, sin θ)` "para evitar a descontinuidade em `2π`". Manter as duas specs coerentes é obrigatório — se o codificador foi escolhido sob um alvo `(cos, sin)`, o ator tem de produzir `(cos, sin)`.

**Opções avaliadas:**

| id | Proposta | Veredito |
|---|---|---|
| **C0** | Gaussiana simples sobre `θ_n ∈ [0,1]`, aceitando o custo | Rejeitada. Falha nos itens 1 e 2. É a linha de base do experimento, não a solução |
| **C1** | Gaussiana envolvida: amostra `θ_n = (z mod 1)`, gradiente com resíduo circular `wrap(z − f(s)) ∈ [−½, ½)` | Corrige o item 1, **não** corrige o item 2. Barata (nenhuma saída extra). Fica como ablação |
| **C2** | **Normal projetada:** a rede emite `u ∈ R²`; amostra-se `z ~ N(u, σ_u² I₂)`; o rumo executado é `θ = atan2(z₂, z₁)` | **Recomendada.** Ver abaixo |
| **C3** | von Mises com `μ = atan2(u₂, u₁)`, `κ` fixo, gradiente `κ·sin(θ−μ)·∂μ/∂u` | Corrige os dois itens, mas `∂μ/∂u = (−u₂, u₁)/‖u‖²` é **ortogonal a `u`** e portanto `‖u‖` é invariante ao gradiente em primeira ordem. Como a memória n-tuple inicializa em zero, `u ≡ 0` no início, `∂μ/∂u` é singular e o algoritmo não sai do lugar. Rejeitada por incompatibilidade com a inicialização do modelo |

**Por que C2 (normal projetada).**

Formalmente: define-se a política sobre uma variável latente `z ∈ R³`,

```
z ~ N( f(encode(o)),  Σ ),      Σ = diag(σ_u², σ_u², σ_s²)
θ  = atan2(z₂, z₁)                     ∈ [0, 2π)
s  = 15 · clip(z₃, 0, 1)               ∈ [0, 15]
```

e o mapa `z ↦ (θ, s)` é declarado **parte do ambiente**, não da política. O teorema do gradiente de política vale para o MDP cuja ação é `z`; a recompensa depende de `z` apenas através de `(θ, s)`, o que é permitido. Consequência prática decisiva:

> **O gradiente continua sendo exatamente a eq. 4.35 de Katopodis, `∇_f ln π = Σ⁻¹ (z − f(s))`, sem nenhuma modificação.** Nada da derivação dele precisa ser refeito.

Quatro propriedades que fecham a escolha:

- **Funciona a partir da memória zerada.** Com `f ≡ 0`, `z ~ N(0, I)` e `atan2(z)` é exatamente uniforme em `[0,2π)`. A exploração inicial é isotrópica e correta, sem inicialização especial. É o oposto de C3.
- **`σ_u` não é hiperparâmetro.** A distribuição angular de `atan2(z)` depende de `(f₁, f₂, σ_u)` só através de `f_{1:2}/σ_u`. Fixar `σ_u = 1` não perde generalidade — a escala é absorvida pela própria rede. Isso **remove um hiperparâmetro da varredura** e é uma vantagem real sobre C0/C1.
- **A concentração angular é aprendida de graça.** `‖f_{1:2}‖` é a razão sinal-ruído do rumo: `‖f‖ = 0` dá rumo uniforme; `‖f‖ = 3` dá desvio angular ≈ 19°. O agente aprende *onde* ser determinístico e *onde* explorar, sem custo de parâmetros e sem a instabilidade clássica de aprender `log σ`.
- **Casa com a SPEC-01.** O alvo `(cos, sin)` do proxy supervisionado T-01.5 é o mesmo objeto que o ator produz.

**O preço, e como pagá-lo.** `‖f_{1:2}‖` cresce sem limite sob reforço consistente (o mesmo fenômeno dos *logits* de uma softmax), e exploração angular tende a zero prematuramente. Duas mitigações, na ordem:

1. **Reaproveitar o fator de esquecimento (RNF-04) no ator.** Decaimento multiplicativo `v ← λ·v + η·δ·g` nas posições endereçadas atua como encolhimento em L2 e fixa um teto para `‖f‖`. É código que já existe para o crítico e unifica os dois lados. `λ` entra na varredura.
2. **Fallback:** projeção `f̃ = r_max · f/‖f‖` quando `‖f‖ > r_max`, com o gradiente calculado sobre `f̃` (ignorando o jacobiano da projeção — aproximação declarada). Só se (1) não segurar.

**Diagnóstico obrigatório:** histograma de `‖f_{1:2}(o)‖` sobre o corpus, ao longo do treino. É o instrumento que distingue "não aprendeu" de "colapsou para determinístico cedo demais" — dois diagnósticos com correções opostas.

**Amostrado ≠ executado.** A regra que evita o bug mais provável desta spec:

> A atualização usa **`z` bruto** (o que a política de fato amostrou). O ambiente recebe `(θ, s)` **derivado e recortado**. Nunca o contrário. Recortar `z₃` e depois usar o valor recortado no resíduo `(z − f)` viesa o gradiente e mata o sinal justamente onde a política pede velocidade máxima — que é a região onde ela vai passar a maior parte do tempo.

**Forma da saída.** O ator emite `d = 3` componentes: `(u₁, u₂, μ_s)`. Isso vale para **os dois** aproximadores. Consequência a registrar: a MLP pareada passa a ser `D→32→32→3`, e não `D→32→32→2` como diz a tabela do PRD §5.5. O efeito na contagem de parâmetros é desprezível (+33), mas o número publicado no relatório tem de ser o correto.

### 4.4 A variância `σ_s²`: fixa, anelada, e por quê

Katopodis trata `σ²` como hiperparâmetro fixo (0,5 em Mountain Car; 0,2 no Hopper por §5.3, **0,3 no Hopper por tab. A.3** — a inconsistência é do original e o valor da tabela A.3 é o que corresponde à avaliação comparativa; registrar no relatório e não citar "0,2" como o valor do experimento comparativo).

**Recomendação: `σ_s` fixa e anelada, não aprendida.** Quatro argumentos:

1. **Escala.** `σ² = 0,5` significa `σ ≈ 0,71`. A faixa de ação do Mountain Car é `[−1, 1]`, então isso é ~35 % da faixa — exploração agressiva, adequada a uma tarefa que exige acumular momento. Nossa ação normalizada vive em `[0,1]`: `σ = 0,71` sobre faixa unitária produziria maioria de amostras recortadas e uma política efetivamente de dois níveis (parado / máximo). **Os valores de Katopodis não são transferíveis; a grandeza transferível é `σ/faixa`.** Faixa inicial: `σ_s ∈ {0,05, 0,10, 0,20}` da faixa.
2. **A recompensa penaliza tempo.** `R_t = (Σ visited − Ns)/Ns` é negativa a cada passo sem coleta. Exploração na velocidade é diretamente cara, e ainda interage com `max_seconds_stalled = 30 s`: 15 decisões sem visita e o episódio é podado. Exploração excessiva não só custa recompensa, **encurta o episódio** e reduz o número de experiências úteis por episódio.
3. **Aprender `σ` é instável no aproximador errado.** Uma cabeça `log σ` numa rede n-tuple herda o modo de falha conhecido: em regiões onde a vantagem é consistentemente positiva, `σ → 0`, o fator `σ⁻²` no gradiente explode e o treino diverge. A asserção da §4.2 detectaria, mas o custo de sintonia é alto e não há resultado publicado de Katopodis a que ancorar.
4. **Custo de modelo.** Uma cabeça a mais é `+33 %` na carga útil de cada posição de memória do ator (`d = 3 → 4`), o que ataca T1 diretamente (§4.7).

**Anelamento** de `σ_s` de `σ₀` a `σ_min` ao longo do treino (linear em passos), porque com `σ` fixa a política ótima determinística é inatingível e sobra ruído residual na avaliação. Reportar a curva junto com os resultados.

**Avaliação é determinística.** Nos 500 episódios de avaliação usa-se a **média**: `θ = atan2(f₂, f₁)`, `s = 15·clip(f₃,0,1)`. Isso não é escolha nossa — é o protocolo de Lamenza ("*during an evaluation the training is temporarily halted and action noise is removed*", §7). Vale igualmente para os dois aproximadores.

**`σ_s` aprendida** fica como ablação declarada, executada só se a fixa+anelada empatar com o guloso e a hipótese for de exploração mal calibrada.

### 4.5 Crítico: qual das duas redes n-tuple

Katopodis §4.3.4 especifica o crítico como **NTRN com esquecimento** (§4.2.1): posições guardam `(v, c)`, a saída é média ponderada `V(x) = (1/N) Σ_i v[i](x)/s[i](x)` com `s = (1−ϕ^c)/(1−ϕ)` (eq. 4.10), **sem taxa de aprendizado**. `ϕ = 0,85` foi o melhor nas quatro tarefas dele.

Há uma alternativa: usar a **rede funcional** também como crítico, com perda quadrática (`∂c/∂f = f − G`), o que dá TD semi-gradiente clássico.

| | NTRN com esquecimento (Katopodis) | Rede funcional + MSE |
|---|---|---|
| Fidelidade à referência | total | desvio de implementação |
| Hiperparâmetros | `ϕ` | `η_value` |
| Memória por entrada | `v` **e** `c` → 2× | só `v` |
| Garantia teórica | — | é TD com aproximação linear sobre `φ(x)`; herda as garantias citadas no PRD §5.3 |
| Código | segundo layout de memória | mesmo layout do ator |

> **Ponto a corrigir no relatório.** O PRD §5.3 argumenta que a rede n-tuple "é tile coding com ladrilhamento aleatório, e portanto herda as garantias de convergência de TD com aproximação linear". Esse argumento vale para a atualização **semi-gradiente**. Ele **não** cobre a média ponderada com esquecimento da NTRN, que não é um passo de gradiente sobre um funcional de risco. Se o crítico de Katopodis for o adotado — e a recomendação é que seja —, a fundamentação teórica precisa dizer que a garantia se aplica ao ator (gradiente funcional) e à variante funcional do crítico, não à NTRN.

**Decisão:** NTRN com esquecimento como **primária** (fidelidade a D6), variante funcional atrás de uma *flag*, ambas medidas em T-05.6. O custo marginal é baixo porque o núcleo esparso é o mesmo.

**Modo de falha do crítico frio.** Uma entrada vista uma única vez tem `s = 1` e devolve o retorno bruto daquela visita: `V` de altíssima variância, e a vantagem `δ = G − V` herda essa variância multiplicada por `η·σ⁻²` no ator. Mitigações a instrumentar: (i) **aquecimento** — `K_warmup` passos com o crítico atualizando e o ator congelado; (ii) encolhimento por confiança para `c` baixo; (iii) monitorar a fração de leituras do crítico que caem em entradas com `c < c_min` por decisão. Entrada nunca endereçada devolve `0`, que na nossa escala de recompensa é **otimista** (estados valem números negativos) — otimismo diante da incerteza, que pode ajudar a exploração ou desestabilizar. É comportamento a medir, não a assumir.

### 4.6 Interface `Policy` e a ponte

A ponte do PRD §7 é por **injeção de política**: `UAVProtocol.handle_timer("decide")` chama a política *inline*. Isso só funciona para métodos on-policy — e é justamente o caso. O contrato tem de absorver que, no simulador, **a recompensa de um passo só existe no tique seguinte**. Um único ponto de entrada elimina a classe inteira de bugs de ordenação:

```python
Obs    = np.ndarray   # shape (D,),  float32,  D = 2*(Na-1) + 24 + 1
EnvAct = np.ndarray   # shape (2,),  float32,  (theta_n, s_n) em [0,1]^2

class Policy(Protocol):
    obs_dim: int
    def step(self, agent_id: int, obs: Obs, reward: float,
             end: Literal[None, "terminal", "truncated"], *, learn: bool) -> EnvAct: ...
    def flush(self, agent_id: int) -> None: ...          # esvazia a janela de m passos
    def reset_episode(self, agent_id: int) -> None: ...
    def snapshot(self) -> dict: ...                      # T1: entradas ocupadas, bytes
```

Semântica de `step`, nesta ordem exata:

1. fecha a transição pendente do agente `agent_id` com `reward` (recompensa do intervalo de 2 s que acabou) e `obs` como `S_{t}`;
2. se `learn` e a janela tem `m+1` elementos, aplica a atualização da §4.1 para `τ = t − m + 1`;
3. amostra `z`, empilha `(φ_t, z_t)` na janela do agente;
4. devolve `(θ_n, s_n)`; o protocolo converte para `SetVelocityMobilityCommand` com `v = (s·cos θ, s·sin θ)`.

**Cadência de atualização: por passo, com retorno de `m` passos.** Não por episódio. Justificativa: (i) é literalmente o Algoritmo 4.4; (ii) o episódio tem no máximo `500/2 = 250` decisões, então `m = 250` é o episódio inteiro e degenera para REINFORCE com linha de base — que Katopodis mediu **pior** (Mountain Car 90,9 vs 93,3; Hopper 499,6 vs 748,8); (iii) atualizar por passo mantém tudo dentro de `simulation.start_simulation()` sem corrotinas nem estado compartilhado, que é o que o PRD §7 pede.

Os `m` de Katopodis (100 no Mountain Car, 1000 no Hopper) são da ordem do episódio das tarefas dele. Faixa inicial nossa: `m ∈ {1, 5, 20, 250}`, com `m = 250` presente só como controle degenerado.

**Rastros de elegibilidade: não.** Retornos de `m` passos são a visão prospectiva do rastro e é o que o algoritmo de referência usa. Um rastro sobre LUT esparsa exigiria tocar todas as entradas já visitadas a cada passo — custo `O(ocupadas)` por passo, crescente, e destrói o argumento de custo de treino que é um resultado do projeto. Registrado como trabalho futuro.

**Terminal vs truncado.** Pela RF-02, o episódio **não** termina quando todos os sensores são coletados. Ele acaba por `max_episode_length` ou por `max_seconds_stalled`. Ambos são **truncamentos**, não terminações do MDP, e portanto o retorno de `m` passos **deve** fazer bootstrap com `γ^m V(S_T)` no fim. Tratar truncamento como terminal (`V = 0`) enviesa a política contra estados de valor alto perto do limite de tempo. O caso do *stalling* é mais delicado — é uma terminação induzida pela política — e é uma **pergunta em aberto** (§9). O padrão do CleanRL, base da implementação de Lamenza, marca `done = True` em ambos; se replicarmos isso, é para replicar deliberadamente, não por omissão.

### 4.7 Compartilhamento de parâmetros entre UAVs

Lamenza §5 treina **um único ator** com todos os agentes segurando a mesma instância (*parameter sharing*, Chu & Ye 2017), e injeta o índice `k` na observação para permitir individualidade. Reproduzimos.

Com `Na = 1` (Etapas 1 e 2) isso é trivial. A generalização, e o que ela quebra:

**(a) Buffers por agente, tabelas compartilhadas.** Uma instância de `NTupleAC`; um dicionário `agent_id → janela de m passos`. A janela é por agente porque o retorno de `m` passos é sobre a trajetória daquele agente.

**(b) Atualizações assíncronas viram off-policy.** Se o agente B atualiza as tabelas enquanto o agente A ainda tem `m` passos amostrados sob os parâmetros antigos, a atualização de A deixa de ser on-policy. Standard em A2C/PPO com compartilhamento, mas aqui há um agravante de reprodutibilidade: a ordem de aplicação depende da ordem de eventos do simulador. **Decisão: barreira por tique de decisão.** Todos os agentes agem; as atualizações maduras do tique são acumuladas num dicionário `{(ram, addr) → Δ}` e aplicadas **uma vez**, ao fim do tique. Resultado determinístico independente da ordem de eventos, e um único caminho de escrita na tabela esparsa. Custo: zero na prática (as chaves colidem pouco).

**(c) O índice do agente é quase invisível a um mapeamento de tuplas aleatório — e isso é quantificável.** O índice é categórico: codifica-se como **one-hot de `Na` bits**, nunca por termômetro (não há relação de ordem entre UAVs; termômetro imporia uma). Mas com `B` bits totais na entrada, a probabilidade de uma RAM de tamanho `n` ler ao menos um bit de índice é

```
p = 1 − (1 − Na/B)^n
```

Para `Na = 4`, `T = 32` sobre 31 escalares (`B ≈ 1000`), `n = 12`: `p ≈ 4,7 %` — cerca de **2 de 48 RAMs** enxergam a identidade do agente. O canal de individualidade que Lamenza obtém de graça numa MLP densa (toda unidade da primeira camada vê o índice) é, na rede n-tuple, praticamente inexistente.

**Mitigações a testar em T-05.5:** (i) replicar o one-hot `R` vezes no padrão binário, elevando `p` a `1 − (1 − R·Na/B)^n`; (ii) reservar um grupo de RAMs cujo mapeamento **inclui obrigatoriamente** os bits de índice; (iii) não fazer nada e medir — se a individualidade não tiver efeito no tempo de conclusão, é resultado a reportar, e um resultado interessante (a coordenação de Lamenza emerge de posições relativas, não de identidade). Isto é um acréscimo ao **codificador**, portanto não entra na tabela de desvios §5.4 — a mesma lógica pela qual a binarização em si é a única linha de lá.

### 4.8 Baseline MLP AC pareado

**O que é idêntico, por construção e por teste:** ambiente, semente, observação (eq. 5-1), recompensa (eq. 5-6), `γ`, `m`, forma do retorno de `m` passos, tratamento de truncamento, parametrização da política (mesma normal projetada de §4.3, mesmo `σ_s` e mesmo anelamento), avaliação determinística, política de barreira do compartilhamento de parâmetros. **A única diferença é `f(·)`.**

**Arquitetura** (Katopodis §5.4): duas camadas ocultas de 32 unidades com ReLU, Adam. Ator `D→32→32→3`; crítico `D→32→32→1`, redes **separadas** (sem tronco compartilhado), espelhando a separação ator/crítico da WNN. `D = 25` para `Na = 1`, `31` para `Na = 4`, `39` para `Na = 8`.

**Entrada.** A MLP recebe a observação **em ponto flutuante**, com normalização afim fixa (`/200` para as coordenadas relativas; one-hot para o índice). A normalização é parte do aproximador, exatamente como a binarização é parte do aproximador do lado WNN (PRD §5.4). Sem ela a comparação seria injusta contra a MLP.

**RNF-11 — NumPy puro dos dois lados.** Medir a latência de uma MLP `31→32→32→3` via PyTorch mede sobretudo o despacho do PyTorch: para ~2 100 MACs, o *overhead* por chamada é da ordem do próprio cálculo, e o resultado seria uma vitória da WNN por artefato de arcabouço.

**Decisão: implementar a MLP AC inteira em NumPy — inferência e retropropagação — e não usar PyTorch em nenhum caminho do experimento.** Uma MLP de 2 camadas com ReLU e Adam são poucas dezenas de linhas, totalmente determinísticas e sem dependência pesada. Isso satisfaz a RNF-11 por construção e não por ressalva, elimina a "coluna com arcabouço" e remove a maior fonte de não determinismo de `make repro`.

**Risco associado e sua mitigação:** retropropagação escrita à mão erra em silêncio. **Teste obrigatório:** verificação de gradiente por diferenças finitas em todos os parâmetros, com tolerância declarada, mais um teste de equivalência numérica contra uma referência PyTorch — a referência vive **só em `tests/`**, como dependência de desenvolvimento, nunca no caminho de medição.

**Contagem de operações por decisão (T3, RNF-12)** — o número que sustenta a alegação estrutural de H1, calculado aqui para `Na = 1`, `T = 32`, `n = 12`, `N = 48`, `d = 3`:

| | MLP AC (ator) | n-tuple AC (ator) |
|---|---|---|
| MACs em ponto flutuante | 25·32 + 32·32 + 32·3 = **1 920** | **0** |
| ReLU | 64 | 0 |
| Comparações (termômetro) | 0 | ≈ **800** + 12 bits de validade |
| Operações de bit (montar `N` endereços) | 0 | 48 · 12 = **576** |
| Consultas (LUT) | 0 | **48** |
| Somas em ponto flutuante | (nas MACs) | 48 · 3 = **144** |

Duas leituras que precisam ir para o relatório: (i) o caminho de inferência da WNN é **livre de MAC em ponto flutuante**, que é a alegação defensável do PRD §5.5; (ii) o **codificador domina** a contagem de operações da WNN — o que a SPEC-01 T-01.7 já antecipava. Uma vitória de latência agregada é do codificador tanto quanto da rede, e o relatório tem de dizer isso.

**Correção material em T1 que esta spec descobre.** A conta de tamanho do PRD §5.5 (`2 268` entradas ocupadas × 2 B int16 = 4,4 KB, + 1,2 KB de mapeamento = **≈ 5,6 KB**) pressupõe posição de memória **escalar**. Nosso ator emite `d = 3` componentes, e a carga útil por entrada triplica:

```
2 268 × 3 × 2 B (int16) = 13,6 KB  +  1,2 KB (mapeamento)  ≈ 14,8 KB
```

contra ≈ 8,6 KB da MLP pareada. Isto é, na estimativa preliminar **o ator WNN é maior que a MLP pareada, não menor** — e o número "realizado" (chave do hash e fator de carga, PRD §5.5) só piora a razão. O PRD já previu o caso ("*Se a WNN acabar maior que a MLP pareada, isso é resultado*"), mas a estimativa da tabela precisa ser corrigida antes de virar expectativa. **Mitigação de projeto:** usar **um único discriminador com posições vetoriais** (Katopodis fig. 4.3a) e não três discriminadores escalares (fig. 4.3b) — um endereço, uma chave, três valores, em vez de três chaves e três valores. Numa tabela esparsa isso economiza ~2 chaves por entrada, o que é da ordem de 40 % do "realizado". Decisão de arquitetura registrada aqui e medida em T-05.7.

---

## 5. Tarefas

### T-05.1 — Núcleo do algoritmo, com o laço corrigido

Implementar `agents/ac_core.py`: janela de `m` passos por agente, retorno de `m` passos com bootstrap, vantagem, barreira de aplicação de atualizações, tratamento de terminal/truncado, `flush` de fim de episódio. **Agnóstico ao aproximador** — recebe dois objetos com `predict(x) → np.ndarray` e `update(x, grad)`.

Isto é o que garante o pareamento: se o laço estivesse duplicado nos dois agentes, H1 mediria a diferença entre duas implementações.

**Saída:** módulo + testes de unidade sobre trajetórias sintéticas com retorno analiticamente conhecido (recompensa constante, `γ` conhecido → `G` fechado); teste de que `m = T` reproduz o retorno Monte Carlo; teste de que truncamento faz bootstrap e terminação não.

### T-05.2 — Parametrização da política contínua

`agents/policies.py`: `ProjectedNormalPolicy` (C2, primária), `WrappedGaussianPolicy` (C1, ablação), `PlainGaussianPolicy` (C0, controle negativo). Cada uma expõe `sample(f, rng) → (z, env_action)`, `log_prob_grad(f, z) → grad` e `mean_action(f) → env_action`.

**Testes:**
- com `f = 0`, a distribuição de `θ` de C2 é uniforme (Kolmogorov–Smirnov contra `U[0,2π)`);
- `∇_f ln π` bate com diferenças finitas de `ln π(z|f)` para C0 e C1;
- para C2, o gradiente é a eq. 4.35 exata sobre `z` (verificação simbólica trivial, mas explícita, porque é a peça que sustenta "nenhuma modificação em Katopodis");
- a concentração angular empírica de C2 segue a curva teórica em `‖f‖`;
- `env_action` está sempre em `[0,1]²`; `z` bruto é o que sai para a atualização (teste de regressão contra o bug "amostrado ≠ executado").

### T-05.3 — Aproximadores

`agents/ntuple_ac.py` (ator: rede funcional com posições **vetoriais** `d = 3`; crítico: NTRN com esquecimento, com a variante funcional atrás de *flag*) e `agents/mlp_ac.py` (NumPy puro, ator e crítico, Adam).

**Testes:** verificação de gradiente por diferenças finitas na MLP; equivalência com referência PyTorch dentro de tolerância (só em `tests/`); asserção de estabilidade da §4.2 ativa e testada por um caso que deve abortar; determinismo bit a bit com semente fixa nos dois aproximadores.

### T-05.4 — Validação em benchmark, antes do ambiente real

**É o portão desta spec.** Dois níveis:

**(a) Reprodução de Katopodis** — `MountainCarContinuous-v0` e, se o orçamento permitir, `Hopper`. Alvos publicados, com IC *bootstrap* de 95 % sobre 1000 reamostragens (mesmo protocolo dele):

| Alvo | n-tuple AC | AC com MLP |
|---|---|---|
| Mountain Car | 93,3 (92,9; 93,8) | 0,02 (−0,03; −0,02) |
| Hopper | 748,8 (741,7; 758,0) | 643,6 (577,0; 746,0) |

Hiperparâmetros do apêndice A.3/A.6 dele, verbatim. O alvo do AC com MLP no Mountain Car (≈ 0, ótimo local de não mover o carro) é um **controle negativo** tão informativo quanto o positivo: se a nossa MLP resolver o Mountain Car, ela não é a MLP dele e a comparação de H1 não é a comparação dele.

**(b) Validação da ação circular** — nenhum benchmark clássico tem ação **angular**, então usamos um brinquedo próprio, um brinquedo próprio em `src/uavwnn/toy/nav.py`: 1 UAV, 1 sensor, área 200 × 200, `dt = 0,5 s`, ação `⟨θ, s⟩`, recompensa `+10` na visita e `−1/passo`. **Não** usa o gerador analítico da SPEC-01, que foi descartado — é um brinquedo independente, sem simulador e sem rede, com episódios de milissegundos.

É aqui que C0, C1 e C2 são comparados com custo desprezível — e é o único ponto do projeto em que a decisão da §4.3 é falsificável de forma barata. Métrica: passos até a visita, e erro angular mediano entre `atan2(f₂,f₁)` e o rumo ótimo.

**Critério:** os quatro números de (a) dentro dos ICs publicados **ou** com discrepância explicada e registrada; C2 estritamente melhor que C0 em (b), com IC separado.

### T-05.5 — Compartilhamento de parâmetros e o canal de identidade

Estender o brinquedo de T-05.4(b) para `Na ∈ {2, 4}` com 4 sensores. Medir:

- `p = 1 − (1 − R·Na/B)^n` teórico contra a fração medida de RAMs que leem ao menos um bit de índice, para `R ∈ {1, 4, 16}`;
- efeito de `R` sobre a diferenciação de comportamento (dispersão dos rumos escolhidos por agentes na mesma observação, exceto o índice);
- determinismo da barreira: mesma semente, duas ordens de evento diferentes → tabelas idênticas bit a bit (**teste automatizado**).

### T-05.6 — Varredura mínima de hiperparâmetros no brinquedo

Não é a varredura da SPEC-06 (que roda no problema real, sobre `T × n × N`). É a varredura que fixa **faixas defensáveis** para entregar à SPEC-06, no ambiente barato.

| Hiperparâmetro | Origem | Faixa inicial | Nota |
|---|---|---|---|
| `γ` | **Lamenza tab. 7.1** | **0,99, fixo** | não se varre |
| `η` (ator) | Katopodis A.3: 8·10⁻⁴ (MC), 2·10⁻⁵ (Hopper) | log-grade `10⁻⁵ … 10⁻²` | convenção de **média** (§4.2); ver absorção de `σ⁻²` abaixo |
| `m` | Katopodis A.3: 100 (MC), 1000 (Hopper) | `{1, 5, 20, 250}` | 250 = episódio inteiro, controle degenerado |
| `ϕ` (esquecimento do crítico) | Katopodis §5.2.3: 0,85 melhor nas 4 tarefas | `{0,7, 0,85, 1,0}` | `1,0` = sem esquecimento, controle |
| `λ` (esquecimento do ator) | novo, §4.3 | `{1,0, 0,999, 0,99}` | regulariza `‖f‖`; `1,0` = desligado |
| `σ_s` | Katopodis: 0,5 (MC), 0,3 (Hopper), em faixa `[−1,1]` | `{0,05, 0,10, 0,20}` **da faixa** | anelada até `σ_min = 0,02` |
| `σ_u` | — | **≡ 1, fixo** | não identificável (§4.3) |
| `K_warmup` | novo, §4.5 | `{0, 2·10³, 2·10⁴}` passos | crítico frio |
| `R` (réplicas do índice) | novo, §4.7 | `{1, 4, 16}` | só `Na > 1` |
| `T`, `n`, `N` | **SPEC-01 / G0** | herdados | com a ressalva abaixo |

> **Absorção de `σ⁻²` na taxa.** O gradiente é `Σ⁻¹(z − f)`; com `σ_s` fixa, `σ_s⁻²` é uma constante multiplicativa que confunde a varredura de `η` (mudar `σ_s` de 0,2 para 0,05 multiplica o passo efetivo por 16). Definir e varrer `η′ = η/σ_s²` explicitamente, e reportar `η′`. Sem isso as duas varreduras se contaminam e nenhum dos dois efeitos fica isolado.

> **Ressalva sobre `n`.** Katopodis usa `n ∈ {16, 32, 64}` e mede, no Mountain Car, que tuplas **maiores** vencem ("*agents are much more benefited from greater model capacity than generalization ability*"), com as melhores curvas em `N = 1` RAM. O PRD varre `n ∈ {8, 10, 12}` — todo o intervalo abaixo do que a referência usou em tarefa contínua. O risco "`2^n` explode" do PRD §9 já está resolvido pelo armazenamento esparso: com hash, `n` **não custa memória** até o endereço ser tocado; o custo de `n` grande é perda de generalização, não de bytes. **Recomendação a levar para a SPEC-06:** incluir ao menos `n = 16` na varredura. Pergunta em aberto (§9).

Também obrigatório: **normalização da vantagem**. `δ` chega a ~100 na nossa escala de recompensa (`+10` por visita, retorno de `m = 20` passos), contra `O(1)` do Mountain Car. Padronizar `δ` por média/desvio móveis, ou recortar em `±δ_max`, **identicamente nos dois aproximadores**. Isto não altera a recompensa (D3) — é normalização dentro do otimizador, e a distinção precisa estar escrita no relatório para não parecer *reward scaling*.

### T-05.7 — Pontos de instrumentação para T1/T2/T3

Não implementar o harness (é a SPEC-03), mas **expor e testar** o que ele consome:

- `snapshot()` → entradas ocupadas por tabela (ator, crítico), bytes de carga útil, bytes de chave, fator de carga, bytes do mapeamento de tuplas (`N × n` índices), separados em coluna **ator** e coluna **ator + crítico** (PRD §5.5 T1);
- limites de estágio para a decomposição da T2: montagem da observação / codificação / endereçamento / acumulação e amostragem;
- contadores estáticos de T3 (a tabela da §4.8) emitidos pela configuração, não medidos em tempo de execução;
- medição da economia do discriminador vetorial vs. três discriminadores escalares (§4.8), em bytes realizados.

---

## 6. Critério de saída

1. **T-05.4(a) fechada:** os quatro números de Katopodis reproduzidos dentro dos ICs publicados, ou a discrepância diagnosticada e registrada por escrito. Sem isto, nenhum resultado negativo da SPEC-06 é atribuível ao problema.
2. **T-05.4(b) fechada:** C2 (normal projetada) estritamente superior a C0 (gaussiana simples) no brinquedo, com IC separado. Se C2 **não** vencer, a §4.3 estava errada e a decisão precisa ser revista antes da SPEC-06 — não depois.
3. **Pareamento auditável:** um teste automatizado que instancia `NTupleAC` e `MlpAC` com a mesma semente e verifica que a sequência de `(obs, reward, terminal/truncated)` consumida e a forma da ação emitida são idênticas passo a passo. O teste falha se qualquer assimetria for introduzida.
4. **`γ = 0,99` fixo**, faixas iniciais dos demais hiperparâmetros registradas em `configs/spec05/` com a justificativa de origem (Katopodis, Lamenza, ou varredura própria).
5. **Nenhuma dependência de PyTorch fora de `tests/`.** Verificado por teste de importação.
6. **Asserção de estabilidade ativa** (§4.2) e coberta por um caso que deve abortar.
7. **Determinismo:** duas execuções com a mesma semente produzem tabelas idênticas bit a bit, inclusive com `Na > 1` e ordens de evento diferentes.
8. **Relatório curto** com: as curvas de validação, a comparação C0/C1/C2, a tabela de contagem de operações da §4.8, a correção da estimativa de T1 (§4.8) e a lista de correções ao Algoritmo 4.4 (§4.1). Vai direto para o capítulo de método de E2.
9. **PRD atualizado** em dois pontos, sem passar pela §5.4 (nenhum deles altera o MDP): a arquitetura do baseline passa a `D→32→32→3`, e a estimativa de tamanho do ator WNN em §5.5 é corrigida pelo fator `d = 3`.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **O ator WNN sai maior que a MLP pareada** (§4.8: ≈ 14,8 KB vs 8,6 KB, e piora no "realizado") | Já previsto pelo PRD §5.5. Discriminador vetorial em vez de escalares; medir e reportar honestamente; migrar o argumento para T3 (zero MAC em ponto flutuante), que é onde a diferença é estrutural. Reportar contra **ambas** as referências (D12) |
| **C2 não resolve a circularidade** e a política não aprende rumo | C1 e C0 implementadas como ablação barata em T-05.2; o brinquedo de T-05.4(b) responde em minutos. Se as três falharem, o diagnóstico é do codificador (volta para a SPEC-01), não do algoritmo |
| **`‖f‖` colapsa (exploração some cedo) ou explode** | Esquecimento no ator (`λ`) como encolhimento; projeção `r_max` como *fallback*; histograma de `‖f‖` como diagnóstico obrigatório — é o instrumento que separa "não aprendeu" de "convergiu cedo demais" |
| **`max_seconds_stalled = 30 s` impede o arranque.** Com a memória zerada o rumo é uniforme e a velocidade média é 7,5 m/s; se o agente não visita nenhum sensor em 15 decisões, o episódio é podado, o crítico só vê prefixos e a janela de `m = 250` nunca amadurece | Medir a taxa de visitação sob a política inicial **antes** de treinar (é uma conta de Monte Carlo no brinquedo de navegação, ou direto no ambiente de referência a 0,017 s por episódio — custo desprezível nos dois casos). Se for baixa demais, as opções são relaxar `max_seconds_stalled` **só durante um aquecimento de treino** — decisão a registrar na §10 do PRD — ou aceitar e reportar. Não improvisar durante a SPEC-06 |
| **Vantagem de variância explosiva por crítico frio** | Aquecimento `K_warmup`; normalização/recorte de `δ`; monitorar a fração de leituras em entradas com `c < c_min` |
| **`σ⁻²` confunde a varredura de `η`** | Varrer `η′ = η/σ_s²` e reportar `η′`, explicitamente |
| **Convenção soma/média trocada** entre o protótipo e as taxas de Katopodis, com fator `N` de erro | Decisão registrada (§4.2), conversão do protótipo, e um teste que verifica `f(x) = (1/N)Σ v[i](x)` |
| **Retropropagação da MLP escrita à mão erra em silêncio** e o baseline fica artificialmente fraco — o que *inflaria* H1 | Verificação de gradiente por diferenças finitas + equivalência com PyTorch em `tests/`; e o controle negativo do Mountain Car (a MLP **tem** de ficar em ≈ 0) |
| **Atualizações assíncronas com parâmetros compartilhados quebram reprodutibilidade** | Barreira por tique de decisão; teste de invariância à ordem de eventos |
| **O índice do agente é invisível ao mapeamento de tuplas** (§4.7: ~2 de 48 RAMs) | Réplicas `R` do one-hot; ou RAMs dedicadas; ou medir e reportar que individualidade não importa neste problema — que também é resultado |
| **Truncamento tratado como terminação** enviesa a política contra estados bons perto do limite de tempo | Explícito no contrato da §4.6, com teste. O caso do *stalling* fica como pergunta em aberto, decidida por medição |
| **A validação em Hopper estoura o orçamento** (Katopodis roda 5000 episódios × 10 repetições) | Hopper é opcional. O par Mountain Car (positivo + controle negativo) já valida os dois lados; Hopper entra se sobrar máquina |

---

## 8. Entregáveis

- `src/uavwnn/agents/ac_core.py` — laço AC on-policy, agnóstico ao aproximador
- `src/uavwnn/agents/policies.py` — `ProjectedNormalPolicy` (C2), `WrappedGaussianPolicy` (C1), `PlainGaussianPolicy` (C0)
- `src/uavwnn/agents/ntuple_ac.py` — ator funcional com posições vetoriais `d = 3`; crítico NTRN com esquecimento (+ variante funcional)
- `src/uavwnn/agents/mlp_ac.py` — NumPy puro, ator e crítico, Adam, sem PyTorch
- `src/uavwnn/agents/interface.py` — `Policy` (§4.6), com tipos e formas documentados
- `src/uavwnn/toy/nav.py` — brinquedo de navegação `⟨θ, s⟩`, independente do ambiente de referência
- `configs/spec05/` — hiperparâmetros de reprodução de Katopodis e faixas iniciais para a SPEC-06
- `reports/spec05-actor-critic.md` + figuras (curvas de validação, C0/C1/C2, histograma de `‖f‖`, tabela T3)
- `tests/test_ac_core.py`, `tests/test_policies.py`, `tests/test_mlp_grad.py`, `tests/test_pareamento.py`, `tests/test_determinismo.py`

---

## 9. Perguntas em aberto

1. **Poda por *stalling*: terminação ou truncamento?** Truncar por tempo pede bootstrap. Podar por inatividade é uma terminação induzida pela política: fazer bootstrap ensina que ficar parado "vale `V`"; não fazer ensina que vale zero, o que é uma penalidade implícita que Lamenza não escreveu na eq. 5-6. As duas escolhas alteram o sinal de aprendizado e nenhuma é obviamente fiel ao original. Decidir por medição no brinquedo, ou inspecionando o repositório de Lamenza (é a mesma investigação de P1/P4).

2. **`n` deve ser estendido para 16 ou 32 na varredura da SPEC-06?** Katopodis mediu, nas duas tarefas contínuas, que o tamanho de tupla domina e que a direção ótima é **oposta** entre elas (maior no Mountain Car, menor no Hopper). O PRD varre `{8, 10, 12}`, todo abaixo da faixa dele. Com armazenamento esparso o custo de `n` grande é generalização, não memória. Ampliar a varredura custa tempo de SPEC-06, que é o recurso escasso.

3. **A variância de Katopodis para o Hopper é 0,2 (§5.3) ou 0,3 (tab. A.3)?** O original é inconsistente. Se reproduzirmos o Hopper e o número publicado só sair com um dos dois, isso resolve — e é um achado de reprodutibilidade a registrar.

4. **Vale um crítico com desconto diferente do ator?** `γ = 0,99` é de Lamenza e não se negocia para a política. Mas Katopodis usa `γ = 1,0` no crítico do AC em três das quatro tarefas, e nosso episódio tem no máximo 250 passos (`γ^250 ≈ 0,08`). Um crítico não descontado teria menos viés de horizonte. É um desvio do algoritmo, não do problema — mas precisa de decisão consciente.

5. **Se `Na = 1`, o índice `k` do agente é constante.** Mantê-lo preserva a dimensão da observação de Lamenza (25 escalares) mas gasta bits do codificador com informação de entropia zero. Remover economiza bits reais mas quebra a identidade dimensional com a eq. 5-1. Recomendação provisória: manter, e medir o custo em bits.

6. **`σ_s` anelada — segundo que cronograma, e ele conta como hiperparâmetro varrido ou como protocolo fixo?** Se for varrido, multiplica a grade da SPEC-06. Se for fixo, é uma escolha não justificada por medição.

7. **A barreira por tique de decisão sobrevive ao `gradysim`?** Ela pressupõe que todos os UAVs disparam `handle_timer("decide")` no mesmo instante simulado. Com `algorithm_iteration_interval = 2 s` e temporizadores iniciados juntos isso deve valer, mas é uma premissa sobre o simulador que só a SPEC-00 pode confirmar. Se os tiques forem desalinhados, a barreira precisa virar uma janela, e o determinismo passa a depender de uma tolerância.

8. **A referência de latência de ~15 µs do protótipo (RNF-03) foi medida com saída escalar.** Com `d = 3` e o codificador dominando a contagem de operações (§4.8), o número esperado muda. Vale medir de novo em T-05.7 antes de a expectativa entrar no relatório?
