# SPEC-07 — Camada de ruído, KF com entrada de controle e validação de consistência

**Etapa:** 2 (extensão de Fusão de Dados) · **Depende de:** nada para a bancada sintética; SPEC-00 apenas para a integração final com o `gradysim` · **Estimativa:** 3 d nominais (ver ressalva em §7)
**Alimenta:** SPEC-08 (portão `G2`) e SPEC-09 · **Referência:** PRD §1 (achados A2 e A3, regra *bin ≈ σ*), §6.2, RF-04, RF-05, RNF-05, §10/P3

---

## 0. Revisão após a SPEC-00 — o ponto de inserção mudou

Esta spec foi escrita supondo que o ambiente seria nosso e que o ruído entraria por uma interface `StateView`. **O ambiente passou a ser o de Lamenza, verbatim, e não é editável.** O ponto de inserção agora é **subclasse do `GrADySEnvironment`, sobrescrevendo apenas `observe_simulation_relative_positions()`**. Ver `specs/ARQUITETURA.md` §3.

Três consequências, todas favoráveis:

**(a) A física fica intocável por construção.** Movimento, alcance de comunicação, coleta, recompensa e terminação vivem no `Simulator`, fora do alcance do método sobrescrito. RF-04 ("a verdade não vaza para a política") deixa de precisar de *taint tracking* — a subclasse só pode corromper o que devolve.

**(b) O erro de GPS próprio é um deslocamento COMUM, e isso simplifica tudo.** Verificado numericamente. Toda entrada da observação tem a forma `(p_k − p_i + S)/(2S)`. Se a estimativa própria erra por `e`:

```
(p̂_k − p_i + S)/(2S) = (p_k − p_i + S)/(2S) + e/(2S)
```

O mesmo `e/(2S)` soma a **todas** as entradas válidas — sensores e vizinhos — e o padding `-1` não é tocado. Logo:

- corromper = somar duas linhas de código, não reconstruir a observação;
- **aplicar o filtro = somar `(p̃_k − p̂_k)/(2S)`**, pela mesma razão;
- a escala do erro em unidades de observação é `σ/(2S)`. Com `S = 100 m`, `σ = 5 m` desloca **0,025** — comparável direto à largura do bin do termômetro. É a regra *bin ≈ σ* em unidades da entrada real.

**(c) Guardas fortes e baratas.** Com `σ = 0` a subclasse devolve observação **bit-idêntica** ao pai (verificado). E o tempo de conclusão com `σ = 0` tem de continuar reproduzindo a fig. 7.3, que a SPEC-00 já travou em 9/9 células.

## 0-CORTE. O ruído entra apenas no GPS do drone (D13, 2026-08-15)

> **Decisão de escopo.** O canal do sensor saiu. Com ele saem: o bloco 2b, o
> filtro de informação, a hipótese **H4** e as pendências **TS3** e **P3**.
>
> | item | estado |
> |---|---|
> | GPS do drone, `p̂_k = p_k + e_k` | **é o canal, e o único** |
> | KF com **entrada de controle** | permanece — é onde vive o achado A3 (18,8×) |
> | ranking sobre a estimativa filtrada | permanece |
> | NEES/NIS | permanece |
> | interseção de covariância entre UAVs (H5) | permanece — Etapa 3 |
> | canal do sensor, `z = p_i + b_i + v_t` | **fora** |
> | filtro de informação (alvo estático) | **fora** |
> | `R(d)` crescente com a distância | **fora** |
> | H4 (busca ativa) | **retirada** — perdeu a causa física, não só o experimento |
> | TS3, P3 | **fechadas por decisão**, não resolvidas |
>
> **O que o corte ganha.** O canal do drone é o único com modelo fechado — ruído
> branco, estado móvel, comando conhecido — e é onde a medição de 18,8× existe. O
> que sai dependia de uma decisão de física nunca tomada (TS3) e tinha ganho
> conjectural.
>
> **O que o corte custa.** A H4 perde a causa física: com ruído só na posição
> própria, a covariância converge para regime estacionário e não há incerteza
> *por sensor* a reduzir, logo aproximar-se não compra informação. O degrau 5 da
> escada continua como ablação, sem hipótese forte associada.
>
> **Trechos abaixo que tratam do canal do sensor estão revogados.** Foram
> mantidos porque a análise é boa e é a primeira coisa a reaproveitar se o escopo
> crescer. Nada deles está em execução.

---

### ~~O modelo de ruído mudou: o canal primário é o SENSOR~~ — REVOGADO por D13

Esta spec supunha o GPS do drone como canal principal. **Inverteu-se.** O desenho agora é: o drone recebe, a cada época, um relatório corrompido da posição de cada sensor e o refina recursivamente; **tudo que ele deriva daí, inclusive o ranking, é computado sobre a estimativa filtrada.** Detalhe completo em `specs/ARQUITETURA.md` §3.

```
z_{i,t} = p_i + b_i + v_{i,t}      b_i ~ N(0, σ_b²)  viés FIXO por sensor
                                   v_{i,t} ~ N(0, R(d))  BRANCO por relatório
R(d) = (σ_0 + κ·d)² I              precisão PIORA com a distância
```

| # | A spec dizia | Correção |
|---|---|---|
| 1 | O erro dos sensores é **constante no episódio**; sortear por passo seria modelar errado | **Invertido.** A componente branca é o que dá sentido à filtragem — sem ela não há o que filtrar. O que existe é branco **mais** um viés fixo, e o viés é o piso irredutível |
| 2 | O canal dos sensores é degenerado na Etapa 2 porque ouvir = coletar (achado C5) | **Resolvido.** O relatório não vem por rádio: vem do **canal global reinterpretado**. Nenhuma mudança de protocolo, e o canal deixa de ser degenerado — há medições repetidas a fundir |
| 3 | O filtro dos sensores é um KF | É **filtro de informação**: estado estático, `F=I`, `Q=0`, sem entrada de controle. Mais barato e mais estável para constante |
| 4 | Ganho ≈ o 18,8× do achado A3 | A3 é sobre a posição **própria**. Com `σ_rel = √(σ_own² + σ_sensor²)`, o ganho no que a política enxerga é limitado pelo pior canal (pendência **T2**) |
| 5 | `dt` a definir | É **0,5 s**, do `benchmark.py`. E a posição lida está **um tique atrasada** (quirk Q3): o KF do canal próprio precisa modelar isso |
| 6 | Mobilidade com inércia como ablação fácil | Está **dentro do `reset()`** de referência; trocá-la é mais invasivo que sobrescrever a observação. Sem inércia o modelo de processo é quase exato e `Q` degenera — declarar como limitação |

### O experimento que esta spec ganha de presente

Se o filtro de informação **ignorar o viés `b_i`**, ele converge para `p_i + b_i` com `P → 0`: fica arbitrariamente confiante num valor errado, e o **NEES estoura**. É predição falsificável, e é o melhor material da Etapa 2 para a disciplina — o filtro ingênuo passa em qualquer inspeção visual e **só a validação de consistência o denuncia**. A correção é limitar `P` inferiormente por `σ_b²`; a diferença entre as duas versões é resultado a reportar.

### O canal do GPS próprio permanece

`σ_own` **fixo**, com KF de entrada de controle, **fora da varredura** (que é toda do canal do sensor). A grade continua `σ_sensor × T` = 12 configurações. Duas razões: o achado A3 só existe neste canal, e sem incerteza na posição própria a **interseção de covariância da Etapa 3 perde o objeto** — drones transmitiriam posições exatas e não haveria erro correlacionado a fundir.

### Uma decisão que esta spec precisa tomar

Com posição própria ruidosa, o agente real ordenaria os sensores pela distância **percebida**; o ambiente ordena pela **verdadeira** e entrega só os `N*_s` mais próximos. Duas leituras:

- **(i)** herdar seleção e ordenação verdadeiras, só deslocar coordenadas — erro de segunda ordem, só importa quando dois sensores distam menos que `σ`;
- **(ii)** recomputar seleção e ordenação a partir das posições estimadas — mais fiel, acessível na subclasse por leitura do `Simulator`.

**Recomendação: (ii)**, com (i) como ablação para quantificar a idealização. Em `Ns = 36` num quadrado de 200 m a distância típica entre vizinhos é ~30 m, comparável a `σ = 10 m` — não é desprezível.

---

## 1. Objetivo

Entregar, **validado fora do laço de RL**, o estimador de estado que substitui o canal global de Lamenza: (i) uma camada de corrupção parametrizada e reprodutível, (ii) um filtro de Kalman linear de 2 estados que usa a **velocidade comandada pela política** como entrada de controle, (iii) um filtro de informação para o canal estático dos sensores, (iv) o aparato de consistência NEES/NIS que autoriza qualquer número a jusante, e (v) o teste automatizado do invariante de vazamento (RF-04).

Ao fim desta spec, a SPEC-08 deve poder rodar a escada de ablação 0–5 e a varredura `σ × T` **sem escrever uma linha de infraestrutura de fusão**, e cada degrau da escada deve ter sido sintonizado de forma justa contra a bancada sintética.

## 2. Por que esta spec existe, e por que agora

**Por que existe.** Lamenza (2025, p. 17–18) é explícito: a observação do agente inclui posição de todos os UAVs e status de visitação de todos os sensores, e isso só se sustenta por um "enlace global de satélite" ao qual os UAVs teriam acesso. O próprio autor registra isso como limitação: *"We still view this as one of the work's current limitations and plan to address this in the future by constraining observable information to locally communicated data."* Esta spec é o preenchimento desse buraco — e é o único ponto do projeto em que a disciplina de Fusão de Dados tem conteúdo próprio, não emprestado.

**Por que agora, antes de qualquer política treinada.** Três razões, em ordem de força:

1. **Nada aqui depende de RL.** Ruído e filtro são funções de uma trajetória e de uma sequência de comandos. Uma trajetória sintética com comandos por partes constantes reproduz a estatística que importa — mudança abrupta de rumo a cada intervalo de decisão — sem simulador e sem treino. É a spec que mais se beneficia de rodar em paralelo (README §"Racional da divisão").
2. **RF-05 é uma trava, não uma verificação.** Um KF mal sintonizado produz erro médio plausível e covariância mentirosa. Foi exatamente o que aconteceu no experimento preliminar (PLANO §9.4): a primeira tentativa, KF de velocidade constante com 4 estados, ficou **pior que a medida crua** e o número só não envenenou a análise porque foi comparado contra a medida crua por acaso. Consistência medida antes de qualquer resultado de RL não é higiene, é o que separa este trabalho de um relatório com número bonito e conclusão errada.
3. **A regra *bin ≈ σ* fecha o ciclo com a SPEC-01.** O KF não melhora a política diretamente: ele derruba o `σ` da estimativa, o que habilita um termômetro mais fino, que é o que recupera a acurácia (PRD §1). A resolução `T*` recomendada para a Etapa 2 é **saída desta spec**, não da SPEC-08.

E o achado que dá substância à proposta (PRD §1, A3; PLANO §9.4): o KF de velocidade constante falha **por motivo estrutural, não por bug**. O UAV é comandado por uma política que muda de rumo bruscamente a cada `algorithm_iteration_interval`; um modelo de velocidade constante trata cada decisão como surpresa e fica permanentemente atrasado. Mas o comando não é surpresa para nós — a política acabou de escolhê-lo. Usá-lo como `u` na predição é a diferença entre 5,06 m e 0,67 m de erro com `σ_GPS = 10 m`.

## 3. Escopo

**Dentro (revisto por D13):** o modelo de corrupção do **GPS próprio** (`p̂_k = p_k + e_k`) e o do **atraso de broadcast** — este último porque a Etapa 3 precisa dele para a interseção de covariância; o **KF de 2 estados com entrada de controle**; os degraus 0–5 da escada; NEES/NIS. ~~Corrupção da posição de sensor por enlace local~~ e ~~filtro de informação para estado constante~~ **saíram do escopo**: a posição dos sensores permanece exata, como no problema original.

**Fora:** interseção de covariância e fusão entre UAVs (SPEC-09; aqui só se provisiona a API de mensagens atrasadas); canais de covariância na observação, degrau 5 (codificação é da SPEC-01/SPEC-08; aqui entrega-se apenas `P` como saída documentada); qualquer experimento com política treinada (SPEC-08); o caso **não linear** de descoberta por distância/RSSI com trilateração (P3 — ver §9).

**Convenções fixadas aqui e usadas por todas as specs a jusante:**

| Símbolo | Significado | Unidade |
|---|---|---|
| `p` | posição verdadeira, `[x, y]` | m, em `[-100, 100]²` |
| `z` | medida corrompida | m |
| `x̂`, `P` | média e covariância da estimativa | m, m² |
| `u` | velocidade **comandada**, `u = s·[cos θ, sin θ]` | m/s, `‖u‖ ≤ 15` |
| `dt_dec` | intervalo de decisão (`algorithm_iteration_interval`) | 2 s |
| `dt_gps` | período entre correções de GPS | s (nominal 1 s — ver §9) |
| `dt` | passo efetivo de uma chamada de `predict` | s |
| `σ_gps` | desvio padrão do ruído de posição própria | m, ∈ {1, 3, 5, 10} |
| `σ_prior` | erro do levantamento de posição do sensor | m |
| `σ_sens` | erro do relato de posição feito pelo sensor | m |

> **Unidade canônica é o metro, sempre.** A corrupção acontece em metros, sobre a posição absoluta, **antes** de qualquer subtração relativa e **antes** de qualquer normalização para o codificador. O `σ` normalizado que aparece no PLANO §9.4 (0,002–0,007) é `σ/(2·scenario_size) = σ/400` e existe só para comparação com a tabela de sensibilidade do achado A2; **nenhum código deve carregar σ normalizado**. Duplicar normalização é o erro silencioso mais provável desta spec.

---

## 4. Bancada sintética

Nenhuma tarefa de §5, exceto a integração final, precisa do `gradysim`. A bancada tem três níveis, do mais controlado ao mais fiel.

| id | Bancada | O que gera | Papel |
|---|---|---|---|
| **B1** | comando sintético paramétrico | `u` constante por partes, sorteado a cada `dt_dec`; distribuição do salto de rumo `Δθ` controlada | isola o efeito que matou o KF-VC; permite varrer de "rumo quase constante" a "rumo i.i.d. uniforme" |
| **B2** | rollout guloso analítico | reusa `env/analytic.py` da SPEC-01 (`dt = 2 s`, `v = 15 m/s`, rumo para o sensor não visitado mais próximo) | é a estatística **on-policy** aproximada; bancada primária para sintonia |
| **B3** | trajetória do `gradysim` | quando a SPEC-00 entregar | fidelidade |

**Dimensões da bancada:** `M = 100` corridas Monte Carlo × `K = 250` passos de decisão (500 s / 2 s) por configuração, para cada `σ_gps ∈ {1, 3, 5, 10}`. `M = 100` é escolha do intervalo qui-quadrado (§5.6), não arbitrária.

> **Invariante V-7.1 (fidelidade do resíduo).** O que a bancada precisa reproduzir não é a trajetória, é o **resíduo de modelo** `e_k = Δp_k − u_k·dt` (§5.5). Assim que B3 existir, comparar a distribuição de `e_k` entre B2 e B3 por Kolmogorov–Smirnov, estratificada por `‖u_k − u_{k−1}‖`. Divergência significativa invalida o `Q` sintonizado e obriga a re-sintonizar sobre B3. É o análogo direto do invariante V-1 da SPEC-01, e pela mesma razão: sintonizar contra a distribuição errada é o modo de falha caro.

**Semeadura e números aleatórios comuns (CRN).** Hierarquia com `numpy.random.SeedSequence(root).spawn()`, um fluxo por `(canal, nó, episódio)`. Duas exigências, ambas testáveis:

1. **Independência entre canais.** Consumir o fluxo de GPS não pode alterar o fluxo do enlace local. Sem isso, ligar um degrau da escada muda o ruído dos outros e a comparação deixa de ser pareada.
2. **CRN em `σ`.** A camada gera o desvio padronizado `ε ~ N(0, I₂)` a partir de um fluxo que depende de `(semente, nó, t)` e **não** de `σ`; a corrupção é `z = p + σ·ε`. Consequência: as quatro condições de `σ` da varredura veem exatamente a mesma realização normalizada, e a comparação entre elas é pareada. O mesmo vale para `T`: mudar a resolução do termômetro não pode mudar o ruído. Isso reduz a variância da varredura `σ × T` — o experimento mais caro do projeto (PRD §6.2) — sem custo nenhum.

---

## 5. Tarefas

Marcação de prioridade: **[N]** = não pode cair; **[C]** = cortável pela linha de corte (§7).

### T-07.1 — Camada de corrupção `noise.py` **[N]**

Três canais, com naturezas estatísticas distintas — e é a diferença entre elas que dá conteúdo ao trabalho.

**(a) Posição própria (GPS).** Modelo nominal, branco e isotrópico:

```
z_k = p_k + σ_gps · ε_k ,   ε_k ~ N(0, I₂) i.i.d.
R    = σ_gps² · I₂          [m²]
```

Modo alternativo `gps_gauss_markov`, **desligado por padrão e obrigatório como ablação de robustez**: erro de GPS real não é branco — tem componente lentamente variável (multicaminho, geometria de constelação, ionosfera). Modelado como Gauss–Markov de 1ª ordem somado a um branco:

```
b_k = e^(-dt/τ) · b_{k-1} + sqrt(1 - e^(-2 dt/τ)) · σ_b · η_k ,  η ~ N(0, I₂)
z_k = p_k + b_k + σ_w · ε_k
```

com `τ ∈ {30, 120, 300} s` e `σ_b² + σ_w² = σ_gps²` (mesma variância marginal, para que a comparação isole a **correlação**, não a magnitude). O KF nominal assume ruído branco; sob este modo ele fica **otimista** — `P` menor que o erro real — e é isso que o NEES tem que detectar. Se não detectar, o detector não presta (ver T-07.7).

**(b) Posição dos sensores, descoberta por enlace local.** Adotado o **caso linear** de P3: o sensor transmite a própria posição estimada. Dois componentes, e a distinção entre eles é o ponto sutil do canal:

```
prior de levantamento:  z_prior,i = p_s,i + σ_prior · β_i      β_i ~ N(0, I₂), sorteado UMA VEZ por episódio
relato por contato:     z_i,k     = p_s,i + σ_sens · w_{i,k}   w i.i.d. a cada relato
```

O prior existe porque Lamenza (p. 17–18) justifica o conhecimento das posições dos sensores exatamente assim: *"sensors are stationary and their positions can be noted during their deployment"*. Corromper esse levantamento é a extensão mínima e honesta. **`β_i` é um viés constante no episódio, não um ruído**: refiltrar o prior não reduz erro nenhum, e um filtro de informação que trate cada leitura do prior como amostra nova converge para uma covariância mentirosa com confiança crescente. Modo `sensor_stored` (o sensor retransmite um valor armazenado, e portanto o mesmo erro toda vez) existe como ablação e serve para exercitar o piso de variância de T-07.3.

**Interação com `communication_range` que precisa estar escrita.** Em Lamenza a coleta é automática: o sensor que responde ao heartbeat é imediatamente marcado como visitado (RF-03) e **sai de `P^s`**. Logo, no modelo fiel, o UAV ouve cada sensor **exatamente uma vez, no instante em que ele deixa de importar para a observação**. Consequência dura: na Etapa 2 fiel, o canal de sensores é **o prior e mais nada**, e o filtro de informação não tem o que fundir. Isso não invalida a peça — ela é exigida pela Etapa 3, tem forma fechada testável e custa pouco — mas exige honestidade: seu ganho na Etapa 2 é aproximadamente nulo **por construção do problema original**, e o relatório precisa dizer isso em vez de exibir um filtro que não filtra. A alternativa (`detection_range > communication_range`, o sensor anuncia-se antes de ser coletado) **modificaria o cenário** e exige linha na tabela de desvios do PRD §5.4. Ver §9.

**(c) Atraso de broadcast (provisão para a SPEC-09).** Mensagem emitida em `t_e` é entregue em `t_e + Δ`, com `Δ = Δ_0 + jitter`, `jitter ~ U[0, Δ_j]`, e perda independente com probabilidade `p_loss`. Nesta spec entrega-se a API, os testes unitários e a **fila de medidas ordenada por carimbo de tempo** exigida por T-07.2; o canal só é exercitado em cenário na SPEC-09. Motivo de estar aqui: o tratamento de medida fora de ordem é uma propriedade do filtro, não do multiagente, e retrofitá-lo depois obrigaria a re-sintonizar tudo.

**Parametrização e reprodutibilidade.** Toda configuração é um dataclass serializável (`NoiseConfig`), gravado junto com o resultado; nenhum parâmetro de ruído com valor default espalhado pelo código. Teste: mesma semente ⇒ fluxo idêntico bit a bit; `σ` diferente ⇒ mesmos `ε`.

### T-07.2 — KF com entrada de controle `kalman.py` **[N]**

**Estado e matrizes.** Duas dimensões, e nada mais: a velocidade **não** é estado, porque é conhecida.

```
x = [x, y]ᵀ ∈ R²                      (posição, m)
F = I₂                                 (2×2, adimensional)
B = dt · I₂                            (2×2, s)
u = s·[cos θ, sin θ]ᵀ ∈ R²             (m/s, comando da política no passo)
H = I₂                                 (GPS entrega posição diretamente)
R = σ_gps² I₂                          (m²)
Q = Q(dt)                              (m², ver abaixo)
```

**Ciclo:**

```
predição:  x̂⁻ = x̂ + dt·u
           P⁻ = P + Q(dt)
inovação:  ỹ  = z − x̂⁻
           S  = P⁻ + R
ganho:     K  = P⁻ S⁻¹
correção:  x̂  = x̂⁻ + K ỹ
           P  = (I − K)P⁻(I − K)ᵀ + K R Kᵀ        (forma de Joseph)
```

`S⁻¹` é uma inversa 2×2 em forma fechada, nunca `np.linalg.inv`. A forma de Joseph é obrigatória: custa ~30 flops a mais (T-07.9) e garante simetria e positividade mesmo com `K` subótimo — e `K` **será** subótimo nas ablações de `Q` mal sintonizado, que é justamente quando a forma simples `P = (I−K)P⁻` pode produzir `P` não positiva definida e um NEES sem sentido.

**`Q` — o que ele representa.** Três fontes de erro de processo, com composições diferentes em `dt`, e confundi-las é um erro sutil real:

| Fonte | Modelo | `Q(dt)` | Composição |
|---|---|---|---|
| Turbulência / vento de banda larga | erro de velocidade branco | `q_w · dt · I₂` | **invariante à taxa**: `Q(2dt) = Q(dt) + Q(dt)` |
| Erro de atuação (comando não cumprido) | erro de velocidade **constante dentro do passo** | `σ_v² · dt² · I₂` | **não invariante**: `Q(2dt) ≠ 2·Q(dt)` |
| Discretização do integrador de mobilidade | ver abaixo | absorvido em `σ_v²` | por passo de comando |

Modelo adotado: `Q(dt) = q_w·dt·I₂ + σ_v²·dt²·I₂`.

> **Armadilha de taxa.** Como `F = I` e `u` é constante entre decisões, propagar a média em 20 passos de 0,1 s ou em 1 passo de 2 s dá **exatamente o mesmo `x̂⁻`**. A covariância, não: o termo `dt²` composto 20 vezes dá `20·σ_v²·(0,1)² = 0,2·σ_v²`, contra `σ_v²·4` de um passo único — fator 20 de diferença. Portanto **a taxa do filtro é parte da definição de `Q`** e precisa estar fixada, gravada na configuração e verificada por teste. Regra adotada: **um `predict` por intervalo entre eventos de correção ou decisão, nunca por tick de mobilidade.** É também o que torna o filtro barato (T-07.9).

**`Q` adaptativo ao salto de comando (proposta, com ablação).** O erro de atuação não é estacionário: ele é grande logo após uma virada e pequeno em rumo constante. Com `DynamicVelocityMobilityHandler` a resposta é de 1ª ordem com constante `τ_m`, e o erro de velocidade logo após uma reversão de rumo chega a `2·s ≈ 30 m/s`. Um `Q` dimensionado para esse pior caso é grande demais no regime permanente; um `Q` médio é inconsistente exatamente nos passos que mais importam. Duas saídas, ambas a medir:

- **Barata:** `Q_k = (q_w·dt + (σ_v0² + κ·‖u_k − u_{k−1}‖²)·dt²)·I₂`. Custa 2 multiplicações e 1 soma por passo; `κ` sai da sintonia de T-07.5.
- **Exata:** predizer com a resposta do modelo de inércia em vez do comando cru, `ū_k = u_k + (v_{k−1} − u_k)·(τ_m/dt)·(1 − e^{−dt/τ_m})`, o que remove a maior parte do viés e deixa um `Q` genuinamente pequeno. Exige conhecer `v_{k−1}` (a velocidade realizada anterior, que o UAV conhece pela própria IMU — suposição declarada) e `τ_m`.

**Sem medida no passo.** Só a predição roda; `P` cresce monotonicamente por `Q(dt)`. O estado exposto à política é sempre `predict_to(t_decisão)` — a estimativa propagada até o instante da decisão, que pode não ter recebido correção nenhuma desde o último fix. A API não permite ler `x̂` sem informar o instante.

> **Armadilha de sincronia — o erro mais provável desta spec.** A medida `z` refere-se ao instante `t_z` do fix, não ao instante `t_d` da decisão. A ordem correta é: `predict_to(t_z)` → `update(z)` → `predict_to(t_d)`. Aplicar uma medida antiga no instante corrente introduz viés `‖u‖·(t_d − t_z)`; a 15 m/s com 0,2 s de latência isso são **3 m**, a mesma ordem do `σ` que se está tentando remover — e o viés é sistemático, logo não aparece como variância nem some com média. Exigências: (i) toda medida carrega `t_meas`; (ii) o filtro carrega `t_state` e só avança por `predict_to(t)`; (iii) `predict_to(t < t_state)` levanta exceção; (iv) medidas fora de ordem são bufferizadas por até `dt_dec` e reprocessadas em ordem, com contador exposto de descartes. Teste de regressão: injetar deliberadamente `z` com carimbo defasado e verificar que o NEES acusa.

**Inicialização.** `x̂_0 = z_0`, `P_0 = R`. Não `x̂_0 = 0` com `P_0` grande: essa escolha produz NEES explodindo nos primeiros passos e obriga a descartar transiente, o que é exatamente onde a política toma as decisões mais consequentes. Com `x̂_0 = z_0` e `P_0 = R` a consistência vale desde `k = 0`, exatamente. Sem *warm start* entre episódios: as posições são re-sorteadas (RF-01) e reaproveitar `P` seria fraude.

**Malha fechada — nota teórica que vai para o relatório.** Na bancada, `u` é exógeno. Com a política real, `u` é escolhido a partir de `x̂`, logo comando e erro de estimativa ficam correlacionados. O estimador continua correto: `u` é **conhecido exatamente** (nós o comandamos), então a predição permanece exata independentemente de como `u` foi escolhido — é equivalência à certeza / princípio da separação. O que **não** se transfere é a estatística de `u`: a política pode gerar padrões de comando (por exemplo circular sob incerteza) ausentes de B2. Por isso `Q` sintonizado em malha aberta precisa ser **reverificado por NEES em malha fechada** na SPEC-08. Isso é um item de aceite da SPEC-08, registrado aqui.

**Verificação por forma fechada (teste unitário, não estatístico).** Com `Q = 0`, `H = I`, `R = σ²I` e medidas a cada passo, o filtro é exatamente a média amostral: `P_k = (σ²/k)·I`. Com `q` escalar e `R = σ²`, o regime permanente escalar é

```
p∞² + q·p∞ − q·σ² = 0   ⇒   p∞ = ( −q + sqrt(q² + 4qσ²) ) / 2 ,   K∞ = (p∞ + q)/(p∞ + q + σ²)
```

Ambos viram asserções numéricas em `tests/test_kalman.py`. Testar filtro contra si mesmo é o modo clássico de não descobrir nada.

### T-07.3 — Filtro de informação para estado constante **[N]**

Sensor é estático: `F = I`, `Q = 0`. Nesse caso a forma de informação é estritamente melhor que a forma de covariância.

```
Y = P⁻¹   (matriz de informação, m⁻²)
y = P⁻¹ x̂ (vetor de informação)

atualização por medida (H = I, R = σ_i² I₂):
   Y ← Y + σ_i⁻² I₂
   y ← y + σ_i⁻² z_i
extração da média (só quando a observação é montada):
   x̂ = Y⁻¹ y
```

Com `R` isotrópico isso colapsa em escalar: `Y = (Σ_i σ_i⁻²)·I` e `x̂ = (Σ_i z_i/σ_i²)/(Σ_i σ_i⁻²)` — **média ponderada recursiva pela precisão**, sem predição, sem ganho, sem inversa no caminho de atualização (T-07.9: ~6 flops por relato). Três propriedades que justificam a escolha e vão para o relatório:

1. **Numericamente estável para estado constante**: `Y` só cresce, nunca há subtração de matrizes, logo não há perda de positividade — que é exatamente o modo de falha do KF em covariância quando `P → 0`.
2. **Aceita ignorância total como prior**: `Y_0 = 0` é legítimo e representa "não sei nada", o que é impossível de escrever em forma de covariância. Aqui, porém, usa-se `Y_0 = σ_prior⁻²·I` e `y_0 = Y_0·z_prior`, porque o levantamento existe (T-07.1b).
3. **Composição trivial**, que é o que a SPEC-09 vai precisar: fundir duas fontes independentes é somar informação.

**Piso de variância — obrigatório.** Sob `sensor_stored` (ou sob qualquer viés constante não modelado), o erro do relato é o **mesmo** em cada contato e a soma de informações é inválida: o filtro converge para o valor enviesado com `P → 0`. Implementar `Y ← min(Y, σ_piso⁻²·I)` com `σ_piso` configurável e **contador de saturação exposto**. O NEES do canal é o detector: sem o piso ele estoura; com o piso ele passa. Essa é uma demonstração barata e didática de por que consistência importa, e merece uma figura no relatório.

### T-07.4 — Coordenadas relativas e propagação de covariância **[N]**

A observação de Lamenza (eq. 5-1) é **relativa**. O que a política enxerga não é `x̂_a` nem `x̂_s,i`, é a diferença:

```
d̂_i   = x̂_s,i − x̂_a
Cov(d̂_i)      = P_s,i + P_a           (fontes independentes: prior/relato do sensor vs GPS próprio)
Cov(d̂_i, d̂_j) = P_a        (i ≠ j)    ← termo comum, NÃO nulo
```

Duas consequências que precisam estar escritas antes de alguém escrever o codificador:

1. **Os 12 slots de `P^s` têm erros correlacionados**, porque todos compartilham `−x̂_a`. Um erro de posição própria desloca os 12 alvos na mesma direção. Isso é boa notícia para a proposta: filtrar bem **um** canal melhora **doze** entradas da observação simultaneamente — é o mecanismo pelo qual o KF paga a política. E é má notícia para qualquer tratamento que assuma independência entre slots, incluindo a interseção de covariância ingênua da SPEC-09.
2. **A regra *bin ≈ σ* é governada pelo pior canal.** `σ_rel = sqrt(σ_a² + σ_s²)`. Se o prior do sensor tem `σ_prior = 5 m` e é irredutível (T-07.1b), derrubar `σ_a` de 10 m para 0,7 m leva `σ_rel` de 11,2 m para 5,05 m — fator 2,2, não 15. **O ganho da fusão na observação é limitado pelo canal que não se pode filtrar**, e reportar o ganho do canal próprio como se fosse o ganho da observação seria escolher o numerador conveniente. Entregável desta tarefa: a tabela `σ_gps × σ_prior → σ_rel` filtrado e não filtrado, que é a entrada real da varredura da SPEC-08.

### T-07.5 — Sintonia de `Q` e `R` **[N]**

**`R` não se sintoniza — mede-se.** Em simulação nós injetamos o ruído, logo `σ_gps` é conhecido exatamente e `R = σ_gps² I` é verdade, não estimativa. Isso é uma alavanca de diagnóstico forte: **qualquer inconsistência detectada é culpa de `Q`, da sincronia ou de um bug — nunca de `R`.** Ajustar `R` para "fazer o NEES passar" é proibido e deve ser rejeitado em revisão. `R` só é varrido em T-07.7, deliberadamente errado, para calibrar o detector.

**`Q` sai de medição direta, não de busca.** Com a verdade disponível na bancada, o resíduo de modelo é observável sem envolver ruído de medição nenhum:

```
e_k = p_{k+1} − (p_k + u_k · dt)          [m]
Q̂   = Cov(e_k), estratificado por ‖u_k − u_{k−1}‖
```

Ajustar `q_w`, `σ_v0²` e `κ` por mínimos quadrados sobre a decomposição da §T-07.2 é o procedimento primário. Busca em grade é só verificação.

**Verificação por grade (a figura do relatório).** Varrer `q` em grade log e traçar, na mesma figura: (i) NEES médio contra a faixa qui-quadrado, (ii) RMSE da estimativa. O `q` de RMSE mínimo e o menor `q` consistente costumam ficar próximos mas não coincidem. **Escolhe-se o menor `q` consistente** — consistência é requisito, RMSE é resultado. Registrar os dois.

**Teste de generalização de `Q`.** `Q` e `R` descrevem fenômenos independentes: o `q` sintonizado **não pode variar com `σ_gps`**. Se variar, é sintoma de que `q` está absorvendo erro de modelo correlacionado com a medida — quase sempre sincronia. Isso é uma asserção automatizável: sintonizar em `σ = 1` e verificar consistência em `σ = 10` sem retocar `q`.

**Por que a mobilidade importa aqui, e a decisão de escopo.** Com o modelo de Lamenza (§3: velocidade constante, mudança de direção instantânea) o modelo `x + u·dt` é **exato** e `Q` seria degenerado (zero, à parte a discretização). O filtro vira um mediador recursivo perfeito, e é provavelmente daí que vem parte dos 18,8× medidos. Com `DynamicVelocityMobilityHandler` o comando não é cumprido instantaneamente e o descasamento vira um `Q` **mensurável** — que é o cenário realista e o único em que a sintonia de `Q` é um exercício de verdade. Mas trocar o modelo de mobilidade **altera o cenário de Lamenza** e exigiria linha na tabela de desvios (PRD §5.4).

> **Decisão adotada:** a configuração **nominal** da Etapa 2 mantém a mobilidade sem massa de Lamenza — **zero desvios**. A inércia entra como **ablação do estimador**, reportada em separado, e só é promovida a configuração com decisão registrada e linha na §5.4. O relatório precisa declarar explicitamente que o número principal do KF é obtido sob um modelo de processo exato, e usar a ablação de inércia/vento como a curva de degradação honesta.

**Erro de modelo que existe mesmo sem inércia, e que ninguém espera.** O `DynamicVelocity`/`update_rate` do simulador integra a posição em passos de `update_rate`. Um comando emitido entre ticks só passa a valer no tick seguinte: o deslocamento erra em até `‖u_k − u_{k−1}‖ · update_rate ≤ 30 · 0,1 = 3 m` por decisão com `update_rate = 0,1` (o valor que a P2 do PRD quer adotar para viabilizar o treino). **A 3 m de erro de discretização, `σ_gps = 1 m` deixa de ser o termo dominante e a Etapa 2 mede o integrador, não o GPS.** Ação obrigatória: medir `Q̂` com `update_rate ∈ {0,01; 0,1}` e reportar a razão; se o resíduo em 0,1 dominar `σ = 1 m`, avaliar a Etapa 2 em `update_rate = 0,01` (mais caro, mas a Etapa 2 tem 1 UAV) ou declarar a limitação. Esta é uma dependência cruzada entre P2 e a Etapa 2 que não está no PRD.

### T-07.6 — Consistência `consistency.py` **[N]**

**NEES** (precisa da verdade; só existe em simulação):

```
ε_k = (p_k − x̂_k)ᵀ P_k⁻¹ (p_k − x̂_k)      ~ χ²(n_x),  n_x = 2
```

**NIS** (não precisa da verdade; é o que existiria num voo real):

```
ν_k = ỹ_kᵀ S_k⁻¹ ỹ_k                        ~ χ²(n_z),  n_z = 2
```

**Protocolo Monte Carlo.** Média sobre `M` corridas independentes no **mesmo índice de tempo `k`**, nunca média temporal como teste primário — passos consecutivos não são independentes e a média temporal tem graus de liberdade efetivos desconhecidos.

```
ε̄_k = (1/M) Σ_{m=1..M} ε_k^(m) ,     M·ε̄_k ~ χ²(M·n_x)
faixa de aceitação bilateral a (1−α):
   [ χ²_{M·n_x, α/2} / M ,  χ²_{M·n_x, 1−α/2} / M ]
```

| `M` | g.l. `M·n_x` | faixa a 95 % para `ε̄` (`n_x = 2`) |
|---|---|---|
| 50 | 100 | ≈ [1,48 ; 2,59] |
| **100** | **200** | **≈ [1,63 ; 2,41]** |

Adotado **`M = 100`, `α = 0,05`, `n_x = n_z = 2`**. Valores da tabela são indicativos e devem ser computados em tempo de execução com `scipy.stats.chi2.ppf`, nunca hard-coded. `M = 50` é o mínimo clássico e fica como fallback de linha de corte.

**Terceiro teste, e o mais informativo dos três: brancura da inovação.** Sob filtro correto, `ỹ_k` é branca. Estimar a autocorrelação normalizada `ρ(l)` da sequência de inovação por corrida e verificar `|ρ(l)| ≤ 1,96/sqrt(K)` para `l = 1..20`. Inovação correlacionada é a assinatura direta de (i) erro de GPS não branco (modo `gps_gauss_markov`) e (ii) erro de modelo não modelado (inércia, discretização) — as duas coisas que NEES médio consegue mascarar quando o viés é pequeno mas persistente. Sem este teste, o aparato de consistência é fraco.

**O que reportar (por canal, por `σ`, por degrau da escada):** curva `ε̄_k × k` com a faixa sombreada, idem para `ν̄_k`, o correlograma da inovação, e a fração de instantes fora da faixa.

**O que fazer quando falha.** O sinal do desvio é diagnóstico e a ordem de investigação não é negociável:

| Sintoma | Significado | Ação, nesta ordem |
|---|---|---|
| `ε̄` **acima** da faixa | filtro **otimista**: `P` menor que o erro real. Perigoso — a política confia demais, e na Etapa 2 esse `P` vai para a observação (degrau 5) | 1) sincronia (`t_meas` vs `t_state`); 2) unidades (m vs normalizado; `σ` vs `σ²`); 3) `Q` subdimensionado ou composto na taxa errada; 4) correlação temporal não modelada |
| `ε̄` **abaixo** da faixa | filtro **conservador**: `P` maior que o erro. Seguro, mas desperdiça informação e infla os canais de `P` na observação | `Q` superdimensionado; verificar se `Q` adaptativo não está saturando |
| `ν̄` fora com `ε̄` dentro | inconsistência no par medida/inovação | `R` errado no filtro (não deveria acontecer — ver T-07.5) ou medidas duplicadas/descartadas |
| `ρ(l)` fora com `ε̄` dentro | erro de modelo persistente e pequeno | promover o modelo (predição com inércia) antes de inflar `Q` |

**Regra de conduta (RF-05):** nenhum resultado de RL é olhado antes de NEES e NIS passarem, para **cada `σ` da varredura**, no canal próprio. Se a consistência só for atingível inflando `Q` acima do medido em T-07.5, isso é resultado e vai para o relatório como "fator de inflação declarado", com o valor. Nunca silenciosamente.

### T-07.7 — Potência do detector: desintonia injetada **[N]**

Um detector que nunca acusa não é evidência de nada. Injetar desintonias conhecidas e verificar que o aparato de T-07.6 as pega:

| Injeção | Efeito esperado |
|---|---|
| `Q ← Q/4` e `Q ← 4Q` | `ε̄` acima / abaixo da faixa |
| `R ← R/4` e `R ← 4R` | `ν̄` fora da faixa |
| `gps_gauss_markov`, `τ ∈ {30, 120, 300}` | `ρ(l)` fora; `ε̄` acima |
| medida com carimbo defasado de `dt_gps` | `ε̄` acima; viés visível na média da inovação |
| `sensor_stored` sem piso de variância (T-07.3) | `ε̄` do canal de sensores divergindo com o número de contatos |

**Saída:** tabela de detecção com a magnitude mínima de desintonia detectável em `M = 100`. Isso responde a pergunta que dá credibilidade ao portão G2: *quão errado o filtro precisaria estar para este teste não perceber?*

### T-07.8 — Invariante de vazamento RF-04 **[N]**

**O que se afirma:** o caminho `observação → política` não é função da posição verdadeira. A verdade é entregue pelo `handle_telemetry` do `gradysim`, é corrompida na entrada do protocolo, e a cópia exata fica guardada **só para métricas** — inclusive para o NEES, que precisa dela por definição.

Provar isso é menos trivial do que parece, porque o vazamento pode ocorrer por valor derivado, por conversão para escalar, por variável global ou por dentro do simulador. Quatro camadas, com papéis distintos:

**Camada 1 — ponto único de custódia (`TruthLedger`).** A verdade existe em um único objeto. O protocolo só escreve (`record(t, node_id, p)`). A leitura exige um token (`MetricsToken`) construível apenas dentro de `eval/metrics.py` e `fusion/consistency.py`; `get()` sem token levanta exceção. Torna o vazamento acidental impossível de escrever distraidamente e documenta, por construção, quais são os dois módulos legitimamente autorizados a ver a verdade.

**Camada 2 — tipos distintos, verificados estaticamente.** `TruePosition` e `MeasuredPosition` como dataclasses congeladas distintas, não tuplas nem `ndarray` crus. `ObservationBuilder` e `Policy.act` aceitam apenas `MeasuredPosition`. `mypy --strict` sobre `fusion/`, `env/protocols.py`, `encoding/` e `agents/` no CI. Custo baixo, pega a maioria dos erros na hora de escrever. Limite conhecido: não sobrevive à conversão para `ndarray`.

**Camada 3 — marcação em tempo de execução (*taint*).** Subclasse de `np.ndarray` com atributo `_truth = True` propagado por `__array_finalize__`, de modo que qualquer array derivado (fatia, soma, diferença) herde a marca. `Policy.act` afirma `not is_tainted(obs)`. Pega o caso do desenvolvedor que calcula `dx = p_true − x̂` e coloca na observação. Ativada por variável de ambiente (`UAVWNN_TAINT=1`) porque tem custo; obrigatória na suíte de testes, desligada nas corridas de medição de latência (senão contamina T2 da SPEC-03). **Limite conhecido e que precisa estar documentado:** a marca não sobrevive a `float(x)`, `.item()` nem `.tolist()`. Por isso ela não é a prova — é a rede de segurança.

**Camada 4 — teste diferencial, e esta é a prova.** Metamórfico, agnóstico à implementação:

1. Corrida A: executa um episódio com semente fixa, com um `NoiseRecorder` que grava **toda** saída da camada de corrupção e um hash da sequência de ações.
2. Corrida B: mesma semente, mas (i) a camada de corrupção é substituída por um `NoiseReplayer` que devolve exatamente os valores gravados em A, e (ii) o valor de verdade entregue pelos acessores (`handle_telemetry` e `TruthLedger.record`) é deslocado por um offset fixo `δ`, sorteado, **sem** tocar no estado interno do simulador — a física continua idêntica, só o número que trafega pelo caminho de telemetria muda.
3. Asserção: o hash da sequência de ações em B é **idêntico** ao de A.

Qualquer dependência da política em relação à verdade — por qualquer canal, incluindo escalares, globais e caminhos dentro do simulador — quebra a igualdade. Repetir com `n ≥ 20` pares `(semente, δ)`. É o único dos quatro testes que prova alguma coisa; os outros três existem para que ele quase nunca falhe.

> **Cuidado de escopo, e é um limite honesto do trabalho.** O deslocamento `δ` deve incidir **apenas** sobre os canais que a Etapa 2 declara corromper: posição própria e posição de sensor. O **status de visitação** continua oracular na Etapa 2 (PRD §6.2 lista dois canais, e só). Ou seja: o canal global de Lamenza é fechado **parcialmente**. O teste não deve perturbar a máscara de visitados, e o relatório precisa dizer que ela permanece perfeita — caso contrário estaríamos alegando ter resolvido uma limitação que resolvemos pela metade.

### T-07.9 — Custo do filtro (RNF-05) e integração com a SPEC-03 **[N]**

**Contagem exata**, por ciclo do KF de 2 estados, com `P` simétrica 2×2 (3 entradas independentes):

| Etapa | Operações | flops |
|---|---|---|
| `x̂⁻ = x̂ + dt·u` | 2 mult + 2 soma | 4 |
| `P⁻ = P + Q` | 3 soma | 3 |
| `S = P⁻ + R` | 3 soma | 3 |
| `S⁻¹` (2×2 fechada) | 2 mult + 1 sub + **1 divisão** + 3 mult | 7 |
| `K = P⁻ S⁻¹` | 8 mult + 4 soma | 12 |
| `ỹ = z − x̂⁻` | 2 sub | 2 |
| `x̂ = x̂⁻ + K ỹ` | 4 mult + 2 soma | 6 |
| `P` forma simples `(I−K)P⁻` | 8 mult + 4 soma | 12 |
| `P` **forma de Joseph** | — | ≈ 42 |
| **Total, forma simples** | | **≈ 49** |
| **Total, forma de Joseph** | | **≈ 79** |

Isto **deriva** a faixa de 50–80 flops/passo afirmada no PRD (RNF-05), em vez de citá-la. Comparação:

- MLP pareada 31→32→32→2 (PRD §5.5): `31·32 + 32·32 + 32·2 = 2 080` MACs por inferência.
- Um ciclo de KF por decisão: **2,4 % (simples) a 3,8 % (Joseph)**.

> **Discrepância a corrigir no PRD.** O PRD §6.2 diz "~2600 MACs de uma MLP 31→32→2". Essa arquitetura dá 1 056 MACs; os ~2 600 vêm da MLP 33→64→8 do PLANO §9.8. A MLP pareada declarada na §5.5 é 31→32→32→2 = 2 080. Adotar 2 080 aqui e corrigir a §6.2.

**Três ressalvas de honestidade, todas obrigatórias no relatório:**

1. **A taxa muda o número.** Se o filtro rodasse a cada tick de mobilidade (0,1 s), seriam 20 predições + 1–2 correções por decisão ≈ 220 flops, e a fração sobe para ~11 %. A regra de taxa da T-07.2 (um `predict` por evento) não é só corretude de `Q` — é o que preserva o argumento de custo.
2. **O KF é termo comum.** A MLP pareada precisa exatamente do mesmo filtro. Na comparação de H1, o custo do KF **cancela**, e deve aparecer como linha separada da tabela, nunca embutido no custo da WNN. Embuti-lo seria inflar o resultado na direção errada e, pior, na direção que um revisor nota.
3. **Divisão em ponto flutuante.** O KF usa 1 divisão por ciclo. Num microcontrolador sem FPU, divisão custa dezenas de ciclos — mais que a soma dos MACs. A alegação defensável não é "sem MACs", é: **o custo aritmético do filtro é uma constante pequena e fixa, independente da capacidade do modelo, enquanto o da MLP cresce com a rede.**

**Ablação de ganho estacionário `[C]`.** Para `dt`, `Q` e `R` fixos, a recursão de Riccati converge e `K → K∞` (forma fechada em T-07.2). O filtro degenera em `x̂ = x̂⁻ + K∞(z − x̂⁻)`: **~10 flops, sem matriz, sem divisão, implementável em inteiro**. Isso preserva a narrativa de embarcabilidade quase intacta. Medir: flops, erro contra o KF completo, e NEES — a consistência do regime permanente vale após a convergência, e o transiente inicial precisa ser reportado, não escondido.

**Integração com a SPEC-03:** registrar `filtragem` como estágio novo na decomposição de T2 (junto de montagem da observação, codificação, endereçamento) e alimentar T3 com a contagem por tipo, **incluindo divisões como categoria própria** — hoje a SPEC-03 conta MACs, lookups e somas, e divisão não cabe em nenhuma das três.

### T-07.10 — Degraus 2 e 3 da escada, sintonizados de verdade **[N]**

A escada só tem valor se os degraus intermediários forem adversários honestos. Ambos são implementados e sintonizados **aqui**, para que a SPEC-08 só rode experimento.

**Degrau 2 — passa-baixa / média móvel.** `x̂_k = (1−α)x̂_{k−1} + α z_k`. Ponto que precisa estar no relatório: para um veículo em movimento este filtro é **enviesado**, não apenas menos eficiente. Rastreando uma rampa de velocidade `v`, o atraso em regime permanente é

```
atraso = v · dt · (1 − α)/α        [m]
```

Com `v = 15 m/s`, `dt = 2 s`, `α = 0,5` → **30 m de atraso**, uma ordem de grandeza acima de `σ_gps`. Varrer `α ∈ {0,1; 0,25; 0,5; 0,75; 1,0}` e reportar o ótimo do compromisso viés×variância por `σ`. É essa análise que explica *por que* o KF com controle ganha: ele não é um passa-baixa melhor, ele é **não enviesado**, porque conhece o movimento.

**Degrau 3 — KF de velocidade constante (4 estados).** `x = [x, y, vx, vy]ᵀ`, `F = [[I, dt·I],[0, I]]`, `H = [I, 0]`, `R = σ²I₂`, e `Q` de aceleração branca:

```
Q = q_a · [[ dt³/3·I ,  dt²/2·I ],
           [ dt²/2·I ,  dt·I    ]]
```

Varrer `q_a ∈ {0,5; 2; 5; 20; 50; 200}` como no experimento preliminar (PLANO §9.4), escolher pelo mesmo critério de T-07.5, e **reportar NEES do degrau 3 também**: a previsão é que ele seja inconsistente sob comando i.i.d. (B1 com `Δθ ~ U[−π,π]`) para todo `q_a`, porque nenhum `Q` de aceleração branca representa um salto de rumo determinístico. Se isso se confirmar, é o resultado que transforma "o KF ingênuo é pior" de anedota em diagnóstico: **ele não está mal sintonizado, ele está mal especificado.** Vale uma figura.

### T-07.11 — Erro ao longo do tempo e a regra *bin ≈ σ* instanciada **[N]**

**O erro médio do episódio esconde o que importa.** Com `σ = 10 m` e o `q` que reproduz `0,67 m` de erro médio, o ganho estacionário fica em `K∞ ≈ 0,005`, ou seja o filtro está efetivamente promediando ~200 passos — praticamente o episódio inteiro (`K = 250`). Nos primeiros passos o erro é próximo de `σ/sqrt(k)`: **4,5 m em `k = 5`**, quase sete vezes a média reportada. E é no início do episódio que a política decide para onde ir. Exigências:

1. Reportar `erro(k)` e `sqrt(tr P_k)` como **curvas**, com o limite ideal `σ/sqrt(k)` sobreposto; a média do episódio é um número secundário.
2. Reportar o instante `k*` em que o erro filtrado cruza a largura de bin do termômetro escolhido — antes de `k*`, o filtro não está entregando resolução nenhuma ao codificador.

**Instanciação da regra.** Termômetro linear global sobre `[-200, 200]` (SPEC-01, E0): largura de bin `= 400/T` → `T = 8` → 50 m; `T = 32` → 12,5 m; `T = 128` → **3,1 m**. Com `σ_rel` filtrado da ordem de 1–5 m (T-07.4), `T = 128` está no limite e a regra sugere ir **além** de 128 na condição filtrada — o que a varredura da SPEC-08 precisa acomodar, e que a SPEC-01 não previu porque parou em `T = 128`. Sob binning por slot (E1/E2), o span dos slots próximos é muito menor e a resolução efetiva é maior; a tabela final `T*(σ_gps, σ_prior, esquema)` é entregável desta tarefa e é a entrada direta do experimento mais caro do projeto.

### T-07.12 — Integração com o `gradysim` **[C se a SPEC-00 atrasar]**

Ponto de corrupção na entrada do protocolo (`UAVProtocol.handle_telemetry`), `TruthLedger.record` em paralelo, `KalmanFilter.predict_to(t)/update(z, t)` no `handle_timer("decide")` antes de montar a observação, e `u` capturado da ação imediatamente após `policy.act` para o passo seguinte. Rodar V-7.1 (KS sobre `e_k`) e re-sintonizar `Q` sobre B3 se divergir. Enquanto a SPEC-00 não entregar, tudo acima roda em B1/B2 e nada fica bloqueado — que é a razão de esta spec existir em paralelo.

---

## 6. Critério de saída

Medido, não implementado.

1. **NEES e NIS do canal próprio dentro da faixa qui-quadrada a 95 % (`M = 100`, `n = 2`)** para todo `σ_gps ∈ {1, 3, 5, 10}`, em B1 e B2, com a curva por instante e não só a média. Fração de instantes fora da faixa reportada.
2. **Brancura da inovação** verificada (`|ρ(l)| ≤ 1,96/sqrt(K)`, `l = 1..20`) na configuração nominal; violações no modo `gps_gauss_markov` reportadas como esperado.
3. **Tabela de potência do detector** (T-07.7) completa: toda desintonia injetada é acusada, com a magnitude mínima detectável registrada.
4. **Reprodução do achado A3** na bancada: `σ_gps = 10 m` → erro filtrado com KF+controle materialmente menor que medida crua e que KF-VC melhor sintonizado, com IC bootstrap. O número não precisa ser 18,8× (`dt_dec` e `dt_gps` mudaram); precisa ser **medido e explicado**.
5. **Degraus 2 e 3 sintonizados** pelo mesmo protocolo da proposta, com `α*` e `q_a*` registrados, e o NEES do degrau 3 reportado.
6. **Filtro de informação** validado contra a forma fechada `P_k = σ²/k` e com o piso de variância demonstrado sob `sensor_stored`.
7. **RF-04:** as quatro camadas implementadas e o teste diferencial (camada 4) verde em `n ≥ 20` pares `(semente, δ)`. Escopo do invariante documentado, incluindo que a máscara de visitados permanece oracular.
8. **Custo:** tabela de flops por tipo (mult, soma, divisão) para KF completo, KF de ganho estacionário e filtro de informação; fração contra os 2 080 MACs da MLP pareada; estágio `filtragem` registrado no harness da SPEC-03.
9. **Tabela `T*(σ)`** entregue à SPEC-08, com `σ_rel` filtrado e não filtrado para a grade `σ_gps × σ_prior`.
10. **Relatório curto** (3–4 páginas com figuras) pronto para o capítulo de extensão de E2 — capítulo separado, extensão declarada (PRD §0).
11. **Testes automatizados:** `tests/test_noise.py`, `tests/test_kalman.py`, `tests/test_consistency.py`, `tests/test_leak_invariant.py`. Os de consistência marcados como lentos, mas dentro de `make repro`.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **Erro de sincronia entre instante da medida e instante da predição.** Viés de `‖u‖·Δt` (3 m a 15 m/s e 0,2 s), sistemático, invisível na variância | carimbo de tempo obrigatório em toda medida; `predict_to(t)` como única forma de avançar; exceção em retrocesso; teste de regressão com defasagem injetada; NEES como detector (T-07.7) |
| **`Q` composto na taxa errada.** O termo `dt²` não é invariante à taxa; um `Q` sintonizado a 0,1 s aplicado a 2 s erra por fator 20 | decomposição explícita `q_w·dt + σ_v²·dt²`; taxa do filtro gravada na configuração; teste que compara 1 passo de 2 s contra 20 de 0,1 s e exige diferença apenas em `P`, jamais em `x̂` |
| **Confusão de unidades**: `σ` vs `σ²`, metro vs normalizado, dupla normalização | metro como unidade canônica em todo o pacote `fusion/`; asserção de faixa na entrada do codificador; nenhum `σ` normalizado em código |
| **Erro de GPS real não é branco.** O KF assume branco, fica otimista, e o `P` otimista vai para a observação no degrau 5 | modo `gps_gauss_markov` como ablação obrigatória; teste de brancura da inovação como detector primário; se falhar, o caminho correto é aumentar o estado (viés) — registrado como trabalho futuro, não aplicado |
| **`Q` degenerado sob mobilidade sem massa**: o modelo é exato e o resultado fica bom demais por construção | declarar isso no relatório; ablação com inércia e vento como curva de degradação honesta; `Q̂` medido em vez de buscado |
| **`update_rate = 0,1` da P2 injeta até 3 m de erro de discretização**, dominando `σ_gps = 1 m` | medir `Q̂` nas duas taxas e reportar; avaliar a Etapa 2 em 0,01 (viável com 1 UAV) ou declarar a limitação; escalar para a P2 |
| **O filtro de informação não tem o que fundir na Etapa 2**, porque ouvir o sensor equivale a visitá-lo | declarado em T-07.1b; a peça é validada por forma fechada e o valor real fica na SPEC-09; `detection_range > communication_range` só com decisão registrada e linha na §5.4 |
| **`σ_rel` limitado pelo canal do sensor**: filtrar o GPS pode render pouco na observação | T-07.4 entrega a tabela conjunta; o relatório reporta ganho por canal **e** ganho na observação, sem escolher o conveniente |
| **Teste de vazamento que não testa nada** (marca que não propaga, tipo que some no `ndarray`) | quatro camadas com papéis distintos; a camada 4 é diferencial e agnóstica à implementação; limites de cada camada documentados no próprio teste |
| **Média do episódio esconde o transiente**, que é onde a política decide | T-07.11: erro e `sqrt(tr P)` como curvas, com `σ/sqrt(k)` sobreposto e o cruzamento `k*` reportado |
| **Estimativa de 3 d não cobre 12 tarefas.** É a spec mais densa do conjunto | linha de corte própria (abaixo); as tarefas **[C]** saem primeiro; T-07.12 pode escorregar sem bloquear ninguém, já que B1/B2 não dependem da SPEC-00 |
| Sintonia em malha aberta não vale em malha fechada | reverificação de NEES em malha fechada é item de aceite da SPEC-08, registrado aqui |

**Linha de corte desta spec, em ordem:** (1) ablação de ganho estacionário; (2) modo `gps_gauss_markov` reduzido a um único `τ`; (3) `M = 100 → 50` nas corridas Monte Carlo; (4) B1 reduzida a dois regimes de `Δθ`; (5) T-07.12 adiada para dentro da SPEC-08. **Não podem cair:** NEES/NIS (RF-05), o teste diferencial de RF-04, os degraus 2 e 3 sintonizados, a tabela de custo.

---

## 8. Entregáveis

- `src/uavwnn/fusion/noise.py` — GPS (branco e Gauss–Markov), prior de levantamento e relato do sensor, atraso e perda de broadcast; `NoiseConfig` serializável; semeadura hierárquica com CRN
- `src/uavwnn/fusion/kalman.py` — KF 2 estados com entrada de controle e forma de Joseph, KF-VC de 4 estados (degrau 3), passa-baixa (degrau 2), variante de ganho estacionário, filtro de informação com piso de variância, fila de medidas por carimbo de tempo
- `src/uavwnn/fusion/consistency.py` — NEES, NIS, faixas qui-quadrado por `scipy`, teste de brancura da inovação, relatório de diagnóstico com a tabela de sintomas
- `src/uavwnn/env/truth.py` — `TruthLedger`, `MetricsToken`, tipos `TruePosition`/`MeasuredPosition`, utilitários de *taint*
- `src/uavwnn/fusion/bench.py` — bancadas B1 e B2 (B2 reusa `env/analytic.py` da SPEC-01)
- `configs/spec07/` — grades de `σ`, `q`, `α`, `q_a`, `τ`
- `reports/spec07-fusao-local.md` + figuras: NEES/NIS por `σ`; correlograma da inovação; `erro(k)` com `σ/sqrt(k)`; NEES × `q` cruzando a faixa; viés do passa-baixa × `α`; inconsistência do KF-VC; tabela de custo; tabela `T*(σ)`
- `tests/test_noise.py`, `tests/test_kalman.py`, `tests/test_consistency.py`, `tests/test_leak_invariant.py`
- **Atualizações de documento:** correção dos ~2 600 MACs no PRD §6.2; registro da decisão de mobilidade em §10; nota em §10/P3 fixando o caso linear

---

## 9. Perguntas em aberto

1. **Qual é a taxa do GPS?** É parâmetro da nossa extensão, não existe em Lamenza, e **determina o número principal**. Com `dt_dec = 2 s` e 1 Hz há 2 correções por decisão; o experimento preliminar usou decisão a 0,5 s, e portanto quatro vezes mais correções por segundo de voo. A reprodução dos 18,8× depende disso. Proposta: `dt_gps = 1 s` nominal, `{0,2; 1; 5} s` como varredura secundária. **Precisa de aprovação, porque escolher a taxa é escolher o resultado.**
2. **`detection_range` separado de `communication_range`?** Sem isso, o canal de sensores da Etapa 2 é um prior estático e o filtro de informação não filtra nada (T-07.1b). Com isso, o cenário muda e é linha nova na tabela de desvios (PRD §5.4). Terceira opção: aceitar o canal degenerado na Etapa 2 e exercitar o filtro de informação apenas na SPEC-09. Recomendação: a terceira, mas é decisão do autor.
3. **A máscara de visitação continua oracular.** A Etapa 2 corrompe só posições, então o canal global de Lamenza é fechado pela metade. Isso deve ser declarado como limitação no relatório, ou vale corromper também o status de visitação (por exemplo, um UAV só sabe o que ele mesmo coletou)? A segunda opção é bem mais próxima da limitação que Lamenza aponta — e é uma mudança grande, provavelmente Etapa 3.
4. **`σ_prior` do levantamento dos sensores: qual valor, e é filtrável?** Se for da ordem de `σ_gps`, ele domina `σ_rel` e limita o ganho da fusão na observação (T-07.4). Se for muito menor, o canal deixa de ser interessante. Não há valor herdado de Lamenza — é escolha nossa e precisa de justificativa.
5. **Mobilidade com inércia: nominal ou ablação?** Adotado aqui como ablação, para preservar zero desvios. Mas é ela que torna `Q` um objeto medido em vez de um resíduo de discretização, e é o argumento sim-to-real mais forte disponível. Se for promovida a nominal, exige decisão registrada e linha na §5.4.
6. **A política conhece `u` ou `ū`?** O filtro usa a velocidade **comandada**. Se o UAV tem IMU, ele conhece a velocidade **realizada**, o que é informação melhor e muda a natureza do trabalho (passa a haver um segundo sensor a fundir). Manter só o comando é a hipótese conservadora e é a que reproduz o achado A3; declarar a IMU como indisponível é uma suposição de modelagem que o relatório precisa assumir explicitamente.
7. **O que fazer se a consistência só for atingível com `Q` inflado acima do medido?** Proposta: reportar o fator de inflação como número declarado, e investigar antes se a causa é sincronia. Mas isso precisa de um limite acordado — a partir de qual fator a resposta certa passa a ser "o modelo está errado, aumente o estado"?
8. **`M = 100` corridas por célula da grade `σ × T` cabe no orçamento?** A varredura da SPEC-08 já é o experimento mais caro do projeto; multiplicá-la por 100 corridas de consistência pode não fechar. Alternativa: consistência validada em bancada (B1/B2, barata) e apenas amostrada em malha fechada.
9. **A verificação de `P` na observação (degrau 5) usa `sqrt(tr P)` ou os autovalores?** O PLANO §9.6 propõe `sqrt(tr P)`, que é escalar e barato, mas descarta a anisotropia — que existe assim que o ruído deixa de ser isotrópico ou o atraso entra (SPEC-09). Decidir antes de a SPEC-08 codificar os canais.
