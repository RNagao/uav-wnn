# SPEC-08 — Etapa 2: escada de ablação 0–5 e varredura conjunta σ × T

**Etapa:** 2 · **Depende de:** SPEC-06 (políticas e números da Etapa 1), SPEC-07 (ruído, KF, NEES/NIS), SPEC-03 (harness de custo) · **Estimativa:** 3 dias
**Portão de saída:** `G2` · **Referência:** PRD §1 (achados A2/A3 e a regra *bin ≈ σ*), §2 (H2 e H3 — a **H4 saiu** com D13), §5.4 (desvios), §5.5 (custo), §6.2 (Etapa 2), §6.4 (métricas)

---

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

## 0. ~~Revisão — a Etapa 2 roda em dois blocos~~ — REVOGADO por D13

*A Etapa 2 roda em **um** bloco: GPS do drone, KF com entrada de controle, varredura `σ_own × T` = 12 configurações, portão G2. A tabela e a discussão de 2a/2b abaixo ficam como registro.*

### A escada, num canal só

| # | Configuração | Papel |
|---|---|---|
| 0 | sem ruído — a política da Etapa 1 | teto |
| 1 | ruído bruto, sem filtro | piso |
| 2 | média móvel / passa-baixa | filtro barato que o KF precisa bater |
| 3 | **KF de velocidade constante** | o "óbvio" que já sabemos que falha |
| 4 | **KF + entrada de controle** | a proposta — onde vive o achado A3 |
| 5 | 4 + covariância própria na observação | ablação, **sem hipótese forte** |

A ressalva antiga — de que os degraus 2 e 3 mudavam de sentido no bloco de alvo
estático — deixa de se aplicar: só há alvo móvel, que é o próprio drone, e o
degrau 3 mantém o sentido literal de espantalho.

**O degrau 5 perdeu a hipótese, não a razão de existir.** A H4 previa *busca
ativa*: com `R(d)` crescente, aproximar-se compraria informação. Sem o canal do
sensor a covariância própria converge para regime estacionário e não há incerteza
*por sensor* a reduzir. O que sobra é a pergunta mais modesta — **a política usa
os canais de `P` para algo?** — e ela continua respondível pelos testes de
intervenção da T-08.9, que medem *uso* e não *busca*.

**Custo:** ~5 h de um núcleo, 2 M passos por corrida com 3 sementes na célula
base `Na=1, Ns=12` (0,0174 s/ep, medido na SPEC-00).

<details>
<summary><b>Registro histórico:</b> por que havia dois blocos (revogado por D13)</summary>

Os dois canais tinham estruturas **opostas** — o do drone é deslocamento comum a
todas as entradas da observação, o do sensor seria por slot —, e ligá-los juntos
tornaria qualquer degradação inatribuível. Daí a separação em 2a/2b, com o 2b
bloqueado pela TS3 (a física do canal do sensor: GPS próprio do sensor, dominado
por viés fixo, ou localização por rádio, com erro branco crescente com a
distância). Com um canal só, não há o que atribuir e a TS3 não tem consequência.

</details>

## 1. Objetivo

Produzir os **números e os vereditos** de H2 e H3 na célula `Na = 1, Ns = 12` — a **H4 saiu com D13** e o degrau 5 fica como ablação sem hipótese —, executando a escada de ablação 0–5 nos dois regimes de treino e a varredura **conjunta** `σ_GPS × T`, e fechar o portão `G2`.

Esta é uma spec de **experimento**. Ela não escreve camada de ruído, filtro nem verificador de consistência — isso é SPEC-07 — nem harness de custo — isso é SPEC-03. Ela **consome** esses artefatos, orquestra rodadas, aplica um protocolo estatístico declarado antes das rodadas, e escreve vereditos.

## 2. Por que

Três razões, nesta ordem.

**Primeira: sem a escada não se sabe qual componente pagou.** "Filtrei e melhorou" é compatível com "qualquer suavização melhoraria". O degrau 2 (média móvel) existe para tornar essa alternativa falsificável, e o degrau 3 (KF de velocidade constante) existe porque já foi **medido** que ele piora (PLANO §9.4: 5,06 m contra 12,54 m da medida crua com σ=10 m, contra 0,67 m do KF com controle) — o que transforma "usei Kalman" em uma afirmação com conteúdo, já que a variante óbvia de Kalman é justamente a que falha.

**Segunda: filtro e codificador são acoplados, e por isso a varredura tem de ser conjunta.** PLANO §9.3 mede que, para σ ≥ 0,05 normalizado, `T = 8`, `32` e `128` **colapsam no mesmo erro** — bits além de `largura_do_bin ≈ σ` codificam ruído. A consequência é a tese central da Etapa 2: *o KF não melhora a política diretamente; ele derruba σ e com isso habilita um termômetro mais fino, que é o que recupera a acurácia*. Varrer σ com `T` fixo mediria a metade errada do efeito; varrer `T` com σ fixo mediria a outra metade errada. Só a grade cruzada mostra a interação, que é o objeto.

**Terceira: o achado A2 remove a objeção mais óbvia à proposta** (a WNN não é mais frágil a ruído que a MLP), mas ele foi medido em regressão sintética, fora do ambiente e fora do laço de RL. H2 exige que isso seja replicado **na tarefa real**, com política treinada, ou o achado não sustenta o relatório.

E há uma razão de higiene, que é o motivo de a §4 existir: um KF mal sintonizado produz números plausíveis. Já aconteceu uma vez neste projeto (PLANO §9.10). Qualquer conclusão de RL tirada antes de NEES/NIS passar é, na melhor das hipóteses, sorte.

## 3. Escopo

**Dentro:** definição operacional dos degraus 0–5; os dois regimes de treino e o teste de equivalência entre eles; a grade `σ_GPS ∈ {1, 3, 5, 10} m × T ∈ {8, 32, 128}`; a bateria de busca ativa de H4; o recálculo de T1/T2/T3 com o filtro embutido; vereditos de H2, H3 e H4; portão G2.

**Fora:** implementação de `fusion/noise.py`, `fusion/kalman.py`, `fusion/consistency.py` (SPEC-07); harness de custo (SPEC-03); qualquer coisa multiagente, inclusive o degrau 6 e a interseção de covariância (SPEC-09); `Ns ∈ {24, 36}`; canal de vizinhos, atraso de broadcast e correlação entre estimativas — não existem com `Na = 1`.

**Herdado sem renegociação:** o MDP da §5.2 do PRD verbatim, o codificador escolhido em G0, `(n, N)` fixados em G1, os 500 episódios de avaliação e o IC bootstrap de 1000 reamostragens da §6.4.

---

## 4. Fluxo de execução e o bloqueio de RF-05

A ordem aqui **não é sugestão de conveniência**. RF-05 do PRD exige NEES/NIS aprovado antes de qualquer conclusão de RL, e este é o ponto do projeto em que essa exigência tem de virar mecanismo, não intenção.

```
SPEC-07 entrega filtros
        │
        ▼
 [B1] atestado de consistência  ── reprova ──►  volta a SPEC-07. NÃO é resultado de G2.
 NEES/NIS por canal e por degrau                Nenhuma rodada de RL desta spec executa.
        │ aprova
        ▼
 [B2] invariante de vazamento (RF-04) ── reprova ──►  bloqueia tudo. Qualquer número
      teste automatizado verde                        produzido antes é descartado.
        │ aprova
        ▼
 [B3] plano de análise congelado (hash registrado)
        │
        ▼
 T-08.5 escada na âncora ─► T-08.6 regimes ─► T-08.7 grade σ×T ─► T-08.8 H2 ─► T-08.9 H4
        │
        ▼
 T-08.10 custo com filtro ─► T-08.11 vereditos ─► G2
```

> **Bloqueio B1 (RF-05).** A spec começa com `results/etapa2/gate_rf05.json` **ausente**. O corredor de execução (`make etapa2`) recusa-se a iniciar qualquer treino ou avaliação enquanto o arquivo não existir com `status: aprovado`, e o atestado carrega o hash da configuração de filtro que foi validada. Se a configuração mudar, o atestado é invalidado automaticamente e o bloqueio volta. Isto é mais forte que "rodar NEES/NIS antes": impede fisicamente a inversão de ordem, inclusive por descuido às três da manhã da semana 3.

> **Bloqueio B2 (RF-04).** Idem para o teste de não vazamento. Vale para os **cinco** degraus com ruído: o degrau 2 é o mais fácil de vazar por acidente, porque uma média móvel mal escrita sobre o buffer errado é uma média móvel sobre a verdade.

> **Bloqueio B3 (pré-registro).** O plano de análise — métricas primárias, margens de equivalência, sinais esperados, família de testes e correção de multiplicidade — é escrito e **hasheado antes** da primeira rodada. Não é burocracia: esta spec tem 12 células, 6 degraus, 2 regimes e uma bateria comportamental com meia dúzia de estatísticas. Sem pré-registro, sempre existe uma leitura que confirma H4, e ela não vale nada.

---

## 5. Tarefas

### T-08.1 — Atestado de consistência e de não vazamento

Consome `fusion/consistency.py`. Produz `results/etapa2/gate_rf05.json`.

Conteúdo mínimo do atestado, **por canal** (posição própria; posição de sensores) e **por degrau** (2, 3, 4, 5):

| Item | Critério |
|---|---|
| NEES médio sobre `K` passos | dentro do intervalo qui-quadrado de 95 % para `K·dim` graus de liberdade |
| NIS médio | idem, com a dimensão da inovação |
| Fração de passos fora do envelope individual | ≤ 5 %, sem estrutura temporal (teste de corridas) |
| Autocorrelação da inovação em lag 1 | dentro do IC de ruído branco |
| Hash da configuração de filtro (`Q`, `R`, `dt`, modelo de atuação) | registrado |

Dois pontos que costumam ser esquecidos e que aqui são obrigatórios:

1. **O degrau 3 tem de ser avaliado com o `q` varrido**, não com um `q` qualquer. PLANO §9.4 varreu `q ∈ {0,5; 2; 5; 20; 50; 200}` justamente para que o KF de velocidade constante não fosse espantalho. Repetir a varredura **dentro do ambiente** e reportar o melhor `q`. Se o degrau 3 for testado com um `q` ruim, todo o argumento a favor do degrau 4 perde valor.
2. **NEES do degrau 3 pode reprovar legitimamente** — um modelo mal especificado é inconsistente por construção. Isso não bloqueia a spec: registra-se como "inconsistente por descasamento de modelo, conforme esperado", e é parte do resultado. O que bloqueia é o **degrau 4** reprovar.

**Consistência sob a mobilidade real.** `Q` do KF com controle depende de o comando ser cumprido. Se a Etapa 2 usa mobilidade com inércia (PRD §9, risco), o descasamento entre comando e deslocamento é o que `Q` tem de representar; medir esse descasamento empiricamente e ajustar `Q` **antes** do atestado, não depois de olhar os resultados de RL.

### T-08.2 — Protocolo experimental

Este é o contrato estatístico da spec inteira.

**Orçamento de treino.** Todas as configurações treinam com o **mesmo** número de passos de decisão `S₂`, fixado como o joelho da curva de aprendizado medida na Etapa 1 (SPEC-06), com valor de trabalho `S₂ = 2·10⁶`. Comparação a orçamento igual é o que torna a escada interpretável; o desempenho absoluto pode ficar abaixo da assíntota da Etapa 1 e isso é aceito, porque todo veredito desta spec é **relativo entre degraus**.

**Sementes.** Três famílias, independentes e nunca reutilizadas entre si:

| Família | Papel | Quantidade |
|---|---|---|
| `seed_train` | inicialização das LUTs, mapeamento das tuplas, amostragem da gaussiana no treino | 5 por configuração no experimento principal; 3 nas células da varredura |
| `seed_eval` | posições de sensores e posição inicial do UAV nos 500 episódios de avaliação | fixa e **compartilhada por toda a spec e com a SPEC-06** |
| `seed_noise` | realizações de ruído | fixa, ver abaixo |

Um veredito escrito só pode se apoiar em configurações com ≥ 5 sementes de treino. Células de 3 sementes servem para desenhar a superfície da varredura, não para declarar hipótese.

**Pareamento por números aleatórios comuns (CRN).** É o que dá poder ao experimento e o que faz caber no orçamento.

- O episódio de avaliação `e` tem cenário determinado por `seed_eval[e]`, **idêntico em todos os degraus, todos os σ, todos os T e ambos os regimes**. O degrau 0 desta spec é literalmente reavaliado nos mesmos episódios da Etapa 1.
- O ruído de GPS é pré-sorteado como uma sequência de **normais padronizadas** `z_{e,t} ~ N(0, I₂)` indexada por `(e, t)`, e a medida é `p̃_t = p_t + σ·z_{e,t}`. Consequência: as células de σ diferente compartilham a mesma realização, e a curva em σ fica suave em vez de ruidosa. Os degraus 1–5 compartilham exatamente o mesmo `z`, portanto a diferença entre eles é atribuível ao estimador e a nada mais.
- O canal de sensores tem número de atualizações dependente da trajetória; o CRN é preservado indexando o sorteio por `(id_sensor, contador_de_contatos)` em vez de por tempo.
- O degrau 0 **consome** o fluxo de ruído sem usá-lo, para que a sequência de decisões não se desalinhe por um sorteio a menos.

**Métrica primária.** Tempo de conclusão **com censura em `max_episode_length = 500 s`**: episódios não concluídos entram com 500 s, não são descartados. Comparar médias só sobre episódios bem-sucedidos é enviesado quando as taxas de sucesso diferem entre degraus — e elas vão diferir, porque é exatamente isso que o ruído faz. Taxa de sucesso é métrica secundária **obrigatória**, reportada ao lado, sempre.

**Intervalos de confiança.** Bootstrap hierárquico de 1000 réplicas: reamostrar com reposição as `seed_train` (nível 1) e, dentro de cada uma, os 500 episódios (nível 2), reaplicando o **mesmo** conjunto de índices de episódio aos dois braços da comparação para preservar o pareamento. Reportar sempre a **diferença pareada** com IC, não dois ICs sobrepostos — dois ICs que se tocam não é um teste. Reportar também o IC condicional a uma semente (só episódios), identificado como tal, porque a diferença entre os dois ICs quantifica quanto da variabilidade é do treino e é informação útil.

**Testes de superioridade** (degrau 4 vs degraus 2 e 3): unilateral, via IC bootstrap da diferença pareada excluindo zero. Família corrigida por Holm.

**Testes de equivalência** (degrau 4 vs degrau 0; regime limpo vs ruidoso): **TOST**, nunca "p > 0,05". A margem é declarada agora:

```
δ = 0,10 · ( t̄₁ − t̄₀ )        (10 % do dano que o ruído causa sem filtro, na mesma célula)
```

Autocalibrada de propósito: a afirmação vira "a lacuna residual do degrau 4 é no máximo um décimo do estrago que o ruído faz". Se `t̄₁ − t̄₀` for pequeno na célula (σ = 1 m, tipicamente), o próprio experimento não tem o que mostrar ali e a célula é reportada como **não informativa**, não como equivalência trivialmente satisfeita. Reportar sempre a meia-largura obtida ao lado de δ: equivalência declarada com meia-largura > δ é falta de poder disfarçada de resultado.

### T-08.3 — Definição operacional dos degraus 0–5

O que exatamente muda em cada degrau, sem ambiguidade. Tudo o mais é idêntico.

| # | Degrau | Estimador que alimenta o codificador | Parâmetros a fixar/varrer | Papel |
|---|---|---|---|---|
| **0** | sem ruído | estado verdadeiro (canal global de Lamenza) | — | teto; é a política da Etapa 1, reavaliada nos mesmos episódios |
| **1** | ruído bruto | `p̃_t = p_t + σ z_t`, sem memória | — | piso |
| **2** | passa-baixa | média móvel exponencial `x̂_t = α p̃_t + (1−α) (x̂_{t−1} + u_{t−1} dt)` e a variante sem termo de comando | `α` varrido em `{0,1; 0,3; 0,5; 0,7; 1,0}`, melhor `α` por σ | filtro barato que o KF precisa bater |
| **3** | KF velocidade constante | 4 estados `[x, y, vx, vy]`, sem entrada de controle | `q` varrido em `{0,5; 2; 5; 20; 50; 200}`, melhor `q` por σ | o "óbvio" que já sabemos que falha |
| **4** | **KF + controle** | 2 estados `[x, y]`, predição `x̂ + u·dt` com `u` = ação da política no passo anterior; sensores por filtro de informação | `Q` do descasamento de atuação, medido em T-08.1 | a proposta |
| **5** | 4 + covariância na observação | idem 4, **mais** canais de incerteza na entrada (T-08.4) | `T_P`, conjunto de canais | política consciente da incerteza |

Três decisões que precisam ficar escritas:

- **O degrau 2 leva duas variantes**, com e sem o termo de comando na predição. A variante *com* comando é a ablação honesta: isola "usar o comando" de "usar Kalman". Se o passa-baixa com comando já capturar a maior parte do ganho, o resultado do projeto muda de "Kalman funciona" para "conhecer o comando funciona, e Kalman é a forma principiada de usá-lo" — que é uma frase diferente e mais verdadeira. Custa quase nada medir.
- **O degrau 4 usa a ação efetivamente executada no passo anterior**, não a ação desejada, quando as duas diferirem por saturação de velocidade. Registrar qual.
- **O degrau 5 não existe no regime de treino limpo.** Uma política treinada sem ruído nunca viu canais de `P` não degenerados; alimentá-los na avaliação é entrada fora de distribuição, e treiná-los com `P ≡ 0` produz canais constantes que o mapeamento de tuplas simplesmente ignora. A célula é marcada **N/A** e a comparação 5 vs 4 acontece inteira dentro do regime ruidoso. Simetricamente, "degrau 0 com treino ruidoso" é diagnóstico, não célula do desenho.

### T-08.4 — Degrau 5: canais de incerteza

**O que entra na observação.** Anexado ao padrão binário da eq. 5-1, nunca substituindo nada:

| Canal | Definição | Quantidade |
|---|---|---|
| `u_self` | `sqrt(tr(P_self))` — incerteza posicional própria, em metros | 1 |
| `u_j` | `sqrt(tr(P^s_j))` — incerteza da estimativa de posição do sensor no slot `j` de `P^s` | 12 |

**Por slot, não por setor.** O PRD fixou em D4 a observação por slots ordenados por distância; um mapa setorial é justamente a ideia registrada na §12 como trabalho futuro por modificar o MDP. Canais por setor exigiriam esse mapa. Por slot, os canais herdam a ordenação e o **bit de validade** já existentes, e o acréscimo é puramente aditivo.

**Termometrização.** Mesmo codificador escolhido em G0, com três especificidades:

- escala em **metros**, limites do suporte empírico de `sqrt(tr P)` medido nas rodadas do degrau 4 (não inventado);
- **bins em quantis (ou log-espaçados)**, porque a distribuição de `sqrt(tr P)` é assimétrica e concentrada perto do regime estacionário do filtro — um termômetro linear gastaria quase todos os bits na cauda;
- resolução própria `T_P`, **não** igual a `T`.

> **Restrição de diluição.** Cada bit de entrada é amostrado por cerca de `n·N/B` das RAMs. Inflar `B` dilui a cobertura de todos os bits, inclusive os da observação original — o degrau 5 pode piorar por engenharia de representação e ser lido como refutação de H4. Regra: os canais de incerteza não podem passar de **15 % de `B`**. Com `T = 32` e 25 escalares (`B ≈ 800`), 13 canais a `T_P = 8` custam 104 bits ≈ 13 % — cabe. Com `T = 8` (`B ≈ 200`), os mesmos 104 bits são 52 % — **não cabe**. Daí a regra `T_P = clamp(T/4, 2, 16)`, e, quando ela ainda estourar, a variante reduzida abaixo.

**Variante 5-lite (obrigatória como ablação interna).** Só `u_self` e `u_1` (incerteza do slot mais próximo): 2 canais. Se 5-lite empatar com 5-completo, o resultado é que a política usa a incerteza **própria**, não a dos sensores, e isso muda a interpretação de H4 — de "busca ativa sobre alvos" para "cautela sobre si". Distinguir as duas coisas vale 2 canais de experimento.

**Desvio §5.4.** O degrau 5 acrescenta canais à observação e portanto **entra na tabela de desvios do PRD quando executado**, com decisão registrada na §10. Texto proposto para a linha 2:

> | 2 | **Canais de incerteza na observação** — só `sqrt(tr P)` **própria**; a versão *por slot* saiu com D13 —, apenas no degrau 5 da Etapa 2 | Testar H4 exige que a política enxergue a covariância que o KF já produz; não há como fazê-lo sem estender a entrada | Executado como **ablação explícita**, com resultado reportado com e sem; anexo ao padrão binário, sem alterar a eq. 5-1; ausente nas Etapas 1 e 3a; se H4 não se confirmar, o degrau 5 não é retido na configuração final e a linha é marcada como ablação não adotada |

A atualização do PRD faz parte do entregável, não é tarefa posterior.

### T-08.5 — A escada na célula-âncora

**Célula-âncora `A` = (`σ_GPS = 5 m`, `T = T_G1`)**, onde `T_G1` é a resolução escolhida na Etapa 1; se G1 não singularizar uma, usar `T = 32`.

Rodar os degraus 0–5 completos em `A`, nos dois regimes de treino, com 5 sementes. É a tabela que vai para o relatório como figura principal da Etapa 2, e é onde os testes de H3 acontecem:

| Comparação | Teste | Predição |
|---|---|---|
| 4 vs 0 | TOST, margem δ | equivalente |
| 4 vs 1 | superioridade | 4 muito melhor |
| 4 vs 2 (ambas as variantes) | superioridade, Holm | 4 melhor |
| 4 vs 3 | superioridade, Holm | 4 melhor; 3 possivelmente pior que 1 |
| 5 vs 4 | superioridade (só regime ruidoso) | H4 |
| 5-lite vs 5 | exploratório | interpretação de H4 |

**Veredito de H3:** confirmado se `4 ≡ 0` por TOST **e** `4 > 2` e `4 > 3` com IC separado após Holm. Se `4 > 2, 3` mas `4 ≢ 0`, o veredito é **parcial** e a lacuna residual é quantificada em segundos e em fração de `t̄₁ − t̄₀`; T-08.7 diz se a lacuna é do filtro ou da resolução do codificador, que é uma distinção que muda a conclusão do trabalho.

### T-08.6 — Regime de treino: transferência vs randomização de domínio

Dois braços, cruzados com toda a escada:

| Regime | Treino | Avaliação | O que responde |
|---|---|---|---|
| **L→R** (transferência) | limpo (política da Etapa 1, reaproveitada) | ruidosa, degrau `d` | o filtro basta para transferir sem retreinar? |
| **R→R** (randomização de domínio) | ruidoso com o **mesmo** σ e o **mesmo** degrau `d` | ruidosa, degrau `d` | quanto o treino sob ruído ainda acrescenta? |

O PRD é explícito: se o degrau 4 tornar os dois **equivalentes**, é o argumento mais forte possível a favor do filtro — porque significa que o filtro devolveu o problema ao regime em que a política sem ruído já é ótima, e que retreinar sob ruído não compra nada.

**Como medir "equivalente".** TOST sobre a diferença `t̄(R→R) − t̄(L→R)` no degrau 4, mesma margem δ da T-08.2, com bootstrap hierárquico — aqui o pareamento é **por episódio** (mesmos cenários e mesmo `z`), mas as políticas são distintas, então a variabilidade de semente de treino entra pelos dois lados e o nível 1 do bootstrap reamostra **pares** de sementes.

> **Controle positivo obrigatório.** A equivalência no degrau 4 só é informativa se, no degrau **1**, o regime R→R for **significativamente melhor** que L→R. Se treinar sob ruído bruto também não ajuda, então não havia nada a ganhar com randomização de domínio em lugar nenhum, e a equivalência no degrau 4 é vacuidade. Este controle roda **antes** da leitura do degrau 4 e faz parte do pré-registro. Se ele falhar, o braço R→R inteiro é reportado como não informativo e H3 se apoia só na comparação 4 vs 0.

Nota de terminologia: "randomização de domínio" em sentido estrito randomiza σ durante o treino. Aqui R→R é treino com σ **casado** ao da avaliação, que é o caso mais favorável ao braço ruidoso e portanto o teste mais duro para a tese do filtro. A variante com σ sorteado por episódio é um terceiro braço **opcional**, só na âncora, se sobrar orçamento.

### T-08.7 — Varredura conjunta σ × T e a regra *bin ≈ σ*

Grade: `σ_GPS ∈ {1, 3, 5, 10} m` × `T ∈ {8, 32, 128}` = **12 células**. Conjunta, pela regra da §1 do PRD.

**Como cabe no orçamento.** A fatoração é o truque, e ela é o que torna o experimento mais caro do projeto executável em 3 dias:

| Bloco | Treinos | Avaliações | Racional |
|---|---|---|---|
| Degrau 0 | **0** | 3 (`T`) × 5 sementes = 15 | políticas reaproveitadas da SPEC-06; independe de σ |
| L→R, degraus 1–4 | **0** | 4σ × 3T × 4 degraus × 5 sementes = 240 | **treino zero**: a mesma política limpa é avaliada sob cada estimador |
| R→R, degrau 4 | 12 × 5 = 60 | 60 | grade completa — é o degrau da proposta |
| R→R, degrau 1 | 4 cantos × 5 = 20 | 20 | piso e controle positivo; cantos `{1,10} × {8,128}` |
| R→R, degrau 5 e 5-lite | 4 cantos × 5 × 2 = 40 | 40 | H4 |
| R→R, degraus 2 e 3 | âncora × 3 = 6 | 6 | referência barata fora da âncora não se justifica |
| MLP pareada (H2) | 3 (âncora) | 2 degraus × 4σ × 3 sementes = 24 | L→R reaproveita a MLP da Etapa 1 |
| Intervenções κ e controles de H4 | 0 | ≈ 60 | avaliação de política já treinada |
| **Total** | **≈ 109** | **≈ 465** | |

Com `S₂ = 2·10⁶` passos, episódio médio de ~150 passos de decisão e custo de parede `E` por episódio: treino ≈ `1,3·10⁴ · E`, avaliação ≈ `500 · E`. Com `E = 0,5 s` (extrapolação de PLANO/P2: 0,11 s por 120 s simulados), dá **≈ 200 h de núcleo de treino + ≈ 65 h de avaliação**, ou ~1,5 dia em 8 núcleos. As rodadas são embaraçosamente paralelas por `(degrau, σ, T, semente)`.

> **Verificação de cabimento, obrigatória e antecipada.** Medir `E` e o comprimento médio de episódio **nas primeiras duas horas**, com uma célula completa, e recalcular a tabela. Se o total projetado passar de 60 % do tempo disponível, aplicar os cortes **nesta ordem declarada**: (i) sementes 5 → 3 fora da âncora; (ii) grade reduzida aos 4 cantos `σ ∈ {1,10} × T ∈ {8,128}` (linha de corte 4 do PRD §8); (iii) degraus 2 e 3 só na âncora — retendo ao menos um dos dois, conforme PRD §8; (iv) `S₂` reduzido pela metade **para todos os degraus simultaneamente**, nunca para alguns. Cortar `S₂` de forma assimétrica destrói a comparabilidade e é pior que cortar células.

**A análise da regra, que é o produto intelectual desta tarefa.** Não basta o mapa de calor. Para cada célula, medir:

- `σ_eff,j` — desvio-padrão **empírico** do erro da coordenada relativa no slot `j`, que é o que o codificador enxerga: combina erro de posição própria e erro de posição do sensor, `σ_eff² ≈ σ_self² + σ_sensor²`. É esta a grandeza que a regra compara com o bin, não `σ_GPS`.
- `w_j(x)` — largura **local** do bin que contém o valor observado, no slot `j`. Se G0 escolheu binning por quantis, `w` não é `400/T`; tem de ser medida.
- `ρ_j = w_j / σ_eff,j` — a razão que a regra prevê governar tudo.

**Predição pré-registrada:** a degradação relativa ao degrau 0 colapsa (as curvas de `T` se juntam) quando `ρ < 1`, e `T` só compra desempenho enquanto `ρ > 1`. Consequências verificáveis, que é o que torna a predição útil:

1. No **degrau 1**, com `σ = 10 m`, `T = 32` e `T = 128` devem empatar; com `σ = 1 m`, `T = 128` deve ganhar.
2. No **degrau 4**, `σ_eff` cai para a faixa de décimos de metro (PLANO §9.4: 0,19–0,67 m) e portanto `ρ > 1` em **toda** a grade — logo `T = 128` deve vencer em todo σ e a dependência em σ deve praticamente desaparecer. Este é o enunciado forte de H3 e o mais fácil de falsificar.

Produto: tabela `T*(σ, degrau)` recomendada e **comparação com a predição de T-01.8 da SPEC-01**, que estimou a mesma regra fora do ambiente. Concordância entre as duas é validação cruzada barata; discordância é achado, e provavelmente aponta para a diferença entre a distribuição analítica e a on-policy real (invariante V-1 da SPEC-01).

### T-08.8 — H2: fragilidade relativa da WNN contra a MLP

A comparação certa **não** é de níveis. A WNN pode ser pior em absoluto e igualmente robusta; H2 fala de fragilidade, isto é, de **degradação relativa**:

```
D_m(σ) = ( t̄_m(σ) − t̄_m(0) ) / t̄_m(0)          m ∈ {WNN(T), MLP}
```

O teste de H2 é sobre a **interação** aproximador × σ: IC bootstrap da diferença `D_WNN(σ) − D_MLP(σ)` em cada σ, com o mesmo pareamento por episódio. H2 cai se esse IC excluir zero pelo lado desfavorável em algum σ, e não simplesmente se as curvas de nível estiverem separadas — separação de níveis é assunto de H1, já resolvido em G1.

Rodar em L→R (política limpa de ambos os lados, avaliação sob degraus 1 e 4) na grade completa de σ, com a WNN em cada `T`. A MLP não tem `T`: comparar contra cada `T` separadamente e contra o melhor `T`, reportando as duas leituras, porque escolher só a favorável é o vício que o PRD §5.5 já proíbe para tamanho.

Reportar também a versão com estimativa filtrada (degrau 4): se ambas voltarem ao regime sem ruído, H2 fica confirmada de forma pouco interessante. O teste com conteúdo é o do degrau 1.

### T-08.9 — ~~H4: medir busca ativa~~ → o degrau 5 usa os canais de `P`?

> **Reenquadrada por D13.** A H4 saiu: sem `R(d)` crescente, aproximar-se não
> compra informação e não há busca ativa a detectar. **A tarefa não é
> descartada** — as métricas B1/B2 medem se a política *usa* os canais de
> incerteza, o que continua sendo pergunta legítima e continua exigindo número em
> vez de anedota.
>
> O que muda: **as predições da coluna "Predição de H4" perdem o respaldo
> teórico.** Deixam de ser predições e passam a ser *observações a reportar*.
> Concretamente, o teste de intervenção `κ` (multiplicar os canais de `P` por um
> fator e ver se a trajetória muda) responde "a política lê esses bits?"; ele
> **não** responde mais "a política busca reduzir incerteza?", porque não há
> incerteza reduzível por deslocamento.
>
> **Veredito possível agora:** *usa* / *não usa* / *usa sem ganho*. O terceiro
> caso — lê os canais mas o tempo de conclusão não melhora — deixa de ser
> surpreendente e passa a ser o resultado esperado por omissão.

**Esta tarefa é a lacuna real da spec e recebe tratamento próprio.** O PRD lista "inspeção de trajetórias" como método para H4, e inspeção de trajetórias não decide nada: sempre se acha uma trajetória bonita. A hipótese — *uma política que enxerga a própria covariância aprende a se aproximar antes de decidir, reduzindo variância antes de comprometer trajetória* — é uma afirmação sobre **política condicional**, e é isso que precisa ser estimado.

**Definições operacionais** (fixadas no pré-registro, antes de qualquer rodada):

- **Alvo aparente** `g_t`: o sensor não visitado mais próximo **segundo a estimativa corrente**, isto é, o alvo que o guloso escolheria com a mesma informação que a política tem.
- **Ação gulosa-sobre-estimativa** `a^g_t`: rumo para `g_t`, velocidade máxima. É o comportamento de referência de quem ignora incerteza.
- **Desvio angular** `Δθ_t = |ang(a_t) − ang(a^g_t)|`, dobrado em `[0, π]`.
- **Incerteza corrente** `U_t = sqrt(tr(P_self,t))`, padronizada dentro do episódio.
- **Instante de compromisso** `τ_c` de um segmento (entre duas coletas): primeiro passo após o qual o rumo comandado permanece dentro de ±15° até a coleta. É o momento em que a trajetória "trava".

**Bateria de métricas.** Primárias em negrito; o resto é secundário e corrigido por Holm.

| id | Métrica | Definição | ~~Predição de H4~~ → observação a reportar |
|---|---|---|---|
| **B1** | **Índice de desvio condicional (IDC)** | coeficiente `β_U` na regressão `Δθ_t ~ β_U·U_t + controles`, com controles de distância ao alvo, número de sensores restantes e passo no episódio, e efeito fixo de episódio | `β_U > 0`: desvia do guloso **quando, e só quando**, está incerta |
| **B2** | **Teste de intervenção κ** | multiplicar **apenas os canais de `P`** por `κ ∈ {0,25; 1; 4}` na avaliação, sem tocar na estimativa nem no ruído; medir variação de velocidade média comandada, de IDC e de tempo de conclusão | efeito **monótono** em κ. Se nada muda, os canais são decorativos e H4 cai, independentemente de qualquer outra métrica |
| B3 | Incerteza no compromisso | `E[U_{τ_c}]`, pareado por episódio | menor no degrau 5 que no 4 |
| B4 | Informação por metro voado | `Σ_t Δ log det P^s` acumulado, dividido pela distância percorrida | maior no degrau 5 |
| B5 | Trocas de alvo por episódio | número de mudanças de `g_t` | menor no degrau 5 (menos *thrashing*) |
| B6 | Elasticidade da velocidade | `β_s` em `s_t ~ β_s·U_t + controles` | `β_s < 0`: desacelera sob incerteza |

**O nível não é o efeito: o efeito é a diferença contra o degrau 4.** `Δθ`, velocidade e `U` são todos correlacionados com a geometria do episódio, então B1 e B6 são não nulos mesmo numa política que **não enxerga** `P` — o ambiente cria a correlação sozinho. Por isso todas as métricas são computadas **também** no degrau 4 (que não tem os canais) e a estatística de H4 é `B(5) − B(4)`, com IC bootstrap hierárquico. O degrau 4 é a distribuição nula empírica, e é a única forma honesta de ler B1.

**Controle de canal embaralhado.** Avaliar o degrau 5 com os canais de `P` substituídos pelo traço de `P` de **outro** episódio (mesma distribuição marginal, informação destruída). Se o comportamento e o desempenho não mudarem, os canais não estão sendo usados. Este controle e B2 são baratos — não exigem retreino — e juntos separam "a política usa a incerteza" de "a política melhorou por acaso ao ganhar 104 bits de entrada".

**Veredito de H4:**

- **confirmada** se B2 for monótono em κ **e** `B1(5) − B1(4) > 0` com IC separado **e** o ganho de tempo de conclusão de 5 sobre 4 for positivo com IC separado;
- **mecanismo sem ganho** se B1/B2 confirmarem o uso dos canais mas o tempo de conclusão empatar. Pelo PRD ("H4 cai se o ganho for nulo") isso é refutação **quanto ao desempenho**, e é assim que deve ser escrito — com a evidência comportamental reportada junto, porque ela é o resultado interessante e é o que aponta trabalho futuro;
- **refutada** se B2 for plano: os canais não são usados, e o degrau 5 não é retido.

Figuras qualitativas de trajetória entram no relatório como **ilustração** do efeito já estabelecido por número, nunca como evidência.

### T-08.10 — Custo com o filtro embutido

O KF é código que voa no UAV. T1/T2/T3 da §5.5 são **recalculados**, não copiados da Etapa 1.

| Item | O que medir |
|---|---|
| T3 — operações | flops do KF por decisão (predição + atualização, 2 estados), medidos e não estimados; contraste com a estimativa de 50–80 flops de PLANO §9.8 e com os ~2600 MACs da MLP; **filtro de informação por atualização de sensor**, reportado como média por decisão **e como pior caso** (várias atualizações no mesmo passo), porque a média esconde o pico e o pico é o que o microcontrolador precisa aguentar |
| T1 — memória | estado do filtro próprio (2 + 3 floats) e dos `Ns` filtros de informação (`Ns` × (2 + 3)); somado ao ator; ainda em poucos KB, mas **reportado**, não presumido |
| T1 — degrau 5 | a ocupação das LUTs **muda** com `B` maior: mais bits de entrada, mais endereços distintos. Refazer a curva de ocupação da RNF-08 para o degrau 5, não reaproveitar a da Etapa 1 |
| T2 — latência | recompor a decomposição por estágio da §5.5 com duas linhas novas: **filtro** e **codificação dos canais de incerteza**; mediana e p99 |
| Ressalva RNF-05 | a fração do orçamento que o KF consome, escrita como número na tabela, com a frase correspondente no relatório. Se passar de ~5 %, a narrativa "sem MACs" precisa ser reescrita como "MACs confinados a um filtro de 2 estados", que continua defensável |

### T-08.11 — Vereditos e relatório

`reports/spec08-etapa2.md`, entrando direto no capítulo de extensão de E2 (capítulo separado da reprodução, conforme §0 do PRD). Conteúdo mínimo: tabela da escada na âncora nos dois regimes; mapa de calor `σ × T` do degrau 1 e do degrau 4 com a análise de `ρ`; curvas de degradação relativa WNN vs MLP; tabela da bateria de busca ativa com os controles; tabela de custo recalculada; e **três vereditos escritos** (H2, H3, H4), cada um com o número que o sustenta e a margem/IC que o qualifica. Hipótese refutada é resultado.

---

## 6. Critério de saída — portão G2

1. **`gate_rf05.json` aprovado** para os canais e degraus usados, com o hash da configuração de filtro que foi de fato executada, e RF-04 verde.
2. **Degrau 4 indistinguível do degrau 0** por TOST com margem δ na célula-âncora, com meia-largura reportada ≤ δ.
3. **Degrau 4 supera os degraus 2 e 3** com IC da diferença pareada separado de zero, após Holm — incluindo a variante do degrau 2 **com** termo de comando.
4. **Varredura σ × T reportada** na grade executada, com a análise de `ρ = w/σ_eff` e a tabela `T*(σ, degrau)`, confrontada com a predição de T-01.8.
5. **Vereditos escritos de H2 e H3**, com número e IC. Para o **degrau 5**, relato de *uso dos canais de `P`* pelas métricas da T-08.9 — não veredito de hipótese, porque a H4 saiu com D13.
6. **T1/T2/T3 recalculados com o filtro embutido**, incluindo a curva de ocupação do degrau 5 se ele for executado.
7. **PRD §5.4 atualizado** com a linha 2 e decisão registrada na §10, se o degrau 5 foi executado.
8. Configurações, sementes e resultados versionados; `make repro` regenera as tabelas desta spec.

### Se G2 não for atingido

Portão não atingido é resultado publicável (PRD §2). O que muda é a ação, e ela depende de **qual** condição falhou:

| Falha | Leitura | Ação |
|---|---|---|
| NEES/NIS reprova no degrau 4 | não é resultado de RL; é filtro mal sintonizado | volta a SPEC-07. Nenhum número de RL desta spec é reportado. **Não** consome o portão |
| `4 ≢ 0`, mas `ρ < 1` na célula | a lacuna é de **resolução**, não de filtro | rodar `T` maior na âncora; se fechar, H3 confirmada com a ressalva de que o filtro sozinho não basta — o par filtro+codificador é que basta. É o enunciado da regra *bin ≈ σ*, e é uma conclusão melhor que a original |
| `4 ≢ 0` com `ρ > 1` folgado | H3 refutada | reportar a lacuna residual em segundos; diagnosticar por canal (a incerteza que sobra é própria ou dos sensores?); a resposta prática passa a ser o braço R→R, e é assim que se escreve |
| `4 ≯ 2` | o filtro barato basta | resultado legítimo e desconfortável: reportar, e manter o degrau 2 como referência na Etapa 3. A tese migra para "conhecer o comando é o que importa" |
| `4 ≯ 3` | contradiz o achado A3, que é **medido** | suspeitar de implementação antes de suspeitar da hipótese: reproduzir a tabela sintética de PLANO §9.4 com o mesmo código; se ela reproduz e o ambiente não, a diferença é do ambiente (inércia, atraso, saturação) e vira achado sobre `Q` |
| Degrau 5 sem uso detectável | degrau 5 não é retido | §5.4 volta a ter uma linha só; reportar a ablação como executada e negativa. Nenhum impacto sobre a Etapa 3 |
| Falta de tempo | — | linha de corte da §8 do PRD, na ordem declarada em T-08.7. **Não podem cair:** degraus 0, 1 e 4; NEES/NIS; RF-04; ao menos um dos degraus 2/3; a tabela de custo |

**Regra de escalonamento (PRD §8):** portão não atingido ao fim da semana 3 aciona a linha de corte da **Etapa 3**, nunca estende a Etapa 2 para dentro dela.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **Conclusão de RL tirada antes do NEES/NIS**, por pressa na semana 3 | o bloqueio B1 é mecânico: `make etapa2` recusa-se a rodar sem o atestado, e o atestado morre quando a config do filtro muda |
| **Degrau 3 vira espantalho** por `q` mal escolhido, e o resultado a favor do degrau 4 fica sem valor | varredura de `q` dentro do ambiente, com o melhor `q` reportado por σ — como já foi feito no sintético |
| **Degrau 5 piora por diluição de bits** e é lido como refutação de H4 | teto de 15 % de `B`, `T_P = clamp(T/4, 2, 16)`, variante 5-lite, e o teste κ (B2), que distingue "não usa os canais" de "os canais atrapalharam" |
| **Busca ativa "confirmada" por correlação do ambiente**, não por uso da covariância | toda métrica comportamental é lida como diferença contra o degrau 4; mais o controle de canal embaralhado e a intervenção κ |
| **Garden of forking paths**: 12 células × 6 degraus × 2 regimes × 6 estatísticas comportamentais | pré-registro hasheado (B3), métricas primárias declaradas, Holm na família secundária |
| **Equivalência declarada por falta de poder** | TOST com δ autocalibrada, meia-largura sempre reportada, células com `t̄₁ − t̄₀` pequeno marcadas como não informativas |
| **Viés de seleção no tempo de conclusão** quando as taxas de sucesso diferem | métrica primária censurada em 500 s; taxa de sucesso sempre ao lado |
| **Orçamento estoura** — é o experimento mais caro do projeto | fatoração L→R sem custo de treino; verificação de cabimento nas duas primeiras horas; ordem de cortes declarada, com `S₂` cortado simetricamente ou não cortado |
| **Ordenação dos slots vira canal de ruído**: os 12 sensores são ordenados por distância **estimada** | medir a taxa de troca de ordenação entre passos consecutivos, por degrau, e reportá-la como covariável — é parte da degradação, e o filtro deve reduzi-la |
| **`Q` errado por descasamento entre comando e movimento** | medir o descasamento empírico com a mobilidade efetivamente usada, antes do atestado |
| **A política limpa da Etapa 1 é medíocre** e a escada mede degradação de algo ruim | toda a Etapa 2 mede degradação **relativa ao próprio degrau 0**, e permanece válida (PRD §6.1, cláusula de G1 não atingido). Reportar o nível absoluto sempre ao lado |
| Diferença entre o corpus analítico da SPEC-01 e a distribuição real | a comparação `T*` predito vs medido em T-08.7 é justamente o teste; divergência realimenta a SPEC-01 |

---

## 8. Entregáveis

- `configs/etapa2/` — escada, grade `σ × T`, regimes de treino, sementes, `S₂`
- `results/etapa2/gate_rf05.json` — atestado de consistência, com hash da configuração de filtro
- `results/etapa2/analysis_plan.md` + hash — pré-registro (métricas primárias, δ, sinais esperados, família de testes)
- `results/etapa2/*.parquet` — resultados por episódio, com `(degrau, σ, T, regime, seed_train, seed_eval)` como chave, para que o pareamento seja verificável a posteriori
- `reports/spec08-etapa2.md` + figuras: escada na âncora, mapa de calor `σ × T`, curva `ρ` vs degradação, degradação relativa WNN vs MLP, bateria de busca ativa com controles, tabela de custo com filtro
- Atualização do **PRD §5.4 (linha 2) e §10 (decisão)** se o degrau 5 for executado
- Trechos de E2: capítulo de extensão com os vereditos de H2 e H3, mais o registro da **H4 como hipótese abandonada com justificativa** (D13)

---

## 9. Perguntas em aberto

1. **O que a política enxerga de um sensor ainda não descoberto?** O PRD §6.2 diz que a posição dos sensores é "descoberta por enlace local". Lido ao pé da letra, um sensor nunca visto não tem posição e não pode ocupar um slot de `P^s` — o que transforma o problema numa tarefa de exploração e **altera o MDP de Lamenza**, violando a §0. A leitura que preserva a formulação é: todas as posições vêm do canal global, **corrompidas** por `σ_sensor`, e o enlace local apenas refina a estimativa pelo filtro de informação. Esta spec assume a segunda leitura. É decisão de SPEC-07 e precisa estar escrita lá antes de T-08.3.
2. **`σ_sensor` não está fixado em lugar nenhum.** A grade varre `σ_GPS`; o erro da posição dos sensores é o outro termo de `σ_eff` e domina a leitura da regra *bin ≈ σ* nos slots distantes. Varrer os dois seria uma grade 4×4×3; a proposta desta spec é fixar `σ_sensor` em um valor (relacionado ao `communication_range` de P1) e varrer só `σ_GPS`, reportando a escolha. Falta decidir o valor.
3. **`S₂ = 2·10⁶` é valor de trabalho**, não medido. Depende do joelho da curva da SPEC-06. Se a Etapa 1 mostrar que a política ainda melhora em 10⁷, a Etapa 2 compara políticas subtreinadas — o que é aceitável para diferenças relativas, mas precisa aparecer como ressalva no relatório.
4. **Mobilidade com ou sem inércia na Etapa 2.** Com inércia, `Q` fica mensurável e o argumento sim-to-real melhora; sem inércia, o degrau 4 fica quase trivialmente correto e o experimento perde graça. A escolha muda a magnitude esperada de todo o resultado e ainda não foi feita.
5. **Cinco sementes de treino sustentam TOST?** O nível 1 do bootstrap hierárquico com `n = 5` é frágil. Se a variância entre sementes for alta, a equivalência pode ficar indeterminável com o orçamento previsto. Alternativa a decidir: mais sementes com `S₂` menor.
6. **A margem δ = 10 % de `t̄₁ − t̄₀` é uma escolha.** É defensável e autocalibrada, mas arbitrária. Vale confrontá-la com uma margem em unidades absolutas (por exemplo, 2 % do tempo do guloso) e reportar as duas leituras se elas discordarem.
7. **O degrau 5 fora do regime ruidoso.** A assimetria do desenho (5 só existe em R→R) enfraquece a leitura de H4, porque a comparação 5 vs 4 fica confinada a um regime. Se houver orçamento, considerar um braço "treino ruidoso com `P` congelado" como controle adicional.
8. **`T = 128` pode não ser o topo.** A predição de T-08.7 é que, sob o degrau 4, `ρ > 1` em toda a grade — o que significa que a grade **não contém** o ponto de saturação e que o melhor `T` está acima de 128. Se isso se confirmar, é preciso decidir se a grade se estende (custo, e `B` cresce) ou se o resultado é reportado como "não saturado no intervalo medido", com a ressalva explícita.
