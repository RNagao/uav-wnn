# Revisão cruzada das specs — lacunas e pontos a decidir

> ## Estado desta revisão
>
> Documento de **um momento**: a revisão cruzada das 11 specs, feita antes de qualquer implementação. Vários achados já foram resolvidos ou reformulados desde então. O registro vigente de pendências é [`../PENDENCIAS.md`](../PENDENCIAS.md).
>
> | Achado | Estado |
> |---|---|
> | **A** — o ator WNN pode sair maior que a MLP pareada | **aberto** (T1). Segue sendo a decisão mais consequente |
> | **B** — parâmetros ausentes (P1, P4–P8) | **fechados**, exceto P4 e P7. Extraídos do código de referência |
> | **C.1** — θ circular vs média linear | **aberto**. Recomendação: normal projetada |
> | **C.2** — esquecimento vs gradiente funcional | **aberto** |
> | **C.3** — estrutura de armazenamento das LUTs | **aberto** |
> | **C.4** — cobertura de bits `N·n/B` | **aberto**, e a SPEC-01 foi corrigida para varrer `(T, N)` juntos |
> | **C.5** — filtro de informação sem o que fundir | **resolvido** pelo novo modelo de ruído: o relatório vem do canal global reinterpretado |
> | **D.1** — canal de sensores degenerado | **resolvido** junto com C.5 |
> | **D.2** — CI reprova no G3 por funcionar | **aberto** (E2). Precisa de critério assimétrico |
> | **E** — achados de API do `gradysim` | **confirmados por execução**, mas na versão **0.6.11**, não 0.8.0 |
> | **E.6** — célula cara é `Ns=36`, não `Na=8` | **corrigido novamente**: em horas de treino a célula cara é `Na=1` |
> | **F.1** — sementes de treino | **aberto** (E1) |
> | **G.1** — cobertura de bits na SPEC-01 | **aplicado** |
>
> Achados novos, posteriores a esta revisão: **T3** (física do ruído no sensor), **P9**/**P10** (parâmetros dos canais), e a separação da Etapa 2 em blocos 2a/2b.

---

Consolidação dos achados das 11 specs, deduplicada e priorizada. Cada item traz o que foi achado, por que importa, e a recomendação.

**Como ler:** os itens de **decisão** precisam de resposta humana antes de codificar. Os de **achado** mudam o plano mas não bloqueiam. Os de **correção** já foram aplicados.

---

## A. A hipótese H1 está em risco — e três specs independentes convergiram nisso

Este é o achado mais importante da revisão. SPEC-03, SPEC-04 e SPEC-05 chegaram, por caminhos diferentes, à mesma conclusão: **o ator n-tuple provavelmente fica MAIOR que a MLP pareada.**

| Fonte | Estimativa do ator WNN | Como chegou |
|---|---|---|
| SPEC-03 | ~12,3 KB | hash aberto, α=0,75, chave+valor 2+2 B, sobre occ=2 268 e N=48 |
| SPEC-04 | ~14,5 KB (congelado) / ~64 KB (hash com chave uint32, α=0,5) | idem, com mapeamento de tuplas incluído |
| SPEC-05 | ~14,8 KB | o ator emite **d=3** saídas, não 2 — ver item C.1 |
| **MLP pareada** | **~8,6 KB** | 2 146 parâmetros fp32, arquitetura de Katopodis |

O PRD §5.5 já previa esse cenário em tese e prescrevia migrar o argumento para contagem de operações. O que mudou é a **magnitude**: não é empate, é provavelmente derrota por ~1,5–2×. E há uma assimetria estrutural que eu não tinha visto: **o custo da WNN escala com o número de cabeças de saída** (cada cabeça tem suas próprias `N` entradas por endereço), enquanto na MLP só muda a última camada. Ir de 2 para 3 saídas triplica o ator WNN e muda a MLP em 32 parâmetros.

Agrava: SPEC-03 mostra que **"zero MACs" é falso como escrito** — a acumulação usa somas em ponto flutuante e a amostragem gaussiana usa transcendentais. E SPEC-05 calculou que **~800 comparações do codificador dominam o lado WNN** na contagem de operações; a vantagem é parcialmente do codificador, não da rede.

> **DECISÃO A.1 — reformular H1 antes da semana 2, não depois.**
> Recomendação: a alegação central passa a ser **"zero multiplicações em ponto flutuante no aproximador"** (já corrigido no PRD), com o tamanho reportado honestamente contra as duas referências. Se a WNN sair maior que a MLP pareada, o trabalho continua tendo resultado — mas o resultado é *"aproximador sem multiplicação com desempenho equivalente"*, não *"modelo menor"*. Precisa de aval, porque muda a manchete.

> **DECISÃO A.2 — escolha da estrutura de armazenamento vale 2–5× no número principal.**
> Hash aberto (premissa atual) vs chaves ordenadas + busca binária (menor **e** de cauda mais previsível, ao custo de O(log occ)) vs densa. SPEC-03 aponta que a segunda pode ganhar nas duas dimensões que nos interessam — tamanho e p99. Se ganhar, a SPEC-04 muda de alvo. Recomendo medir as três antes de congelar.

> **DECISÃO A.3 — persistente vs residente.** O mapeamento das tuplas é uma permutação de PRNG: uma semente de 4 B substitui ~1,2 KB em flash, mas não em RAM. SPEC-03 propõe reportar as duas colunas com a residente como número principal. Concordo, mas registro o risco: é o tipo de contabilidade que um revisor pode ler como conveniente, então precisa vir com as duas colunas sempre.

---

## B. Parâmetros ausentes bloqueiam a comparabilidade — e são mais que o P1

Eu tinha uma pendência (P1, `communication_range`). A SPEC-00 achou **mais quatro** parâmetros que Lamenza não publica e sem os quais o ambiente diverge silenciosamente do original:

| # | Parâmetro | Por que é bloqueante |
|---|---|---|
| P1 | `communication_range` | define quando o sensor é coletado; sem ele nenhum número é comparável |
| **P5** | **Período do heartbeat** | se `v_max · heartbeat > communication_range`, o UAV **atravessa a cobertura sem coletar** e o ambiente fica mais difícil que o original, sem aparecer na tabela de desvios |
| **P6** | Raio da área de largada dos UAVs | "small area near the center", sem número |
| **P7** | `max_episode_length` "permissivo" do §7.1 | sem ele, os 100 % publicados não são alvo comparável |
| **P8** | Tratamento de borda | o `gradysim` não limita posição e a eq. 5-6 não penaliza borda; se Lamenza satura em [−100,100]² e nós não, o ambiente diverge |
| **P9** | `σ_sensor` (erro do levantamento dos sensores) | não existe no PRD, mas **domina a regra bin ≈ σ** nos slots distantes (ver D.3) |

A SPEC-06 achou uma verificação de sanidade independente e engenhosa para P1: a diagonal da tab. 7.3 de Lamenza (13,47 s com 4 agentes, 22,71 s com 2) implica, a 15 m/s em 200×200 m, um `communication_range` da ordem de **dezenas de metros**. Não substitui o valor real, mas rejeita valores absurdos.

> **DECISÃO B.1** — P1 e P5 sobem a **bloqueantes da semana 1**. O repositório `Project-GrADyS/gradys-data-collection@dissertation` é a fonte; se não tiver, contatar o autor. P5 é tão crítico quanto P1 e eu não o tinha percebido.

---

## C. Duas lacunas matemáticas reais no aproximador

### C.1 — θ é circular e a saída n-tuple é uma média linear

A melhor contribuição técnica da rodada, da SPEC-05. Eu sabia da descontinuidade em 0/2π. O problema é pior: **a saída da rede n-tuple é média linear entre RAMs, e média linear de ângulos é inválida** — 0,02 e 0,98 promediam para 0,5, que é o oposto do ângulo correto. Isso invalida gaussiana envolvida escalar *e* von Mises com média angular escalar.

Recomendação da SPEC-05: **normal projetada** — a rede emite `u ∈ R²`, amostra-se `z ~ N(f, Σ)`, executa-se `θ = atan2(z₂, z₁)`. Preserva a formulação de Katopodis sem modificação, funciona a partir da memória zerada (dá rumo uniforme, que é a exploração inicial correta), e torna `σ_u` não identificável — um hiperparâmetro a menos. Preço: `‖f‖` vira concentração aprendida e pode explodir.

Consequência: **o ator emite d=3** (`u₁`, `u₂`, `μ_s`), não 2. Daí o item A acima.

> **DECISÃO C.1** — adotar normal projetada. É a recomendação, e ela precisa de validação empírica num brinquedo de navegação antes da SPEC-06. Note que von Mises foi rejeitada por incompatibilidade com inicialização em zero, não por complexidade — vale confirmar que concorda com o descarte.

### C.2 — esquecimento e gradiente funcional são duas arquiteturas diferentes em Katopodis

SPEC-03 e SPEC-04 acharam o mesmo problema independentemente. Katopodis §4.2.1 (rede com esquecimento) usa **valor + contador** com média auto-normalizada e soma de série geométrica; §4.3.1 (gradiente funcional) **não tem contador nem denominador**. O PRD (RNF-04) pede gradiente funcional **com** esquecimento multiplicativo — combinação que nenhuma das duas seções cobre.

A SPEC-04 provou uma correspondência (`c ↔ 1−φ`, a NTRN-φ é EWMA por entrada), mas mostrou que **ela não cobre a saída agregada**: a NTRN-φ pondera por confiança, o gradiente funcional não. A consequência é concreta e ruim — em região fria (endereços pouco visitados) a média da ação é puxada para **zero**, que sob a normalização [0,1]² de Lamenza é o canto do espaço de ação: **UAV parado apontando para leste**.

Se exigir contador, o "ocupado" **dobra** e a comparação de tamanho com a MLP muda de sinal outra vez.

> **DECISÃO C.2** — escolher entre: (a) parametrização centrada, (b) modo auto-normalizado com contador, (c) aceitar o viés e documentá-lo. Recomendo (a), que é a mais barata em memória e resolve o canto do espaço de ação. Precisa de decisão antes da SPEC-04.

### C.3 — três correções menores mas que mudam código

- **Convenção da soma:** o protótipo `ntuple.py` usa `f = Σ v[i]`; Katopodis usa `f = (1/N) Σ v[i]`. A regra `η = c/N` do PRD é a convenção de média reexpressa. Migrar invalida as taxas anotadas do protótipo.
- **Erros de transcrição em Katopodis:** o Algoritmo 4.4 aparece sem a taxa `η` na atualização do ator, com o crítico indexado por ação (sendo função só de estado), e com `v_policy` no lado direito da atualização do crítico. A eq. 4.11 tem sinal de subida sobre a perda, inconsistente com o alg. 4.3. Duas specs acharam isso separadamente. **Confirmar contra o PDF original antes de codificar.**
- **A fundamentação do relatório precisa de ajuste:** o crítico de Katopodis usa NTRN com esquecimento, que **não é passo de gradiente**. A garantia de "TD com aproximação linear" que eu cito no PRD §5.3 cobre o ator e a variante funcional, **não** esse crítico.

---

## D. Buracos de desenho que eu deixei nas Etapas 2 e 3

### D.1 — o filtro de informação não tem o que fundir na Etapa 2

Achado da SPEC-07, e é um furo real no meu desenho. Em Lamenza, **ouvir um sensor equivale a coletá-lo** (RF-03), e ele sai de `P^s` no mesmo instante. Logo o canal de sensores é o *prior* de levantamento e mais nada — não há múltiplas observações do mesmo sensor para fundir.

Duas saídas: aceitar o canal degenerado (filtro validado por forma fechada, exercitado de fato só na SPEC-09), ou criar `detection_range > communication_range`, que é **linha nova na tabela de desvios §5.4**.

> **DECISÃO D.1** — recomendo aceitar o canal degenerado na Etapa 2 e deixar a fusão de sensores para a Etapa 3, onde ela existe de verdade (dois UAVs, mesmo sensor). Criar `detection_range` seria modificar o problema para caber no método.

### D.2 — CI reprova no portão G3 por estar funcionando

Achado da SPEC-09, e é uma falha do critério que eu escrevi. A interseção de covariância é **conservadora por construção**: seu NEES/NIS fica *sistematicamente abaixo* do intervalo qui-quadrado. Com o critério simétrico do G3, **6c reprovaria justamente por funcionar como projetado**.

> **DECISÃO D.2** — adotar critério assimétrico (rejeita otimismo, aceita conservadorismo) mais um fator de conservadorismo `κ` reportado à parte. Precisa de acordo explícito, porque muda um portão.

### D.3 — o ganho da fusão é limitado pelo pior canal

SPEC-07: como a observação é **relativa**, `σ_rel = sqrt(σ_a² + σ_s²)`. Se `σ_prior` do levantamento dos sensores for ~5 m e irredutível, filtrar o GPS de 10 m para 0,7 m leva `σ_rel` de 11,2 m para 5,05 m — **fator 2,2, não 15**. O achado A3 (18,8×) é sobre a posição própria, não sobre o que a política enxerga.

> **DECISÃO D.3** — fixar `σ_sensor` (P9) e recalibrar a expectativa do resultado da Etapa 2. Isto muda o que a manchete da fusão pode prometer.

### D.4 — a taxa do GPS escolhe o resultado

SPEC-07: o achado A3 foi medido com decisão a **0,5 s**; com `algorithm_iteration_interval = 2 s` e GPS a 1 Hz são só 2 correções por decisão. **Escolher `dt_gps` é escolher o resultado.** Proposta: 1 s nominal, varredura em {0,2; 1; 5}.

### D.5 — interação entre `update_rate` e a varredura de σ

SPEC-07: `update_rate = 0,1` injeta até 3 m de erro de discretização por decisão, o que **domina `σ_gps = 1 m`**. Nessa célula da varredura mede-se o integrador, não o GPS. Escala para a P2.

---

## E. Achados sobre o `gradysim` que invalidam premissas do `PLANO.md`

A SPEC-00 inspecionou o wheel 0.8.0. **Ressalva:** o pacote **não está instalado neste ambiente**, então estes achados vêm de leitura de código, não de execução — confirmar no ambiente real antes de agir.

| # | Achado | Consequência |
|---|---|---|
| E.1 | `SetVelocityMobilityCommand` é **incompatível** com `MasslessMobilityHandler`: o handler lê só `param_1`, então o comando vira `speed = vx` **silenciosamente, sem erro** | A tabela do `PLANO.md` está errada. A ação ⟨θ,s⟩ precisa ser `SetSpeed` + `GotoCoords` |
| E.2 | `Simulator.__init__` adiciona `StreamHandler` ao logger raiz e nunca remove | Um `Simulator` por episódio × 10 000 episódios = vazamento e degradação progressiva |
| E.3 | `start_simulation()` não permite parar cedo (a mobilidade se reagenda, o loop nunca esvazia) | A poda por `max_seconds_stalled` obriga a bombear `step_simulation()` manualmente. A injeção de política sobrevive; muda quem gira a manivela |
| E.4 | `algorithm_iteration_interval` precisa ser **múltiplo inteiro** de `update_rate` | Restringe os candidatos de P2 a divisores de 2 s |
| E.5 | O alcance de comunicação é avaliado em **3D** | Todos os nós precisam de `z = 0`, senão o raio no solo é `sqrt(r²−h²)` |
| E.6 | Modelo de custo `eventos ≈ (T/update_rate)·(1+Na+Ns)`, validado contra as duas medições do PRD (204 000 vs 205 653; 20 400 vs 22 080) | **A célula cara é `Ns=36`, não `Na=8`** — ao contrário do que o `PLANO.md` supunha. Muda a prioridade da linha de corte |
| E.7 | Não há `reset()` no simulador | Um `Simulator` novo por episódio, com overhead de construção a medir |

---

## F. Metodologia estatística — uma ameaça que atravessa tudo

Três specs (02, 06, 10) chegaram ao mesmo ponto: **o IC bootstrap mede variância de cenário, não variância de treino.** Com uma única semente de treino por configuração, todo IC do projeto é condicional a um treino específico — e Lamenza relata **falhas catastróficas de treino** (§7.3, a curva de 4 agentes que não melhora). Uma corrida por configuração não distingue método ruim de semente ruim.

> **DECISÃO F.1 — quantas sementes de treino?** SPEC-06 propõe `R = 5` para permitir teste de permutação com `p < 0,01`; SPEC-10 propõe `R ≥ 3` nas células não-cortáveis. Isso multiplica o custo justamente onde ele é maior. Depende de P2 (orçamento de simulação, ainda não medido). **Se ficar em R=1, isso tem que aparecer em E2 como limitação, não em rodapé.**

Outros pontos de protocolo que julgo bem resolvidos pelas specs e só precisam de aval:

- **Métrica primária censurada** (imputação em 500 s, estilo PAR-1) em vez de média sobre sucessos — sem isso há viés de sobrevivência e "desistir cedo" vira vantagem. SPEC-02 e SPEC-06 convergiram nisso.
- **Como Lamenza agregou episódios sem sucesso** é indeterminável pelo texto e é o **maior risco de incomparabilidade numérica**. Mitigação: reportar três estatísticas para que ao menos uma case com a leitura dele.
- **Os tempos do §7.2 estão em figura, não em tabela** (SPEC-06). A única tabela numérica de tempo (7.3) é de outra campanha, com `α` diferente. Comparação rigorosa é por **razão contra o guloso reimplementado**, não por valor absoluto em segundos.
- **Lamenza não publica `Na = 1`** — nossa célula base da Etapa 1 não tem alvo externo; o primeiro ponto comparável é `Na = 2`.
- **A comparação WNN vs MLP mede aproximador + binarização juntos.** SPEC-06 propõe uma coluna **MLP-bin** (mesma MLP, mesmos bits) para separar os dois efeitos. É campanha de treino a mais; cortá-la precisa ser decisão consciente.

---

## G. Uma lacuna de capacidade que ninguém tinha visto

SPEC-04: com sorteio independente por RAM, `N·n = 576` bits lidos contra `B ≈ 3 200` bits de entrada em `T=128`. **A rede lê 18 % da entrada e ignora o resto silenciosamente** — possivelmente incluindo os bits de validade do zero-padding, que são justamente o mecanismo de desambiguação que projetamos.

Ou se exige `N ≥ B/n` (o que forçaria `N ≥ 268`, muito acima da varredura `{16, 48, 128}`), ou se aceita subamostrar e **reporta a fração de cobertura**.

Isso tem consequência direta na SPEC-01: **a varredura de `T` e a de `N` não são independentes**, e a spec hoje as trata como se fossem. Aumentar `T` sem aumentar `N` reduz a fração de entrada observada.

> **DECISÃO G.1** — a SPEC-01 precisa incluir cobertura de bits como métrica de primeira classe em T-01.4, e a varredura precisa ser sobre o par `(T, N)` com a razão `N·n/B` reportada. Corrigir antes de rodar.

---

## H. Métricas novas que as specs inventaram e que precisam de aval

As três lacunas que eu tinha deixado explicitamente em aberto foram desenvolvidas:

- **Busca ativa (H4)** — SPEC-08 propõe uma bateria (desvio angular do guloso-sobre-estimativa regredido na incerteza, intervenção `κ` nos canais de `P`, canal embaralhado como controle, informação por metro voado). Ponto central bem visto: toda estatística é lida como **diferença contra o degrau 4**, porque o ambiente gera essas correlações sozinho. Risco: é a parte mais autoral e pode estar sobredimensionada para 3 dias.
- **Divisão de trabalho (H6)** — SPEC-09 propõe disjunção de cobertura, alvo implícito por cone, redundância de coleta. SPEC-06 propõe o índice de sobreposição `Ω` para quantificar a fig. 7.1 vs 7.4 sem inspeção visual. **A definição do "alvo implícito" (cone de ±30°) é arbitrária e precisa ser fixada antes de olhar resultados**, senão a métrica é ajustável a posteriori.
- **Vazamento da verdade (RF-04)** — SPEC-07 propõe quatro camadas, e é honesta sobre os buracos: *taint* em `ndarray` não sobrevive a `float()`/`.item()`. A prova real é a camada 4, **diferencial/metamórfica**: gravar-e-repetir o fluxo de ruído, deslocar a verdade por δ só no caminho de telemetria, exigir hash idêntico da sequência de ações. Elegante — mas precisa ser verificado se é implementável sem reescrever a ponte da SPEC-00.

---

## I. Correções já aplicadas

| # | Erro | Onde |
|---|---|---|
| I.1 | "~2600 MACs de uma MLP 31→32→2" — número vinha da MLP 33→64→8 do PLANO §9.8, e a arquitetura citada não era a pareada. O valor correto é **2 080 MACs** para `31→32→32→2` | PRD §6.2, corrigido |
| I.2 | "zero MACs no caminho de inferência" é falso — a acumulação usa somas fp e a amostragem usa transcendentais. Alegação verificável: **zero multiplicações fp no aproximador** | PRD §5.5 e H1, corrigido |
| I.3 | SPEC-01 referenciava SPEC-05/SPEC-06 para Kalman e Etapa 2, numeração anterior à divisão final | SPEC-01, corrigido para 07/08 |
| I.4 | Critério de H1 ainda prometia vantagem de tamanho sem ressalva | PRD §2, corrigido com o alerta de A |
| I.5 | A contagem de operações não previa **divisões**, que num MCU sem FPU custam mais que MACs e que o KF usa | PRD §6.2, anotado |

---

## J. O que eu faria primeiro

1. **Resolver P1 e P5** no repositório público — bloqueiam a comparabilidade inteira e P5 eu não tinha visto.
2. **Decidir A.1** (reformulação de H1) — muda a manchete do trabalho e é melhor decidir agora que na semana 3.
3. **Decidir C.1 e C.2** (normal projetada; esquecimento vs gradiente funcional) — bloqueiam a SPEC-04, que bloqueia tudo do lado do modelo.
4. **Corrigir G.1** na SPEC-01 antes de ela rodar — cobertura de bits muda a varredura.
5. **Confirmar E.1–E.7** no ambiente real com o `gradysim` instalado.
6. **Decidir F.1** (sementes de treino) — depende de medir P2, que é tarefa da semana 1.
