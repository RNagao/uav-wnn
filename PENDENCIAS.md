# Pendências — registro único

Consolidação das ~90 perguntas em aberto espalhadas pelo PRD e pelas 11 specs. Este arquivo é a **fonte única de verdade** sobre o que está pendente; as seções "Perguntas em aberto" das specs são o detalhe, este é o índice com impacto.

> **Atualização (código do Katopodis).** `rkatopodis/RLWorktable.jl` + `rkatopodis/Ramnet.jl` (branch `faster_hashing`), MIT, são a implementação do próprio autor. **C2 fechada** — a pergunta estava mal formulada. **C1 confirmada como problema nosso**, com a solução encaixando nativamente na arquitetura dele. Ver a memória do projeto e SPEC-04/05 §0.
>
> **Atualização (SPEC-03).** **T1 fechada por medição, com resposta negativa** — a alegação de "modelo menor" cai e o argumento migra para a aritmética. Ver `reports/spec03-custo.md`.
>
> **Atualização.** Após a SPEC-00 e a revisão do modelo de ruído: **P1, P5, P6, P8, L1 e P2 fechadas** (código de referência e medição); **C5 resolvida** pelo novo modelo; **C4, T1, C1, C2, C3 seguem abertas**. Entram **P9** (parâmetros do canal do sensor) e **P10** (`σ_own`).

**Leitura rápida:** das ~90, **16 têm impacto real no projeto**. Destas, **5 estão no caminho crítico** e precisam de resposta na semana 1. O resto é local, já tem recomendação embutida na spec e pode seguir com o padrão proposto.

| Classe | Quantas | O que acontece se não for resolvida |
|---|---|---|
| **A — Bloqueiam a reprodução** | 6 | Não podemos afirmar que reproduzimos Lamenza; os números viram incomparáveis com os publicados |
| **B — Mudam a tese** | 2 | A manchete do trabalho está errada e será reescrita tarde |
| **C — Bloqueiam código** | 5 | A SPEC-04 não pode ser implementada, e ela bloqueia todo o lado do modelo |
| **D — Higiene estatística** | 3 | Os resultados saem, mas não sustentam a conclusão que se quer tirar deles |
| Local às specs | ~74 | Nada — têm recomendação e seguem no padrão |

---

## Classe A — Bloqueiam a reprodução (informação externa que falta)

Todas dependem do repositório público `Project-GrADyS/gradys-data-collection`, branch `dissertation`, ou do autor. **A latência de uma resposta humana é de dias**, então o pedido tem que sair no dia 1 e o trabalho segue parametrizado enquanto isso.

| ID | Pendência | Impacto se não resolvida | Plano B |
|---|---|---|---|
| **P1** | `communication_range` | **A maior.** Define quando o sensor é coletado, o atalho das pernas do TSP, o limite inferior LB-1, o disco das métricas de cobertura da Etapa 3 e a escala da leitura visual da fig. 7.4. Bloqueia 4 specs (00, 02, 06, 09) | Família de curvas sobre 2–3 valores candidatos; a diagonal da tab. 7.3 restringe a ordem de grandeza a dezenas de metros |
| **P5** | Período do heartbeat | **Tão crítico quanto P1 e eu não o tinha catalogado.** Se `v_max · heartbeat > communication_range`, o UAV atravessa a cobertura **sem coletar** e nosso ambiente fica mais difícil que o original — invisivelmente, sem aparecer na tabela de desvios | Fixar por regra `α · r / v_max` com α declarado |
| **P7** | `max_episode_length` "permissivo" do §7.1 | Sem ele, os **100 % de taxa de sucesso publicados não são alvo comparável** — e a réplica 1a é justamente o experimento cujo valor é ter um alvo externo | SPEC-06 propõe `3 × CT_p99` do guloso, piso 500 s, com sensibilidade em três horizontes. É invenção nossa e precisa ser declarada |
| **P8** | Tratamento de borda | O `gradysim` não limita posição e a eq. 5-6 não penaliza borda. Se Lamenza satura em [−100,100]² e nós não (ou o inverso), o ambiente diverge silenciosamente. A caminhada aleatória é **extremamente** sensível a isso; o guloso quase não é | Escolher, declarar na §5.4, e reportar o piso aleatório sob as duas regras |
| **P6** | Raio da área de largada dos UAVs | Afeta quanto a coordenação importa nos primeiros segundos. A fig. 7.1 mostra trajetórias praticamente sobrepostas na largada, o que é consequência direta desse raio ser pequeno | Valor declarado |
| **P4** | Arquitetura do ator do MADDPG | Sem ela, metade da razão dupla de tamanho (§5.5) é estimativa marcada. Se a comparação virar argumento central, **sobe de pendência a bloqueante** | Padrão do CleanRL (256×256), declarado como suposição |
| **L1** | Como Lamenza agregou episódios sem sucesso no "average completion time" da fig. 7.3 | **Indeterminável pelo texto** e é o maior risco de incomparabilidade numérica com os valores publicados | Reportar três estatísticas (penalizada, sobre sucessos, taxa de sucesso) para que ao menos uma case com a leitura dele. **Já resolvido pela SPEC-02** — fica só o registro no relatório |

**Impacto agregado da classe A:** se nada for resolvido, a Etapa 1 deixa de ser "reprodução" e vira "problema inspirado em Lamenza". A mitigação que preserva o valor do trabalho é comparar por **razão contra o guloso reimplementado**, não por valor absoluto em segundos — o que a SPEC-06 já adota como comparação rigorosa, porque os tempos do §7.2 estão em figura e a única tabela numérica (7.3) é de outra campanha com `α` diferente.

---

## Classe B — Mudam a tese do trabalho

| ID | Pendência | Impacto |
|---|---|---|
| ~~**T1**~~ | ~~"modelo menor" → "aproximador sem multiplicação"?~~ | **FECHADA POR MEDIÇÃO, com resposta negativa.** Não são os ~12–15 KB estimados: medido, a WNN é **133,9 KB** (chaves ordenadas) contra **7,6 KB** da MLP pareada — **17,5×** — e ainda 5× menos precisa na clonagem (5,6° contra 1,1°). A estimativa errou uma ordem de grandeza porque partia de um protótipo de **2 variáveis de entrada**; a observação real tem 25. A alegação de tamanho **cai**. Sobra, medido: **zero multiplicações em ponto flutuante no aproximador**, e vantagem de ciclos **condicional ao alvo** — 1,6× em x86, 4,1× em Cortex-M4F, **19,9× em Cortex-M0 sem FPU**. Ver `reports/spec03-custo.md` |
| **T4** | **A barra da H6 mudou.** O baseline da fig. 7.3 (`greedy`) **não coordena** — separação média entre UAVs de 0,1 m contra alcance de 20 m. A `smart` do mesmo `heuristics.py` coordena lendo só a observação e fica a **5–17 % do MADDPG publicado**, superando-o com 2 agentes. Uma heurística sem treino empata com 32 dias de GPU. Decidir: reformular a H6 em torno da `smart`, ou dividir em H6a (bater `greedy`) e H6b (bater `smart`)? Ver `reports/spec00-parametros.md` §9 |
| **T3** | **Qual é a física do ruído no sensor?** Decide se a H4 tem mecanismo. Se o sensor **transmite o próprio GPS**, o erro é dele, não depende da distância e é dominado por **viés fixo** — o canal vira piso irredutível e a busca ativa perde a causa. Se o drone **localiza o sensor por rádio**, o erro é branco e cresce com a distância, a filtragem tem função e aproximar-se compra informação — ao custo de um alcance de detecção maior que o de coleta, declarado na §5.4. **Bloqueia o bloco 2b; não bloqueia 2a** |
| **T2** | **Recalibrar o que a fusão pode prometer** | A observação é **relativa**, então `σ_rel = √(σ_a² + σ_s²)`. Se o levantamento dos sensores tiver ~5 m irredutíveis, o 18,8× medido no GPS vira **~2,2×** no que a política enxerga. O achado A3 é sobre a posição própria, não sobre a entrada da política |

**Impacto agregado:** nenhuma das duas mata o projeto. Ambas mudam o que o relatório pode afirmar. Adiar as duas é o cenário caro: significa descobrir na redação que a manchete não se sustenta.

---

## Classe C — Bloqueiam código (caminho crítico do modelo)

Estas travam a SPEC-04, que trava SPEC-05 → SPEC-06 → SPEC-08 → SPEC-09. **Profundidade 5 na cadeia de dependências** — é a corrente mais longa do projeto.

| ID | Pendência | Impacto | Recomendação |
|---|---|---|---|
| **C1** | **Como parametrizar θ, que é circular?** **Confirmado como problema NOSSO:** o código do Katopodis (`ContinousActionPolicy`) usa gaussiana isotrópica sem nenhum tratamento de wrap-around — para o Hopper, torques numa caixa, basta; para `θ ∈ [0,2π)` não | Determina se o ator emite 2 ou 3 saídas, o que realimenta T1 | **Normal projetada**, e ela **encaixa nativamente**: `FunctionalDiscriminator{AS}` já é multi-saída, basta `AS=3`. E a covariância dele é **recozida, não aprendida**, o que elimina o hiperparâmetro que era o risco desta opção. Falta só validar empiricamente |
| ~~**C2**~~ | ~~Esquecimento e gradiente funcional no mesmo módulo~~ | **FECHADA pelo código do autor.** A pergunta partia de premissa errada: a combinação **não existe**. São dois modelos para dois papéis — **ator** `FunctionalDiscriminator` (média simples, só valor, **sem γ**) e **crítico** `RegressionDiscriminator` (auto-normalizado por `Σ(1−γ^c)/(1−γ)`, valor **e** contador). O `NTupleRegressor` da SPEC-01 já é o ator. Consequência para T1: o crítico custa **o dobro por entrada** | — |
| **C3** | **Estrutura de armazenamento das LUTs** | Vale **2–5×** no número principal de T1. Hash aberto (premissa atual) vs chaves ordenadas + busca binária vs densa. A segunda pode ganhar em **tamanho e p99 ao mesmo tempo** | Medir as três antes de congelar. Se a ordenada vencer, a SPEC-04 muda de alvo |
| **C4** | **Cobertura de bits: `N·n` vs `B`** | Com `N·n = 576` contra `B ≈ 3 200` em `T=128`, a rede lê **18 % da entrada** e ignora o resto silenciosamente — possivelmente incluindo os bits de validade, que são o mecanismo de desambiguação que projetamos. **A varredura de `T` e a de `N` não são independentes**, e a SPEC-01 hoje as trata como se fossem | Incluir cobertura como métrica de primeira classe em T-01.4 e varrer o par `(T, N)` reportando `N·n/B`. **Corrigir antes de a SPEC-01 rodar** |
| **C5** | **O filtro de informação não tem o que fundir na Etapa 2** | Em Lamenza, ouvir um sensor equivale a coletá-lo: ele sai de `P^s` no mesmo instante, então não há múltiplas observações do mesmo sensor. Metade do desenho da Etapa 2 fica sem objeto | Aceitar o canal degenerado e deixar a fusão de sensores para a Etapa 3, onde ela existe de fato. Criar `detection_range > communication_range` seria **modificar o problema para caber no método** |

---

## Classe D — Higiene estatística

| ID | Pendência | Impacto |
|---|---|---|
| **E1** | **Quantas sementes de treino por configuração?** | O IC bootstrap mede variância de **cenário**, não de **treino**. Com `R = 1`, todo IC do projeto é condicional a um único treino — e Lamenza relata falhas catastróficas de treino (§7.3). Não se distingue método ruim de semente ruim. SPEC-06 pede `R = 5` para permutação com `p < 0,01`; SPEC-10 pede `R ≥ 3` nas células não-cortáveis. **Multiplica o custo onde ele já é maior.** Depende de medir P2 |
| **E2** | **Critério de consistência para estimador conservador (portão G3)** | A interseção de covariância fica **sistematicamente abaixo** do intervalo qui-quadrado — é a propriedade, não o defeito. Com o critério simétrico que escrevi, **6c reprova por funcionar como projetado**. Precisa de critério assimétrico (rejeita otimismo, aceita conservadorismo) com `κ` reportado |
| **E3** | **Coluna MLP-bin** | A comparação WNN vs MLP mede **aproximador + binarização juntos**, não o aproximador. Uma coluna MLP-bin (mesma MLP, mesmos bits) separa os dois efeitos. É campanha de treino a mais — cortá-la precisa ser decisão consciente, não omissão |

---

## Parâmetros nossos que ainda não têm valor

Não são "pendências de informação externa", são decisões de projeto que ninguém tomou:

| ID | Parâmetro | Por que importa |
|---|---|---|
| **P9** | Parâmetros do canal do sensor: `σ_b` (viés fixo), `σ_0` e `κ` de `R(d) = (σ_0 + κ·d)²` | Agora é o canal **primário**. `σ_b` é o piso irredutível do filtro; `κ` é o que faz aproximar-se ser informativo e portanto o que sustenta a H4. Faixas a fixar na SPEC-07 |
| ~~**P10**~~ | ~~`σ_own` — valor fixo do GPS próprio~~ | **RESOLVIDA por construção.** O bloco 2a passa a **varrer** `σ_own` e a medir qual valor usar em 2b, em vez de fixá-lo por decreto |
| **N1** | `dt_gps` — taxa do GPS | **Escolher `dt_gps` é escolher o resultado.** O achado A3 (18,8×) foi medido com decisão a 0,5 s; com `algorithm_iteration_interval = 2 s` e GPS a 1 Hz são só 2 correções por decisão |
| **N2** | Mobilidade com ou sem inércia na Etapa 2 | Sem inércia o modelo de processo é exato, `Q` degenera e o degrau 4 fica quase trivialmente correto — o experimento perde valor. Com inércia, `Q` vira mensurável |
| **N3** | Orçamento de treino `S₂` da Etapa 2 | Se a Etapa 1 mostrar melhora até `10⁷` passos e a Etapa 2 rodar `2·10⁶`, comparam-se políticas subtreinadas |
| **P2** | Orçamento real de simulação | **Ainda é estimativa, não medição.** Dele dependem E1 (sementes), N3 e o dimensionamento de todas as varreduras |

---

## Caminho crítico — o que decidir na semana 1

Por ordem de quanto destrava:

1. **P1 e P5** — pedido ao repositório/autor no dia 1. Bloqueiam 4 specs e a latência é externa.
2. **C1 e C2** — bloqueiam a SPEC-04, que está na corrente de profundidade 5. São decisões técnicas que eu já recomendei; precisam só de aval.
3. **C4** — corrigir a SPEC-01 **antes** de ela rodar, senão o estudo de binarização mede a coisa errada.
4. **T1** — decidir a reformulação de H1 antes de a semana 2 gerar números sob a premissa antiga.
5. **P2** — medir na semana 1; dele saem E1 e N3.

**As quatro últimas (C1, C2, C4, T1) eu posso destravar com uma decisão sua de uma linha cada.** P1, P5 e P2 dependem de ação no mundo.

---

## O que NÃO precisa de decisão agora

As ~74 perguntas restantes têm recomendação embutida na spec correspondente e seguem no padrão proposto se ninguém disser nada. Exemplos: `B = 1000` vs `10 000` reamostragens; se o k-means recebe passo de melhoria de *makespan*; se a fig. 7.3 é digitalizada como linha de referência; idioma do relatório; formato-fonte dos slides. Estão registradas para não se perderem, não para travar o começo.

Duas exceções que **parecem** locais e não são:

- **SPEC-09 Q7** — cortar a Etapa 3 para `Na = 2` está na linha de corte do PRD como item 6. Mas com um único vizinho perdem-se o equilíbrio de carga, a dependência de ordem da CI e o achado de retornos decrescentes de Lamenza. **Deveria ser a última opção, não a sexta.**
- **SPEC-09 Q4** — o gatilho para ligar o degrau 5 na Etapa 3 depende do resultado do G2, então não pode ser decidido agora. Mas **precisa ser acordado antes**, senão vira decisão pós-hoc.

---

## Adendo — 2026-08-15: a fonte primária é a dissertação, não o código

Com a leitura de `tmp/947841_compressed.pdf` (Katopodis, *Reinforcement
Learning with Weightless Neural Networks*, PESC/COPPE-UFRJ, jan/2022) a
hierarquia de fontes fica clara, e ela **reordena** algumas pendências.

Os algoritmos da dissertação são 4.1 (SARSA com esquecimento), 4.3 (REINFORCE) e
**4.4 (Actor-Critic)**. **PPO aparece uma única vez em 87 páginas**, no capítulo
6 de trabalhos futuros (p. 70), como sugestão para corrigir a instabilidade de
passo fixo do SGD. Logo:

- **`rkatopodis/RLWorktable.jl` + `Ramnet.jl` = a dissertação implementada.**
- **`anbubus/ntn_neurocomputing_v0` (PyTorch + Ray RLlib PPO) = o capítulo 6.**

### Fechadas

- ~~**C2** — esquecimento com gradiente funcional.~~ São **duas redes**
  (§4.2.1 crítico, §4.3.1 ator). Implementadas em `wnn/redes.py` e **idênticas
  bit a bit** ao `Ramnet.jl` em 5 configurações.
- ~~**C4** — cobertura de bits.~~ Fechada **pela estrutura**: o mapeamento por
  partição (`:uniform_random`) lê cada bit exatamente uma vez.

### Nova

- **P10 — Σ é covariância ou desvio-padrão?** A eq. (4.34) diz *"predefined
  **covariance** matrix"*, logo a amostragem é `√Σ·z`. O
  `ContinousActionPolicy.jl` amostra `Σ·z` mas usa `inv(Σ)` no gradiente —
  trata o mesmo número como desvio num lugar e variância no outro. **Seguimos o
  artigo**; `escala_como_desvio=True` reproduz o Julia. Com Σ = 0,25 a
  diferença é fator 2 na exploração, então precisa aparecer na tabela de
  hiperparâmetros. *Resolvida em recomendação; falta aval.*

- **P11 — o regime de `n` foi escolhido pelo problema ou pelo armazenamento?**
  O apêndice A da dissertação usa `n ∈ {16, 32, 64}`. Nós escolhemos `n = 12`
  (SPEC-01) e achamos `n = 10` dominante (SPEC-03) — **mas as duas rodaram
  sobre armazenamento denso**, que aloca `2ⁿ` por RAM e torna `n = 32` caro e
  `n = 64` impossível. Agora que as redes esparsas alcançam 64, isso é
  verificável. **Revarrer `n ∈ {8, 12, 16, 24, 32}` antes de fixar a Etapa 1**,
  porque muda a configuração de referência de todas as specs seguintes. *Aberta,
  e é a de maior impacto no momento.*

### Reordenada

- **C1 — ação circular.** Segue aberta e segue nossa: a dissertação usa
  gaussiana isotrópica sem tratamento de wrap-around. O `CircularEncoder` do
  repositório PyTorch resolve **entrada** circular, não **ação** — não fecha
  C1, mas vale portar (~15 linhas) porque as tabelas do apêndice A usam
  codificação circular em CartPole e Hopper, e nós não temos nada equivalente.

### Achado que afeta a SPEC-06

A instabilidade de passo fixo é **mensurável**: no bandido contextual, o pior
erro entre 5 sementes cresce monotonicamente com α (0,044 em α=0,002 contra
0,238 em α=0,05), e em 0,05 uma semente diverge. A busca de α precisa de faixa
larga e várias sementes. Se o Algoritmo 4.4 não passar o portão G1, a
instabilidade é a primeira suspeita — e a saída já está nomeada pelo autor.
