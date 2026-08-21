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

> **Namespace, para evitar uma confusão real.** `T1`, `T2` e `T3` são as
> **figuras de mérito de custo** do PRD §5.5 — tamanho, latência e contagem de
> operações — e aparecem no código, em `src/uavwnn/custo/`. As pendências da
> Classe B usam o prefixo **`TS`** (de *tese*) justamente para não colidir: antes
> desta correção, "T3 fechada por decisão" (pendência) convivia com "T3
> sobrevive" (métrica medida na SPEC-03) no mesmo repositório.

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
| **TS4** | **A barra da H6 mudou.** O baseline da fig. 7.3 (`greedy`) **não coordena** — separação média entre UAVs de 0,1 m contra alcance de 20 m. A `smart` do mesmo `heuristics.py` coordena lendo só a observação e fica a **5–17 % do MADDPG publicado**, superando-o com 2 agentes. Uma heurística sem treino empata com 32 dias de GPU. Decidir: reformular a H6 em torno da `smart`, ou dividir em H6a (bater `greedy`) e H6b (bater `smart`)? Ver `reports/spec00-parametros.md` §9 |
| **TS3** | **Qual é a física do ruído no sensor?** Decide se a H4 tem mecanismo. Se o sensor **transmite o próprio GPS**, o erro é dele, não depende da distância e é dominado por **viés fixo** — o canal vira piso irredutível e a busca ativa perde a causa. Se o drone **localiza o sensor por rádio**, o erro é branco e cresce com a distância, a filtragem tem função e aproximar-se compra informação — ao custo de um alcance de detecção maior que o de coleta, declarado na §5.4. **Bloqueia o bloco 2b; não bloqueia 2a** |
| **TS2** | **Recalibrar o que a fusão pode prometer** | A observação é **relativa**, então `σ_rel = √(σ_a² + σ_s²)`. Se o levantamento dos sensores tiver ~5 m irredutíveis, o 18,8× medido no GPS vira **~2,2×** no que a política enxerga. O achado A3 é sobre a posição própria, não sobre a entrada da política |

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

- **P13 — Σ é covariância ou desvio-padrão?** A eq. (4.34) diz *"predefined
  **covariance** matrix"*, logo a amostragem é `√Σ·z`. O
  `ContinousActionPolicy.jl` amostra `Σ·z` mas usa `inv(Σ)` no gradiente —
  trata o mesmo número como desvio num lugar e variância no outro. **Seguimos o
  artigo**; `escala_como_desvio=True` reproduz o Julia. Com Σ = 0,25 a
  diferença é fator 2 na exploração, então precisa aparecer na tabela de
  hiperparâmetros. *Resolvida em recomendação; falta aval.*

- **P14 — o regime de `n` foi escolhido pelo problema ou pelo armazenamento?**
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

### Correção — P14 era menos grave do que eu disse (2026-08-15)

Registrei a P14 como "a de maior impacto no momento", supondo que o nosso
`n = 12` fosse artefato do armazenamento denso diante dos `n ∈ {16, 32, 64}` do
apêndice A. **Ao normalizar por `B` isso não se sustenta.**

As dimensões de observação estão no §5.1 da dissertação (CartPole 4, Lunar
Lander 8, Mountain Car 2, Hopper 12), e `B = dim × resolução`:

| tarefa | `B` | `n` | `n/B` | `N` |
|---|---|---|---|---|
| CartPole | 512 | 64 | 0,125 | 8 |
| Lunar Lander | 1024 | 64 | 0,062 | 16 |
| Mountain Car | 256 | 32 | 0,125 | 8 |
| Hopper | 384 | 16 | 0,042 | 24 |
| **nosso** | **200** | **12** | **0,060** | **17** |

**Estamos dentro da faixa do artigo em `n/B` e em `N`**, praticamente sobre o
Lunar Lander. O que governa a generalização é a razão, não `n` sozinho — e os
testes de §3.2.4 em `tests/test_identidades_artigo.py` mostram por quê: a
fração de consultas sem resposta cresce monotonicamente com a tupla **relativa
à entrada**.

A diferença real é a **decomposição**: o artigo usa resolução alta (32–128) com
tupla grande; nós, resolução baixa (`T = 8`) com tupla pequena. Chegamos ao
mesmo regime por outro caminho.

**P14 rebaixada de bloqueio para refinamento.** A varredura continua valendo,
mas o eixo é `n/B`, e o achado esperado é ajuste fino, não correção de erro.

---

## Adendo — 2026-08-15 (b): corte de escopo, o ruído só no GPS do drone (D13)

O canal do sensor saiu do escopo. Isto **fecha três pendências sem resolvê-las**
e **retira uma hipótese**.

### Fechadas por decisão, não por resposta

| ID | era | por que fecha |
|---|---|---|
| **TS3** | Qual é a física do ruído no sensor — GPS próprio (viés fixo dominante) ou localização por rádio (branco, crescente com a distância)? | Sem o canal, a pergunta não tem consequência. **É o primeiro item a reabrir se o escopo crescer**, porque a resposta determina se a H4 tem mecanismo |
| **P3** | O sensor transmite posição estimada (linear → filtro de informação) ou distância/RSSI (não linear → EKF/UKF)? | Idem. Fica emparelhada com a T3 |
| **P9** | Parâmetros do canal do sensor: `σ_b`, `σ_0` e `κ` de `R(d) = (σ_0 + κ·d)²` | Não há canal a parametrizar |

Também cai a **C5** como preocupação ativa (o filtro de informação não tinha o
que fundir porque ouvir = coletar): o mecanismo que a contornava — reinterpretar
o canal global — sai junto com o canal.

### Hipótese retirada

**H4** — *"uma política que enxerga a própria covariância aprende busca ativa"*.
Não é adiamento, é retirada: o mecanismo era `R(d)` crescente com a distância,
que faz **aproximar-se comprar informação**. Com ruído só na posição própria a
covariância converge para regime estacionário e não existe incerteza *por sensor*
a reduzir. **A hipótese perde a causa física, não só o experimento.**

O degrau 5 da escada (política que enxerga o próprio `P`) permanece como
**ablação sem hipótese forte associada** — a pergunta fica sendo "ajuda?" em vez
de "produz busca ativa?".

### O que o corte NÃO toca

O conteúdo de Fusão de Dados estava em três lugares e dois permanecem:

- **KF com entrada de controle** (Etapa 2) — onde vive o achado medido A3, `12,54
  m → 0,67 m`, 18,8×. É o único canal cujo modelo estava fechado.
- **Interseção de covariância entre UAVs** (Etapa 3, **H5**) — funde as
  estimativas de posição **própria** que os vizinhos transmitem
  (`⟨k, t, x̂_k, P_k⟩`), construída sobre o degrau 4. Indiferente ao canal do
  sensor.
- **NEES/NIS** como validação de consistência.

Nota: se o corte tivesse sido na outra metade — tirar o GPS próprio e manter o
sensor —, a **H5 ficaria sem objeto**, porque os drones transmitiriam posições
exatas e não haveria erro correlacionado a fundir. O corte foi na metade certa.

### Renumeração

Ao registrar as pendências novas da rodada anterior eu reutilizei o **P10**, que
já estava em uso (`σ_own`, fechada por construção). Corrigido:

| era | agora | assunto |
|---|---|---|
| P10 | **P13** | `Σ` é covariância ou desvio-padrão (artigo vs código do autor) |
| P11 | **P14** | revarrer o regime de tupla no eixo `n/B` |
| P12 | **P15** | replicar §5.2.2 com orçamento cheio |

- **P15 — replicar §5.2.2 com orçamento cheio?** Nossa replicação parcial dá
  **empate** na configuração que o artigo declara para CartPole (`R=128, n=64`:
  0,96×, IC cruza zero) e **vitória do circular** em baixa resolução (`R=32,
  n=16`: 1,68×, IC95% [+39,6; +56,1], 20 de 20 sementes). Orçamento ~1/50 do
  original. **Não é bloqueio** — o codificador de produção é o quantil, por
  outras razões —, mas se o relatório citar §5.2.2 como fundamento, a ressalva
  precisa vir junto. Ver `reports/replica-katopodis-52.md`.

---

## Adendo — 2026-08-20: P4 fechada por medição, e a D7 se estreita

**O submodule já tinha o que faltava.** `vendor/gradys-data-collection` traz, além
do cenário, o **código de treino do MADDPG** (`main.py`, base CleanRL), **9
checkpoints treinados** (`Na ∈ {2,4,8} × Ns ∈ {12,24,36}`, 45 MB) e o **harness
de avaliação** (`test.py`). Nada disso precisava ser treinado.

### ~~P4~~ — arquitetura do ator do MADDPG. **FECHADA POR MEDIÇÃO.**

| `Na` | obs | params | ator fp32 |
|---|---|---|---|
| 2 | 27 | 73 474 | 287,0 KB |
| 4 | 31 | 74 498 | 291,0 KB |
| 8 | 39 | 76 546 | 299,0 KB |

`obs → 256 → 256 → 2`, ReLU, `tanh` na saída — **exatamente o default do CleanRL
que a estimativa supunha**. A estimativa registrada era "≈74 000 params, ≈298 KB";
o palpite estava certo. A razão da §5.5 (`0,46×` o ator do MADDPG) deixa de sair
marcada como estimativa.

Duas validações cruzadas de graça: a fórmula `D = 2(Na−1) + 25` do PRD §5.2
confere com as formas de `fc1.weight`, e `obs_dim` não muda com `Ns`, confirmando
`N*_s = 12` fixo com zero-padding.

### D7 estreitada, não derrubada

A justificativa original — *"reproduzi-lo exigiria replicar exatamente simulador,
observação e recompensa; qualquer divergência invalidaria a comparação
silenciosamente"* — **não se aplica a carregar os pesos dele**. Não há nada a
replicar: são os pesos dele, no ambiente dele, que é o nosso submodule.

O que continua valendo: **não retreinamos**. Isso custaria ~7 dias de GPU por
célula, pelo número da própria dissertação.

### Sobe de prioridade: T-03.6 (paridade de arcabouço)

A latência do ator do MADDPG agora é mensurável (**35 a 36 µs** de mediana, 1
thread, batch 1, fp32). Mas é PyTorch em CPU, e a nossa WNN é NumPy — medir tempo
de parede favorece quem tem a biblioteca mais otimizada, não o modelo mais barato.
**A comparação independente de arcabouço é a contagem de operações (T3)**, e o
contraste que se sustenta é estrutural: ~73 000 multiplicações em ponto flutuante
por decisão contra **zero** no nosso aproximador.

### Nova, e é limitação de escopo, não pendência de decisão

- **Não há checkpoint para `Na = 1`.** As células publicadas começam em 2 agentes,
  e a Etapa 1 começa em `Na = 1`. Ali **não existe baseline MADDPG** — as
  referências são a heurística e a MLP pareada. Precisa aparecer no relatório;
  não há o que decidir.


---

## Adendo — 2026-08-20 (b): SPEC-02 concluída, o bloqueio estatístico caiu

**O teste A/A está calibrado.** Cobertura 95,5 % (sintética) e 93,5 % (campanha
MADDPG), com o IC de Wilson contendo 95 % nas duas. O contra-exemplo —
reamostragem independente — falha como deve, com IC **5,2× mais largo**.

Consequência retroativa: **os ICs já reportados se sustentam.** Isso cobre a
validação 9/9 do guloso e o `greedy` vs `smart` (SPEC-00), a comparação de custo
(SPEC-03), a replicação de §5.2.2 (T-B2) e a validação 3/3 do MADDPG.

### Achado novo que reforça a TS4

O guloso **não escala com `Na`**: de 1 para 4 drones acelera **1,10×**, contra
3,32× do planejador. Quadruplicar o enxame melhora o baseline em 9,5 %.

O mecanismo está medido: coletas redundantes saltam de 0,0 (`Na=1`) para **10,9**
(`Na=2` e `Na=4`), e o Gini das visitas **piora** com mais agentes (0,469 →
0,707). A TS4 dizia que a barra da H6 era baixa porque o baseline não coordena;
agora se sabe também que **acrescentar drones a ele quase não ajuda**.

### Achado sobre o problema, não sobre o método

Em `Na = 1` não há particionamento, então o `kmeans_tsp` degenera em ordenação
ótima de rota. Isso separa as fontes de ganho:

| fonte | ganho sobre o guloso |
|---|---|
| roteamento só (`Na = 1`) | **8,1 %** |
| roteamento + particionamento (`Na = 4`) | **69,4 %** |

**O gargalo da coleta por enxame é *quem vai onde*, não *em que ordem*.** Vale
para a leitura da Etapa 1: se a nossa política aprender particionamento, ganha
quase tudo; se aprender só roteamento, ganha 8 %.

### Não feito, e registrado como tal

**T-02.9 §7 — sensibilidade a `communication_range`.** O motivo original era
decidir entre candidatos da P1, que foi fechada na SPEC-00 com o valor lido do
código (20 m). Continua valendo como análise de robustez.

---

## Adendo — 2026-08-21: SPEC-05 fechada, C1 resolvida por medição

### ~~C1~~ — ação circular. **FECHADA.**

A pendência mais antiga da classe C. Estava marcada como *"falta só validar
empiricamente"*, e a validação aconteceu — com o critério escrito para **falsificar
a nossa própria decisão**:

```
C2 (normal projetada) − C0 (gaussiana simples), erro angular, pareado por alvo
    Δ = −13,45°    IC95 % [−14,53; −12,43]    MDE 1,06°    n = 400
    vitória do C2 em 98,5 % dos alvos      Wilcoxon p = 4,4·10⁻⁶⁷
```

O IC está a 12 MDEs de zero, e o C0 recebeu **quatro concessões** contra
espantalho: varredura própria de `σ` (que significa coisas diferentes nos dois
esquemas), varredura própria de taxa, variante centrada em 0,5 (que foi de fato a
melhor dele e é a que entrou na comparação), e `d = 2` contra `d = 3`.

**O mecanismo, medido em isolamento.** No bandido de alvo fixo o C0 resolve
*apenas* o alvo em 0° — e a sua curva de recompensa é **exatamente `cos(alvo)`**,
isto é, ele aponta sempre para 0° e recebe o que a sorte do alvo der. Com `f₁`
abaixo de zero com margem, **toda** amostra recorta na mesma ação, a recompensa
deixa de variar, `(r − b) → 0` e o gradiente morre. É zona absorvente: não é
convergência lenta, é ausência de sinal, e nenhuma quantidade de episódios sai de
lá.

Consequência a jusante já propagada: `d = 3` para os dois aproximadores, o que
corrige o PRD §5.5 (arquitetura pareada `D→32→32→3`) e triplica a carga útil por
posição de memória da WNN.

### Nova — TS5: crédito temporal no brinquedo sequencial

`toy/nav.py` não é resolvido por nenhum dos dois braços (0–8 % de sucesso), embora
a política ótima analítica faça **7,86 passos com 100 % de sucesso**. Descartados
por medição: anelamento de `σ` (persiste com `σ` fixa), recompensa esparsa
(persiste com termo denso de progresso), e a maquinaria de atualização (o bandido
com a mesma maquinaria aprende perfeitamente).

Resta crédito temporal. **Não bloqueia a SPEC-06** — o portão foi decidido no
bandido contextual, onde os dois braços aprendem —, mas fica registrado porque o
ambiente real tem horizonte muito maior (250 decisões) e, se o modo de falha
transferir, aparecerá lá como "não aprende" sem diagnóstico pronto.

### Nova — TS6: invariância à ordem é inatingível com esquecimento

O critério de saída 7 da SPEC-05 pede tabelas idênticas "inclusive com `Na > 1` e
**ordens de evento diferentes**". Medido: impossível para o crítico, por aritmética.
`v ← φ·v + alvo` não comuta, e a diferença entre duas ordens é exatamente
`(φ−1)(a₁−a₂)`.

E **a colisão de endereço é o regime normal**: com `Na = 4` e `R = 1`, só 18 % das
RAMs leem bit de índice, então nas outras 82 % dois agentes com a mesma observação
posicional escrevem na mesma posição.

O que se entrega é **determinismo** com ordem canônica (índice crescente), que é o
que `make repro` precisa. Invariância exigiria acumular alvos por endereço dentro
do tique e aplicar `φ` uma vez só — desvio da eq. (4.8), não detalhe de
implementação. Decisão a tomar se a SPEC-08/09 precisar de replay paralelo.

### Perguntas do artigo resolvidas de passagem

- **Variância do Hopper (0,2 ou 0,3):** a tabela A.3 diz **0,3**. A inconsistência
  é entre o texto da §5.3 e o apêndice, não dentro do apêndice. Qual reproduz o
  número publicado só se decide rodando o Hopper (MuJoCo, fora de escopo).
- **Cronograma do anelamento de `σ_s`:** **não há anelamento.** A primeira
  tentativa tinha um bug — `decaimento` conta transições, não episódios, e num
  brinquedo de 40 passos colapsava `σ` em ~20 episódios. Desligado, porque a
  avaliação já é determinística (Lamenza §7).
- **Cobertura do canal de identidade:** a fórmula `p = 1 − (1 − R·Na/B)^n` da
  SPEC-05 §4.7 supõe amostragem **com reposição** por RAM — a mesma suposição que
  gerou o achado C4. Com mapeamento particionado a conta é **hipergeométrica**.
  Medida, ela bate melhor em 5 das 6 configurações; a binomial fica como
  aproximação, não identidade.

### Correção de redação, não de fato

O critério de saída 5 ("nenhuma dependência de PyTorch fora de `tests/`") é violado
por construção: `scripts/spec02_maddpg_*.py` importam `torch` porque é o que lê os
pesos publicados de Lamenza. O que a RNF-11 protege é **`src/uavwnn/`**, que está
limpo e agora tem teste. O critério estava escrito mais forte do que se pretende.

### ~~T-05.4(a) fechada com discrepância~~ — RETRATADO, ver abaixo

A reprodução do Katopodis no `MountainCarContinuous-v0` rodou com o orçamento
publicado (2000 episódios × 10 sementes, §5.4 p.65):

| | medido | publicado |
|---|---|---|
| AC com MLP (controle negativo) | −0,0370 [−0,0393; −0,0347] | **−0,02** (−0,03; −0,02) |
| n-tuple AC | −0,1989 [−0,2124; −0,1842] | **93,3** (92,9; 93,8) |

O controle negativo reproduz o fenômeno; o n-tuple cai no **mesmo ótimo local de
não mover o carro** que o artigo atribui à MLP, com **0,00 % de chegada à bandeira**
em 1000 episódios de avaliação.

Três erros nossos corrigidos no caminho: retornos de treino em vez de avaliação
determinística (a MLP foi de −31,86 para −0,04 com a correção), sinal trocado no
alvo publicado da nossa própria tabela (`0,02` em vez de `−0,02`), e orçamento a
37 % do publicado.

**O que a T-05.4(a) deixa de garantir, e é o motivo de isto virar pendência.** O
propósito escrito do critério era *"sem isto, nenhum resultado negativo da SPEC-06 é
atribuível ao problema"*. Continua garantido: redes bit-idênticas ao `Ramnet.jl`,
codificador com semântica de resolução provada, mapeamento por partição provado,
protocolo de avaliação correto, e capacidade/generalização eliminada por varredura
de `n ∈ {8,16,32,64}`. **Não** garantido: que o laço do Algoritmo 4.4 atribua
crédito corretamente sobre horizonte longo e recompensa quase esparsa — e ele é o
único componente da cadeia **sem oráculo diferencial**.

Se a SPEC-06 der resultado negativo no braço n-tuple, **a atribuição fica em aberto
e o primeiro suspeito é o laço.**

### Nova — TS7: os dois braços do `spec05_mountaincar.py` usam laços diferentes

Viola o *"a única diferença é `f(·)`"* da §4.8: o n-tuple usa `AtorCriticoNTupla`
(Algoritmo 4.4, retorno de `m` passos por transição) e a MLP um laço próprio com um
único ponto de truncamento por episódio.

Consequência que importa: **a MLP reproduzir não valida o laço**, porque o alvo que
ela reproduz é um ótimo local degenerado que quase qualquer esquema de crédito
encontra. Unificar antes de qualquer nova rodada; até então, só a comparação de cada
braço com o **seu** número publicado é válida, nunca a comparação entre eles.

### Retratação — a varredura de tupla não responde a pergunta aberta 2

Quase registrei que ela respondia. O retorno melhora monotonicamente com tupla maior
(−1,73 → −0,94 → −0,13 → −0,01) e o **sinal** coincide com o achado da §5.3 do
Katopodis. Mas zero é o teto do "não mover o carro": convergir para zero é assentar
no ótimo local com mais precisão, não resolver. A chegada em avaliação é 0,00 % nos
quatro pontos, e a de treino não sobe com `n`.

A recomendação de incluir `n = 16` na varredura da SPEC-06 **continua válida**, mas
pelo argumento de fidelidade à referência — Katopodis usa `n ∈ {16,32,64}` em tarefa
contínua e o PRD varre `{8,10,12}` — e **não** por esta medição.

---

## Adendo — 2026-08-21 (b): T-05.4(a) reproduz; a "falha" era bug nosso

**Retrata o adendo anterior por inteiro.** O n-tuple **resolve o Mountain Car**:

| | antes | depois da correção | publicado |
|---|---|---|---|
| retorno (avaliação) | −0,199 | **96,72** [96,70; 96,75] | 93,3 (92,9; 93,8) |
| chegada à bandeira | 0,00 % | **100,0 %** (10/10 sementes) | — |
| MLP (controle negativo) | −0,037 | −0,04 | −0,02 (−0,03; −0,02) |

O excesso é **para cima**: gastamos 3,28 de energia contra 6,70 dele. A explicação
econômica é que o número dele inclui ~3 % de episódios que não chegam — a conta é
robusta ao valor assumido para a falha (96,5 % a 97,5 % de chegada implicada), mas
não é verificável porque ele não publica a taxa.

### O bug, e por que ele passou tanto tempo escondido

`AtorCriticoNTupla.reiniciar()` **apaga as duas redes**. O laço de treino do
`spec05_mountaincar.py` a chamava a **cada episódio**, acreditando que só esvaziava a
janela de `m` passos. Foram 2000 episódios independentes, cada um de rede em branco.

O que torna isso perigoso: o resultado — retorno ≈ 0, o ótimo local de "não mover o
carro" — **é indistinguível de "o método não funciona"**, e é um ótimo local que o
próprio artigo descreve para a MLP. A falha imitava um resultado plausível.

O sintoma que denunciou não foi métrica de RL nenhuma: foi a **ocupação de memória
oscilando** (188 → 722 → 283 → …) numa estrutura *append-only*, que não pode
encolher.

Corrigido: `novo_episodio()` para a operação de início de episódio, aviso na
docstring do `reiniciar()`, e **teste de regressão** que exige ocupação
monotonicamente crescente entre episódios — mais o par que fixa o contrato
destrutivo do `reiniciar` para ninguém "consertá-lo".

### Retratações

- **~~TS5~~ pode não ser o que eu disse.** Registrei que `toy/nav.py` falha por
  crédito temporal nos dois braços. Aquele brinquedo usa laço inline do
  `spec05_circular.py`, não o `AtorCriticoNTupla`, então **não** compartilha este
  bug — mas a lição se aplica: antes de concluir "crédito temporal", auditar o
  ponto de chamada. Fica **reaberta para reexame**, não fechada.
- **A varredura de tamanho de tupla está inválida.** Rodou com o bug, medindo
  aprendizado de um episódio só. A monotonia observada era convergência ao ótimo
  local dentro do episódio. Refeita.
- **"O primeiro suspeito é o laço do Algoritmo 4.4" está retirado.** A lista de
  eliminação estava correta — protocolo, hiperparâmetros, codificador, mapeamento,
  orçamento, redes —, mas faltava nela **o nosso próprio código de chamada**, que
  não é componente do método e por isso não estava sendo auditado.

### O que a T-05.4(a) agora entrega

O propósito escrito era *"sem isto, nenhum resultado negativo da SPEC-06 é
atribuível ao problema"*. **Agora entrega.** O laço continua sem oráculo diferencial
(`PLANO-TESTES.md` §5, item 1), mas atravessa uma validação de ponta a ponta contra
número publicado — evidência muito mais forte que inspeção de código.

**TS7 continua aberta:** os dois braços do `spec05_mountaincar.py` usam laços
diferentes, violando a §4.8. Não invalida a reprodução (cada braço é comparado ao
**seu** número publicado), mas invalida comparações **entre** os braços.

### A lição, que é a da SPEC-03 outra vez

Antes de atribuir uma discrepância ao método, esgotar as hipóteses sobre o arnês —
**e o arnês inclui o código de chamada, não só as bibliotecas.** Auditei sete
componentes do método com evidência documental, extraindo provas do PDF, e o erro
estava numa linha do meu laço de treino.
