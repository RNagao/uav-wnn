# T-05.4(a) — reprodução do Katopodis no `MountainCarContinuous-v0`

**Veredito: um dos dois alvos reproduz o fenômeno; o outro não, e a discrepância
está diagnosticada.** É o que o critério de saída 1 admite — *"os quatro números
dentro dos ICs publicados **ou** com discrepância explicada e registrada"* — mas
vale dizer de saída o que isso custa: **um resultado negativo da SPEC-06 no braço
n-tuple não será atribuível ao problema com confiança total.** Era exatamente essa
atribuição que a T-05.4(a) existia para garantir.

## Resultado, com o orçamento publicado (2000 episódios × 10 sementes)

| | medido | publicado | |
|---|---|---|---|
| **AC com MLP** | −0,0370 [−0,0393; −0,0347] | **−0,02** (−0,03; −0,02) | erra por 0,005 |
| **n-tuple AC** | −0,1989 [−0,2124; −0,1842] | **93,3** (92,9; 93,8) | falha total |

Taxa de chegada à bandeira: **0,34 %** no treino do n-tuple, **0,00 %** na
avaliação determinística (1000 episódios). O n-tuple cai no **mesmo ótimo local de
não mover o carro** que o artigo atribui à MLP.

O controle negativo reproduz o **fenômeno** com desvio de 0,005 numa escala em que
o efeito discriminante é 93,3. Formalmente os ICs não se sobrepõem — são
adjacentes, e o nosso é ~2× mais negativo —, mas a leitura qualitativa do artigo
(*"optimized their agents to a local optimum of not moving the car"*) é reproduzida
sem ambiguidade.

## Três erros meus, corrigidos antes de o resultado valer algo

### 1. Eu reportava retornos de treino, com ruído ligado

A regra oposta está escrita na §4.4 desta spec e eu mesmo a codifiquei em
`politica_continua.py` — *"avaliação é determinística (Lamenza §7): usa-se a
média"* — e a violei neste script. A conta fecha exatamente: com `Σ = 0,4` a MLP
tem `σ = 0,632`, logo `E[a²] = 0,4`, e o custo de ação sozinho vale
`−0,1 · 0,4 · 999 = −39,96` por episódio.

| | com ruído (errado) | sem ruído (correto) | publicado |
|---|---|---|---|
| MLP | −31,86 | **−0,04** | −0,02 |
| n-tuple | −36,56 | −0,20 | 93,3 |

**A primeira "discrepância" era do meu protocolo, não do algoritmo.** O número
publicado é inatingível sob ruído: ele exige média ≈ 0 **e** ruído removido.

E o protocolo sai do próprio artigo, por dedução em vez de citação direta (p.66):

> *"the MLP methods having optimized their agents to a local optimum of not moving
> the car. In this way, they **never incur the penalty of expending energy**"*.

Um número que incluísse exploração **não pode** "never incur the penalty" — logo o
reportado é sem ruído.

### 2. A tabela de alvos desta spec tinha erro de sinal

Ela dizia `0,02 (−0,03; −0,02)` para a MLP: um estimador pontual **fora do próprio
intervalo**, o que já denunciava o problema. A tabela 5.8 (p.67) do original diz
**−0,02**. Importa porque o alvo é o controle negativo, e o sinal é justamente o que
distingue "o carro não se move" de "o carro se move um pouco".

### 3. O orçamento estava a 37 % do publicado

A spec não registrava o orçamento. O §5.4 (p.65) diz:

> *"each pair of algorithm and benchmark task is executed **10 times**, with
> distinct random initializations, for **2000 episodes**"*, com IC bootstrap de
> 95 % sobre 1000 reamostragens.

A primeira execução usou 1500 × 5 — 75 % dos episódios e 50 % das sementes. Um
resultado negativo ali não seria atribuível ao método.

## O que está eliminado, com evidência

Sem esta lista o diagnóstico seria opinião.

| Hipótese | Como foi eliminada |
|---|---|
| Protocolo de avaliação | corrigido; a MLP passou a reproduzir |
| Hiperparâmetros errados | tabela A.3 verbatim. A **tabela 5.6 dá resolução 16**, mas é do experimento de taxa com n-tuple **REINFORCE** (§5.3), não do §5.4 — para o Actor-Critic vale a A.3 |
| Semântica de "resolution" | **provada** bits-por-dimensão: *"with a resolution of 16 bits and a state vector of length 4, the encoded cartpole observation inputs consist of only 64 bits"* (p.67). Mountain Car: 128 × 2 = 256 bits ✓ |
| Mapeamento errado | **provado** partição: *"with a resolution of 128, the network has eight times the number of units than with the 16 bit resolution"* (p.67) — isto é `N = B/n` ✓ |
| Orçamento insuficiente | agora 2000 × 10, o publicado |
| `Σ` desvio vs covariância | seguimos a eq. (4.34): `σ = √Σ`. A variante P13 fica atrás de `--p13` |
| Redes com bug | **bit-idênticas ao `Ramnet.jl`** em 5 configurações (`tests/test_redes.py`) |
| Divergência numérica | `c_max=None` aqui e todos os retornos finitos; nenhum `TreinoInstavel` |
| Codificador | `TermometroArtigo`, eq. (3.23), o *Simple* do artigo |

## Um defeito que esta execução revelou no próprio script

**Os dois braços usam laços diferentes**, o que viola o *"a única diferença é
`f(·)`"* da §4.8:

- **n-tuple:** `AtorCriticoNTupla`, a transcrição do Algoritmo 4.4 — retorno de `m`
  passos **por transição**, com bootstrap.
- **MLP:** laço meu, de passagem reversa com **um único ponto de truncamento por
  episódio** (`if i == m: g = V(x)`). É aproximação mais grosseira, não o
  Algoritmo 4.4.

Consequência para o diagnóstico, e é ela que importa: **o fato de a MLP reproduzir
não valida o laço.** O alvo que ela reproduz é um ótimo local degenerado — "não
mover o carro" — que quase qualquer esquema de crédito encontra, porque basta levar
a média da ação a zero. Reproduzir isso não é evidência de que a atribuição de
crédito funciona.

E o laço do Algoritmo 4.4 é justamente **o único componente da cadeia sem oráculo
diferencial** (item 1 da `PLANO-TESTES.md` §5): validá-lo exigiria rodar o agente
Julia inteiro, amarrado aos tipos de ambiente e buffer do `RLWorktable.jl`.

## A hipótese de mecanismo que eu tinha, e o que a medição de fato mostrou

Minha suspeita era **generalização**: com `n = 32` em 256 bits são só **8 RAMs**,
quase memorização, então a trajetória bem-sucedida imprimiria nos endereços dela e
não recuperaria no episódio seguinte, partindo de outro lugar. Previsão: tuplas
**menores** (mais RAMs, mais generalização) resolveriam.

Medido, 2000 episódios × 3 sementes por ponto:

| `n` | N RAMs | aval | chegada (treino) | chegada (aval) |
|---|---|---|---|---|
| 8 | 32 | −1,73 | 0,20 % | **0,00 %** |
| 16 | 16 | −0,94 | 0,25 % | **0,00 %** |
| 32 | 8 | −0,13 | 0,42 % | **0,00 %** |
| 64 | 4 | **−0,01** | 0,37 % | **0,00 %** |

A previsão está refutada — o retorno melhora monotonicamente com tupla **maior**.
Mas a leitura óbvia dessa monotonia também está errada, e foi o ponto `n = 64` que
expôs isso.

### A monotonia não é aprendizado: é convergência para o ótimo local

**Zero é o teto do "não mover o carro".** Sem alcançar a bandeira não se faz melhor
que ≈ 0, porque a única recompensa positiva é o `+100` da chegada; todo o resto é
custo de ação. Os quatro valores — −1,73, −0,94, −0,13, −0,01 — são todos `≤ 0` e
**convergem para 0 por baixo**.

Então tupla maior não está aprendendo mais: está assentando no ótimo local com mais
precisão. Tuplas menores generalizam mais, o que **borra** a saída "emita zero" entre
estados vizinhos e deixa custo de ação residual. Quanto mais memorização, mais limpo
o zero.

A prova está na coluna que mede progresso de verdade: a **chegada em avaliação é
0,00 % em todas as quatro configurações**, e a chegada em treino não sobe com `n`
(0,42 % em `n=32` contra 0,37 % em `n=64`). **Nenhuma configuração aprende a tarefa.**

### Retratação: isto NÃO reproduz o achado de tamanho de tupla do Katopodis

Eu ia escrever que a direção reproduz a §5.3 dele — *"agents are much more benefited
from greater model capacity than generalization ability"*, melhores curvas em `N = 1`
RAM. **O sinal da tendência coincide e o significado é oposto.** No caso dele, tupla
maior levava o retorno **para 93,3**, isto é, resolvendo. No nosso, leva **para 0**,
isto é, desistindo com mais capricho. Chamar isso de reprodução seria usar a
coincidência de sinal para encobrir a ausência do fenômeno.

O que a varredura estabelece é mais modesto e ainda útil: **o tamanho de tupla não é
a alavanca desta falha.** Quatro pontos cobrindo 32→4 RAMs, nenhum sai do ótimo
local. Isso remove capacidade/generalização da lista de suspeitos — por eliminação,
não por reprodução.

## Consequência para a pergunta aberta 2 da SPEC-05

> *"`n` deve ser estendido para 16 ou 32 na varredura da SPEC-06? O PRD varre
> `{8, 10, 12}`, todo abaixo da faixa dele."*

**Não respondida, e é importante não fingir que foi.** A monotonia que medimos é
convergência ao ótimo local, não desempenho de tarefa, então ela não diz nada sobre
qual `n` aprende melhor — nenhum aprendeu. Usar esta curva para justificar ampliar a
faixa seria construir a recomendação sobre um artefato.

O que se pode dizer, e só isso: a faixa `{8, 10, 12}` do PRD fica **inteiramente
abaixo** dos `n ∈ {16, 32, 64}` que Katopodis usa nas tarefas contínuas, e o
argumento do PRD para o teto (`2^n` explode) já está refutado pela SPEC-03 — com
armazenamento esparso, `n` grande custa generalização, não bytes. A recomendação de
incluir `n = 16` continua válida **pelo argumento de fidelidade à referência**, que é
independente desta medição.

## O que resta como suspeito, em ordem de confiança

Com capacidade/generalização, protocolo, hiperparâmetros, codificador, mapeamento,
orçamento e as redes todos eliminados, sobram três, e nenhum é confortável:

**1. O laço do Algoritmo 4.4 — o único componente sem oráculo diferencial.** É o
item 1 da `PLANO-TESTES.md` §5, e é exatamente onde um erro silencioso viveria.
Inspecionei o retorno de `m` passos linha a linha e ele fecha: `passo_intermediario`
faz *bootstrap* de `V(s_{t+1})` sobre uma janela de `m` transições e aplica à mais
antiga, o que é `r_τ + … + r_t + γ^m V(s_{t+1})` para `τ = t − m + 1`. Correto na
leitura, sem oráculo na execução.

**2. Exploração temporalmente descorrelacionada.** O `MountainCarContinuous` é o
caso clássico difícil de exploração: ruído gaussiano independente por passo injeta
pouca energia na frequência de ressonância do vale, e é por isso que o DDPG usa
Ornstein-Uhlenbeck. Medimos 0,34 % de chegada, ou ~7 episódios bem-sucedidos por
semente em 2000. Se o do Katopodis for materialmente maior, algo difere no caminho
da ação — mas ele usa a mesma gaussiana isotrópica da eq. (4.34), então esta
hipótese exige que a diferença esteja em outro lugar.

**3. `Σ` como desvio em vez de covariância (P13).** Seguimos a eq. (4.34), `σ = √Σ`.
O `ContinousActionPolicy.jl` amostra `Σ·z` mas usa `inv(Σ)` no gradiente. Se o
número publicado saiu do código e não do artigo, a exploração dele foi **mais
estreita** (`σ = 0,5` contra os nossos 0,707) — o que pioraria a exploração, não a
melhoraria, então esta hipótese prevê o sinal errado. Fica registrada e medível por
`--p13`.

**Um quarto, que é conjectura e vai marcado como tal:** ~7 sucessos por semente
podem simplesmente não bastar, e a diferença entre 93,3 e −0,2 pode ser um efeito de
limiar em que a primeira gravação bem-sucedida ou pega ou não pega. Com `α = 8·10⁻⁴`,
`δ ≈ +100` na janela vitoriosa e `∇log ≈ 1,4`, o passo é ~0,11 por atualização e
~11 acumulado na janela — imprime forte. Mas isto é aritmética, não medição.

## Impacto declarado na SPEC-06

**A T-05.4(a) existia para garantir que um resultado negativo da SPEC-06 fosse
atribuível ao problema, e ela não entrega isso para o braço n-tuple.** Registrar o
que sobra:

- **O que a T-05.4(a) ainda garante:** as redes estão certas (bit-idênticas ao
  `Ramnet.jl`), o codificador está certo (semântica de resolução provada), o
  mapeamento está certo, o protocolo de avaliação está certo, e a dependência
  qualitativa do tamanho de tupla reproduz.
- **O que ela não garante:** que o laço do Algoritmo 4.4, sobre horizonte longo e
  recompensa quase esparsa, atribua crédito corretamente.
- **Mitigação a levar para a SPEC-06:** o ambiente de Lamenza tem recompensa
  **muito mais densa** que o Mountain Car — `R_t = (Σ visited − Ns)/Ns` muda a cada
  coleta, e o guloso já atinge as coletas com regularidade —, então o regime de
  quase-ausência de sinal que domina aqui pode não transferir. Isso é razão para
  esperança, não evidência: se a SPEC-06 der resultado negativo no braço n-tuple, a
  atribuição precisa ser tratada como **em aberto**, e o primeiro suspeito é o laço.
- **Correção obrigatória no script antes de qualquer nova rodada:** unificar o laço
  dos dois braços, pela §4.8. Hoje o n-tuple usa `AtorCriticoNTupla` e a MLP um laço
  próprio, e enquanto isso durar nenhuma comparação **entre** os braços deste script
  é válida — só a comparação de cada um com o seu número publicado.

## Uma nota sobre honestidade de relato

O primeiro relatório que eu quase escrevi dizia "os dois alvos discrepantes,
provável falha de exploração". Estaria errado em três pontos ao mesmo tempo:
protocolo (retornos de treino em vez de avaliação), alvo (sinal trocado na tabela) e
orçamento (37 % do publicado). Os três eram meus, e o único que estava no artigo
desde o começo — o `−0,02` — eu havia transcrito errado nesta própria spec.

Vale registrar porque o padrão é o mesmo da SPEC-03 e da SPEC-05 §4.8: **antes de
atribuir uma discrepância ao método, esgotar as hipóteses sobre o próprio arnês.**
Aqui, três das quatro camadas de erro eram nossas, e a quarta — a que sobra — só
ficou visível depois de as outras três saírem do caminho.

## Reprodução

```bash
# o orçamento publicado: 2000 episódios × 10 sementes
uv run python scripts/spec05_mountaincar.py --episodios 2000 --sementes 10 --processos 10

# a variante P13: Σ como desvio, e não covariância
uv run python scripts/spec05_mountaincar.py --p13
```

Artefato: `reports/spec05-mountaincar.json`, com as colunas de treino e de avaliação
separadas e a taxa de chegada nas duas. Paralelizar por semente **não altera
resultado**: cada semente instancia redes, `rng` e `reset` próprios.

**Hopper não foi rodado.** Exige MuJoCo, e a T-05.4(a) já o trata como condicional
(*"e, se o orçamento permitir, Hopper"*). Consequência: a pergunta aberta 3 —
variância 0,2 (§5.3) ou 0,3 (tab. A.3) — fica **respondida na parte factual** (a
A.3 diz 0,3, então a inconsistência é com o texto da §5.3, não dentro do apêndice) e
**sem resposta na parte empírica** (qual das duas reproduz o 748,8 publicado).
