# T-05.4(a) — reprodução do Katopodis no `MountainCarContinuous-v0`

**Veredito: reproduz o fenômeno nos dois braços. A discrepância anterior era um bug
meu, num ponto de chamada do script — não no método.**

| | medido | publicado | |
|---|---|---|---|
| **n-tuple AC** | **96,72** [96,70; 96,75] | 93,3 (92,9; 93,8) | resolve, e gasta **2× menos energia** |
| **AC com MLP** (controle negativo) | −0,04 [−0,04; −0,03] | −0,02 (−0,03; −0,02) | erra por 0,02 |

Chegada à bandeira na avaliação determinística: **100,0 %** no n-tuple, nas dez
sementes; **0,0 %** na MLP. É exatamente o contraste que o artigo descreve.

Formalmente nenhum dos dois IC se sobrepõe ao publicado. Mas os desvios são 0,02 e
3,4 numa escala em que o efeito discriminante entre os dois métodos é ~93 — e o
sinal do nosso desvio no n-tuple é **para cima**, isto é, a nossa política é melhor
que a publicada, não pior. A §"o que sobra" abaixo quantifica.

## O bug, que consumiu a maior parte deste esforço

`AtorCriticoNTupla.reiniciar()` **apaga as duas redes** — é o `reset!` do agente. O
laço de treino deste script a chamava no começo de **cada episódio**, acreditando que
ela só esvaziava a janela de `m` passos:

```python
for ep in range(episodios):
    obs, _ = env.reset(seed=...)
    agente.reiniciar()          # <-- zerava ator e crítico, todo episódio
```

O efeito foi treinar 2000 episódios **independentes**, cada um partindo de uma rede
em branco. Nunca houve aprendizado ao longo do treino. E o resultado disso —
retorno ≈ 0, o ótimo local de "não mover o carro" — é **indistinguível de "o método
não funciona"**, que é precisamente o que eu concluí e comecei a diagnosticar.

### O sintoma que denunciou

Instrumentei a taxa de chegada por bloco de 200 episódios, esperando decidir entre
"exploração insuficiente" e "o laço não retém crédito". A coluna que respondeu foi
outra — a **ocupação de memória**:

```
   ep  chegada bloco  |f| médio  ocupadas
  200         0.50%      0.0443       188
  400         1.00%      0.0412       722
  600         0.50%      0.0389       283     <-- caiu
  800         0.50%      0.0397       689
 ...
 2000         0.50%      0.0429       729
```

`entradas_ocupadas()` soma `len()` de dicionários que **só recebem chaves**. Ela não
pode encolher, e após 2 milhões de atualizações deveria estar nas centenas de
milhares, não em 729. A oscilação era a assinatura de a rede estar sendo zerada.

### Correção, e a blindagem

- `novo_episodio()` — a operação de início de episódio, que **preserva** o
  aprendizado. Só esvazia a janela de `m` passos.
- `reiniciar()` mantém o comportamento destrutivo (é necessário para começar uma
  execução nova), agora com aviso explícito na docstring.
- **Teste de regressão** em `tests/test_actor_critic.py`: exige ocupação
  monotonicamente crescente ao longo de episódios, e um par que fixa o contrato
  destrutivo do `reiniciar` para ninguém "consertá-lo".

O bug estava confinado a **um único ponto de chamada** — verificado por `grep` em
`src/`, `scripts/` e `tests/`. Nenhum outro código chamava `reiniciar()` por
episódio.

### Efeito da correção, medido

| | antes | depois |
|---|---|---|
| retorno (avaliação) | −0,199 | **96,72** |
| chegada à bandeira (avaliação) | 0,00 % | **100,0 %** |
| chegada à bandeira (treino) | 0,34 % | **98,2 %** |

## Outros três erros meus, corrigidos antes deste

Vale a lista porque o primeiro relatório que eu quase escrevi estaria errado em
**quatro** camadas ao mesmo tempo, e nenhuma delas era do método.

**1. Reportava retornos de treino, com ruído ligado.** A regra oposta está na §4.4
e está codificada em `politica_continua.py` — *"avaliação é determinística (Lamenza
§7)"* — e o script a violava. Com `Σ = 0,4` o ruído sozinho custa
`−0,1 · 0,4 · 999 = −39,96` por episódio, então o `−0,02` publicado é **inatingível**
sob ruído. Corrigido, a MLP foi de −31,86 para −0,04.

O protocolo sai do próprio artigo, por dedução (p.66): *"the MLP methods having
optimized their agents to a local optimum of not moving the car. In this way, they
**never incur the penalty of expending energy**"*. Um número que incluísse exploração
não pode "never incur the penalty".

**2. A tabela de alvos desta spec tinha o sinal trocado** — dizia `0,02` em vez de
`−0,02`, um estimador pontual **fora do próprio intervalo**, o que já denunciava. A
tabela 5.8 (p.67) diz **−0,02**.

**3. O orçamento estava a 37 % do publicado.** O §5.4 (p.65) diz *"executed **10
times** … for **2000 episodes**"*; rodávamos 1500 × 5.

## Confirmação independente do protocolo de avaliação

Com o bug corrigido, os dois protocolos podem ser comparados de frente, e um se
refuta:

| protocolo | n-tuple | MLP | pior erro |
|---|---|---|---|
| **avaliação** (sem ruído) | 96,72 vs 93,3 → **+3,4** | −0,04 vs −0,02 → **+0,02** | **3,4** |
| treino (com ruído) | 92,47 vs 93,3 → −0,83 | −31,73 vs −0,02 → **−31,7** | **31,7** |

O protocolo de treino é **tentador** — 92,47 fica a só 0,43 do limite inferior
publicado do n-tuple — e é aí que ele engana. A MLP o desmonta: 31,7 de erro. O
protocolo de avaliação erra no máximo 3,4, e é o que o texto do artigo sustenta.

## O que sobra de discrepância, e o que ela provavelmente é

O n-tuple **excede** o publicado: 96,72 contra 93,3. Em energia gasta,
`100 − 96,72 = 3,28` contra `100 − 93,30 = 6,70` — **ele gasta 2,0× mais que nós**.

A explicação mais econômica é que o número dele inclui episódios que **não chegam**.
Se uma fração `p` chega (retorno ≈ 96,7) e o resto falha com retorno `X`:

| valor da falha | taxa de chegada dele implicada |
|---|---|
| 0,0 | 96,46 % |
| −5,0 | 96,64 % |
| −40,0 | 97,50 % |

A conta é **robusta ao valor assumido**: qualquer hipótese razoável para o custo de
um episódio falho dá uma taxa de chegada de 96–97,5 %, contra os nossos 100,0 %.
Isto é, a diferença é compatível com a política dele falhar em ~3 % das avaliações
e a nossa em nenhuma.

**Não é verificável** — ele não publica a taxa de chegada, e nem o número de
episódios de avaliação. Fica registrado como explicação quantificada e não
confirmada. O que é verificável, e o que importa para o propósito da T-05.4(a), é
que o **fenômeno** reproduz: o n-tuple resolve, a MLP fica no ótimo local, e a
distância entre os dois é da ordem de 97 pontos de retorno.

## Retratações

**A varredura de tamanho de tupla que reportei antes está inválida.** Ela rodou com
o bug, então media aprendizado de **um episódio só**, e a monotonia que observei
(−1,73 → −0,01 com `n` crescente) era convergência para o ótimo local dentro de um
episódio, não desempenho de tarefa. Refeita; ver a seção seguinte.

**A conclusão "o primeiro suspeito é o laço do Algoritmo 4.4" está retirada.** Eu a
escrevi depois de eliminar protocolo, hiperparâmetros, codificador, mapeamento,
orçamento e as redes — e a lista de eliminação estava correta. O que faltava na
lista era **o meu próprio ponto de chamada**, que não é componente do método e por
isso não estava sendo auditado. O laço está, até onde esta tarefa consegue mostrar,
correto: com ele intacto e a chamada consertada, a tarefa é resolvida em 100 % das
avaliações.

**O que continua sem oráculo diferencial** é o laço, e isso não muda (item 1 da
`PLANO-TESTES.md` §5). Mas agora há uma validação **de ponta a ponta contra número
publicado** que ele atravessa, o que é evidência bem mais forte que a inspeção de
código que eu tinha antes.

## A varredura de tamanho de tupla, refeita com dados válidos

2000 episódios × 3 sementes por ponto, hiperparâmetros da A.3 salvo `tamanho_tupla`:

| `n` | N RAMs | retorno | energia | chegada | sementes que resolvem | pior semente |
|---|---|---|---|---|---|---|
| 8 | 32 | 30,35 | 4,80 | 66,7 % | **2 de 3** | **−99,37** |
| 16 | 16 | 96,09 | 3,91 | 100 % | 3 de 3 | 95,76 |
| 32 | 8 | 96,55 | 3,45 | 100 % | 3 de 3 | 95,97 |
| 64 | 4 | **97,42** | **2,58** | 100 % | 3 de 3 | 97,25 |

Duas leituras, e a segunda é a que interessa.

**A média esconde o achado.** O `30,35` de `n = 8` não é "desempenho intermediário":
são duas sementes resolvendo (94,99 e 95,42) e **uma falhando por completo**
(−99,37, com 0 % de chegada). Energia ~99 em 999 passos significa `|a| ≈ 1,0` — o
acelerador no fundo, oscilando sem nunca chegar. É falha de **confiabilidade**, não
de qualidade média, e reportar só a média a tornaria invisível.

**Entre as configurações confiáveis, tupla maior é levemente mais eficiente**: a
energia gasta cai de 3,91 para 2,58 indo de `n = 16` a `n = 64`. Agora sim a direção
reproduz o achado da §5.3 do Katopodis — *"agents are much more benefited from
greater model capacity than generalization ability"* —, e desta vez com o
significado certo, porque os retornos representam **resolver a tarefa** e não
assentar num ótimo local.

Ressalva de amostra: 3 sementes por ponto é pouco para afirmar confiabilidade. O que
se pode dizer é que `n = 8` produziu uma falha total e `n ≥ 16` não produziu nenhuma,
em 9 execuções.

## Isto responde a pergunta aberta 2 — e a resposta desfavorece a faixa do PRD

> *"`n` deve ser estendido para 16 ou 32 na varredura da SPEC-06? O PRD varre
> `{8, 10, 12}`, todo abaixo da faixa dele."*

**Sim, e por dois caminhos independentes que apontam para o mesmo lado.**

*Pelo eixo `n` (capacidade por RAM, `2^n`):* `n = 8` foi o único a falhar; `n ≥ 16`
foi uniformemente confiável. A faixa `{8, 10, 12}` do PRD está justamente na borda
que quebrou.

*Pelo eixo `N` (número de RAMs):* a falha aqui aconteceu com `N = 32`. No nosso
problema, com `B ≈ 800` bits em 25 dimensões, a faixa do PRD dá:

| `n` | N no nosso problema |
|---|---|
| 8 | 100 |
| 10 | 80 |
| **12** | **66** |
| 16 | 50 |
| 32 | 25 |

Toda a faixa do PRD cai **mais fundo** na região que falhou aqui — `n = 12` já dá 66
RAMs, o dobro dos 32 que quebraram.

**A ressalva de transferência continua séria e é o motivo de isto não ser
conclusivo:** foram 2 dimensões contra as nossas 25, e com 25 dimensões pode ser
*necessário* ter mais RAMs para cobrir a entrada. Katopodis mediu direção **oposta**
entre as duas tarefas contínuas dele (maior no Mountain Car, menor no Hopper), então
o sinal não é universal. Mas os dois eixos concordarem é o mais forte que uma
medição em tarefa auxiliar consegue ser, e a recomendação fica: **incluir `n = 16` e
`n = 32` na varredura da SPEC-06, e tratar `n = 8` como ponto de risco a monitorar,
não como padrão.**

## Um defeito de pareamento que continua aberto (TS7)

**Os dois braços deste script usam laços diferentes**, o que viola o *"a única
diferença é `f(·)`"* da §4.8: o n-tuple usa `AtorCriticoNTupla` (retorno de `m`
passos por transição) e a MLP um laço com um único ponto de truncamento por
episódio.

Isso **não** invalida a reprodução — cada braço é comparado ao **seu** número
publicado, não ao outro. Mas invalida qualquer comparação **entre** os braços deste
script, e precisa ser unificado antes de o script servir para outra coisa.

## A lição, que é a mesma da SPEC-03

Antes de atribuir uma discrepância ao método, esgotar as hipóteses sobre o próprio
arnês — **e o arnês inclui o código de chamada, não só as bibliotecas.** Eu auditei
sete componentes do método com evidência documental, alguns extraindo provas do PDF,
e o erro estava numa linha do meu laço de treino. O sinal estava disponível desde o
começo: uma estrutura *append-only* cuja contagem oscila.

## Reprodução

```bash
uv run python scripts/spec05_mountaincar.py --episodios 2000 --sementes 10 --processos 10
```

Artefato: `reports/spec05-mountaincar.json`, com treino e avaliação em colunas
separadas e a taxa de chegada nas duas. Paralelizar por semente não altera
resultado: cada semente instancia redes, `rng` e `reset` próprios.

**Hopper não rodado.** Exige MuJoCo, e a T-05.4(a) já o trata como condicional
(*"e, se o orçamento permitir, Hopper"*). A pergunta aberta 3 fica respondida na
parte factual — a tabela A.3 diz **0,3**, logo a inconsistência é com o texto da
§5.3, não dentro do apêndice — e sem resposta na parte empírica.
