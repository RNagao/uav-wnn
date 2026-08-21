# SPEC-05 — n-tuple Actor-Critic (ação contínua) e baseline MLP AC pareado

**Etapa:** 1 · **Depende de:** SPEC-04 (núcleo n-tuple esparso), SPEC-01 (codificador, G0) · **Estimativa:** 3 dias
**Alimenta:** SPEC-06 (Etapa 1, portão G1) · **Referência:** PRD §5.2 (MDP verbatim), §5.3 (aproximador), §5.5 (custo), §10 (D2, D6, D7)
**Fonte primária:** Katopodis (2022) §4.3.1–§4.3.4, §5.3, §5.4, apêndice A · Lamenza (2025) §5.1–§5.3, tab. 5.1

---


## 0-quater. Fechamento da spec (2026-08-21)

Estado por critério de saída, e **quatro correções que a execução impôs aos
próprios critérios**. Elas estão aqui em cima porque um critério satisfeito por
leitura generosa é pior que um critério aberto.

| # | Critério | Estado |
|---|---|---|
| 1 | T-05.4(a) — os quatro números do Katopodis | **FECHADO, e o fenômeno reproduz nos dois braços.** n-tuple **96,72** com **100 % de chegada à bandeira** (publicado 93,3); MLP −0,04 (publicado −0,02). A "falha" anterior era **bug meu no ponto de chamada** — ver (e) |
| 2 | T-05.4(b) — C2 > C0 com IC separado | **FECHADO.** `Δ = −13,45°`, IC95 % `[−14,53; −12,43]`, 12× o MDE. `reports/spec05-circular.md` |
| 3 | Pareamento auditável | **FECHADO.** `tests/test_pareamento.py` |
| 4 | `γ = 0,99` fixo e faixas com procedência | **FECHADO.** `configs/spec05/hiperparametros.toml`, verificado por teste |
| 5 | Sem PyTorch fora de `tests/` | **FECHADO com redação corrigida** — ver (b) |
| 6 | Asserção de estabilidade ativa, com caso que aborta | **FECHADO.** `TreinoInstavel`; ligada em `NTuplaAprox`, não na rede — ver (c) |
| 7 | Determinismo com `Na > 1` e ordens diferentes | **FECHADO em parte, com achado** — ver (d) |
| 8 | Relatório curto | `reports/spec05-circular.md`, `spec05-identidade.md`, `spec05-mountaincar.md` |
| 9 | PRD atualizado em dois pontos | **FECHADO.** `PRD.md` §5.5, RNF-05, RNF-11 e a linha de risco |

### (a) O primeiro portão media exploração, não parametrização

A T-05.4(b) começou apenas no brinquedo sequencial `toy/nav.py` e deu
**inconclusivo com os dois braços piores que rumo aleatório** (150° de erro, 3–8 %
de sucesso). Diagnóstico: `δ ≈ −19` constante, porque com sucesso quase nulo o
retorno vale ≈ −60 **qualquer que seja o rumo** e o gradiente não carrega direção.

O portão foi decomposto em três instrumentos — bandido fixo, bandido contextual,
nav sequencial — e **só o contextual decide**, porque é o único em que os dois
braços demonstravelmente aprendem. Um braço que não aprende não é evidência sobre
o outro, e um empate entre dois fracassos não é empate.

O nav sequencial permanece **pendência aberta**: a política ótima analítica faz
7,86 passos com 100 % de sucesso, então o brinquedo é resolvível e a falha é de
crédito temporal — nos dois braços. Fica registrado, fora do veredito.

### (b) O critério 5 estava escrito mais forte do que se pretende

"Nenhuma dependência de PyTorch fora de `tests/`" é violado **por construção**:
`scripts/spec02_maddpg_*.py` importam `torch` porque é o que lê os pesos publicados
de Lamenza, e o baseline MADDPG não existe sem eles. O que a regra protege é a
**nossa** implementação, e `src/uavwnn/` está limpo — verificado por teste
(`test_src_nao_importa_torch`), que é a leitura que RNF-11 de fato exige.

### (c) A asserção da §4.2 não pode viver na rede

Ligar `c_max = 0,5` em `RedeGradienteFuncional` **aborta 13 testes existentes**: os
diferenciais alimentam gradientes arbitrários para casar bit a bit com o
`Ramnet.jl`, e os de identidade usam passo unitário para sondar as eqs. (4.16) e
(4.22). Essa validação contra o Julia é a evidência mais forte do projeto.

A asserção foi movida para `NTuplaAprox` — o ponto onde o gradiente de política de
fato entra na tabela, e onde a divergência da RNF-01 foi **medida**. A rede é
primitiva de uso geral e não policia quem a chama. Um teste garante que o padrão
não seja invertido por descuido.

### (d) Invariância à ordem é inatingível para o crítico com esquecimento

O critério 7 pede tabelas idênticas "inclusive com `Na > 1` e **ordens de evento
diferentes**". Medido, isso é impossível, e a razão é aritmética:

    ator      `v ← v + passo`        soma, COMUTA
    crítico   `v ← φ·v + alvo`       NÃO comuta

Com dois agentes escrevendo na mesma posição no mesmo tique,
`(φv + a₁)φ + a₂ ≠ (φv + a₂)φ + a₁`, e a diferença é exatamente `(φ−1)(a₁−a₂)`.
**A colisão é o regime normal, não o de canto:** com `Na = 4` e `R = 1`, só 18 %
das RAMs leem bit de índice, então nas outras 82 % dois agentes com a mesma
observação posicional endereçam a mesma posição.

Invariância exigiria acumular os alvos por endereço dentro do tique e aplicar `φ`
uma vez só — o que muda a eq. (4.8) de "por observação" para "por tique", desvio
de método e não detalhe de implementação. **O que se entrega é determinismo** com
ordem canônica (índice crescente), que é o que `make repro` precisa. A distinção
está em `tests/test_barreira.py`, incluindo um teste que **afirma a desigualdade**
para que mudá-la seja decisão e não efeito colateral.

### (e) T-05.4(a) reproduz — depois de quatro erros nossos, o último deles caro

**O n-tuple resolve o Mountain Car em 100 % dos episódios de avaliação, nas dez
sementes, com retorno 96,72 contra os 93,3 publicados** — e o excesso é para cima:
gastamos 3,28 de energia contra 6,70 dele. A MLP fica no ótimo local de não mover o
carro (−0,04 contra −0,02), que é o contraste que o artigo descreve.

**O erro que custou mais.** `AtorCriticoNTupla.reiniciar()` apaga as duas redes, e o
laço de treino do script a chamava a **cada episódio**, acreditando que ela só
esvaziava a janela de `m` passos. Resultado: 2000 episódios **independentes**, cada
um de rede em branco — e o retorno ≈ 0 que isso produz é **indistinguível de "o
método não funciona"**.

O sintoma que denunciou não foi nenhuma das métricas de RL: foi a **ocupação de
memória oscilando** (188 → 722 → 283 → …) numa estrutura que só recebe chaves e
portanto não pode encolher. Corrigido com `novo_episodio()`, aviso na docstring do
`reiniciar()`, e teste de regressão que exige ocupação monotonicamente crescente
entre episódios.

| | antes | depois |
|---|---|---|
| retorno (avaliação) | −0,199 | **96,72** |
| chegada à bandeira | 0,00 % | **100,0 %** |

**Os outros três**, todos corrigidos antes: retornos de treino em vez de avaliação
determinística (com `Σ = 0,4` o ruído custa `−39,96` por episódio, então o `−0,02`
publicado é inatingível sob ruído — a MLP foi de −31,86 para −0,04); sinal trocado
no alvo publicado da nossa própria tabela; e orçamento a 37 % do publicado.

**Confirmação independente do protocolo.** Com o bug corrigido os dois protocolos
podem ser comparados de frente, e um se refuta: avaliação erra no máximo **3,4**
(n-tuple +3,4, MLP +0,02); treino erra **31,7** (a MLP o desmonta). O treino é
tentador — 92,47 fica a 0,43 do limite inferior do n-tuple — e é aí que engana.

**RETRATAÇÕES.** A varredura de tamanho de tupla que eu havia reportado rodou com o
bug e está **inválida** — media aprendizado de um episódio só. **Refeita**, ela
responde a pergunta aberta 2: `n = 8` (32 RAMs) foi o único a falhar — 1 das 3
sementes em −99,37 com 0 % de chegada —, e `n ≥ 16` resolveu em 100 % das avaliações
nas três sementes, com energia caindo de 3,91 (`n=16`) para 2,58 (`n=64`).
**A faixa `{8, 10, 12}` do PRD está na borda que quebrou** — e pelo eixo do número de
RAMs é pior ainda: no nosso problema (`B ≈ 800`) ela dá `N ∈ {66, 80, 100}`, contra
os 32 que falharam aqui. Recomendação: **incluir `n = 16` e `n = 32`**, e tratar
`n = 8` como ponto de risco. Ressalva de transferência mantida: 2 dimensões contra
as nossas 25, e Katopodis mediu direção oposta entre Mountain Car e Hopper. E a conclusão *"o
primeiro suspeito é o laço do Algoritmo 4.4"* está **retirada**: a lista de
eliminação estava correta, mas faltava nela o meu próprio ponto de chamada, que não
é componente do método e por isso não estava sendo auditado.

**A lição, que é a da SPEC-03 outra vez:** antes de atribuir uma discrepância ao
método, esgotar as hipóteses sobre o arnês — **e o arnês inclui o código de chamada,
não só as bibliotecas.**

### (e-bis) O que continua sem oráculo, e o que mudou

O propósito escrito do critério 1 é: *"sem isto, nenhum resultado negativo da
SPEC-06 é atribuível ao problema"*. **Ele entrega isso em parte, e a parte que falta
precisa ficar declarada.**

Três erros nossos foram corrigidos no caminho, e o primeiro relatório que quase
escrevi estaria errado nos três ao mesmo tempo:

1. **Reportava retornos de treino, com ruído.** A regra oposta está na §4.4 e está
   codificada em `politica_continua.py`; o script a violava. Com `Σ = 0,4` o ruído
   sozinho custa `−0,1 · 0,4 · 999 = −39,96`, e o `−0,02` publicado é inatingível
   sob ruído. Corrigido: a MLP foi de **−31,86 para −0,04**.
2. **A tabela de alvos desta spec tinha o sinal trocado** (`0,02` em vez de
   `−0,02`) — um estimador pontual fora do próprio intervalo.
3. **O orçamento estava a 37 % do publicado.** O §5.4 (p.65) diz 10 sementes ×
   2000 episódios; rodávamos 1500 × 5.

**O que ficou eliminado com evidência:** protocolo de avaliação (a MLP passou a
reproduzir), hiperparâmetros (A.3 verbatim; a tabela 5.6 divergente é de outro
experimento), semântica de resolução (**provada** bits-por-dimensão pela aritmética
do cartpole na p.67), mapeamento por partição (**provado** pelo *"eight times the
number of units"*), orçamento, as duas redes (bit-idênticas ao `Ramnet.jl`), e
capacidade/generalização — uma varredura de `n ∈ {8, 16, 32, 64}` (32→4 RAMs) não
tira nenhuma configuração do ótimo local.

> **Cuidado de leitura na varredura, que quase me pegou.** O retorno melhora
> monotonicamente com tupla maior (−1,73 → −0,01), e o sinal dessa tendência
> coincide com o achado da §5.3 do Katopodis. **O significado é oposto:** zero é o
> teto do "não mover o carro", então convergir para zero é assentar no ótimo local
> com mais precisão, não resolver. A chegada em avaliação é **0,00 % nos quatro
> pontos**. Isso também significa que a **pergunta aberta 2 continua aberta** — a
> curva não mede desempenho de tarefa, e usá-la para justificar ampliar a faixa de
> `n` seria construir a recomendação sobre um artefato.

**O que sobra como suspeito**, e o primeiro é desconfortável: o **laço do Algoritmo
4.4**, que é o único componente da cadeia **sem oráculo diferencial** (item 1 da
`PLANO-TESTES.md` §5). Inspecionado linha a linha e correto na leitura; sem
verificação na execução.

E esta spec revelou um defeito no próprio script que impede usar a MLP como
validação do laço: **os dois braços usam laços diferentes**, violando o *"a única
diferença é `f(·)`"* da §4.8 — o n-tuple usa `AtorCriticoNTupla` e a MLP um laço
com um único ponto de truncamento por episódio. Como o alvo que a MLP reproduz é um
ótimo local degenerado que quase qualquer esquema de crédito encontra, reproduzi-lo
**não é evidência** de que a atribuição de crédito funciona.

O laço do Algoritmo 4.4 continua **sem oráculo diferencial** (item 1 da
`PLANO-TESTES.md` §5), e isso não muda. Mas agora ele atravessa uma validação **de
ponta a ponta contra número publicado**, o que é evidência muito mais forte que a
inspeção de código que havia antes.

**O que a T-05.4(a) agora entrega:** um resultado negativo da SPEC-06 no braço
n-tuple **é** atribuível ao problema, que era o propósito escrito do critério.

**O que continua aberto (TS7):** os dois braços deste script usam laços diferentes,
violando o *"a única diferença é `f(·)`"* da §4.8. Isso não invalida a reprodução —
cada braço é comparado ao **seu** número publicado — mas invalida qualquer
comparação **entre** os braços deste script.

### Perguntas em aberto que a execução resolveu

- **Nº 3** (variância do Hopper, 0,2 ou 0,3): a tabela A.3 diz **0,3**. A
  inconsistência é entre o texto da §5.3 e o apêndice, não dentro do apêndice.
  Qual reproduz o número publicado fica sem resposta sem rodar o Hopper (MuJoCo,
  fora do orçamento — a própria T-05.4(a) o trata como condicional).
- **Nº 6** (cronograma do anelamento de `σ_s`): **não há anelamento.** Medido um
  bug na primeira tentativa — `decaimento` conta transições, não episódios, e num
  brinquedo de 40 passos colapsava `σ` em ~20 episódios. Desligado, porque a
  avaliação já é determinística (Lamenza §7) e ruído residual de treino não
  contamina o que é reportado.
- **T-05.5, cobertura do canal:** a fórmula `p = 1 − (1 − R·Na/B)^n` da §4.7 supõe
  amostragem **com reposição** por RAM — o que gerou o achado C4. Com mapeamento
  particionado a conta certa é **hipergeométrica**,
  `1 − C(B−K,n)/C(B,n)`. Medido, a hipergeométrica bate melhor em 5 das 6
  configurações; a binomial fica registrada como aproximação, não identidade.

---

## 0-ter. A dissertação corrige o código do autor em dois pontos (2026-08-15)

A §0-bis tratava o `ContinousFunctionalAC.jl` como referência. Com a leitura da
**dissertação** — fonte primária — dois pontos dela **prevalecem sobre o código**:

**1. `Σ` e `α` são constantes, não recozidas.** O apêndice A lista as duas como
hiperparâmetros fixos (`Σ` = 0,5 em Mountain Car, 0,3 em Hopper). O recozimento
existe só no Julia. Implementado desligado por padrão, e a **D6 do PRD foi
corrigida** — ela prescrevia recozimento.

**2. `Σ` é covariância, não desvio (P13).** A eq. (4.34) diz *"predefined
**covariance** matrix"*, logo a amostragem correta é `√Σ·z`. O
`ContinousActionPolicy.jl` amostra `Σ·z` mas usa `inv(Σ)` no gradiente — trata o
mesmo número como desvio num lugar e variância no outro. **Seguimos o artigo**;
`escala_como_desvio=True` reproduz o Julia. Com `Σ = 0,25` a diferença é **fator
2 na exploração**, então a escolha tem de aparecer na tabela de hiperparâmetros.

**Estado da implementação.** `src/uavwnn/rl/politica.py` e
`src/uavwnn/rl/actor_critic.py`. As duas redes são **idênticas bit a bit** ao
`Ramnet.jl` em 5 configurações (`tests/test_redes.py`), e as identidades do
capítulo 4 estão testadas (`tests/test_identidades_artigo.py`).

**O laço do Algoritmo 4.4 continua sem oráculo diferencial** — validá-lo exigiria
rodar o agente Julia inteiro, amarrado aos tipos de ambiente e buffer do
`RLWorktable.jl`. Há invariantes estruturais e dois testes de comportamento. **O
relatório final precisa declarar isso**, e está registrado em
`specs/PLANO-TESTES.md` §5.

---

## 0-bis. Implementado (2026-08-15)

`src/uavwnn/rl/actor_critic.py` traz o **Algoritmo 4.4** transcrito do
pseudocódigo da p. 48, e `src/uavwnn/rl/politica.py` a gaussiana de covariância
predefinida das eqs. (4.34)/(4.35).

Dois pontos que o pseudocódigo esconde e que o código do autor esclarece: a
vantagem `delta = G - f_value(S_tau)` usa o valor **antes** do treino do
crítico; e o crítico **acumula** `G` pela eq. (4.8), não faz um passo de
gradiente para o alvo `G`.

Recozimento de `alpha` e de `Sigma` existem no `ContinousFunctionalAC.jl` mas
**não na dissertação** (apêndice A lista ambos constantes). Ficam desligados por
padrão.

**Divergência P13:** a eq. (4.34) diz "predefined **covariance** matrix", logo a
amostragem correta é `sqrt(Sigma)*z`. O Julia amostra `Sigma*z` mas usa
`inv(Sigma)` no gradiente. Seguimos o artigo; `escala_como_desvio=True`
reproduz o Julia. Com `Sigma = 0,25` a diferença é fator 2 na exploração.

**Achado que afeta a SPEC-06:** a instabilidade de passo fixo é mensurável. No
bandido contextual, o pior erro entre 5 sementes cresce **monotonicamente** com
`alpha` (0,044 em 0,002 -> 0,238 em 0,05), e em 0,05 uma semente diverge. A
busca de `alpha` precisa de faixa larga e várias sementes; meu primeiro palpite
era 25x grande demais.

**Sem oráculo:** as duas redes são idênticas ao Ramnet.jl, mas o *laço* não foi
validado diferencialmente — só por invariantes estruturais e dois testes de
comportamento. O relatório final precisa declarar isso.

Relatório: `reports/spec04-redes-dissertacao.md`.

---

## 0. Revisão — o algoritmo está no código do autor, e pode ser copiado

`rkatopodis/RLWorktable.jl`, arquivo `src/Agents/WNN/PG/ContinousFunctionalAC.jl`. MIT. Deixa de ser reconstrução a partir do texto e passa a ser adaptação de código existente.

### O algoritmo, verbatim

**Actor-critic de `m` passos com retorno Monte Carlo dentro do buffer.** Não é TD(0):

```
se episódio terminou:            G = 0
senão, com buffer cheio:         G = V(s_atual)          # bootstrap
para cada transição, de trás para frente:
    G  = r + γ·G
    δ  = G − V(s)                                        # vantagem
    atualiza ator  com δ
    treina crítico por REGRESSÃO para G                  # não para o erro TD
```

O crítico é treinado por **regressão para o retorno `G`**, não para o erro TD. É isso que justifica o esquecimento nele: o alvo se desloca conforme a política melhora.

### A política contínua, e o que ela não resolve

`ContinousActionPolicy` emite um vetor em `R^AS` e amostra:

```
a = cov·randn(AS) + predict(f, obs)          # gaussiana isotrópica
grad = inv(cov)·(pred − a)                   # gradiente da perda
train!(f, obs, lr · grad · δ)                # escalado pela vantagem
```

**Katopodis não trata ação circular.** Para o Hopper — torques numa caixa — a gaussiana isotrópica basta. Para o nosso `θ ∈ [0, 2π)` não: a média linear de ângulos continua inválida.

**A recomendação de normal projetada (C1) se mantém, e encaixa nativamente.** O `FunctionalDiscriminator{AS}` já é multi-saída: basta `AS = 3` — `u₁, u₂` para o rumo, `μ_s` para a velocidade —, aplicar `θ = atan2(z₂, z₁)` sobre a amostra, e o resto do laço fica idêntico. É o `d = 3` que esta spec já previa.

### Detalhes de treino a copiar

| Detalhe | Como no original |
|---|---|
| **Covariância da política** | **Recozida**, não aprendida: `(1−α)·start_cov + α·end_cov`, `α = update_count / cov_decay`. **Elimina o hiperparâmetro** que esta spec listava como risco da normal projetada |
| **Taxa de aprendizado** | Recozida pela mesma forma linear |
| **Escala do gradiente do ator** | Pela **vantagem `δ`**, não pelo retorno `G` |
| **Particionamento das tuplas** | `:uniform_random` é o default |
| **Crítico** | `RegressionDiscriminator` com γ; **ator** `FunctionalDiscriminator` sem γ (ver SPEC-04 §0) |

> **Ressalva de leitura.** O default `end_learning_rate = 1e5` no construtor de `ContinousActionPolicy` é quase certamente um erro de digitação por `1e-5` — uma taxa final maior que a inicial contraria o propósito do recozimento. Não copiar sem verificar contra os experimentos da dissertação.

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
| Mountain Car | 93,3 (92,9; 93,8) | **−0,02** (−0,03; −0,02) |
| Hopper | 748,8 (741,7; 758,0) | 643,6 (577,0; 746,0) |

> **Erro de transcrição corrigido (2026-08-21).** Esta tabela dizia `0,02
> (−0,03; −0,02)` — um estimador pontual **fora do próprio intervalo**, o que já
> denunciava o problema. A tabela 5.8 (p.67) do original diz **−0,02**; o sinal se
> perdeu na transcrição. Importa porque o alvo é o controle negativo, e o sinal é
> justamente o que distingue "o carro não se move" de "o carro se move um pouco".
>
> **Orçamento, que esta spec não registrava** (§5.4, p.65): *"each pair of
> algorithm and benchmark task is executed **10 times**, with distinct random
> initializations, for **2000 episodes**"*, com IC bootstrap de 95 % sobre 1000
> reamostragens. A primeira execução nossa usou 1500 × 5 — 75 % dos episódios e
> 50 % das sementes —, e um resultado negativo ali não seria atribuível ao método.
>
> **O protocolo de avaliação sai por dedução, não por citação.** A p.66 explica o
> resultado da MLP assim: *"the MLP methods having optimized their agents to a
> local optimum of not moving the car. In this way, they **never incur the penalty
> of expending energy**"*. Com `Σ = 0,4` o ruído sozinho custa
> `−0,1 · 0,4 · 999 = −39,96` por episódio, então um número que inclua exploração
> **não pode** "never incur the penalty" — logo o reportado é **sem ruído**, e a
> avaliação determinística da §4.4 é o protocolo certo. Confirmado por medição: com
> ruído medimos −31,86; sem ruído, −0,01.
>
> **Cuidado com a tabela 5.6.** Ela dá resolução 16 para o Mountain Car, contra 128
> da A.3. Não é contradição: a 5.6 é do experimento de **taxa de aprendizado com
> n-tuple REINFORCE** (§5.3), não do §5.4. Para o Actor-Critic vale a **A.3**.

Hiperparâmetros do apêndice A.3/A.6 dele, verbatim. O alvo do AC com MLP no Mountain Car (≈ 0, ótimo local de não mover o carro) é um **controle negativo** tão informativo quanto o positivo: se a nossa MLP resolver o Mountain Car, ela não é a MLP dele e a comparação de H1 não é a comparação dele.

**(b) Validação da ação circular** — nenhum benchmark clássico tem ação **angular**, então usamos um brinquedo próprio, um brinquedo próprio em `src/uavwnn/toy/nav.py`: 1 UAV, 1 sensor, área 200 × 200, `dt = 0,5 s`, ação `⟨θ, s⟩`, recompensa `+10` na visita e `−1/passo`. **Não** usa o gerador analítico da SPEC-01, que foi descartado — é um brinquedo independente, sem simulador e sem rede, com episódios de milissegundos.

É aqui que C0, C1 e C2 são comparados com custo desprezível — e é o único ponto do projeto em que a decisão da §4.3 é falsificável de forma barata. Métrica: passos até a visita, e erro angular mediano entre `atan2(f₂,f₁)` e o rumo ótimo.

**Critério:** os quatro números de (a) dentro dos ICs publicados **ou** com discrepância explicada e registrada; C2 estritamente melhor que C0 em (b), com IC separado.

### T-05.5 — Compartilhamento de parâmetros e o canal de identidade

Estender o brinquedo de T-05.4(b) para `Na ∈ {2, 4}` com 4 sensores. Medir:

> **Implementado** em `scripts/spec05_identidade.py`, relatório em
> `reports/spec05-identidade.md`. Usa o **bandido contextual** de T-05.4(b), e não
> o `nav` sequencial, porque aquele tem pendência aberta de crédito temporal que
> afetaria as duas condições e mascararia o efeito de `R`.
>
> O brinquedo **exige** diferenciação em vez de só permiti-la: `Na` agentes veem a
> observação **idêntica** exceto o índice, e o rumo correto do agente `k` é o do
> sensor `k`. Sem usar o índice, nenhuma política acerta mais de uma das `Na`
> respostas — então "o canal passou" e "o canal não passou" são separáveis sem
> ambiguidade.

- `p = 1 − (1 − R·Na/B)^n` teórico contra a fração medida de RAMs que leem ao menos um bit de índice, para `R ∈ {1, 4, 16}`;

> **Correção medida:** esta fórmula supõe que cada RAM sorteie seus `n` bits **com
> reposição** — a mesma suposição do `NTupleRegressor` do protótipo, que gerou o
> achado **C4** (bits sem nenhuma RAM lendo). O `mapeamento_particionado` da
> SPEC-04 **particiona** (`randperm` fatiado, cobertura 1,0 por construção), então
> a distribuição certa é **hipergeométrica**:
>
>     p = 1 − C(B − K, n) / C(B, n),    K = R·Na
>
> Amostragem sem reposição contra com reposição. A diferença é pequena quando
> `K, n ≪ B`, e medida ela favorece a hipergeométrica em 5 das 6 configurações. A
> binomial fica registrada como **aproximação**, não identidade.
- efeito de `R` sobre a diferenciação de comportamento (dispersão dos rumos escolhidos por agentes na mesma observação, exceto o índice);
- determinismo da barreira: mesma semente, duas ordens de evento diferentes → tabelas idênticas bit a bit (**teste automatizado**).

> **Este item é impossível como escrito, e a medição está em
> `tests/test_barreira.py`.** O ator é invariante à ordem sempre (`v ← v + passo`
> comuta); o crítico **não pode ser**, porque `v ← φ·v + alvo` não comuta e a
> diferença entre duas ordens é exatamente `(φ−1)(a₁−a₂)`. A colisão de endereço é
> o **regime normal**: com `Na = 4` e `R = 1`, apenas 18 % das RAMs leem bit de
> índice, então nas outras 82 % dois agentes com a mesma observação posicional
> escrevem na mesma posição.
>
> O que se entrega é **determinismo** com ordem canônica (índice crescente), que é
> o que `make repro` precisa. A distinção entre *determinístico* e *invariante à
> ordem* está escrita, com um teste que **afirma a desigualdade** para que mudá-la
> seja decisão consciente e não efeito colateral. Ver §0-quater (d).

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
