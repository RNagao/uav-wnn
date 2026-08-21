# T-05.5 — o canal de identidade sobrevive à binarização?

**Resposta curta: com o one-hot simples, quase não.** Com `Na = 4` e `R = 1`, o
erro angular é 48,3° contra 58,4° de uma política **cega ao índice** — o canal
recupera 17 % do que havia para recuperar. Replicar o one-hot 16 vezes leva a
10,8°, que é 82 %.

Isso importa porque `R = 1` é exatamente o que a observação de Lamenza (eq. 5-1)
entrega: um índice, um bit por agente. Numa rede densa isso basta; sob binarização
com uma rede n-tuple, **não basta**, e a mitigação (i) da §4.7 deixa de ser
opcional.

## O problema

Uma rede n-tuple **só** para todos os UAVs (§4.7). A observação carrega o índice
`k`, então em princípio a rede pode diferenciar comportamento. Mas o índice são
`Na` bits num padrão de ~800, e **cada RAM lê apenas `n` deles**: a maioria das
RAMs não vê o canal, e para elas dois agentes com a mesma observação posicional são
o mesmo agente.

## O brinquedo exige diferenciação, em vez de só permiti-la

`Na` agentes, 4 sensores, **observação idêntica para todos exceto o índice**. O rumo
correto do agente `k` é o do sensor `k`. Sem usar o índice, nenhuma política acerta
mais de uma das `Na` respostas — então "o canal passou" e "o canal não passou" são
separáveis sem ambiguidade.

Usa o **bandido contextual** de T-05.4(b), não o `nav` sequencial: aquele tem
pendência aberta de crédito temporal (TS5) que afetaria as duas condições e
mascararia o efeito de `R`.

## 1. Cobertura: a fórmula da spec supõe o mapeamento errado

| `Na` | `R` | `B` | `N` | p binomial | p hipergeom. | medido | medido − hiper |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 258 | 22 | 0,0892 | 0,0910 | 0,0909 | −0,0001 |
| 2 | 4 | 264 | 22 | 0,3088 | 0,3143 | 0,3136 | −0,0007 |
| 2 | 16 | 288 | 24 | 0,7567 | 0,7638 | 0,7771 | +0,0133 |
| 4 | 1 | 260 | 22 | 0,1698 | 0,1732 | 0,1727 | −0,0004 |
| 4 | 4 | 272 | 23 | 0,5169 | 0,5244 | 0,5196 | −0,0048 |
| 4 | 16 | 320 | 27 | 0,9313 | 0,9348 | 0,9315 | −0,0033 |

A §4.7 escreve `p = 1 − (1 − R·Na/B)^n`, que é a probabilidade de uma RAM ler ao
menos um bit de índice **se cada RAM sorteasse seus `n` bits com reposição**. É a
mesma suposição do `NTupleRegressor` do protótipo, e foi ela que gerou o achado
**C4** — bits sem nenhuma RAM lendo.

O `mapeamento_particionado` da SPEC-04 **particiona** (`randperm` fatiado,
cobertura 1,0 por construção), então a conta certa é **hipergeométrica**:

```
p = 1 − C(B − K, n) / C(B, n),     K = R·Na
```

Medido, a hipergeométrica bate melhor em 5 das 6 configurações. A diferença é
pequena porque `K, n ≪ B`, então a binomial serve como **aproximação** — mas não é
identidade, e num regime de `K/B` maior a lacuna cresceria. Fica registrada como
aproximação na spec.

## 2. Efeito de `R`, contra um controle cego

`R = 0` é o **controle cego**: nenhum bit de índice. É a melhor política possível
sem saber qual sensor é o seu, e sem ele os outros números não têm escala — "48°"
poderia ser bom ou desastroso.

| `Na` | `R` | cobertura | erro | dispersão | do cego eliminado |
|---|---|---|---|---|---|
| 2 | **0 (cego)** | 0,000 | **48,2°** | **0,000** | — |
| 2 | 1 | 0,091 | 33,4° | 0,098 | 31 % |
| 2 | 4 | 0,288 | 16,2° | 0,266 | 66 % |
| 2 | 16 | 0,764 | 9,5° | 0,360 | 80 % |
| 4 | **0 (cego)** | 0,000 | **58,4°** | **0,000** | — |
| 4 | 1 | 0,182 | 48,3° | 0,106 | **17 %** |
| 4 | 4 | 0,551 | 26,9° | 0,359 | 54 % |
| 4 | 16 | 0,963 | 10,8° | 0,512 | 82 % |

```
Na=2   R=16 − R=1:  Δ = −23,90°  IC95 % [−25,87; −22,09]  MDE 1,82°
Na=4   R=16 − R=1:  Δ = −37,49°  IC95 % [−40,09; −34,84]  MDE 2,47°
```

Os dois ICs estão a mais de 13 MDEs de zero.

A **dispersão** é a medida direta do canal: `1 − ‖média vetorial dos rumos‖` entre
agentes na mesma observação. No cego ela sai **exatamente 0,000**, porque entrada
idêntica dá saída idêntica por construção — o que também serve de verificação do
próprio experimento. Com `R = 1` sobe a apenas ~0,10: os agentes quase não se
distinguem.

### Três leituras

**(a) `R = 1` deixa a maior parte do canal na mesa.** Com `Na = 4`, o one-hot
simples recupera 17 %. Não é ruído: é o canal chegando degradado a 82 % das RAMs.

**(b) `Na` maior precisa de `R` maior, e isso é contraintuitivo.** A fórmula sozinha
diz que `p` **cresce** com `Na` para `R` fixo (mais bits de índice, mais chance de
uma RAM ler algum) — e de fato a cobertura vai de 0,091 para 0,182 quando `Na`
dobra. Ainda assim o desempenho **piora**: 31 % contra 17 % de recuperação. A razão
é que a tarefa passa a exigir mais do canal — distinguir 4 comportamentos em vez de
2 — e essa exigência cresce mais rápido que a cobertura. Cobertura não é
capacidade.

**(c) A degradação é graciosa, não abrupta.** Com 9 % das RAMs lendo o canal
(`Na=2, R=1`) já se recupera 31 %. Não há limiar em que o canal simplesmente
desliga; ele perde resolução continuamente. Isso é boa notícia para robustez e má
notícia para diagnóstico — um canal parcialmente mudo se parece com "aprendeu
menos", não com "quebrou".

## Determinismo da barreira — o terceiro item de T-05.5

O item pedia tabelas idênticas bit a bit sob **ordens de evento diferentes**.
Medido, isso é **impossível** para o crítico com esquecimento:

```
ator      v ← v + passo        soma, COMUTA
crítico   v ← φ·v + alvo       NÃO comuta
```

Com dois agentes escrevendo na mesma posição no mesmo tique,
`(φv + a₁)·φ + a₂ ≠ (φv + a₂)·φ + a₁`, e a diferença é exatamente
`(φ−1)(a₁−a₂)` — zero apenas se `φ = 1` (sem esquecimento, e aí recupera a eq.
3.16) ou `a₁ = a₂`.

**E a colisão é o regime normal, não o caso de canto.** A tabela acima é a
evidência: com `Na = 4` e `R = 1`, só 18 % das RAMs leem bit de índice. Nas outras
82 %, dois agentes com a mesma observação posicional endereçam a mesma posição.

O que o projeto entrega é **determinismo** com ordem canônica (índice crescente),
que é o que `make repro` precisa. Invariância exigiria acumular os alvos por
endereço dentro do tique e aplicar `φ` uma vez só — o que muda a eq. (4.8) de "por
observação" para "por tique", desvio de método e não detalhe de implementação.

Coberto por `tests/test_barreira.py`, incluindo um teste que **afirma a
desigualdade** para que mudá-la seja decisão registrada e não efeito colateral de
refatoração.

## Recomendação para a SPEC-06

Levar `R` como hiperparâmetro varrido, com `R = 1` **como controle e não como
padrão**. O custo é baixo: `R = 16` com `Na = 4` são 64 bits contra 260, ou seja
~20 % mais bits de entrada, e num armazenamento esparso bits de entrada não custam
memória de LUT — custam apenas endereço mais largo, que o `uint64` já acomoda.

A mitigação (iii) da §4.7 (*"não fazer nada e medir"*) fica **descartada por
medição**: haveria efeito nulo se a coordenação emergisse só de posições relativas,
e não é o caso — `R` mexe no resultado com IC a 13 MDEs de zero.

## Reprodução

```bash
uv run python scripts/spec05_identidade.py --tentativas 15000 --sementes 3
```

Artefato: `reports/spec05-identidade.json`. Alvos de avaliação com semente fixa
`4242`, o que autoriza o pareamento; bootstrap com raiz `20260821` via
`ArvoreSementes`.
