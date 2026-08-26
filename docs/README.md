# Documentação

Fundamentação teórica e mapas do código. Separada de `reports/`, que guarda
**resultados de experimentos** — aqui está o que explica *como as coisas
funcionam*, não *o que foi medido*.

## `teoria/` — fundamentação

| documento | assunto |
|---|---|
| [`wnn-dissertacao.md`](teoria/wnn-dissertacao.md) | a rede sem peso: endereçamento, as duas redes (ator e crítico), a política, o Algoritmo 4.4, e o custo de uma decisão |
| [`wnn-anbubus.md`](teoria/wnn-anbubus.md) | a segunda família WNN (`NTNModel` do artigo da Neurocomputing): as três diferenças estruturais e por que o braço PPO falhou |
| [`maddpg.md`](teoria/maddpg.md) | o baseline do Lamenza: CTDE, o laço de treino do fonte original, e o que exatamente usamos dele |
| [`filtro-de-kalman.md`](teoria/filtro-de-kalman.md) | o KF do degrau 4: variáveis, equações, implementação linha a linha, e como o `q_w` é descoberto pelo NEES |

## Mapas do código

| documento | assunto |
|---|---|
| [`codebase.md`](codebase.md) | a base inteira — 42 scripts, 45 módulos, 8 pipelines |
| [`pipeline-treinamento.md`](pipeline-treinamento.md) | o pipeline de treino, do `env.step` até a escrita na LUT |

## Referência que vive em `reports/`

Estes são **entregáveis da série de specs** — `specs/SPEC-NN-*.md` aponta para
`reports/specNN-*.md`, e quebrar esse mapeamento custaria mais que a arrumação
ganharia. Mas leem-se como fundamentação:

| documento | assunto |
|---|---|
| [`../reports/spec00-parametros.md`](../reports/spec00-parametros.md) | o ambiente do Lamenza: proveniência, por que não reimplementamos, e o que a versão do simulador implica |
| [`../reports/spec04-redes-dissertacao.md`](../reports/spec04-redes-dissertacao.md) | as redes e o Algoritmo 4.4 contra a dissertação, com os testes diferenciais |
| [`../reports/spec02-baseline-maddpg.md`](../reports/spec02-baseline-maddpg.md) | tamanho e latência dos pesos publicados do MADDPG |

## Convenção

Todo número de linha citado nestes documentos é **verificado contra o código**
antes do commit — um script confere que o trecho citado está de fato naquela
linha. Citações que envelhecem são erro, não detalhe: elas mandam o leitor para
o lugar errado com aparência de precisão.
