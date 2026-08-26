# Documentação

Fundamentação teórica e mapas do código. Separada de `reports/`, que guarda
**resultados de experimentos** — aqui está o que explica *como as coisas
funcionam*, não *o que foi medido*.

## `teoria/` — fundamentação

| documento | assunto |
|---|---|
| [`filtro-de-kalman.md`](teoria/filtro-de-kalman.md) | o KF do degrau 4: variáveis, equações, implementação linha a linha, e como o `q_w` é descoberto pelo NEES |

## Mapas do código (em `reports/`, por ora)

| documento | assunto |
|---|---|
| [`../reports/codebase.md`](../reports/codebase.md) | a base inteira — 42 scripts, 45 módulos, 8 pipelines |
| [`../reports/pipeline-treinamento.md`](../reports/pipeline-treinamento.md) | o pipeline de treino, do `env.step` até a escrita na LUT |

## Convenção

Todo número de linha citado nestes documentos é **verificado contra o código**
antes do commit — um script confere que o trecho citado está de fato naquela
linha. Citações que envelhecem são erro, não detalhe: elas mandam o leitor para
o lugar errado com aparência de precisão.
