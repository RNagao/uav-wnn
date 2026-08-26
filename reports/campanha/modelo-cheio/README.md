# Campanha medida no modelo CHEIO — superseded

Estes CSVs avaliaram `models/T*/…` (ator completo, `float64`). A campanha passou
a avaliar `models/deploy/T*/…` — o artefato **podado a 70 %**, que é o mesmo cujo
tamanho a tese reporta.

A razão da troca: publicar o tamanho de um artefato e o desempenho de outro
afirmaria um modelo de 144 KB que nunca foi avaliado nas 20 mil sementes.

Ficam aqui porque são dado válido de uma pergunta diferente — *"quanto o modelo
cheio entrega"* — e servem de referência para medir o custo da poda em escala
grande (20 mil episódios contra as 500 da verificação original).

Não misture com os novos: a coluna `params_ator` distingue (cheio ≈ 36 547 /
78 132 entradas; podado ≈ 10 965 / 23 440).

## Escopo

Só `Na ≤ 2`. As corridas de `Na = 4` e `Na = 8` foram apagadas em 26/08, junto
com a redução da campanha a 1 e 2 drones — elas mediam células que saíram do
escopo, e ficariam aqui como 7,6 MB sem leitor. Estão no histórico do git, se
precisar.

O que fica são as **9 corridas com par na campanha atual**: mesma célula, mesmas
sementes, modelo cheio contra podado. É o que permite medir o custo da poda em
20 mil episódios, contra as 500 sementes da verificação original.
