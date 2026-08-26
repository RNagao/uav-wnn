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
