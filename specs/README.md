# Specs de execução — `uav-wnn`

Decomposição do PRD (`../PRD.md`) em unidades de trabalho pequenas, cada uma com escopo fechado, critério de saída **medido** e dependências explícitas.

**Regras:**
1. Uma spec não começa antes de suas dependências saírem.
2. Uma spec não termina sem seu critério de saída **medido** — não "implementado", medido.
3. Toda alteração ao problema original de Lamenza exige linha nova na **§5.4 do PRD** e decisão registrada na §10. O padrão é não alterar.
4. Toda spec entrega um relatório curto com figuras, que vai direto para o capítulo correspondente de E2. Nada é medido duas vezes.

## Divisão

| # | Spec | Natureza | Etapa do PRD | Depende de | Est. | Estado |
|---|---|---|---|---|---|---|
| **00** | [Ambiente fiel a Lamenza](SPEC-00-ambiente.md) | infra | 1 | — | 3 d | escrita, em revisão |
| **01** | [Estudo de binarização](SPEC-01-binarizacao.md) | estudo | pré-1 | — | 3–4 d | escrita, em revisão |
| **02** | [Baselines e harness de avaliação](SPEC-02-baselines.md) | infra | 1 | 00 | 2 d | escrita, em revisão |
| **03** | [Harness de custo — T1/T2/T3](SPEC-03-custo.md) | infra | todas | — | 2 d | escrita, em revisão |
| **04** | [Núcleo n-tuple esparso](SPEC-04-ntuple.md) | modelo | 1 | 01 | 2 d | escrita, em revisão |
| **05** | [n-tuple Actor-Critic + MLP AC pareado](SPEC-05-actor-critic.md) | modelo | 1 | 04 | 3 d | escrita, em revisão |
| **06** | [Etapa 1 — réplicas 1a/1b e varreduras → **G1**](SPEC-06-etapa1.md) | experimento | 1 | 00, 02, 03, 05 | 4 d | escrita, em revisão |
| **07** | [Ruído, KF com entrada de controle, NEES/NIS](SPEC-07-fusao-local.md) | fusão | 2 | 00 | 3 d | escrita, em revisão |
| **08** | [Etapa 2 — escada 0–5 e varredura σ×T → **G2**](SPEC-08-etapa2.md) | experimento | 2 | 06, 07 | 3 d | escrita, em revisão |
| **09** | [Etapa 3 — multiagente e interseção de covariância → **G3**](SPEC-09-etapa3.md) | fusão + experimento | 3 | 08 | 4 d | escrita, em revisão |
| **10** | [Reprodutibilidade, relatório e slides](SPEC-10-entrega.md) | entrega | — | contínua | contínua | escrita, em revisão |

Todas as specs foram escritas e passaram por uma revisão cruzada; os achados que exigem decisão humana estão em [`REVISAO.md`](REVISAO.md).

## Racional da divisão

**Separação entre infra, modelo, experimento e fusão.** Specs de infra (00, 02, 03) entregam capacidade e são testáveis sozinhas. Specs de modelo (04, 05) entregam aproximadores validados fora do ambiente. Specs de experimento (06, 08, 09) só produzem números e portões — se uma delas está escrevendo código de infraestrutura, a divisão falhou.

**SPEC-01 antes de SPEC-00, de propósito.** O estudo de binarização não depende do `gradysim`: a observação de Lamenza é função pura das posições, então o corpus é analítico. Isso desacopla a decisão mais cara de reverter (o codificador) da peça de engenharia mais chata (a ponte orientada a eventos).

**SPEC-03 é transversal e vem cedo.** Tamanho de modelo e tempo de inferência são figuras de mérito de primeira classe (PRD §5.5, D11). Se o harness de custo só aparecer no fim, os números saem improvisados e sem protocolo.

**SPEC-07 é independente do RL.** Ruído e filtro de Kalman são testáveis contra trajetórias sintéticas, sem política treinada. Adiantá-los protege a Etapa 2 do risco de cronograma — é a spec que mais se beneficia de rodar em paralelo.

## Paralelismo

```
dia 1 ──────────────────────────────────────────────────────────► dia 28
 SPEC-01 ═════╗
 SPEC-03 ═════╬═════════════════╗
 SPEC-00 ═════╬══► SPEC-02 ═════╬═══════════════╗
              ╚══► SPEC-04 ══► SPEC-05 ═════════╬═► SPEC-06 (G1) ═╗
              ╚══► SPEC-07 ════════════════════════════════════════╬═► SPEC-08 (G2) ═► SPEC-09 (G3)
 SPEC-10 ════════════════════════════════════════════════════════════════════════════════════════
```

## Portões

| Portão | Spec | Critério |
|---|---|---|
| **G0** | 01 | Codificador escolhido com evidência medida; `(n, N)` pré-selecionados |
| **G1** | 06 | Réplica 1a vs os 100 % publicados; 1b dentro do IC do guloso; tabela de custo completa; veredito de H1 |
| **G2** | 08 | NEES/NIS no intervalo qui-quadrado; degrau 4 indistinguível do 0 e superior a 2 e 3 |
| **G3** | 09 | 6c supera 6a com NEES/NIS consistente; 6b demonstravelmente inconsistente; vereditos de H5 e H6 |
