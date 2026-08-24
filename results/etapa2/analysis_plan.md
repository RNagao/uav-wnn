# SPEC-08 — Plano de análise pré-registrado (bloqueio B3)

**Congelado antes da primeira rodada de RL.** Este documento fixa métricas
primárias, margens, sinais esperados, família de testes e correção de
multiplicidade **antes** de qualquer número sair, para que não exista uma leitura
escolhida depois dos dados. É o bloqueio B3 da §4 da SPEC-08.

**Desenho hasheado:** `configs/etapa2/experimento.json`
`sha256 = b86bbb8a8d6f4169280dd29926e0330925e3fafb482c5d5b0a2573cb4424f78d`
(ver `results/etapa2/analysis_plan.sha256`). Alterar o desenho **invalida** este
pré-registro e o atestado do gate; ambos têm de ser refeitos.

## 1. Métrica primária e secundária

- **Primária:** tempo de conclusão **censurado em `max_episode_length = 500 s`** —
  episódios não concluídos entram com 500 s, nunca descartados. Comparar médias só
  sobre sucessos é enviesado quando as taxas de sucesso diferem entre degraus, e
  elas vão diferir.
- **Secundária obrigatória:** taxa de sucesso (SR), reportada **sempre ao lado**.
- Nível absoluto reportado junto: toda conclusão é **relativa ao próprio degrau 0**
  (a política limpa pode ser medíocre; a escada mede degradação relativa).

## 2. Intervalos de confiança

Bootstrap **hierárquico** de 1000 réplicas: reamostra `seed_train` (nível 1) e,
dentro de cada, os episódios (nível 2), aplicando o **mesmo** conjunto de índices
aos dois braços para preservar o pareamento. Reporta a **diferença pareada** com
IC — nunca dois ICs marginais sobrepostos. O IC condicional a uma semente é
reportado à parte, identificado como tal.

## 3. Pareamento por números aleatórios comuns (CRN)

- Cenário do episódio `e` fixado por `seed_eval[e]`, **idêntico** em todos os
  degraus, σ, T e regimes. O degrau 0 é reavaliado nos mesmos episódios da Etapa 1.
- Ruído pré-sorteado como `z_{e,t} ~ N(0, I₂)`; a medida é `p̃ = p + σ·z_{e,t}`.
  Células de σ diferente e degraus 1–4 compartilham o mesmo `z` — a diferença
  entre degraus é atribuível **só** ao estimador. O degrau 0 consome o fluxo sem
  usar, para não desalinhar a sequência de decisões.

## 4. Testes, sinais esperados e correção

**Superioridade** (unilateral, via IC bootstrap da diferença pareada excluindo
zero), família corrigida por **Holm**:

| Comparação | Sinal esperado |
|---|---|
| 4 > 1 | degrau 4 muito melhor que o bruto |
| 4 > 2 (sem comando) | degrau 4 melhor |
| 4 > 2 (com comando) | degrau 4 melhor — a ablação honesta de "usar o comando" |
| 4 > 3 | degrau 4 melhor; degrau 3 possivelmente pior que 1 |

**Equivalência** (TOST, forma por IC de (1−2α); ver `uavwnn.eval.stats.tost`):

| Comparação | Margem δ | Predição |
|---|---|---|
| 4 ≡ 0 | `0,10·(t̄₁ − t̄₀)` na célula | equivalente |
| RR ≡ LR (degrau 4) | mesma δ | equivalente ⇒ retreinar sob ruído não compra nada |

- δ **autocalibrada**: 10 % do dano que o ruído causa sem filtro, na mesma célula.
- Célula com `t̄₁ − t̄₀` pequeno (σ = 1 m, tipicamente) é **não informativa**, não
  "equivalência trivial". A **meia-largura** do IC é reportada ao lado de δ;
  equivalência com meia-largura > δ é falta de poder, não resultado.
- **Controle positivo (R→R):** a equivalência no degrau 4 só é informativa se, no
  degrau **1**, R→R for significativamente melhor que L→R. Roda **antes** da
  leitura do degrau 4. Se falhar, o braço R→R é reportado como não informativo.

## 5. Vereditos que este plano habilita

- **H3** confirmado se `4 ≡ 0` por TOST **e** `4 > 2` e `4 > 3` com IC separado
  após Holm. Se `4 > 2,3` mas `4 ≢ 0`, veredito **parcial**, com a lacuna residual
  quantificada em segundos e em fração de `t̄₁ − t̄₀`, e diagnosticada como filtro
  vs resolução do codificador (regra `bin ≈ σ`) pela varredura σ×T.
- **H2** sobre a **interação** aproximador × σ: IC de `D_WNN(σ) − D_MLP(σ)`, onde
  `D_m(σ) = (t̄_m(σ) − t̄_m(0))/t̄_m(0)`. (Adiado neste incremento — exige MLP.)

## 6. Bloqueios que precedem qualquer rodada de RL (§4)

1. **B1 (RF-05):** `results/etapa2/gate_rf05.json` com `status: aprovado`, NEES/NIS
   no intervalo qui-quadrado por degrau (2, 3, 4), com o **hash da config de
   filtro** que foi validada. O degrau 3 pode reprovar legitimamente (mal
   especificado); o que bloqueia é o **degrau 4** reprovar.
2. **B2 (RF-04):** teste de não vazamento verde para todos os degraus com ruído.
3. **B3 (este documento):** hash de `experimento.json` registrado acima.

`scripts/spec08_escada.py` recusa-se a rodar enquanto B1 não existir aprovado com
o hash de filtro correspondente — o bloqueio é mecânico, não uma intenção.
