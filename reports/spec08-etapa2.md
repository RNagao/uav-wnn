# SPEC-08 — Etapa 2: escada de ablação sob ruído de GPS (bloco L→R)

**Situação honesta:** a fundação da Etapa 2 está construída e validada; os
bloqueios **RF-05 (consistência) e RF-04 (não vazamento) passam**; o bloco
**L→R** da escada rodou de ponta a ponta com 500 sementes de avaliação. O
**portão G2 não é declarado** — e a razão é um achado, não uma falha de
infraestrutura: **no regime D13 (ruído só no GPS do drone, σ ≤ 10 m), o ruído não
degrada a política — ele a melhora**, e o KF, ao removê-lo, remove o ganho.

Este relatório reporta o que foi medido, a interpretação, e o que fica aberto
para R→R e para os degraus/varreduras adiados.

## 1. O que foi executado

| Bloco | Estado |
|---|---|
| Infra da escada 0–4 no ambiente ruidoso (degraus 2/3 ligados; inovação logada) | **feito** |
| TOST (equivalência) no harness estatístico | **feito** |
| Pré-registro (plano de análise + hash do desenho) — B3 | **feito** |
| Gate RF-05/RF-04 sob a **mobilidade real do WNN** — B1/B2 | **APROVADO** |
| Escada L→R (degraus 0–4 × σ ∈ {1,3,5,10}, 500 sementes, 1 semente de treino) | **feito** |
| R→R, degrau 5, H2 (WNN×MLP), varredura em T, T1/T2/T3 com filtro | **adiado** |

**Modelo:** rede da dissertação (`ntupla_dissertacao`), config vencedora da Etapa
1 (passo 8e-4, σ 0,35, T 32, n 12, φ 0,85), checkpoint de melhor validação.

## 2. Bloqueio B1 — RF-05/RF-04 (atestado `results/etapa2/gate_rf05.json`)

Medido com a **mobilidade real do WNN** (não o guloso): a consistência do filtro
tem de valer sob a trajetória que vai ser avaliada (T-08.1).

| Degrau | Parâmetro | NEES | NIS | Veredito |
|---|---|---|---|---|
| 3 (KF vel. const.) | melhor `q_a = 200` | consistente | consistente | consistente **por inflar Q** |
| 4 (KF + controle) | melhor `q_w = 4,0` | 2,12 (consistente) | consistente | **consistente** |

- **`q_w = 4,0`, não 1,0.** Sob a mobilidade do guloso o filtro era consistente
  com `q_w = 1`; sob a mobilidade do WNN (mais agressiva) o mesmo `q_w = 1` dava
  NEES ≈ 6,5 (otimista). Ajustar `q_w` ao descasamento de atuação **real** — como
  a T-08.1 manda — restaura a consistência. É a diferença que justifica atestar
  sob a política, não sob um proxy.
- **RF-04 verde** (`tests/test_vazamento.py`). A verdade não entra na observação.
- **degrau 4 consistente ⇒ o bloqueio B1 libera** a leitura de RL. (O degrau 3 só
  fica consistente inflando `Q` a `q_a = 200`, o que o transforma em quase-inércia
  — coerente com "mal especificado", T-07.10.)

## 3. A escada L→R (500 sementes, pareado por CRN)

Tempo de conclusão **censurado em 500 s**; SR ao lado. Teto (degrau 0, sem ruído):
**CT 130,4 s · SR 81 %**.

| σ (m) | d1 bruto | d2 passa-baixa | d3 KF-VC | d4 KF+ctrl | NEES d4 |
|---|---|---|---|---|---|
| 1 | 90,1 / 90% | 100,5 / 88% | 111,7 / 85% | 103,7 / 87% | 2,09 |
| 3 | 79,8 / 92% | 86,7 / 91% | 83,7 / 91% | 90,1 / 90% | 2,24 |
| 5 | 74,8 / 94% | 79,3 / 93% | 73,7 / 94% | 92,8 / 89% | 2,57 |
| 10 | 94,3 / 91% | 75,1 / 94% | 84,8 / 92% | 86,8 / 91% | 2,54 |

(CT em s / SR em %.)

**O dano do ruído é negativo em todo σ:** `t̄₁ − t̄₀ ∈ [−56; −36] s`. O degrau 1
(ruidoso) **conclui mais rápido e com mais sucesso** que o teto limpo.

## 4. O achado: o ruído é dithering, não dano

Diagnóstico controlado (100 sementes, mesmo cenário):

```
degrau 0 (referência):        CT 129,5 · SR 81%
degrau 1 σ=0 (reimplementado): CT 129,5 · SR 81%   ← idêntico → sem bug de caminho
degrau 1 σ=5:                  CT  71,3 · SR 95%   ← o ruído melhora
```

O caminho de observação reimplementado é **bit-idêntico** ao de referência em
σ = 0 (confirmado para 1 drone, além do teste de 2 drones), então a melhora vem
**só do ruído**. A leitura:

- A política limpa **empaca** em ~19 % dos cenários (estado determinístico que
  vira 500 s censurados). O ruído de GPS, mesmo 1 m, **perturba a observação o
  suficiente para quebrar o ciclo** — sobe o SR de 81 % para 88–95 %. É dithering
  determinístico clássico.
- **O KF remove o dithering.** O degrau 4, por construção, filtra o ruído de volta
  para perto da verdade — e com isso **remove o ganho**. Em σ = 5, o degrau 1 cru
  (74,8 s / 94 %) bate o degrau 4 (92,8 s / 89 %). O filtro não é neutro: ele
  **piora** ao restaurar o determinismo que empacava.

## 5. Vereditos e o portão G2

**H3 (L→R) não confirmado — e refutado quanto ao mecanismo esperado.**

| Teste | Resultado |
|---|---|
| degrau 4 ≡ degrau 0 (TOST, δ = 0,10·\|dano\|) | **não informativo** em todo σ (meia-largura ≈ 12 s > δ ≈ 4–6 s): não há dano a reparar, δ é minúsculo |
| degrau 4 > degrau 1 | **não** (Δ = +10 a +18 s; o KF é *pior* que o bruto) |
| degrau 4 > degrau 2 | **não** |
| degrau 4 > degrau 3 | **não** |

Pela tabela de desfechos da §6 da SPEC-08 (`4 ≯ 2/3`): resultado legítimo e
desconfortável. Aqui ele é mais forte que "o filtro barato basta" — **o filtro
atrapalha**, porque o que ajudava era o ruído.

**Estado dos critérios de G2:**

| # | Critério G2 | Estado |
|---|---|---|
| 1 | `gate_rf05.json` aprovado + RF-04 | ✅ (sob mobilidade real, q_w=4) |
| 2 | degrau 4 ≡ degrau 0 (TOST) | ⚠️ não informativo (sem dano a reparar) |
| 3 | degrau 4 > degraus 2 e 3 | ❌ **refutado** (degrau 4 é pior) |
| 4 | varredura σ×T + regra `bin≈σ` | ⏳ adiado (exige modelos por T) |
| 5 | vereditos H2 e H3 | H3 escrito; H2 adiado |
| 6 | T1/T2/T3 com filtro | ⏳ adiado |

**G2 não é declarado.** "Portão não atingido é resultado publicável" (PRD §2). O
resultado desta rodada é que, **no escopo D13 (ruído só no GPS próprio, σ ≤ 10 m),
o experimento da escada não tem o que mostrar a favor do KF** — o ruído é um
deslocamento comum pequeno que a política absorve (e aproveita), e o achado A3
(KF ≫ velocidade-constante, 18,8× no banco sintético de saltos de rumo) **não se
traduz** em ganho de tempo de conclusão no ambiente real nessas condições.

## 6. Consequências e próximos passos

1. **A causa é a própria D13.** A spec já avisava (§0): com ruído só na posição
   própria, "aproximar-se não compra informação". Este resultado é a versão
   quantitativa disso — o KF não compra desempenho porque não há dano de tarefa a
   reparar. Reabrir o canal do sensor (fora do escopo atual) traria o regime onde
   o KF morde.
2. **O dithering é achado sobre a política, não sobre o filtro.** A rede da
   dissertação empaca em ~19 % dos cenários limpos; ruído a destrava. Vale como
   observação sobre determinismo de políticas WNN gulosas — e sugere que uma
   política com exploração residual na avaliação não teria esse piso.
3. **Onde o KF ainda pode aparecer:** σ maior (onde o dano supere o dithering),
   R→R (treinar sob ruído pode remover a dependência do dithering), e a varredura
   σ×T (a regra `bin≈σ`). Tudo adiado, com o arnês pronto para rodá-los.

## 7. Ressalvas

- **Uma semente de treino.** Os IC são condicionais ao modelo da Etapa 1; o
  veredito pleno pede ≥ 5 (T-08.2). O sinal (KF pior, ruído ajuda) é grande e
  consistente nos quatro σ, mas a magnitude é de uma semente.
- **σ ≤ 10 m é brando** para a escala do cenário; a grade não contém um regime
  onde o ruído domine o dithering.
- Artefatos: `results/etapa2/gate_rf05.json` (atestado), `results/etapa2/escada_lr.json`
  (por célula, com IC pareado), `configs/etapa2/experimento.json` +
  `results/etapa2/analysis_plan.md` (desenho congelado e hasheado).

## 8. Reprodução

```bash
# atestado RF-05/RF-04 sob a mobilidade do WNN (B1/B2)
uv run python scripts/spec08_gate_rf05.py --checkpoint <run>/checkpoints/melhor --episodios 100
# escada L→R, 500 sementes
uv run python scripts/spec08_escada.py --checkpoint <run>/checkpoints/melhor --episodios 500
```
