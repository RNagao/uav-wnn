# uav-wnn — Relatório geral: o que foi feito e conclusões até aqui

**Escopo do trabalho.** Substituir o aproximador de função do problema de coleta
de dados por UAVs de Lamenza (2025) — troca do **MADDPG** (RL profundo) por uma
**rede sem peso n-tuple** (Katopodis, 2022) — e acrescentar **ruído de GPS no
drone com fusão por filtro de Kalman**. O problema em si (cenário, observação,
ação, recompensa, métricas) é reproduzido sem alteração; mudam só o aproximador e
o canal de ruído.

> ## ⚠️ Ressalva central sobre o modelo
> Todos os números de desempenho deste relatório vêm de **um** modelo WNN,
> **provisório e parcial** — a rede da dissertação (`ntupla_dissertacao`) na
> configuração eleita sobre uma **fração** do grid search (não o vencedor
> definitivo), treinada em **uma única semente** na célula **Na=1, Ns=12**.
> **O braço PPO (`anbubus` via PPO) não foi testado**, e a família `anbubus`
> abortou nas corridas longas. Portanto:
> - Nenhum portão (G1, G2) é declarado.
> - Os vereditos são **provisórios** e condicionais a essa semente/config.
> - A comparação de **desempenho** WNN×MADDPG por cenário ainda não existe (não
>   há célula em comum; ver §7).
> As **figuras de custo** (tamanho, inferência, operações), por serem
> propriedades do modelo, são as conclusões mais firmes até aqui.

---

## 1. O que foi construído (estado das specs)

| # | Spec | Estado |
|---|---|---|
| 00 | Ambiente de referência | concluída |
| 01 | Binarização (termômetro) | concluída — G0 |
| 02 | Baselines + harness estatístico | concluída — MADDPG medido, bootstrap pareado calibrado (A/A) |
| 03 | Harness de custo T1/T2/T3 | T1/T3 concluídos; **T2 medido agora** (este ciclo) |
| 04 | Núcleo n-tuple esparso | concluída — duas famílias, desvio zero vs `Ramnet.jl` |
| 05 | Actor-Critic (Alg. 4.4) + MLP pareado | Algoritmo implementado; MLP pareado parcial |
| 06 | Etapa 1 — grid + treino → G1 | **grid parcial**; config provisória; **1 modelo treinado**; G1 **não** declarado |
| 07 | Ruído, KF, NEES/NIS | concluída — escopo D13 (ruído só no GPS do drone) |
| 08 | Etapa 2 — escada sob ruído → G2 | **fundação + bloco L→R** feitos; G2 **não** declarado |
| 09 | Etapa 3 — multiagente, interseção de covariância | não iniciada |
| 10 | Entrega / reprodutibilidade | contínua |

---

## 2. O modelo treinado (SPEC-06) — e por que é provisório

- **Config:** `ntupla_dissertacao`, passo 8e-4, σ 0,35, T 32, n 12 (N=67), φ 0,85,
  m 8; célula Na=1, Ns=12; réplica 1b.
- **Treino:** ~6,75 M passos, **parada antecipada** com cobertura de validação = 1,0.
- **Desempenho (limpo, 300–500 sementes):** SR 81–91%, cobertura ~90%, tempo de
  sucesso ~40 s. Referências: guloso 100% / ~34 s; aleatório cobertura ~3,3%.

**Por que não é o vencedor definitivo:**
- O grid search fechou só parcialmente; a config foi eleita sobre a **parcial**.
- **Uma** semente de treino (o protocolo pede ≥5 para declarar hipótese).
- O braço **`anbubus`** (a segunda família WNN) **abortou** em corrida longa nos
  dois pontos testados (passo 1,5e-3 aos 115 k; 5e-4 aos 722 k) — a assertiva de
  estabilidade `c_max` disparou. **O braço PPO do `anbubus` não foi avaliado.**
- Achado do grid (parcial): no regime estável (passo ≤ 5e-4) a dissertação supera
  o anbubus (+4,3 pp), mas o ótimo do anbubus é instável.

---

## 3. Figuras de mérito de custo — WNN × MADDPG (conclusão firme)

Propriedades do modelo, medidas (SPEC-02/03), pouco sensíveis ao cenário. Modelos
de **deploy** (`models/deploy`), a mesma fonte da latência medida.

| eixo | WNN (a2s24, Na=2, deploy) | MADDPG (256×256, Na=2) | vencedor |
|---|---|---|---|
| **T1 pico de RAM (processo, medido, 100 processos)** | 45,9 ± 0,1 / 53,3 ± 0,2 MB (NumPy) | 506,0 ± 0,1 MB (PyTorch) | **WNN** no processo (~0,1×) — quase todo o pico do MADDPG é arcabouço |
| **T2 inferência (ciclos)** | 4,8 k (x86) · 21 k (M0) | 147 k (x86) · 9,15 M (M0) | **WNN** — 31× (x86) a **438× (Cortex-M0 sem FPU)** |
| **T2 inferência (latência medida)** | 22–25 µs (a2s24, NumPy) | 49 µs (PyTorch) | **WNN** — ~2× no relógio |

**Mesma célula (a2s24, Na=2)** dos dois lados — WNN de deploy e MADDPG publicado —,
sem a ressalva de "célula diferente" de versões anteriores. Como cada figura foi
calculada: ver `spec06-merito.md` (seção *Metodologia*).

- **Latência empírica (SPEC-03, a2s24):** um processo por modelo, mesma observação,
  menor mediana de 8×1000 — a WNN é **~2× mais rápida** que o ator MADDPG, mesmo
  NumPy × PyTorch. (Corrige uma versão anterior que citava 173/35 µs hardcoded
  dizendo o wall-clock invertido.) O overhead de arcabouço achata a vantagem que os
  ciclos preveem; compilada, a WNN realizaria mais dela.
- **KF da fusão:** ~20 µs/decisão medidos (protótipo NumPy) = **~0,001 %** do
  orçamento de tempo real de 2 s — e é **termo comum** (WNN e MADDPG usam o mesmo,
  cancela em H1). Contabilizado à parte, não cobrado de nenhuma política.
- **Operações fp (MACs):** figura **retirada** — contar multiplicação favorece a
  WNN por construção (ela é *weightless* e quase não multiplica); o custo de
  computação passou a ser reportado por ciclos e pela latência empírica.

**Conclusão:** a rede sem peso **ganha em inferência** — nos ciclos e no relógio
medido —, com vantagem que explode em hardware **sem FPU**. Em memória a figura é
o **pico de RAM medido** do processo (a comparação por bytes de artefato foi
retirada): o processo da WNN cabe em ~46–53 MB contra ~506 MB do MADDPG em PyTorch (média de
100 processos, desvio < 0,2 MB), mas quase todo esse pico é arcabouço — descontado,
o ator MADDPG acresce 10 MB e a WNN 15–22 MB (o load monta dicionários antes da LUT plana). É o lado de custo de H1,
e é o resultado mais sólido do trabalho até aqui (independe de a política ser ótima).

---

## 4. Ruído, filtro de Kalman e consistência (SPEC-07)

- Infra construída e testada: modelo de ruído (branco / Gauss-Markov), **KF com
  entrada de controle** (degrau 4), passa-baixa (2) e KF de velocidade constante
  (3), **NEES/NIS** (consistência), e o invariante **RF-04** (a verdade não vaza
  para a política).
- Escopo **D13** (2026-08-15): ruído **só no GPS do drone** — um deslocamento
  comum a toda a observação. O canal do sensor e a hipótese de busca ativa (H4)
  saíram.

---

## 5. Etapa 2 sob ruído (SPEC-08) — o achado do *dithering*

Bloqueios de qualidade passaram: **RF-05** (consistência do filtro sob a
mobilidade **real** do WNN — `q_w=4` dá NEES 2,12/NIS consistente) e **RF-04**
(não vazamento) verdes; plano de análise pré-registrado e hasheado.

Escada de ablação em L→R (500 sementes, σ ∈ {1,3,5,10}). **Achado central:**

> No regime D13 (ruído só no GPS próprio, σ ≤ 10 m), **o ruído não degrada a
> política — ele a melhora.** A rede limpa **empaca** em ~19 % dos cenários
> (coleta ~90 % dos sensores mas não fecha o episódio); o ruído funciona como
> *dithering* que quebra esses estados. E o **KF, ao filtrar o ruído, remove esse
> ganho.**

Consequência para G2: o degrau 4 (KF) **não supera** os degraus 2/3, e a
equivalência 4≡0 é não informativa (**não há dano a reparar**). O portão **não é
declarado** — resultado publicável (PRD §2), coerente com o próprio aviso da D13.

---

## 6. Desempenho sob ruído, com e sem filtro (300 sementes)

| config | tempo conclusão | SR | cobertura |
|---|---|---|---|
| limpo (sem ruído) | 130,7 s | 81% | 89,8% |
| **ruído CRU, sem filtro** | **78–100 s** | **90–93%** | **94–98%** |
| ruído + KF | 83–104 s | 87–92% | 92–96% |

**Sem o filtro a rede vai melhor** — e por margem estatisticamente significativa
contra o KF em σ=1 e σ=5 (IC pareado exclui zero). Ou seja: **a WNN de 1 drone
aguenta o ruído de GPS sozinha, sem filtro, até σ=10 m**, e o KF neste regime
brando só atrapalha. Isso é positivo para a proposta (robustez), e negativo para
a tese do filtro **neste escopo** — que precisaria de σ maior ou do canal do
sensor (cortado pela D13) para ter dano a reparar.

---

## 7. Conclusões até aqui

1. **Custo (H1):** a rede sem peso é estruturalmente mais barata em inferência
   (31–438× menos ciclos, sobretudo sem FPU; **~2× mais rápida no relógio medido**),
   na **mesma célula** (a2s24). Em memória ela **paga**: o T=32 ocupa ~2× o ator
   MADDPG e o T=16 empata. É a contribuição mais firme e independe da qualidade da
   política.
2. **Robustez a ruído:** a WNN treinada tolera ruído de GPS (σ ≤ 10 m) **sem
   filtro**, e até se beneficia dele (*dithering*). A cobertura sobe de ~90 % para
   94–98 %.
3. **O filtro, no escopo D13, não compra desempenho** — não há dano de tarefa a
   reparar. É um resultado honesto que redireciona a tese do KF para regimes mais
   severos.
4. **Desempenho absoluto:** a política (provisória, 1 semente) fica **abaixo do
   guloso ideal** (SR 81–93% vs 100%) — sólida, não ótima.

---

## 8. Ressalvas e limitações

- **Modelo provisório e parcial** (grid incompleto; 1 semente; **PPO não
  testado**; anbubus instável). Reforçado em toda leitura.
- **Head-to-head de desempenho ainda parcial e injusto no orçamento:** só Na=2
  fechou (MADDPG vence), com o WNN em 1/5 do orçamento e 1 semente. Na=4/8 em
  curso; veredito justo pede orçamento igual (50 M) e ≥5 sementes.
- **G1 e G2 não declarados.**
- σ ≤ 10 m é brando; a grade não contém um regime onde o ruído domine o dithering.
- Wall-clock de inferência não é comparável entre arcabouços (NumPy × PyTorch).

---

## 8.5. Escalada multiagente e head-to-head vs MADDPG (SPEC-06 T-06.5)

Treinamos a WNN da dissertação nos cenários **multiagente** (Na∈{2,4,8}, Ns=24)
com a **mesma** config parcial (8e-4/σ0,35/T32/n12/φ0,85/m8), **parameter sharing**
(uma rede compartilhada + Na trajetórias de AC) e **recompensa/Na** (normalização
que estabiliza — sem ela o δ estoura o `c_max` em Na>1; medido: aborto em Na=2).
Isso dá, pela primeira vez, **células em comum** com o MADDPG.

**Head-to-head parcial (Na=2/24, avaliação pareada nas mesmas 100 sementes):**

| | tempo de conclusão | SR | cobertura |
|---|---|---|---|
| **MADDPG** | **28,0 s** | **100%** | **100%** |
| WNN (10 M, 1 semente) | 80,9 s | 91% | 96,7% |

**MADDPG vence** este cenário (~2,9× mais rápido, SR e cobertura maiores). **Mas
não é comparação justa de método:** o WNN treinou com **1/5 do orçamento** do
Lamenza (10 M vs 50 M) e **1 semente**. O eixo de custo segue a favor do WNN.

**Coordenação (paths, `reports/img/spec06-path/`):** Na=2 e Na=4 **dividem a arena
e fecham** (24/24 — Na=4 em quatro quadrantes limpos); **Na=8 aglomera** numa
região e deixa gaps (21/24, não fecha, SR 0) — o limite estrutural do crítico
**descentralizado** onde o CTDE do MADDPG mais ajudaria. (Na=4/8 ainda treinando;
regenerar dos checkpoints finais.)

---

## 9. O que falta

- **Fechar o grid search**, incluindo o **braço PPO** — só então há vencedor
  definitivo.
- **Escalada T-06.5** (treinar a WNN em Na∈{2,4,8}) — **em curso**; Na=2 fechou,
  Na=4/8 rodando. Falta orçamento igual (50 M) e ≥5 sementes para veredito justo.
- Etapa 2: braço **R→R** (treino sob ruído), degrau 5, H2 (WNN×MLP), varredura
  σ×T; e um regime de ruído onde o KF tenha dano a reparar.
- **Etapa 3 (SPEC-09):** multiagente e interseção de covariância.
- Múltiplas sementes de treino para os vereditos.

---

## Artefatos

- Custo: `reports/spec06-merito.md` + `reports/img/spec06-merito/`.
- Etapa 2: `reports/spec08-etapa2.md`, `results/etapa2/` (gate + escada).
- Multiagente: `scripts/spec06_treino_multiagente.py`, `runs/spec06-multi/`,
  paths em `reports/img/spec06-path/`.
- Procedência (modelo + hiperparâmetros): `reports/modelo-e-hiperparametros.md`.
- Grid/treino: `reports/spec06-grid-search.md`, `reports/spec06-etapa1.md`.
- Baselines/custo: `reports/spec02-baseline-maddpg.md`, `reports/spec03-custo.md`.
- Fusão: `reports/spec07-*`.
