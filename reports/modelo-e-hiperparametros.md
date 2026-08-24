# Modelo e hiperparâmetros — procedência dos dados

Registro de **qual modelo e quais hiperparâmetros** geraram cada resultado deste
projeto, e por que o modelo é **parcial** (não o vencedor definitivo do grid).

## A configuração usada em TODOS os dados da dissertação

| item | valor |
|---|---|
| aproximador | **`ntupla_dissertacao`** (gradiente funcional §4.3.1 + crítico NTRN com esquecimento §4.2.1) |
| passo (taxa do ator, α) | **8e-4** (convenção de média → `taxa_ator = passo`) |
| σ (normal projetada) | **0,35** |
| T (resolução do termômetro) | **32** |
| n (tamanho da tupla) | **12** |
| φ (esquecimento do crítico) | **0,85** |
| m (passos do Actor-Critic) | **8** |
| γ (desconto) | **0,99** |
| d_saída | **3** (u₁, u₂, μ_s — normal projetada) |
| réplica | **1b** (recompensa `punish`/eq. 5-6, escala 1,0, horizonte 500 s) |
| `c_max` (guarda de estabilidade) | **0,5** |

**Fonte:** é o **ponto #1 de `PONTOS_LONGA`** (`spec06_busca.py`) — o **melhor
ponto da parcial do grid** (`desempate`): 50,1 % de cobertura, SR 21,3 %, 0/3
abortos. É a mesma config em 1 drone e no multiagente.

`N = ⌈D·T/n⌉` cresce com Na (é o análogo estrutural da entrada do MADDPG crescer):
**Na=1 → 67 · Na=2 → 72 · Na=4 → 83 · Na=8 → 104**.

## Onde essa config foi usada (checkpoints)

| dado / experimento | célula | checkpoint |
|---|---|---|
| Figuras de custo T1/T2/T3 (PR #18) | Na=1/Ns=12 | `runs/spec06-longo/spec06-busca-1b/15b49e642647/0` |
| SPEC-08 — ruído/escada L→R (PR #17) | Na=1/Ns=12 | idem |
| Escalada multiagente (PR #19) | Na=2/4/8 / Ns=24 | `runs/spec06-multi/ntupla_dissertacao-a{2,4,8}s24` |
| Paths (PR #19) | Na=2/4/8 | checkpoints `melhor` das corridas acima |

**Adições exclusivas do multiagente** (não mudam a rede): **parameter sharing**
(rede compartilhada + Na trajetórias de AC) e **recompensa/Na** (normalização que
estabiliza; escalar positivo → não muda a política ótima). Réplica e demais
hiperparâmetros idênticos ao de 1 drone.

## Por que é PARCIAL (não o vencedor definitivo)

- O **grid search não fechou** — a config foi eleita sobre a **parcial** do
  `desempate` (17–22 % das 720 corridas).
- **Uma** semente de treino (o protocolo pede ≥5 para declarar hipótese).
- O **braço PPO** (`ntupla_anbubus` via PPO) **não foi testado**; a família
  `anbubus` **abortou** nas corridas longas (1,5e-3 aos 115 k; 5e-4 aos 722 k).
- **Nenhum portão (G1, G2) declarado.** Vereditos são provisórios.

## Reconciliação — para não confundir com outras "provisórias"

Houve **três** configs `passo/σ` no histórico; só uma gerou os dados:

| config | o que é | usada nos dados? |
|---|---|---|
| **8e-4 / 0,35** (`PONTOS_LONGA` #1) | melhor ponto da parcial | **SIM — todos os dados** |
| 5e-4 / 0,20 (§9 do `grid-search.md`) | ponto **seguro** para dev paralelo | não |
| 5e-4 / 0,20 (1ª tentativa ad-hoc, `spec06_treino_wnn.py`) | standalone sem a correção `(1−γ)`/reward | **abortou** (c_max, 6,84 M) — superada |

Os `anbubus` treinados (`6356fa883ca1` = 5e-4/0,20; `88b75c732880` = 1,5e-3/0,35)
foram tentativas do outro braço — **abortaram ou ficaram no piso**, e **não**
entram nos dados reportados.

## Ressalva de método (multiagente)

O MADDPG do Lamenza é **CTDE** (crítico **centralizado**); este AC (Algoritmo
4.4) tem crítico **descentralizado** on-policy. O ator implantado é
descentralizado+compartilhado nos dois (comparável no deploy); os algoritmos de
treino diferem → WNN×MADDPG é comparação **cross-método**. A comparação "mesmo
algoritmo" do projeto é WNN-AC × MLP-AC.
