# Modelo e hiperparâmetros — procedência dos dados

Registro de **qual modelo e quais hiperparâmetros** geraram cada resultado deste
projeto. **Atualizado após o fechamento do grid search:** a config abaixo, antes
rotulada "parcial", foi **confirmada como 1ª de 120 pontos** na grade completa.

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

**Fonte:** é o **ponto #1 de `PONTOS_LONGA`** (`spec06_busca.py`), eleito sobre a
**parcial** do `desempate` e depois **confirmado pela grade completa** (720 corridas,
120 pontos, 3 sementes): **1º lugar** com 50,1 % de cobertura, SR 21,3 %, 0/3
abortos. A grade fechada **não trocou o vencedor**. É a mesma config em 1 drone e no
multiagente.

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

## O que já se resolveu, e o que continua em aberto

**Resolvido pelo fechamento do grid:**

- A grade **fechou** — 720 corridas, 120 pontos por família. A config é **1ª de 120**;
  a escolha feita sobre a parcial se sustentou.
- O **braço PPO foi testado** — **153 corridas / 51 pontos** (`spec06_grid_ppo.py`).
  O melhor ponto chega a **44,6 % de cobertura com SR 0,0 %**; o melhor SR de qualquer
  ponto é **3,3 %**, e **1 corrida em 153** passa de 5 % de SR. **49 das 153 falharam.**
  O braço PPO **não** produz um competidor.

**Continua em aberto:**

- **Duas** sementes de treino na corrida longa (o protocolo pede ≥5 para declarar
  hipótese).
- **Nenhum portão (G1, G2) declarado.** Vereditos seguem provisórios nesse sentido.
- A grade ordena a **200 mil passos**, 0,4 % do orçamento de operação. `T = 16`
  (3º de 120, SR 0,7 % na grade) **empata em SR** com o 1º na corrida longa, usando
  metade da memória — ver §9 de `spec06-grid-search.md`. A ordenação no topo da grade
  não é evidência confiável do regime de operação.

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
