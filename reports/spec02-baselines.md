# SPEC-02 — Baselines e harness de avaliação

Fecha a spec. Reprodução:

```bash
uv run python scripts/spec02_baselines.py --n-episodios 500   # ~3 min
uv run python scripts/spec02_aa.py --execucoes 200            # teste A/A
```

Artefatos em `results/spec02-baselines/`: `agregado.csv`, `comparacoes.csv` e,
por célula, `episodios.csv` (grão fino, fora do git) e `manifest.json`.

---

## 1. Os três baselines, 500 episódios pareados

`Ns = 12`, `phase="eval"`. Todas as políticas nos **mesmos** 500 cenários.

| célula | política | `t_conc_pen` | IC95 % | sucesso | mediana | p90 | Gini | redund. |
|---|---|---|---|---|---|---|---|---|
| `Na1` | `random_uniforme` | 500,00 | [500,00; 500,00] | 0,000 | 500,00 | 500,00 | — | 0,0 |
| `Na1` | `guloso` | 33,85 | [33,42; 34,24] | 1,000 | 34,00 | 39,50 | 0,000 | 0,0 |
| `Na1` | **`kmeans_tsp`** | **31,12** | [30,79; 31,48] | 1,000 | 31,00 | 36,00 | 0,000 | 0,0 |
| `Na2` | `random_uniforme` | 500,00 | [500,00; 500,00] | 0,000 | 500,00 | 500,00 | 0,329 | 0,1 |
| `Na2` | `guloso` | 32,51 | [32,03; 32,99] | 1,000 | 33,00 | 39,00 | 0,469 | **10,9** |
| `Na2` | **`kmeans_tsp`** | **16,99** | [16,75; 17,21] | 1,000 | 17,00 | 20,50 | 0,101 | 0,1 |
| `Na4` | `random_uniforme` | 499,43 | [498,28; 500,00] | 0,002 | 500,00 | 500,00 | 0,537 | 0,2 |
| `Na4` | `guloso` | 30,64 | [30,10; 31,20] | 1,000 | 31,50 | 38,00 | 0,707 | **10,9** |
| `Na4` | **`kmeans_tsp`** | **9,38** | [9,25; 9,49] | 1,000 | 9,50 | 11,50 | 0,198 | 0,4 |

`LB-1` fica em 6,4 a 6,5 s nas três células.

**Ordenação esperada confirmada nas três células**, com as comparações adjacentes
todas conclusivas (`0 ∉ IC` da diferença pareada). Isso valida o harness — é o
critério de saída T-02.9 §1.

## 2. O achado: o guloso não escala com `Na`

| política | `Na = 1` | `Na = 4` | ganho |
|---|---|---|---|
| `guloso` | 33,85 s | 30,64 s | **1,10×** |
| `kmeans_tsp` | 31,12 s | 9,38 s | **3,32×** |

O ideal seria ~4×. **Quadruplicar o número de drones acelera o guloso em 9,5 %.**

E o mecanismo está medido: as **coletas redundantes** — sensores coletados com
mais de um UAV no alcance simultaneamente — saltam de 0,0 (`Na = 1`, sem pares)
para **10,9** em `Na = 2` e permanecem 10,9 em `Na = 4`. O Gini das visitas
**piora** com mais agentes (0,469 → 0,707), o oposto do que divisão de trabalho
produziria.

O `kmeans_tsp` fica em 0,1 a 0,4 coletas redundantes e Gini 0,101 a 0,198.

Isto **reforça a TS4** (a barra da H6) com um número novo. Não é só que o
baseline da fig. 7.3 não coordena — é que **acrescentar drones a ele quase não
ajuda**, e portanto "o enxame supera a heurística gulosa" é ainda mais fraco do
que parecia.

## 3. Decomposição limpa da vantagem do planejador

Em `Na = 1` não há o que particionar, então o `kmeans_tsp` degenera em *ordenação
ótima de rota contra vizinho-mais-próximo*. Isso separa as duas fontes de ganho:

| fonte | ganho sobre o guloso |
|---|---|
| **roteamento** só (`Na = 1`) | **8,1 %** |
| roteamento + particionamento (`Na = 2`) | 47,7 % |
| roteamento + particionamento (`Na = 4`) | 69,4 % |

**A quase totalidade da vantagem é particionamento, não roteamento.** Ordenar
bem a rota de um único UAV vale 8 %; dividir o trabalho entre quatro vale 69 %.

É um resultado sobre o problema, não sobre o método: o gargalo da coleta por
enxame é *quem vai onde*, não *em que ordem*.

## 4. LB-1 e quão frouxo ele é

| célula | `kmeans_tsp` / LB-1 |
|---|---|
| `Na1` | 4,79× |
| `Na2` | 2,63× |
| `Na4` | 1,46× |

LB-1 ignora que um agente não pode estar em dois lugares, então fica cada vez
menos frouxo à medida que há mais agentes — em `Na = 4` já está a 46 % do
planejador. É limite **válido** e barato, e serve de asserção: um resultado
abaixo dele é bug, e o harness levanta.

## 5. O que a tabela **não** autoriza a dizer

**`kmeans_tsp` não é limite inferior e não é ótimo.** O k-means minimiza inércia,
não *makespan*; o problema alvo (mTSP min-max com alcance de coleta) é NP-difícil
e não está sendo resolvido. A rotulação correta é **"planejador centralizado
offline com posições verdadeiras"**, e a coluna `usa_estado_verdadeiro` carrega
isso na própria tabela.

**A comparação com o MADDPG não é pareada.** Os números de
`reports/spec02-baseline-maddpg-rollout.md` vêm de `Ns ∈ {12,24,36}` com
`Na ∈ {2,4,8}` e de outra árvore de sementes. Comparar `kmeans_tsp` (9,38 s em
`Na4_Ns12`) com o MADDPG (10,07 s em `a_4-s_12`) é sugestivo — o planejador com
verdade fica ~7 % à frente da política treinada — mas **não** é diferença
pareada e não tem IC da diferença. Fica para a SPEC-06, quando as duas rodarem no
mesmo harness.

**A caminhada aleatória degenera o subconjunto `sucesso_comum`.** Com taxa de
sucesso 0,000 não há episódio em que ela e outra política tenham concluído, e a
T-02.5 manda não reportar. O harness omite a linha em vez de produzir `nan`.

## 6. Decisões e suposições registradas

| id | decisão |
|---|---|
| **D-02.1** | `kmeans_tsp` é política **executável**, não cálculo analítico. Um tempo analítico teria de modelar a discretização de `dt`, o corte por `communication_range` e o `max_episode_length` — cada uma uma aproximação *diferente* das que o simulador aplica às outras políticas |
| **D-02.2** | métrica primária é `t_conc_pen`, com imputação em `max_episode_length` (PAR-1). Monótona, não manipulável por travar cedo, escalar único |
| **A-02.1** | o guloso voa a `v_max = 15 m/s`. Lamenza não especifica a velocidade da heurística; usamos o máximo, que é o que a torna mais forte |

**Desvio declarado:** a T-02.8 pede `episodios.parquet`; usamos **CSV**, para não
acrescentar `pyarrow` por causa de um arquivo. Grão, colunas e exclusão do git são
os mesmos.

## 7. Validação do harness — T-02.9

| item | estado |
|---|---|
| §1 ordenação esperada, adjacentes conclusivas | ✅ nas 3 células |
| **§2 teste A/A** | ✅ **calibrado** — `reports/spec02-aa.md` |
| §3 teste de sinal (guloso aponta *para* o sensor) | ✅ 4 direções + versão dinâmica |
| §4 privilégio: sem flag, sem verdade | ✅ com política-espião no laço real |
| §5 disjunção treino/avaliação | ✅ asserção antes de gastar simulação |
| §6 degenerescências (`Ns < 12`, `Na > Ns`, `Na = 1`) | ✅ |
| §7 sensibilidade a `communication_range` | ⬜ **não feito** — ver abaixo |

**O que ficou de fora:** a sensibilidade a `communication_range` (T-02.9 §7). A
P1 foi fechada na SPEC-00 com o valor lido do código (20 m), então o motivo
original — decidir entre candidatos — desapareceu. Continua valendo como análise
de robustez, e fica registrado como não feito em vez de silenciosamente omitido.

## 8. O que isto destrava

**A interface `Policy` está congelada** (`src/uavwnn/agents/base.py`), com os
cinco contratos da T-02.1 sob teste. A SPEC-05 implementa a mesma interface e
entra no mesmo harness — mesma censura, mesmo bootstrap, mesma tabela.

**Todo IC do projeto agora tem procedimento verificado.** Era o último bloqueio
estatístico.
