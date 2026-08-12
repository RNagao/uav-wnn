# uav-wnn — Coleta de dados por enxame de UAVs com redes sem peso e fusão de dados

Trabalho da disciplina **Fusão de Dados** — PESC/COPPE-UFRJ.
Raphael Nagao Ferreira · `rnagao@cos.ufrj.br`

---

## O que é

Reprodução do problema de coordenação de enxame de UAVs formulado por **Lamenza (2025, PUC-Rio)**, trocando **apenas o aproximador de função**: rede n-tuple sem peso (Katopodis, 2022, COPPE/UFRJ) no lugar das redes profundas do MADDPG. Depois da reprodução, o trabalho estende o problema para o domínio da disciplina: **medição ruidosa, filtro de Kalman com entrada de controle e fusão distribuída por interseção de covariância**.

O princípio que rege o repositório está no [PRD §0](PRD.md): **o problema de Lamenza não é modificado.** Toda alteração exige uma linha na tabela de desvios (§5.4) e uma decisão registrada (§10). Hoje a tabela tem uma linha: a binarização da entrada, que é inevitável porque redes n-tuple endereçam RAMs com bits.

## Etapas

Cada etapa acrescenta **uma única** fonte de dificuldade, para que qualquer degradação seja atribuível a uma causa identificada.

| Etapa | Configuração | Dificuldade nova | Hipóteses | Portão |
|---|---|---|---|---|
| 1 | 1 UAV, sem ruído | aproximação de função sobre entrada binarizada | H1 | G1 |
| **2a** | 1 UAV, ruído no **GPS do drone** | KF com entrada de controle | H2, H3 | G2 |
| **2b** | 1 UAV, + ruído no **canal do sensor** | filtro de informação; ranking sobre estimativas | H4 | G2 |
| 3 | `Na` UAVs, com ruído | coordenação e erros correlacionados | H5, H6 | G3 |

A Etapa 2 roda em dois blocos porque os canais têm estruturas opostas — o do drone é deslocamento comum a todas as entradas da observação, o do sensor é por slot — e ligá-los juntos tornaria qualquer degradação inatribuível. Como as features se encaixam sem tocar no cenário: [`specs/ARQUITETURA.md`](specs/ARQUITETURA.md).

## Documentos

| Arquivo | O que é |
|---|---|
| [`PRD.md`](PRD.md) | Requisitos, hipóteses, MDP, figuras de mérito, cronograma, decisões |
| [`PENDENCIAS.md`](PENDENCIAS.md) | Registro único das pendências, ordenadas por impacto |
| [`specs/`](specs/) | 11 specs de execução, cada uma com critério de saída **medido** |
| [`specs/REVISAO.md`](specs/REVISAO.md) | Revisão cruzada das specs: lacunas e pontos a decidir |
| [`specs/ARQUITETURA.md`](specs/ARQUITETURA.md) | Como as features entram sem tocar no cenário, com os diagramas |
| [`reports/`](reports/) | Relatórios com números medidos |

## Estado

| Spec | Estado |
|---|---|
| SPEC-00 — Ambiente de referência | **pronta**; o ambiente é o de Lamenza verbatim (submodule), 15 testes, P1 e P2 fechadas |
| SPEC-01 — Estudo de binarização | revisada; **próxima** |
| SPEC-02, 03, 07 | revisadas; rodam em paralelo com a 01 |
| SPEC-04 a SPEC-06, 08 a SPEC-10 | revisadas |

## Instalação e uso

Requer [uv](https://docs.astral.sh/uv/). O ambiente vem de um **submodule**:

```bash
git submodule update --init --recursive
uv sync
uv run pytest -q                          # 15 testes
uv run python scripts/spec00_budget.py    # custo por célula da grade
```

Um episódio com a heurística gulosa **do próprio trabalho de referência**:

```python
from uavwnn.refenv import LAMENZA_003, load_heuristics, run_episode

cfg = LAMENZA_003(num_drones=4, num_sensors=24)
greedy = load_heuristics().create_greedy_heuristics(3, 12)
print(run_episode(cfg, greedy, seed=1).as_row())
```

## O ambiente é o de Lamenza, verbatim

Não há reimplementação do MDP. O ambiente vem do submodule `vendor/gradys-data-collection`, fixado na tag `dissertation`, cenário `003-collaboration`. Nosso `src/uavwnn/refenv/` apenas **carrega, configura e semeia** aquele código — nada altera observação, ação, recompensa, coleta ou terminação.

O simulador está fixado em **`gradysim==0.6.11`**: o `environment.py` de referência importa `VisualizationController`, que existe até essa versão e sumiu na série 0.7. Rodar código intacto contra outro simulador seria, por si só, uma divergência do baseline.

Proveniência de cada parâmetro e as ressalvas sobre os números publicados estão em [`reports/spec00-parametros.md`](reports/spec00-parametros.md) — incluindo que a fig. 7.3 compara o MADDPG a 0,5 s por decisão com o guloso a 0,1 s, uma variável não controlada que vale ~10 % no tempo de conclusão.

## Referências

- LAMENZA, T. S. *A Deep Reinforcement Learning Approach for Sensor Data Collection by a Swarm of UAVs*. Dissertação de mestrado, PUC-Rio, 2025.
- KATOPODIS, R. F. *Reinforcement Learning with Weightless Neural Networks*. Dissertação de mestrado, COPPE/UFRJ, 2022.
- LOWE, R. et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), 2017.
- NG, A. Y.; HARADA, D.; RUSSELL, S. *Policy invariance under reward transformations*, 1999.
- JULIER, S.; UHLMANN, J. *A non-divergent estimation algorithm in the presence of unknown correlations*, 1997.

As dissertações de Lamenza e Katopodis **não** estão versionadas: são obras de terceiros. As citações no PRD e nas specs referenciam seção, equação e página.
