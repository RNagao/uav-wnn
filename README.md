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
| 2 | 1 UAV, com ruído | observabilidade parcial e filtragem | H2, H3, H4 | G2 |
| 3 | `Na` UAVs, com ruído | coordenação e erros correlacionados | H5, H6 | G3 |

## Documentos

| Arquivo | O que é |
|---|---|
| [`PRD.md`](PRD.md) | Requisitos, hipóteses, MDP, figuras de mérito, cronograma, decisões |
| [`PENDENCIAS.md`](PENDENCIAS.md) | Registro único das pendências, ordenadas por impacto |
| [`specs/`](specs/) | 11 specs de execução, cada uma com critério de saída **medido** |
| [`specs/REVISAO.md`](specs/REVISAO.md) | Revisão cruzada das specs: lacunas e pontos a decidir |
| [`reports/`](reports/) | Relatórios com números medidos |

## Estado

| Spec | Estado |
|---|---|
| SPEC-00 — Ambiente fiel a Lamenza | **implementada**; 40 testes; P1 e P2 fechadas. Falta o invariante V-1, que depende da SPEC-01 |
| SPEC-01 — Estudo de binarização | especificada; próxima |
| SPEC-02 a SPEC-10 | especificadas, em revisão |

## Instalação e uso

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                          # 40 testes
uv run python scripts/spec00_budget.py    # custo e fidelidade do integrador
```

Um episódio com a heurística gulosa:

```python
from uavwnn.env import EpisodeRunner, GreedyPolicy, ScenarioConfig

cfg = ScenarioConfig(n_agents=4, n_sensors=24)
runner = EpisodeRunner(cfg, root_seed=0)
result = runner.run(GreedyPolicy(cfg.effective_n_star_a), episode_index=0)
print(result.as_row())
```

## Parâmetros e proveniência

Os valores do cenário vieram do **código de referência** (`Project-GrADyS/gradys-data-collection@dissertation`, cenário `003-collaboration`), não do texto da dissertação, sempre que os dois divergem. Proveniência linha a linha em [`reports/spec00-parametros.md`](reports/spec00-parametros.md). Três divergências relevantes:

- a observação é **normalizada** e o padding é **`-1`**, não `(0,0)` — o que torna desnecessário o bit de validade que havíamos projetado;
- `algorithm_iteration_interval` é **0,5 s** no código contra 2 s na tab. 7.1 — e 2 s é fisicamente inconsistente com `communication_range = 20 m`;
- o `update_rate` do código de referência (`interval/2`) introduz **viés sistemático de +7,9 %** no tempo de conclusão; ver [`reports/spec00-orcamento.md`](reports/spec00-orcamento.md).

## Referências

- LAMENZA, T. S. *A Deep Reinforcement Learning Approach for Sensor Data Collection by a Swarm of UAVs*. Dissertação de mestrado, PUC-Rio, 2025.
- KATOPODIS, R. F. *Reinforcement Learning with Weightless Neural Networks*. Dissertação de mestrado, COPPE/UFRJ, 2022.
- LOWE, R. et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), 2017.
- NG, A. Y.; HARADA, D.; RUSSELL, S. *Policy invariance under reward transformations*, 1999.
- JULIER, S.; UHLMANN, J. *A non-divergent estimation algorithm in the presence of unknown correlations*, 1997.

As dissertações de Lamenza e Katopodis **não** estão versionadas: são obras de terceiros. As citações no PRD e nas specs referenciam seção, equação e página.
