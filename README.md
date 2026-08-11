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

## Estado

| Spec | Estado |
|---|---|
| SPEC-00 — Ambiente fiel a Lamenza | especificada; implementação em PR separado |
| SPEC-01 — Estudo de binarização | especificada |
| SPEC-02 a SPEC-10 | especificadas, em revisão |

## Referências

- LAMENZA, T. S. *A Deep Reinforcement Learning Approach for Sensor Data Collection by a Swarm of UAVs*. Dissertação de mestrado, PUC-Rio, 2025.
- KATOPODIS, R. F. *Reinforcement Learning with Weightless Neural Networks*. Dissertação de mestrado, COPPE/UFRJ, 2022.
- LOWE, R. et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), 2017.
- NG, A. Y.; HARADA, D.; RUSSELL, S. *Policy invariance under reward transformations*, 1999.
- JULIER, S.; UHLMANN, J. *A non-divergent estimation algorithm in the presence of unknown correlations*, 1997.

As dissertações de Lamenza e Katopodis **não** estão versionadas: são obras de terceiros. As citações no PRD e nas specs referenciam seção, equação e página.
