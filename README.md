# uav-wnn — Coleta de dados por enxame de UAVs com redes sem peso e fusão de dados

Trabalho da disciplina **Fusão de Dados** — PESC/COPPE-UFRJ.
Raphael Nagao Ferreira · `rnagao@cos.ufrj.br`

---

## O que é

Reprodução do problema de coordenação de enxame de UAVs formulado por **Lamenza (2025, PUC-Rio)**, trocando **apenas o aproximador de função**: rede n-tuple sem peso (Katopodis, 2022, COPPE/UFRJ) no lugar das redes profundas do MADDPG. Depois da reprodução, o trabalho estende o problema para o domínio da disciplina: **medição ruidosa, filtro de informação, filtro de Kalman com entrada de controle e fusão distribuída por interseção de covariância**.

O princípio que rege o repositório está no [PRD §0](PRD.md):

> **O problema de Lamenza não é modificado.** Toda alteração exige uma linha na tabela de desvios (§5.4) e uma decisão registrada (§10).

E isso deixou de ser promessa: **o ambiente é o `GrADySEnvironment` de Lamenza, verbatim**, carregado de um submodule fixado na tag `dissertation`. Não há reimplementação do MDP. O `src/uavwnn/refenv/` apenas carrega, configura e semeia aquele código — nada altera observação, ação, recompensa, coleta ou terminação.

O simulador está fixado em **`gradysim==0.6.11`**: o `environment.py` de referência importa `VisualizationController`, que existe até essa versão e sumiu na série 0.7. Rodar código intacto contra outro simulador seria, por si só, uma divergência do baseline.

## Estado

| Spec | Estado |
|---|---|
| SPEC-00 — Ambiente de referência | **concluída**: 18 testes, P1 e P2 fechadas, validação contra a fig. 7.3 |
| SPEC-01 — Estudo de binarização | **concluída**: portão G0 fechado, codificador escolhido com evidência |
| SPEC-03 — Harness de custo | **T1 e T3 medidos**; T2 depende da SPEC-04 |
| SPEC-02, 07 | revisadas; **próximas**, rodam em paralelo |
| SPEC-04 a 06, 08 a 10 | revisadas |

**Bloqueio único:** a pendência **T3** — a física do ruído no sensor — decide se a H4 tem mecanismo. Ela trava o bloco 2b e nada mais.

## Resultados medidos

### A montagem está correta

A heurística gulosa **do próprio trabalho de referência**, no ambiente de referência, reproduz os tempos de conclusão da fig. 7.3 em **9 de 9 células** da grade: erro relativo mediano **1,7 %**, máximo 3,3 %, viés médio **+0,5 %** — sem tendência sistemática. Ver [`reports/spec00-parametros.md`](reports/spec00-parametros.md) §8.

Isso vale mais que qualquer teste unitário: se submodule, versão do simulador, configuração ou laço de avaliação estivessem errados, essa curva não fecharia.

### Custo: 1,8 a 5,2 h por núcleo

Para o orçamento de 50 M passos de Lamenza, contra 7 dias de GPU por célula no original. O PRD estimava 8–30 h; o real é menor, com folga. E a célula cara em horas de treino é **`Na = 1`**, não `Na = 8` — o contador de passos é `passos × Na`. Ou seja, a Etapa 1 é a mais cara do projeto. Ver [`reports/spec00-orcamento.md`](reports/spec00-orcamento.md).

### O custo: a alegação de "modelo menor" não se sustenta

Medido na configuração escolhida, na tarefa de clonagem, **com os dois modelos convergidos**:

| | erro | persistente |
|---|---|---|
| **MLP pareada 25-32-32-2** | **1,1°** | **7,6 KB** |
| WNN E2/T=8/N=48 | 5,6° | 133,9 KB |

A MLP é 5× mais precisa e **17,5× menor**. A estimativa anterior (~12–15 KB) errou uma ordem de grandeza porque partia de um protótipo com **2 variáveis de entrada**; a observação real tem 25.

O que sobrevive, e está medido: **zero multiplicações em ponto flutuante no aproximador**, com vantagem de ciclos **condicional à classe de alvo** — 1,6× em x86, 4,1× em Cortex-M4F, **19,9× em Cortex-M0 sem FPU**. Ver [`reports/spec03-custo.md`](reports/spec03-custo.md).

*(Ressalva: a comparação é sobre clonagem supervisionada. As vantagens que a literatura atribui às redes sem peso — aprendizado online, sem replay, esquecimento — não são exercitadas por regressão estacionária.)*

### O codificador binário: quantis por slot

A distribuição da observação depende fortemente do **índice do slot** — o sensor mais próximo concentra, o 12º espalha, com razão de amplitude **3,1×**. Binning global desperdiça bits justamente no slot 1, que é o que dirige a ação: usa 2,40 dos 4,00 bits disponíveis.

Medido pela clonagem do guloso (erro angular mediano): **quantil por slot 4,3°** contra 6,7° do linear por slot e 7,6° do global. Configuração escolhida: **E2, T=8, N=48** — 5,8° com 200 bits, melhor que o *melhor* resultado do global, que precisa de 400 bits e 128 RAMs. Ver [`reports/spec01-binarizacao.md`](reports/spec01-binarizacao.md).

### O baseline da figura publicada não coordena

![Painel comparando greedy e smart com 8 UAVs e 36 sensores: na greedy os oito drones voam empilhados na mesma trilha e o episódio leva 55 s; na smart cada UAV cobre um território e leva 11,5 s](reports/img/greedy-vs-smart-8a36s.png)

A `greedy` — o baseline da fig. 7.3 — tem **separação média entre UAVs de 0,1 m**, contra alcance de comunicação de 20 m. Os drones voam colados, cobrindo o mesmo disco. Não é coordenação fraca; é ausência de coordenação.

A `smart`, do mesmo `heuristics.py`, coordena lendo **apenas o vetor de observação** — nenhuma informação privilegiada. No mesmo intervalo de decisão, `Ns = 24`, 60 sementes:

| Na | MADDPG publicado | `smart` medido | `greedy` medido |
|---|---|---|---|
| 2 | 29,6 | **25,18 ± 0,91** | 40,79 ± 2,07 |
| 4 | 13,8 | 16,12 ± 0,73 | 43,04 ± 2,21 |
| 8 | 9,7 | 10,21 ± 0,40 | 39,33 ± 2,51 |

**Uma heurística sem treino algum fica a 5–17 % do MADDPG e o supera com 2 agentes** — contra 32 dias de GPU. Isso muda a barra da H6 e está registrado como pendência **T4**.

## Etapas

Cada etapa acrescenta **uma única** fonte de dificuldade, para que qualquer degradação seja atribuível a uma causa identificada.

| Etapa | Configuração | Dificuldade nova | Hipóteses | Portão |
|---|---|---|---|---|
| 1 | 1 UAV, sem ruído | aproximação de função sobre entrada binarizada | H1 | G1 |
| **2a** | 1 UAV, ruído no **GPS do drone** | KF com entrada de controle | H2, H3 | G2 |
| **2b** | 1 UAV, + ruído no **canal do sensor** | filtro de informação; ranking sobre estimativas | H4 | G2 |
| 3 | `Na` UAVs, com ruído | coordenação e erros correlacionados | H5, H6 | G3 |

A Etapa 2 roda em dois blocos porque os canais têm estruturas opostas — o do drone é deslocamento **comum** a todas as entradas da observação, o do sensor é **por slot** — e ligá-los juntos tornaria qualquer degradação inatribuível.

Como as features entram **sem tocar no cenário**, com os diagramas do fluxo: [`specs/ARQUITETURA.md`](specs/ARQUITETURA.md).

## Instalação

Requer [uv](https://docs.astral.sh/uv/). O ambiente vem de um **submodule**:

```bash
git submodule update --init --recursive
uv sync
uv run pytest -q                    # 18 testes, ~1 min
```

## Scripts

| Script | O que faz |
|---|---|
| `spec00_demo.py` | um episódio com traço passo a passo; `--plot` salva trajetórias, `--compare` gera o painel greedy vs smart, `--estilo lamenza` reproduz o visual do `plot_paths.py` de referência |
| `spec00_visualizar.py` | **visualização 3D ao vivo** no navegador; `--diagnostico` testa o servidor sem navegador |
| `spec00_validar_guloso.py` | reproduz a fig. 7.3 nas 9 células |
| `spec00_budget.py` | custo de simulação por célula |

```bash
uv run python scripts/spec00_demo.py --policy smart --plot traj.png
uv run python scripts/spec00_demo.py --compare greedy-vs-smart.png
uv run python scripts/spec00_demo.py --agents 2 --sensors 12 --estilo lamenza --plot path-a2s12.png
uv run python scripts/spec00_visualizar.py --policy greedy
```

Um episódio em três linhas:

```python
from uavwnn.refenv import LAMENZA_003, load_heuristics, run_episode

cfg = LAMENZA_003(num_drones=4, num_sensors=24)
greedy = load_heuristics().create_greedy_heuristics(3, 12)
print(run_episode(cfg, greedy, seed=1).as_row())
```

### Visualização 3D ao vivo

O simulador sobe um servidor **WebSocket** em `localhost:5678`. Ele fala apenas WebSocket e **não serve página** — abrir `localhost:5678` no navegador dá *"You cannot access a WebSocket server directly with a browser"*.

Quem serve a página é o visualizador hospedado, e é ele que conecta no seu servidor:

```
https://project-gradys.github.io/gradys-sim-nextgen-visualization/
```

Use **Chrome ou Edge**: a página é HTTPS e precisa abrir um WebSocket inseguro em `localhost`. O Chrome permite por tratar `localhost` como origem confiável; o Firefox costuma bloquear.

Se não conectar, isole o problema sem depender do navegador com `--diagnostico`: ele conecta como cliente WebSocket em Python e imprime os quadros recebidos. Se passar, o servidor está bom e o problema é do navegador.

## Documentos

| Arquivo | O que é |
|---|---|
| [`PRD.md`](PRD.md) | Requisitos, hipóteses, MDP, figuras de mérito, cronograma, decisões |
| [`PENDENCIAS.md`](PENDENCIAS.md) | Registro único das pendências, ordenadas por impacto |
| [`specs/ARQUITETURA.md`](specs/ARQUITETURA.md) | Como as features entram sem tocar no cenário, com os diagramas |
| [`specs/`](specs/) | 11 specs de execução, cada uma com critério de saída **medido** |
| [`specs/REVISAO.md`](specs/REVISAO.md) | Revisão cruzada das specs (registro de um momento) |
| [`reports/`](reports/) | Relatórios com números medidos e as figuras |

## Proveniência

Os valores do cenário vieram do **código de referência** (`Project-GrADyS/gradys-data-collection@dissertation`, cenário `003-collaboration`), não do texto da dissertação, sempre que os dois divergem. Proveniência linha a linha em [`reports/spec00-parametros.md`](reports/spec00-parametros.md), incluindo três ressalvas que precisam acompanhar qualquer citação dos números publicados:

- `algorithm_iteration_interval` é **0,5 s** no código, contra 2 s na tab. 7.1 — e 2 s é fisicamente inconsistente com `communication_range = 20 m`, porque a 15 m/s o UAV atravessaria a cobertura de um sensor entre dois heartbeats;
- a observação é **normalizada** e o padding é **`-1`**, não `(0,0)` — o que torna desnecessário o bit de validade que o projeto havia previsto;
- na fig. 7.3 o MADDPG é avaliado a 0,5 s por decisão e o guloso a **0,1 s**, uma variável não controlada que vale ~10 % no tempo de conclusão.

## Referências

- LAMENZA, T. S. *A Deep Reinforcement Learning Approach for Sensor Data Collection by a Swarm of UAVs*. Dissertação de mestrado, PUC-Rio, 2025.
- KATOPODIS, R. F. *Reinforcement Learning with Weightless Neural Networks*. Dissertação de mestrado, COPPE/UFRJ, 2022.
- LOWE, R. et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), 2017.
- NG, A. Y.; HARADA, D.; RUSSELL, S. *Policy invariance under reward transformations*, 1999.
- JULIER, S.; UHLMANN, J. *A non-divergent estimation algorithm in the presence of unknown correlations*, 1997.

As dissertações de Lamenza e Katopodis **não** estão versionadas: são obras de terceiros. As citações no PRD e nas specs referenciam seção, equação e página.
