# SPEC-00 T-00.1 — Proveniência dos parâmetros

**Fonte primária de mecanismo:** repositório público `Project-GrADyS/gradys-data-collection`, branch `dissertation`.
Arquivos consultados: `scenario/003-collaboration/environment.py` (680 linhas), `.../main.py`, `.../heuristics.py`, `.../benchmark.py`.
**Fonte primária de valores publicados:** Lamenza (2025), tab. 3.1, 5.1, 7.1, 7.2.

O cenário `003-collaboration` é o que corresponde ao §7.2 da dissertação ("Explored the ability of MADDPG to learn a collaborative strategy for data collection").

---

## 1. Parâmetros resolvidos

| Parâmetro | Valor | Proveniência | Pendência fechada |
|---|---|---|---|
| `communication_range` | **20 m** | `environment.py:157` — `communication_range: float = 20` | **P1** |
| Período do heartbeat | **`0,99 × interval`** | `environment.py:102` — `schedule_timer("", now + algorithm_interval * 0.99)`, e `handle_timer` chama `_collect_packets()` | **P5** |
| Raio da área de largada | **2 m** (quadrado, não disco) | `environment.py:533` — `random.uniform(-2, 2)` para x e y | **P6** |
| Tratamento de borda | **destino limitado**; o UAV não sai de `[-100, 100]²` | `environment.py:71-95` — bloco "Maintain direction but bound destination within scenario" | **P8** |
| Terminação ao completar | **episódio NÃO termina**; a poda por parada continua valendo | `benchmark.py` passa `--no-end-when-all-collected`; `environment.py:654` | **RF-02** |
| Agregação de episódios sem sucesso | **censura em `max_episode_length`** | `environment.py:674` — `"completion_time": max_episode_length if not all_collected else ...`; e `collection_times` inicializado em `max_episode_length` (`:551`) | **L1** |
| Recompensa de equipe | **mesmo escalar para todos** | `environment.py:646` — `rewards = {agent: reward for agent in self.agents}` | SPEC-00 Q9 |
| Velocidade do guloso | **máxima** (`1` normalizado = 15 m/s) | `heuristics.py` — `np.array([(angle + pi)/(2pi), 1])` | SPEC-02 Q3 |
| Guloso reativo | **recomputa o alvo a cada passo** | `heuristics.py` — sem estado entre chamadas | SPEC-02 Q4 |
| `N*_s`, `N*_a` | **12** e **`Na − 1`** | `benchmark.py` — `--state-num-closest-sensors=12`, `--state-num-closest-drones={agents-1}` | confirma tab. 7.1 |
| Modo de recompensa | **`punish`** = eq. 5-6 | `benchmark.py` — `--reward=punish` | confirma eq. 5-6 |
| `total_timesteps` | **50 000 000** | `benchmark.py` | confirma §7.2 |
| Arquitetura do ator do MADDPG | *pendente* — `main.py` define `actor_model_size` por CLI | — | **P4 segue aberta** |

---

## 2. Divergências entre o texto da dissertação e o código

Estas são as descobertas que mudam decisões do projeto. Em cada caso registramos qual fonte adotamos e por quê.

### 2.1 `algorithm_iteration_interval`: 2 s (texto) contra 0,5 s (código)

A tab. 7.1 diz **2 s**. O `benchmark.py` que gerou os resultados passa `--algorithm-iteration-interval=0.5`.

**Adotamos 0,5 s.** Não é só uma questão de qual fonte vale: **2 s é fisicamente inconsistente com `communication_range = 20 m`.** A 15 m/s o UAV percorre 30 m entre heartbeats, isto é, mais que o diâmetro útil da cobertura — ele atravessaria sensores sem coletá-los. Com 0,5 s percorre 7,4 m, e a asserção `v·hb ≤ 0,5·r` da SPEC-00 T-00.3 é satisfeita com folga.

Esta é a razão de `ScenarioConfig` **recusar** `algorithm_iteration_interval=2.0` com `communication_range=20` (teste 24): o valor da tabela publicada não é construível sem quebrar o mecanismo de coleta.

### 2.2 A observação é NORMALIZADA e o padding é `-1`

A eq. 5-1 do texto descreve coordenadas relativas cruas. O código (`environment.py:331-375`) faz:

```python
closest_unvisited_sensors.fill(-1)                       # padding = -1
...
(agent_position - closest[:n] + scenario_size) / (scenario_size * 2)   # normalização
```

**Três consequências para o projeto**, todas em favor:

1. A entrada da rede vive em **[0, 1]** (valores válidos), não em `[-200, 200]` metros. A SPEC-01 estava dimensionando o termômetro para a faixa errada.
2. **O padding `-1` está fora da faixa dos valores válidos.** Logo "slot vazio" e "sensor exatamente sobre o UAV" (que normaliza para 0,5) são distinguíveis **por construção**. O **bit de validade** que a SPEC-00 §5.4 e a SPEC-01 propunham como desvio obrigatório é **desnecessário** — e com ele cai a única linha da tabela de desvios que não era a binarização.
3. O índice do agente entra como `k / Na`, não como `k` cru — já é uma quantidade em [0, 1).

**Ressalva medida:** `p_k − p_i` vive em `[-2S, 2S]`, então a normalização produz `[-0,5, 1,5]` e não `[0, 1]`. O código de referência declara `Box(-1, 1)` para este modo, o que também não bate. A SPEC-01 precisa binarizar a faixa **real**, não a declarada.

### 2.3 A regra de não poda é a leitura (B), não a (A)

A SPEC-00 §9.2 propunha como default a leitura (A): o episódio corre até 500 s após a coleta completa. O `benchmark.py` passa `--no-end-when-all-collected`, e o `step()` mantém a poda por parada ativa — logo o episódio termina **`max_seconds_stalled` (30 s) após a última visita**. É a leitura **(B)**.

Isso importa para o orçamento: sob (A) todo episódio bem-sucedido rodaria os 500 s completos. Sob (B), os episódios da grade terminam entre 58 s e 82 s.

---

## 3. Quirks do original reproduzidas de propósito

Registradas porque afetam o comportamento e porque um leitor do código pode achar que são bugs nossos.

| # | Quirk | Efeito |
|---|---|---|
| Q1 | O waypoint mede a distância até a borda **mais próxima** em cada eixo (`limit − |x|`), independentemente do **sentido** do movimento | Perto de uma borda o UAV quase para, mesmo apontando para dentro. Reproduzida; `edge_behavior="corrected"` existe só para quantificar o efeito |
| Q2 | A primeira decisão ocorre em `t = update_rate`, não em `t = 0`, e a primeira época é mais curta | `reset()` consome um evento antes da primeira observação. Efeito de um único epoch por episódio |
| Q5 | `DroneProtocol` inicializa `current_position = (0,0,0)` em vez da posição de spawn | Ver §4.2: "corrigir" isto quebra a equivalência com a referência por efeito caótico |
| Q3 | A posição lida numa decisão está **um tique de mobilidade atrasada** | Barreira e tique compartilham timestamp; a barreira é processada primeiro. **Importa para a SPEC-07**: o KF vai estimar uma posição que a política observa com atraso |
| Q4 | O guloso de referência filtra slots por `== 0` (posição relativa nula), mas o padding é `-1` | Não filtra o padding; funciona só porque `(-3, -3)` tem norma grande. Nossa `GreedyPolicy` filtra pelo sentinela correto |

---

## 4. Correções aplicadas em relação à SPEC-00

| # | O que a spec dizia | O que o código mostrou |
|---|---|---|
| C1 | Observação sem normalização, padding `(0,0)`, bit de validade obrigatório | Normalizada, padding `-1`, bit de validade desnecessário (§2.2) |
| C2 | Ação aplicada com waypoint a `speed·dt` de distância | Destino colocado na **borda** ao longo da direção (`environment.py:80-99`) |
| C3 | Leitura (A) da regra de não poda | Leitura (B) (§2.3) |
| C4 | Um timer `"decide"` por agente | **Barreira global de época** (`GrADySHandler`, `environment.py:485-508`): todos os agentes decidem no mesmo instante, sobre o mesmo estado. Isto fecha a pergunta nº 10 da SPEC-05 |
| C5 | Nós inseridos com UAVs primeiro | Sensores primeiro, depois UAVs |
| C6 | `MasslessMobilityHandler` / `MasslessMobilityConfiguration` | O código usa os aliases `MobilityHandler` / `MobilityConfiguration` — verificado: são o mesmo objeto |

### 4.1 Equivalência provada, não argumentada

`tests/test_referencia.py` roda o `GrADySEnvironment` de referência e o nosso `EpisodeRunner` sobre o **mesmo cenário** e a **mesma sequência de ações**, e compara passo a passo:

| Grandeza | Pior diferença em 60 passos × 5 sementes |
|---|---|
| Observação | **7,9 × 10⁻⁸** (precisão de float32) |
| Recompensa | **0** |

**O MDP é o mesmo.** As diferenças de código são de andaime — API, semeadura, validações, ponto de inserção de ruído —, não de formulação.

### 4.2 Uma "correção" nossa que quebrava a equivalência, e foi revertida

O `DroneProtocol` de referência inicializa `current_position = (0,0,0)`, o que é tecnicamente errado enquanto a primeira telemetria não chega — o spawn é sorteado em `[-2, 2]²`. Na primeira versão desta spec nós "corrigimos" isso para a posição real de spawn, e registramos como micro-correção benigna.

**Não era benigna.** Medido: com a correção, `max|Δobs|` contra a referência salta de **7,9 × 10⁻⁸ para 8,5 × 10⁻¹** em 60 passos. A causa não é magnitude, é **caos**: a posição errada entra no cálculo da escala do waypoint, altera levemente o rumo da primeira ação, e as trajetórias divergem completamente.

Revertido. O PRD §0 manda reproduzir, não melhorar — e este é o exemplo concreto de por que a regra existe. O teste diferencial agora trava isso.

**Confirmado por execução, não por leitura** (os três fatos mais caros da §4.1 da spec):

- `SetVelocityMobilityCommand(vx, vy, vz)` tem `command_type == SET_SPEED` e `param_1 == vx`. Sob o handler massless viraria `speed = vx` **silenciosamente**. A tabela do `tmp/PLANO.md` está errada neste ponto.
- `Simulator.__init__` adiciona um `StreamHandler` ao logger raiz e nunca o remove: 3 `Simulator` → 3 handlers. Poda implementada em `bridge.py`, coberta pelo teste 21.
- Identificadores de nó são o índice de inserção.

---

## 5. Pendências que permanecem

| # | Pendência | Situação |
|---|---|---|
| **P4** | Arquitetura do ator do MADDPG | `main.py` recebe `actor_model_size` por CLI; o `benchmark.py` do §7.2 não o passa, então vale o default do `main.py` — falta extrair. Bloqueia metade da razão dupla de tamanho da §5.5 do PRD |
| **P7** | `max_episode_length` "permissivo" do §7.1 | O cenário exploratório é o `001-prototype`/`002-scaling`, não consultado ainda |
| **P9** | `σ_sensor` | É parâmetro **nosso** (Etapa 2), não do original. Segue em aberto |
