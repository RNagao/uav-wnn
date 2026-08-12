# SPEC-00 — Ambiente fiel a Lamenza

**Etapa:** 1 (infra) · **Depende de:** nada · **Estimativa:** 3 dias
**Portão de saída:** contribui para `G0` (P1 e P2 resolvidas; orçamento de simulação medido) · **Referência:** PRD §5.1, §5.2, §7; Lamenza §3 (tab. 3.1), §5.1–5.3 (eq. 5-1, 5-6), §6, §7 (tab. 7.1)

---

> ## ⚠ Documento histórico — executado, e a execução mudou a abordagem
>
> Esta spec foi escrita **antes** de o código de referência ser lido, e projetava uma **reimplementação** do MDP: `build_observation`, `EpisodeRunner`, `StateView`, protocolos próprios, gerador analítico e o invariante V-1.
>
> **Nada disso está no repositório.** A execução da spec levou à conclusão oposta: o ambiente passou a ser o `GrADySEnvironment` de Lamenza, verbatim, num submodule fixado na tag `dissertation`, com `gradysim==0.6.11`. O que está implementado é `src/uavwnn/refenv/`, que apenas carrega, configura e semeia aquele código.
>
> | O que a spec projetava | O que foi entregue |
> |---|---|
> | `build_observation` como fonte única | o método do ambiente de referência, intocado |
> | `EpisodeRunner` com injeção de política | `run_episode()` sobre a API PettingZoo do próprio ambiente |
> | `StateView` como ponto de inserção de ruído | **subclasse** sobrescrevendo `observe_simulation_relative_positions()` |
> | Gerador analítico + invariante V-1 (teste KS) | **descartados** — o corpus sai do ambiente real, que roda a 0,02–0,15 s por episódio |
> | Padding `(0,0)` + bit de validade | padding **`-1`**, fora da faixa dos válidos; bit de validade **desnecessário** |
> | `update_rate` escolhido por nós (`interval/16`) | fixado pelo ambiente de referência em `interval/2` |
> | Superfície de API verificada no `gradysim 0.8.0` | **0.6.11** — o `environment.py` importa `VisualizationController`, que sumiu na série 0.7 |
> | P1 (`communication_range`) bloqueante | **20 m**, resolvida no código de referência |
>
> **Por que a reversão.** A reimplementação era provadamente equivalente (`max|Δobs| = 7,9×10⁻⁸` contra a referência, em 60 passos × 5 sementes). Mas uma "correção" de **uma linha** — inicializar a posição do drone no spawn em vez de `(0,0,0)`, que é o que o original faz — levava a divergência a `8,5×10⁻¹` por efeito caótico. Manter implementação paralela é superfície de risco permanente contra o único ativo do trabalho: a comparabilidade com o baseline.
>
> **Estado dos critérios de saída:** P1 e P2 fechadas; grade completa rodando; **o guloso de referência reproduz a fig. 7.3 em 9/9 células** (erro mediano 1,7 %). V-1 deixou de existir.
>
> Proveniência e ressalvas: [`../reports/spec00-parametros.md`](../reports/spec00-parametros.md). Arquitetura vigente: [`ARQUITETURA.md`](ARQUITETURA.md).
>
> O restante deste documento fica como **registro do raciocínio** que levou até aqui — em particular a §4, cujo levantamento de API foi o que revelou as armadilhas do simulador. Não o use como especificação do que construir.

---

## 1. Objetivo

Entregar o ambiente de simulação que reproduz o problema de Lamenza **sem alteração**, sobre o `gradysim` (GrADyS-SIM NextGen 0.8.0, Python): geração de cenário semeada, `SensorProtocol` e `UAVProtocol`, coleta automática por proximidade, gestão de episódio, a ponte entre o simulador orientado a eventos e o treinador, e a recompensa verbatim.

O critério não é "roda". É: **o ambiente produz a mesma distribuição de observações que o gerador analítico da SPEC-01** (invariante V-1), **o orçamento de simulação está medido** (pendência P2) e **os parâmetros não publicados estão resolvidos ou declarados como pendência com valor de trabalho registrado** (pendência P1).

## 2. Por que esta spec existe e por que ela é curta

O PRD §0 é um contrato: o problema não é modificado. Esse contrato só é verificável se existir **uma** implementação do MDP — observação, ação, recompensa, terminação — e se ela for fiel linha a linha ao texto original. Toda ambiguidade não resolvida aqui vira uma diferença silenciosa em relação aos números publicados, e uma diferença silenciosa é pior que uma declarada: ela contamina a comparação sem aparecer na tabela de desvios (§5.4).

O segundo motivo é de cronograma. A ponte simulador↔treinador é, pelo próprio PRD §7, "o pedaço de engenharia não trivial do projeto". Ela é chata, é curta, e não produz nenhum número. Se ela vazar para dentro da SPEC-06, a Etapa 1 vira depuração de simulador em vez de experimento.

O terceiro é o mais concreto: **três decisões do ambiente controlam o custo de todo o projeto** — `update_rate` da mobilidade, período de heartbeat e a política de terminação. As três estão medidas ou explicitadas aqui, antes de 10 000 episódios serem gastos com a configuração errada.

## 3. Escopo

**Dentro:** geração de cenário e semeadura; `SensorProtocol` e `UAVProtocol`; coleta automática por proximidade via heartbeat; estado verdadeiro do episódio; construção do vetor de observação (eq. 5-1); aplicação da ação contínua; `rewards.py` (eq. 5-6 e eq. 7-2); gestão de episódio (`max_episode_length`, `max_seconds_stalled`, regra de não poda); ponte por injeção de política; validação de `update_rate`; teste KS do invariante V-1; testes de determinismo.

**Fora:**
- Baselines (aleatório, guloso, k-means+TSP) e o harness de avaliação com IC — **SPEC-02**. Esta spec entrega apenas a política gulosa *mínima* necessária para gerar trajetórias de validação (V-1 e P2), sem instrumentação de métricas.
- Harness de custo T1/T2/T3 — **SPEC-03**.
- Qualquer aproximador de função, codificador binário ou laço de aprendizado — **SPEC-04/05**.
- Ruído, filtro de Kalman, `fusion/` — **SPEC-07**. Esta spec só entrega o **ponto de inserção** (a interface `StateView`, T-00.4), porque retrofitá-la depois é caro e é o que faz o invariante RF-04 ser verificável por construção.
- A ponte alternativa (b) do PRD §7 — estado compartilhado com API PettingZoo de verdade. Fica registrada e só é implementada se o baseline MLP da SPEC-05 exigir.
- Modelo de mobilidade com inércia (`DynamicVelocityMobilityHandler`). Lamenza §3 é explícito: velocidade constante e mudança de direção instantânea. Inércia é assunto da Etapa 2 (PRD §9, linha "`Q` do KF depende de o comando ser cumprido") e entra pela SPEC-07.

---

## 4. Superfície verificada do `gradysim` 0.8.0

Esta seção existe para que a implementação não invente API. O conteúdo abaixo foi **verificado por inspeção do wheel `gradysim-0.8.0-py3-none-any.whl` obtido do PyPI**, e não de memória. Cada linha traz o arquivo de origem dentro do pacote. O que **não** está aqui está marcado como incerto em §4.3.

### 4.1 O que está confirmado

| Elemento | Assinatura / comportamento | Arquivo |
|---|---|---|
| `IProtocol` | métodos abstratos: `initialize()`, `handle_timer(timer: str)`, `handle_packet(message: str)`, `handle_telemetry(telemetry: Telemetry)`, `finish()`; atributo `provider: IProvider` | `protocol/interface.py` |
| `IProtocol.instantiate` | `@classmethod instantiate(cls, provider) -> IProtocol`; faz `protocol = cls()` e depois `protocol.provider = provider`. **O `__init__` não recebe argumentos** | `protocol/interface.py` |
| `IProvider` | `send_communication_command(cmd)`, `send_mobility_command(cmd)`, `schedule_timer(timer: str, timestamp: float)`, `cancel_timer(timer: str)`, `current_time() -> float`, `get_id() -> int`, `tracked_variables: Dict[str, Any]` | `protocol/interface.py` |
| `Telemetry` | dataclass com um único campo: `current_position: Position`, onde `Position = Tuple[float, float, float]` | `protocol/messages/telemetry.py`, `protocol/position.py` |
| Comandos de mobilidade | `GotoCoordsMobilityCommand(x, y, z)`, `SetSpeedMobilityCommand(speed)`, `SetVelocityMobilityCommand(vx, vy, vz)` | `protocol/messages/mobility.py` |
| Comandos de comunicação | `BroadcastMessageCommand(message: str)`, `SendMessageCommand(message: str, destination: int)` | `protocol/messages/communication.py` |
| `MasslessMobilityConfiguration` | `update_rate: float = 0.01`, `default_speed: float = 10`, `reference_coordinates` | `simulator/handler/mobility/massless.py` |
| `CommunicationMedium` | `transmission_range: float = 60`, `delay: float = 0`, `failure_rate: float = 0` | `simulator/handler/communication.py` |
| `SimulationConfiguration` | `duration`, `max_iterations`, `real_time`, `debug`, `log_file`, `execution_logging: bool = True`, `profile` | `simulator/simulation.py` |
| `SimulationBuilder` | `add_handler(h) -> self`, `add_node(protocol_type, position) -> int`, `build() -> Simulator` | `simulator/simulation.py` |
| `Simulator` | `step_simulation() -> bool` (processa **um** evento; `False` quando terminou), `start_simulation() -> None` (bloqueante), `is_simulation_done() -> bool`, `get_node(id) -> Node` | `simulator/simulation.py` |
| `Node` | `id: int`, `position: Position`, `protocol_encapsulator` | `simulator/node.py` |
| `TimerHandler.get_current_time()` | devolve o tempo corrente do event loop; é o **único acessor público** de tempo fora do protocolo | `simulator/handler/timer.py` |

**Fatos de comportamento que restringem o desenho** (todos verificados no código, não inferidos):

1. **Identificadores de nó são o índice de inserção.** `SimulationBuilder.build()` chama `create_node(pos, proto, index)` com `enumerate` sobre a lista de nós adicionados. Logo, adicionando os `Na` UAVs primeiro, o **índice do agente `k` da eq. 5-1 é exatamente `provider.get_id()`**, e os sensores ocupam `Na … Na+Ns−1`.

2. **O alcance de comunicação é avaliado em 3D.** `can_transmit()` usa a distância euclidiana nos três eixos contra `transmission_range`. Como Lamenza §3 abstrai a altitude, **todos os nós devem ser criados com `z = 0`**; caso contrário o raio efetivo no solo seria `sqrt(r² − h²)` e não `r`.

3. **`BROADCAST` não entrega ao próprio emissor** (o laço pula `destination == sender.id`). Com `delay = 0`, a entrega é agendada no **mesmo timestamp** da emissão.

4. **`MasslessMobilityHandler` não aceita vetor de velocidade.** Ele trata `SET_SPEED` lendo **apenas `param_1`**. `SetVelocityMobilityCommand(vx, vy, vz)` usa o mesmo `command_type` com três parâmetros e, sob o handler *massless*, seria silenciosamente interpretado como `speed = vx`, **sem erro e mantendo o alvo anterior**. Isto contradiz a tabela da §1 do `tmp/PLANO.md` e é a armadilha mais cara desta spec. Consequência: a ação `⟨θ, s⟩` se aplica como **`SetSpeedMobilityCommand(s)` + `GotoCoordsMobilityCommand(waypoint)`** (ver T-00.6). `SetVelocityMobilityCommand` só é válido com `DynamicVelocityMobilityHandler`, que por sua vez **rejeita** `GOTO_COORDS`.

5. **A mobilidade emite telemetria para todos os nós a cada tique**, inclusive para os sensores (estáticos), porque `Simulator.create_node` registra todo nó em todos os handlers. Isso dá o modelo de custo:

   ```
   eventos ≈ (T_sim / update_rate) · (1 + Na + Ns)  +  eventos de comunicação e de temporizador
   ```

   O modelo **reproduz as duas medições do PRD §6.1** para `T_sim = 120 s`, `Na = 4`, `Ns = 12`: `update_rate = 0,01` → `12000 · 17 = 204 000` contra 205 653 medidos; `update_rate = 0,10` → `1200 · 17 = 20 400` contra 22 080 medidos. A diferença (~1 650 eventos nos dois casos) é exatamente a parcela independente de `update_rate`. **O custo é dominado por `(Na + Ns)`, não por `Na`** — o que significa que a célula cara não é a de 8 UAVs, é a de 36 sensores.

6. **`Simulator` não tem `reset()`.** As posições são sorteadas por episódio (RF-01), logo **cada episódio constrói um `Simulator` novo** via `SimulationBuilder`. O custo de construção por episódio precisa ser medido (T-00.10).

7. **`Simulator.__init__` adiciona um `StreamHandler` ao logger raiz**, incondicionalmente (`setup_simulation_formatter`), e nunca o remove. Um `Simulator` por episódio × 10 000 episódios = **10 000 handlers acumulados no logger raiz**. É vazamento de memória e degradação progressiva garantidos. Mitigação obrigatória em T-00.12.

8. **`execution_logging=True` (o default) formata uma string por evento** (`scope_event` monta `timedelta` e f-string a cada `step_simulation`). Com dezenas de milhares de eventos por episódio isso não é ruído: é custo de primeira ordem. `execution_logging=False` faz `scope_event` retornar imediatamente.

9. **Parar cedo exige bombear o laço manualmente.** `start_simulation()` só devolve quando o event loop esvazia ou `duration` é atingido — e o event loop **nunca esvazia**, porque `MasslessMobilityHandler._update_movement` sempre reagenda a si mesmo. Portanto a poda por `max_seconds_stalled` **não é implementável dentro de `start_simulation()`**: o driver do episódio precisa chamar `step_simulation()` em laço e checar a terminação. Isso é compatível com a injeção de política do PRD §7 (a política continua sendo chamada *inline* dentro de `handle_timer`); o que muda é apenas quem gira a manivela.

10. **Se o episódio para cedo, `finish()` dos protocolos não é chamado** (`_finalize_simulation` é privado e só roda pelas condições internas). Consequência de projeto: **nenhuma métrica pode morar em `protocol.finish()`**. A verdade do episódio é do driver, não dos protocolos.

11. **Determinismo do event loop:** `EventLoop` é um min-heap de `Event`, com `__lt__` comparando **apenas `timestamp`**. Empates são resolvidos pela estrutura do heap — determinístico para a mesma ordem de inserção, mas **não semanticamente definido**. Ver o risco de ordenação em §7.

12. **`PositionScheme.random` e `failure_rate` usam o `random` global da stdlib.** Nada do nosso código deve depender deles: as posições são geradas pelo nosso RNG e passadas explicitamente a `add_node`, e `failure_rate = 0`.

### 4.2 Configuração-alvo dos handlers

```python
SimulationConfiguration(
    duration=max_episode_length,     # 500.0 s (tab. 7.1)
    execution_logging=False,         # ver fato 8
    debug=False,
    real_time=False,
)
TimerHandler()
MasslessMobilityHandler(MasslessMobilityConfiguration(update_rate=?, default_speed=0.0))
CommunicationHandler(CommunicationMedium(transmission_range=communication_range, delay=0.0, failure_rate=0.0))
```

`update_rate` fica com valor a **decidir por medição** em T-00.10 (candidato 0,1). `default_speed=0.0` garante que nenhum UAV se mova antes da primeira decisão.

### 4.3 Incertezas remanescentes da API — a confirmar

Nenhuma destas bloqueia o desenho, mas todas devem ser confirmadas ao codificar, e **nenhum nome de método deve ser inventado**:

- **Ordem de execução de eventos com timestamp idêntico.** Verificado que é determinística dada a ordem de inserção, mas não que seja estável entre versões nem entre plataformas. Precisa de teste próprio (T-00.12).
- **Se `handle_telemetry` é sempre entregue antes do `handle_timer` de mesma marca temporal.** Não há garantia no código. O desenho abaixo **não depende disso** (a verdade é lida diretamente de `Node.position`, não da telemetria), e essa independência é deliberada.
- **Custo real de `SimulationBuilder.build()` por episódio.** Não medido; entra em T-00.10.
- **Comportamento do `MasslessMobilityHandler` quando o alvo é atingido exatamente:** o código faz *snap* para o alvo quando `speed·update_rate ≥ distância restante` e mantém o alvo registrado (não há `STOP` acessível por comando). O nó fica parado sobre o alvo. Confirmado por leitura; confirmar por teste.
- **`gradysim` 0.8.0 é a versão fixada.** Qualquer atualização de versão exige repetir §4.1. Fixar em `pyproject.toml` com `==`, não `>=`.

---

## 5. Tarefas

### T-00.1 — Resolver P1 e o conjunto de parâmetros não publicados

**É a primeira tarefa do projeto** (PRD §10, P1). Não é opcional e não é adiável: sem ela, nenhum número desta spec é comparável aos publicados.

Fonte: repositório `Project-GrADyS/gradys-data-collection`, branch `dissertation` (Lamenza §7.1, nota de rodapé 1).

Extrair, com **citação de arquivo e linha** para cada item:

| Parâmetro | Por que importa | Status |
|---|---|---|
| `communication_range` | define a dinâmica de coleta inteira; não consta nas tabelas 3.1/7.1/7.2 | **P1, bloqueante** |
| período do heartbeat do UAV | ver T-00.3: controla se um UAV rápido pode *atravessar* o alcance de um sensor sem detectá-lo | não consta na dissertação |
| `update_rate` da mobilidade usada por Lamenza | fidelidade da trajetória; é o objeto de P2 | não consta |
| tratamento de borda (o UAV é limitado a `[−100, 100]²`?) | Lamenza §3 diz que as posições "assumem valores no intervalo"; não diz se há saturação, reflexão ou nada, e não há penalidade de borda na recompensa | não consta |
| regra exata de terminação quando todos os sensores foram coletados | ver T-00.8 — as duas leituras do §5 de Lamenza divergem | ambígua no texto |
| como o tempo de conclusão é agregado em episódios que **não** completam | afeta a comparação com a fig. 7.3 | não consta |
| arquitetura do ator do MADDPG | P4 do PRD; barato de extrair na mesma visita | pendente |

**Saída:** uma tabela em `reports/spec00-parametros.md` com, para cada linha, **valor encontrado + referência** ou **"não encontrado"**. Para todo item não encontrado: registrar um **valor de trabalho** com justificativa explícita e marcá-lo como suposição no PRD.

> **Se `communication_range` não for encontrado:** rodar toda a spec com 2–3 valores candidatos e reportar a sensibilidade, exatamente como a SPEC-01 já faz. Se as conclusões de V-1 ou de P2 mudarem entre eles, P1 escala para bloqueante do projeto e precisa de decisão humana (contato com o autor).

### T-00.2 — Cenário e semeadura reprodutível (`env/scenario.py`)

Estruturas de dados. Nada de simulador aqui: é geração pura, para ser compartilhada com o gerador analítico da SPEC-01.

```python
@dataclass(frozen=True)
class ScenarioConfig:
    n_agents: int                    # Na
    n_sensors: int                   # Ns
    scenario_size: float = 100.0     # tab. 7.1; area = 2*scenario_size por lado
    altitude: float = 0.0            # ver fato 2 da §4.1: z=0 em TODOS os nós
    spawn_radius: float = ?          # raio da "small area near the center" (Lamenza §3) -- PENDENTE, ver §9
    max_speed: float = 15.0          # eq. 3-2
    algorithm_iteration_interval: float = 2.0
    max_episode_length: float = 500.0
    max_seconds_stalled: float = 30.0
    communication_range: float = ?   # P1
    heartbeat_interval: float = ?    # T-00.3
    update_rate: float = ?           # T-00.10

@dataclass(frozen=True)
class Scenario:
    config: ScenarioConfig
    agent_positions: NDArray[np.float64]   # (Na, 2)
    sensor_positions: NDArray[np.float64]  # (Ns, 2)

def generate_scenario(config: ScenarioConfig, rng: np.random.Generator) -> Scenario: ...
def episode_seed(root_seed: int, episode_index: int) -> np.random.Generator: ...
```

Contratos:
- Sensores: **uniformes** em `[−scenario_size, scenario_size]²` (Lamenza §3: "randomly distributed... in a flat square area with size 2·scenario_size").
- UAVs: uniformes em uma **área pequena perto do centro**. O raio dessa área não consta na dissertação — é pendência (§9). A figura 7.4 de Lamenza mostra as trajetórias partindo praticamente do mesmo ponto central, o que sugere raio pequeno em relação a 100 m.
- `episode_seed` deriva de um `np.random.SeedSequence` raiz por `spawn(episode_index)`, para que o episódio `i` seja reproduzível **isoladamente**, sem depender de rodar os `i−1` anteriores. Isso é o que faz `make repro` funcionar sob paralelismo.
- Semear também `random.seed(...)` da stdlib por episódio, por precaução (§4.1, fato 12), mesmo com `failure_rate = 0`.
- **Nenhum uso de `PositionScheme`.**

### T-00.3 — Protocolos e coleta automática por proximidade (`env/protocols.py`)

**RF-03: coletar não é ação do agente.** Lamenza §3: o UAV emite heartbeats periódicos; o sensor que recebe responde com seus dados e passa a ser considerado visitado.

O `gradysim` instancia protocolos por `cls()` sem argumentos (§4.1, fato "instantiate"). Portanto **a injeção de dependências precisa de um mecanismo explícito**. Duas opções, ambas legítimas; escolher uma e documentar:

- **(a) subclasse por episódio:** `type("BoundUAVProtocol", (UAVProtocol,), {"_ctx": ctx})`, passando a classe a `add_node`.
- **(b) registro global de contexto por episódio**, lido em `initialize()` via `self.provider.get_id()`.

Recomendação: **(a)** — não usa estado global, é seguro sob paralelismo por processos e mantém `add_node(protocol_type, position)` na forma que a API exige.

```python
class SensorProtocol(IProtocol):
    """Nó estático. Não emite nada por conta própria; só responde a heartbeat."""
    def initialize(self) -> None: ...            # nada a agendar
    def handle_timer(self, timer: str) -> None: ...   # nunca dispara
    def handle_packet(self, message: str) -> None:
        # message == f"hb:{uav_id}" -> registra a visita no EpisodeState e responde
    def handle_telemetry(self, t: Telemetry) -> None: ...  # no-op (chega mesmo assim, ver fato 5)
    def finish(self) -> None: ...                # no-op (pode não ser chamado, ver fato 10)

class UAVProtocol(IProtocol):
    def initialize(self) -> None:
        # agenda "decide" em t=0 e "heartbeat" em t=phase
    def handle_timer(self, timer: str) -> None:
        # "decide"    -> ver T-00.9
        # "heartbeat" -> BroadcastMessageCommand(f"hb:{self.provider.get_id()}"); reagenda
    def handle_packet(self, message: str) -> None:
        # f"data:{sensor_id}" -- resposta do sensor
```

**Onde a visita é registrada.** O estado verdadeiro é do `EpisodeState` (T-00.4), não dos protocolos. Duas variantes possíveis: marcar (i) quando o **sensor recebe** o heartbeat, ou (ii) quando o **UAV recebe** a resposta com os dados. Com `delay = 0` e `failure_rate = 0`, as duas entregas ocorrem **no mesmo timestamp** e a resposta nunca falha por alcance — logo **as duas variantes são equivalentes na configuração desta etapa**. Adotar (i), que é a leitura literal de Lamenza ("If the agent is within range of a sensor, data harvesting will be automatically performed and that sensor will be considered as visited"), manter a resposta por fidelidade, e registrar em teste que (i) e (ii) coincidem — porque na Etapa 2, com `delay > 0` ou perda, elas deixam de coincidir e a escolha passa a importar.

**O período do heartbeat é um parâmetro de fidelidade, não um detalhe.** Entre dois heartbeats o UAV percorre até `v_max · heartbeat_interval` metros. Se isso exceder o diâmetro do alcance, o UAV pode **atravessar** a cobertura de um sensor sem detectá-lo, e o ambiente passa a ser mais difícil que o de Lamenza sem que nada apareça na tabela de desvios. Regra a codificar como asserção:

```
v_max · heartbeat_interval ≤ α · communication_range,   α ≤ 1 (recomendado α = 0,5)
```

Com `v_max = 15 m/s`, isso amarra `heartbeat_interval` a `communication_range` — que é P1. **Não fixar um número antes de T-00.1.** Se P1 não resolver, reportar a sensibilidade da taxa de coleta a `heartbeat_interval` junto com a de `communication_range`.

Custo: cada heartbeat gera um broadcast que percorre `Na + Ns − 1` destinos e agenda um evento por destino **em alcance**. Isso entra no orçamento de T-00.10.

### T-00.4 — Estado verdadeiro do episódio e a interface `StateView`

**Quem é dono da verdade:** o `EpisodeState`, criado pelo driver e compartilhado com os protocolos pelo contexto. Nem `SensorProtocol` nem `UAVProtocol` mantêm cópia autoritativa de nada.

```python
class EpisodeState:
    scenario: Scenario
    visited: NDArray[np.bool_]              # (Ns,)
    visit_time: NDArray[np.float64]         # (Ns,), NaN se não visitado
    last_visit_time: float                  # para max_seconds_stalled
    completion_time: float | None           # t do instante em que visited.all() virou True
    _nodes: dict[int, Node]                 # referências vivas; Node.position é mutado in loco

    def agent_positions(self) -> NDArray   # (Na, 2), lido de Node.position
    def sensor_positions(self) -> NDArray  # (Ns, 2)
    def mark_visited(self, sensor_index: int, t: float) -> bool  # idempotente; devolve True só na 1a vez
```

`Node.position` é reatribuído (não mutado internamente) pelo handler de mobilidade, e a referência ao `Node` continua válida — verificado. Guardar as referências obtidas por `simulator.get_node(i)` na construção evita um lookup por acesso.

**A interface `StateView` (ponto de inserção da SPEC-07).** O construtor de observação **não** recebe o `EpisodeState`; recebe uma vista:

```python
class StateView(Protocol):
    def agent_positions(self) -> NDArray[np.float64]   # (Na, 2)
    def sensor_positions(self) -> NDArray[np.float64]  # (Ns, 2)
    def visited_mask(self) -> NDArray[np.bool_]        # (Ns,)
```

Na Etapa 1, `TruthStateView(episode_state)` devolve a verdade — é o "canal global via satélite" de Lamenza §3, e é **fiel ao original**. Na Etapa 2, `NoisyStateView` entra no lugar e a verdade continua no `EpisodeState`, exclusivamente para métricas. É isso que torna **RF-04** (a verdade não vaza para a política) um teste de tipo e não uma promessa.

Custo de introduzir isso agora: praticamente zero. Custo de introduzir depois: reescrever o construtor de observação e todos os testes que dependem dele.

### T-00.5 — Observação (Lamenza eq. 5-1) — `env/observation.py`

**Regra de fonte única.** Existe **uma** função que constrói o vetor de observação, e tanto o simulador (esta spec) quanto o gerador analítico (`env/analytic.py`, SPEC-01) a chamam. Se houvesse duas implementações, o teste KS de V-1 mediria a diferença entre dois códigos e não a diferença entre duas dinâmicas — que é o que se quer medir. A SPEC-01 deve ser refatorada para consumir esta função.

```python
def build_observation(
    view: StateView,
    agent_index: int,          # k, base 0
    n_star_a: int,             # Na - 1 (tab. 7.1); 0 na réplica exploratória (Lamenza §7.1)
    n_star_s: int = 12,        # tab. 7.1
) -> NDArray[np.float32]:
    """Devolve vetor de forma (2*n_star_a + 2*n_star_s + 1,)."""
```

Contratos, todos verbatim de Lamenza §5.1:

1. **Sinal das coordenadas relativas.** Eq. 5-2 e 5-5: `(x_k − x_i, y_k − y_i)`, isto é, **posição do agente menos a do outro** — o vetor que aponta do outro *para* o agente. É o contrário da convenção intuitiva "vetor até o alvo" e é o erro de sinal mais fácil de cometer neste projeto. Codificar como está e cobrir com teste.
2. **Ordenação por distância crescente** (eq. 5-4), agentes e sensores separadamente. **Desempate por índice de nó**, para determinismo.
3. `P^s` contém **apenas sensores não visitados** (eq. 5-5). Sensores já visitados nunca aparecem — incluir `visited_i` mediu pior no original.
4. **Zero-padding ao fim** quando restam menos de `n_star_s` sensores não visitados: slots vazios valem `(0.0, 0.0)` e ficam **depois** dos preenchidos.
5. **Sem normalização.** As unidades são metros brutos, suporte em `[−200, 200]` (diferença de duas coordenadas em `[−100, 100]`). A normalização em Lamenza é **da ação**, não da observação (§5.2). Escalar a observação aqui seria alterar o problema para a MLP pareada e mudaria o alvo do estudo de binarização.
6. **O próprio agente não entra em `P^a`.**
7. **Último elemento = `float(agent_index)`.** Lamenza indexa `i ∈ {1..Na}`; adotamos base 0 e documentamos. Para `Na = 1` o valor é constante 0 e a componente é informativa apenas de que existe um único agente.
8. **Dimensão:** `2(Na−1) + 24 + 1`. `Na = 1` → 25; `Na = 4` → 31; `Na = 8` → 39 (confere com PRD §5.2).
9. **A função é pura** — mesma vista, mesma saída. Sem estado, sem RNG, sem tempo.

Nota sobre o custo: a ordenação é `O(Ns log Ns)` e é o estágio "montagem da observação" da tabela T2 (PRD §5.5), **compartilhado por todos os métodos**, inclusive pelo guloso. Esta função é o alvo de medição daquela linha; a instrumentação é da SPEC-03.

### T-00.6 — Ação contínua (Lamenza eq. 3-2, §5.2)

A política devolve `u ∈ [0, 1]²`. A desnormalização é do ambiente:

```python
def denormalize_action(u: NDArray) -> tuple[float, float]:
    theta = 2.0 * math.pi * clip(u[0], 0.0, 1.0)     # [0, 2pi]
    speed = 15.0 * clip(u[1], 0.0, 1.0)              # [0, 15] m/s
    return theta, speed
```

**Aplicação no `gradysim`** — depende do fato 4 da §4.1 e é o ponto onde o `PLANO.md` está errado:

```
provider.send_mobility_command(SetSpeedMobilityCommand(speed))
provider.send_mobility_command(GotoCoordsMobilityCommand(
    x + speed * dt * cos(theta),
    y + speed * dt * sin(theta),
    0.0))
```

com `dt = algorithm_iteration_interval` e `(x, y)` a posição corrente do UAV. O waypoint é colocado **exatamente** na distância que o UAV percorreria no intervalo: quando o alvo é alcançado, o handler faz *snap* e o nó fica parado, o que torna o esquema auto-limitado e imune a *overshoot* numérico. A alternativa (waypoint muito distante, nunca alcançado) percorre a mesma distância mas deixa o UAV livre para acumular deriva se uma decisão falhar.

**Asserção obrigatória:** `algorithm_iteration_interval` deve ser **múltiplo inteiro de `update_rate`**. O handler move `speed · update_rate` por tique; se `dt` não for múltiplo, o UAV chega ao waypoint no meio de um tique e a distância efetivamente percorrida por intervalo deixa de ser `speed · dt`. Com `dt = 2 s` e `update_rate = 0,1` a razão é 20 — exata. Com `update_rate = 0,11` não seria, e a velocidade efetiva ficaria silenciosamente abaixo da comandada. Codificar como `assert` no construtor do cenário, não como comentário.

**`speed = 0`.** Direção fica indefinida e o waypoint colapsa na posição atual. Comportamento: o UAV não se move. É válido e admissível (`s ∈ [0, 15]`, extremo inclusive). Cobrir com teste para garantir que não há divisão por zero nem `NaN`.

**Borda do cenário.** Nada no `gradysim` limita a posição. Se Lamenza satura em `[−100, 100]²`, é preciso saturar o waypoint; se não satura, o UAV pode sair da área — e como não há penalidade de borda na eq. 5-6, sair da área é apenas desperdício de tempo. **É pendência de T-00.1 e §9.** Implementar como opção de configuração (`clamp_to_bounds: bool`) com o default definido pela resposta de T-00.1, e não escolher em silêncio.

### T-00.7 — `env/rewards.py` (eq. 5-6 e eq. 7-2, verbatim)

```python
def reward_5_6(visited_now: NDArray[np.bool_], visited_prev: NDArray[np.bool_]) -> float:
    """Lamenza eq. 5-6. Recompensa de EQUIPE: a formula nao tem indice de agente."""
    delta = int(visited_now.sum()) - int(visited_prev.sum())
    if delta > 0:
        return 10.0 * delta
    n_s = visited_now.size
    return (int(visited_now.sum()) - n_s) / n_s

def reward_7_2(visited_now: NDArray[np.bool_]) -> float:
    """Lamenza eq. 7-2, usada SOMENTE na replica do experimento exploratorio."""
    return int(visited_now.sum()) / visited_now.size
```

Contratos:
- **A recompensa é global.** `R_t` na eq. 5-6 não tem índice de agente. Todos os `Na` agentes recebem o **mesmo escalar** no mesmo passo. Isso não é uma escolha nossa e não é negociável sem linha nova na §5.4 do PRD.
- **`Δvisited_t` é somado sobre todos os sensores** e é a diferença entre dois passos de decisão consecutivos — não entre dois eventos do simulador. O passo de decisão é `algorithm_iteration_interval`.
- **Uma recompensa por época de decisão, memoizada pelo timestamp.** Os `Na` temporizadores "decide" disparam no mesmo instante em ordem arbitrária (§4.1, fato 11); calcular a recompensa por agente produziria `Δvisited` diferente para o primeiro e para os demais. O `EpisodeState` calcula uma vez por época e serve a todos.
- Sem *shaping*, sem penalidade de borda, sem termo de coordenação (PRD D3).
- Observação que decorre da fórmula e explica a regra de não poda: quando todos os sensores estão visitados, `Δ = 0` e o ramo "caso contrário" vale exatamente **0**, em vez de negativo. Continuar o episódio após a coleta completa entrega, portanto, uma sequência de recompensas `0` que domina o retorno de qualquer estado incompleto (sempre negativo). É consistente com a explicação de Lamenza §5 ("longer duration positive reward signal"). Registrar como leitura nossa, não como afirmação do autor.

### T-00.8 — Gestão de episódio e terminação

Três regras, e a terceira é a que Lamenza mediu como crítica:

1. **Limite duro:** o episódio termina em `t ≥ max_episode_length = 500 s`. Implementado por `SimulationConfiguration(duration=500.0)` **e** por checagem no driver (cinto e suspensório: o driver precisa saber *por que* terminou).
2. **Poda por parada (`max_seconds_stalled = 30 s`):** se `t − last_visit_time > 30`, o episódio é podado. `last_visit_time` é inicializado em `0.0` (o relógio de parada corre desde o começo).
3. **RF-02 — não podar quando todos os sensores foram coletados.** Lamenza §5 é explícito: podar imediatamente episódios completos degrada fortemente o desempenho. Portanto a regra 2 é **suspensa** assim que `visited.all()` for verdadeiro.

```python
class TerminationReason(str, Enum):
    TIME_LIMIT = "time_limit"      # t >= max_episode_length
    STALLED    = "stalled"         # sem visita ha max_seconds_stalled, com sensores restantes
    # nao existe reason "completed": completar NAO termina o episodio
```

> **Ambiguidade a decidir por humano (§9).** "Não podar quando todos foram coletados" tem duas leituras: **(A)** a regra de parada é suspensa e o episódio corre até 500 s; **(B)** a regra de parada continua valendo, e o episódio acaba 30 s após a última visita — o que já dá "mais experiências após o objetivo", mas encerra bem antes de 500 s. O texto de Lamenza favorece (A) ("letting the agents collect more experiences"), mas o custo de simulação de (A) é várias vezes maior, porque **todo episódio bem-sucedido passa a rodar os 500 s completos**. Isso muda o orçamento de T-00.10 por um fator grande. **Default proposto: (A)**, por ser a leitura literal, com a decisão registrada e (B) medida como variante em T-00.10 para quantificar o custo.

**Métricas cruas expostas pelo episódio** (agregação e IC são da SPEC-02):

```python
@dataclass(frozen=True)
class EpisodeResult:
    scenario: Scenario
    seed: int
    reason: TerminationReason
    end_time: float
    completion_time: float | None     # instante da ULTIMA visita, se completou; None se nao
    n_visited: int
    visit_times: NDArray[np.float64]  # (Ns,), NaN para nao visitados
    n_decisions: int
    n_events: int
    wall_time: float
```

`completion_time` é o instante da última visita, **não** o instante do fim do episódio — é a métrica "completion time" da fig. 7.3 de Lamenza. Episódios que não completam têm `None`; como agregá-los na média é pendência de T-00.1 e responsabilidade da SPEC-02.

### T-00.9 — Ponte simulador ↔ treinador (`env/bridge.py`)

O `gradysim` é orientado a eventos: `step_simulation()` processa **um evento**, não um instante. O RL quer `obs → ação`. A conciliação é a **injeção de política** (PRD §7, alternativa (a)): o `UAVProtocol` chama a política *inline* dentro de `handle_timer("decide")`. Sem corrotinas, sem inversão de controle, válido para métodos **on-policy** — que é exatamente o caso do n-tuple Actor-Critic (PRD D6).

**Contrato da política** — interface de agente clássica (estilo RL-Glue), que é a que encaixa numa chamada síncrona:

```python
class Policy(Protocol):
    def start_episode(self, agent_index: int, obs: NDArray) -> NDArray: ...
    def step(self, agent_index: int, reward: float, obs: NDArray) -> NDArray: ...
    def end_episode(self, agent_index: int, reward: float) -> None: ...
```

Devolve `u ∈ [0,1]²` (`shape (2,)`, `float32`). A política é **compartilhada** pelos `Na` agentes (*parameter sharing*, PRD §5.3) e recebe `agent_index` para manter o estado por agente que o aprendizado exigir (obs/ação anteriores). Ela **não** conhece o simulador, o `EpisodeState` nem a verdade — só recebe o vetor de observação. Esse é o contrato que a SPEC-05 implementa e que o baseline guloso desta spec também satisfaz.

**Fluxo em `handle_timer("decide")`:**

```
1. r  = episode_state.reward_for_epoch(t)     # memoizado por timestamp
2. o  = build_observation(view, k, ...)
3. u  = policy.step(k, r, o)                  # ou start_episode na primeira decisao
4. aplica a acao (T-00.6)
5. reagenda "decide" para t + algorithm_iteration_interval
```

**Coerência dentro de uma época.** Os `Na` eventos "decide" de um mesmo instante disparam em ordem arbitrária, mas a observação é a mesma para todos eles: posições só mudam no tique de mobilidade e a máscara de visitados só muda por heartbeat. Vale a pena afirmar isso em teste, porque é o que garante que a ordem de disparo não altera o resultado.

**Driver do episódio:**

```python
class EpisodeRunner:
    def __init__(self, config: ScenarioConfig, root_seed: int): ...
    def run(self, policy: Policy, episode_index: int) -> EpisodeResult: ...
```

Internamente:
```
scenario = generate_scenario(config, episode_seed(root_seed, episode_index))
constroi handlers e SimulationBuilder; UAVs primeiro (ids 0..Na-1), sensores depois
sim = builder.build()
remove o StreamHandler que Simulator.__init__ adicionou ao logger raiz   # fato 7
while not episode_state.terminated:
    if not sim.step_simulation(): break
    episode_state.check_termination(timer_handler.get_current_time())
policy.end_episode(k, r_final) para cada k
return EpisodeResult(...)
```

`timer_handler.get_current_time()` é o único acessor **público** de tempo fora do protocolo (§4.1). Não usar `Simulator._current_timestamp`.

O treino inteiro é, então, `for i in range(n_episodes): runner.run(policy, i)` — sem processos, sem filas, sem corrotinas.

**Baseline guloso mínimo.** Esta spec entrega `GreedyPolicy`, que satisfaz `Policy` e escolhe `θ` na direção do sensor não visitado mais próximo com `s = 15`. Ele existe aqui apenas como **instrumento de validação** (T-00.10 e T-00.11) e é lido diretamente da observação — o slot `j = 1` de `P^s`, com o sinal invertido conforme o contrato 1 de T-00.5. Sua versão instrumentada, com métricas e IC, é da SPEC-02.

### T-00.10 — `update_rate`, orçamento de simulação e fidelidade (pendência P2)

Duas perguntas distintas, frequentemente confundidas: **(i) quanto custa** e **(ii) o resultado muda**.

**(i) Custo.** Validar o modelo de eventos da §4.1 (fato 5) e medir o custo real:

- eventos por episódio e tempo de parede, na grade `Na ∈ {1,2,4,8} × Ns ∈ {12,24,36}`, com `update_rate ∈ {0,01; 0,05; 0,10; 0,25}` e episódios de 500 s;
- **custo de `SimulationBuilder.build()` por episódio**, separado do custo do laço de eventos — é o overhead fixo de não haver `reset()`, e com episódios curtos pode dominar;
- efeito de `execution_logging=False` contra o default (§4.1, fato 8), medido, não presumido;
- custo das duas leituras da regra de não poda (§T-00.8, A vs B): sob (A) todo episódio completo roda os 500 s.

Extrapolação a reportar: passos de treino → tempo de parede, contra a estimativa de 8–30 h em um núcleo do PRD §6.1. **O PRD marca essa estimativa como estimativa; esta tarefa a substitui por medição.**

**(ii) Fidelidade.** Comparar `update_rate` candidato contra `0,01` (a referência) com **política fixa e determinística** (`GreedyPolicy`), mesmas sementes, para toda a grade:

| Critério | Aceite proposto |
|---|---|
| Sequência de sensores visitados (ordem e identidade) | **idêntica** |
| `completion_time` | diferença ≤ 1 intervalo de decisão (2 s) |
| Taxa de sucesso sobre ≥ 200 episódios | diferença dentro do IC |
| Desvio máximo de posição ao longo da trajetória | ≤ uma fração pequena de `communication_range` (fixar a fração após P1) |

Restrição estrutural que precede a medição: `algorithm_iteration_interval` deve ser múltiplo inteiro de `update_rate` (T-00.6). Isso já elimina candidatos como `0,11` ou `0,3`.

**Atenção especial ao `Na = 8`** — é o caso que o `PLANO.md` levantou explicitamente como não validado. Mas o modelo de eventos diz que o custo cresce com `(1 + Na + Ns)`: **a célula cara é `Ns = 36`, não `Na = 8`**. A grade acima cobre as duas e o relatório deve dizer qual das duas dominou.

**Saída:** valor de `update_rate` fixado com evidência, tabela de custo por célula e extrapolação de horas de treino — em `reports/spec00-orcamento.md`. Isso fecha P2.

### T-00.11 — Invariante V-1: teste Kolmogorov–Smirnov contra o gerador analítico

O invariante V-1 da SPEC-01 diz: se a distribuição de observações do gerador analítico divergir da do simulador, **o estudo de binarização foi otimizado para a distribuição errada e precisa ser refeito**. Este é o teste que fecha ou dispara isso, e é a tarefa de maior valor por hora desta spec.

**Protocolo:**

1. Rodar `GreedyPolicy` nos dois lados — simulador e gerador analítico da SPEC-01 — com **as mesmas sementes, o mesmo `ScenarioConfig` e o mesmo `communication_range`**, sobre `Na ∈ {1,2,4}` × `Ns ∈ {12,24,36}`.
2. Coletar as observações emitidas em cada época de decisão. Como as duas pontas chamam a **mesma** `build_observation` (T-00.5), qualquer divergência é atribuível à **dinâmica**, que é o objeto do teste.
3. Para cada componente escalar (as `2·n_star_a + 24` coordenadas): **KS de duas amostras**.
4. Para a componente discreta `k`: KS não se aplica. Usar distância de variação total ou qui-quadrado.
5. Estatísticas agregadas que importam mais que qualquer componente isolada, e que devem ser comparadas explicitamente: **distância ao sensor não visitado mais próximo**, **número de sensores não visitados ao longo do tempo**, **tempo de conclusão** e **taxa de sucesso**.

**Como julgar** — e aqui está a armadilha: com ~10⁶ amostras, o p-valor do KS rejeita qualquer diferença, inclusive irrelevantes. Além disso, observações consecutivas de um mesmo episódio são **fortemente autocorrelacionadas**, o que viola a hipótese de independência do teste e infla ainda mais a rejeição. Duas mitigações, ambas obrigatórias:

- **Afinar a amostra:** uma observação por episódio por agente (época sorteada), o que restaura a independência aproximada entre amostras.
- **Julgar por tamanho de efeito, não por p-valor:** limiar declarado *a priori* sobre a estatística `D` do KS (proposta: `D ≤ 0,05` por componente), acompanhado da sobreposição visual dos histogramas. Registrar `D`, `n` e o p-valor, mas **o critério é `D`**.

**Divergência já prevista, e que deve ser procurada primeiro.** O rollout analítico da SPEC-01 marca o sensor como visitado quando `d ≤ communication_range` **na época de decisão** (a cada 2 s). O simulador marca por **heartbeat**, que é mais frequente. Logo, o simulador detecta passagens rápidas pela cobertura que o gerador analítico perde: **o gerador analítico deve coletar menos**, e a distribuição de "número de sensores não visitados" deve divergir de forma sistemática, não aleatória. Se for esse o caso, a correção é do **gerador analítico** — checar o alcance ao longo do segmento percorrido (distância mínima do segmento ao sensor), ou subamostrar o segmento no período do heartbeat — e não do simulador, que é a referência.

**Saída:** tabela `D` por componente e por célula, figuras de sobreposição, e um **veredito escrito**: V-1 satisfeito, ou SPEC-01 a refazer com o corpus corrigido. O veredito vale mais que a tabela.

### T-00.12 — Testes, determinismo e higiene

**Testes de contrato do MDP** (`tests/test_env.py`):

| # | Teste |
|---|---|
| 1 | Dimensão da observação = `2(Na−1) + 24 + 1` para `Na ∈ {1,2,4,8}` |
| 2 | Sinal das coordenadas relativas: cenário montado à mão, comparação com `(x_k − x_i, y_k − y_i)` calculado na mão |
| 3 | Ordenação por distância crescente, com desempate por índice |
| 4 | Sensores visitados **nunca** aparecem em `P^s` |
| 5 | Zero-padding: slots vazios valem `(0,0)` e ficam ao fim |
| 6 | `build_observation` é pura: mesma vista → saída idêntica, bit a bit |
| 7 | `denormalize_action`: `u = (0,0) → (θ=0, s=0)`; `u = (1,1) → (θ=2π, s=15)`; `s = 0` não move o UAV nem produz `NaN` |
| 8 | Distância percorrida em um intervalo = `s · dt` dentro de tolerância, para `s ∈ {0; 7,5; 15}` |
| 9 | `reward_5_6`: `Δ = 2 → 20,0`; `Δ = 0` com 3 de 12 visitados → `−0,75`; **todos visitados e `Δ = 0` → `0,0`** |
| 10 | Recompensa é idêntica para todos os agentes na mesma época |
| 11 | Coleta por proximidade: sensor a `communication_range − ε` é coletado; a `communication_range + ε` não é |
| 12 | Marcar visita é idempotente; o segundo heartbeat não recompensa de novo |
| 13 | **RF-02:** episódio com todos os sensores coletados aos 100 s **não** termina aos 130 s (sob a leitura A) |
| 14 | Poda por parada dispara em `max_seconds_stalled` quando há sensores restantes |
| 15 | Episódio termina em `max_episode_length` mesmo sem parada |
| 16 | **RF-04 (pré-registro):** `build_observation` recebe apenas um `StateView`; teste de tipo garantindo que a assinatura não expõe `EpisodeState`. O teste de vazamento propriamente dito é da SPEC-07 |
| 17 | Ordem de disparo dos "decide" na mesma época não altera as observações nem a recompensa |

**Determinismo** (`tests/test_determinism.py`):

| # | Teste |
|---|---|
| 18 | Mesma semente → `EpisodeResult` idêntico (tempos de visita bit a bit) |
| 19 | Episódio `i` reproduzido isoladamente == episódio `i` dentro de uma sequência (`SeedSequence.spawn`) |
| 20 | Nenhum uso de `random` global nem `np.random` legado no caminho do episódio |

**Higiene** (`tests/test_hygiene.py`):

| # | Teste |
|---|---|
| 21 | **Vazamento de logging:** 200 episódios seguidos deixam `len(logging.getLogger().handlers)` constante (§4.1, fato 7) |
| 22 | Sem crescimento de memória ao longo de 200 episódios (referências a `Simulator`/`Node` coletadas) |
| 23 | Asserção ativa: `algorithm_iteration_interval % update_rate == 0` |
| 24 | Asserção ativa: `v_max · heartbeat_interval ≤ α · communication_range` |
| 25 | Todos os nós criados com `z = 0` (§4.1, fato 2) |

---

## 6. Critério de saída

1. **P1 fechada** (T-00.1): `communication_range` e o período de heartbeat com valor e referência, ou com valor de trabalho declarado e sensibilidade reportada.
2. **P2 fechada** (T-00.10): `update_rate` fixado por medição, com tabela de custo por célula e extrapolação de horas de treino substituindo a estimativa do PRD §6.1.
3. **V-1 com veredito escrito** (T-00.11): tabela `D` do KS por componente, e a decisão de aceitar o corpus da SPEC-01 ou refazê-lo.
4. **Os 25 testes de T-00.12 passando**, com destaque para os invariantes RF-02 (não poda), RF-03 (coleta não é ação) e o pré-registro de RF-04.
5. **Um episódio completo executa ponta a ponta** com `GreedyPolicy` e produz `EpisodeResult` preenchido, em todas as células da grade `Na ∈ {1,2,4,8} × Ns ∈ {12,24,36}`.
6. **Relatório curto** (`reports/spec00-ambiente.md`, 2–3 páginas) com: tabela de parâmetros e sua proveniência, tabela de orçamento, resultado de V-1, e a lista de **suposições** feitas onde a fonte era silenciosa. Essa última lista é a parte que mais importa para a honestidade do E2: ela é o que separa "reprodução fiel" de "reprodução com lacunas não declaradas".
7. **`pyproject.toml` fixa `gradysim==0.8.0`**, exatamente.

**Não é critério de saída:** desempenho de qualquer política, qualquer número de custo T1/T2/T3, qualquer comparação com o guloso com IC. Tudo isso é SPEC-02, SPEC-03 e SPEC-06.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **P1 não resolve** (`communication_range` não está no repositório) | 2–3 valores candidatos em toda a spec, com sensibilidade reportada; se V-1 ou P2 mudarem de veredito entre eles, escalar para decisão humana (contato com o autor). É o risco mais caro do projeto: sem ele, nenhum número é comparável aos publicados |
| **`SetVelocityMobilityCommand` sob mobilidade *massless*** — comando aceito, interpretado como `speed = vx`, **sem erro** | Contrato explícito de T-00.6 (`SetSpeedMobilityCommand` + `GotoCoordsMobilityCommand`) e teste 8, que mede a distância percorrida em vez de confiar no comando |
| **Vazamento de handlers de logging** (um `StreamHandler` por episódio no logger raiz) | Poda explícita após `build()` e teste 21. Sem isso, o treino degrada progressivamente e o diagnóstico é caro |
| **Heartbeat esparso demais** faz o UAV atravessar a cobertura sem coletar, tornando o ambiente mais difícil que o original sem aparecer na tabela de desvios | Asserção `v_max · heartbeat_interval ≤ α · r` (teste 24) e sensibilidade reportada junto com P1 |
| **`algorithm_iteration_interval` não múltiplo de `update_rate`** → velocidade efetiva silenciosamente menor que a comandada | Asserção no construtor do cenário (teste 23); candidatos de `update_rate` restritos a divisores de 2 s |
| **Ambiguidade da regra de não poda (A vs B)** muda o orçamento de simulação por um fator grande | Default (A), decisão registrada, e (B) medida em T-00.10 para que o custo da escolha seja conhecido e não presumido |
| **V-1 falha** e a SPEC-01 precisa ser refeita | Causa mais provável já prevista e diagnosticada (detecção de coleta na época vs por heartbeat); a correção é do gerador analítico e é barata. Rodar V-1 **cedo**, não no fim da spec |
| **KS rejeita tudo** por `n` grande e por autocorrelação intraepisódio | Afinamento a uma observação por episódio + critério declarado sobre `D`, não sobre p-valor |
| **Erro de sinal nas coordenadas relativas** — `(x_k − x_i)` é o contrário da convenção intuitiva | Teste 2 com cenário calculado à mão; o guloso desta spec lê a direção da própria observação, então um erro de sinal aparece como um guloso que foge dos sensores — falha ruidosa, não silenciosa |
| **Ordem de eventos com timestamp idêntico** não é semanticamente definida no `gradysim` | Teste 17 mostra que o resultado não depende dela; onde depender, o desenho é alterado, não o simulador |
| **Custo real estoura a estimativa do PRD** (8–30 h em um núcleo) | O modelo de eventos da §4.1 já permite extrapolar antes de medir; se estourar, a linha de corte do PRD §8 age sobre `Ns ∈ {24,36}` e `Na = 8`, que são exatamente as células caras segundo o modelo |
| **Sem `reset()`, o overhead de construção domina** episódios curtos | Medido separadamente em T-00.10; se dominar, avaliar reaproveitar o `Simulator` reposicionando nós — mas isso toca `_event_loop`, que é privado, e só se justifica com número na mão |
| Ambiente e gerador analítico divergirem no futuro por edição de um só lado | Regra de fonte única em T-00.5: `build_observation` é uma função, chamada pelos dois |

---

## 8. Entregáveis

- `src/uavwnn/env/scenario.py` — `ScenarioConfig`, `Scenario`, `generate_scenario`, `episode_seed`
- `src/uavwnn/env/protocols.py` — `SensorProtocol`, `UAVProtocol` (`IProtocol`), ligação de contexto por subclasse
- `src/uavwnn/env/state.py` — `EpisodeState`, `StateView`, `TruthStateView`, `EpisodeResult`, `TerminationReason`
- `src/uavwnn/env/observation.py` — `build_observation` (eq. 5-1), **fonte única**, consumida também pela SPEC-01
- `src/uavwnn/env/rewards.py` — `reward_5_6`, `reward_7_2`, verbatim
- `src/uavwnn/env/bridge.py` — `Policy` (protocolo), `EpisodeRunner`, `GreedyPolicy` mínima
- `configs/spec00/` — cenários da grade, `update_rate`, sementes
- `reports/spec00-parametros.md` — proveniência de cada parâmetro (fecha P1, adianta P4)
- `reports/spec00-orcamento.md` — custo e fidelidade de `update_rate` (fecha P2)
- `reports/spec00-ambiente.md` — relatório curto, com V-1 e a lista de suposições
- `tests/test_env.py`, `tests/test_determinism.py`, `tests/test_hygiene.py`

---

## 9. Perguntas em aberto — decidir antes de codificar

Cada item precisa de decisão humana. Onde há recomendação, ela é proposta, não conclusão.

1. **`communication_range` (P1).** Bloqueante. Se o repositório não responder, decidir entre (i) escolher um valor e declará-lo como suposição no PRD §5.4, (ii) reportar tudo como família de curvas sobre 2–3 valores, ou (iii) contatar o autor. *Recomendação: (ii) até o fim da semana 1; (iii) em paralelo, porque a latência de uma resposta humana é dias.*

2. **Regra de não poda: leitura (A) ou (B)?** (A) o episódio corre até 500 s após a coleta completa; (B) a poda por parada continua valendo e ele acaba 30 s depois da última visita. Muda o orçamento de simulação por um fator grande e muda o sinal de aprendizado. *Recomendação: (A), leitura literal, com o custo de (B) medido para comparação.*

3. **Período do heartbeat.** Não consta na dissertação e controla diretamente a taxa de coleta. Está acoplado a `communication_range` pela asserção de T-00.3. Decidir se é (i) fixado por regra (`α · r / v_max`), (ii) igual a `algorithm_iteration_interval`, ou (iii) o que o repositório disser. *Recomendação: (iii), com (i) como plano B.*

4. **Raio da área de largada dos UAVs.** Lamenza §3 diz "small area near the center" e não dá número. Afeta a dificuldade inicial e o quanto a coordenação importa nos primeiros segundos (a fig. 7.1 mostra trajetórias praticamente sobrepostas na largada, o que é consequência direta desse raio ser pequeno). Precisa de valor declarado.

5. **Tratamento de borda.** O UAV pode sair de `[−100, 100]²`? Não há penalidade de borda na eq. 5-6 e o `gradysim` não limita nada. Se Lamenza satura e nós não, nosso ambiente é diferente; se Lamenza não satura e nós saturamos, idem. *Precisa vir de T-00.1; sem resposta, escolher e declarar.*

6. **Índice do agente: base 0 ou base 1, e como entra no vetor?** Lamenza usa `i ∈ {1..Na}`. Como escalar cru, `k` é uma variável categórica disfarçada de número — para a MLP pareada isso é o que o original faz, mas para a WNN o codificador tem de tratá-la como categórica (one-hot), o que é assunto da SPEC-01. Decidir a convenção **aqui** e mantê-la nos dois lados.

7. **Como o tempo de conclusão é agregado quando o episódio não completa?** Média só sobre os que completaram (viesa a favor), `max_episode_length` como censura (viesa contra), ou reportar taxa de sucesso e tempo condicional separadamente. Afeta a comparação direta com a fig. 7.3. *Recomendação: reportar os dois separadamente, e a média censurada só se for isso o que Lamenza fez.*

8. **Réplica exploratória (Lamenza §7.1): qual é o `max_episode_length` "permissivo"?** O texto diz "permissive" sem número. Sem ele, a taxa de sucesso de 100 % não é comparável — uma taxa de sucesso alta com tempo infinito é trivial. Precisa de valor do repositório ou de uma declaração explícita nossa.

9. **A recompensa de equipe é mesmo entregue igual a todos os agentes?** A eq. 5-6 não tem índice de agente, o que é evidência forte. Mas MADDPG admite recompensa por agente, e o repositório pode revelar uma atribuição individual (por exemplo, `Δvisited` creditado a quem coletou). Se o repositório divergir da equação, **a equação vence** (PRD D1/D3) e a divergência é registrada.

10. **Instrumentação de observações para V-1: sempre ligada ou opcional?** Gravar 10⁶ observações custa I/O e memória. *Recomendação: opcional, ligada por configuração, desligada no treino.*

11. **Se o custo medido em T-00.10 estourar o orçamento**, a ordem de corte é a do PRD §8 (células `Ns ∈ {24,36}` e `Na = 8` primeiro) ou vale subir `update_rate` acima de 0,1 sacrificando fidelidade? *Recomendação: cortar células, nunca fidelidade — a fidelidade é o que o princípio da §0 do PRD protege; o número de células é só cobertura.*
