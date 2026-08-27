# O simulador e o cenário

O problema que todos os modelos deste projeto resolvem: o que o drone vê, o que
ele pode fazer, o que ele ganha, e quando acaba.

**Ressalva que define o escopo:** o ambiente é o de Lamenza, **verbatim**. Vem do
submodule `vendor/gradys-data-collection`, tag `dissertation` (`be94ca7`),
cenário `003-collaboration`. Nosso código apenas **carrega, configura e semeia** —
nada altera observação, ação, recompensa, coleta ou terminação.

---

## 1. A tarefa

Um enxame de drones sobrevoa uma arena quadrada onde há sensores parados no chão.
Cada sensor guarda dados. Um drone que chega perto o bastante **coleta** os dados
daquele sensor. O episódio acaba quando todos foram coletados.

O objetivo é **coletar tudo o mais rápido possível**.

```
   arena 200 × 200 m  (scenario_size = 100, de −100 a +100)

   ┌───────────────────────────────┐
   │   ●        ●         ●        │   ● sensor não coletado
   │        ✈                ●     │   ○ sensor coletado
   │   ○          ●    ✈           │   ✈ drone (raio de 20 m)
   │        ●            ○         │
   │  ●            ●          ●    │
   └───────────────────────────────┘
```

---

## 2. Glossário

### Do cenário

| símbolo | nome | valor na tab. 7.1 | o que é |
|---|---|---|---|
| `Na` | drones | 1, 2, 4 ou 8 | quantos agentes |
| `Ns` | sensores | 12, 24 ou 36 | quantos pontos a coletar |
| `scenario_size` | meia-aresta | **100 m** | a arena vai de −100 a +100 |
| `communication_range` | alcance | **20 m** | distância para coletar |
| `algorithm_iteration_interval` | `dt` | **0,5 s** | tempo entre decisões |
| `max_episode_length` | horizonte | **500 s** | teto do episódio |
| `max_seconds_stalled` | | 30 s | corta episódio empacado |

### Do drone

| símbolo | o que é | faixa |
|---|---|---|
| `θ` | rumo comandado | `[0, 2π)` |
| `s` | velocidade comandada | `[0, 15]` m/s |

### Da observação

| símbolo | o que é |
|---|---|
| `p` | posição do drone que observa |
| `state_num_closest_sensors` | quantos sensores entram na observação (= `Ns`) |
| `state_num_closest_drones` | quantos outros drones entram (= `Na − 1`) |
| `id_on_state` | se a identidade do drone entra (**sim**) |

---

## 3. O simulador — `gradysim`

`gradysim==0.6.11`, fixado. É um simulador de eventos discretos para redes de
sensores e VANTs. Duas coisas dele importam aqui:

**Mobilidade.** O drone recebe comandos `SetSpeedMobilityCommand` e
`GotoCoordsMobilityCommand`; o simulador integra o movimento entre eventos. O
drone **não teleporta** — ele viaja.

**Comunicação.** Sensor e drone trocam pacotes quando estão dentro do
`communication_range`. É isso que efetiva a coleta.

O ambiente RL é uma casca `ParallelEnv` (PettingZoo) sobre isso —
`environment.py:129`.

> **Por que não reimplementamos.** A primeira versão do projeto reimplementou o
> MDP e provou equivalência por teste diferencial (`max|Δobs| = 7,9·10⁻⁸`). Foi
> descartada assim mesmo: uma "correção" inócua — inicializar `current_position`
> na posição de spawn em vez de `(0,0,0)` — fazia a divergência saltar para
> **8,5·10⁻¹** em 60 passos. Não por magnitude, por **caos**. Detalhes em
> `reports/spec00-parametros.md`.

---

## 4. A observação — o que o drone vê

Modo `relative` (`environment.py:331–375`). Para cada drone:

```python
345  sensor_distances = np.linalg.norm(unvisited_sensor_nodes - agent_position, axis=1)
346  sorted_sensor_indices = np.argsort(sensor_distances)
```

**Só os sensores NÃO coletados entram**, ordenados por distância. Depois:

```python
365  closest_agents[:len(sorted_agent_indices) - 1] = (agent_position - closest_agents[:len(
366      sorted_agent_indices) - 1] + self.scenario_size) / (self.scenario_size * 2)
```

Posições **relativas ao drone** e normalizadas para `[0, 1]`. O vetor final:

```python
370  state[f"drone{agent_index}"] = np.concatenate([
371      closest_agents.flatten(),              # 2 · (Na−1) valores
372      closest_unvisited_sensors.flatten(),   # 2 · Ns valores
373      np.array([agent_index / self.num_drones]) if self.id_on_state else []
374  ])
```

### O tamanho — e por que ele NÃO depende de `Ns`

```
d_obs = 2·(Na − 1)  +  2·closest_sensors  +  1
        ─────────      ──────────────────    ─
        outros drones   sensores vistos      id
```

E `closest_sensors` é **fixado em 12** na tab. 7.1, em todas as células. O drone
vê os **12 sensores não coletados mais próximos**, tenha o cenário 12, 24 ou 36.

O vetor tem sempre `2 × 12 = 24` posições para sensores, **mesmo quando quase
todos foram coletados**: as vagas sobrando ficam preenchidas com `−1`
(linha 350, `closest_unvisited_sensors.fill(-1)`).

Medido:

| célula | `closest_sensors` | `closest_drones` | `d_obs` |
|---|---|---|---|
| Na=1, Ns=12 | 12 | 0 | **25** |
| Na=2, Ns=**12** | 12 | 1 | **27** |
| Na=2, Ns=**24** | 12 | 1 | **27** |
| Na=2, Ns=**36** | 12 | 1 | **27** |
| Na=4, Ns=24 | 12 | 3 | 31 |
| Na=8, Ns=24 | 12 | 7 | 39 |

As três células de `Na=2` dão **o mesmo `d_obs`**, com 12, 24 e 36 sensores.

> **É o que torna a transferência possível.** Uma rede treinada em `Ns=24` roda em
> `Ns=12` ou `Ns=36` sem retreino, porque a entrada tem a mesma forma. A campanha
> explora isso, e marca essas corridas como `transferencia=1` — o MADDPG tem
> checkpoint por célula e não transfere, então a assimetria fica registrada.

### Três consequências

**A observação é parcial.** O drone não sabe onde estão os sensores fora dos 12
mais próximos, nem onde estão drones distantes.

**Ela é egocêntrica.** Tudo é relativo à posição própria — não há sistema de
coordenadas global na entrada.

**O `−1` é sinal de ausência.** Vaga preenchida com `−1` significa "não há sensor
aqui", e a política precisa aprender isso.

---

## 5. A ação — o que o drone faz

`Box(0, 1, shape=(2,))` — dois números em `[0,1]`. Viram movimento em
`environment.py:59–103`:

```python
 62  direction: float = action[0] * 2 * math.pi
 65      speed: float = action[1] * 15
```

`a₀` → rumo em `[0, 2π)`. `a₁` → velocidade em `[0, 15]` m/s.

### O drone não pode sair da arena

```python
 71  distance_to_x_edge = coordinate_limit - abs(self.current_position[0])
 78      scale = min(scale_x, scale_y)
 80      destination = [
```

O destino é escalado para parar **na borda** — o drone mantém o rumo mas não
atravessa. Se já estiver na borda apontando para fora, o destino é recortado
(linhas 93–94).

> É por isso que uma política ruim **empaca**: apontar para fora da arena
> significa ficar parado na borda. Foi o mecanismo por trás do achado do
> *dithering* — ruído de GPS destrava a política porque perturba o rumo o
> suficiente para sair do canto.

---

## 6. A recompensa

`environment.py:626–652`. Três variantes; usamos a `punish` (a réplica 1b):

```python
628  if self.reward == 'punish':
632          reward = (after - before) * 10
634          reward = -(self.num_sensors - sum(sensor_is_collected)) / self.num_sensors
```

| situação | recompensa |
|---|---|
| coletou `k` sensores neste passo | `+10·k` |
| não coletou nada | `−(sensores restantes)/Ns` |

**A punição é proporcional ao que falta.** Com 12 sensores, ficar parado no começo
custa `−1,0` por passo; faltando um só, custa `−0,083`. A pressão diminui à medida
que o trabalho avança.

> As duas recompensas ficam em `campanha/recompensas.py`, calculadas **fora** do
> ambiente. É o que permite treinar com uma e avaliar com a métrica de outra sem
> tocar no simulador.

---

## 7. A terminação

```python
654  simulation_ended = (all_sensors_collected and self.end_when_all_collected) or not simulation_ongoing
657  truncations = {agent: False for agent in self.agents}
```

Três formas de acabar:

| | quando |
|---|---|
| **sucesso** | todos os sensores coletados |
| **horizonte** | 500 s de simulação |
| **empacado** | 30 s sem progresso (`max_seconds_stalled`) |

### `end_when_all_collected` — a diferença que muda a métrica

| fase | valor | efeito |
|---|---|---|
| **treino** | `False` | o episódio continua após a coleta — dá experiência pós-conclusão |
| **avaliação** | `True` | para na coleta — é o protocolo do `visualize_performance.py` |

Usar uma no lugar da outra **muda a métrica sem erro nenhum**. É a razão de
`_cfgs` devolver duas configurações.

---

## 8. As métricas

| métrica | o que é |
|---|---|
| **CT** (`completion_time`) | tempo até coletar tudo, em segundos |
| **CT_cens** | CT com os fracassos contados como 500 s (**censurado**) |
| **CT_succ** | CT só dos episódios que completaram |
| **SR** (taxa de sucesso) | fração de episódios que coletaram tudo |
| **cobertura** | fração média de sensores coletados |

**Por que `cobertura` para selecionar e `SR` para reportar.** O `SR` é binário e
satura: no começo do treino ele é 0 % em quase todo cenário, e não distingue "quase
lá" de "não fez nada". A `cobertura` é contínua e dá gradiente de decisão. Mas o
número que Lamenza reporta é o `SR`, então é ele que vai ao relatório.

**A censura importa.** Média de CT sobre episódios que completaram é enviesada: a
política que desiste dos cenários difíceis parece rápida. O `CT_cens` conta o
fracasso como 500 s, e é o que compara justo.

---

## 9. A configuração canônica

`LAMENZA_003` é a tab. 7.1 codificada (`refenv/runner.py`). Para `Na=1, Ns=12`:

```
algorithm_iteration_interval  0.5      max_episode_length      500.0
communication_range           20.0     max_seconds_stalled     30
scenario_size                 100.0    reward                  punish
speed_action                  True     state_mode              relative
id_on_state                   True     state_num_closest_drones  0
end_when_all_collected        True     state_num_closest_sensors 12
                                       → observation_size = 25
```

As nove células publicadas: `Na ∈ {2,4,8} × Ns ∈ {12,24,36}`. **Não há `Na = 1`**
nos checkpoints do MADDPG — Lamenza não treinou single-drone. É a célula que
usamos para a grade da WNN, e ali a referência é o guloso.

---

## 10. Os pisos e tetos de referência

Medidos em `Na=1, Ns=12` (`agents/baselines.py`):

| política | cobertura | SR | CT |
|---|---|---|---|
| caminhante aleatório | 3,3 % | 0 % | — |
| **guloso** (vê o mapa inteiro) | 100 % | **100 %** | **≈ 34 s** |
| PPO não treinado | 10,8 % | 0 % | — |

O **guloso** é o teto prático: ele conhece todas as posições e vai sempre ao
sensor não coletado mais próximo. Uma política com observação parcial não deveria
batê-lo — chegar perto já é o resultado.

> A melhor WNN treinada faz `CT_succ ≈ 38,5 s` — **13 % acima do guloso**,
> decidindo por observação local e sem sinalização entre drones.

---

## 11. Onde cada coisa vive

| assunto | arquivo:linha |
|---|---|
| ambiente (fonte) | `vendor/gradys-data-collection/scenario/003-collaboration/environment.py` |
| ação → movimento | `environment.py:59–103` |
| observação `relative` | `environment.py:331–375` |
| espaços | `environment.py:201–233` |
| recompensa | `environment.py:626–652` |
| terminação | `environment.py:654–657` |
| nosso carregador | `src/uavwnn/refenv/loader.py` |
| config e laço | `src/uavwnn/refenv/runner.py` |
| recompensas (fora do env) | `src/uavwnn/campanha/recompensas.py` |
| baselines | `src/uavwnn/agents/baselines.py` |
| proveniência e ressalvas | `reports/spec00-parametros.md` |

## 12. O que fica registrado

**O ambiente é intocado.** Toda a nossa liberdade está em quem decide a ação —
nunca em o que a ação faz, o que o drone vê, ou o que ele ganha.

**A versão está fixada** — `gradysim==0.6.11`, submodule em `be94ca7`. Trocar
qualquer uma das duas invalida a comparação com os números publicados, e não há
erro que avise.

**Os bancos de cenários são versionados.** As sementes geram posições, e regenerar
um banco com outra versão do NumPy daria cenários diferentes com a mesma semente
— comparações já registradas deixariam de valer em silêncio.
