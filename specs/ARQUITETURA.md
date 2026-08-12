# Arquitetura de integração — como as features entram sem tocar no cenário

O ambiente é o `GrADySEnvironment` de Lamenza, verbatim, num submodule. **Nenhuma linha dele é editada.** Este documento define onde cada feature do trabalho se encaixa e qual teste guarda cada costura.

Duas features:

1. **Substituir o MADDPG pela rede sem peso** (Etapa 1).
2. **Ruído gaussiano na medição de posição dos drones**, e o filtro que o trata (Etapas 2–3).

---

## 1. O mapa

```
┌──────────────────────────────────────────────────────────────┐
│  vendor/gradys-data-collection  (submodule, tag dissertation) │
│  GrADySEnvironment  —  PettingZoo ParallelEnv                 │
│  reset() -> obs ;  step(actions) -> obs, rewards, term, info  │
│  VERBATIM. Física, coleta, recompensa e terminação são dele.  │
└───────────────┬──────────────────────────────────────────────┘
                │  obs[agent] : (2·N*_a + 2·N*_s + 1,) em [0,1], padding -1
                │
      ┌─────────▼──────────┐   COSTURA A — só nas Etapas 2 e 3
      │ NoisyGrADysEnv     │   subclasse; sobrescreve APENAS
      │ (sobrescreve       │   observe_simulation_relative_positions()
      │  a observação)     │   Guarda: com sigma=0, saída bit-idêntica ao pai
      └─────────┬──────────┘
                │  obs corrompida + medição z_k da posição própria
                │
      ┌─────────▼──────────┐   COSTURA B — Etapas 2 e 3
      │ Estimador (KF)     │   entrada de controle = ação anterior
      │  p̂ -> p̃           │   corrige por deslocamento comum
      └─────────┬──────────┘
                │  obs filtrada
                │
      ┌─────────▼──────────┐   COSTURA C — todas as etapas
      │ Codificador        │   termômetro: reais -> bits  (SPEC-01)
      └─────────┬──────────┘
                │  vetor binário
      ┌─────────▼──────────┐   COSTURA D — Etapa 1 em diante
      │ Aproximador        │   n-tuple Actor-Critic  |  MLP AC (pareado)
      └─────────┬──────────┘
                │  u ∈ [0,1]²  =  (direção, velocidade)
                └──────────────► env.step()
```

Nenhuma costura está **dentro** do ambiente. A física — movimento, alcance de comunicação, coleta, recompensa, terminação — permanece exatamente a de Lamenza em todas as etapas.

---

## 2. Feature 1 — substituir o MADDPG pela rede sem peso

### 2.1 Onde o MADDPG vive hoje

No `main.py` de referência, não no `environment.py`. O ambiente não sabe que algoritmo o controla: entrega `obs` e recebe `actions`. **A substituição do aproximador não toca o cenário** — é troca de quem consome a mesma API.

### 2.2 O que exatamente é substituído

| Peça | Referência (MADDPG) | Nosso |
|---|---|---|
| Ator | MLP `Actor` (CleanRL), saída `Box(0,1,shape=(2,))` | rede n-tuple, mesma saída |
| Crítico | `QNetwork` centralizado, recebe obs+ações de todos | crítico n-tuple de `V`, descentralizado |
| Algoritmo | MADDPG (off-policy, replay, target networks) | Actor-Critic on-policy (Katopodis §4.3.4) |
| Entrada | vetor real de dimensão `2N*_a + 2N*_s + 1` | **mesmo vetor**, binarizado (COSTURA C) |

**O algoritmo muda junto com o aproximador**, e isso é deliberado: o n-tuple não tem gradiente para retropropagar por um crítico centralizado. Por isso o PRD (D7) já estabelece que o pareamento honesto é **n-tuple AC contra MLP AC** — mesmo algoritmo, aproximadores diferentes — e o MADDPG entra como referência de literatura, nunca como baseline pareado.

### 2.3 Onde o código mora

```
src/uavwnn/
├── refenv/          # PRONTO — carrega, configura e semeia o ambiente
├── encoding/        # COSTURA C — termômetro (SPEC-01)
├── wnn/             # núcleo n-tuple esparso (SPEC-04)
├── agents/
│   ├── ntuple_ac.py # ator+crítico n-tuple (SPEC-05)
│   └── mlp_ac.py    # MESMO algoritmo, aproximador denso (o pareado)
└── train/runner.py  # laço on-policy sobre a API PettingZoo
```

O laço de treino consome a mesma interface que o `visualize_performance.py` usa:

```python
obs, _ = env.reset()
while True:
    actions = {a: policy(obs[a]) for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    ...
```

### 2.4 Restrições que o ambiente impõe ao aproximador

Descobertas ao ler o código, e que a SPEC-01/SPEC-05 precisam respeitar:

1. **A observação já vem normalizada** — `(p_k − p_i + S)/(2S)`, com valores válidos em torno de `[0,1]`. Não é preciso normalizar de novo, e **não se deve**: seria alteração de entrada.
2. **Padding é `-1`**, fora da faixa dos válidos. Slot vazio e "sensor sobre o UAV" (0,5) são distinguíveis por construção. O **bit de validade projetado na SPEC-01 é desnecessário**.
3. **Faixa real é `[-0,5; 1,5]`**, não `[0,1]`: `p_k − p_i` vive em `[-2S, 2S]`. O `observation_space` de referência declara `Box(-1,1)`, o que também não bate. O termômetro precisa cobrir a faixa **medida**, não a declarada.
4. **A ação é `Box(0,1,shape=(2,))`** — `u₀` vira `θ = 2π·u₀` e `u₁` vira `v = 15·u₁`. A circularidade de `θ` é problema do ator (decisão C1: normal projetada), não do ambiente.

### 2.5 Guarda desta costura

`tests/test_refenv.py::test_guloso_reproduz_a_figura_7_3` — a heurística de referência reproduz a fig. 7.3 em 9/9 células (erro mediano 1,7 %). Qualquer coisa que quebre a montagem aparece como desvio dessa curva **antes** de qualquer política aprendida entrar.

---

## 3. Feature 2 — ruído gaussiano na posição dos drones

### 3.1 O princípio: corromper a percepção, nunca a física

Em Lamenza a política recebe o estado verdadeiro por um "canal global". O ruído fecha esse buraco. Mas ele entra **só na observação**:

- a posição **verdadeira** continua governando movimento, alcance de comunicação e coleta;
- a posição **estimada** é o que a política enxerga.

Isso é RF-04 do PRD, e agora é verificável por construção: a subclasse só sobrescreve o método que **constrói a observação**. Tudo que é física está no `Simulator`, fora do alcance dela.

### 3.2 O achado que simplifica tudo: o erro de GPS próprio é deslocamento comum

Toda entrada da observação tem a forma `(p_k − p_i + S)/(2S)`, com `p_k` = posição do agente. Se a estimativa própria erra por `e`:

```
(p̂_k − p_i + S)/(2S)  =  (p_k − p_i + S)/(2S)  +  e/(2S)
```

**O mesmo `e/(2S)` é somado a todas as entradas válidas** — sensores e vizinhos —, e o padding `-1` não é tocado. Verificado numericamente.

Três consequências práticas:

1. **A corrupção é uma soma vetorial de duas linhas**, não uma reconstrução da observação.
2. **A correção do filtro também é**: aplicar o KF equivale a somar `(p̃_k − p̂_k)/(2S)`.
3. **A escala do erro é `σ/(2S)`.** Com `S = 100 m`, `σ = 5 m` desloca 0,025 em unidades de observação — o que se compara diretamente à largura do bin do termômetro. É a regra *bin ≈ σ* do PRD, agora em unidades da entrada real.

### 3.3 Os três canais e onde cada um entra

| Canal | Modelo | Como entra na observação | Etapa |
|---|---|---|---|
| **Posição própria (GPS)** | `p̂_k = p_k + e_k`, `e_k ~ N(0, σ_own² I)` | deslocamento **comum** `+e_k/(2S)` em todas as entradas válidas | 2 |
| **Posição dos sensores** | `p̂_i = p_i + e_i`, do levantamento; **fixa no episódio** | deslocamento `−e_i/(2S)` na entrada daquele sensor | 2 |
| **Posição dos vizinhos** | `p̂_j = p_j + e_j`, recebida por broadcast, **atrasada** | deslocamento `−e_j/(2S)` na entrada daquele vizinho | 3 |

Duas observações que mudam expectativa:

**O erro dos sensores é constante no episódio, não branco por passo.** O sensor não se move e a posição vem de um levantamento. Sortear `e_i` a cada passo seria modelar errado — e tornaria o problema artificialmente mais fácil, porque a média temporal cancelaria o erro.

**O ganho do filtro é limitado pelo pior canal.** Como a observação é relativa, `σ_rel = √(σ_own² + σ_sensor²)`. Filtrar `σ_own` de 10 m para 0,7 m com `σ_sensor = 5 m` leva `σ_rel` de 11,2 m para 5,05 m — fator **2,2×**, não 15×. O achado A3 do PRD (18,8×) é sobre a posição própria, não sobre o que a política enxerga. Isto é a pendência **T2**.

### 3.4 A ordenação: uma idealização a declarar

O ambiente seleciona e ordena os `N*_s` sensores não visitados mais próximos usando as posições **verdadeiras**. Com posição própria ruidosa, um agente real ordenaria pela distância **percebida**, e ocasionalmente veria outro conjunto ou outra ordem.

Duas leituras possíveis:

- **(i) Herdar seleção e ordenação verdadeiras** e apenas deslocar as coordenadas. Simples, e o erro é de segunda ordem — só muda quando dois sensores distam menos que `σ` entre si.
- **(ii) Reordenar pelas posições estimadas.** Mais fiel, e exige recomputar a seleção a partir de todos os sensores — acessível na subclasse por leitura do `Simulator`.

**Recomendação: (ii)**, porque é o que um agente real faria e porque em `Ns = 36` num quadrado de 200 m a distância típica entre vizinhos é da ordem de 30 m, comparável a `σ = 10 m`. A opção (i) fica como ablação para quantificar quanto a idealização vale. A escolha entra na tabela de desvios §5.4 do PRD como **efeito de observação**, não de MDP.

### 3.5 Onde o filtro de Kalman se encaixa

O KF estima a posição própria a partir do GPS ruidoso **e da ação que a política acabou de tomar**:

```
predição:   p̃⁻ = p̃ + u·dt,    u = 15·u₁·[cos(2π·u₀), sin(2π·u₀)]
            P⁻  = P + Q
correção:   K   = P⁻(P⁻ + R)⁻¹,  R = σ_own² I
            p̃   = p̃⁻ + K(z − p̃⁻),  P = (I − K)P⁻
```

Três amarras com o que já foi medido:

1. **`u` é a ação do passo anterior**, disponível no wrapper — é o que o achado A3 explora, e a razão de o KF de velocidade constante falhar: a política muda de rumo a cada decisão.
2. **`dt = algorithm_iteration_interval = 0,5 s`**, e o comando é cumprido quase exatamente (mobilidade sem massa), então `Q` é pequeno e representa sobretudo a discretização. Com mobilidade sem inércia o modelo de processo é quase exato — a Etapa 2 deve declarar isso como limitação, porque torna o degrau 4 mais fácil do que seria com dinâmica realista.
3. **A posição lida numa decisão está um tique de mobilidade atrasada** (quirk Q3 do relatório). O KF precisa saber disso, ou o resíduo terá viés sistemático.

A saída do KF entra na observação como o deslocamento `(p̃_k − p̂_k)/(2S)` da §3.2 — nenhuma reconstrução.

### 3.6 A observação consciente da incerteza (degrau 5)

O KF entrega `P` além de `p̃`. Acrescentar `√tr(P)` à observação **aumenta a dimensão do vetor** e portanto é a única parte desta feature que altera a entrada da rede. Isto **exige linha na tabela de desvios §5.4** e é ablação explícita, reportada com e sem.

### 3.7 Guardas desta costura

| Guarda | O que garante |
|---|---|
| `σ = 0` ⇒ observação **bit-idêntica** à do pai | a subclasse não introduz diferença por si |
| Deslocamento medido == `e/(2S)` em todas as entradas válidas; padding intacto | a álgebra da §3.2 vale na implementação |
| Posições verdadeiras no `Simulator` inalteradas após corrupção | o ruído não vazou para a física |
| Tempo de conclusão com `σ = 0` == fig. 7.3 | a corrupção desligada não muda o baseline |
| NEES/NIS no intervalo qui-quadrado | o filtro é consistente (RF-05) |

Os dois primeiros já estão verificados; entram como teste na SPEC-07.

---

## 4. O que muda nas specs

| Spec | Mudança |
|---|---|
| **SPEC-00** | Concluída. Ambiente é o de referência; escopo vira carregar/configurar/semear + validação contra a fig. 7.3 |
| **SPEC-01** | Faixa da entrada é `[-0,5; 1,5]` e não `[-200, 200]` m; **bit de validade cai**; padding `-1` é sentinela natural; varredura passa a ser sobre o par `(T, N)` por causa da cobertura de bits (C4) |
| **SPEC-02** | Baselines saem do `heuristics.py` de referência, verbatim — guloso, `smart` e aleatório. Não reimplementar |
| **SPEC-04/05** | Inalteradas no essencial; ganham a restrição de que a entrada já vem normalizada |
| **SPEC-07** | Ponto de inserção passa a ser **subclasse do ambiente**, não `StateView`. A álgebra do deslocamento comum simplifica a implementação e dá guardas fortes |
| **SPEC-08/09** | Inalteradas; herdam as ressalvas de comparação da §4 do relatório de parâmetros |

---

## 5. O que **não** entra por aqui

- **Sinalização entre agentes** (mensagem de reivindicação): mudaria o protocolo, que é do ambiente. Fora de escopo (PRD D5).
- **`detection_range` separado do `communication_range`**: exigiria alterar o mecanismo de coleta. Descartado — seria modificar o problema para caber no método.
- **Mobilidade com inércia**: é configuração do simulador dentro do `reset()` de referência. Só entraria por subclasse do `reset()`, o que é mais invasivo que sobrescrever a observação. Fica como trabalho futuro declarado.
