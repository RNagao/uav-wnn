# SPEC-00 — Ambiente de referência: proveniência e ressalvas

**O ambiente do trabalho é o de Lamenza, verbatim.** Não há reimplementação. O código vem do submodule `vendor/gradys-data-collection`, fixado na tag `dissertation` (`be94ca7`), cenário `003-collaboration`.

Nosso código (`src/uavwnn/refenv/`) apenas **carrega, configura e semeia** aquele ambiente. Nada nele altera observação, ação, recompensa, coleta ou terminação.

---

## 1. Por que não reimplementamos

A primeira versão desta spec reimplementou o MDP e provou, por teste diferencial, que era equivalente ao original (`max|Δobs| = 7,9 × 10⁻⁸`, `Δrecompensa = 0`, em 60 passos × 5 sementes).

**Ainda assim foi descartada.** O motivo é uma medição feita naquele processo: uma "correção" aparentemente inócua — inicializar `current_position` na posição de spawn em vez de `(0,0,0)`, que é o que o original faz — fazia a divergência de observação saltar de **7,9 × 10⁻⁸ para 8,5 × 10⁻¹** em 60 passos. A causa não é magnitude, é **caos**: a posição errada entra no cálculo da escala do waypoint, altera levemente o rumo da primeira ação, e as trajetórias divergem por completo.

Se uma linha desse porte tem esse efeito, manter uma implementação paralela é superfície de risco permanente contra o único ativo que o trabalho tem: a comparabilidade com o baseline. O ambiente passa a ser o deles.

---

## 2. Versão do simulador — e por que ela importa

**`gradysim==0.6.11`**, fixado.

O `environment.py` de referência importa `VisualizationController`. Verificado por inspeção dos wheels do PyPI:

| versão | `VisualizationController` |
|---|---|
| 0.6.9, 0.6.10, 0.6.11 | **presente** |
| 0.7.0, 0.7.1, 0.7.3–0.7.5, 0.8.0 | ausente |

Rodar o código intacto contra outra versão do simulador seria, por si só, uma divergência do baseline — e exigiria patches no arquivo de referência, que é justamente o que se quer evitar. Sob 0.6.11 o arquivo importa e roda **sem nenhuma alteração**.

`tests/test_refenv.py` trava as duas coisas: o commit do submodule e a versão do gradysim.

---

## 3. Parâmetros: onde cada valor foi encontrado

Fonte de mecanismo: `scenario/003-collaboration/`. Fonte de valores: `benchmark.py` (treino) e `visualize_performance.py` (avaliação) — **não** os defaults do `main.py`, que são de teste.

| Parâmetro | Valor | Proveniência | Pendência |
|---|---|---|---|
| `communication_range` | **20 m** | `environment.py:157` | **P1** |
| Período do heartbeat | **`0,99 × interval`** | `environment.py:102` | **P5** |
| Raio da área de largada | **2 m** (quadrado) | `environment.py:533` | **P6** |
| Tratamento de borda | destino limitado a `[-100,100]²` | `environment.py:71-95` | **P8** |
| Agregação sem sucesso | censura em `max_episode_length` | `environment.py:674` | **L1** |
| Recompensa de equipe | mesmo escalar para todos | `environment.py:646` | Q9 |
| `N*_s`, `N*_a` | 12 e `Na − 1` | `benchmark.py` | tab. 7.1 |
| Modo de recompensa | `punish` = eq. 5-6 | `benchmark.py` | eq. 5-6 |
| `min_sensor_priority` | 1 | `benchmark.py` | — |
| `total_timesteps` | 50 000 000 | `benchmark.py` | §7.2 |
| `algorithm_iteration_interval` | **0,5 s** | `benchmark.py` | ver §4.1 |
| Arquitetura do ator do MADDPG | *pendente* | `main.py`, via CLI | **P4** |

---

## 4. Ressalvas sobre os números publicados

Estas não são divergências nossas — são propriedades do trabalho de referência que precisam ser declaradas ao citarmos a fig. 7.3.

### 4.1 `algorithm_iteration_interval`: 2 s no texto, 0,5 s no código

A tab. 7.1 diz **2 s**; o `benchmark.py` passa `--algorithm-iteration-interval=0.5`.

Além de o código ser a fonte que gerou os resultados, **2 s é fisicamente inconsistente com `communication_range = 20 m`**: a 15 m/s o UAV percorre 30 m entre heartbeats e atravessaria a cobertura de um sensor sem coletá-lo.

### 4.2 Treino e avaliação usam configurações diferentes

| Fase | Script | `end_when_all_collected` | `interval` |
|---|---|---|---|
| Treino | `benchmark.py` | **False** (`--no-end-when-all-collected`) | 0,5 |
| Avaliação da política | `visualize_performance.py` | **True** (default; não é passado) | 0,5 |
| Avaliação da heurística | `visualize_performance.py` | **True** | **0,1** |

No treino o episódio continua após a coleta completa — é o "longer duration positive reward signal" de Lamenza §5. Na avaliação ele termina na conclusão, e é esse instante que vira o `completion_time` da fig. 7.3.

### 4.3 A comparação da fig. 7.3 tem uma variável não controlada

No mesmo script, **o MADDPG é avaliado a 0,5 s por decisão e o guloso a 0,1 s** — cinco vezes mais reativo, com integrador de mobilidade proporcionalmente mais fino.

Medido por nós, guloso na célula 4×24, 8 sementes pareadas:

| Configuração | tempo de conclusão |
|---|---|
| guloso a `interval = 0,5` | 43,45 s |
| guloso a `interval = 0,1` | **39,13 s** |

O guloso mais reativo é **~10 % mais rápido**. O efeito favorece o baseline, então a vantagem publicada do MADDPG sobre ele é **conservadora** — mas a assimetria existe e será declarada sempre que citarmos aqueles números. Coberto por `test_avaliar_guloso_a_0_1_e_mais_rapido`.

---

## 5. O que o nosso código acrescenta, e por que não é modificação

| Acréscimo | Por quê | Toca o MDP? |
|---|---|---|
| Carregamento por caminho de arquivo | Não poluir o espaço de importação e deixar explícito que o código é externo | não |
| `EnvConfig` com os valores publicados | Evitar que os defaults de teste do `main.py` entrem por engano | não |
| `random.seed(seed)` antes de `reset()` | O `reset()` de referência declara o argumento `seed` e **o ignora**; sorteia com o `random` da stdlib. É a única forma de reproduzir cenários sem tocar no arquivo | não |
| Poda de handlers de logging | O `Simulator` acrescenta um `StreamHandler` ao logger raiz por instância e nunca o remove; sem `reset()`, cada episódio constrói um. 10 000 episódios = 10 000 handlers | não |
| Condição de parada do laço | Verbatim do `visualize_performance.py`: quebra na **primeira** `info` não vazia. Capturar a última daria o fim do episódio, não a conclusão | não |

---

## 6. Características do ambiente que a SPEC-01 precisa conhecer

Estas mudam o desenho do codificador binário e foram descobertas ao ler o código:

1. **A observação é normalizada**: `(p_k − p_i + S)/(2S)`. A entrada vive em **[0,1]** para valores válidos, não em `[-200,200]` metros.
2. **O padding é `-1`**, fora da faixa dos válidos. Logo "slot vazio" e "sensor exatamente sobre o UAV" (que normaliza para 0,5) são distinguíveis **por construção** — o bit de validade que a SPEC-01 projetava é **desnecessário**.
3. **Ressalva de faixa:** `p_k − p_i` vive em `[-2S, 2S]`, então a normalização produz `[-0,5, 1,5]`, e não `[0,1]`. O `observation_space` de referência declara `Box(-1,1)`, o que também não bate. A SPEC-01 precisa binarizar a faixa **real**.
4. O índice do agente entra como `k / Na`.

---

## 7. Pendências remanescentes

| # | Pendência | Situação |
|---|---|---|
| **P4** | Arquitetura do ator do MADDPG | `main.py` a recebe por CLI; falta extrair o default |
| **P7** | `max_episode_length` "permissivo" do §7.1 | Cenário exploratório é outro (`001-prototype`/`002-scaling`) |
| **P9** | `σ_sensor` | Parâmetro nosso, da Etapa 2 |
| **V-1** | Invariante KS contra o gerador analítico | Depende da SPEC-01. Agora compara contra o ambiente de referência diretamente |
