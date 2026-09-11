# O MADDPG — o baseline do Lamenza

O que o baseline é, como foi treinado, e o que exatamente usamos dele.

**Ressalva que define o escopo deste documento:** o MADDPG **não é código nosso**.
Ele vem do submodule `vendor/gradys-data-collection`, e usamos apenas os **pesos
publicados** — nunca o retreinamos. Este documento descreve a implementação do
Lamenza, lida do fonte, não um MADDPG genérico de livro-texto.

---

## 1. O que a sigla significa

**M**ulti-**A**gent **D**eep **D**eterministic **P**olicy **G**radient.

Cada palavra carrega uma decisão de projeto:

| termo | o que significa aqui |
|---|---|
| **Multi-Agent** | vários drones, e o crítico vê **todos** eles (§4) |
| **Deep** | aproximador neural treinado por retropropagação — em oposição a tabular ou linear. **Não** significa convolução ou muitas camadas |
| **Deterministic** | a política devolve **uma ação**, não uma distribuição. A exploração vem de ruído somado por fora |
| **Policy Gradient** | o ator é atualizado pelo gradiente que **o crítico fornece** |

O "Deep" é a herança do DQN (2013): trocar a tabela Q por uma rede. Uma MLP de
duas camadas ocultas é deep learning por essa definição, e é o mesmo sentido do
artigo original do DDPG.

---

## 2. Glossário

| símbolo | nome | o que é |
|---|---|---|
| `o` | observação de **um** drone | 27–39 floats, dependendo de `Na` |
| `a` | ação de um drone | 2 valores (rumo, velocidade) |
| `μ(o)` | **o ator** | política determinística: `o → a` |
| `Q(O, A)` | **o crítico** | valor de estado-ação **conjunto** |
| `O` | observações de **todos** | `o₁ ‖ o₂ ‖ … ‖ o_Na` (concatenadas) |
| `A` | ações de **todos** | `a₁ ‖ a₂ ‖ … ‖ a_Na` |
| `μ'`, `Q'` | **redes-alvo** | cópias lentas de `μ` e `Q`, para estabilidade |
| `τ` | coeficiente de suavização | 0,005 — quão rápido o alvo persegue |
| `γ` | desconto | 0,99 |
| `D` | *replay buffer* | 10⁶ transições guardadas |

---

## 3. As duas redes — do fonte

`003-collaboration/main.py:154–172`:

```python
154  class Actor(nn.Module):
157      self.fc1 = nn.Linear(prod(observation_space.shape), args.actor_model_size)
158      self.fc2 = nn.Linear(args.actor_model_size, args.actor_model_size)
159      self.fc_mu = nn.Linear(args.actor_model_size, prod(action_space.shape))
168  def forward(self, x):
169      x = F.relu(self.fc1(x))
170      x = F.relu(self.fc2(x))
171      x = torch.tanh(self.fc_mu(x))
172      return x * self.action_scale + self.action_bias
```

```
o(31) ──[Linear 31→256]──ReLU──[256→256]──ReLU──[256→2]──tanh──> a
          7 936 + 256        65 536 + 256      512 + 2
                      74 502 parâmetros · 287 KB
```

`003-collaboration/main.py:139–152`:

```python
139  class Critic(nn.Module):
142      self.fc1 = nn.Linear(
143          np.array(observation_space.shape).prod() * args.num_drones
143          + np.prod(action_space.shape) * args.num_drones, args.critic_model_size)
147  def forward(self, x, a):
148      x = torch.cat([x, a], 1)          # ← observações E ações de TODOS
```

```
[O ‖ A](132) ──[512]──[512]──[1]        331 265 parâmetros · 1 294 KB
```

**O crítico é 4,4× maior que o ator.** E a entrada dele **cresce com `Na`** — é o
que o torna centralizado.

### `tanh` e os buffers de reescala

A saída passa por `tanh` (−1 a 1) e depois é reescalada:

```python
161  self.register_buffer(
162      "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, ...))
164  self.register_buffer(
165      "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, ...))
```

São **constantes do espaço de ação**, não pesos aprendidos — 4 valores, 16 bytes.
Ficam fora da contagem de parâmetros (73 474 contra 73 478), e vêm **do
checkpoint**: reconstruí-los com valores inventados daria ações fora de faixa e um
baseline silenciosamente errado.

---

## 4. CTDE — o que faz o MADDPG ser MADDPG

**C**entralized **T**raining, **D**ecentralized **E**xecution.

```
TREINO                              EXECUÇÃO
                                    
 O = [o₁, o₂, …]  ┐                  o₁ ──> μ ──> a₁
 A = [a₁, a₂, …]  ├──> Q ──> valor   o₂ ──> μ ──> a₂
                  ┘                       (a MESMA rede)
 o crítico vê TUDO                   o ator vê só o próprio
```

Durante o treino, o crítico observa o estado conjunto — resolve a
**não-estacionaridade** que atrapalha multiagente: do ponto de vista de um drone
isolado, o ambiente muda porque os outros também estão aprendendo. Com o crítico
vendo todos, o alvo é estacionário.

Na execução o crítico é **descartado**. Cada drone roda o ator sozinho, com a
própria observação.

> **É a ressalva de método que atravessa todas as nossas comparações.** O nosso
> Algoritmo 4.4 tem crítico **descentralizado** on-policy. Os atores implantados
> são comparáveis (os dois são descentralizados e compartilhados), mas os
> algoritmos de treino diferem — WNN × MADDPG é comparação **cross-método**.

---

## 5. O treinamento — do fonte

`main.py:410–444`. Quatro peças, na ordem em que rodam.

### O alvo do crítico

```python
416  next_actions = target_actor(data.next_observations).view(...)
418  qf1_next_target = qf1_target(next_state, next_actions)
419  next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
420      qf1_next_target).view(-1)
```

```
y = r + γ·(1 − done)·Q'(O', μ'(O'))
```

Note que são as redes-**alvo** (`target_actor`, `qf1_target`). Usar as redes vivas
faria o alvo perseguir a si mesmo e divergir.

### A perda do crítico

```python
424  qf1_a_values = qf1(current_state, all_actions).view(-1)
425  qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
```

Erro quadrático entre o `Q` previsto e o alvo. Regressão comum.

### A perda do ator — a peça central

```python
433  actor_actions = actor(data.observations).view(data.observations.shape[0], -1)
435  actor_loss = -qf1(current_state, actor_actions).mean()
```

```
L_ator = −Q(O, μ(O))
```

**Minimizar `−Q` é maximizar `Q`.** O ator é empurrado a produzir ações que o
crítico avalia bem. O gradiente flui *através* do crítico até os pesos do ator —
por isso "deterministic policy gradient".

É a diferença essencial para o Algoritmo 4.4: lá, `δ = G − V(x)` é um **escalar**
que escala o gradiente da log-probabilidade; aqui, o gradiente vem **de dentro** do
crítico.

### As redes-alvo

```python
442  target_param.data.lerp_(param.data, args.tau)
```

```
θ' ← (1 − τ)·θ' + τ·θ,      τ = 0,005
```

O alvo persegue a rede viva devagar — 0,5 % por atualização. É o que estabiliza o
laço.

### Atualização atrasada do ator

```python
432  if global_step % args.policy_frequency == 0:
```

O crítico atualiza a cada passo; o ator, a cada **3**. Deixa o crítico se
estabilizar antes de guiar o ator.

---

## 6. Os hiperparâmetros publicados

`main.py:23–115` — os defaults que geraram os pesos que usamos:

| parâmetro | valor |
|---|---|
| `total_timesteps` | **1 000 000** — o *default*, **não** o usado |
| `actor_learning_rate` | 3e-6 |
| `critic_learning_rate` | 3e-6 |
| `buffer_size` | 10⁶ |
| `batch_size` | 256 |
| `gamma` | 0,99 |
| `tau` | 0,005 |
| `exploration_noise` | 0,1 |
| `learning_starts` | 25 000 |
| `policy_frequency` | 3 |
| `actor_model_size` | 256 |
| `critic_model_size` | 512 |

As larguras **não são palpite nosso** — são os defaults do `main.py`, e são o que
gerou estes pesos.

> **Cuidado com o `total_timesteps`.** O default de 1 M foi sobrescrito na linha
> de comando: os tensorboard logs mostram **49 999 999 passos** nas nove células.
> Em 1 M o SR ainda era **zero**. Ver
> [`maddpg-treinamento.md`](maddpg-treinamento.md).

### A exploração

```python
277  actions[agent] += torch.normal(0, actor.action_scale * args.exploration_noise)
```

Ruído gaussiano somado à ação, **fora** da rede. É o que "determinística" implica:
a política não tem aleatoriedade própria, então a exploração vem por fora.

> Contraste com o nosso: a `NormalProjetada` amostra `z = f + σ·ε` **dentro** da
> política, e o `∇log π` depende disso. São mecanismos diferentes de explorar.

---

## 7. O que NÓS usamos

Só o **ator**, e só para inferência.

`spec02_maddpg_rollout.py:82–103` transcreve a classe e carrega os pesos:

```python
 97  rede.load_state_dict(torch.load(caminho, map_location="cpu",
                                     weights_only=True)[0])
 98  rede.eval()
100  def politica(observacao):
101      with torch.no_grad():
102          return rede(torch.as_tensor(observacao, dtype=torch.float32)).numpy()
```

O `[0]` é o índice do ator na tupla `(ator, crítico)` — é assim que o `test.py`
deles lê.

### As nove células publicadas

`Na ∈ {2,4,8} × Ns ∈ {12,24,36}`. **Não há checkpoint para `Na = 1`** — Lamenza
não treinou single-drone. É por isso que a célula 1d/12s da nossa campanha só tem
WNN, e a comparação com o baseline vive em `Na ≥ 2`.

### O que o T1 conta

`spec06_figuras_merito.py` chama `custo_mlp([d, 256, 256, 2])` — **só o ator**, do
mesmo jeito que conta só o ator da WNN. O crítico está medido (1 294 KB) mas fora
da razão de tamanho, porque não voa.

> Contar os críticos **nos favoreceria**: o do MADDPG é 4,4× o ator dele, o nosso
> é ~0,7×. A razão cairia de 6,2× para 2,1×. Não conto por isso.

---

## 8. As três diferenças que importam para a comparação

| | MADDPG | Algoritmo 4.4 (nosso) |
|---|---|---|
| **crítico** | centralizado — vê `O` e `A` de todos | descentralizado — vê só `o` |
| **sinal do ator** | gradiente **através** do crítico: `−Q(O, μ(O))` | escalar `δ = G − V(x)` escala `∇log π` |
| **exploração** | ruído somado **fora** da rede | amostragem **dentro** da política |
| **dados** | off-policy, replay de 10⁶ | on-policy, buffer de `m = 8` |
| **alvos** | redes-alvo com `τ = 0,005` | nenhuma |
| **orçamento** | **50 M** de passos (medido nos logs) | 6,75 M (o nosso) |

A linha do orçamento merece atenção, e na direção contrária à que eu supunha: o
MADDPG usou **7,4× MAIS** passos que as nossas corridas longas (50 M contra
6,75 M). Quando a WNN perde em tempo, parte da diferença é orçamento.

---

## 9. Por que uma MLP é o baseline certo

Poderia parecer que comparar contra "só uma MLP" enfraquece o argumento. É o
oposto.

A entrada são **27–39 floats** — posições relativas e distâncias. Não há estrutura
espacial para convolver nem temporal para recorrer. Uma MLP é a arquitetura
**correta** para um vetor de 31 números; uma CNN aqui seria erro de projeto.

E a alegação central do trabalho é sobre **multiplicações**:

| | ator MADDPG | ator WNN |
|---|---|---|
| mult. fp por decisão | **72 964** | **4** (do amostrador) |
| no aproximador | 72 960 | **0** |
| pico de RAM do processo (T1, medido, a2s24, 100 processos) | 506,0 ± 0,1 MB (PyTorch) | 45,9 ± 0,1 / 53,3 ± 0,2 MB (NumPy) |

Se o baseline fosse uma CNN, a comparação seria contra algo que ninguém rodaria
num microcontrolador, e o resultado perderia sentido prático. O baseline ser
pequeno e rápido é o que torna a comparação difícil — e é o que a torna válida.

---

## 10. Onde cada coisa vive

| assunto | arquivo:linha |
|---|---|
| fonte original (treino) | `vendor/gradys-data-collection/scenario/003-collaboration/main.py` |
| `Actor` | `main.py:154–172` |
| `Critic` | `main.py:139–152` |
| laço de treino | `main.py:410–444` |
| hiperparâmetros | `main.py:23–115` |
| nossa transcrição do ator | `scripts/spec02_maddpg_rollout.py:82–103` |
| tamanho e custo | `scripts/spec02_baseline_maddpg.py` |
| campanha de referência | `scripts/spec02_maddpg_campanha.py` |
| sob ruído | `scripts/spec07_maddpg_ruido.py` |
| relatório | `reports/spec02-baseline-maddpg.md` |

## 11. O que fica registrado como ressalva

**A comparação é cross-método.** CTDE contra crítico descentralizado on-policy.
Os atores implantados são comparáveis; os treinos, não. A comparação "mesmo
algoritmo" do projeto é WNN-AC × MLP-AC, com a MLP pareada rodando o **mesmo**
laço.

**Não há `Na = 1`.** A célula mais difícil da grade — um drone para todos os
sensores — não tem baseline. Ali a referência é o guloso (SR 100 %, CT ≈ 34 s).

**O orçamento difere, e a favor do baseline.** 50 M contra 6,75 M — 7,4×.
Igualar exigiria treinar a WNN por 50 M de passos (~52 h por célula à nossa
vazão), não retreinar o MADDPG. Ver [`maddpg-treinamento.md`](maddpg-treinamento.md).
