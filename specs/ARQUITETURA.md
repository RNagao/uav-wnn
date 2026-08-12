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

## 3. Feature 2 — medição ruidosa e fusão

### 3.1 O modelo: o drone recebe a posição suja do sensor e a filtra

O canal **primário** é a **posição do sensor**, não o GPS do drone. A cada época de decisão o drone recebe um relatório corrompido da posição de cada sensor que enxerga, e o refina recursivamente. Tudo que ele deriva daí — inclusive **o ranking dos sensores** — passa a ser computado sobre a estimativa filtrada, nunca sobre a verdade.

**Onde esse relatório entra.** Não pode ser por rádio: no mecanismo de referência, ouvir o drone **é** ser coletado, no mesmo instante — um sensor que transmitisse sua posição já teria saído do problema, e não haveria medições a acumular. O relatório entra **reinterpretando o canal global** de Lamenza, que é justamente como o drone sabe onde os sensores estão. Declaramos que esse canal é ruidoso. Nenhuma linha do cenário é tocada.

### 3.2 Modelo de medição

```
z_{i,t}  =  p_i  +  b_i  +  v_{i,t}

b_i     ~ N(0, σ_b² I)        viés FIXO por sensor, sorteado no início do episódio
v_{i,t} ~ N(0, R_i(d_t))      ruído BRANCO por relatório
R_i(d)  = (σ_0 + κ·d)² I      a precisão PIORA com a distância
```

Três decisões registradas, cada uma com consequência:

**O ruído é branco por relatório, mais um viés fixo.** É preciso ser branco, senão não há o que filtrar — um sensor que reporta sempre o mesmo valor errado faz o filtro convergir para o valor errado. E é preciso ter viés, senão o filtro levaria o erro a zero, o que nenhum GPS real faz. O viés é o **piso irredutível**, e medi-lo é resultado.

> **Correção a uma afirmação anterior.** A SPEC-07 dizia que "o erro dos sensores é constante no episódio, não branco por passo — sortear a cada passo seria modelar errado". Sob este desenho isso está invertido: a componente branca é o que dá sentido à filtragem. O que era "modelar errado" é, na verdade, **as duas componentes juntas**.

**`R` cresce com a distância.** É isto que faz **aproximar-se ser informativo**, e portanto é o mecanismo físico que sustenta a **H4** (a política que enxerga a própria incerteza aprende busca ativa). Com `R` constante, só o tempo reduz incerteza e a H4 fica sem motivação.

**O drone conserva ruído de GPS próprio, com σ fixo.** Ver §3.5.

### 3.3 O filtro certo é de informação, não KF de movimento

O sensor é **estático**: `F = I`, `Q = 0`, sem entrada de controle. O KF degenera no filtro de informação — média ponderada recursiva pela precisão:

```
Ω_i ← Ω_i + R_i(d_t)⁻¹              (matriz de informação)
ξ_i ← ξ_i + R_i(d_t)⁻¹ z_{i,t}      (vetor de informação)
p̃_i = Ω_i⁻¹ ξ_i ,   P_i = Ω_i⁻¹
```

Mais barato e numericamente mais estável que a forma de covariância para estado constante. Com `R` constante, `σ_k = σ/√k`; com `R(d)`, medições feitas de perto pesam mais — que é exatamente o efeito procurado.

**Uma armadilha desenhada de propósito.** Se o filtro ignorar o viés `b_i`, ele converge para `p_i + b_i` **com `P → 0`** — isto é, fica arbitrariamente confiante num valor errado. O NEES vai estourar. Esta é uma predição falsificável e o melhor material que a Etapa 2 tem para a disciplina: o filtro ingênuo passa em toda inspeção visual e **só a validação de consistência o denuncia**. A correção é limitar `P` inferiormente por `σ_b²`, e a diferença entre as duas versões é resultado a reportar.

### 3.4 O ranking é computado sobre as estimativas

Consequência direta do modelo: o drone ordena e seleciona os `N*_s` sensores por `|p̂_k − p̃_i|`, com **ambas** as posições estimadas. Nunca pela verdade.

Isto não é detalhe. **Medido**, com ruído só na posição própria, 2400 decisões amostradas:

| | σ=1 m | σ=3 m | σ=5 m | σ=10 m |
|---|---|---|---|---|
| `Ns=12` — muda o sensor mais próximo | 3,3 % | 8,6 % | 14,8 % | 26,3 % |
| `Ns=12` — muda os membros do top-12 | 0 % | 0 % | 0 % | 0 % |
| `Ns=36` — muda o sensor mais próximo | 6,1 % | 17,1 % | 24,3 % | **42,9 %** |
| `Ns=36` — muda os membros do top-12 | 17,5 % | 44,3 % | 64,1 % | **83,0 %** |

Herdar o ranking verdadeiro devolveria, por outra porta, a maior parte do canal global que a Etapa 2 existe para fechar.

**Refinamento estrutural.** A **seleção** só é ambígua enquanto `n_não_visitados > N*_s = 12`. Como a célula base das Etapas 1–2 é `Ns = 12`, ali só a **ordenação** está em jogo — e ela se resolve pós-processando a saída do pai: aplicar os deslocamentos e reordenar os 12 slots pela própria norma. A recomputação completa da seleção fica restrita à Etapa 3, como função pura com teste bit-exato em `σ = 0`.

**A opção de herdar o ranking verdadeiro fica como ablação**, não descarte: a diferença entre as duas mede quanto vale o "oráculo de ranking" — isto é, quanto do resultado publicado vem de a política enxergar o estado verdadeiro.

### 3.5 O canal do GPS próprio: presente, com σ fixo, fora da varredura

O erro de posição própria **continua existindo**, com `σ_own` fixo num valor declarado e filtrado por **KF com entrada de controle** (a ação que a política acabou de tomar). Ele **não** entra na varredura experimental — a varredura é toda do canal do sensor.

Isto custa **um módulo a mais, não uma campanha maior**: a grade permanece `σ_sensor × T` = 12 configurações, a mesma que teria sem este canal. O custo é escrever o KF e validar sua consistência, trabalho da SPEC-07, que roda em paralelo.

Duas razões para mantê-lo:

1. **O achado A3 depende dele.** KF com entrada de controle dando 18,8× de redução, e o KF de velocidade constante ficando *pior* que a medida crua, é sobre a posição própria — o estado se move e o comando é conhecido. Para alvo estático não há comando a injetar.
2. **A Etapa 3 perde o objeto sem ele.** O que um drone transmite por broadcast é a estimativa da **própria** posição. Se ela for exata, o broadcast é exato, não há erro correlacionado a fundir, e a **interseção de covariância deixa de ter razão de existir** — a H5 desaparece junto.

### 3.6 Como os dois canais compõem, e por que isso limita o ganho

A observação é relativa, então o erro do que a política vê é `e_k − e_i`:

```
σ_rel  =  √( σ_own²  +  σ_sensor² )
```

Filtrar um canal só melhora a percepção até o piso do outro. Com `σ_sensor = 5 m` irredutível, levar `σ_own` de 10 m a 0,7 m move `σ_rel` de 11,2 m para 5,05 m — fator **2,2×**, não 15×. **O achado A3 é sobre a posição própria, não sobre o que a política enxerga**, e essa distinção precisa aparecer no relatório (pendência T2).

Contrapartida favorável: os dois canais têm naturezas de estimação **opostas** — estático sem controle contra móvel com controle —, e tê-los lado a lado é o contraste de livro-texto que a disciplina pede.

### 3.7 A álgebra que barateia a implementação

Toda entrada da observação tem a forma `(p_k − p_i + S)/(2S)`. Logo:

- erro na posição **própria** `e_k` ⇒ **deslocamento comum** `+e_k/(2S)` em todas as entradas válidas;
- erro na posição do **sensor** `e_i` ⇒ deslocamento `−e_i/(2S)` **só na entrada daquele sensor**;
- o padding `-1` nunca é tocado.

Verificado numericamente para o canal próprio. A escala em unidades de observação é `σ/(2S)`: com `S = 100 m`, `σ = 5 m` desloca **0,025** — comparável direto à largura do bin do termômetro. É a regra *bin ≈ σ* em unidades da entrada real.

### 3.8 Guardas desta costura

| Guarda | O que garante |
|---|---|
| `σ = 0` ⇒ observação **bit-idêntica** à do pai | a subclasse não introduz diferença por si |
| Deslocamentos medidos == previstos; padding intacto | a álgebra da §3.7 vale na implementação |
| Posições verdadeiras no `Simulator` inalteradas | o ruído não vazou para a física (RF-04) |
| Tempo de conclusão com `σ = 0` == fig. 7.3 | a corrupção desligada não muda o baseline |
| `P_i` decai como `R/k` sob `R` constante | o filtro de informação está correto |
| **NEES do filtro ingênuo estoura; do filtro com piso, não** | o viés está sendo tratado (§3.3) |

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
