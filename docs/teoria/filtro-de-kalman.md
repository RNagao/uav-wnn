# O filtro de Kalman neste projeto

Como o filtro funciona, o que cada variável significa, onde cada etapa está
implementada, e como o parâmetro de sintonia é descoberto.

Escopo: o **degrau 4** da escada da SPEC-08 — `FiltroKalmanControle`, a proposta
do trabalho. Os outros degraus aparecem só por contraste.

---

## 1. O problema

O drone precisa saber **onde ele está**. O GPS erra ~10 m a cada leitura, de
forma aleatória. Confiar cegamente nele faria o drone acreditar que pula de um
lado para o outro a cada instante, mesmo voando reto.

O filtro combina duas fontes que sozinhas são ruins:

* **o que o GPS diz** — informação externa, ruidosa, mas sem viés acumulado;
* **o que o drone previa** — "mandei voar a 10 m/s ao norte por 0,5 s, devo estar
  5 m ao norte" — precisa a curto prazo, mas acumula erro.

---

## 2. Glossário — cada variável antes de qualquer fórmula

### O que o filtro estima

| símbolo | nome | o que é | dimensão | unidade | no código |
|---|---|---|---|---|---|
| `x̂` | estimativa do estado | onde o filtro **acha** que o drone está | vetor 2 | m | `self._x` |
| `P` | covariância do estado | o **erro típico ao quadrado** dessa estimativa — a incerteza que o filtro anuncia sobre si mesmo | 2×2 | m² | `self._P` |
| `t` | instante | tempo a que `x̂` e `P` se referem | escalar | s | `self._t` |

O estado tem **2 componentes** — só `x` e `y`. Não há velocidade nele, e isso é
deliberado (§7).

### O que entra de fora

| símbolo | nome | o que é | no código |
|---|---|---|---|
| `z` | medição | leitura crua do GPS: posição verdadeira + ruído | argumento de `atualizar` |
| `σ_gps` | desvio do GPS | quanto o GPS erra tipicamente — **10 m** no nosso regime | `sigma_gps` |
| `R` | covariância da medição | `R = σ_gps²·I`. Com σ=10 → **100 m²**. É **fixo** | `self._R` |
| `u` | comando de velocidade | a **ação que a política acabou de escolher**: `u = s·[cos θ, sin θ]` | `self._u` |
| `dt` | intervalo | tempo desde a última predição — **0,5 s** entre decisões | `t − self._t` |

### Parâmetros de sintonia

| símbolo | o que é | valor nas corridas |
|---|---|---|
| `q_w` | quanto o drone **desobedece o comando**, por segundo | 1 a 16 (do gate) |
| `σ_v0` | erro de atuação base | 0 |
| `κ` | sensibilidade ao salto de rumo | 0 |
| `Q` | incerteza **acrescentada** a cada predição | calculada |

### Intermediárias da correção

| símbolo | o que é |
|---|---|
| `y` | **inovação** — o quanto o GPS discorda da minha previsão |
| `S` | o quanto de discordância eu **esperava** |
| `K` | **ganho** — a fração do desacordo que aceito. 0 = ignoro o GPS, 1 = obedeço |
| `I` | identidade 2×2 |

---

## 3. As equações — cada uma, de onde vem e para que serve

O ciclo tem duas fases. A **predição** responde *"onde acho que estou, antes de
olhar o GPS?"* e roda uma vez por intervalo entre eventos. A **correção**
responde *"como reparto a confiança entre essa previsão e o fix que chegou?"* e
roda uma vez por medida. Cada equação abaixo existe para uma dessas perguntas —
e cada uma tem um modo de falhar específico se for trocada pela alternativa
"óbvia".

### 3.1 Inicialização

```
x̂ = z₀        a primeira estimativa É a primeira leitura de GPS
P  = R         logo, a incerteza inicial é a do GPS
```

**De onde vem.** Antes do primeiro fix o filtro não sabe nada; depois dele, a
melhor estimativa possível *é* o próprio fix. O erro dessa estimativa é então
exatamente o ruído do GPS, cuja covariância é `R` — declarar `P = R` é a única
escolha honesta.

**Por que importa.** Com essa escolha o erro inicial `e₀ = z₀ − p` é, por
construção, `N(0, R)` — logo o NEES `ε₀ = e₀ᵀP₀⁻¹e₀ ~ χ²(2)` **exatamente, desde
o passo zero**. A alternativa de livro-texto (`x̂ = 0` com `P` "grande") faz o
NEES explodir nos primeiros passos e obriga a descartar o transiente da
análise — que é justamente onde a política toma as decisões mais consequentes.
Aqui não há transiente a esconder.

### 3.2 Predição da média

```
x̂ ← x̂ + dt·u          ando na direção que mandei
```

**De onde vem.** É a forma geral do Kalman, `x̂⁻ = F·x̂ + B·u`, com `F = I` e
`B = dt·I`. `F = I` porque a posição não evolui sozinha — sem comando, o drone
paira; não há dinâmica a extrapolar só do estado. O termo `dt·u` é física pura
(dead reckoning): posição nova = posição velha + velocidade × tempo, e `u` é
constante durante o intervalo porque a política só decide nos eventos.

**Para que serve.** É o que transporta informação **através** do intervalo sem
GPS — e é o ingrediente que falta a qualquer filtro que só faz média de
medidas. Um passa-baixa perseguindo um alvo a `v` m/s carrega atraso de regime
`v·dt·(1−α)/α` (~30 m no nosso regime, uma ordem acima do próprio σ do GPS):
não é só mais ruidoso, é **enviesado**. O comando elimina esse viés de graça,
porque o agente o conhece — foi ele quem o escolheu.

**Detalhe de taxa.** Como `F = I` e `u` é constante, propagar a média em 1
passo de 2 s ou em 20 de 0,1 s dá o mesmo `x̂`. A covariância **não** tem essa
indiferença — ver §3.4.

### 3.3 Predição da covariância

```
P ← P + Q             e fico menos certo de onde estou
```

**De onde vem.** Forma geral `P⁻ = F·P·Fᵀ + Q`; com `F = I` sobra a soma.

**Por que só cresce.** Prever não acrescenta informação nenhuma — só reposiciona
a estimativa — então `P` não pode encolher aqui. E cresce por `Q` porque o
modelo "andar exatamente `dt·u`" é imperfeito: enquanto o filtro voa às cegas
entre fixes, a desobediência do voo real acumula erro que ninguém mediu ainda.

**Para que serve.** Duas funções, e as duas são vitais:

1. **Honestidade**: `P` precisa acompanhar o erro real para o NEES ter
   significado e para quem consome `x̂` saber a barra de erro.
2. **Calibrar o ganho**: o `P` que sai daqui é o numerador de `K` no próximo
   fix. `Q` pequeno demais → `P` cresce de menos → `K` pequeno → o filtro
   ignora o GPS e vira **otimista** (erra mais do que admite). `Q` grande
   demais → **conservador** (joga fora predição boa). `Q` é o botão; `P` é a
   contabilidade que ele alimenta.

### 3.4 O ruído de processo `Q`, termo a termo

```
Q(dt) = [ q_w·dt + (σ_v0² + κ·‖u − u_anterior‖²)·dt² ] · I
```

`Q` responde: *"quanto o deslocamento real desobedece `dt·u`?"* A equação
decompõe a desobediência em duas fontes **fisicamente distintas**, que se
distinguem por como escalam com `dt`:

**`q_w·dt` — desobediência sem memória (linear em `dt`).** Perturbações de
velocidade descorrelacionadas de um instante para o outro — rajada, vibração,
e no nosso caso a fonte dominante: o tranco da própria política a cada decisão
(§6). Empurrões independentes somam **em variância**, daí o crescimento linear.
`q_w` é densidade (m²/s), não variância. Esse termo é **invariante à taxa**:
20 composições de `q_w·0,1` = 1 composição de `q_w·2`.

**`(σ_v0² + κ‖Δu‖²)·dt²` — erro constante dentro do passo (quadrático em
`dt`).** Modela um erro de velocidade `δ` sorteado uma vez e mantido o passo
inteiro (atuador entregando `u + δ`): vira `δ·dt` de posição, logo variância
`∝ dt²`. Esse termo **não** é invariante à taxa — 20 passos de 0,1 s dão
`σ²·0,2`, mas 1 passo de 2 s dá `σ²·4`, fator 20 de diferença. Daí a regra dura
do módulo: **um `predizer_ate` por intervalo entre eventos, nunca um por tique
de mobilidade** — a taxa de chamada é parte da definição de `Q` (a armadilha
da T-07.2). Dentro do fator:

* `σ_v0²` é o piso — erro de atuação que existe mesmo em rumo constante;
* `κ‖Δu‖²` torna `Q` **adaptativo ao salto de comando**: logo após uma virada,
  inércia e dinâmica do controlador fazem o modelo cinemático errar mais, e o
  filtro abre a guarda exatamente aí (o `_u_ant` do código existe só para
  isso). Com `κ = 0`, `Q` volta a ser estacionário.

Nas corridas deste projeto `σ_v0 = κ = 0`: toda a desobediência medida coube no
termo linear (§6).

**O `·I`** — ruído isotrópico: o rumo comandado é uniforme, não há direção
privilegiada; mesma variância em `x` e `y`, sem correlação entre eixos.

Números: `q_w = 4`, `dt = 0,5` → **Q = 2 m²** (1,41 m acrescentados por passo);
`q_w = 16` (WNN T=16, 8 drones) → 8 m² (2,83 m).

### 3.5 Inovação

```
y = z − x̂             o GPS discorda de mim em `y` metros
```

**Para que serve.** É a **única informação nova** do ciclo: o que o GPS sabe e
a previsão não anteviu. Todo o resto da correção decide apenas que fração de
`y` aceitar.

**O subproduto de graça.** Se o modelo está certo, `y ~ N(0, S)` e a sequência
de inovações é **branca** (descorrelacionada no tempo). Inovação correlacionada
é a assinatura direta de erro não modelado — GPS não branco, inércia,
discretização — e é o diagnóstico mais informativo do arsenal de consistência,
porque acusa viés pequeno-mas-persistente que o NEES médio mascara.

### 3.6 Covariância da inovação

```
S = P + R             eu esperava discordância dessa magnitude
```

**De onde vem.** Forma geral `S = H·P·Hᵀ + R`; como o GPS mede diretamente a
posição, `H = I` e sobra a soma. A discordância esperada entre previsão e
medida é o erro de um **mais** o erro do outro, porque os dois são
independentes — variâncias de fontes independentes somam.

**Para que serve, duas vezes.** É o denominador do ganho (§3.7) — a régua que
diz se um `y` de 15 m é escândalo ou rotina — e é a escala do **NIS**
(`ν = yᵀS⁻¹y`), o teste de consistência que não precisa da verdade e por isso é
o único que existiria num voo real.

### 3.7 O ganho

```
K = P · S⁻¹           aceito a fração K do desacordo
```

**É aqui que mora a otimalidade.** A derivação escalar cabe em três linhas:
combinar duas estimativas independentes da mesma grandeza — a previsão `a`
(variância `P`) e a medida `b` (variância `R`) — como `x̂ = (1−k)·a + k·b` dá
erro quadrático médio `(1−k)²P + k²R`; derivando em `k` e zerando,

```
k* = P / (P + R)
```

— que é exatamente `K = P·S⁻¹`. Qualquer outro peso erra mais, em média.

**Comportamento nos extremos.** `P ≫ R` (muito tempo sem fix, ou logo após
virada com `κ > 0`) → `K → 1`, obedece o GPS. `P ≪ R` (previsão fresca, sensor
ruim) → `K → 0`, segura a previsão. Ninguém programou esse número: ele cai da
contabilidade que as equações 3.3 e 3.4 mantêm — é o que a tabela da §4 mostra
acontecendo passo a passo.

### 3.8 Correção da média

```
x̂ ← x̂ + K·y           ando parte do caminho até o GPS
```

Rearranjada, é uma média ponderada explícita: `x̂ = (1−K)·previsão + K·z`. A
diferença para o passa-baixa não é a fórmula — é que lá o peso `α` é um chute
fixo, e aqui `K` é recalculado a cada ciclo a partir de quem está mais incerto
naquele momento.

### 3.9 Correção da covariância — forma de Joseph

```
P ← (I−K)·P·(I−K)ᵀ + K·R·Kᵀ       e fico mais certo
```

**Por que não a forma curta `P ← (I−K)·P` dos livros?** As duas são
algebricamente iguais **apenas quando `K` é exatamente o ótimo**. A curta ainda
subtrai duas quantidades quase iguais (cancelamento catastrófico em ponto
flutuante) e pode devolver um `P` assimétrico ou com autovalor negativo — que
não é covariância de coisa nenhuma, e um NEES calculado com ele não significa
nada. A forma de Joseph é soma de dois termos positivos-semidefinidos, logo `P`
sai simétrica e positiva-definida **para qualquer `K`** — e "qualquer `K`" é
exatamente o regime das ablações de desintonia da §6, onde o ganho é subótimo
de propósito e é o NEES que precisa acusar. Custa ~30 flops a mais; barato
pelo que garante. A linha 183 ainda simetriza `P ← (P+Pᵀ)/2` contra deriva
numérica residual.

---

## 4. O ganho `K` é o árbitro, e ele se auto-ajusta

`K = P/(P+R)` compara as duas incertezas. Traço real, σ=10 m, `q_w=4`:

```
  k   P antes     K    erro real   P depois   o filtro confia em…
  1     102.0   0.505    19.75 m      50.5    meio a meio
  2      52.5   0.344    12.68 m      34.4    própria predição (34% no GPS)
  3      36.4   0.267     8.65 m      26.7    própria predição (27%)
  4      28.7   0.223     2.55 m      22.3    própria predição (22%)
  5      24.3   0.195     1.53 m      19.5    própria predição (20%)
  6      21.5   0.177     2.17 m      17.7    própria predição (18%)
```

O erro cai de 19,75 m para ~2 m **enquanto o GPS continua errando 10 m**. Ninguém
programou "confie mais na predição": o filtro descobre isso comparando `P` com `R`.

---

## 5. Onde cada etapa está implementada

Três arquivos, com papéis separados:

| arquivo | papel |
|---|---|
| `src/uavwnn/fusion/kalman.py:68–228` | **o filtro** — a matemática, sem saber de drone |
| `src/uavwnn/fusion/ambiente_ruidoso.py` | **o laço** — chama na ordem certa, por drone |
| `scripts/spec08_gate_rf05.py` | **o ajuste** — descobre `q_w` medindo o NEES |

### As variáveis nascem — `kalman.py:101–107`

```python
102  self._x = np.zeros(2)                     # x̂
103  self._P = np.eye(2) * self.sigma_gps**2   # P
104  self._R = np.eye(2) * self.sigma_gps**2   # R
106  self._u = np.zeros(2)                     # u
107  self._u_ant = np.zeros(2)                 # u anterior (para κ)
```

`P` e `R` nascem iguais porque ainda não há informação além do sensor. **`R` nunca
mais muda**; `P` muda a cada passo.

### Inicialização — `kalman.py:120–122`

```python
120  self._x = np.asarray(z0).copy()      # x̂ = z₀
121  self._P = self._R.copy()             # P = R
```

### O comando entra — `kalman.py:133–134`, chamado de `ambiente_ruidoso.py:152`

```python
133  self._u_ant = self._u.copy()
134  self._u = np.asarray(u).copy()
```

O vetor vem da ação da política, convertido em `ambiente_ruidoso.py:129`:

```python
129  self._acao[indice] = velocidade * np.array([cos(direcao), sin(direcao)])
```

É aqui que *"o drone sabe o que mandou fazer"* entra no filtro.

### `Q` — `kalman.py:139–141`

```python
139  salto = np.dot(self._u - self._u_ant, self._u - self._u_ant)   # ‖Δu‖²
140  var = self.q_w*dt + (self.sigma_v0**2 + self.kappa*salto) * dt**2
```

### Predição — `kalman.py:158–160`

```python
158  self._x = self._x + dt * self._u      # x̂ ← x̂ + dt·u
159  self._P = self._P + self._Q(dt)       # P  ← P + Q
```

Guardas nas linhas 146–157: recusa predizer sem `iniciar()`, e levanta
`ErroDeSincronia` se `dt < 0` — medida fora de ordem corromperia o estado em
silêncio.

### Correção — `kalman.py:171–183`

```python
171  self.predizer_ate(t)                  # propaga ATÉ o instante do fix
173  S = self._P + self._R
175  K = self._P @ S_inv
176  y = z - self._x
177  self._x = self._x + K @ y
180  self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T     # Joseph
183  self._P = 0.5 * (self._P + self._P.T)                   # simetriza
```

Três decisões visíveis:

* **171** — a predição está *dentro* do `atualizar`. A medida se refere ao
  instante do fix, não ao da decisão; ler `z` no instante errado é o viés
  sistemático da T-07.2.
* **180** — forma de Joseph, não `(I−K)P`. Custa ~30 flops e mantém `P`
  positiva-definida mesmo com `K` subótimo — o regime das ablações de `Q` mal
  sintonizado, onde a forma simples daria um NEES sem significado.
* **183** — simetriza contra deriva numérica; o NEES é sensível a assimetria
  residual.

### O laço — `ambiente_ruidoso.py:146–170`

Por drone, a cada decisão:

```python
146  filt = self._kf[i]                    # um filtro POR drone
149      filt.iniciar(z, t)                # 1ª vez
152      filt.comando(self._acao[i])       # registra u
157      filt.predizer_ate(t)              # prediz…
158      inov, ess = filt.inovacao(z)      # …lê a inovação ANTES de corrigir
159      filt.atualizar(z, t)              # corrige
160  x_hat = filt.media                    # x̂ vai para a OBSERVAÇÃO
```

A ordem 157–159 não é arbitrária: o **NIS** precisa da inovação depois de predizer
e antes de corrigir. O `atualizar` seguinte reprediz com `dt ≈ 0` (no-op).

E a linha 166 grava `P` em `log_estimacao` — **nunca na observação**. A verdade
também só entra no log (invariante RF-04, com teste próprio).

---

## 6. O ajuste — como `q_w` é descoberto

O filtro não sabe qual `q_w` usar. `R` vem do datasheet do sensor; `Q` depende de
**como aquela política pilota**, e isso não está em datasheet nenhum.

### A conta que mede honestidade: NEES

`consistencia.py:44–45`:

```python
44  p_inv = np.linalg.inv(covs)                                    # P⁻¹
45  return np.einsum("...i,...ij,...j->...", erros, p_inv, erros)  # ε = eᵀP⁻¹e
```

É o erro real medido **na escala da incerteza que o filtro declarou**. Se o filtro
é honesto, `ε ~ χ²(n)` com `n = 2`, logo `E[ε] = 2`.

### O veredito — `consistencia.py:128–137`

```python
128  lo, hi = faixa_chi2(M, n, alpha)   # [1,63 ; 2,41] com M=100, n=2
133      veredito = "otimista"          # nees > hi : P menor que o erro
135      veredito = "conservador"       # nees < lo : P maior que o erro
137      veredito = "consistente"
```

| veredito | significa | risco |
|---|---|---|
| **otimista** | o filtro erra mais do que admite | **perigoso** — quem consome confia demais |
| **consistente** | `P` corresponde ao erro real | ✅ |
| **conservador** | erra menos do que admite | seguro, mas desperdiça informação |

### A varredura — `spec08_gate_rf05.py:111–124`

```python
115  for q in grade:
116      r = _consistencia(cfg, degrau, episodios, politica, **{chave: q})
118      nees = r.get("nees", {}).get("medio")
120          dist = abs(nees - 2.0)
122              melhor_dist, melhor_q = dist, q     # fica o mais perto de 2
```

Para cada candidato, roda **100 episódios com a política real** e calcula o NEES
médio. A curva é monotônica — mais `Q` sempre baixa o `ε` — e o ajuste é achar
onde ela cruza o 2:

```
q_w =  0,00  →  ε = 13,27   otimista demais
q_w =  1,00  →  ε =  2,51   quase
q_w =  4,00  →  ε =  1,62   passou do ponto
q_w = 64,00  →  ε =  1,52   conservador
```

### Por que um `q_w` por modelo E por célula

O `q_w` mede a política, não o hardware. Valores medidos neste projeto:

| quem pilota | `q_w` | desvio acrescentado por passo |
|---|---|---|
| MADDPG | 1,0 | 0,71 m |
| WNN T=32 | 1,5 – 4,0 | 0,87 – 1,41 m |
| WNN T=16, 2 drones | 8,0 | 2,00 m |
| **WNN T=16, 8 drones** | **16,0** | **2,83 m** |

**Fator 16 entre extremos.** O MADDPG voa suave; a WNN voa aos trancos, e piora
com mais drones. Usar o `q_w` de um na outra faria o filtro acreditar num drone
obediente que não existe — e o experimento mediria *"o KF mal sintonizado
ajuda?"*, que não é a pergunta.

### Onde o resultado é consumido

`spec08_escada.py`, `_params_filtro`:

```python
if degrau == 4:
    return {"q_w": float(gate["degrau4"]["melhor"])}
```

O `q_w` sai do JSON do atestado e entra no construtor. **O gate mede, o JSON
guarda, a escada usa** — e a escada recusa rodar sem atestado aprovado
(bloqueio B1).

---

## 7. Por que só 2 estados

Um KF de rastreamento típico teria 4 — `[x, y, vx, vy]` — e estimaria a
velocidade a partir das posições. É o que o **degrau 3**
(`FiltroVelocidadeConstante`) faz.

Aqui não precisa: **a velocidade é conhecida**, é o `u` que a política acabou de
comandar. Por isso ela entra como *entrada de controle* na predição em vez de
virar estado a estimar.

É a diferença central entre os dois degraus, e a razão de o degrau 4 ser a
proposta: o de velocidade constante trata cada virada de rumo como surpresa e
fica permanentemente atrasado; o degrau 4 já sabe da virada antes de ela aparecer
na medida.

---

## 8. O ciclo inteiro

| etapa | código | o que faz |
|---|---|---|
| nascem as variáveis | `kalman.py:101–107` | `x̂`, `P`, `R`, `u` |
| 1ª estimativa | `kalman.py:120–122` | `x̂ = z₀`, `P = R` |
| registra o comando | `kalman.py:133` ← `ambiente:152` | `u` |
| calcula `Q` | `kalman.py:139–141` | `q_w·dt` |
| **prediz** | `kalman.py:158–159` | `x̂+dt·u`, `P+Q` |
| **corrige** | `kalman.py:173–183` | `S`, `K`, `y`, Joseph |
| loga (não observa) | `ambiente:166–170` | `P`, inovação, erro |
| **mede a honestidade** | `consistencia.py:44` | `ε = eᵀP⁻¹e` |
| **escolhe `q_w`** | `gate:115–122` | o `ε` mais perto de 2 |
| **usa** | `escada.py` | `q_w` → construtor |

---

## Referências no repositório

* Implementação: `src/uavwnn/fusion/kalman.py`
* Costura com o ambiente: `src/uavwnn/fusion/ambiente_ruidoso.py`
* Consistência (NEES/NIS): `src/uavwnn/fusion/consistencia.py`
* Atestado: `scripts/spec08_gate_rf05.py`
* Experimento: `scripts/spec08_escada.py`, `reports/spec08-etapa2.md`
