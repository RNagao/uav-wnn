# A rede sem peso da dissertação — do treinamento à inferência

Como a WNN de Katopodis (2022) aprende e decide neste projeto: a matemática, o
que cada símbolo significa, e onde cada equação está implementada.

Escopo: `ntupla_dissertacao` — a família que venceu a grade. São **duas redes
diferentes** (ator e crítico) dentro do **Algoritmo 4.4**.

---

## 1. A ideia central

Uma rede neural convencional guarda **pesos** e faz produtos internos. Uma rede
sem peso guarda **valores em tabelas** e faz consultas.

A entrada binária é fatiada em pedaços; cada pedaço vira um **endereço**; cada
endereço aponta para uma posição de memória. A saída é a combinação do que está
nessas posições. Não há multiplicação — há endereçamento e soma.

O preço: a entrada precisa ser binária. Daí o termômetro (§3).

---

## 2. Glossário

### Estrutura da rede

| símbolo | nome | o que é | código |
|---|---|---|---|
| `x` | entrada binária | a observação já binarizada | argumento `bits` |
| `B` | número de bits | tamanho de `x` | `self.n_bits` |
| `n` | tamanho da tupla | quantos bits cada RAM lê | `self.tamanho_tupla` |
| `N` | número de RAMs | `N = ⌈B/n⌉` — **derivado, não escolhido** | `self.n_ram` |
| `τ[i](x)` | endereço | o inteiro que os `n` bits da RAM `i` formam | `enderecos()` |
| `v[i]` | posição de memória | o valor guardado no endereço `τ[i](x)` da RAM `i` | `self.memoria[i]` |
| `d` | dimensão da saída | 3 no ator, 1 no crítico | `self.n_saidas` |

### O que o ator produz

| símbolo | o que é | dimensão |
|---|---|---|
| `f(x)` | saída do ator — os parâmetros da política | 3 |
| `θ` | rumo, extraído de `f` por `atan2(f₂, f₁)` | escalar |
| `s` | velocidade normalizada, `f₃` recortado em [0,1] | escalar |
| `z` | **ação amostrada bruta**, em R³ | 3 |
| `σ` | desvio da amostragem | escalar (0,35) |

### O que o crítico produz

| símbolo | o que é |
|---|---|
| `V(x)` | valor estimado do estado — "quanto ainda dá para ganhar daqui" |
| `c[i]` | **contador de visitas** da posição `i` |
| `φ` | fator de esquecimento (0,85) |
| `s(c)` | peso acumulado de uma posição, dado seu contador |

### Do Algoritmo 4.4

| símbolo | o que é |
|---|---|
| `r` | recompensa de um passo |
| `γ` | desconto (0,99) |
| `m` | passos do retorno (8) |
| `G` | retorno de `m` passos, com bootstrap |
| `δ` | **a surpresa**: `G − V(x)` |
| `α` | passo efetivo do ator (8e-4) |

---

## 3. Da observação aos bits — o termômetro

A observação são **27 floats** (posições relativas, normalizadas). A rede precisa
de bits.

O **termômetro quantil** divide cada dimensão em `T` níveis, e liga os bits
abaixo do valor:

```
valor 0,6 numa dimensão com T=4 e limiares [0,2 · 0,4 · 0,6 · 0,8]
      →  [1, 1, 1, 0]
```

Com `T = 32` e 27 dimensões: `B = 27 × 32 = 864 bits`.

**Os limiares são estado treinado.** Vêm dos quantis das observações reais
(`encoding/corpus.py`), viajam com o modelo e entram na conta de memória — omiti-los
inflaria a nossa vantagem.

> Codificação é a única assimetria permitida entre os braços (§4.8): MADDPG, WNN
> e MLP recebem a **mesma** observação e cada um a prepara à sua maneira.

---

## 4. O endereçamento — comum às duas redes

### A partição

`mapeamento_particionado`, `redes.py:83`:

> sorteia uma permutação de `1..B` e a fatia em blocos de `n`

Cada bit é lido **exatamente uma vez**. É o que torna `N = ⌈B/n⌉` determinado em
vez de hiperparâmetro livre. Com `B=864`, `n=12`: **N = 72 RAMs**.

### O endereço

```
τ[i](x) = Σₖ x[mapa[i][k]] · 2ᵏ
```

Cada RAM lê seus `n` bits e os interpreta como um número binário. Com `n=12`, o
endereço vai de 0 a 4095.

`redes.py:221–234`:

```python
231  estendido = np.zeros((bits.shape[0], self.n_bits + 1), dtype=np.uint64)
232  estendido[:, :self.n_bits] = bits
234  return (estendido[:, self._tuplas] * self._pot_mat).sum(axis=2)
```

Uma multiplicação matricial em vez de `N` laços — o comentário do código registra
que a versão anterior custava ~5 µs por RAM. O bit sentinela em zero (linha 231)
existe para tuplas curtas apontarem sem efeito.

---

## 5. O ator — gradiente funcional (§4.3.1)

### A saída — eq. (4.12)

```
f(x) = (1/N) · Σᵢ v[i](τ[i](x))
```

A **média** dos valores nas `N` posições endereçadas. Posição nunca escrita vale
zero e **ainda assim entra no denominador** — é o que a equação diz.

`redes.py:554–570`:

```python
566  for i, e in enumerate(linha):
567      v = self.memoria[i].get(int(e))
568      if v is not None:
569          saida[m] += v
570  return saida / self.n_ram if self.convencao == "media" else saida
```

### A atualização — eq. (4.11)

```
v[i](τ[i](x)) ← v[i](τ[i](x)) + passo,   para as N posições endereçadas
```

Soma o **mesmo** `passo` às `N` posições. As não endereçadas ficam intactas.

`redes.py:578–598`:

```python
591  for i, e in enumerate(self.enderecos(bits)[0]):
594      if v is None:
596          self.memoria[i][chave] = passo.copy()    # v₀ = 0, logo v₁ = passo
598          v += passo
```

### Por que a média importa

Somar `passo` às `N` posições muda `f` em **exatamente `passo`**, porque
`f = (1/N)Σv` e há `N` termos. Isso torna o teto de estabilidade exato:

```python
276  def _verificar(self, passo):     # asserção da §4.2
```

`c_max = 0,5` é verificado sobre o **passo efetivo** `δ·∇log π`, não sobre `α`
nominal — `α` pequeno com `δ` explodindo diverge igual.

### O que isso é, de verdade — eq. (4.16)

Desenrolando a recursão:

```
f_t(x) = Σⱼ αⱼ · Δⱼ · k(x, xⱼ),      k(x,w) = 1 − δ_Hamming(x,w)/N
```

**É uma máquina de núcleo.** A saída é a soma ponderada de todas as atualizações
passadas, com peso dado pela similaridade entre os endereços. Duas consequências:

* uma atualização em `x` muda a saída em **qualquer** ponto próximo — é a
  generalização;
* a memória cresce com o **suporte da experiência**, não com a arquitetura. É a
  raiz do custo de memória documentado em `reports/spec06-memoria-wnn.md`.

---

## 6. O crítico — regressão com esquecimento (§4.2.1)

Rede **diferente**. Guarda dois números por posição: valor e contador.

### A acumulação — eq. (4.8)

```
v ← φ·v + y        primeiro o esquecimento, DEPOIS a observação
c ← c + 1
```

`redes.py:794–820`:

```python
818      v *= self.phi          # primeiro o esquecimento...
819      v += alvo              # ...depois a observação nova
820      c += 1                 # uint32: contagem de visitas
```

A ordem é a do texto — inverter daria peso errado à observação nova.

### O peso acumulado — eq. (4.10)

```
s(c) = (1 − φᶜ)/(1 − φ)         soma dos c primeiros termos da série geométrica
```

`redes.py:736–746`. Com `φ=0,85`, `s` satura em `1/(1−φ) = 6,67`.

### A predição — eq. (4.9)

```
V(x) = Σᵢ v[i] / Σᵢ s(c[i])
```

**Média ponderada**, não média simples. `Σs = 0 → V = 0` é explícito na equação —
cobre o caso de nenhuma posição ter sido escrita.

`redes.py:748–768`.

### Por que esquecer

O alvo do crítico **muda durante o treino**: conforme o ator melhora, o retorno de
um mesmo estado sobe. Sem esquecimento, o crítico ficaria preso na média de tudo
que já viu — inclusive de quando a política era ruim.

### A diferença essencial entre as duas redes

| | ator (4.3.1) | crítico (4.2.1) |
|---|---|---|
| guarda | **só o valor** | valor **e** contador |
| combina | soma / N (média simples) | média **ponderada** por `s(c)` |
| esquece | não | sim, por `φ` |
| eq. | (4.16) soma de núcleos | (3.22) média ponderada de núcleo |

O ator ser **soma sem normalização** é o que lhe permite representar funções que o
crítico não representa.

---

## 7. A política — normal projetada

O ator produz `f ∈ R³`. Vira ação assim (`politica_continua.py:124–136`):

```python
133  theta = float(np.arctan2(z[1], z[0]))     # rumo, do círculo
134  s = float(np.clip(z[2], 0.0, 1.0))        # velocidade
```

`atan2` é liso e sobrejetivo no círculo — não há descontinuidade em `θ=0` nem
correção de fase. É a razão de `d=3` para uma ação de 2 graus de liberdade.

**Treino** amostra (`:141–151`):
```
z = f + σ·ε,     ε ~ N(0, I₃)
```

**Avaliação** usa a média, sem ruído (`:139`) — `treinando=False`.

O gradiente (`:157–170`):
```
∇_f log π(z|x) = (z − f)/σ²
```

`z` é o **bruto**, não a ação recortada. Passar a recortada é o bug que a §4.3
nomeia, e é por isso que `amostrar` devolve os dois separados.

---

## 8. O treinamento — Algoritmo 4.4

### O laço, por transição

`ac_continuo.py:530–590`:

```python
530  def act(self, obs):
531      x = self.codificar(obs.vetor)          # ① observação → bits
532      f = self.aproximador.f(x)              # ② bits → f (ator)
541      self._ultimo = (x, z)                  # guarda para fechar a transição

557  def registrar(self, recompensa):
560      self._buffer.append((x, z, float(recompensa)))     # ③ fecha com r

576  def passo_intermediario(self, obs_atual):
577      if len(self._buffer) < self.passos: return         # espera m passos
580      g = self._retorno(self.aproximador.v(x_atual))     # ④ G com bootstrap
582      self._aplicar(x, z, g)
```

### O retorno — `ac_continuo.py:562–566`

```
G = r_t + γ·r_{t+1} + … + γ^{m-1}·r_{t+m-1} + γ^m · V(x_{t+m})
```

Os primeiros `m` termos são recompensas **reais**; o último é a estimativa do
crítico — o *bootstrap*.

### O núcleo — `ac_continuo.py:568–574`

```python
570  delta = g - self.aproximador.v(x)                              # δ = G − V(x)
571  f = self.aproximador.f(x)
572  self.aproximador.ascender_ator(x, delta * self.politica.grad_log(f, z))
573  self.aproximador.regredir_critico(x, g)
```

Quatro linhas, e cada uma tem consequência:

**570 — `δ` é a surpresa.** `G` é o que aconteceu; `V(x)` é o que o crítico
esperava. A diferença é o que vale aprender.

**A ordem importa:** `δ` usa o valor **antes** de o crítico treinar (linha 573).
Inverter faria o crítico aprender a resposta e se surpreender menos com ela — o
sinal do ator encolheria sem que nada acusasse.

**572 — o passo do ator.** Combinando com `ascender_ator` (`:230–233`) e o
gradiente da política:

```
passo = α · δ · (z − f)/σ²
```

Note o `σ²` no **denominador**: σ pequeno amplifica o passo. É o mecanismo por
trás da divergência que o `c_max` detecta.

**573 — o crítico ACUMULA `G`**, não dá um passo de gradiente na direção dele. É
a média ponderada da eq. (4.9) que converte acumulação em estimativa.

### Fim de episódio — `ac_continuo.py:584–590`

Sem bootstrap: `g = 0` e o buffer é drenado de trás para frente.

### O que cada rede recebe

| rede | recebe | equação |
|---|---|---|
| ator | `α·δ·∇log π` — direção **e** magnitude da surpresa | (4.11) |
| crítico | `G` — o retorno observado | (4.8) |

---

## 9. A inferência

Muito mais simples: **sem buffer, sem δ, sem escrita**.

```python
535  if not self.treinando:
536      return self.politica.media(f)          # avaliação: sem ruído
```

O ciclo inteiro é:

```
obs → termômetro → τ[i](x) → soma das LUTs → f → atan2 → ação
```

Por decisão, medido em `custo/operacoes.py`:

| | WNN | ator MADDPG |
|---|---|---|
| multiplicações fp | **4** (do amostrador) | 72 964 |
| consultas | 216 | 0 |
| somas fp | 218 | 73 476 |

As 4 multiplicações são do amostrador Box-Muller, comum aos dois braços. **O
aproximador tem zero** — é a alegação central de H1.

---

## 10. Do treino ao deploy

```
treino  →  checkpoints/final  →  poda 70 % por |v|  →  f32/uint16
                                        ↓
                              verifica SR no banco val
                                        ↓
                    caiu ≤ 2 pp? publica : regrava sem poda
```

A poda funciona porque a contribuição de cada posição é `v/N`: com `N=72`, uma
entrada com `|v|=0,02` desloca a ação em 0,0003. E a massa é concentrada — 50 %
dela está em 6,3 % das entradas.

Detalhes em `reports/spec06-memoria-wnn.md`.

---

## 11. Resumo — cada equação e onde ela vive

| eq. | o que faz | arquivo:linha |
|---|---|---|
| τ[i](x) | endereçamento | `redes.py:221–234` |
| (4.12) | `f = (1/N)Σv` — saída do ator | `redes.py:554–570` |
| (4.11) | `v ← v + passo` — atualização do ator | `redes.py:578–598` |
| (4.8) | `v ← φv + y`, `c ← c+1` — crítico | `redes.py:794–820` |
| (4.10) | `s(c) = (1−φᶜ)/(1−φ)` — peso | `redes.py:736–746` |
| (4.9) | `V = Σv/Σs` — predição do crítico | `redes.py:748–768` |
| (4.35) | `∇log π = (z−f)/σ²` | `politica_continua.py:157–170` |
| — | `G` com bootstrap | `ac_continuo.py:562–566` |
| (4.36) | `δ = G − V(x)` | `ac_continuo.py:570` |
| Alg. 4.4 | as duas atualizações | `ac_continuo.py:568–574` |
| §4.2 | asserção `c_max` | `redes.py:276–290` |

## Referências no repositório

* Redes: `src/uavwnn/wnn/redes.py`
* Algoritmo 4.4: `src/uavwnn/agents/ac_continuo.py`
* Política: `src/uavwnn/agents/politica_continua.py`
* Codificação: `src/uavwnn/encoding/thermometer.py`
* Validação contra o Julia: `scripts/spec04_diferencial_redes.py`
* Custo de memória: `reports/spec06-memoria-wnn.md`
