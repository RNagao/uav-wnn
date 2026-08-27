# A rede sem peso do anbubus — a segunda família

A `NTNModel` do repositório `anbubus/ntn_neurocomputing_v0`, portada para este
projeto como `ntupla_anbubus`. Existe para que a comparação contra o MADDPG tenha
**duas** opções de rede sem peso, e não uma.

Este documento pressupõe [`wnn-dissertacao.md`](wnn-dissertacao.md) — o
endereçamento por tuplas e o vocabulário são os mesmos. Aqui está **o que
difere**, e por quê.

> **Estado do braço:** abandonado. 153 corridas de PPO, SR ≈ 0 %. O resultado
> negativo é reportável e está em `reports/spec06-grid-search.md`; a rede fica
> documentada porque a comparação depende dela ter sido tentada de verdade.

---

## 1. A origem

O código de referência está em `vendor/ntn_neurocomputing_v0/wnn/ntn_model.py`.
São ~120 linhas de PyTorch. A rede inteira é isto:

```python
self.policy_mem = nn.EmbeddingBag(
    num_embeddings=self.n_nodes * 2 ** tuple_size,
    embedding_dim=self.policy_output_dim[0],
    mode='sum'
)
nn.init.zeros_(self.policy_mem.weight)
```

Uma `EmbeddingBag` **é** uma rede n-tuple: uma tabela de consulta, indexada por
inteiros, que soma as linhas selecionadas. O `mode='sum'` é a convenção; a
inicialização em zero é a condição `v₀ = 0`.

---

## 2. Glossário — o que muda em relação à dissertação

| símbolo | o que é | anbubus | dissertação |
|---|---|---|---|
| `res` | resolução do termômetro | `encoding["resolution"]`, 512 padrão | `T`, 16 ou 32 |
| `input_dim` | bits de entrada | `prod(obs_space) · res` | `B = D·T` |
| `tuple_size` | bits por RAM | `8` nas configs deles | `n`, 12 |
| `n_nodes` | RAMs | `⌈input_dim/tuple_size⌉` | `N = ⌈B/n⌉` |
| `policy_mem` | memória do ator | `EmbeddingBag` densa | dict esparso |
| `vf_mem` | memória do crítico | **outra `EmbeddingBag`, mesma classe** | rede **diferente** |
| `mapping` | permutação dos bits | `torch.randperm` | `mapeamento_particionado` |

A estrutura é a mesma. As três diferenças que importam estão abaixo.

---

## 3. Diferença 1 — o crítico é da mesma família do ator

**No anbubus:** `policy_mem` e `vf_mem` são as duas `EmbeddingBag`, idênticas
exceto pela dimensão de saída. As duas treinam por autograd.

**Na dissertação:** o ator é gradiente funcional (§4.3.1) e o crítico é regressão
com esquecimento (§4.2.1) — **modelos diferentes**, com equações diferentes.

### Por que a divergência

É o algoritmo que decide. O **PPO** treina ator e crítico por gradiente, então um
modelo só serve para os dois. O **Algoritmo 4.4** acumula retorno no crítico pela
eq. (4.8) — `v ← φv + G` — que não é passo de gradiente.

### Como isso aparece na nossa aritmética

`ac_continuo.py:408–417` — o crítico do anbubus faz **descida de gradiente**:

```python
416  erro = self.v(x) - float(alvo)
417  self.critico.atualizar(np.atleast_2d(x), np.array([-self.taxa_critico * erro]))
```

Contra o da dissertação (`ac_continuo.py:235–238`), que **acumula**:

```python
238  self.critico.treinar(np.atleast_2d(x), np.array([float(alvo)]))
```

Um tem taxa de aprendizado; o outro não tem — tem fator de esquecimento.

> Esta era a pendência **C2** do projeto: descobrir se ator e crítico eram o mesmo
> modelo. O código do Katopodis (`RLWorktable.jl` + `Ramnet.jl`) resolveu — são
> diferentes. O do anbubus mostra a alternativa.

---

## 4. Diferença 2 — convenção de soma

**No anbubus:** `mode='sum'`. A saída é `f(x) = Σᵢ v[i]`, sem dividir.

**Na dissertação:** eq. (4.12), `f(x) = (1/N)·Σᵢ v[i]` — média.

### A consequência que morde

Somar `passo` às `N` posições muda `f` em:

| convenção | Δf |
|---|---|
| média | `passo` |
| **soma** | `passo · N` |

Com `N = 100`, a **mesma** taxa produz um passo 100× maior. **As taxas deles não
são transferíveis** para a nossa convenção sem dividir por `N`.

`redes.py:237–245` expõe isso como propriedade:

```python
237  def ganho_do_passo(self) -> float:
245      return 1.0 if getattr(self, "convencao", "media") == "media" else self.n_ram
```

E `redes.py:247–260` faz a conversão explícita:

```python
247  def taxa_equivalente(self, taxa: float, para: str) -> float:
260      return float(taxa * self.n_ram) if para == "media" else float(taxa / self.n_ram)
```

### Por que isso não é detalhe

O teto de estabilidade `c_max` é sobre a **mudança em `f`**, não sobre o número
escrito na memória. Sem o `ganho_do_passo`, `c_max = 0,5` significaria coisas
diferentes nas duas famílias, e a comparação seria entre réguas distintas.

> **Erro real que isto causou:** numa campanha, o `ap_taxa` convertia com `n=8`
> fixo enquanto o treinador rodava `n=12`. A taxa saiu **1,49× alta**, e o braço
> anbubus ficou instável por motivo de conversão, não de método. Registrado em
> `reports/spec06-grid-search.md`.

---

## 5. Diferença 3 — alocação densa

**No anbubus:**

```python
num_embeddings = self.n_nodes * 2 ** tuple_size
```

Toda posição possível é alocada na inicialização, escrita ou não.

**Na dissertação (nossa implementação):** dicionário por RAM — só o que foi
tocado ocupa memória.

### O teto duro que isso impõe

| `tuple_size` | posições por RAM | com `n_nodes = 100` |
|---|---|---|
| 8 | 256 | 25 600 |
| 12 | 4 096 | 409 600 |
| 24 | 16 777 216 | 1,68 · 10⁹ |
| 32 | 4,29 · 10⁹ | **4,29 · 10¹¹ — inviável** |

**É por isso que as configurações deles usam `tuple_size = 8`.** Não é escolha de
qualidade — é o que a alocação densa permite.

Nós medimos os dois regimes em vez de assumir: `variante_lut="denso"` reproduz o
custo deles no harness da SPEC-03 (`ac_continuo.py:382–385`).

> **Isto nos custou 18 processos mortos.** Na grade do PPO, `n = 24` pedia
> 6,7–13,4 GB por corrida e o sistema matava tudo. A fila ganhou uma guarda:
> `if ((25*T)//n)*(2**n)*4*4/1e9 > 4.0: continue`.

---

## 6. O que NÃO importamos — e é decisão declarada

O `NTNModel` emite `num_outputs`, que no PPO contínuo do RLlib é
`2 · dim(ação)`: média **e `log σ`**. Isto é, **variância aprendida**. O trecho
comentado deles confirma:

```python
# nn.init.constant_(self.policy_mem.weight.chunk(2, dim=1)[1], math.log(.5))
```

Nós mantemos **σ fixa** (0,35), pelas razões da §4.4, e a política é a mesma nas
duas famílias **de propósito**: trocar aproximador e política ao mesmo tempo
mediria duas coisas de uma vez.

A variância aprendida fica registrada como a diferença remanescente, **não
importada**.

---

## 7. O endereçamento deles — `conv1d` como extrator de tuplas

O truque mais bonito do código deles:

```python
def _keys(self, x):
    return F.conv1d(
        x[:, self.mapping].unsqueeze(1),
        self.key_weights,
        stride=self.tuple_size
    ).squeeze(1)
```

Com `key_weights = [2⁰, 2¹, …, 2ⁿ⁻¹]` e `stride = tuple_size`, a convolução
**é** a conversão binário→inteiro: cada janela sem sobreposição vira um endereço.

Depois:

```python
keys += self.mem_offsets      # mem_offsets = (2^n)·[0, 1, …, n_nodes−1]
return mem(keys)
```

As `N` RAMs vivem numa tabela **única**, empilhadas; o offset desloca cada uma
para o seu bloco. É a mesma coisa que `N` dicionários separados, com layout
diferente.

### A restrição escondida

`stride=tuple_size` sem padding exige que **`n` divida `input_dim`**. Se não
dividir, o `conv1d` descarta os bits do fim — silenciosamente.

> **Isto quebrou uma grade nossa.** `n = 24` com `input_dim = 25·T` só funciona
> quando `24 | 25·T`. A fila ganhou `if (25*T) % n: continue`.

---

## 8. O codificador deles

`vendor/…/wnn/ntn_encodings.py`:

```python
enc = torch.arange(self.resolution).expand(*X.shape, self.resolution)
threshold = (X - self.minimum)/(self.maximum - self.minimum)*self.resolution
return (enc < threshold).to(torch.long)
```

Termômetro **de faixa fixa** — `minimum` e `maximum` são parâmetros, não quantis.
A dissertação usa **quantis** dos dados reais.

### A faixa, e o erro que ela causou

As oito configurações publicadas deles usam:

```
observation_filter = "MeanStdFilter"     em TODAS
encoding range     = (-1.5, 1.5)         em TODAS
```

O filtro **padroniza** a observação (média 0, desvio 1). Então a faixa `(-1,5;
1,5)` está em **desvios-padrão**, não em unidades do ambiente — cobre ~87 % de
uma normal.

> **Eu quebrei isto e medi o estrago.** Ao adicionar o `MeanStdFilter`, mudei a
> faixa para `(-1,0; 1,5)` sem perceber que invalidava a própria justificativa.
> Medido: nossa faixa saturava **24,1 %** das dimensões contra **12,4 %** da
> original. Corrigido para `(-1.5, 1.5)` em `spec06_ppo_anbubus.py:156`.

---

## 9. Resumo — as duas famílias lado a lado

| | dissertação | anbubus |
|---|---|---|
| **ator** | gradiente funcional §4.3.1 | `EmbeddingBag` + autograd |
| **crítico** | regressão c/ esquecimento §4.2.1 | `EmbeddingBag` — mesma família |
| **crítico guarda** | valor **e** contador | só valor |
| **crítico atualiza** | `v ← φv + G` (acumula) | `Δ = −η(v − alvo)` (gradiente) |
| **convenção** | média `(1/N)Σv` | **soma** `Σv` |
| **Δf por passo** | `passo` | `passo · N` |
| **memória** | dicionário esparso | densa, `n_nodes · 2ⁿ` |
| **teto de `n`** | nenhum | ~8, pela alocação |
| **codificador** | termômetro por **quantis** | termômetro de faixa **fixa** |
| **σ da política** | fixa (0,35) | aprendida (não importada) |
| **algoritmo** | Algoritmo 4.4 | PPO |

---

## 10. Onde cada coisa vive

| assunto | arquivo:linha |
|---|---|
| fonte original | `vendor/ntn_neurocomputing_v0/wnn/ntn_model.py` |
| codificador original | `vendor/ntn_neurocomputing_v0/wnn/ntn_encodings.py` |
| nosso port (Alg. 4.4) | `ac_continuo.py:327–420` |
| crítico por gradiente | `ac_continuo.py:408–417` |
| conversão de convenção | `redes.py:237–260` |
| treino com PPO real | `scripts/spec06_ppo_anbubus.py` |
| grade do PPO | `scripts/spec06_grid_ppo.py` |
| invariantes de arquitetura | `tests/test_anbubus.py` |

## 11. O veredito

O braço PPO rodou **153 corridas / 51 pontos**. O melhor ponto chegou a 44,6 % de
cobertura com **SR 0,0 %**; o melhor SR de qualquer ponto foi **3,3 %**; 49 das
153 falharam.

Duas leituras honestas:

* **A rede não é o problema.** Ela é uma n-tuple legítima, e o artigo dela reporta
  resultados em outras tarefas. O que não funcionou foi a combinação
  *rede + PPO + este problema*.
* **O braço não foi descartado por conveniência.** Ele foi tentado com as oito
  configurações publicadas respeitadas (`MeanStdFilter`, `num_sgd_iter=1`, faixa
  em desvios, 1 M de passos), e três defeitos nossos foram corrigidos no caminho
  — a faixa do codificador, a conversão de taxa e a guarda de memória. O
  resultado negativo é do método, não da implementação.
