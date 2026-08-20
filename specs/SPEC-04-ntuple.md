# SPEC-04 — Núcleo n-tuple esparso

**Etapa:** 1 · **Depende de:** SPEC-01 (codificador e `(n, N)` pré-selecionados) · **Estimativa:** 2 dias
**Portão de saída:** nenhum próprio — alimenta `G1` através da SPEC-05 · **Referência:** PRD §5.3 (o aproximador), §5.5 (custo), RNF-01, RNF-02, RNF-08, RNF-09, RNF-10, RNF-12

---


## 0-quater. A fundamentação teórica passou a ser testada (2026-08-15)

A leitura da **dissertação** (fonte primária, `tmp/947841_compressed.pdf`)
fechou a auditoria do capítulo 3 e do 4. O que era analogia agora é identidade
verificada, em `tests/test_identidades_artigo.py`:

| eq. | o que prova |
|---|---|
| (3.22) | a rede de regressão **é** um estimador de Nadaraya-Watson |
| (4.16) | a rede de gradiente funcional **é** máquina de núcleo |
| (4.19) | a atualização **é** um passo de gradiente funcional: `Δf(x) = αΔ·k(x_t,x)` |
| (4.22)/(4.23) | mapa de características explícito e `⟨φ(w),φ(x)⟩ = k(w,x)` → núcleo **positivo-definido** |

A (4.23) é a que sustenta as outras: o texto assinala que a passagem de (4.18)
para (4.19) *"hinges on the kernel being positive-definite"*. Verificada no
espectro — Gram 60×60, autovalor mínimo `> −1e−9`.

**Consequência para a RNF-01 do PRD, que estava errada.** Ela prescrevia
`η = c/N` supondo `‖φ‖² = N`. A eq. (4.22) normaliza por `1/√N`, então
`‖φ‖² = 1`, e com a média da (4.12) o passo efetivo é **exatamente `η`,
independente de `N`**. A regra foi retirada. O que de fato limita `α` é a
variância do gradiente de política — medido, e monótono em `α`.

**Novo método:** `treinar_componente(bits, indice, alvo)`, que o Algoritmo 4.1
exige (`v[i](S_code, A)` atualiza só o discriminador da ação). É o que justifica
o **contador por componente** — um contador compartilhado degradaria a média
ponderada de (4.9) para as ações menos visitadas.

---

## 0-bis. Implementado — e validado contra o Ramnet.jl (2026-08-15)

As duas redes estão em `src/uavwnn/wnn/redes.py`, com **armazenamento esparso
por dicionário**. A motivação não é elegância: o apêndice A da dissertação usa
tuplas de **16, 32 e 64 bits**, e `2^64` posições por RAM não cabem em array
denso. O `NTupleRegressor` de `ntuple.py` segue servindo a SPEC-01 e a SPEC-03,
preso a `n <= 16`.

Mapeamento por **partição** (`randperm(B)` fatiado), como o `:uniform_random` do
Ramnet: cada bit lido exatamente uma vez, logo **C4 fica fechada pela
estrutura**. Quando `B` não divide `n` a última tupla é mais curta — fiel ao
`Iterators.partition` do Julia, e exatamente o caso que quebra o `NTNModel` em
PyTorch (`ceil` para os nós, `floor` na `conv1d`).

Teste diferencial contra `Ramnet.jl@faster_hashing`: **desvio zero** em 5
configurações, cobrindo os dois papéis e `phi` em {0,50; 0,85; 1,00}. Fixtures
congeladas em `tests/fixtures/diferencial_redes/`.

Relatório: `reports/spec04-redes-dissertacao.md`.

**Pendência nova, e é a que mais importa aqui:** a SPEC-01 escolheu `n = 12` e a
SPEC-03 achou `n = 10` dominante, mas as duas rodaram sobre o armazenamento
denso, que torna `n = 32` caro e `n = 64` impossível. Essa escolha pode ser
consequência do armazenamento, não do problema. Revarrer com
`n` em {8, 12, 16, 24, 32} antes de fixar a Etapa 1.

---

## 0. Revisão — o código do Katopodis resolve a C2, e ela estava mal formulada

A implementação de referência é do **próprio autor**: `rkatopodis/RLWorktable.jl` (agentes) e `rkatopodis/Ramnet.jl` (núcleo). Atenção: os `AltDiscriminators` estão na branch **`faster_hashing`**, não na `master`. Ambos MIT.

**A pendência C2 partia de uma premissa errada.** O PRD pedia "gradiente funcional **com** fator de esquecimento" — combinação que **não existe** no código dele. São **dois modelos distintos, para papéis distintos**:

| papel | modelo | `predict` | `train!` | guarda |
|---|---|---|---|---|
| **ator** | `FunctionalDiscriminator` | `Σ vᵢ / N` — média simples | `v -= grad` | só **valor** |
| **crítico** | `RegressionDiscriminator` | `Σ vᵢ / Σ (1−γ^cᵢ)/(1−γ)` | acumula valor e contador | **valor + contador** |

O denominador do crítico é a soma da série geométrica dos pesos de esquecimento — é a eq. 4.10 da dissertação, implementada literalmente.

### Três consequências para esta spec

**1. Implementar dois modelos, não um.** O escopo muda de "um núcleo com esquecimento opcional" para dois tipos com interfaces próximas mas semânticas diferentes. O `NTupleRegressor` que a SPEC-01 já escreveu (média sobre RAMs, `v -= η(f−y)`, sem contador) **já é o `FunctionalDiscriminator`** — o ator está pronto em forma mínima.

**2. O crítico custa o dobro por entrada.** Valor **e** contador por posição endereçada. A medição de T1 da SPEC-03 mediu só o ator; a coluna *ator + crítico* precisa desse fator, e o medidor já lê `bytes_por_entrada` do modelo em vez de assumir.

**3. O esquecimento não é ajuste do ator.** Ele existe porque o crítico faz **regressão para o retorno `G`**, cujo alvo se desloca conforme a política melhora. No ator não há distribuição-alvo se movendo dessa forma, e por isso não há γ.

### Ainda em aberto

**C3 — estrutura de armazenamento.** A SPEC-03 já mediu, na configuração da SPEC-01: chaves ordenadas **133,9 KB** contra hash aberto **273,6 KB**. A ordenada ganha por 2× em memória; falta medir a cauda de latência (T2) para decidir, já que o argumento a favor do hash era previsibilidade.

**A ocupação não satura** (elasticidade 0,194 contra o critério de 0,05 do RNF-08). A poda por esquecimento deixa de ser refinamento e vira requisito: medir `occ_ε` — entradas com `|v| > ε` — junto com `occ`.

---

## 1. Objetivo

Entregar a biblioteca do aproximador — a rede n-tuple com gradiente funcional, esquecimento, armazenamento esparso e múltiplas cabeças de saída — **validada fora do laço de RL**, com contratos de API, layout de memória e instrumentação de custo já no formato que a SPEC-03 consome.

Esta spec é **endurecimento e extensão de um protótipo existente**, não escrita do zero. O `ntuple.py` marcado como PRONTO no PRD já implementa gradiente funcional e esquecimento e já produziu os três achados medidos (estabilidade, retorno decrescente em `N`, ocupação de 1,2 %). O trabalho aqui é: auditar o que existe, corrigir o que a auditoria encontrar, e acrescentar o que falta — esparsidade real, cabeças múltiplas, quantização, instrumentação e testes.

## 2. Por que fora do RL, e por que antes da SPEC-05

Três razões, em ordem de peso:

1. **Um núcleo com erro sutil é indistinguível de "o RL não converge".** Convenção de sinal do gradiente invertida, endereçamento que ignora metade dos bits de entrada, passo efetivo escalado por `N` sem que ninguém perceba — todos produzem exatamente o mesmo sintoma: curva de aprendizado plana. Diagnosticar dentro do laço de treino custa dias; contra uma regressão sintética com solução conhecida, custa minutos.
2. **O custo é medido aqui, não no experimento.** Tamanho realizado, ocupação, contagem de operações e latência são figuras de mérito de primeira classe (PRD D11). São propriedades da estrutura de dados, não da política. Medi-las com o núcleo isolado dá números limpos e reprodutíveis; medi-las com o simulador rodando por volta mistura tudo.
3. **A matemática de Katopodis precisa de adaptação, e a adaptação precisa de teste.** O modelo do PRD (`f = ⟨W, φ⟩`, soma pura) e o de Katopodis (`f = (1/N) Σ v`, eq. 4.12) não são o mesmo objeto até que se fixe a relação entre as taxas de aprendizado. Pior: a rede com esquecimento (eq. 4.8–4.10) e a rede de gradiente funcional (eq. 4.11–4.12) são **duas arquiteturas diferentes** no texto original, e o PRD pede as duas coisas no mesmo módulo. A §4 desta spec resolve isso explicitamente; se ficar implícito, vira bug.

## 3. Escopo

**Dentro:** sorteio e armazenamento do mapeamento de tuplas; cálculo de endereços; núcleo de gradiente funcional com perda plugável; as variantes de esquecimento; guarda de estabilidade `η = c/N`; backends de armazenamento (hash esparso, denso, hashing com colisão); cabeças múltiplas; quantização fp32/int16; serialização; regressão sintética de validação; suíte de testes de propriedade; instrumentação de ocupação e de contagem de operações.

**Fora:** qualquer algoritmo de RL — perda de TD, gaussiana de política, vantagem, laço de episódios são da **SPEC-05**. O codificador binário é da **SPEC-01** (aqui ele entra só como `bits: np.ndarray` de entrada). O protocolo de medição de latência e de tabela de custo é da **SPEC-03**; esta spec entrega os *contadores*, não o relatório. Nenhuma decisão de hiperparâmetro de aprendizado (`T`, `n`, `N`, `c`, `φ`) é fechada aqui — a varredura é da SPEC-06.

---

## 4. O modelo de referência, e onde Katopodis precisa ser adaptado

Esta seção é normativa. Ela existe porque três discrepâncias entre o PRD e o texto de Katopodis são pequenas na aparência e determinantes na prática.

### 4.1 Definição

Entrada binária `x ∈ {0,1}^B` (saída do codificador da SPEC-01). Um mapeamento de tuplas fixo `M ∈ [0,B)^{N×n}` define, para cada RAM `i`, o endereço

```
a_i(x) = Σ_{j=0}^{n-1} x[ M[i,j] ] · 2^j
```

`φ(x)` é a concatenação dos one-hots por RAM: vetor de dimensão `N·2^n` com exatamente `N` bits ligados. Para cada cabeça `k`:

```
f_k(x) = ⟨W_k, φ(x)⟩ = Σ_{i=1}^{N} W[k, i, a_i(x)]
```

Isto é **tile coding com ladrilhamento aleatório**: cada RAM é um ladrilhamento, cada endereço é um ladrilho, e o modelo é linear nos parâmetros. É de onde vêm as garantias de convergência de TD com aproximação linear, e é o argumento da fundamentação do relatório (PRD §5.3).

### 4.2 Discrepância 1 — a normalização `1/N` e o passo efetivo

Katopodis (eq. 4.12) define `f(x) = (1/N) Σ_i v[i](x)`. O PRD define a soma sem normalizar. **Os dois modelos são idênticos a menos de uma reparametrização da taxa de aprendizado**, e é aí que mora o achado de estabilidade.

Com a atualização de gradiente funcional aplicada às `N` entradas endereçadas:

| Forma | Saída | Efeito de um passo sobre `f` |
|---|---|---|
| Katopodis (normalizada) | `f = (1/N) Σ v_i` | `Δf = −η · ∂L/∂f` |
| PRD (soma pura) | `f = Σ W_i` | `Δf = −η · N · ∂L/∂f` |

**Decisão normativa.** Mantemos a forma do PRD (soma pura, que é a leitura linear `⟨W, φ⟩`) e parametrizamos a taxa como

```
η = c / N
```

de modo que `Δf = −c · ∂L/∂f` nas duas formas. **`c` é a taxa de aprendizado no espaço de funções** e é o hiperparâmetro que aparece em todas as configurações e em todos os relatórios. `η` é derivada e nunca é exposta na configuração.

Consequência de projeto, e é obrigatória: **a varredura de `N` da SPEC-06 varre `c`, não `η`.** Varrer `η` com `N` variando confunde capacidade com passo, e a comparação `N = 16 / 48 / 128` deixa de significar o que se pensa que significa.

### 4.3 Discrepância 2 — a fronteira de estabilidade, e o NaN medido

Para a perda quadrática `L = ½(y − f)²`, um passo sobre a mesma amostra dá `f' = f + c·(y − f)`. A recorrência converge se, e somente se, `|1 − c| < 1`, isto é `0 < c < 2`. Este é o limite de LMS clássico, `η‖φ‖² < 2`, com `‖φ(x)‖² = N`.

O achado do protótipo bate exatamente: `η = 0,05` com `N = 48` dá `c = η·N = 2,4 > 2` — divergência para NaN não é acidente numérico, é a recorrência instável. A regra `c ≲ 0,5` é uma margem de 4× sobre a fronteira teórica, e existe porque em RL o alvo `y` também se move.

**Esta análise sobrevive à correção da RNF-01 — só fica mais simples.** A fronteira LMS `0 < c < 2` vale nas duas convenções; o que muda é a relação entre `c` e a taxa exposta. Na convenção de **soma**, `c = η·N`. Na de **média**, que é a do artigo (eq. 4.12) e a que adotamos, `‖φ‖² = 1` e portanto **`c = η` diretamente**: a regra é `η ≲ 0,5`, sem divisão por `N`. O NaN medido continua explicado, como artefato da convenção de soma do protótipo.

**E há um segundo limite, muito mais apertado, que não é este.** A fronteira LMS governa a **regressão do crítico**. A taxa do **ator** é limitada pela *variância do gradiente de política*, e isso foi medido: no bandido contextual o pior erro entre 5 sementes cresce monotonicamente com `α`, com divergência já em `α = 0,05` — **dez vezes abaixo** da fronteira LMS. Confundir os dois faria a busca de `α` da SPEC-06 começar uma ordem de grandeza alto demais. É a instabilidade de passo fixo que a dissertação nomeia no capítulo 6.

**RNF-01 codificado.** Asserção em tempo de construção **e** em tempo de execução (ver T-04.5). O valor divergente medido vira caso de teste de regressão.

### 4.4 Discrepância 3 — a mais séria: esquecimento e gradiente funcional são duas redes diferentes

Em Katopodis, §4.2.1 e §4.3.1 propõem **arquiteturas distintas**:

- **Rede com esquecimento (NTRN-φ)** — cada posição guarda `(v, c)`; atualização `v ← φ·v + y`, `c ← c + 1` (eq. 4.8); saída `f = Σ v_i / Σ s_i` (eq. 4.9), com `s = (1 − φ^c)/(1 − φ)` (eq. 4.10), a soma da série geométrica dos pesos. É uma **média ponderada auto-normalizada** e minimiza implicitamente o erro quadrático.
- **Rede de gradiente funcional** — cada posição guarda só `v`; atualização `v ← v − η·∂L/∂f` (eq. 4.11, ver ressalva de sinal em T-04.1); saída `f = (1/N) Σ v_i` (eq. 4.12). Aceita **qualquer** perda diferenciável em `f`, e é isso que habilita erro TD e policy gradient.

O Actor-Critic de Katopodis (§4.3.4, alg. 4.4) usa **as duas ao mesmo tempo**: NTRN-φ no crítico, gradiente funcional no ator. O PRD, por outro lado, descreve um único módulo "gradiente funcional + esquecimento". Isso precisa ser conciliado, e a conciliação é quantitativa:

**Resultado que reconcilia as duas.** Para uma única posição de memória da NTRN-φ, com `v' = φv + y` e `s' = φs + 1`, a estimativa por posição `f_i = v_i/s_i` satisfaz

```
f_i' = (φ·s·f_i + y) / (φ·s + 1) = f_i + (y − f_i) / s'
```

isto é, **a NTRN-φ é, por posição, exatamente uma EWMA com passo adaptativo `α_t = 1/s_t`**, que decai de `1` (primeira visita) até o regime permanente `α_∞ = 1 − φ`. Ou seja: a rede com esquecimento é o mesmo gradiente funcional com perda quadrática, sob um **cronograma de passo com correção de viés**. Daí sai a correspondência de hiperparâmetros

```
c  ↔  1 − φ        (φ = 0,99 ⇔ c = 0,01;  φ = 0,9 ⇔ c = 0,1)
```

que deve ser usada para não varrer duas vezes a mesma grandeza.

**O que a equivalência NÃO cobre — e é o risco sutil desta spec.** A saída agregada difere:

```
NTRN-φ:              f = Σ v_i / Σ s_i  =  Σ s_i f_i / Σ s_i     (média ponderada pela confiança)
gradiente funcional: f = (1/N) Σ v_i                              (média uniforme sobre as N RAMs)
```

As duas coincidem quando todas as `N` posições endereçadas têm o mesmo contador. Diferem quando não têm — e em RL nunca têm. A NTRN-φ é **auto-normalizada sobre as posições endereçadas**: uma posição fria entra com peso ~0 no numerador e no denominador, e é simplesmente ignorada. O gradiente funcional não normaliza: uma posição fria contribui com `0` para a soma e **encolhe `f` em direção a zero**.

Consequência concreta, específica deste projeto: em região pouco visitada, o ator com gradiente funcional produz média de ação próxima de zero. Com a ação de Lamenza normalizada para `[0,1]²`, zero é o **canto** do espaço de ação — `θ = 0`, `s = 0`, UAV parado apontando para leste. Não é um viés inofensivo, é um atrator degenerado. Mitigações a implementar e comparar em T-04.4: (i) parametrizar a saída em `[-1,1]` e mapear o centro do intervalo de ação em `f = 0`; (ii) modo auto-normalizado opcional, `f = Σ v_i / max(1, #posições visitadas)`.

**Decisão normativa.** Um único núcleo (gradiente funcional, tabela de valores) com **modo de passo plugável**:

| Modo | Passo | Estado por entrada | Uso previsto |
|---|---|---|---|
| `constant` | `c` fixo | `v` | ator (fiel a Katopodis §4.3.1) |
| `ewma` | `α = 1/s`, `s ← φ·s + 1` | `v`, `s` | crítico (reproduz a NTRN-φ, eq. 4.8–4.10, com `s` armazenado em vez de recalculado por `φ^c`) |
| `global_decay` | `c` fixo + decaimento multiplicativo global por passo | `v` + escala global | experimental, ver T-04.4 |

Armazenar `s` diretamente, e não o contador `c`, evita calcular `φ^c` a cada consulta e evita a questão de saturação do contador. É desvio de forma em relação à eq. 4.10, não de conteúdo: `s` é literalmente o valor que a eq. 4.10 computa.

---

## 5. Tarefas

### T-04.1 — Auditoria do protótipo `ntuple.py`

Antes de estender, verificar. Checklist fechada, cada item com veredito escrito no relatório da spec:

1. **Convenção de sinal.** A eq. 4.11 de Katopodis, como impressa, é `v ← v + η·∂c/∂f` — isto é, *subida* de gradiente sobre a perda. Já o alg. 4.3 (REINFORCE) usa `v ← v + α·G·∂_f ln f`, que é subida sobre o objetivo, equivalente a *descida* sobre `R = −G ln f` (eq. 4.25). **As duas passagens são inconsistentes no texto**; a impressa em 4.11 é quase certamente um sinal trocado. Fixamos `W ← W − η·∂L/∂f` e testamos por monotonicidade (T-04.9). Verificar qual das duas o protótipo implementou.
2. **Cobertura de bits do mapeamento.** Se o sorteio das tuplas for independente por RAM, `N·n` posições de bit são amostradas de `B`. Com o codificador da SPEC-01 em `T = 128` e 25 escalares, `B ≈ 3 200`; com `N = 48`, `n = 12`, isso é `576` posições — **a rede lê 18 % dos bits da entrada e ignora o resto, em silêncio**. Medir a cobertura do protótipo. Se for esse o caso, é candidato número um a explicar qualquer resultado ruim em `T` alto, e a correção está em T-04.2.
3. **Tuplas duplicadas.** Duas RAMs com o mesmo conjunto de índices são uma só RAM com peso 2 — reduz `N` efetivo sem reduzir `N` nominal, e portanto quebra a relação `c = η·N`. Contar duplicatas.
4. **Escala do passo.** Confirmar se o protótipo expõe `η` ou `c`. Se expuser `η`, os números de RMSE por `N` da tabela de retorno decrescente (0,0366 / 0,0325 / 0,0319) foram obtidos com passos efetivos diferentes e **precisam ser refeitos com `c` fixo** antes de entrarem no relatório.
5. **Esquecimento.** Qual das três variantes da §4.4 está implementada, e se há denominador. Se o esquecimento for `v ← φ·v` aplicado só nas entradas visitadas e sem normalizador, é decaimento em direção a zero e **não** o esquecimento de Katopodis — ver T-04.4.
6. **Ordem dos bits no endereço.** Convenção LSB/MSB documentada e fixada; qualquer mudança invalida checkpoints salvos.
7. **Inicialização.** `W = 0` implica `f = 0` para entrada inédita. Registrar; a implicação para o crítico e para o ator está na §4.4.
8. **Tipos.** Acumulação em fp64 mascarando problema que aparecerá em fp32/int16; índices em `int64` inflando o custo de memória do mapeamento.

Saída: tabela de vereditos e lista de correções, com as que alteram números já medidos marcadas como **remedição obrigatória**.

### T-04.2 — Mapeamento de tuplas: sorteio, cobertura e reprodutibilidade

**Construção normativa — permutações concatenadas.** Gerar `⌈N·n/B⌉` permutações independentes de `[0, B)`, concatenar, truncar em `N·n`, reagrupar em `(N, n)`. Propriedades, todas verificáveis:

- cobertura exata `min(1, N·n/B)` da entrada; nenhum bit lido `⌈N·n/B⌉+1` vezes;
- quando `N·n ≥ B`, **todo bit é lido ao menos uma vez** — inclusive os bits de validade do zero-padding (PRD §5.4), que sob sorteio independente podem simplesmente nunca ser lidos, tornando o remendo inócuo sem aviso;
- determinismo total dado `(B, n, N, seed)`.

**Reprodutibilidade.** O gerador pseudoaleatório é parte do contrato. Não usar `numpy.random` legado nem depender de estabilidade de fluxo entre versões: implementar um Fisher–Yates alimentado por um gerador de contador determinístico e documentado (por exemplo SplitMix64 semeado por `seed`), ou congelar `numpy.random.Generator(PCG64(seed))` com o teste de ouro abaixo detectando qualquer mudança de fluxo.

Contrato:

```python
@dataclass(frozen=True)
class TupleMapping:
    idx: np.ndarray        # (N, n) int16 ou int32; int16 exige B <= 32767 (asserção)
    n_bits_in: int         # B
    seed: int

    @classmethod
    def draw(cls, n_bits_in: int, tuple_size: int, n_rams: int, seed: int) -> "TupleMapping": ...
    def address(self, bits: np.ndarray) -> np.ndarray:        # (B,) uint8  -> (N,) int32
    def address_batch(self, bits: np.ndarray) -> np.ndarray:  # (M,B) uint8 -> (M,N) int32
    def coverage(self) -> dict                                # bits lidos, min/max repetições, duplicatas
    def digest(self) -> str                                   # SHA-256 de idx, para teste de ouro
```

**Teste de ouro:** `digest()` de `(B=256, n=8, N=16, seed=42)` gravado em `tests/golden/tuple_mapping.json`. Falha nesse teste é sinal de que qualquer checkpoint anterior ficou inválido — e é exatamente o erro que passa despercebido em uma troca de versão de NumPy.

**Custo (entra em T1, PRD §5.5).** `N·n` índices em int16. Com `N = 48`, `n = 12`: 576 índices, 1,2 KB. Reportar as duas contabilizações: **mapeamento armazenado** (1,2 KB, o número honesto por omissão) e **mapeamento regenerado da seed** (8 bytes de seed, mas exige o gerador embarcado e `O(B)` de trabalho no boot). O PRD já exige que isso não seja omitido; a escolha entre as duas é declarada, não escondida.

### T-04.3 — Núcleo de gradiente funcional

Contrato mínimo, com a fronteira do módulo posta exatamente onde a SPEC-05 precisa:

```python
@dataclass(frozen=True)
class NTupleConfig:
    n_bits_in: int
    tuple_size: int          # n
    n_rams: int              # N
    n_heads: int             # K
    c: float                 # taxa no espaço de funções; eta = c / N (nunca expor eta)
    step_mode: Literal["constant", "ewma", "global_decay"]
    phi: float = 1.0         # fator de esquecimento; usado por "ewma" e "global_decay"
    storage: Literal["hash", "dense", "hashed_dense"] = "hash"
    dtype: Literal["fp32", "int16"] = "fp32"
    seed: int = 0
    c_max: float = 0.5       # guarda RNF-01

class NTupleNet:
    def forward(self, bits: np.ndarray) -> np.ndarray:            # (B,) -> (K,) float32
    def forward_addr(self, addr: np.ndarray) -> np.ndarray:       # (N,) -> (K,)
    def update_addr(self, addr: np.ndarray, dL_df: np.ndarray) -> None   # (K,)
    def update(self, bits: np.ndarray, dL_df: np.ndarray) -> None
    def occupancy(self) -> OccupancyStats
    def op_counts(self) -> OpCounts
    def freeze(self) -> "FrozenNTupleNet"
    def save(self, path) -> None
    @classmethod
    def load(cls, path) -> "NTupleNet"
```

**Regras de contrato, todas testáveis:**

- `update_addr` recebe **`∂L/∂f`**, não o erro nem o alvo. Quem sabe a perda é a SPEC-05. O núcleo aplica `W[k, i, a_i] += −(c/N)·dL_df[k]` para as `N` entradas endereçadas e para cada `k`.
- Separar `address` de `forward`/`update` é obrigatório: o laço de RL calcula o endereço **uma vez** e o reutiliza no ator, no crítico e na atualização. É a única otimização de latência que muda ordem de grandeza, e ela só existe se a API expuser `*_addr`.
- Perdas de referência vivem em `losses.py` com a assinatura `(f, y) -> (L, dL/df)`: `mse` para validação; `td` e `gaussian_logprob` ficam declaradas aqui e são exercidas pela SPEC-05. Registrar a identidade que a SPEC-05 vai testar: para política gaussiana `L = −A·ln π(a|s)`, `∂L/∂f = −A·Σ⁻¹(a − f)`, logo `W += (c/N)·A·Σ⁻¹(a − f)`, que é a eq. 4.35 de Katopodis com o sinal já reconciliado (T-04.1).
- `forward` de entrada inédita retorna `0`, não erro, e **não cria entradas** na tabela. Consulta que aloca é bug de memória e envenena a curva de ocupação do RNF-08.

### T-04.4 — Esquecimento: implementar as três variantes e decidir por medida

Implementar os três modos da §4.4 e comparar em um teste de não-estacionariedade controlado, sem RL: regressão de alvo com **degrau** (`y = g₁(x)` por 20k passos, depois `y = g₂(x)`), medindo tempo de readaptação e erro em regime.

| Modo | Semântica | Custo | Cuidado |
|---|---|---|---|
| `constant` | passo fixo `c`; a própria iteração de gradiente já esquece exponencialmente com constante de tempo `≈ 1/c` por entrada | `v` só | é o que Katopodis argumenta ser suficiente para policy search (§4.3.2) |
| `ewma` | `s ← φ·s + 1`, `α = 1/s`; reproduz eq. 4.8–4.10 com correção de viés | `v` e `s` — **dobra os bytes por entrada** | só o crítico usa; o crítico não embarca (PRD §5.5 T1), então não contamina a alegação de embarcabilidade |
| `global_decay` | `W ← φ·W` em **todas** as entradas a cada passo | `v` + escala global | ver abaixo |

**Sobre `global_decay`, e por que ele merece existir.** Aplicar `φ` só nas entradas visitadas — que é o que uma leitura apressada da eq. 4.8 sugere para o gradiente funcional — acopla a taxa de esquecimento à **frequência de visita**: regiões muito visitadas esquecem rápido em tempo de relógio, regiões raras não esquecem nunca. Não é o que a fig. 4.2 de Katopodis descreve, que é decaimento em tempo de amostra, uniforme. O decaimento uniforme é implementável em `O(1)` amortizado com uma escala global `g`: manter `f = g·Σ v_i` e atualizar `v_i += −(c/N)·dL_df / g`. Riscos a tratar explicitamente: `g → 0` em subfluxo (renormalizar `v ← g·v`, `g ← 1` quando `g < 1e-6`, custo `O(ocupadas)` e frequência declarada) e amplificação de `1/g` no incremento, que interage com a quantização de T-04.8.

**Decisão a registrar ao fim da tarefa:** padrão `constant` para o ator e `ewma` para o crítico (fiel ao alg. 4.4), com `global_decay` retido como ablação apenas se a medida do degrau mostrar vantagem.

**Além disso, a ablação de auto-normalização** descrita na §4.4 (`f = Σ v_i / max(1, visitadas)`), com a métrica sendo o viés de `f` em região fria — medido diretamente na regressão sintética, em pontos fora do suporte de treino.

### T-04.5 — Guarda de estabilidade (RNF-01)

1. **Construção — reformulada.** A guarda `assert c <= c_max` com `c = η·N` pertencia à convenção de **soma**. Na convenção de **média** da eq. (4.12) o passo é independente de `N`, então não há `c` a limitar. A guarda que sobra é sobre o **passo efetivo**: `assert |α · Δ| <= passo_max` no momento da atualização, com `passo_max` configurável. O NaN medido no protótipo (`η = 0,05`, `N = 48`) foi artefato da soma e **não se reproduz** no modelo do artigo.
2. **Execução:** verificação de finitude de `f` e do gradiente recebido, em modo `debug` sempre e em modo produção a cada `K` passos (`K` configurável, por omissão 1000) — o custo por passo é o que decide, e ele entra na medida de latência da SPEC-03.
3. **Teste de regressão do NaN:** configuração `N = 48`, `c = 2,4` deve **falhar na construção**; forçando a guarda desligada, deve divergir em menos de `X` passos na regressão sintética. Congelar `X` medido.
4. **Diagnóstico exportado:** `‖W‖∞` das entradas ocupadas e `max |Δf|` por passo, para a SPEC-05 conseguir distinguir "divergiu" de "não aprendeu".

Documentar no README do módulo que `c` — e não `η` — é o hiperparâmetro, e por quê. É a regra do PRD que mais facilmente se perde em uma refatoração.

### T-04.6 — Armazenamento: esparso, denso, e a variante que resolve os dois

Base medida (PRD §5.5, achado A1): `n = 12`, `N = 48`, uma cabeça, 80k passos → **2 268 de 196 608 entradas endereçadas (1,2 %)**, 4,4 KB em int16 contados como "ocupado".

O ponto desta tarefa é que **"ocupado" não é "realizado"**, e a diferença é grande o bastante para mudar a conclusão de H1.

**Backend A — `hash` (treino).** Endereçamento aberto com sondagem linear, capacidade potência de dois, crescimento por duplicação. Chave composta empacotada:

```
chave = (ram_idx << n) | endereço            # com cabeças vetoriais, k não entra na chave
```

Com `N ≤ 128` (7 bits) e `n ≤ 16`, a chave cabe em `uint32`; com `n ≤ 8` cabe em `uint16`. Arrays paralelos `keys: uint32[cap]`, `vals: dtype[cap, K]` (e `s: dtype[cap]` no modo `ewma`). Sentinela de vazio explícita (chave `0` é válida — usar um array de ocupação ou reservar o valor máximo). Fator de carga máximo **0,5** para manter a sondagem curta; é essa folga que faz o custo realizado explodir.

Conta a fazer e reportar, para `2 268` entradas, `K = 2`, int16: capacidade `8 192` (potência de dois acima de `2268/0,5`), `4 + 2·2 = 8` bytes por slot ⇒ **64 KB realizados** contra 9 KB "ocupados". Uma ordem de grandeza de diferença, e pior que os ~8,6 KB da MLP pareada. **Este é o número que o PRD §5.5 chama de honesto, e ele não favorece a hipótese.**

**Backend B — `dense`.** `W` como array `(K, N, 2^n)`. Latência determinística, sem chave, sem colisão — é a mitigação prescrita pelo RNF-10 para o risco de perder no p99. Custo: `K·N·2^n·bytes`, que só é viável para `n` pequeno. Tabela a preencher e publicar, porque ela impõe um acoplamento que não é óbvio:

| `n` | denso, `K=2`, int16 | viável embarcado? |
|---|---|---|
| 8 | 48 KB | limítrofe |
| 10 | 192 KB | não |
| 12 | 768 KB | não |

Isto é, **a mitigação de p99 do RNF-10 só existe para `n ≤ 8`**, o que a acopla à varredura `n ∈ {8, 10, 12}` da SPEC-06. Registrar esse acoplamento no relatório: não é livre escolher `n = 12` e prometer latência previsível.

**Backend C — `hashed_dense` (o truque de hashing).** Array plano de tamanho `m` fixo, índice `h(ram_idx, endereço) mod m`, **sem armazenar chave**, colisões aceitas como ruído. É a solução clássica de tile coding com hashing e resolve RNF-02 e RNF-10 simultaneamente: memória limitada por construção, latência de um único acesso, zero bytes de chave. Custo: colisões viram interferência entre estados. Medir a taxa de colisão em função de `m` (com `m = 4096` e 2 268 chaves distintas, esperar ~24 % de entradas em colisão) e o efeito no RMSE da regressão sintética. Se o efeito for pequeno, este vira o backend recomendado para implantação e o argumento de embarcabilidade se recupera.

**Estrutura congelada de implantação (`freeze()`).** Distinguir explicitamente a estrutura de **treino** (mutável, com folga de crescimento, com `s` no crítico) da de **implantação** (imutável, compacta, só o ator): por RAM, array ordenado de endereços `uint16` + valores, com deslocamentos por RAM e busca binária ou hash perfeito mínimo. Sem folga de carga: `2 268 × (2 + 2·2) = 13,3 KB` mais `N·4` bytes de deslocamentos mais 1,2 KB de mapeamento. **É este o número que vai para T1 na coluna "realizado" do ator.**

Contrato de instrumentação, consumido pela SPEC-03:

```python
@dataclass
class OccupancyStats:
    allocated_entries: int      # K * N * 2^n
    occupied_entries: int       # chaves distintas endereçadas
    capacity_slots: int
    load_factor: float
    bytes_keys: int
    bytes_values: int
    bytes_counters: int         # modo ewma
    bytes_mapping: int
    bytes_realized: int         # a soma que vale
```

E o **RNF-08**: `occupied_entries` amostrada em uma grade logarítmica de passos, persistida em CSV, para a curva de ocupação × passos com verificação de saturação. O gancho é do núcleo; o gráfico é da SPEC-03.

### T-04.7 — Múltiplas cabeças de saída

Katopodis fig. 4.3 apresenta as duas formas de modelar uma função vetorial. Comparação:

| | (a) um discriminador, posições vetoriais | (b) `K` discriminadores reais |
|---|---|---|
| Mapeamento de tuplas | um, compartilhado | `K` (ou um compartilhado) |
| Cálculo de endereço por inferência | `N·n` leituras, **uma vez** | `K·N·n` leituras |
| Chave no hash | uma por entrada, amortizada entre `K` valores | uma por entrada **por cabeça** |
| Bytes por entrada ocupada (int16, hash) | `4 + 2K` | `K·(4 + 2)` |
| Padrão de acesso | `K` valores contíguos, uma linha de cache | `K` sondagens independentes |
| Flexibilidade | `n`, `N` e ocupação comuns a todas as saídas | por cabeça |
| Generalização | acoplada: o que generaliza para `θ` generaliza para `s` | independente por saída |

**Recomendação: (a), posições vetoriais, para o ator `⟨θ, s⟩`.** Três razões, na ordem em que pesam: o custo de inferência é dominado pelo cálculo de endereço, e (a) o paga uma vez; o custo de chave, que é o que estraga o número realizado de T-04.6, é amortizado por `K`; e a sondagem única melhora o p99, que é o risco explícito do RNF-10. A perda de flexibilidade é real mas não morde aqui — `θ` e `s` são componentes da mesma decisão, sobre a mesma observação.

Manter (b) implementável como caso trivial (`K` instâncias de `K=1`) e usá-lo como **ablação**, para que a afirmação acima seja medida e não asseverada. O crítico é `K = 1` e é instância separada.

**Extensão registrada, não implementada por omissão:** ator e crítico podem compartilhar o cálculo de endereço se tiverem `(n, N, seed)` iguais — economiza ~50 % da latência de inferência de treino. Acopla a generalização dos dois, e Katopodis os trata com hiperparâmetros separados (alg. 4.4 tem `n_value`, `N_value`, `n_policy`, `N_policy`). Fica como opção de configuração desligada, com nota de que ligá-la muda o modelo.

### T-04.8 — Quantização fp32 / int16 (RNF-09)

**Ponto de partida normativo: treinar em fp32, quantizar depois (post-training), avaliar a política quantizada.** Reportar int16 ao lado de desempenho fp32 é exatamente o que o RNF-09 proíbe.

- Escala fixa por rede, `val_int16 = round(v / Δ)` com `Δ = V_max / 32767`, `V_max` derivado do percentil 99,9 de `|v|` observado no fim do treino, com saturação registrada (contar entradas saturadas; saturação não nula é resultado, não detalhe).
- Acumulação da soma das `N` entradas em `int32`; conversão a float só na saída. É o que sustenta a alegação de "caminho de inferência sem MAC em ponto flutuante" (RNF-12) — com a ressalva de que a amostragem gaussiana da ação continua sendo ponto flutuante, e isso precisa aparecer na contagem.
- **Erro de quantização:** limite de pior caso `N·Δ/2` na saída; medir o real. Com `N = 48` e `Δ = V_max/32767`, o pior caso é `7,3e-4·V_max` — pequeno, mas a verificação é barata e evita uma surpresa.
- **Treino em int16 é armadilha declarada:** com `c = 0,05` e `N = 48`, `η·∂L/∂f` é da ordem de `1e-3·Δ⁻¹` e frequentemente arredonda para zero — zona morta que congela o aprendizado sem qualquer sinal de erro. Se alguém insistir, exige acumulador de erro (dithering) e é outro projeto. Registrar como fora de escopo.
- Quantizar também o mapeamento de tuplas em int16 (asserção `B ≤ 32767`).

Saída: par `(tamanho, desempenho)` para fp32 e int16 na regressão sintética; o par equivalente sobre a política é da SPEC-06.

### T-04.9 — Validação sem RL

**Regressão sintética.** Alvo `sin(3x)·cos(2y)` em `[-1,1]²`, imagem em `[-1,1]`, entrada binarizada pelo codificador da SPEC-01. Reproduzir os números do protótipo **com `c` fixo entre as configurações** (ver T-04.1 item 4):

| `N` | RMSE do protótipo | memória alocada |
|---|---|---|
| 16 | 0,0366 | 256 KB |
| 48 | 0,0325 | 768 KB |
| 128 | 0,0319 | 2 048 KB |

O achado a preservar é o **retorno decrescente medido**: 8× de memória para 13 % menos erro. O critério de saída é reproduzir a *ordenação e a ordem de grandeza*, não o dígito — mudança de mapeamento (T-04.2, cobertura) e de escala de passo (T-04.1) altera legitimamente os valores. **Discrepância grande é sinal de que a auditoria encontrou algo real, e a explicação vai no relatório.**

Acrescentar à mesma bancada: erro fora do suporte de treino (para a ablação de auto-normalização de T-04.4), o teste de degrau de T-04.4 e a varredura de colisão do backend C.

**Suíte de testes de propriedade** (`tests/test_ntuple.py`), todos determinísticos e sem tolerância frouxa:

1. `‖φ(x)‖₀ = N` exatamente, para toda entrada.
2. Endereçamento determinístico; `digest()` do mapeamento contra o arquivo de ouro.
3. `W = 0 ⇒ f = 0`; linearidade: `f(W₁ + W₂) = f(W₁) + f(W₂)` sobre o mesmo endereço.
4. **Passo exato:** com perda quadrática, uma atualização sobre `(x, y)` leva `f(x)` de `f` para exatamente `f + c·(y − f)`. É igualdade algébrica, testável com tolerância de fp32; é o teste que pega inversão de sinal, `N` faltando e normalização errada de uma vez só.
5. **Sinal:** a perda não aumenta após um passo com `c < 1` (contra o item 1 de T-04.1).
6. **Esquecimento:** no modo `ewma`, com entrada fixa e alvo constante `y`, `f → y` para qualquer `φ ∈ (0,1]`; após degrau no alvo, a constante de tempo é `≈ 1/(1−φ)` amostras.
7. **Isolamento de cabeças:** atualizar a cabeça 0 não altera a saída da cabeça 1.
8. **Contabilidade de ocupação:** `occupied_entries` é igual ao número de chaves distintas endereçadas, sem contar consultas (`forward` não aloca).
9. **Hash sob colisão forçada:** construir com capacidade mínima para forçar sondagem longa; leitura e escrita continuam corretas; crescimento preserva todos os valores.
10. **Serialização:** `save`/`load` idempotente, incluindo mapeamento, escala global do `global_decay` e parâmetros de quantização; `freeze()` preserva `f` em todas as entradas ocupadas.
11. **Equivalência entre backends:** para o mesmo mapeamento e a mesma sequência de atualizações, `hash` e `dense` produzem `f` bit a bit idêntico (o `hashed_dense` não, por construção — e a diferença é a métrica de colisão).
12. **Guarda RNF-01:** `c > c_max` falha na construção.

### T-04.10 — Instrumentação de custo (alimenta SPEC-03)

Expor contadores, não medições — o protocolo é da SPEC-03 e nada é medido duas vezes (regra 4 do `specs/README.md`).

```python
@dataclass
class OpCounts:
    bit_reads: int          # N * n por inferência
    shifts_ors: int         # N * n
    int_lookups: int        # N (layout vetorial) ou K * N
    int_adds: int           # K * N
    float_macs: int         # 0 no caminho de inferência da LUT; > 0 só na amostragem da ação
```

Com `n = 12`, `N = 48`, `K = 2`, layout (a): 576 leituras de bit, 48 consultas, 96 somas inteiras, **zero MAC em ponto flutuante** na LUT. É a linha do RNF-12 que sustenta o argumento estrutural de H1, e ela precisa vir com a ressalva de que codificador e amostragem gaussiana não são zero.

Além disso: modo de contagem ligável/desligável com custo declarado, e o gancho de amostragem de ocupação do RNF-08.

---

## 6. Critério de saída

1. **Auditoria de T-04.1 concluída**, com veredito escrito por item e a lista do que exigiu remedição.
2. **Regressão sintética converge** e reproduz o retorno decrescente em `N` com `c` fixo; qualquer divergência em relação aos números do protótipo explicada pela auditoria.
3. **Suíte de propriedade verde**, incluindo os itens 4 (passo exato), 6 (esquecimento) e 12 (guarda RNF-01).
4. **Reprodutibilidade demonstrada:** mesma seed ⇒ mesmo `digest()` do mapeamento ⇒ mesmos pesos após a mesma sequência de atualizações, em duas máquinas ou dois ambientes.
5. **Três backends implementados e medidos** na mesma bancada: `hash`, `dense`, `hashed_dense`, com `bytes_realized` e latência mediana/p99 tabelados. A tabela de T1 do ator para `n = 12`, `N = 48`, `K = 2` fechada com as quatro linhas — alocado, ocupado, realizado (treino), realizado (congelado) — mais mapeamento.
6. **Decisão de esquecimento registrada** com o número do teste de degrau que a sustenta, e a correspondência `c ↔ 1 − φ` documentada para a SPEC-06 não varrer duas vezes a mesma grandeza.
7. **Decisão de cabeças registrada**, com (a) e (b) medidos em latência e em bytes realizados.
8. **Par `(tamanho, desempenho)` fp32 e int16** na regressão sintética.
9. **Contadores de `OpCounts` e a curva de ocupação** expostos no formato que a SPEC-03 consome, com um CSV de exemplo gerado.
10. **Relatório curto** (2–3 páginas) com a tabela de custo do núcleo e a seção "adaptações de Katopodis" (§4 desta spec) pronta para o capítulo de método de E2.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **O tamanho realizado do hash mata a alegação de embarcabilidade** (~64 KB contra ~8,6 KB da MLP pareada) | é o risco central desta spec. Três frentes: estrutura congelada de implantação (T-04.6), backend `hashed_dense` sem chave, e — se nenhuma resolver — migrar o argumento para contagem de operações e ausência de FPU, como o próprio PRD §5.5 já prevê. Reportar o número desfavorável de qualquer forma |
| **Cobertura de bits baixa** (`N·n ≪ B`): a rede lê 18 % da entrada em `T = 128` e ninguém percebe | construção por permutações concatenadas (T-04.2) e `coverage()` reportado em toda configuração; a SPEC-06 precisa ver `N·n/B` ao lado de cada resultado, senão vai atribuir a `T` um efeito que é de cobertura |
| **Encolhimento para zero em região fria** empurra a média da ação para o canto do espaço de ação | ablação de auto-normalização e parametrização centrada em `[-1,1]` (§4.4, T-04.4); métrica direta de viés fora do suporte de treino |
| **Sinal do gradiente invertido** — a eq. 4.11 impressa é subida sobre a perda | convenção fixada, teste 4 (passo exato) e teste 5 (monotonicidade); é o erro que produz "não converge" sem nenhuma outra pista |
| **Esquecimento acoplado à frequência de visita** em vez de ao tempo de amostra | as três variantes implementadas e comparadas no teste de degrau; `global_decay` com escala global e renormalização |
| **Confusão entre `c` e `η` numa refatoração** reintroduz a divergência para NaN | `η` nunca aparece em configuração; asserção na construção; teste de regressão do caso `c = 2,4` |
| Erros de transcrição do próprio texto de referência (alg. 4.4 mistura `v_policy` e `v_value` na atualização do crítico, e indexa `f_value` por `(S, A)` sendo função de estado) | seguir o texto de §4.2.1/§4.3.1/§4.3.4, não os quadros de algoritmo; o crítico aproxima `V(S)`, sem argumento de ação. Registrar as divergências no relatório para que a SPEC-05 não as reintroduza |
| **Perda no p99 por causa do hash** (RNF-10) | backends `dense` e `hashed_dense` medidos desde já; declarar o acoplamento com `n ≤ 8` para a variante densa |
| **Zona morta na quantização int16** se alguém tentar treinar quantizado | quantização só pós-treino; treino int16 declarado fora de escopo |
| Números do protótipo (RMSE por `N`) obtidos com `η` fixo em vez de `c` fixo | remedição obrigatória em T-04.9 antes de qualquer uso no relatório |
| Instabilidade de fluxo do PRNG entre versões invalida checkpoints silenciosamente | gerador determinístico próprio ou fluxo congelado, mais teste de ouro do `digest()` |

---

## 8. Entregáveis

- `src/uavwnn/wnn/ntuple.py` — núcleo de gradiente funcional, modos de passo, cabeças vetoriais, guarda RNF-01, instrumentação (endurecimento do protótipo)
- `src/uavwnn/wnn/mapping.py` — `TupleMapping`, sorteio por permutações concatenadas, `coverage()`, `digest()`
- `src/uavwnn/wnn/sparse.py` — backends `hash`, `dense`, `hashed_dense`; `freeze()` e a estrutura compactada de implantação; `OccupancyStats`
- `src/uavwnn/wnn/losses.py` — `mse`, e as assinaturas de `td` e `gaussian_logprob` que a SPEC-05 preenche
- `configs/spec04/` — bancada de regressão sintética, teste de degrau, varredura de colisão
- `tests/test_ntuple.py`, `tests/golden/tuple_mapping.json`
- `reports/spec04-ntuple.md` + figuras: RMSE × `N`, curva de ocupação × passos, bytes realizados por backend, latência mediana/p99 por backend

---

## 9. Perguntas em aberto

1. **A conta de tamanho realizado sobrevive?** Com folga de carga e chave, o hash de treino chega a ~64 KB para 2 268 entradas — pior que a MLP pareada. A estrutura congelada leva a ~14,5 KB e o `hashed_dense` a menos ainda, mas nenhuma das duas foi medida. Se nem a congelada couber em "poucas dezenas de KB" com margem confortável, o critério de H1 no PRD §5.5 precisa ser reescrito **antes** da SPEC-06 rodar, não depois de ver o número.
2. **`hashed_dense` é aceitável como o modelo, ou é outro modelo?** Colisões sem chave mudam a função aproximada — deixa de ser tile coding puro e vira tile coding com hashing. É defensável (é literatura clássica), mas se o backend de implantação não for o mesmo do treino, a política avaliada não é a política treinada. Decidir: treinar já com colisões (honesto, mais barato, muda o modelo) ou treinar limpo e congelar com colisões (exige reavaliar após congelar).
3. **O crítico paga `ewma` ou `constant`?** O modo `ewma` dobra os bytes por entrada e é fiel ao alg. 4.4; o `constant` é mais simples e Katopodis argumenta em §4.3.2 que o passo constante já lida com a não-estacionariedade. Como o crítico não embarca, o custo é de treino — mas a escolha muda o comportamento e não há medida ainda.
4. **Cobertura de bits vira restrição de projeto?** Se a exigência for `N·n ≥ B`, então `N` deixa de ser hiperparâmetro livre e passa a ser `N ≥ B/n` — com `B ≈ 3 200` e `n = 12`, isso é `N ≥ 268`, muito acima dos `N ∈ {16, 48, 128}` da varredura do PRD. Ou se aceita subamostrar a entrada (e se reporta a fração), ou a varredura de `T` fica limitada por cima. Interage diretamente com o resultado da SPEC-01 e precisa ser resolvido junto com ela.
5. **A correspondência `c ↔ 1 − φ` permite cortar uma dimensão da varredura?** Se sim, a grade `T × n × N × esquema × esquecimento` da SPEC-06 encolhe de forma material. Se os dois hiperparâmetros tiverem efeitos separáveis na prática (passo do ator vs. esquecimento do crítico), não permite. É medível no teste de degrau, mas com custo.
6. **Ator e crítico compartilham endereçamento?** Vale ~50 % da latência de treino e acopla a generalização dos dois. Katopodis os mantém separados. Fica desligado por omissão — mas se o orçamento de simulação da P2 apertar, é a primeira alavanca.
7. **Qual `c_max` de fato?** `c < 2` é a fronteira de LMS para alvo fixo; `0,5` é margem escolhida, não medida. Em RL o alvo se move e a fronteira efetiva é menor. O valor deveria ser calibrado contra o crítico com bootstrap, o que só é possível na SPEC-05 — até lá, `0,5` é uma guarda conservadora e não uma constante justificada.
8. **A perda por colisão é interferência ou regularização?** No regime de 1,2 % de ocupação a probabilidade de colisão entre estados *visitados* pode ser tolerável e até benéfica para generalização. Vale ao menos uma medida na bancada sintética antes de tratá-la como custo puro.
