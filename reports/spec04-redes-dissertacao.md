# SPEC-04/05 — As redes e o algoritmo, fiéis à dissertação

Fonte primária: Katopodis, *Reinforcement Learning with Weightless Neural
Networks*, dissertação de mestrado, PESC/COPPE-UFRJ, janeiro de 2022
(`tmp/947841_compressed.pdf`, 87 p.). Seções §4.2.1, §4.3.1, §4.3.3 e
Algoritmo 4.4.

Reprodução: `uv run pytest tests/test_redes.py tests/test_actor_critic.py`.

---

## 1. Por que isto foi refeito

O `NTupleRegressor` da SPEC-01 é um regressor supervisionado denso: uma rede,
uma regra de atualização, `2ⁿ` posições alocadas por RAM. A dissertação pede
outra coisa em três dimensões.

**Duas redes, não uma.** O capítulo 4 define modelos separados para papéis
separados, e isso fecha a pendência C2 — que perguntava como conciliar
gradiente funcional *com* esquecimento. Não se concilia:

| papel | seção | modelo | guarda | esquecimento |
|---|---|---|---|---|
| ator | §4.3.1 | gradiente funcional | só valor | não |
| crítico | §4.2.1 | regressão com esquecimento | valor **e** contador | sim (φ) |

O motivo aparece no Algoritmo 4.4: o crítico é alimentado com o retorno `G`,
cujo alvo se desloca conforme a política melhora. É a distribuição mutável que
o fator de esquecimento existe para acompanhar. O ator não tem alvo se movendo
assim.

**Armazenamento esparso.** Não é detalhe de implementação. O apêndice A da
dissertação usa tuplas de **16, 32 e 64 bits**; `2⁶⁴` posições por RAM não
cabem em array denso. O `NTupleRegressor` fica preso a `n ≲ 16`.

**Mapeamento por partição.** `randperm(B)` fatiado em `⌈B/n⌉` tuplas, como o
`:uniform_random` do `Ramnet.jl`. Cada bit é lido **exatamente uma vez**, logo
a cobertura é 1,0 por construção — o achado C4 resolvido pela estrutura em vez
de por verificação.

O que **não** mudou: o termômetro. A SPEC-01 mediu que o quantil por slot vence
o linear global por 4,3° contra 7,6°, e essa escolha é nossa, não da
dissertação. Fica fora das redes, injetada pelo chamador.

---

## 2. As equações, como implementadas

**Ator** — eqs. (4.11) e (4.12):

```
v[i]_t(x) = v[i]_{t-1}(x) + α_t · Δ_t,   i = 1..N
f(x)      = (1/N) Σ_i v[i](x)
```

A eq. (4.12) traz o `1/N` **explícito**. É a convenção que desacopla a taxa de
aprendizado de `N`: com `Δ = −η(f − y)`, a variação da saída num passo é
exatamente `−η(f − y)`, seja qual for `N`. O `EmbeddingBag(mode='sum')` do
repositório em PyTorch usa soma, e aí o passo efetivo seria `−Nη(f − y)`.

**Crítico** — eqs. (4.8), (4.9) e (4.10):

```
v[i]_{t+1}(x) = φ · v[i]_t(x) + y_{t+1}
c[i]_{t+1}(x) = c[i]_t(x) + 1

f(x) = Σ_i v[i](x) / Σ_i s[i](x)          se Σ s ≠ 0, senão 0

s[i](x) = (1 − φ^c[i](x)) / (1 − φ)       se c > 0 e φ ≠ 1
        = c[i](x)                          caso contrário
```

O denominador é **soma de pesos**, não de contadores: com esquecimento as
observações passadas contribuem menos, e a média precisa ser ponderada. A forma
fechada é a soma dos primeiros termos de uma série geométrica.

**Política** — eqs. (4.34) e (4.35), gaussiana de covariância **predefinida**:

```
∇_f ln f(a|s) = Σ⁻¹ (a − f(s)) · k(s, ·)
```

O fator `k(s, ·)` é a função de base do núcleo; na rede n-tuple ele é
implícito — escrever nas `N` posições endereçadas por `s` **é** aplicar a base.

**Algoritmo 4.4**, com os dois pontos que o pseudocódigo esconde:

- **A vantagem usa o valor ANTES da atualização.** `δ = G − f_value(S_τ)` é
  calculado com a rede no estado corrente; o crítico só treina depois. Ator e
  crítico enxergam a mesma estimativa.
- **O crítico não é regredido para `G` — ele acumula `G`.** A eq. (4.8) faz
  `v ← φv + G`, e a média ponderada de (4.9) é quem converte acumulação em
  estimativa. É diferente de um passo de gradiente para o alvo `G`, e é a razão
  de o crítico precisar de contador.

---

## 3. Validação: as duas redes são idênticas ao Ramnet.jl

Teste diferencial contra `Ramnet.jl@faster_hashing`, Julia 1.10.5. Mesmo
particionamento, mesmos dados, mesma sequência de atualizações.

| configuração | \|Δ\| máx |
|---|---|
| ator, B=64, n=8, d=2 | **0** |
| ator, B=96, n=12, d=3 | **0** |
| crítico, B=64, n=8, φ=0,85 | **0** |
| crítico, B=96, n=12, φ=1,00 | **0** |
| crítico, B=64, n=8, φ=0,50 | **0** |

**Zero em todas** — idênticas bit a bit, não apenas dentro de tolerância.

O crítico é o que mais precisava de oráculo: a média ponderada de (4.9) com o
denominador de (4.10) tem várias formas plausíveis de errar — somar contadores
em vez de pesos, aplicar φ na ordem trocada, esquecer o ramo `φ = 1` — e todas
produzem números com cara de certos. Os três valores de φ cobrem o ramo
geométrico e o degenerado.

As predições ficam congeladas em `tests/fixtures/diferencial_redes/`, então a
suíte não exige Julia.

### O que NÃO tem oráculo

O laço do Algoritmo 4.4 é transcrição do pseudocódigo da página 48. Validá-lo
diferencialmente exigiria rodar o agente Julia inteiro, amarrado aos tipos de
ambiente e buffer do `RLWorktable.jl`. O que existe é mais fraco, e o relatório
final precisa dizer isso: os invariantes estruturais são verificados um a um
(retorno de `m` passos com bootstrap, vantagem antes do treino do crítico,
buffer consumindo uma transição por vez, fim de episódio sem bootstrap), e dois
testes de comportamento fecham a ponta — a política converge para os alvos de
um bandido contextual contínuo, e o crítico estima `−Σ` como deveria.

---

## 4. Achado: a instabilidade de passo fixo é mensurável

No bandido contextual, varrendo α em 5 sementes com 20 000 passos:

| α | pior erro entre as sementes |
|---|---|
| 0,002 | **0,044** |
| 0,005 | 0,059 |
| 0,010 | 0,072 |
| 0,020 | 0,144 |
| 0,050 | 0,238 |

**Monótono.** E em α = 0,05 uma das sementes chega a divergir — a trajetória
piora de 6 000 para 20 000 passos antes de recuperar em 60 000.

Isto é exatamente o que a dissertação aponta no capítulo 6:

> *"SGD is a simple and often effective algorithm but can lead to unstable
> policy updates due to its **fixed step size**."*

É a frase que motiva a sugestão de TRPO/PPO — e que o repositório
`anbubus/ntn_neurocomputing_v0` implementa. Vale reter duas consequências:

1. **A busca de α da SPEC-06 precisa de faixa larga e de várias sementes.** Meu
   primeiro palpite, α = 0,05, era 25× grande demais, e produziria uma curva de
   aprendizado ruim que pareceria limitação do modelo.
2. **Se o Algoritmo 4.4 não atingir o portão G1, a instabilidade é a primeira
   suspeita**, e a saída já está identificada pelo próprio autor.

---

## 5. Divergência entre o artigo e o código do autor (P13)

| | amostragem | gradiente |
|---|---|---|
| dissertação, eqs. (4.34)/(4.35) | `N(f(s), Σ)` | `Σ⁻¹(a − f(s))` |
| `ContinousActionPolicy.jl` | `Σ·z + f(s)` → Σ como **desvio** | `inv(Σ)·(f(s) − a)` |

Se Σ é a covariância, como (4.34) diz explicitamente, a amostragem correta é
`√Σ·z`. O código Julia usa Σ direto na amostragem mas `inv(Σ)` no gradiente,
tratando o mesmo número como desvio num lugar e como variância no outro.

**Seguimos o artigo.** Com Σ = 0,25, isso dá desvio 0,5 em vez de 0,25 — fator
2 na exploração, o que não é desprezível. A alternativa fica disponível em
`PoliticaGaussiana(..., escala_como_desvio=True)` para reproduzir o Julia, e a
escolha precisa aparecer na tabela de hiperparâmetros do relatório final.

Duas outras coisas do código Julia que **não estão na dissertação** e ficaram
desligadas por padrão: o recozimento linear da taxa de aprendizado e o da
covariância. O apêndice A lista α e Σ como constantes.

E uma ressalva sobre copiar o Julia: o default `end_learning_rate = 1e5` no
construtor é quase certamente erro de digitação por `1e-5` — taxa final maior
que a inicial contraria o propósito do recozimento.

---

## 6. O que isto muda nas pendências

| pendência | estado |
|---|---|
| **C2** — esquecimento com gradiente funcional | **fechada.** São duas redes; ambas implementadas e validadas contra o oráculo. |
| **C4** — cobertura de bits | **fechada pela estrutura.** A partição lê cada bit uma vez. |
| **C1** — ação circular | **segue aberta.** A dissertação usa gaussiana sem tratamento de wrap-around; o `CircularEncoder` do PyTorch resolve entrada circular, não ação. A normal projetada continua de pé. |
| **P13** — Σ como covariância ou desvio | **nova**, resolvida em recomendação (seguir o artigo), registrada. |

## 7. A pendência nova que mais me incomoda

**Estamos operando em `n ≈ 10–12` enquanto a dissertação opera em `n = 16–64`.**

A SPEC-01 escolheu `n = 12` e a SPEC-03 achou que `n = 10` domina em memória.
Mas as duas rodaram sobre o `NTupleRegressor` **denso**, que aloca `2ⁿ` por RAM
— e isso torna `n = 32` caro e `n = 64` impossível. A escolha pode ter sido
consequência silenciosa do armazenamento, não do problema.

Agora é verificável: as redes esparsas alcançam `n = 64`. Vale revarrer o
estudo de binarização da SPEC-01 com `n ∈ {8, 12, 16, 24, 32}` antes de fixar a
configuração da Etapa 1.
