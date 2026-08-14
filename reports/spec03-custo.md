# SPEC-03 — Harness de custo: T1 e T3

Configuração medida: a que a SPEC-01 escolheu — **E2 quantil, `T = 8`, `N = 48`, `n = 12`**, célula `Na = 1, Ns = 24`, `D = 25`, `B = 200` bits. Corpus de 11 526 observações do ambiente de referência.

Reprodução: `uv run python scripts/spec03_custo.py`.

---

## 1. T1 está refutada — e por uma margem grande

| modelo | erro de clonagem | persistente | residente |
|---|---|---|---|
| **MLP pareada 25-32-32-2** | **1,1°** | **7,6 KB** | 8,1 KB |
| WNN, chaves ordenadas | 5,6° | 133,9 KB | 134,1 KB |
| WNN, hash aberto | 5,6° | 273,6 KB | 273,9 KB |
| WNN, densa | 5,6° | 1 537,4 KB | 1 537,6 KB |

**A MLP é 5× mais precisa e 17,5× menor.** Não é empate nem derrota apertada.

Razão dupla (D12), variante mais econômica: **17,5× a MLP pareada** e **0,47× o ator do MADDPG** — este último ainda uma estimativa marcada, porque **P4** segue aberta (hipótese: 256×256 do CleanRL).

### O que tornou a estimativa anterior errada por uma ordem de grandeza

O PRD partia de **2 268 entradas ocupadas → 4,4 KB**, número do protótipo do `PLANO.md`. Medido aqui: **13 555 entradas**.

A causa é que aquele número veio de uma regressão de brinquedo com **2 variáveis de entrada** (`sin(3x)cos(2y)`), e a observação real tem **25**. Mais dimensões significam muito mais padrões binários distintos, logo muito mais endereços tocados. Extrapolar de um brinquedo 2-D para o problema 25-D foi a origem do erro.

### Dois cuidados que tomei para não construir espantalho

**Contra a MLP.** Meu primeiro treino usava descida simples e dava 23,3° — o que teria "provado" a superioridade da WNN. Era subtreino. Com Adam e 600 épocas a MLP converge a 1,1°, e o erro de treino iguala o de teste (1,1° / 1,1°), sem sinal de sobreajuste.

**Contra a WNN.** Dei o mesmo tratamento: varri `η ∈ {0,10; 0,25; 0,50; 0,80}` e épocas `∈ {4, 8, 20, 40}`. O erro fica em **5,6–6,5°** em toda a faixa, com platô em 5,6°. **A WNN convergiu** — não é subtreino, é o teto dela nesta tarefa e nesta configuração.

Também varri o tamanho da tupla, que é o parâmetro que mais move a memória:

| `n` | erro | ocupadas | persistente (ordenado) |
|---|---|---|---|
| 8 | 6,6° | 3 608 | **37,0 KB** |
| 10 | 5,7° | 8 341 | 83,4 KB |
| 12 | 5,7° | 13 555 | 133,9 KB |

`n = 10` dá o mesmo erro de `n = 12` com 38 % menos memória — a escolha `n = 12` da SPEC-01 não era ótima nessa dimensão, porque aquela spec otimizou precisão, não custo. Mesmo assim, **a melhor configuração da WNN (37 KB) ainda é 4,9× maior que a MLP e 6× menos precisa**.

---

## 2. A ocupação não satura

| amostras vistas | ocupadas | elasticidade `d log occ / d log t` |
|---|---|---|
| 200 | 3 908 | — |
| 1 000 | 7 938 | 0,397 |
| 4 000 | 11 675 | 0,244 |
| 8 644 | 13 555 | **0,194** |

O critério do RNF-08 é elasticidade **< 0,05**. Estamos em 0,194 e caindo devagar. **Não saturou.**

Consequência: qualquer alegação de tamanho precisa sair **com o horizonte declarado**. Extrapolando ingenuamente para os 50 M passos do orçamento de Lamenza — e é extrapolação, não medição —, a ocupação continuaria crescendo. Isso reforça a §3: a poda por esquecimento deixa de ser refinamento e passa a ser requisito, e a SPEC-04 precisa medir `occ_ε` junto com `occ`.

---

## 3. T3 sobrevive — e é condicional à classe de hardware

| operação | WNN | MLP |
|---|---|---|
| **multiplicações em ponto flutuante (aproximador)** | **0** | **1 888** |
| somas em ponto flutuante | 98 | 1 956 |
| divisões | 2 | 0 |
| transcendentais | 5 | 5 |
| comparações | 200 | 64 |
| lookups | 96 | 0 |
| somas inteiras | 576 | 0 |
| operações de bits | 1 152 | 0 |
| **total** | **2 133** | **3 917** |

**A alegação central se sustenta como zero literal:** nenhuma multiplicação em ponto flutuante no aproximador. As 4 que aparecem no total são da amostragem gaussiana, comum aos dois.

Mas o total conta outra história: **1,8× menos operações**, não ordens de grandeza. A WNN troca 1 888 multiplicações fp por 1 152 operações de bits e 576 somas inteiras. Se essa troca vale a pena depende inteiramente do alvo:

| classe de alvo | WNN | MLP | razão |
|---|---|---|---|
| x86-64 (com FPU e SIMD) | 2 642 | 4 112 | **1,56×** |
| Cortex-M4F (com FPU) | 2 954 | 12 108 | **4,10×** |
| **Cortex-M0 (sem FPU)** | 12 290 | 244 084 | **19,86×** |

*(ciclos por decisão; modelo de ciclos por ordens de grandeza de manuais de arquitetura, não medição nossa)*

**A vantagem da WNN é condicional, e o gradiente é íngreme.** Num processador com FPU ela é marginal; num microcontrolador sem FPU, onde a multiplicação em ponto flutuante é emulada em software, ela é de quase 20×.

Isso é exatamente o enquadramento que o PRD §5.5 estabelece — *"a métrica não é cumprir prazo, é que classe de hardware o modelo exige"* — agora com número. E note que o codificador responde por 200 das 2 133 operações da WNN (9 %): a vantagem é do aproximador, não do termômetro.

---

## 4. O que isto significa para a tese

**A pendência T1 pode ser fechada, com resposta negativa.** A alegação de "modelo menor" não se sustenta: a WNN é 5 a 18× maior que a MLP pareada, e menos precisa, na tarefa medida.

O que sobra é sólido mas mais estreito:

> **Aproximador sem nenhuma multiplicação em ponto flutuante, que roda ~20× mais barato que uma MLP equivalente em hardware sem FPU.**

Isso é o que o PRD §5.5 já prescrevia como recuo, e agora está medido.

### Três ressalvas que o relatório final precisa carregar

1. **A comparação é sobre clonagem supervisionada, não sobre RL.** As vantagens que a literatura atribui às redes sem peso — aprendizado online, sem *replay*, esquecimento para não-estacionariedade — não são exercitadas por regressão estacionária. Os resultados de Katopodis são em RL, não em clonagem. **Esta medição não refuta H1 no cenário de RL; refuta a alegação de tamanho, que é independente do cenário.**
2. **Custo de treino é outra figura de mérito, não medida aqui.** A MLP precisou de 600 épocas com Adam e retropropagação; a WNN, de 8 passadas de atualização direta. Se a comparação de custo incluir treino, o resultado pode mudar de sinal — e vale medir.
3. **`n = 10` domina `n = 12`** em memória sem custo de precisão. Vale revisitar a escolha da SPEC-01 com o custo na função objetivo, não só a precisão.

---

## 5. O que não foi feito nesta passagem

- **T2 (latência)** — o harness de medição com regimes de cache, 5 blocos em processos separados e separação de cauda de hash *versus* cauda de escalonador. É trabalho considerável e depende de decidir a variante de armazenamento; a SPEC-04 é quem implementa o esparso de verdade.
- **T-03.4 (quantização como ablação)** — exige política treinada para reavaliar tempo de conclusão, o que só existe depois da SPEC-06.
- **T-03.6 (paridade de arcabouço)** — a MLP já está em NumPy puro aqui; falta a coluna PyTorch e o teste de equivalência.

Nenhum deles muda o veredito de T1, que é o que esta passagem existia para resolver.
