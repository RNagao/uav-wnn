# SPEC-01 — Estudo de binarização (portão G0)

Corpus colhido do **ambiente de referência**: célula `Na=1, Ns=24`, 30 episódios por política (guloso e aleatório), ~2 300 observações. O gerador analítico e o invariante V-1 que a spec original previa foram descartados — o ambiente roda a 0,02 s por episódio, então o corpus sai dele direto.

Reprodução: `uv run python scripts/spec01_estudo.py`.

---

## 1. A distribuição depende do slot — confirmado

Hipótese central da spec: o sensor não visitado mais próximo concentra perto de um valor; o 12º espalha. Medido, amplitude p05–p95 da componente `x`:

| slot | amplitude | entropia sob binning global (T=16, máx 4,00) |
|---|---|---|
| 1 | **0,506** | **2,40** |
| 4 | 1,104 | 3,43 |
| 8 | 1,435 | 3,73 |
| 12 | **1,568** | 3,81 |

**Razão de amplitude entre o 12º e o 1º slot: 3,1×.**

O binning global desperdiça bits, e desperdiça **exatamente onde mais importa**: o slot 1 usa 2,40 dos 4,00 bits disponíveis — 60 % da resolução — e é justamente o slot que dirige a ação, porque o guloso age sobre o sensor mais próximo. Nos slots distantes, que quase não influenciam a decisão, o binning global vai bem.

Binning por quantis recupera: **3,91 bits uniformemente em todos os slots**, +0,45 bits por dimensão sobre o global.

---

## 2. Clonagem do guloso — o filtro decisivo (T-01.5)

Alvo: a ação do guloso, regredida como `(cos θ, sin θ)` para evitar a descontinuidade em 0/2π — média linear de ângulos é inválida, o mesmo problema que a SPEC-05 enfrenta no ator (decisão C1). Métrica: erro angular mediano no conjunto de teste.

**Melhor erro mediano por esquema:**

| esquema | melhor erro | onde |
|---|---|---|
| E0 — termômetro global | 7,6° | T=16, N=128 |
| E1 — linear por slot | 6,7° | T=32, N=128 |
| **E2 — quantil por slot** | **4,3°** | T=32, N=128 |

**E2 vence com folga** — quase o dobro de precisão do global. E a vantagem é maior justamente onde os recursos são escassos: com `T=8, N=16`, E2 dá **8,5°** contra **23,4°** do global, uma diferença de 2,8×.

### As duas configurações registradas (exigência do portão G0)

| # | configuração | erro mediano | bits | cobertura | entradas ocupadas |
|---|---|---|---|---|---|
| **1ª** | **E2, T=8, N=48** | 5,8° | 200 | 96 % | **11 407** |
| 2ª | E2, T=16, N=128 | 4,6° | 400 | 98 % | 39 732 |

A primeira é a recomendada: chega a 5,8° com **200 bits e 48 RAMs**, melhor que o *melhor* resultado do esquema global (7,6°), que precisa de 400 bits e 128 RAMs. A segunda troca 3,5× mais memória por 1,2° de precisão.

---

## 3. `T` e `N` não são independentes (T-01.4, achado C4)

A cobertura de bits — a fração da entrada que alguma RAM lê — cai de 100 % a **22 %** ao longo da grade. E a consequência aparece direto no erro:

| esquema | T=8, N=16 | T=16, N=16 | T=32, N=16 |
|---|---|---|---|
| cobertura | 62 % | 42 % | 22 % |
| E2, erro mediano | **8,5°** | 9,8° | 9,1° |

**Aumentar `T` com `N` fixo piorou o resultado.** Mais bits sem mais RAMs significa uma fração menor da entrada efetivamente lida — a rede ignora o resto silenciosamente. Confirma que as varreduras precisam rodar **conjuntas**, como a spec revista determina, e que `N·n/B ≳ 1` é a restrição a respeitar.

---

## 4. A geometria não discrimina — e isso é resultado (T-01.3)

A spec propunha escolher o codificador também pela preservação de similaridade: correlação entre distância na observação e distância de tupla, e fração de pares saturados (sem nenhuma RAM em comum).

Medido, o esquema global tem geometria **melhor**: Spearman 0,85–0,90 contra 0,65–0,76 do quantil, com menos saturação em pares aleatórios (23–39 % contra 38–57 %).

Isso **contradiz** o resultado da clonagem — e a contradição é aparente. O codificador por quantis espalha a massa de propósito: torna pontos próximos em valor mais distinguíveis, o que reduz a correlação monotônica global mas aumenta a discriminabilidade local. É exatamente o que se quer para uma função que varia depressa onde os dados se concentram.

**O que resolve a aparente contradição é a faixa que importa.** A spec já dizia: o que interessa é a distância entre **passos consecutivos**, porque é aí que o TD faz bootstrap de `V(s′)` a partir de `V(s)`. Medido:

| | pares aleatórios | **passos consecutivos** |
|---|---|---|
| saturação, E0 | 23–39 % | **0,5–1,1 %** |
| saturação, E2 | 38–57 % | **0,9–1,5 %** |

**Na faixa que importa, todos os esquemas saturam ~1 %** — nenhum impede o crítico de propagar valor. A métrica de geometria sobre pares aleatórios mede uma faixa que o aprendizado não usa.

Conclusão: a geometria **não é critério de decisão** aqui, e a escolha cai inteiramente sobre a clonagem. Registrar isso é mais útil que registrar o número — evita que alguém no futuro escolha o codificador errado por otimizar a métrica errada.

---

## 5. Uma ressalva que piora T1

O codificador que ganha é também o que **mais ocupa memória**. Nas mesmas `T` e `N`:

| configuração | E0 global | E2 quantil | razão |
|---|---|---|---|
| T=8, N=48 | 3 475 | 11 407 | **3,3×** |
| T=32, N=128 | 11 266 | 41 793 | **3,7×** |

O quantil espalha os dados por mais endereços — é *por isso* que ele é mais preciso, e é o mesmo motivo pelo qual ocupa mais. Como a pendência **T1** já aponta que o ator n-tuple pode sair **maior** que a MLP pareada, esta escolha empurra na direção errada da figura de mérito de tamanho.

Não muda a recomendação: 5,8° contra 7,6° é diferença grande demais para trocar por memória, e a configuração recomendada (T=8, N=48) já é a mais econômica das boas. Mas o número precisa entrar na contabilidade da §5.5 do PRD, e reforça que a alegação central deve ser **ausência de multiplicação em ponto flutuante**, não tamanho.

---

## 6. Portão G0

| Critério | Estado |
|---|---|
| Esquema e faixa de `T` escolhidos com evidência | ✅ **E2 quantil por slot**, `T = 8`, decidido pela clonagem |
| `(n, N)` pré-selecionados | ✅ `n = 12`, `N = 48` (`N·n/B = 2,88`, cobertura 96 %) |
| Duas melhores configurações registradas | ✅ E2/T=8/N=48 e E2/T=16/N=128 |
| Tabela de desvios §5.4 atualizada | ✅ **sem mudança** — E3 (polar) não foi adotado, e o padding `-1` dispensa o bit de validade |
| Relatório | este documento |
| Testes | `tests/test_encoding.py` |

**O que não foi feito:** o esquema **E3 (polar com codificador circular)** e o proxy de regressão de valor (T-01.6). O E3 exigiria linha nova na tabela de desvios e o ganho do E2 já é grande; o T-01.6 testa o que o **crítico** precisa, que é função diferente da do ator, e cabe melhor junto da SPEC-05, quando o crítico existir. Ambos ficam registrados como opcionais, não como pendências.
