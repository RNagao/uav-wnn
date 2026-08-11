# SPEC-01 — Estudo de binarização

**Etapa:** pré-Etapa 1 · **Depende de:** nada · **Estimativa:** 3–4 dias
**Portão de saída:** `G0` · **Referência:** PRD §5.2 (observação), §5.4 (desvios), §5.5 (custo)

---

## 1. Objetivo

Escolher, **com evidência medida**, o codificador binário que alimenta a rede n-tuple: esquema, resolução `T`, tratamento dos casos degenerados, e uma pré-seleção de `(n, N)`. Antes de treinar qualquer política.

## 2. Por que antes do treino

A rede n-tuple só enxerga bits: toda a informação passa por um endereço de RAM. Katopodis §3.2.3 é explícito — o codificador precisa **preservar similaridade**, mapeando entradas próximas para padrões com pequena distância de tupla — e dedica seções inteiras (§5.2.2, §5.3.2) ao efeito do esquema de codificação sobre o aprendizado.

O argumento prático é mais direto: **um codificador ruim é indistinguível de "o RL não converge".** Diagnosticar isso dentro do laço de treino custa dias por iteração. Fora dele, custa horas, sem simulador e sem RL.

## 3. Escopo

**Dentro:** distribuição da observação de Lamenza; esquemas de codificação candidatos; resolução; geometria da representação binária; utilização de endereços; dois proxies supervisionados; custo do codificador; sensibilidade a ruído de entrada.

**Fora:** qualquer treino de política; escolha de algoritmo de RL; o filtro de Kalman (SPEC-07). A tarefa T-01.8 adianta a interação resolução × ruído, mas **sem** filtro — a varredura com estimativa filtrada é da SPEC-08.

---

## 4. Gerador analítico de cenários

Não depende do `gradysim`. A observação (PRD §5.2, Lamenza eq. 5-1) é função pura das posições e da máscara de visitados.

**Amostragem estática:** `Ns` sensores uniformes em `[-100, 100]²`; UAVs em área pequena no centro; máscara de visitados sorteada com fração `ρ ∈ [0, 1]` para cobrir todo o ciclo do episódio.

**Rollout analítico do guloso:** `dt = 2 s`, `v = 15 m/s`, direção para o sensor não visitado mais próximo; sensor marcado como visitado quando `d ≤ communication_range`. Reproduz a dinâmica suficiente para gerar a distribuição **on-policy** das observações.

> **Invariante V-1 (fidelidade do corpus).** Quando a SPEC-00 entregar o ambiente real, comparar os histogramas de cada componente da observação entre o gerador analítico e o `gradysim` por teste de Kolmogorov–Smirnov. **Divergência significativa invalida este estudo e obriga a refazê-lo.** Sem esse teste, corre-se o risco de otimizar o codificador para a distribuição errada.

> **Dependência de P1.** `communication_range` ainda não foi confirmado (PRD §10). O rollout guloso depende dele através da dinâmica da máscara de visitados. Rodar com 2–3 valores candidatos e **reportar a sensibilidade**; se as conclusões mudarem entre eles, o estudo fica bloqueado em P1.

### Corpus

| Conjunto | Tamanho | Origem | Papel |
|---|---|---|---|
| `on-policy` | ~10⁶ observações | rollouts gulosos | é a distribuição que a política vai encontrar |
| `uniforme` | ~10⁵ observações | amostragem estática com `ρ` uniforme | cobertura; evita ajustar o codificador ao guloso |

Células: `Na ∈ {1, 2, 4}` × `Ns ∈ {12, 24, 36}`. Seeds fixas, corpus versionado e regenerável.

---

## 5. Tarefas

### T-01.1 — Caracterização da distribuição de entrada

Para cada componente escalar da observação — `(dx_j, dy_j)` de cada slot `j` de `P^s`, idem `P^a`, e o índice `k` — levantar histograma, quantis e suporte efetivo.

**Hipótese central a testar:** a distribuição depende fortemente do **índice do slot**. O sensor não visitado mais próximo (`j = 1`) concentra perto de zero; o 12º espalha por todo o quadrante. Se confirmado, **um termômetro global uniforme desperdiça a maior parte dos bits**: quase toda a massa de `j = 1` cai num punhado de bins, e a resolução onde ela importa é justamente a menor.

Métrica: entropia da distribuição quantizada, por slot, sob binning global uniforme vs. binning por slot. Saída: figura e tabela; decisão explícita sobre binning por slot.

### T-01.2 — Esquemas candidatos

| id | Esquema | Observação |
|---|---|---|
| **E0** | termômetro linear global | bins uniformes em `[-200, 200]`, iguais para todos os slots — a linha de base ingênua |
| **E1** | termômetro linear **por slot** | limites do suporte empírico de cada slot |
| **E2** | termômetro **distributivo** (quantis) por slot | bins equiprováveis segundo o corpus |
| **E3** | E2 + **polar** `(distância, ângulo)`, ângulo com codificador **circular** | ver ressalva abaixo |

Varredura: `T ∈ {4, 8, 16, 32, 64}` bits por escalar × `{com, sem}` bit de validade.

> **Ressalva sobre E3.** Converter `(dx, dy)` para `(d, θ)` é bijeção — a informação disponível ao agente é idêntica, e a conversão acontece **dentro do codificador**. Ainda assim, muda o vetor que a rede recebe e portanto **exige uma linha nova na tabela de desvios do PRD §5.4 se for adotado**. Há razão para testá-lo: o codificador circular de Katopodis existe exatamente para grandezas angulares, e sob distância de tupla a representação cartesiana degrada mal sob rotação. Há razão para desconfiar: Lamenza testou "ângulos e distâncias em vez de coordenadas" e mediu **pior** — para MLP. É uma pergunta empírica aberta e o mérito do estudo é respondê-la barato.

O **bit de validade por slot** (PRD §5.4) é obrigatório na configuração final; a ablação existe só para quantificar quanto ele vale.

### T-01.3 — Geometria: preservação de similaridade

Amostrar pares de observações `(o, o′)`; medir `d_obs = ‖o − o′‖₂` normalizada e `d_tuple` = distância de tupla entre `φ(o)` e `φ(o′)` (Katopodis eq. 3.17).

Métricas:
- **ρ de Spearman** entre `d_obs` e `d_tuple` — monotonicidade
- curva `d_obs × d_tuple` com bandas de dispersão
- **fração de pares saturados** (`d_tuple = N`, isto é, sem nenhuma RAM em comum): saturação é morte da generalização, porque duas observações saturadas são "infinitamente distantes" independentemente de quão próximas estejam de fato

**Faixa que importa:** a de `d_obs` que ocorre entre **passos consecutivos** de um episódio. É essa a faixa em que o TD faz bootstrap de `V(s′)` a partir de `V(s)`; se a codificação satura aí, o crítico não tem como propagar valor. Reportar a distribuição de `d_obs` entre passos consecutivos e sobrepô-la à curva.

### T-01.4 — Utilização de endereços

Por RAM: quantos dos `2^n` endereços são efetivamente visitados no corpus, entropia da distribuição de endereços e índice de Gini.

Varrer `n ∈ {8, 10, 12, 16}` × `N ∈ {16, 48, 128}`.

Isto **prediz T1 (ocupação de memória, PRD §5.5) antes de qualquer treino**, e identifica os dois regimes ruins: subutilização (a maioria das RAMs vê poucas dezenas de endereços — capacidade morta) e explosão (ocupação cresce sem saturar — memorização, e a alegação de "kilobytes" cai).

### T-01.5 — Proxy supervisionado A: clonagem do guloso

**É o filtro decisivo desta spec.**

Alvo: a ação da heurística gulosa `a = (θ, s)` na observação `o`. Regredir `θ` como `(cos θ, sin θ)` para evitar a descontinuidade em `2π`.

Modelo: regressor n-tuple já existente (`ntuple.py`), com capacidade idêntica em todos os esquemas comparados.

Métricas: erro angular mediano e p90 (graus); acurácia de "escolheu o mesmo sensor-alvo que o guloso".

Racional: o guloso é determinístico, simples e é o baseline que a política precisa bater. **Se a codificação não sustenta nem clonar o guloso, ela não vai sustentar RL** — e descobrir isso agora custa horas em vez de uma semana de treino.

### T-01.6 — Proxy supervisionado B: regressão de valor

Alvo: retorno Monte Carlo sob a política gulosa, com a recompensa de Lamenza (eq. 5-6) **verbatim** e `γ = 0,99`.

Métricas: RMSE normalizado e — mais informativo — **erro relativo entre estados consecutivos**, que é a quantidade de que o TD depende.

Testa o que o **crítico** vai precisar, que é função diferente da do ator: valor é suave e de longo alcance, ação é local. Um codificador pode servir a um e não ao outro.

### T-01.7 — Custo do codificador

Alimenta diretamente PRD §5.5:

- tempo de codificação por observação (mediana e p99) × `T` × esquema
- número de bits `B` resultante
- tamanho dos limiares armazenados (parte de **T1**, se o binning não for uniforme — E1/E2/E3 pagam esse preço, E0 não)

**Ponto de atenção:** se a codificação dominar o tempo de inferência, a vantagem de latência da WNN é do codificador e não da rede, e o relatório precisa dizer isso.

### T-01.8 — Resolução × ruído (opcional; adianta a Etapa 2)

Corromper a observação com `σ ∈ {0, 1, 3, 5, 10} m` **antes** de codificar; medir a degradação dos proxies A e B por `T`.

Saída: a regra *largura do bin ≈ σ* (PRD §1) instanciada sobre a distribuição real da observação, e uma tabela `T*(σ)` recomendada. Reduz o custo da SPEC-08, que é o experimento mais caro do projeto.

---

## 6. Critério de saída — portão G0

1. **Um esquema e uma faixa de `T` escolhidos**, sustentados por T-01.3 (geometria) e T-01.5 (clonagem). Escolha justificada por número, não por preferência.
2. **`(n, N)` pré-selecionados** por T-01.4, com a ocupação prevista já estimada.
3. **As duas melhores opções registradas**, não só a vencedora — ver riscos.
4. **PRD §5.4 atualizado** se E3 for adotado.
5. **Relatório curto** (2–3 páginas, com as figuras) pronto para entrar no capítulo de método de E2.
6. **Testes automatizados:** codificador determinístico; monotônico na entrada; tamanho de saída constante para configuração fixa; bit de validade correto em slots com padding; invariante V-1 registrado como teste pendente da SPEC-00.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| **Corpus da distribuição errada** (gerador analítico ≠ simulador) | invariante V-1: teste KS assim que a SPEC-00 entregar; refazer se divergir. É o risco mais caro desta spec |
| **P1 não resolvido** (`communication_range`) | rodar 2–3 valores candidatos e reportar sensibilidade; se as conclusões mudarem, escalar P1 como bloqueante |
| **O proxy não prediz o RL.** Clonar o guloso é tarefa mais fácil que aprender por reforço | registrar as **duas** melhores configurações; a SPEC-06 tem orçamento para a segunda se a primeira falhar no treino. Não tratar G0 como veredito final |
| **Sobreajuste do codificador ao guloso** | corpus uniforme além do on-policy; reportar as métricas nos dois |
| Binning por slot deixa de valer quando restam poucos sensores não visitados (slots viram padding) | avaliar as métricas estratificadas por fração de sensores restantes, não só agregadas |
| E3 (polar) vence no proxy mas é rejeitado por alterar o problema | decisão registrada antecipadamente: se vencer com margem, adota-se **com** a linha na §5.4 e a comparação cartesiana mantida como ablação no relatório |

---

## 8. Entregáveis

- `src/uavwnn/encoding/thermometer.py` — linear, distributivo, circular
- `src/uavwnn/encoding/observation.py` — observação de Lamenza (eq. 5-1) → bits, com bit de validade
- `src/uavwnn/env/analytic.py` — gerador de cenários e rollout guloso sem simulador
- `configs/spec01/` — varreduras
- `reports/spec01-binarizacao.md` + figuras
- `tests/test_encoding.py`
