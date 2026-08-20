# Baseline MADDPG completo — tempo de conclusão, trajetórias e coordenação

Fecha a terceira figura de mérito. Tamanho e latência estão em
[`spec02-baseline-maddpg.md`](spec02-baseline-maddpg.md); aqui a política
**roda** no ambiente de referência.

```bash
uv sync --group baseline
uv run --group baseline python scripts/spec02_maddpg_rollout.py --episodios 200
```

200 episódios por célula, `phase="eval"` — a mesma configuração do `test.py` do
submodule, verificada parâmetro por parâmetro. Tempo total: ~4 min para as 9
células.

---

## 1. As três figuras de mérito, todas medidas

| figura | valor | onde |
|---|---|---|
| **tamanho do modelo** | 287 / 291 / 299 KB (`Na` = 2/4/8) | `spec02-baseline-maddpg.md` |
| **tempo de inferência** | 35–36 µs mediana, 1 thread | idem |
| **tempo de conclusão** | tabela abaixo | **aqui** |
| trajetórias | `reports/img/maddpg/` | **aqui** |

## 2. Tempo de conclusão, e a validação do pipeline

| célula | medido (s) | IC95% | sucesso | MADDPG publicado | erro |
|---|---|---|---|---|---|
| `a_2-s_12` | 18,0 | [17,7; 18,5] | 100,0 % | — | — |
| `a_2-s_24` | 29,8 | [27,1; 35,0] | 99,5 % | **29,6** | **+0,7 %** |
| `a_2-s_36` | 33,4 | [32,8; 34,1] | 100,0 % | — | — |
| `a_4-s_12` | 10,2 | [10,0; 10,5] | 100,0 % | — | — |
| `a_4-s_24` | 13,5 | [13,3; 13,8] | 100,0 % | **13,8** | **−1,9 %** |
| `a_4-s_36` | 15,9 | [15,7; 16,1] | 100,0 % | — | — |
| `a_8-s_12` | **17,3** | [14,5; 22,4] | 99,5 % | — | — |
| `a_8-s_24` | 9,6 | [9,4; 9,8] | 100,0 % | **9,7** | **−1,3 %** |
| `a_8-s_36` | 12,8 | [12,5; 13,1] | 100,0 % | — | — |

**3/3 dentro de ±2 %** nas células com valor publicado disponível. Junto com a
validação 9/9 do guloso da SPEC-00, isto fecha o pipeline: submodule, versão do
simulador, configuração, semeadura e laço de avaliação estão todos certos, para
política aprendida **e** para heurística.

Os valores publicados só existem para `Ns = 24` porque foram lidos das barras das
figuras em `statistics/completion_times/` (precisão ~0,3 s). **A partir de agora
não precisamos mais ler barras** — medimos.

### A anomalia de `a_8-s_12`, e ela é real

17,3 s é **pior** que `a_8-s_24` (9,6 s) e `a_8-s_36` (12,8 s): mais sensores,
menos tempo. Não é monotônico, e o IC é o mais largo da tabela — [14,5; 22,4]
contra [9,4; 9,8] da célula vizinha.

Há explicação no repositório. A pasta de `a_8-s_12` tem carimbo de tempo
**1726583488 (set/2024)**, enquanto as outras oito têm **1723034081 (ago/2024)**:
aquela célula foi **retreinada depois**. E o `test.py` de referência ainda aponta
para `a_8-s_12-checkpoint1740` — um checkpoint de treino interrompido, que **não
existe mais no repositório**. O que sobrou é o retreino, e ele é a política mais
fraca da linha `Na = 8`.

Precisa ir para o relatório como ressalva sobre o baseline, não como resultado
nosso: é uma propriedade dos pesos publicados.

## 3. Coordenação, agora com número

A separação média entre pares de UAVs ao longo do episódio, contra um alcance de
comunicação de **20 m**:

| célula | MADDPG | `greedy` (SPEC-00) |
|---|---|---|
| `a_2-s_12` | 70,3 m | ~0,1 m |
| `a_2-s_24` | 108,2 m | ~0,1 m |
| `a_2-s_36` | 131,5 m | ~0,1 m |
| `a_4-s_12` | 57,6 m | ~0,1 m |
| `a_4-s_24` | 93,1 m | ~0,1 m |
| `a_4-s_36` | 94,0 m | ~0,1 m |
| `a_8-s_12` | 65,1 m | ~0,1 m |
| `a_8-s_24` | 69,6 m | ~0,1 m |
| `a_8-s_36` | 89,0 m | ~0,1 m |

**Duas a três ordens de grandeza de diferença.** As trajetórias em
`reports/img/maddpg/` mostram o mecanismo: os agentes partem do mesmo ponto e se
abrem em setores disjuntos, cada um cobrindo um quadrante. O `greedy` voa em
formação, todos atrás do mesmo sensor mais próximo.

Isto reforça a **TS4** (a barra da H6). "A coordenação emerge: o enxame supera a
heurística gulosa" é afirmação fraca quando o baseline não coordena — e agora
temos o contraste medido nos dois lados, não só no do baseline.

## 4. Por que não usamos o `test.py` do submodule

Ele faz quase isto e serviu de referência para o protocolo. Três razões, e a
primeira é dura:

**1. Ele não roda como está.** Os caminhos apontam para `runs/results__…` quando
os checkpoints vivem em `runs/results/results__…`, e a entrada de `a_8-s_12`
referencia um arquivo que não existe mais. O `plot_paths.py`, do mesmo diretório,
tem os caminhos certos — os dois divergiram.

**2. `from main import Actor` executa o `main.py`**, que faz `tyro.cli` no nível
de módulo e tenta configurar `wandb`. Transcrever a classe `Actor` (12 linhas) é
mais barato que contornar isso.

**3. Semeadura.** O `run_episode` do nosso `refenv` chama `random.seed` antes do
`reset()`, porque o `reset()` de referência **ignora** o argumento `seed` que
declara. Sem isso não há reprodutibilidade, e o `test.py` não faz.

A configuração é a mesma, conferida parâmetro por parâmetro:
`algorithm_iteration_interval=0.5`, `scenario_size=100`,
`max_seconds_stalled=30`, `state_num_closest_sensors=12`,
`state_num_closest_drones=Na−1`, `min_sensor_priority=1`,
`full_random_drone_position=False`, `speed_action=True`. O `test.py` **não** passa
`end_when_all_collected`, então vale o default `True` do ambiente — que é
exatamente o nosso `phase="eval"`.

## 5. Ressalvas

**1. O guloso da fig. 7.3 foi avaliado a 0,1 s por decisão, o MADDPG a 0,5 s.**
Cinco vezes mais reativo. Já estava registrado na SPEC-00 §4.3, e continua
valendo: ao confrontar as duas colunas, declarar a assimetria. O efeito favorece
o guloso, então a vantagem do MADDPG é conservadora.

**2. Não há checkpoint para `Na = 1`.** A Etapa 1 começa ali, e não há baseline
MADDPG. As referências seguem sendo a heurística e a MLP pareada.

**3. As trajetórias são de uma semente (`seed=0`) por célula.** Servem para
mostrar o mecanismo, não para quantificar — quem quantifica é a tabela de
separação, que usa o episódio inteiro.

**4. `a_2-s_24` e `a_8-s_12` têm 99,5 % de sucesso**, não 100 %: um episódio em
200 não completa a coleta. O tempo de conclusão desses episódios entra na média
como o instante em que o episódio terminou, não como censura explícita — o mesmo
tratamento que a L1 da SPEC-00 documenta.

---

## 6. Campanha de 900 runs — os números definitivos

`scripts/spec02_maddpg_campanha.py`, **100 runs × 200 episódios × 9 células**.
Cada run é uma réplica da campanha de avaliação de Lamenza (`total_runs = 200`
no `test.py`), então o desvio abaixo é **entre runs**: mede quanto o número
publicado variaria numa nova campanha.

| célula | conclusão (s) | sucesso | separação (m) | latência (µs) |
|---|---|---|---|---|
| `a_2-s_12` | 18,44 ± 0,71 | 0,9995 ± 0,0015 | 92,8 ± 13,2 | 34,8 ± 0,4 |
| `a_2-s_24` | 30,32 ± 2,37 | 0,9944 ± 0,0050 | 100,8 ± 15,4 | 35,0 ± 1,4 |
| `a_2-s_36` | 35,36 ± 2,19 | 0,9964 ± 0,0046 | 116,6 ± 19,2 | 34,7 ± 0,4 |
| `a_4-s_12` | 10,07 ± 0,13 | 1,0000 ± 0,0000 | 80,1 ± 9,2 | 34,6 ± 0,5 |
| `a_4-s_24` | 13,73 ± 0,15 | 1,0000 ± 0,0000 | 94,2 ± 6,6 | 34,5 ± 0,4 |
| `a_4-s_36` | 16,13 ± 0,14 | 1,0000 ± 0,0000 | 95,1 ± 6,7 | 34,6 ± 0,9 |
| `a_8-s_12` | **15,11 ± 0,47** | 0,9999 ± 0,0009 | 76,6 ± 9,6 | 34,6 ± 0,4 |
| `a_8-s_24` | 9,65 ± 0,08 | 1,0000 ± 0,0000 | 73,7 ± 6,2 | 34,6 ± 0,4 |
| `a_8-s_36` | 12,87 ± 0,16 | 1,0000 ± 0,0000 | 88,2 ± 5,3 | 34,7 ± 0,5 |

**Esta tabela substitui a da §2**, que vinha de uma campanha só.

### Validação: 3/3 dentro de um desvio

| célula | medido | publicado | erro | |
|---|---|---|---|---|
| `a_2-s_24` | 30,32 ± 2,37 | 29,6 | +2,4 % | dentro de 1 desvio |
| `a_4-s_24` | 13,73 ± 0,15 | 13,8 | −0,5 % | dentro de 1 desvio |
| `a_8-s_24` | 9,65 ± 0,08 | 9,7 | −0,5 % | dentro de 1 desvio |

Afirmação mais forte que o "±2 %" da §2: com 100 runs sabemos **quanto o número
deveria variar**, e a diferença contra o publicado cabe nessa variação. Não é
concordância dentro de uma tolerância arbitrária — é concordância dentro do ruído
próprio da medição.

### Achado: a linha `Na = 2` é uma ordem de grandeza menos estável

| linha | desvio entre runs |
|---|---|
| `Na = 2` | **0,71 a 2,37 s** |
| `Na = 4` | 0,13 a 0,15 s |
| `Na = 8` | 0,08 a 0,47 s |

**Até 15× de diferença.** Com dois drones a política é frágil; com quatro ou mais
é notavelmente estável. E as falhas de coleta se concentram na mesma linha: a
taxa de sucesso é exatamente 1,0000 em todas as células de `Na = 4` e cai a
0,9944 em `a_2-s_24`.

Isto tem consequência para o nosso plano: **a Etapa 1 percorre `Na = 1 → 2 → 4 →
8`, e a célula de 2 drones é a mais ruidosa do baseline.** Comparar a nossa
política ali exige mais sementes do que nas outras, ou o ruído do baseline
domina a diferença.

### A anomalia de `a_8-s_12` confirma-se — e corrige o meu número anterior

**15,11 ± 0,47 s**, contra 9,65 de `a_8-s_24` e 12,87 de `a_8-s_36`. Mais
sensores, menos tempo: não é monotônico, e agora está medido sobre 100 runs em
vez de uma campanha.

A campanha também **corrige** a estimativa da §2, que dava 17,3 s com IC
[14,5; 22,4] — largo o suficiente para conter 15,11. O valor de uma campanha só
estava alto.

A explicação segue a da §2: aquela pasta tem carimbo de **set/2024** contra
ago/2024 das outras oito, e foi retreinada depois de um treino interrompido.

### O desvio entre runs se comporta como deveria

Verificação de sanidade: se os episódios fossem independentes, o desvio entre
runs seria o desvio entre episódios dividido por `√200 ≈ 14,1`.

| célula | entre runs | entre episódios | razão |
|---|---|---|---|
| `a_4-s_24` | 0,15 s | 2,17 s | **14,2×** |
| `a_2-s_24` | 2,37 s | 29,76 s | 12,5× |

`a_4-s_24` bate na mosca. `a_2-s_24` fica um pouco abaixo, o que é o esperado de
uma distribuição de cauda mais pesada — consistente com ser a linha instável.

Isto valida a estrutura da campanha: o número que reportamos é a variabilidade da
**campanha**, não a do episódio, e as duas se relacionam como a teoria manda.

### Reprodução

```bash
uv run --group baseline python scripts/spec02_maddpg_campanha.py
uv run --group baseline python scripts/spec02_maddpg_campanha.py --so-resumo   # só agrega
```

CSVs em `reports/maddpg/campanha-a{Na}s{Ns}.csv`, uma linha por run, com colunas
que o resumo não mostra: `completion_time_std_episodios`, `coleta_media`,
`latencia_p99_us` e `wall_s`.
