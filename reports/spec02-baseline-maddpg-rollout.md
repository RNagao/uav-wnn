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
