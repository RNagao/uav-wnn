# SPEC-02 — Baselines não aprendidos e harness de avaliação

**Etapa:** 1 (infra) · **Depende de:** SPEC-00 (ambiente) · **Estimativa:** 2 dias
**Alimenta:** `G1` (SPEC-06), `G3`/H6 (SPEC-09), toda tabela de desempenho de E2
**Referência:** PRD §5.1, §5.2, §6.1, §6.4, RNF-06 · Lamenza §3, §5.1–5.3, §7.1–7.2, tab. 3.1 e 7.1

---

## 0. Revisão após a SPEC-00 — os baselines já existem

Esta spec foi escrita para **implementar** os baselines. O repositório de referência já os traz, e eles são os que geraram a fig. 7.3:

| Baseline | Onde | Situação |
|---|---|---|
| Guloso (sensor não visitado mais próximo) | `heuristics.py::create_greedy_heuristics` | **usar verbatim** — reimplementar reintroduz o risco que a SPEC-00 eliminou |
| Aleatório | `heuristics.py::create_random_heuristics` | usar verbatim |
| `smart` (atribuição gulosa entre agentes) | `heuristics.py::create_smart_heuristics` | **brinde**: é coordenação sem aprendizado, exatamente o baseline 3 que esta spec projetava construir |
| k-means + TSP | não existe | **continua sendo nosso** — é o teto quase-ótimo |

Perguntas fechadas pelo código: o guloso usa velocidade máxima (`1` normalizado) e **recomputa o alvo a cada passo** (Q3 e Q4 desta spec).

**Validação já feita.** O guloso de referência reproduz a fig. 7.3 em **9/9 células**, erro mediano 1,7 %, viés médio +0,5 %. Ver `reports/spec00-parametros.md` §8 e `tests/test_refenv.py`.

**Ressalva que esta spec precisa carregar.** Em `visualize_performance.py` o MADDPG é avaliado a `interval = 0,5 s` e o guloso a `0,1 s`. Medido: o guloso a 0,1 s completa em 39,1 s contra 43,5 s a 0,5 s — **~10 %**. Ao comparar nossa política com o guloso, ambos devem rodar no **mesmo** `interval`; ao citar os números publicados, a assimetria deve ser declarada.

O escopo desta spec encolhe para: **k-means+TSP**, o harness de avaliação com IC pareado, e a decisão sobre agregação de episódios sem sucesso.

---

## 1. Objetivo

Entregar (a) os três baselines não aprendidos do PRD §6.1 como políticas executáveis dentro do mesmo ambiente, (b) a interface `Policy` que baselines e políticas aprendidas implementam sem que o avaliador saiba qual é qual, e (c) o harness que roda 500 episódios de avaliação **pareados por semente** e produz números com intervalo de confiança, incluindo o **IC da diferença pareada**, que é a estatística sobre a qual H6 e o portão G1 são decididos.

Ao final desta spec, nenhuma política aprendida existe ainda — e mesmo assim a tabela de resultados do relatório já está preenchida nas linhas `aleatório`, `guloso` e `k-means+TSP`, com IC, para cada célula `Na × Ns`.

## 2. Por que antes de qualquer treino

Três razões, em ordem de importância.

**O baseline é o denominador do trabalho inteiro.** H6 diz "o enxame supera a heurística gulosa". O portão G1 diz "1b supera a caminhada aleatória decisivamente e fica dentro do IC da heurística gulosa". Nenhuma dessas frases tem significado antes de o guloso estar medido, com IC, nas mesmas células. Medir o baseline depois da política é como escolher o denominador depois de ver o numerador.

**O harness é um instrumento e instrumentos se calibram com sinal conhecido.** As três políticas desta spec têm ordenação esperada conhecida *a priori* — `LB-1 ≤ k-means+TSP ≤ guloso ≤ aleatório` em tempo de conclusão. Se o harness não reproduzir essa ordenação, o defeito é do harness (sinal trocado, censura mal tratada, semeadura vazando) e não da política. Descobrir isso com uma política aprendida no meio é caro: um erro de sinal na conversão de coordenada relativa para rumo produz um agente que **foge** dos sensores, o que é indistinguível de "o RL não aprendeu".

**O harness é o único lugar onde a comparação com Lamenza acontece.** Lamenza reporta tempo de conclusão médio sobre 500 episódios de avaliação (§7.2) e taxa de sucesso (§7.1, eq. 7-1). Se o nosso protocolo de censura, de imputação e de pareamento não estiver escrito antes de rodar, ele será escolhido depois de ver os resultados — e passa a ser um grau de liberdade do pesquisador.

## 3. Escopo

**Dentro:** interface `Policy` e contrato de execução de episódio; as três políticas não aprendidas; um limite inferior analítico válido; métricas por episódio com política explícita de censura e imputação; semeadura pareada e disjunção treino/avaliação; bootstrap pareado e IC da diferença; esquema dos artefatos de resultado; testes de sanidade do harness, inclusive um teste A/A.

**Fora:** qualquer aproximador ou laço de treino (SPEC-04, SPEC-05); medição de tamanho de modelo, latência e contagem de operações — o harness só expõe os **ganchos** de instrumentação, o protocolo é da SPEC-03; ruído e filtro (SPEC-07); a execução das varreduras da Etapa 1 (SPEC-06). Esta spec entrega capacidade e três linhas de tabela, não o experimento.

**Explicitamente fora:** reproduzir MADDPG (PRD D7). Os números da fig. 7.3 de Lamenza entram no relatório como **anotação de referência de literatura**, nunca como linha da tabela estatística — ver Perguntas em aberto Q12.

---

## 4. Contratos de módulo

### T-02.1 — Interface `Policy` e contrato de execução

O harness não pode distinguir baseline de política aprendida. Contrato mínimo, em `src/uavwnn/agents/base.py`:

```python
@dataclass(frozen=True)
class Observacao:
    vetor: np.ndarray        # eq. 5-1 achatada, exatamente como Lamenza: 2(Na-1) + 24 + 1
    p_a: np.ndarray          # (N*_a, 2) coords relativas dos agentes, ordem crescente de distância
    p_s: np.ndarray          # (12, 2)   coords relativas dos sensores NÃO visitados, idem
    valido_s: np.ndarray     # (12,) bool — False onde há zero-padding
    k: int                   # índice do agente
    t: float                 # instante simulado, em segundos

@dataclass(frozen=True)
class Acao:
    theta_norm: float        # [0,1] -> θ ∈ [0, 2π]
    s_norm: float            # [0,1] -> s ∈ [0, 15] m/s

class Policy(Protocol):
    id: str                          # identificador estável, vai para a tabela de resultados
    usa_estado_verdadeiro: bool      # False para tudo que é candidato a implantação
    def reset(self, ctx: ContextoEpisodio, rng: np.random.Generator) -> None: ...
    def act(self, obs: Observacao) -> Acao: ...
    def metadados(self) -> dict[str, Any]: ...   # hiperparâmetros, vira coluna/manifesto

class PoliticaPrivilegiada(Policy, Protocol):
    def observar_verdade(self, estado: EstadoVerdadeiro) -> None: ...
```

Regras do contrato, todas com teste:

1. **A ação é sempre normalizada.** `[0,1]²`, como Lamenza §5.2. A desnormalização é do ambiente, não da política. Isso mantém a mesma moeda entre baselines e política gaussiana.
2. **`vetor` é canônico e é o que a MLP pareada consome.** `p_a`, `p_s` e `valido_s` são vistas convenientes do mesmo conteúdo. `valido_s` **não** é informação nova — é a convenção de padding tornada explícita — mas seu consumo é regulado: o codificador da WNN pode usá-la (PRD §5.4, bit de validade); a **MLP pareada não pode**, sob pena de deixar de ser fiel à entrada de Lamenza. Teste: a MLP recebe apenas `vetor`.
3. **Privilégio é declarado, não inferido.** O runner só chama `observar_verdade` se `usa_estado_verdadeiro is True`. Teste: para toda política com a flag `False`, `EstadoVerdadeiro` nunca é passado — é o mesmo invariante de RF-04 (PRD §6.2), antecipado aqui e reaproveitado na Etapa 2.
4. **Avaliação é determinística por padrão.** Lamenza §7 é explícito: "During an evaluation the training is temporarily halted and action noise is removed". O harness chama a política em modo determinístico (média da gaussiana, sem ruído de exploração). Políticas estocásticas por natureza — a caminhada aleatória — declaram `deterministica = False` nos metadados e continuam consumindo o `rng` que recebem no `reset`.
5. **`reset` recebe o `rng`; a política nunca cria o seu.** Fonte única de aleatoriedade, derivada pelo harness (T-02.6).

### Contrato requisitado à SPEC-00

Esta spec depende de a SPEC-00 expor, no mínimo:

```python
def rodar_episodio(politica: Policy, ctx: ContextoEpisodio, seed_cenario: int,
                   gravar_eventos: bool = True) -> ResultadoEpisodio
```

com `ResultadoEpisodio` contendo: instante de cada evento de visita e qual agente o causou; posição inicial dos UAVs e posições dos sensores; motivo de término; número de decisões por agente; distância percorrida por agente. O harness não sabe que o simulador é orientado a eventos e não deve saber (PRD §7).

Além disso, a SPEC-00 precisa **fixar e documentar a regra de fronteira** (o que acontece quando o UAV atinge `|x| = scenario_size`): a caminhada aleatória é o método mais sensível a essa escolha e a dissertação não a especifica — ver Q2.

---

## 5. Baselines

Todas as três vivem em `src/uavwnn/agents/heuristics.py` e são `Policy` comuns. Nenhuma delas tem estado global compartilhado entre agentes, exceto a do §T-02.4, que é declaradamente centralizada e privilegiada.

### T-02.2 — Baseline 1: caminhada aleatória (piso)

A pergunta que este baseline responde é estreita e precisa ser dita: **quanto do tempo de conclusão é explicado por simplesmente se mover numa arena limitada de 200 × 200 m com sensores uniformes?** Ele não é um competidor; é o valor abaixo do qual qualquer resultado é evidência de que nada foi aprendido.

"Aleatória" é ambíguo e a ambiguidade muda o número. Três variantes, todas implementadas, com uma designada canônica:

| id | Rumo `θ` | Velocidade `s` | Papel |
|---|---|---|---|
| `random_uniforme` | `U[0, 2π)` reamostrado a cada passo de decisão | `U[0, 15]` m/s reamostrada a cada passo | **canônica**: é a distribuição uniforme sobre o espaço de ação, sem parâmetro livre — exatamente o que uma política gaussiana não treinada com σ grande produz |
| `random_vmax` | `U[0, 2π)` por passo | fixa em 15 m/s | isola rumo de velocidade; é o piso *mais forte* dos três em quase todo cenário |
| `random_persistente(L)` | mantido com probabilidade `1 − 1/L`, reamostrado com `1/L`; `L ∈ {5, 15}` passos | fixa em 15 m/s | caminhada correlacionada: balística no curto prazo, cobre mais área por unidade de tempo |

Decisões e seus porquês:

- **Reamostragem por passo de decisão**, não por segundo simulado. O passo é `algorithm_iteration_interval = 2 s` (tab. 7.1), o mesmo de qualquer política. Reamostrar mais rápido daria à caminhada aleatória uma frequência de controle que a política aprendida não tem.
- **Sem persistência na variante canônica.** Com `dt = 2 s` e `v = 15 m/s`, cada passo é um deslocamento de até 30 m numa arena de diagonal ≈ 283 m; `max_episode_length = 500 s` dá 250 decisões por agente. Persistência é o parâmetro que mais muda o resultado e é arbitrário — daí ficar fora da canônica e entrar como sensibilidade.
- **O piso reportado no portão G1 é o melhor dos três**, não o canônico. Bater a variante mais fraca não prova nada; o critério "supera a caminhada aleatória decisivamente" só é informativo contra a versão mais forte. A tabela principal traz a canônica; a linha de portão traz `min` sobre as variantes, identificando qual venceu.
- **Sem viés de retorno ao centro, sem repulsão de fronteira, sem memória.** Qualquer um desses já é uma heurística, não um piso.

### T-02.3 — Baseline 2: guloso pelo sensor não visitado mais próximo

É o baseline do próprio Lamenza (§7.2) e é contra ele que H6 se mede. Fidelidade é o requisito, não qualidade.

O texto de Lamenza: *"a naive heuristic was developed. This heuristic is greedy, with agents always traveling towards the closest unvisited sensor. This strategy means that agents do not split the work between them; they work independently instead of collaborating."*

Especificação operacional:

1. A cada passo de decisão, o agente `k` lê o **slot 1** de `P^s` — que já é, por construção da observação (eq. 5-5), o sensor não visitado mais próximo dele.
2. Converte para rumo. **Atenção ao sinal**: a coordenada relativa de Lamenza é `Δ = (x_k − x_i, y_k − y_i)`, isto é, o vetor **do sensor para o agente**. O rumo de aproximação é
   ```
   θ = atan2(−Δy₁, −Δx₁)  mod 2π
   ```
   Um sinal invertido aqui produz um agente que foge dos sensores, com taxa de sucesso próxima de zero — plausível o bastante para ser confundido com "o baseline é ruim mesmo". Teste dedicado: em cenário com um único sensor, a distância ao sensor decresce monotonicamente.
3. Velocidade `s = 15 m/s` sempre. **Suposição A-02.1**: Lamenza não declara a velocidade da heurística; "always traveling towards" lida naturalmente como velocidade máxima, e qualquer valor menor produz uma heurística estritamente pior, o que enfraqueceria artificialmente o baseline que precisamos bater. Registrada como suposição, não como fato.
4. **Sem divisão de trabalho, sem reivindicação, sem desempate coordenado.** Dois agentes podem perseguir o mesmo sensor; é assim no original e é exatamente o comportamento que a fig. 7.1 de Lamenza documenta.
5. **Reativa, não comprometida.** O alvo é recomputado a cada passo; se outro agente coleta o alvo, o agente redireciona no passo seguinte. Uma variante `guloso_comprometido` (mantém o alvo até ser visitado) é implementada e reportada como ablação, mas **não** é o baseline — a leitura literal do texto é a reativa.
6. **Desempate determinístico** por menor índice de sensor. Evento de medida nula com posições contínuas, mas necessário para reprodutibilidade bit a bit.
7. **Caso degenerado**: se nenhum slot de `P^s` é válido (nenhum sensor não visitado), a política devolve `s = 0`. Só ocorre depois do sucesso do episódio.
8. **A heurística consome a mesma `Observacao` que as políticas aprendidas.** Consequência importante para as Etapas 2 e 3: quando a observação passar a ser ruidosa (SPEC-07), o guloso degrada junto, e continua sendo o baseline pareado correto. Uma variante `guloso_oraculo` com `usa_estado_verdadeiro = True` fica registrada como teto do degrau de ruído.

### T-02.4 — Baseline 3: k-means + TSP por UAV, centralizado e offline (referência forte)

Acréscimo nosso. Não altera o problema porque **não participa do aprendizado, não entra na observação e não é candidato a implantação** — é uma régua. Mas precisa ser rotulada com precisão, e é aqui que a spec toma a decisão que o PRD deixou em aberto.

**Decisão D-02.1 — é uma política executável, não um cálculo analítico.** Implementada como `PoliticaPrivilegiada` que planeja uma vez, em `reset`, a partir do estado verdadeiro, e depois **executa waypoints** dentro do mesmo simulador, com o mesmo `dt = 2 s`, a mesma coleta automática por proximidade e as mesmas regras de término.

Razões, e elas são fortes:

- Um tempo calculado analiticamente teria de modelar a discretização de 2 s (o rumo só muda a cada passo, logo há *overshoot* em cada waypoint), o corte antecipado por `communication_range` (não se chega ao sensor, chega-se ao disco) e o `max_episode_length`. Cada uma dessas seria uma aproximação **diferente** das que o simulador aplica às outras políticas, e a comparação passaria a misturar planejador com modelo de execução.
- Executando, o número sai do mesmo cano que os demais: mesma censura, mesmo bootstrap, mesma tabela.
- E fica disponível de graça um diagnóstico valioso: a razão entre o **tempo planejado** (analítico, T-02.4c) e o **tempo realizado** (simulado). A diferença é exatamente o custo da discretização temporal e do executor. Se essa razão for grande, o planejador está bom e o executor é o gargalo — informação que muda a leitura da tabela.

**Consequência a declarar sem rodeios:** por ser executável e por usar informação privilegiada, esta linha **não é um limite inferior e não é ótima**. O particionamento por k-means minimiza inércia, não *makespan*; o problema alvo (mTSP min-max com alcance de coleta) é NP-difícil e não está sendo resolvido. A rotulação correta no relatório é **"planejador centralizado offline com posições verdadeiras"**, e a coluna `usa_estado_verdadeiro` carrega isso na própria tabela. Para não ficar sem nenhum limite honesto, ver T-02.4d.

**T-02.4a — Particionamento.** k-means sobre as posições **verdadeiras** dos `Ns` sensores, `k = Na`, inicialização k-means++, `R` reinícios com escolha pela menor inércia, `rng` derivado da semente do episódio. Cluster vazio (possível com `Na` grande e `Ns` pequeno) recebe o sensor mais distante do seu centroide. Atribuição cluster → UAV pelo problema de designação (Hungarian) minimizando a soma das distâncias das posições iniciais aos centroides; como os UAVs partem todos de uma área pequena no centro (Lamenza §3), a designação quase não importa, mas precisa ser determinística.

**T-02.4b — Rota por partição.** Caminho aberto (sem retorno), origem fixa na posição inicial do UAV.

- `m ≤ 12` sensores na partição: **Held–Karp exato** (`2^m · m²` ≈ 5,9 × 10⁵ operações no pior caso, desprezível).
- `m > 12`: vizinho mais próximo + **2-opt** + Or-opt até convergir, com reinícios semeados.
- Nas células em que ambos são viáveis, medir e reportar o **gap** do heurístico contra o exato. Sem isso, "TSP" na tabela é uma palavra, não um número.

**T-02.4c — Execução e tempo planejado.** O executor mantém, por agente, um índice de waypoint; a cada decisão aponta para o waypoint corrente a `v = 15 m/s`; **pula** waypoints já marcados como visitados (inclusive os coletados incidentalmente por outro UAV — isso é pular, não replanejar); termina com `s = 0` quando a rota acaba. O tempo planejado, só para diagnóstico, é

```
T_plan(k) = Σ_j max(0, d(w_{j-1}, w_j) − communication_range) / v_max
```

com a ressalva registrada de que essa fórmula aproxima o ponto de partida da perna seguinte pelo waypoint, e não pelo ponto do disco onde a coleta de fato ocorreu. É aproximação, é diagnóstico, e o número autoritativo é sempre o simulado. Variante opcional `kmeans_tsp_replan` (replanejar quando um waypoint é coletado por outro agente) fica na linha de corte.

**T-02.4d — LB-1, um limite inferior de fato válido.** Barato e demonstrável: todo sensor `i` precisa ficar a menos de `communication_range` de algum agente, e nenhum agente anda mais rápido que `v_max`, portanto

```
T ≥ LB-1 = max_i  min_k  ( ‖p_i − p_k^0‖ − communication_range )₊ / v_max
```

É frouxo (ignora que um agente não pode estar em dois lugares), mas é **válido**, custa microssegundos e vai para a tabela como coluna própria. Reportar também a razão `k-means+TSP / LB-1` como medida de quão frouxo o limite é nesta geometria. Um resultado abaixo de LB-1 é bug, não descoberta — asserção no harness.

---

## 6. Harness de avaliação

Vive em `src/uavwnn/eval/harness.py`, `metrics.py` e `stats.py`.

### T-02.5 — Métricas por episódio, censura e imputação

Duas métricas de desempenho, ambas de Lamenza (PRD §6.4):

- **Taxa de sucesso** — fração de episódios com `Σ_i visited_i^T = Ns` (eq. 7-1).
- **Tempo de conclusão** — instante do evento de visita que completa o conjunto.

O problema que ninguém pode deixar implícito: **o que é o tempo de conclusão de um episódio que falhou?** Com `max_seconds_stalled = 30 s`, a caminhada aleatória vai falhar na esmagadora maioria dos episódios, e a média sobre os que ela conseguiu completar é a média sobre os cenários mais fáceis — viés de sobrevivência que faz o piso parecer bom. Pior: se a imputação usar o instante em que o episódio foi podado, uma política que trava cedo é **premiada** com tempo baixo.

**Decisão D-02.2 — a métrica primária é o tempo de conclusão com imputação em `max_episode_length`:**

```
t_conc_pen = t_conc            se sucesso
             max_episode_length  caso contrário   (= 500 s, tab. 7.1)
```

É monótona nas duas dimensões que importam (falhar é sempre pior que concluir; concluir mais cedo é sempre melhor), não é manipulável por travar cedo, e é um único escalar sobre o qual o bootstrap opera sem tratamento especial. É o análogo do escore PAR-1 usado em competições com limite de tempo. A variante PAR-2 (`2 × max_episode_length`) fica registrada e não usada.

Reportadas **sempre junto**, porque a primária sozinha esconde o mecanismo:

| Métrica | Definição | Papel |
|---|---|---|
| `t_conc_pen` | acima | **primária**, é sobre ela que o IC da diferença decide G1 e H6 |
| `taxa_sucesso` | eq. 7-1 | métrica de Lamenza §7.1; primária na réplica 1a |
| `t_conc_sucesso` | média só sobre episódios bem-sucedidos, com `n` explícito | comparável com a leitura mais provável da fig. 7.3; **inválida entre métodos com taxas de sucesso diferentes** — reportada com esse aviso preso a ela |
| `t_conc_comum` | média restrita aos episódios em que **todos** os métodos comparados tiveram sucesso | secundária pareada, sem viés de sobrevivência; degenera (conjunto vazio) quando a caminhada aleatória entra na comparação, e nesse caso não é reportada |
| mediana e p90 de `t_conc_pen` | — | a distribuição é assimétrica e censurada; a média sozinha engana |

Diagnósticos por episódio, baratos e que sustentam a leitura de coordenação de H6: `n_visitados` final, `motivo_fim ∈ {sucesso, stalled, timeout}`, distância percorrida por agente, **visitas por agente** (Gini sobre a distribuição mede divisão de trabalho) e **coletas redundantes** (sensores coletados com mais de um UAV no alcance simultaneamente — é a assinatura quantitativa do "todos vão para o mesmo sensor" da fig. 7.1).

**Nota de fidelidade:** RF-02 (não podar quando todos os sensores são coletados) é uma característica do **treino**, medida por Lamenza §5. Na avaliação, o instante registrado é o da última visita; se o episódio é encerrado logo depois, isso não afeta métrica nenhuma. O `max_seconds_stalled` **continua ativo na avaliação** por ser parâmetro do cenário (tab. 3.1 e 7.1), e com a imputação D-02.2 a poda não introduz viés. Ver Q7.

### T-02.6 — Semeadura pareada

O ponto central desta spec, e o que faz 500 episódios renderem muito mais do que 500 episódios não pareados.

**Todos os métodos são avaliados exatamente nos mesmos 500 cenários.** O cenário é função determinística de `seed_cenario`: posições dos `Ns` sensores, posições iniciais dos `Na` UAVs. Como o guloso e o k-means+TSP são determinísticos e a variância entre episódios é dominada pela geometria do cenário — não pela política —, o pareamento remove a maior parte da variância da **diferença**. Sem pareamento, a mesma conclusão exigiria ordens de grandeza mais episódios.

Regras:

1. **Árvore de sementes explícita.** Raiz fixa no config; `numpy.random.SeedSequence(raiz).spawn(...)` gera ramos nomeados e disjuntos: `treino`, `avaliacao`, `bootstrap`, `politica`. Teste automatizado: as listas de `seed_cenario` de `treino` e `avaliacao` são **disjuntas**. Avaliar num cenário visto no treino é vazamento silencioso e é fácil de cometer com `seed = i`.
2. **Números aleatórios comuns só fazem sentido no fluxo do cenário.** A semente de política é derivada como `spawn_key = (raiz_politica, hash(policy.id), idx_episodio)` — determinística e reprodutível, porém **independente** entre métodos. Tentar parear o fluxo de aleatoriedade da política é ilusão: métodos diferentes consomem números aleatórios em quantidades diferentes, e após o primeiro descasamento não há correspondência nenhuma.
3. **O pareamento é dentro da célula.** `Na × Ns` diferentes geram cenários diferentes a partir da mesma semente; comparações entre células nunca são pareadas e a tabela precisa deixar isso claro.
4. **Na Etapa 2, o fluxo de ruído tem semente própria pareada por episódio.** Mesma semente de ruído entre métodos não produz as mesmas realizações (o ruído depende da trajetória), mas mantém a comparação reprodutível e reduz variância de nuisance. Documentar essa limitação em vez de sugerir um pareamento que não existe.
5. **A incerteza que o IC mede é a do cenário, não a do treino.** Um IC bootstrap sobre 500 episódios de **um** modelo treinado não diz nada sobre a variabilidade entre execuções de treino. O esquema de resultados já carrega a coluna `seed_treino` e a SPEC-06 deve rodar ≥ 3 sementes de treino onde o orçamento permitir, reportando a dispersão entre elas **separadamente** do IC. Omitir isso é o erro mais comum na literatura de RL e não vamos cometê-lo por acidente.

### T-02.7 — Estatística: bootstrap pareado e IC da diferença

**Agregado marginal.** Média com IC bootstrap de 95 % por percentis, `B = 1000` reamostragens (PRD §6.4, seguindo Katopodis §5.4), semente de bootstrap fixa e registrada.

**Diferença pareada — e é isto que decide os portões.** Para cada par de métodos (A, B) na mesma célula:

1. Sortear **uma única vez por réplica** um conjunto de índices de episódio com reposição, e aplicar **o mesmo conjunto de índices aos dois métodos**. Reamostrar independentemente destrói o pareamento e infla o IC.
2. Estatística da réplica: `Δ_b = média_b(A) − média_b(B)`. IC de `Δ` por percentis de `{Δ_b}`.
3. Regra de decisão pré-registrada: a diferença é declarada quando **`0 ∉ IC₉₅(Δ)`**.

> **Aviso metodológico que vai também para o relatório:** ICs marginais que **se sobrepõem NÃO implicam ausência de diferença significativa**. A implicação vale só numa direção — ICs que **não** se sobrepõem implicam diferença. Com dados pareados a distância entre as duas leituras é grande, porque a correlação entre métodos no mesmo cenário é alta. Toda afirmação comparativa deste projeto se sustenta no IC da diferença pareada; ICs marginais existem para descrever cada método, não para compará-los.

**Acompanhamentos obrigatórios**, porque a distribuição é assimétrica, censurada e o bootstrap da média é sensível a isso:

- **Taxa de vitória** de A sobre B: fração dos 500 episódios em que `t_conc_pen(A) < t_conc_pen(B)`, com IC. É interpretável mesmo quando a média não é.
- **Wilcoxon dos postos sinalizados** sobre as diferenças pareadas (livre de distribuição) e **McNemar** para a diferença de taxa de sucesso (binária pareada). Reportados como apoio, não como critério primário — o critério é o IC.
- **MDE (efeito mínimo detectável)**, coluna obrigatória: `1,96 · σ̂_d / √500`, com `σ̂_d` o desvio padrão da diferença pareada. Sem ela, um resultado nulo é ininterpretável — "não achamos diferença" precisa vir com "e detectaríamos qualquer diferença acima de X segundos".

**Multiplicidade.** As comparações primárias são poucas e pré-registradas: `política vs guloso` (H6/G1) e `WNN vs MLP pareada` (H1). Tudo mais — varreduras `T × n × N`, células adicionais — é exploratório e assim rotulado na tabela. Correção de Holm disponível como coluna opcional nas tabelas de varredura; nunca aplicada retroativamente às primárias.

**Ressalva sobre `B = 1000`.** Com 1000 reamostragens, cada cauda de 2,5 % é estimada por ~25 réplicas, o que torna os extremos do IC ruidosos na terceira casa. Mantemos 1000 como canônico por seguir o protocolo de referência; `B = 10 000` roda no apêndice onde for barato, e a diferença entre os dois é reportada uma vez, para mostrar que a conclusão não depende disso.

### T-02.8 — Artefatos de resultado

Três arquivos por experimento, em `results/<experimento>/`, com formatos escolhidos por função e não por gosto.

**`episodios.parquet`** — grão: um episódio × método × célula (formato longo). É dado de máquina, cresce, e não vai para o git.

| Coluna | Tipo | Nota |
|---|---|---|
| `experimento`, `celula` | str | `celula` = `"Na4_Ns24"` |
| `Na`, `Ns` | int | |
| `communication_range`, `dt`, `v_max`, `max_episode_length`, `max_seconds_stalled` | float | parâmetros do cenário, replicados por linha — a tabela precisa ser autossuficiente (RNF-06) |
| `politica`, `variante` | str | `politica` = `Policy.id` |
| `config_hash` | str | hash do config resolvido |
| `idx_episodio` | int | 0..499 |
| `seed_cenario`, `seed_politica`, `seed_treino` | uint64 | `seed_treino` nula para baselines |
| `sucesso` | bool | eq. 7-1 |
| `n_visitados` | int | |
| `t_conc_s` | float | NaN se falhou |
| `t_conc_pen_s` | float | métrica primária |
| `t_fim_s`, `motivo_fim` | float, categoria | `sucesso` / `stalled` / `timeout` |
| `n_decisoes` | int | |
| `dist_por_agente_m`, `visitas_por_agente` | list<float>, list<int> | diagnósticos de divisão de trabalho |
| `coletas_redundantes` | int | |
| `lb1_s`, `t_plan_s` | float | LB-1 sempre; `t_plan_s` só para `kmeans_tsp` |
| `usa_estado_verdadeiro` | bool | leitura direta do privilégio na tabela |
| `sigma_gps`, `degrau_fusao` | float, str | NaN / `"n/a"` na Etapa 1; colunas já existem para as Etapas 2–3 não mudarem o esquema |
| `git_commit`, `versao_env`, `timestamp_utc` | str | |

**`agregado.csv`** — grão: método × célula. Pequeno, legível, **versionado no git**, de modo que `make repro` produza um diff inspecionável a olho. Colunas: `celula, politica, variante, n_ep, media_t_conc_pen, ic_lo, ic_hi, mediana, p90, taxa_sucesso, ts_ic_lo, ts_ic_hi, n_sucesso, media_t_conc_sucesso, media_lb1, B, seed_bootstrap`.

**`comparacoes.csv`** — grão: par (A, B) × célula. Também versionado. Colunas: `celula, politica_a, politica_b, subconjunto, n_pareado, delta_media, delta_ic_lo, delta_ic_hi, contem_zero, mde_95, taxa_vitoria_a, tv_ic_lo, tv_ic_hi, p_wilcoxon, delta_taxa_sucesso, dts_ic_lo, dts_ic_hi, p_mcnemar, primaria` (bool), `holm_ajustado`. `subconjunto ∈ {todos, sucesso_comum}`.

**`manifest.json`** — sementes-raiz e ramos, config resolvido, hashes, commit do git, versões de pacotes, hardware declarado (RNF-13), modo de execução, tempo de parede, contagem de episódios efetivamente rodados.

**Determinismo.** Rodar duas vezes a mesma célula produz `episodios.parquet` idêntico exceto pelas colunas de tempo de parede. Por isso, **nenhuma medição de latência entra neste arquivo** — ela vive em `custo.parquet` sob protocolo da SPEC-03. Teste: hash das colunas determinísticas estável entre execuções e entre `--modo desempenho` e `--modo custo`.

**Dois modos de execução, e a distinção é obrigatória.** `--modo desempenho` paraleliza sobre episódios (independentes) e não mede tempo; `--modo custo` roda serial, em um núcleo, sem carga concorrente, e é o único que produz números de latência (RNF-13). Misturá-los contamina a medição da SPEC-03.

CLI: `python -m uavwnn.eval.harness --config configs/spec02/etapa1_na4_ns12.yaml --politicas random_uniforme,guloso,kmeans_tsp --n-episodios 500 --raiz 20260809 --modo desempenho`.

### T-02.9 — Validação do harness

O harness é instrumento; instrumento se calibra antes de medir.

1. **Ordenação esperada.** Em todas as células: `LB-1 ≤ kmeans_tsp ≤ guloso ≤ random_*` em `t_conc_pen`, com as comparações adjacentes tendo `0 ∉ IC(Δ)`. Violação de `LB-1` é asserção dura (bug). Violação de `kmeans_tsp ≤ guloso` é resultado a investigar antes de prosseguir — provável falha do executor de waypoints, não do planejador; o par (`t_plan_s`, `t_conc_s`) diz qual dos dois.
2. **Teste A/A.** Avaliar `random_uniforme` contra ela mesma com sementes de política diferentes, repetido em ~200 execuções: o IC da diferença pareada deve conter 0 em aproximadamente 95 % delas. É o único teste que verifica de fato a cobertura do bootstrap pareado; um erro de reamostragem independente aparece aqui como cobertura errada e em nenhum outro lugar.
3. **Teste de sinal.** Cenário sintético de um sensor: sob `guloso`, a distância ao sensor decresce monotonicamente. Pega a inversão de sinal do §T-02.3.
4. **Teste de privilégio.** Nenhuma política com `usa_estado_verdadeiro = False` recebe `EstadoVerdadeiro` (antecipa RF-04).
5. **Teste de disjunção de sementes** treino/avaliação.
6. **Teste de degenerescências:** `Ns < N*_s` (padding total), `Na > Ns` (cluster vazio no k-means), episódio que conclui no primeiro passo, episódio com zero visitas.
7. **Sensibilidade a `communication_range`** (P1): rodar as três políticas em 2–3 valores candidatos e reportar como as conclusões se movem. Se a ordenação mudar entre candidatos, P1 vira bloqueante para a comparação numérica com Lamenza.

### T-02.10 — Relatório curto

2–3 páginas com: tabela de baselines por célula (`Na ∈ {1,2,4}` × `Ns = 12`, mais `Ns ∈ {24,36}` conforme o tempo), figura de tempo de conclusão por célula no mesmo formato da fig. 7.3 de Lamenza (barras por `Ns`, sub-figuras por `Na`) já com barras de IC, distribuição de `t_conc_pen` (violino ou ECDF, para mostrar a censura), e a tabela de comparações pareadas. Entra direto no capítulo de método/resultados de E2. Nada é medido duas vezes.

---

## 7. Critério de saída

1. Os **três baselines** medidos em `Na ∈ {1, 2, 4}` × `Ns = 12` sobre os **mesmos** 500 episódios, com `agregado.csv` e `comparacoes.csv` gerados.
2. **Ordenação esperada confirmada** com IC da diferença separado de zero nas comparações adjacentes; ou, se não confirmada, causa diagnosticada e registrada antes de a SPEC-06 começar.
3. **Teste A/A com cobertura ≈ 95 %** — sem isso, nenhum IC deste projeto é confiável.
4. **Determinismo verificado**: duas execuções, colunas determinísticas idênticas.
5. **Interface `Policy` congelada**, com os quatro testes de contrato do §T-02.1 passando, e a SPEC-05 conseguindo implementá-la sem alteração da interface.
6. **Suposições registradas** (A-02.1 velocidade do guloso; D-02.1 executabilidade do k-means+TSP; D-02.2 imputação) e **sensibilidade a `communication_range`** reportada.
7. **MDE reportado** em toda comparação primária.
8. **Relatório curto** pronto.

---

## 8. Riscos

| Risco | Mitigação |
|---|---|
| **Censura mal tratada infla o piso** — média só sobre sucessos faz a caminhada aleatória parecer boa | D-02.2: métrica primária penalizada; `t_conc_sucesso` sempre acompanhada de `n_sucesso` e do aviso |
| **Imputar o instante da poda em vez de `max_episode_length`** premiaria travar cedo | asserção: `t_conc_pen ≥ t_fim_s` para episódios sem sucesso; teste dedicado |
| **Sinal invertido na conversão coordenada relativa → rumo** produz um guloso que foge dos sensores, indistinguível de "baseline ruim" | teste de monotonicidade com um sensor; ordenação esperada do §T-02.9 |
| **ICs marginais lidos como teste de hipótese** | IC da diferença pareada é a estatística primária; o aviso está na spec e vai ao relatório; `comparacoes.csv` tem coluna `contem_zero` para não sobrar interpretação |
| **Reamostragem independente no bootstrap** destrói o pareamento e infla o IC silenciosamente | um só conjunto de índices por réplica; teste A/A detecta a falha por cobertura |
| **Vazamento de sementes** treino → avaliação | árvore `SeedSequence` com ramos nomeados; teste de disjunção |
| **IC lido como incerteza do treino** | coluna `seed_treino` no esquema; SPEC-06 com ≥ 3 sementes onde couber; dispersão entre sementes reportada em separado |
| **`kmeans_tsp` apresentado como ótimo ou como limite inferior** | rotulação "planejador centralizado offline com posições verdadeiras"; coluna `usa_estado_verdadeiro`; LB-1 como o único limite realmente válido, com a folga reportada |
| **P1 (`communication_range`) não resolvido** muda o guloso, o LB-1, o atalho das pernas do TSP e o piso | varredura de 2–3 candidatos com sensibilidade reportada; escalar como bloqueante se a ordenação mudar |
| **Regra de fronteira indefinida** muda muito a caminhada aleatória (clamp faz o agente deslizar na parede) | exigir a decisão à SPEC-00, registrá-la no manifesto e reportar o piso sob a regra declarada |
| **`B = 1000` com caudas ruidosas** | rodar `B = 10 000` uma vez e mostrar que a conclusão não muda |
| **500 episódios não bastam** para separar métodos próximos | MDE obrigatório; se o efeito de interesse ficar abaixo do MDE, a conclusão correta é "inconclusivo com n = 500", não "sem diferença" |
| Custo do Held–Karp explodir em `Ns = 36` | limiar `m ≤ 12` exato, heurístico acima, com gap medido onde ambos rodam |

---

## 9. Entregáveis

- `src/uavwnn/agents/base.py` — `Policy`, `PoliticaPrivilegiada`, `Observacao`, `Acao`, `ContextoEpisodio`
- `src/uavwnn/agents/heuristics.py` — `random_uniforme` / `random_vmax` / `random_persistente`, `guloso` (+ `guloso_comprometido`, `guloso_oraculo`), `kmeans_tsp` (+ `kmeans_tsp_replan`), `LB-1`
- `src/uavwnn/eval/harness.py` — laço de avaliação, semeadura, dois modos de execução, CLI
- `src/uavwnn/eval/metrics.py` — métricas por episódio, censura, imputação, diagnósticos de coordenação
- `src/uavwnn/eval/stats.py` — bootstrap pareado, IC da diferença, taxa de vitória, MDE, Wilcoxon, McNemar, Holm
- `configs/spec02/` — uma config por célula, com a raiz de sementes e os candidatos de `communication_range`
- `results/spec02/` — `episodios.parquet`, `agregado.csv`, `comparacoes.csv`, `manifest.json`
- `reports/spec02-baselines.md` + figuras
- `tests/test_policies.py`, `tests/test_harness.py`, `tests/test_stats.py` (inclui o A/A)
- Alvo `make baselines` encadeado em `make repro`

---

## 10. Perguntas em aberto

**Q1 — `communication_range` (P1 do PRD).** Bloqueia o número absoluto de todo baseline: define quando o sensor é coletado, o atalho das pernas do TSP e o LB-1. Enquanto não resolvida, tudo aqui roda parametrizado com sensibilidade reportada.

**Q2 — Regra de fronteira.** A dissertação diz que as posições vivem em `[−scenario_size, scenario_size]` mas não diz o que ocorre quando o comando levaria o UAV para fora. *Clamp*, reflexão, ou movimento livre com a posição saturada? A caminhada aleatória é extremamente sensível a isso, e o guloso quase não é. Decisão pertence à SPEC-00 e precisa ir ao manifesto.

**Q3 — Velocidade da heurística gulosa (A-02.1).** Assumimos `v_max`. Confirmar no repositório `Project-GrADyS/gradys-data-collection`, branch `dissertation`; se lá for outra coisa, o baseline muda e H6 muda com ele.

**Q4 — Guloso reativo ou comprometido.** Adotamos reativo pela leitura literal. Confirmável no mesmo repositório.

**Q5 — A frequência de decisão da heurística é a mesma da política?** Assumimos `algorithm_iteration_interval = 2 s` para ambas. Se no original a heurística for contínua, ela fica artificialmente melhor no nosso harness — ou pior no dele.

**Q6 — Como Lamenza tratou episódios sem sucesso no "average completion time" da fig. 7.3.** Não é possível saber pelo texto. É o maior risco de incomparabilidade numérica com os valores publicados. Mitigação adotada: reportar `t_conc_pen`, `t_conc_sucesso` e `taxa_sucesso`, de modo que ao menos uma das leituras seja comparável qualquer que tenha sido a escolha dele. Precisa ser dito no relatório.

**Q7 — `max_seconds_stalled` na avaliação.** É otimização de treino (§5) ou parâmetro de cenário (tab. 3.1)? Adotamos ativo também na avaliação. Se desativado, os episódios da caminhada aleatória rodam até 500 s e o custo de simulação do piso sobe bastante, sem mudar a métrica primária.

**Q8 — Estado de visitação é global e instantâneo?** Assumimos que sim, coerente com o "canal global" de Lamenza §3 (p18). Se houver atraso de propagação, o guloso passa a poder perseguir um sensor já coletado, o que muda o baseline — e vira material da Etapa 3.

**Q9 — Particionamento consciente do *makespan*.** k-means minimiza inércia, não o máximo dos tempos. Vale gastar o passo de melhoria (transferência/troca entre clusters reduzindo o máximo)? Ganho provável em `Na` grande; custo pequeno. Fica na linha de corte por ora — decidir quando a folga `kmeans_tsp / LB-1` estiver medida.

**Q10 — `B = 1000` vs `10 000`.** Mantido 1000 por fidelidade ao protocolo de referência. Se as caudas se mostrarem instáveis entre execuções, promover 10 000 a canônico e registrar o desvio.

**Q11 — Baselines nas Etapas 2 e 3: ruidosos, oráculo, ou os dois?** Proposta: `guloso` sobre a observação ruidosa é o baseline pareado (degrada junto com a política, mantendo a comparação honesta) e `guloso_oraculo` é o teto. Precisa de decisão registrada antes da SPEC-08, porque muda a leitura de toda a escada de ablação.

**Q12 — Digitalizar a fig. 7.3 de Lamenza?** Os valores de MADDPG só existem como barras num gráfico. Proposta: digitalizar, plotar como linha de referência anotada com a incerteza de leitura declarada, e **nunca** colocar em `comparacoes.csv` — coerente com D7. Precisa de aprovação, porque é a única ponte numérica com o resultado publicado.

**Q13 — Diagnósticos de coordenação dependem do log de eventos da SPEC-00.** `visitas_por_agente` e `coletas_redundantes` exigem que o ambiente registre qual agente causou cada visita e quantos estavam no alcance no instante da coleta. Se o simulador não expuser isso barato, as colunas caem para `null` e a evidência de divisão de trabalho passa a ser só visual (trajetórias), como em Lamenza.

**Q14 — Quantas sementes de treino a SPEC-06 vai rodar?** O harness já suporta a coluna. Se for uma só, todo IC deste projeto é condicional a um único treino, e essa ressalva precisa aparecer em E2, não numa nota de rodapé.
