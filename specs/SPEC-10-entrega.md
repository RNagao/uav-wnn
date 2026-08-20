# SPEC-10 — Reprodutibilidade, relatório e slides

**Etapa:** transversal (todas) · **Depende de:** nada · **Estimativa:** contínua, dia 1 ao dia 28
**Portão de saída:** entregáveis E1, E2 e E3 conformes ao PRD §11 (global)
**Referência:** PRD §4 (entregáveis), §5.4 (desvios), §6.4 (RNF-06), §8 (cronograma e linha de corte), §11 (definition of done), §12 (trabalho futuro) · `specs/README.md` regra 4

---

## 1. Objetivo

Garantir que, ao fim das quatro semanas, existam:

- **E1** — repositório em que `make repro` regenera **todas** as tabelas e figuras do relatório a partir de sementes fixas;
- **E2** — relatório em formato de artigo, com reprodução e extensão em capítulos separados, resultados com IC e limitações honestas;
- **E3** — 12–15 slides no padrão visual da proposta já apresentada;

e que isso seja **consequência mecânica do trabalho diário**, não de um esforço concentrado na semana 4.

Há um quarto produto, que não aparece na lista do PRD §4 mas é o que sustenta os outros três: o **livro-razão de estado** (`STATUS.md`) — hipóteses, decisões, pendências, desvios e portões — vivo desde o dia 1.

## 2. Por que esta spec é contínua

O risco dominante deste projeto **não é técnico, é de fechamento**. Todas as outras specs falham de forma visível: um portão não é atingido, um número sai ruim, um teste fica vermelho. Esta falha em silêncio, e só na última sexta-feira.

Três fatos do PRD tornam a versão "escrever no fim" estruturalmente impossível:

1. **PRD §8:** "a redação não espera a semana 4: cada tabela entra no relatório assim que sai".
2. **`specs/README.md`, regra 4:** toda spec entrega um relatório curto que vai direto para o capítulo correspondente de E2. O relatório é, portanto, **montagem**, não redação. Se na semana 4 ainda houver texto de resultado a escrever do zero, alguma spec não cumpriu sua regra 4.
3. **PRD §11, item 6:** cada hipótese precisa de veredito **com o número que o sustenta**. Veredito não é algo que se descobre no fim; é um campo que existe desde o dia 1 com valor `aberta` e vai sendo preenchido.

A consequência de projeto é direta: **nada é medido duas vezes**. Se uma seção do relatório precisa de um número, ou existe um artefato que uma spec produziu, ou a spec dona está incompleta — e o caminho é abrir tarefa nela, nunca calcular dentro do relatório. O relatório **não contém nenhuma análise nova**. Esta spec constrói e faz cumprir esse contrato.

## 3. Escopo

**Dentro:** gestão de sementes e determinismo; pareamento da avaliação entre métodos; layout de `artifacts/` e o contrato spec→artefato→seção; `make repro` em três modos; esqueleto de E2 e o mapa seção ↔ spec ↔ artefato; texto de fundamentação (que não depende de medição); livro-razão `STATUS.md`; registro e verificação dos vereditos H1–H6; verificação automática da tabela de desvios §5.4; checagens de higiene de resultado (RNF-06); deck E3; empacotamento final e ritual semanal.

**Fora:** **produzir qualquer número novo**. Toda medição pertence a outra spec. Se esta spec estiver medindo alguma coisa, a divisão do `specs/README.md` falhou. Também fora: escolha de conteúdo científico (o que medir, qual configuração vence) — esta spec cuida do formato, da proveniência e do fechamento, não da substância.

---

## 4. Tarefas

### T-10.1 — Livro-razão de estado (`STATUS.md`) — dia 1

Fonte única de verdade sobre o estado do projeto. Markdown versionado na raiz, com blocos de formato fixo, **atualizado no mesmo commit que produz o número**. Nunca é reconstruído retroativamente.

Blocos obrigatórios:

| Bloco | Conteúdo | Regra |
|---|---|---|
| **Hipóteses** | H1–H6: estado, evidência, artefato, data, spec | Formato completo em T-10.8. Nasce com estado `aberta` |
| **Portões** | G0–G3: data, veredito, evidência, e se foi **dispensado por decisão** (PRD §2 permite) | Portão dispensado exige decisão nova no bloco seguinte |
| **Decisões** | D1–D12 do PRD §10 + D13 em diante | Colunas: ID, decisão, razão, data, estado (`vigente` / `revista` / `revogada`), o que a substituiu |
| **Pendências** | P1–P4 do PRD §10 | Colunas: ID, estado (`aberta` / `resolvida` / `caducada`), **valor resolvido**, **fonte** (commit, tabela, página da dissertação), data, impacto se continuar aberta |
| **Desvios** | espelho da tabela §5.4 do PRD | Coluna extra: `run_ids` que exercem o desvio. Alimentada pelo verificador de T-10.5, não à mão |
| **Seções de E2** | uma linha por seção do relatório | Estado: `ESQUELETO` / `RASCUNHO` / `COM NÚMEROS` / `FECHADA` |
| **Linha de corte** | itens da PRD §8 acionados | Data, razão, e a **limitação correspondente** que vai para o capítulo 9 de E2 |
| **Diário** | uma entrada datada por ritual semanal (T-10.11) | Cinco entradas ao fim |

Regras de estado:

- **Pendência aberta ao fim da semana 1 escala.** P1 (`communication_range`) e P2 (orçamento de simulação) são condição do portão G0 (PRD §8); se seguirem abertas, o `STATUS.md` registra o **valor assumido** e o **impacto declarado**, e esse impacto vai literalmente para o capítulo de limitações. Assumir sem registrar é o modo de falha.
- **Decisão revogada não é apagada.** O histórico é parte do relatório: o próprio PRD documenta reversões do v1 (D2, D3, D4) como conteúdo.
- Qualquer acréscimo à tabela de desvios exige, no mesmo commit, uma decisão nova neste arquivo (PRD §5.4).

### T-10.2 — Sementes, determinismo e pareamento da avaliação

**Uma semente mestre, derivação determinística.** `SEED_MASTER` é um único inteiro em `configs/base.yaml`. Toda outra semente é derivada, nunca escrita à mão:

```
seed(experiment_id, role, index) = int(SHA256(SEED_MASTER ‖ experiment_id ‖ role ‖ index)) mod 2^32
```

Papéis (`role`) separados, cada um com seu gerador independente:

| `role` | Governa | Depende do método? |
|---|---|---|
| `scenario` | posições dos sensores, posição inicial dos UAVs, máscara inicial | **não** |
| `noise` | realização de ruído GPS, enlace local, atraso de broadcast | **não** |
| `tuple_mapping` | sorteio dos `N × n` índices de bit das RAMs | sim (é o modelo) |
| `policy_init` | inicialização de LUTs / pesos | sim |
| `action_sampling` | amostragem da política gaussiana | sim |
| `bootstrap` | reamostragem do IC | não |

**Pareamento — o ponto crítico.** O conjunto de 500 episódios de avaliação (PRD §5.1) é gerado por `seed('eval', 'scenario', i)`, `i = 0..499`, e **não depende do método**. WNN, MLP pareada, caminhada aleatória, guloso, k-means+TSP e cada degrau da escada de ablação veem **exatamente os mesmos 500 cenários**. Isto é *common random numbers*: reduz variância e, mais importante, permite bootstrap **sobre a diferença** e não só sobre cada média — que é a estatística que H1, H2, H3 e H5 realmente exigem ("os ICs separam?").

**Ruído pré-gerado por episódio, indexado pelo passo.** A sequência de ruído `ε_t`, `t = 0..T_max`, é sorteada por `seed('eval', 'noise', i)` **antes** do episódio e consumida por índice de passo. Isso mantém a realização idêntica entre degraus da escada mesmo quando as trajetórias divergem — o que inevitavelmente acontece, porque políticas diferentes visitam estados diferentes. Sortear ruído "sob demanda a partir do estado" quebraria o pareamento silenciosamente e é proibido.

> **Ressalva a declarar no relatório.** Pareamento por realização de ruído é exato apenas enquanto as trajetórias coincidem; a partir da primeira divergência, os métodos veem a mesma *sequência* de perturbações aplicada a estados diferentes. É a melhor redução de variância disponível num ambiente episódico com política no laço, e não é equivalência perfeita. Dizer isso é uma linha do apêndice de método.

**Bootstrap pareado.** IC de 95 % por reamostragem de 1000 réplicas (PRD §6.4, Katopodis §5.4), reamostrando o **índice do episódio** — o mesmo índice reamostrado para todos os métodos dentro de uma réplica. O IC da diferença sai da distribuição das diferenças, não da subtração de dois ICs marginais.

**Sementes de treino são outra fonte de variância e não podem ser confundidas com as de avaliação.** Uma célula treinada com `R` sementes independentes (`role = policy_init`, `tuple_mapping`, `action_sampling`, `index = 0..R-1`) produz `R` políticas; cada uma é avaliada nos **mesmos** 500 episódios. Reportar as duas dispersões separadamente: IC dentro-de-semente (bootstrap sobre episódios) e dispersão entre sementes (min–max ou IC sobre `R` médias). Com orçamento de 4 semanas, `R = 1` para a maioria das células é o cenário realista; **então `R = 1` precisa estar escrito no relatório e na legenda de cada tabela**, porque um IC dentro-de-semente com `R = 1` não é IC sobre o método, é IC sobre a avaliação de um treino específico. Alvo mínimo: `R ≥ 3` nas células que sustentam vereditos que não podem cair (PRD §8).

**Determinismo verificável.** `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` — o último também é exigência de RNF-13 (um núcleo, sem GPU). Teste `test_determinism`: roda duas vezes uma configuração curta e compara o hash de `eval/episodes.parquet`. Divergência é bug, não ruído.

### T-10.3 — Layout de artefatos e contrato spec → relatório

```
artifacts/
├── runs/<run_id>/
│   ├── manifest.json           # config resolvida, git sha, sementes, hardware, versões
│   ├── checkpoint/             # CARO — LUTs/pesos finais (ator, crítico)
│   ├── train_log.parquet       # curva de treino; curva de ocupação (RNF-08)
│   └── eval/episodes.parquet   # uma linha por episódio de avaliação
├── tables/<id>.csv + <id>.json # BARATO — agregados com IC
├── figures/<id>.pdf + <id>.png # BARATO — versão clara (relatório)
├── figures/dark/<id>.png       # BARATO — versão escura (slides)
├── index.json                  # catálogo artefato → run_ids → spec → seção de E2
└── LOCK.sha256                 # hashes dos artefatos caros
```

`run_id = <spec>-<experimento>-<config_hash>-s<semente>`, com `config_hash` = hash canônico da configuração **resolvida** (após defaults e overrides). Dois runs com o mesmo `config_hash` e a mesma semente devem produzir o mesmo `episodes.parquet` — é o que `test_determinism` verifica.

**`episodes.parquet` é o nível de granularidade canônico.** Uma linha por `(run_id, episode_index)`, com no mínimo: `tempo_conclusao_s`, `sucesso` (bool), `n_visitados`, `passos`, e — nas Etapas 2 e 3 — `erro_pos_medio_m`, `nees_medio`, `nis_medio`, `nees_frac_no_intervalo`. **Toda tabela e todo IC do relatório é derivado deste arquivo por script.** Nenhum número agregado é digitado. É isto que torna "IC obrigatório" (RNF-06) mecanizável em vez de aspiracional.

**`manifest.json` — campos obrigatórios:**

| Campo | Por quê |
|---|---|
| `git_sha`, `git_dirty` | run gerado com árvore suja não entra no relatório |
| `seed_master`, `derived_seeds` | reexecução |
| `config` (resolvida) e `config_hash` | proveniência |
| `T, n, N, Na, Ns, sigma_gps, precision, encoding_scheme` | **RNF-06**: nenhum número de tamanho/latência sem a configuração que o gerou |
| `fusion_enabled` (bool) | RNF-07 e a separação reprodução × extensão |
| `deviations: [DEV-1, ...]` | preenchido **pelo código**, não à mão (T-10.5) |
| `hardware`: modelo de CPU, threads, RAM, ausência de GPU | RNF-13 |
| `versions`: python, numpy, gradysim, SO | reprodutibilidade honesta |
| `wall_time_train_s`, `wall_time_eval_s` | métrica de processo (PRD §6.4) |
| `framework`: `numpy` \| `torch` | RNF-11 (comparação justa de arcabouço) |

**Fronteira caro/barato — é exatamente onde `make repro` se divide.** `runs/*/checkpoint` e `train_log.parquet` custam horas; `eval/episodes.parquet` custa minutos; `tables/` e `figures/` custam segundos. Corolário de projeto: **nenhuma decisão de apresentação pode exigir retreino**. Se mudar a cor de uma figura ou o agrupamento de uma tabela obriga a treinar de novo, o pipeline está errado.

**Versionamento.** No git: código, `configs/`, `STATUS.md`, `tables/`, `figures/`, `index.json`, todos os `manifest.json` e `LOCK.sha256`. Fora do git: `checkpoint/`, `train_log.parquet`, `episodes.parquet` (tarball em armazenamento externo ou release). `make artifacts-lock` grava o sha256 de cada arquivo caro em `LOCK.sha256`; `make artifacts-verify` confere. Assim o repositório detecta um checkpoint trocado silenciosamente — o modo de falha em que a figura versionada deixa de corresponder ao artefato que a gerou.

**`index.json` é o contrato spec → relatório.** Uma entrada por tabela e por figura:

```json
{ "id": "e1-tempo-conclusao",
  "tipo": "tabela",
  "spec": "06",
  "secao_e2": "5.2",
  "run_ids": ["06-1b-a3f91c-s0", "06-1b-a3f91c-s1"],
  "script": "src/uavwnn/eval/plots.py:tabela_tempo_conclusao",
  "hipoteses": ["H1", "H6"] }
```

Regra: **zero órfãos nas duas direções**. Toda entrada de `index.json` aponta para `run_ids` existentes; toda tabela/figura citada no relatório existe em `index.json`. Verificado em `make repro`.

### T-10.4 — `make repro` em três modos

O PRD §4 exige `make repro` a partir de sementes fixas. Retreinar tudo pode não caber no prazo — então a alegação de reprodutibilidade é **estratificada e declarada**, nunca vaga.

| Alvo | O que refaz | Entrada | Alvo de tempo |
|---|---|---|---|
| `make repro` (= `repro-fast`, padrão) | **todas** as tabelas e figuras | `episodes.parquet` versionados | **≤ 15 min**, um núcleo |
| `make repro-eval` | reavalia 500 episódios a partir dos checkpoints, depois `repro-fast` | `checkpoint/` | minutos a poucas horas |
| `make repro-full` | retreina do zero a partir das sementes, depois `repro-eval` | só `configs/` + `SEED_MASTER` | horas a dias — declarado |

- `make repro` é o alvo do dia a dia e do CI. Roda **a cada dia**. Se ele quebra, quebrou hoje e não na semana 4.
- `make repro-eval` é o que dá garantia científica a custo aceitável: valida toda a metade do pipeline que transforma política em número publicado, sem pagar o treino.
- `make repro-full` precisa rodar **integralmente ao menos uma vez** sobre o **núcleo reprodutível**, definido como os runs que sustentam o que PRD §8 diz que não pode cair: réplica 1a, 1b contra o guloso, degraus 0/1/4/6c, sub-passo 3a, tabela de custo T1/T2/T3. O que fica fora do núcleo é reprodutível **por construção** (semente e configuração estão gravadas) mas não foi reexecutado — e o Apêndice A diz isso com essas palavras.
- Sub-alvos por spec (`make repro-spec06`) para regeneração parcial durante o projeto.
- **`make repro` compara e falha.** Regenera para um diretório temporário e compara com o versionado: exato para CSV/JSON de inteiros e strings, tolerância relativa declarada (`1e-9`) para ponto flutuante, e comparação de conteúdo (não de bytes) para figuras — PDF carrega timestamp. Divergência é *drift* e reprova.
- `make repro` executa também `check_deviations` (T-10.5) e `check_hygiene` (T-10.6). Falha em qualquer um reprova o alvo. `make ci` = testes + `repro` + linters.
- `make repro` **recusa** artefato cujo manifesto tenha `git_dirty: true`, exceto em modo `DEV=1`.

### T-10.5 — Verificação da tabela de desvios (PRD §5.4)

O PRD §11.3 exige que a tabela de desvios "reflita exatamente o que foi executado". Conferir isso a olho é o mesmo que não conferir.

**Desvios são declarados pelo código, não pelo autor.** Cada módulo que implementa um desvio registra seu ID no manifesto do run em tempo de execução:

| ID | Desvio | Estado |
|---|---|---|
| `DEV-1` | binarização da observação por termômetro | sempre ativo (é a única linha da §5.4 hoje) |
| `DEV-2` | codificação polar E3 (SPEC-01) | condicional — só se E3 for adotado |
| `DEV-3` | covariância na observação (degrau 5, PRD §6.2) | condicional — ativo só nos runs do degrau 5 |

`tools/check_deviations.py` faz três verificações:

1. **Executado ⊆ documentado.** União dos `deviations` de todos os runs referenciados por `index.json` deve estar contida na tabela §5.4 do PRD (parseada do markdown). Desvio executado sem linha na tabela → **falha**.
2. **Documentado ⇒ decidido.** Toda linha da tabela §5.4 tem decisão correspondente vigente no `STATUS.md`. Sem decisão → **falha**.
3. **Documentado e não executado → aviso.** Linha prevista que nenhum run exerce: legítimo (o degrau 5 pode cair pela linha de corte), mas obriga nota no relatório dizendo que o desvio foi previsto e não executado.

O verificador também emite `artifacts/tables/desvios.json`, que é o que o relatório renderiza — a tabela de desvios de E2 **é gerada**, não transcrita. É a única forma de ela refletir o executado por construção.

**Simetria RNF-07.** Teste automatizado: um run curto com `fusion_enabled=false` e asserção de que nenhum módulo de `fusion/` foi importado (inspeção de `sys.modules`). Com `fusion/` desligado, o sistema executa o problema de Lamenza puro — verificável, não prometido. Junto com o teste de RF-04 (a verdade não vaza para a política), formam os dois invariantes do PRD §11.2.

### T-10.6 — Higiene de resultados forçada (RNF-06)

RNF-06 tem três cláusulas. Cada uma vira uma checagem executável em `tools/check_hygiene.py`, rodada por `make repro` e `make ci`. Mais duas que a prática exige.

| Regra | Cláusula do PRD | Como é forçada |
|---|---|---|
| **HG-1** | nenhum resultado agregado sem IC | Todo agregado é escrito pela função única `emit_aggregate(...)`, que **exige** `value`, `ci_low`, `ci_high`, `n`, `ci_method`. O linter valida `tables/*.json` contra um JSON Schema: registro derivado de mais de uma observação sem campos de IC → falha |
| **HG-2** | nenhuma conclusão de RL antes do NEES/NIS passar | Toda tabela/figura com `domain: rl` e `etapa ∈ {2,3}` carrega `consistency_ref` apontando para o registro NEES/NIS do mesmo run e canal, e esse registro precisa ter `status: pass`. Ausente ou `fail` → o artefato é gerado **com tarja "PRELIMINAR — consistência não verificada"** e `make repro` reprova em modo estrito |
| **HG-3** | nenhum número de tamanho ou latência sem a configuração | Registros de T1/T2/T3 exigem `config: {T, n, N, Na, Ns, precision, framework, hardware_id}` não nulo. E o renderizador **imprime a configuração na legenda automaticamente** — não depende de o autor lembrar |
| **HG-4** | (corolário) nada digitado | No corpo dos capítulos de resultado (5, 6, 7) e no capítulo 8, **dígitos literais são proibidos**, exceto lista branca de constantes do problema (200 m, 2 s, 500 episódios, 0,99, 32 dias, 100 %). Todo número vem de `{{tbl:e1-tempo.wnn.media}}`, resolvido no build. Placeholder não resolvível → falha do build |
| **HG-5** | (corolário) reprodução não contamina extensão | Nenhuma figura ou tabela do capítulo 5 (Reprodução) pode consumir run com `fusion_enabled: true`. Verificado por `index.json` × manifestos. É o princípio da PRD §0 verificado no relatório, não só no código |

**Válvula de escape explícita.** Falso positivo se suprime com anotação `# hygiene: allow HG-N — <razão>` no ponto exato, e a supressão aparece no diff. Um linter sem escape barato é um linter que alguém desliga inteiro na semana 4; um com escape visível preserva a intenção.

### T-10.7 — Esqueleto de E2 e o mapa seção ↔ spec ↔ artefato

**Criado no dia 1**, com todas as seções presentes, vazias, cada uma marcada com a spec que a alimenta e com seu estado. Formato-fonte: Markdown com placeholders, compilado por `tools/build_report.py` (pandoc) para PDF em template de artigo.

| § de E2 | Conteúdo | Spec que produz os números | Artefatos consumidos |
|---|---|---|---|
| 1 Introdução | contexto, lacuna, contribuições (as três: aproximador barato, custo como figura de mérito, fusão sobre o canal global) | — (PRD §1, §2) | — |
| 2 Trabalhos relacionados | Lamenza; Katopodis; MADDPG; interseção de covariância (Julier & Uhlmann) | — | — |
| 3.1 Fundamentação — rede n-tuple | **Não escrever por analogia.** A dissertação *prova* o que precisamos, e nós *testamos* cada passo: a rede de regressão **é** um estimador de Nadaraya-Watson (3.22); a de gradiente funcional **é** máquina de núcleo, `f(x) = Σ_j α_j k(x,x_j)` (4.16); a atualização **é** um passo de gradiente funcional que se propaga pelo núcleo, `Δf(x) = αΔ·k(x_t,x)` (4.19); e o núcleo `k = 1 − δ/N` **é positivo-definido**, via o mapa explícito da (4.22) e a identidade `⟨φ(w),φ(x)⟩ = k(w,x)` da (4.23). É a (4.23) que sustenta as outras — o texto assinala que (4.18)→(4.19) *hinges on the kernel being positive-definite*. **A RNF-01 antiga (`η = c/N`) não entra: ela supunha `‖φ‖² = N`, e a (4.22) normaliza por `1/√N`.** Fator de esquecimento e não-estacionariedade (RNF-04) | `tests/test_identidades_artigo.py` (21 testes), `specs/PLANO-TESTES.md` | figura de ladrilhamento; curva `d_obs × d_tuple` (SPEC-01 T-01.3); espectro da matriz de Gram |
| 3.2 Fundamentação — filtro de Kalman | KF linear; **KF com entrada de controle** (`x̂ + u·dt`) e por que a ação da política é informação disponível de graça; filtro de informação para fonte estática; NEES/NIS como teste de consistência | 07 | — |
| 3.3 Fundamentação — interseção de covariância | por que o KF ingênuo diverge sob correlação desconhecida; CI como limite conservador | 07, 09 | — |
| 4.1 Método — o problema reproduzido | MDP de Lamenza verbatim: observação eq. 5-1, ação contínua eq. 3-2, recompensa eq. 5-6, cenário tab. 7.1 | 00 | tabela de parâmetros do cenário |
| 4.2 Método — codificador | esquema escolhido, `T`, binning por slot. **A cobertura de bits deixou de ser número a reportar e passou a ser propriedade estrutural:** com o mapeamento por partição do `:uniform_random` ela é 1,0 por construção (achado C4 fechado). Mencionar o contraste com a amostragem por RAM, onde `N·n/B` podia ficar abaixo de 1 | **01** | `tables/spec01-*`, figuras de geometria |
| 4.3 Método — aproximadores | n-tuple Actor-Critic; MLP AC pareada; parameter sharing | 04, 05 | — |
| 4.4 Método — protocolo de custo | T1 (alocado/ocupado/**realizado** + mapeamento de tuplas), T2 (mediana/p99, decomposto, com e sem arcabouço), T3 (operações por tipo) | **03** | — |
| 4.5 Método — protocolo de avaliação | 500 episódios pareados, bootstrap, sementes, hardware | **10 (esta)** | `manifest` agregado |
| **5 Reprodução (Etapa 1)** | 1a vs os 100 % publicados; 1b vs guloso; varreduras `T×n×N`; **pareado WNN × MLP**; tabela de custo contra as duas referências; curva de ocupação | **06** | `tables/e1-*`, `figures/e1-*` |
| **6 Extensão I — fusão local (Etapa 2)** | onde o ruído entra; escada 0–5; NEES/NIS por canal; varredura σ×T; regra *bin ≈ σ*; treino limpo vs ruidoso; T1/T2 com o filtro embutido (RNF-05) | **07, 08** | `tables/e2-*` |
| **7 Extensão II — multiagente (Etapa 3)** | 3a sem ruído; 3b degraus 1/4; 6a/6b/6c; divergência do KF ingênuo; CI | **09** | `tables/e3-*` |
| **8 Discussão** | **vereditos H1–H6**; tabela de desvios (gerada); custo contra ambas as referências; o que o resultado significa se a WNN não for menor | **10 (compila)** | `tables/verdicts.json`, `tables/desvios.json` |
| **9 Limitações** | `R` sementes de treino; células não executadas; itens da linha de corte acionados; pendências abertas; horizonte da curva de ocupação (RNF-08); pareamento de ruído após divergência de trajetória; MADDPG não reproduzido (D7) | **10** | `STATUS.md` |
| 10 Trabalho futuro | PRD §12 verbatim: mapa setorial, ação discretizada, shaping, reivindicação, `Na` variável | — | — |
| Apêndice A — Reprodutibilidade | `SEED_MASTER`, derivação, layout de artefatos, os três modos de `make repro`, **o que foi reexecutado e o que não foi**, hardware, versões | **10** | `manifest` agregado |
| Apêndice B — Varreduras completas | tabelas longas que não cabem no corpo | 06, 08 | `tables/*-full` |

**Duas regras estruturais:**

1. **Reprodução e extensão são capítulos separados e o separador é verificável.** O capítulo 5 não cita nenhum run com `fusion_enabled: true` (HG-5). O capítulo 6 abre declarando: *"a partir daqui o problema é estendido"*. É a PRD §0 materializada no sumário.
2. **O relatório não calcula.** Se falta um número, abre-se tarefa na spec dona. Consumir `episodes.parquet` dentro do build do relatório é violação de arquitetura, não atalho.

**Cadência — a regra das 24 h.** Quando uma spec fecha um portão ou entrega seu relatório curto (regra 4 do `specs/README.md`), a seção correspondente de E2 sai de `ESQUELETO` para pelo menos `COM NÚMEROS` **em até 24 h**. Estado por seção no `STATUS.md`.

**Antecipação deliberada.** As seções que **não dependem de nenhum resultado** — 1, 2, 3.1, 3.2, 3.3, 4.1 — são escritas nas **semanas 1 e 2**, quando o custo de oportunidade é menor e o simulador ainda está sendo domado. São aproximadamente metade do texto do artigo. Esta é a mitigação mais concreta contra o modo de falha "o relatório ficou para o fim": na semana 4, o que resta é montagem de resultado, discussão e limitações.

**Marcos:** fim da semana 1 — seções 1, 2, 3 em `RASCUNHO`. Fim da semana 2 — seção 4 completa e seção 5 em `COM NÚMEROS`. Fim da semana 3 — **≥ 70 % das seções em `COM NÚMEROS`**. Semana 4 — capítulos 7, 8, 9 e fechamento.

### T-10.8 — Vereditos H1–H6

PRD §11.6: cada hipótese tem veredito escrito — confirmada, refutada ou inconclusiva — **com o número que o sustenta**. Hipótese refutada é resultado, não fracasso.

**Onde vivem:** bloco `Hipóteses` do `STATUS.md` desde o dia 1, exportado por script para `artifacts/tables/verdicts.json`, que é o que o capítulo 8 renderiza. Não existe versão do veredito que só apareça no fim.

**Formato do registro:**

```yaml
H3:
  etapa: 2
  enunciado: "O KF com a ação da política como entrada de controle devolve
              a rede ao regime praticamente sem ruído"          # PRD §2, verbatim
  falsificacao: "cai se a degradação com filtro for estatisticamente
                 distinguível do oráculo"                        # PRD §2, verbatim
  estado: confirmada            # aberta | confirmada | refutada | inconclusiva
  data: 2026-08-27
  spec: 08
  precondicoes:
    - "NEES/NIS pass"  ->  tables/e2-consistencia.json#canal=gps
  evidencia:
    - metrica: tempo_conclusao_s
      comparacao: "degrau 4 vs degrau 0 (oráculo), sigma_gps=10, T=32"
      valores: "118,4 vs 115,9"
      diferenca_pareada: "+2,5 s  IC95 [-1,8; +6,9]  bootstrap pareado, 1000 réplicas, n=500"
      artefato: tables/e2-escada.json#row=deg4
      run_ids: ["08-escada-7c21ab-s0"]
  nota: "IC da diferença contém zero: indistinguível do oráculo, que é o
         critério de H3. Vale para sigma<=10 m; não testado além."
```

**Regras de integridade, verificadas por `check_hygiene`:**

- Veredito com estado diferente de `aberta` **exige** ao menos uma linha de evidência cujo `artefato` seja resolvível em `index.json`. Ponteiro morto → falha.
- Estado `inconclusiva` **exige** `razao ∈ {nao_executado, ic_nao_separa, precondicao_falhou}`. Sem essa restrição, "inconclusiva" vira fuga para hipótese que ninguém mediu; com ela, `nao_executado` denuncia exatamente isso e obriga a linha correspondente no capítulo 9.
- Estado `refutada` **exige** `nota` com **diagnóstico** (por quê caiu) — PRD §6.1 já pede isso para G1: "registrar o veredito, diagnosticar (resolução? tamanho de tupla? variância da gaussiana?) e prosseguir".
- Nenhuma hipótese pode terminar o projeto em `aberta`. Ao fim da semana 4, `aberta` vira `inconclusiva / nao_executado`, com a limitação declarada.
- Hipótese cuja pré-condição falhou (HG-2: NEES/NIS reprovou) **não pode** ser marcada `confirmada`. É o mesmo que a cláusula "nenhuma conclusão de RL antes do NEES/NIS passar", aplicada ao nível do veredito.

**Revisão obrigatória** em cada portão (G0–G3) e em todo ritual semanal (T-10.11). O placar das seis vai para um slide dedicado de E3.

**Correspondência hipótese → portão → spec:**

| Hipótese | Portão | Spec que a decide | Métrica decisória |
|---|---|---|---|
| H1 | G1 | 06 (com 03) | tempo de conclusão pareado + T1 realizado + T3 (zero MACs fp) |
| H2 | G2 | 08 | curvas de degradação WNN × MLP sobre σ; ICs separam? |
| H3 | G2 | 08 | degrau 4 vs degrau 0, diferença pareada |
| ~~H4~~ | — | — | **RETIRADA por D13**: o mecanismo era o canal do sensor, fora do escopo. O degrau 5 permanece como ablação sem hipótese associada |
| H5 | G3 | 09 | erro de estimativa 6c vs 6a + NEES/NIS de 6b |
| H6 | G3 | 09 | tempo de conclusão vs heurística gulosa, `Na ≥ 2` |

### T-10.9 — Slides (E3)

**12–15 slides, mesmo padrão visual da proposta já apresentada.** Padrão extraído do PDF da proposta e registrado em `reports/estilo.md`:

| Elemento | Valor |
|---|---|
| Fundo | `#10243A` (marinho), painéis em `#13293D` / `#1B3B57` |
| Realce primário | `#1C7293` (teal) — faixas, títulos de seção, séries principais |
| Realce secundário | `#F2A93B` (âmbar) — número da seção, destaque de resultado, série de contraste |
| Texto | `#F4F6F8` (corpo claro), `#CFE2ED` (secundário), `#8FB3C7` (legendas) |
| Tipografia | títulos em Cambria Bold; corpo e legendas em Calibri (regular / itálico / bold) |
| Estrutura | número de seção grande em âmbar à esquerda; rodapé `PESC/COPPE-UFRJ · Fusão de Dados · <título curto>`; slide de contato e slide de referências ao fim |

**Figuras dos slides são as mesmas do relatório.** `artifacts/figures/dark/<id>.png` é gerado pelo **mesmo script** que gera `artifacts/figures/<id>.pdf`, trocando só o tema (paleta acima, fundo transparente). Captura de tela é proibida: é o mecanismo pelo qual o número do slide passa a discordar do número do relatório.

Cada slide com figura carrega, no campo de **notas do apresentador**, a proveniência: `id` do artefato, `run_ids` e configuração. Serve para a arguição e é gratuito.

**Roteiro proposto (13 slides, folga para 15):**

| # | Slide |
|---|---|
| 1 | Título, autor, disciplina, data |
| 2 | O problema: sensores IoT sem infraestrutura, enxame de UAVs, cenário 200×200 m |
| 3 | O ponto de partida: Lamenza resolve com MADDPG — e custa 32 dias de GPU |
| 4 | Rede sem peso: `f(x) = ⟨W, φ(x)⟩` sobre bits esparsos = tile coding aleatório; sem MAC, sem FPU |
| 5 | **O que trocamos e o que não trocamos** — a tabela de desvios com sua única linha |
| 6 | Hipóteses H1–H6 e os três portões |
| 7 | Método: as três etapas e a escada de ablação 0→5, 6a/6b/6c |
| 8 | Etapa 1 — reprodução: tempo de conclusão vs guloso, WNN vs MLP pareada |
| 9 | Custo: T1/T2/T3 contra as **duas** referências (MLP pareada e ator do MADDPG) |
| 10 | Etapa 2 — ruído, KF com entrada de controle, NEES/NIS |
| 11 | A regra que amarra o trabalho: largura do bin ≈ σ da estimativa filtrada |
| 12 | Etapa 3 — interseção de covariância vs KF ingênuo |
| 13 | Placar H1–H6, limitações honestas e trabalho futuro |
| (14) | Contato / referências |
| (15) | Backup: varreduras completas |

**Marco antimodo-de-falha:** ao fim da **semana 1** já existe um deck de 5 slides (1, 2, 3, 5, 7) — os que não dependem de resultado nenhum. O deck **cresce**; nunca se parte do zero na semana 4.

### T-10.10 — Higiene de repositório e empacotamento (E1)

- `README.md` do repositório: o que é, como instalar, como rodar cada modo de `repro`, o que está no núcleo reprodutível, e o mapa `spec → artefato → seção de E2`.
- `pyproject.toml` com dependências travadas + lockfile; versão do Python declarada; `gradysim` fixado por commit.
- `make ci` = testes + `repro-fast` + `check_hygiene` + `check_deviations` + `artifacts-verify`. Se houver CI hospedado, roda lá; se não, hook de `pre-push` — o requisito é ser **um comando**, não uma lista de lembretes.
- Tag final `v1.0-entrega`: `STATUS.md` fechado, `LOCK.sha256` atualizado, `reports/relatorio.pdf`, `slides.pdf`, e o tarball de artefatos caros com seu sha256 no `README`.

### T-10.11 — Ritual semanal (o mecanismo que faz o resto acontecer)

Sexta-feira, ~60 minutos, pauta fixa, resultado registrado como entrada datada no `STATUS.md § Diário`:

1. Rodar `make ci`. Vermelho vira a primeira tarefa de segunda — nunca "depois".
2. Atualizar `STATUS.md`: portões, P1–P4, decisões novas, estado de cada seção de E2.
3. **Revisar os seis vereditos.** Cada um: mudou? tem evidência? aponta para artefato vivo?
4. **Decidir explicitamente se a linha de corte da PRD §8 é acionada.** Registrar o item cortado, a razão, e — obrigatoriamente — a **limitação correspondente** já redigida para o capítulo 9. Corte sem limitação escrita não é corte, é omissão. Vale a regra de escalonamento da §8: portão não atingido aciona a linha de corte **da etapa seguinte**.
5. Estender o deck com o que apareceu na semana.

Cinco entradas ao fim do projeto (semanas 1–4 e fechamento). É a única tarefa desta spec com hora marcada, e é a que converte "spec contínua" em comportamento observável.

---

## 5. Critério de saída

1. `make repro` verde a partir de checkout limpo, **≤ 15 min**, regenerando **100 %** das tabelas e figuras do relatório, com divergência nula ou dentro da tolerância declarada.
2. `make repro-full` executado **integralmente ao menos uma vez** sobre o núcleo reprodutível, com tempo de parede registrado no Apêndice A; e o que ficou fora declarado com essas palavras.
3. `make ci` verde: testes (incluindo **RF-04** — a verdade não vaza — e **RNF-07** — com `fusion/` desligado roda o Lamenza puro), `check_hygiene` (HG-1 a HG-5), `check_deviations`, `artifacts-verify`.
4. **Zero órfãos:** toda tabela/figura do relatório resolve a `index.json` → `run_ids` → `manifest.json`; e toda entrada de `index.json` é citada.
5. **Seis vereditos escritos**, cada um com evidência apontando para artefato existente; nenhum em `aberta`; refutados com diagnóstico; inconclusivos com razão classificada.
6. **Tabela de desvios do relatório gerada** a partir dos manifestos e idêntica ao conjunto executado — verificada, não conferida a olho.
7. Capítulo de limitações contendo, no mínimo: número de sementes de treino por célula, células não executadas, itens da linha de corte acionados, pendências não resolvidas, horizonte da curva de ocupação (RNF-08), ressalva do pareamento de ruído, e a não reprodução do MADDPG (D7).
8. **E2** em PDF, formato de artigo, reprodução e extensão em capítulos separados, todos os agregados com IC.
9. **E3** com 12–15 slides no padrão visual da proposta, todas as figuras rastreadas a artefatos e com proveniência nas notas.
10. `STATUS.md` fechado: P1–P4 em estado terminal, decisões vigentes, G0–G3 com veredito, data e evidência, cinco entradas de diário.

---

## 6. Riscos

| Risco | Mitigação |
|---|---|
| **O relatório é empurrado para a semana 4 e não fecha** — o risco dominante | Esqueleto completo no dia 1; seções 1–4.1 (metade do texto) escritas nas semanas 1–2 porque **não dependem de resultado**; regra das 24 h; estado por seção no `STATUS.md`; meta de 70 % em `COM NÚMEROS` ao fim da semana 3; ritual semanal com pauta fixa |
| **`make repro-full` não cabe no prazo** e "reprodutível" vira alegação vaga | Três modos; **núcleo reprodutível declarado e pequeno**; `repro-fast` roda todo dia e é o alvo do PRD §4; o que não foi reexecutado é declarado no Apêndice A |
| **Números do relatório envelhecem** em relação aos artefatos (copiar-colar) | HG-4: dígito literal proibido nos capítulos de resultado; placeholders resolvidos no build; build falha em placeholder órfão |
| **Avaliação não pareada** infla ou esconde diferença; ICs marginais usados para responder "os ICs separam?" | Sementes de cenário e ruído independentes do método; ruído pré-gerado indexado pelo passo; bootstrap **pareado** sobre índice de episódio; teste que assere que dois métodos receberam a mesma lista de cenários |
| **`R = 1` semente de treino** vira alegação mais forte do que a evidência | Declarado em cada legenda e no capítulo 9; `R ≥ 3` obrigatório nas células que sustentam vereditos que não podem cair; sem isso, o veredito registra a limitação |
| **Determinismo quebra em silêncio** (threads, versão de biblioteca, hash de dicionário) | `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, versões travadas, `test_determinism`, `git_dirty` no manifesto e recusa de artefato sujo |
| **Checkpoints perdidos ou não versionáveis** (grandes demais para o git) | `LOCK.sha256` versionado + tarball externo; `repro-eval` reconstrói tudo a partir de checkpoints; perda de checkpoint degrada para `repro-full` daquele run, com custo conhecido |
| **Desvio entra sem passar pela §5.4** (degrau 5, polar E3) | `deviations` emitido pelo código no manifesto; `check_deviations` no CI; tabela do relatório **gerada** dos manifestos |
| **Conclusão de RL tirada antes do NEES/NIS** — já aconteceu no experimento preliminar (PRD §9) | HG-2: `consistency_ref` obrigatório, tarja "PRELIMINAR" e reprovação em modo estrito; veredito com pré-condição falha não pode ser `confirmada` |
| **Slides feitos do zero na semana 4** | Deck de 5 slides ao fim da semana 1; figuras vêm do mesmo pipeline do relatório em variante escura |
| **Padrão visual não reproduzível** (paleta/fonte na cabeça de ninguém) | `reports/estilo.md` com os hex extraídos da proposta e a tipografia; tema escuro implementado no gerador de figuras |
| **Os linters viram burocracia e alguém os desliga** | Escopo mínimo (cinco regras); supressão pontual e visível via `# hygiene: allow HG-N — razão`; nenhuma regra que exija trabalho manual recorrente |
| **Esta spec canibaliza tempo das specs que produzem resultado** | Escopo fora explícito: esta spec não mede nada. Orçamento: ~30 min/dia + 60 min de ritual semanal. Se estiver custando mais, é porque está fazendo trabalho de outra spec |
| **Linha de corte acionada sem virar limitação declarada** | Item 4 do ritual: corte só é registrado junto com o texto da limitação correspondente |

---

## 7. Entregáveis

- `Makefile` — alvos `repro`, `repro-fast`, `repro-eval`, `repro-full`, `repro-spec<NN>`, `ci`, `artifacts-lock`, `artifacts-verify`, `report`, `slides-figs`
- `STATUS.md` — livro-razão (hipóteses, portões, decisões, pendências, desvios, seções de E2, linha de corte, diário)
- `src/uavwnn/repro/seeds.py` — derivação determinística por `(experiment_id, role, index)`
- `src/uavwnn/repro/manifest.py` — captura de configuração, git, hardware, versões, desvios
- `src/uavwnn/eval/emit.py` — `emit_aggregate(...)`, o único caminho para um número agregado
- `tools/check_hygiene.py` — HG-1 a HG-5 e integridade dos vereditos
- `tools/check_deviations.py` — verificação da tabela §5.4 e geração de `tables/desvios.json`
- `tools/build_report.py` — resolução de placeholders + pandoc → PDF
- `artifacts/` — layout, `index.json`, `LOCK.sha256`
- `reports/relatorio/` (fonte, esqueleto no dia 1) → `reports/relatorio.pdf`
- `reports/estilo.md` — paleta, tipografia, gabaritos de figura claro/escuro
- `slides/` — fonte, `slides/figuras/` gerada do pipeline, notas de proveniência → `slides.pdf`
- `tests/test_repro.py` — determinismo, pareamento de cenários entre métodos, ausência de órfãos, RNF-07

---

## 8. Perguntas em aberto

1. **Formato-fonte de E2.** Markdown + pandoc (versionável, casa com HG-4 e com placeholders) ou LaTeX direto (melhor tipografia, mas o linter de placeholders precisa ser escrito para `.tex`)? Há exigência de template da disciplina — ABNT, SBC, IEEE — ou o formato é livre? Isso muda o esqueleto e precisa ser decidido no dia 1.
2. **Formato-fonte de E3.** A proposta foi feita em Google Slides. Manter (fidelidade visual garantida, mas não versionável nem gerável por `make`) ou migrar para Marp/Beamer com a paleta replicada (versionável, mas risco de o padrão visual não bater)? A recomendação padrão é manter o Google Slides e importar as figuras de `artifacts/figures/dark/` — o que deixa E3 fora de `make repro` por construção. Isso é aceitável?
3. **Quantas sementes de treino por célula?** O critério de saída pede `R ≥ 3` nas células críticas, mas isso multiplica por três o custo de treino justamente onde ele é maior. Se o orçamento medido em P2 não permitir, qual é a preferência: `R ≥ 3` em menos células, ou `R = 1` em mais células com a limitação declarada?
4. **Onde ficam os artefatos caros?** Sem definição de armazenamento externo (release do GitHub, Drive, disco local), `LOCK.sha256` verifica integridade mas não garante disponibilidade para um terceiro — o que enfraquece a alegação de reprodutibilidade de E1.
5. **Existe CI hospedado?** Se o repositório não for para uma forge com Actions, `make ci` roda por hook de `pre-push` local — mais frágil, porque é suprimível com `--no-verify`.
6. **Modo estrito de HG-2.** Reprovar `make repro` quando o NEES/NIS não passou é o comportamento correto no fim, mas trava o trabalho do dia enquanto o filtro está sendo sintonizado. Proposta: estrito no `make ci` e na tag final, permissivo (só tarja) no `repro` do dia a dia. Confirmar.
7. **Numeração das specs de fusão.** A SPEC-01 cita "o filtro de Kalman (SPEC-05)" e "a varredura com estimativa filtrada é da SPEC-06", enquanto o `specs/README.md` atribui a fusão local à SPEC-07 e a Etapa 2 à SPEC-08. O mapa da §T-10.7 seguiu o README. A divergência precisa ser resolvida numa das duas fontes, porque o `index.json` vai carregar o campo `spec` e ele tem que apontar para algo estável.
8. ~~**P3** (posição estimada vs distância/RSSI)~~ **fechada por D13**: o canal do sensor saiu do escopo, e o capítulo 6 do relatório perde essa dependência. Texto antigo: A recomendação do PRD é o caso linear. Se ficar aberta até a semana 3, o esqueleto da seção 3.2 (filtro de informação vs EKF/UKF) precisa ser reescrito tarde. Vale fechá-la por decisão na semana 1, mesmo sem experimento?
9. **Idioma de E2.** Português (disciplina) ou inglês (aproveitamento posterior como artigo)? Muda o esforço de redação e o reaproveitamento do texto de fundamentação.
10. **O deck de 5 slides da semana 1 é apresentado a alguém?** Se houver um ponto de controle intermediário com o professor, ele deve virar marco explícito do cronograma — feedback na semana 2 vale muito mais que na semana 4.
