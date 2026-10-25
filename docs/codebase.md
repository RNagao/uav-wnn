# A base de código — arquivos, responsabilidades e pipelines

Mapa completo do repositório: **42 scripts**, **45 módulos de biblioteca**
(34 de código + 11 `__init__`), **28 arquivos de teste** (385 testes). O que cada
um faz, e os **oito pipelines** que os combinam.

Gerado por análise AST do grafo de imports, não por leitura aproximada.

---

## Parte I — A biblioteca (`src/uavwnn/`)

O código reutilizável. Nenhum módulo aqui tem CLI: são peças que os scripts
compõem.

### `refenv/` — a fronteira com o problema do Lamenza

A **única** porta para o simulador. É o que torna "não modificar o problema"
verificável num lugar só.

| arquivo | responsabilidade |
|---|---|
| `loader.py` | localiza e carrega o submodule `vendor/gradys-data-collection`. Único ponto de contato do projeto inteiro |
| `runner.py` | `EnvConfig`, `make_env`, `run_episode`, semeadura, `LAMENZA_003` (a config canônica da tab. 7.1) |

### `wnn/` — as redes sem peso

Não sabem o que é RL: recebem bits e um passo, devolvem números.

| arquivo | responsabilidade |
|---|---|
| `redes.py` | **o núcleo**. `RedeGradienteFuncional` (ator §4.3.1 — só valores), `RedeRegressaoEsquecimento` (crítico §4.2.1 — valor + contador `uint32`), endereçamento vetorizado, `salvar`/`carregar`, `TreinoInstavel` |
| `ntuple.py` | variante **densa** do regressor, usada nos testes de identidade do núcleo |

### `agents/` — políticas e o algoritmo

| arquivo | responsabilidade |
|---|---|
| `base.py` | **o contrato**: `Policy`, `Observacao`, `Acao`, `ContextoEpisodio`. Congelado — é o que os três braços cumprem igualmente |
| `ac_continuo.py` | **o Algoritmo 4.4**: buffer de `m` passos, retorno, `δ`, as duas atualizações. E `APROXIMADORES` — a família selecionável por nome, que garante que os três atravessem o MESMO laço |
| `politica_continua.py` | `NormalProjetada` (a da dissertação) e `GaussianaSimples` (o C0 que perdeu o portão da SPEC-05) |
| `baselines.py` | `CaminhadaAleatoria`, `Guloso`, `KMeansTSP` — os pisos e tetos de referência |
| `mlp_ac.py` | MLP pareada com `Adam` — o braço de controle, mesmo laço, aproximador diferente |

### `campanha/` — o protocolo experimental

O que torna a medição comparável entre corridas.

| arquivo | responsabilidade |
|---|---|
| `bancos.py` | bancos `val` e `test` **versionados**. Regenerá-los com outro NumPy mudaria os cenários com a mesma semente, e comparações registradas deixariam de valer sem erro aparente |
| `sementes.py` | os três papéis de semente (treino, avaliação, tupla) — separá-los permite variar um sem contaminar os outros |
| `treino.py` | `Corrida` (artefatos), `Avaliacao`, `ParadaAntecipada` (média móvel 5, paciência 8, mínimo 20 avaliações) |
| `recompensas.py` | as duas recompensas de Lamenza, calculadas **fora** do ambiente |
| `normalizacao.py` | `NormalizadorMovel` — o `MeanStdFilter` do RLlib, para o braço PPO |

### `encoding/` — floats viram bits

| arquivo | responsabilidade |
|---|---|
| `thermometer.py` | `TermometroQuantil` (o usado), `TermometroGlobal`, `TermometroPorSlot`. Os limiares são **estado treinado** e entram no T1 |
| `corpus.py` | colhe do ambiente as observações que calibram os limiares |
| `artigo.py` | codificadores **literais** da dissertação (§3.2.3), para os testes diferenciais |

### `eval/` — medir com honestidade

| arquivo | responsabilidade |
|---|---|
| `metricas.py` | métricas por episódio, **censura** e imputação — o que fazer quando o episódio não completa |
| `stats.py` | bootstrap pareado, IC da diferença, MDE, TOST (equivalência) |
| `sementes.py` | árvore de sementes — CRN entre braços |
| `harness.py` | orquestra avaliação e comparação |

### `custo/` — as figuras de mérito

| arquivo | responsabilidade |
|---|---|
| `memoria.py` | **T1**: bytes, nas três variantes (`ordenado`, `hash`, `denso`). Conta LUTs, mapeamento, limiares e contador |
| `operacoes.py` | **T3**: operações por decisão, separando aproximador de codificador e amostrador |
| `realizado.py` | **validação cruzada do T1**: constrói `LUTOrdenada`/`LUTHash`/`LUTDensa` de verdade e MEDE, em vez de só modelar |

### `fusion/` — a Etapa 2 (ruído e Kalman)

| arquivo | responsabilidade |
|---|---|
| `ruido.py` | `ConfigRuido`, `CanalGPS` — a camada de corrupção |
| `kalman.py` | `FiltroKalmanControle` (degrau 4, a proposta), `PassaBaixa` (2), `FiltroVelocidadeConstante` (3), `FiltroGanhoEstacionario` |
| `ambiente_ruidoso.py` | costura filtro + ambiente; subclasse criada em tempo de execução, via `refenv.loader` |
| `consistencia.py` | NEES/NIS, faixas χ² — o que o gate RF-05 usa |
| `verdade.py` | **invariante RF-04**: tipos que impedem a verdade de entrar na observação |
| `bancada.py` | bancada sintética, sem simulador |

### `rl/` e `toy/` — as validações do algoritmo

| arquivo | responsabilidade |
|---|---|
| `rl/actor_critic.py` | Algoritmo 4.4 **isolado** do ambiente, para validar contra o Julia |
| `rl/sarsa.py` | Algoritmos 4.1 e 4.2 (SARSA de `m` passos) |
| `rl/politica.py` | `PoliticaGaussiana` da §4.3.3 |
| `toy/nav.py` | `NavAngular` — brinquedo sem simulador, onde o portão da SPEC-05 rodou |

---

## Parte II — Os oito pipelines

### 1. Treino WNN — Algoritmo 4.4

```
spec06_busca.py  (--modo grade|aleatoria|longa; --na/--ns escalares ou listas revezadas, ex. --na 1,2,4 --ns 12,24)
spec06_treino_multiagente.py  (Na > 1, parameter sharing; --config-de <grade>, --ns em lista)
        └─> spec06_treinar.py  ─> refenv · encoding · agents · campanha · wnn
                                └─> runs/…/checkpoints/{melhor,final,deploy}
```

Detalhado em [`pipeline-treinamento.md`](pipeline-treinamento.md).

### 2. Treino WNN — braço PPO

```
spec06_grid_ppo.py  ─> spec06_ppo_anbubus.py  ─> RLlib + NTNModel (anbubus)
                                              └─> spec06_importar_ppo.py
                                                  (tabela RLlib → nosso .npz)
```

Braço abandonado: 153 corridas, SR ≈ 0. Mantido porque o resultado negativo é
reportável.

### 3. Exportação de implantação

```
spec06_exportar_deploy.py  ─> poda 70 % por |v| ─> f32/uint16 ─> VERIFICA no val
                           └─> models/deploy/  (com fallback sem poda)
```

Roda automaticamente ao fim de todo treino (§ pós-treino padrão).

### 4. Campanha comparável — o experimento principal

```
spec06_campanha_fila.sh     (a fila de 33 corridas — a campanha publicada)
spec06_campanha_fila.py     (fila livre: --modelos × --celulas × --regimes, manifesto por corrida)
spec06_campanha_novos.sh    (embrulho da fila livre: modelos revezados, cada um só nas células do seu Na)
spec06_campanha_uma.sh      (uma corrida)
spec06_campanha_supervisor.py (mantém N em voo sem derrubar treino)
        └─> spec06_campanha_completa.py  ─> MADDPG | WNN T16 | WNN T32
                                         └─> reports/campanha/*.csv
```

100 runs × 200 episódios por célula, mesma semeadura para todos os modelos.

### 5. Baseline MADDPG

```
spec02_baseline_maddpg.py   (tamanho do modelo, dos pesos liberados)
spec02_maddpg_rollout.py    (carregar_ator, trilhas)
spec02_maddpg_campanha.py   (campanha R runs — o formato que a nossa copia)
```

### 6. Fusão e ruído (Etapas 2–3)

```
spec07_fusao.py           (bancada sintética: KF, NEES/NIS, detector)
spec08_gate_rf05.py       (ATESTADO por modelo e célula — bloqueio B1)
        └─> spec08_escada.py   (escada 0–4 × σ, L→R)
spec07_maddpg_ruido.py    (o mesmo, sobre o MADDPG)
```

O gate é **pré-requisito**: `spec08_escada.py` recusa rodar sem atestado
aprovado.

### 7. Custo e figuras de mérito

```
spec03_custo.py           (harness T1/T3 sobre redes sintéticas)
spec06_figuras_merito.py  (T1/T2/T3 sobre as redes REAIS treinadas)
spec06_orcamento.py       (T-06.0 — quanto custa simular)
```

### 8. Validação do núcleo — contra o código do Katopodis

```
spec04_diferencial.py        (NTupleRegressor × Julia)
spec04_diferencial_redes.py  (as duas redes × Ramnet.jl)
replica_katopodis_52.py      (§5.2.2: circular × termômetro)
spec05_{circular,identidade,mountaincar}.py  (os portões da SPEC-05)
```

É a evidência mais forte do projeto: casamento numérico com a implementação de
referência em Julia.

---

## Parte III — Ferramentas de acompanhamento

Executáveis que não entram em pipeline nenhum — você os roda direto.

| script | o que mostra |
|---|---|
| `spec06_acompanhar.py --multi` | grades do Alg. 4.4 e do PPO, e a escalada multiagente |
| `spec06_campanha_painel.py` | progresso da campanha comparável, com ETA global |
| `spec06_campanha_supervisor.py` | mantém N corridas em voo medindo o impacto no treino |
| `spec06_path.py` | imagem das trajetórias de uma política |
| `spec00_visualizar.py` | visualização 3D ao vivo de um episódio |
| `spec06_bancos.py` | inspeciona os bancos `val`/`test` |

---

## Parte IV — O que está órfão ou obsoleto

Levantado por alcançabilidade AST mais busca de citações.

| arquivo | situação |
|---|---|
| `spec06_aval_melhor.py` (59 L) | **órfão** — sem CLI, sem citação. Produziu `spec06-longa-melhor-ckpt.json`, que sustenta a §9 do grid-search |
| `spec06_aval_multi.py` (30 L) | **órfão** — idem, para as células multiagente |
| `spec06_treino_wnn.py` (265 L) | **obsoleto e declarado**: `modelo-e-hiperparametros.md` registra que foi a 1ª tentativa ad-hoc, sem a correção `(1−γ)`, e que abortou aos 6,84 M |

Os dois órfãos são scripts descartáveis que acabaram commitados. Não são lixo —
os artefatos deles estão em uso — mas quem os encontrar não vai saber se pode
apagar. O `spec06_treino_wnn.py` fica como registro histórico.

---

## Parte V — Como tudo se sustenta

Três invariantes atravessam a base inteira:

**A porta única.** Só `refenv/loader.py` carrega o simulador. Qualquer
modificação do problema do Lamenza teria de passar por ali.

**O contrato congelado.** `agents/base.py` define `Policy`, e MADDPG, WNN e MLP
o cumprem igualmente — é o que permite trocar o aproximador por parâmetro e
saber que o resto é idêntico.

**`val` decide, `test` reporta.** Parada antecipada, escolha de checkpoint,
fração de poda e sintonia do filtro consomem `val`. O `test` só aparece no
número final. Decisão que toca o teste transforma estimativa em ajuste.
