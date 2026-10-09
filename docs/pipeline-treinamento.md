# Pipeline de treinamento — arquivos, fluxo de dados e responsabilidades

Mapa do que roda quando se treina uma WNN neste projeto. O grafo abaixo é o
**grafo real de imports**, não um desenho aproximado.

## 1. Visão geral

```
                    ┌──────────────────────────────────────┐
   DRIVERS          │ spec06_busca.py      (grade / longa) │
   (scripts/)       │ spec06_treino_multiagente.py (Na>1)  │
                    └───────────────┬──────────────────────┘
                                    │ ambos importam
                    ┌───────────────▼──────────────────────┐
   MONTAGEM         │ spec06_treinar.py                    │
   (scripts/)       │  _cfgs · _ctx · avaliar · RECOMPENSAS│
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬────────────────┐
        ▼               ▼           ▼           ▼                ▼
   refenv/         encoding/    agents/     campanha/         wnn/
   (ambiente)      (bits)       (política)  (protocolo)       (LUTs)
```

Duas fronteiras importam:

* **`refenv/loader.py`** é o único ponto que LOCALIZA e CARREGA o simulador do
  Lamenza (submodule `vendor/gradys-data-collection`). É o que torna
  "não modificar o problema" verificável num arquivo só.
  `fusion/ambiente_ruidoso.py` estende a classe de referência para injetar ruído
  de GPS — mas a obtém **através desse loader**, e a subclasse é criada em tempo
  de execução, não de importação: o módulo importa sem o `gradysim` presente.
* **`wnn/redes.py`** não sabe o que é RL. Recebe bits e um passo, devolve
  números. Quem decide o que é o passo é o Algoritmo 4.4.

## 2. O fluxo de UMA transição

É este ciclo que roda milhões de vezes. Segue o caminho de um único passo de
ambiente, com o arquivo responsável por cada etapa:

```
 ┌─ 1. env.step / reset ──────────────── refenv/runner.py ──┐
 │      obs: 27 floats (posições relativas, normalizadas)   │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌─ 2. Observacao.decodificar ────────── agents/base.py ────┐
 │      vetor cru → objeto com semântica (o contrato)       │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌─ 3. codificar ──────────── encoding/thermometer.py ──────┐
 │      27 floats → 864 BITS  (T=32 níveis por dimensão)    │
 │      os limiares são ESTADO TREINADO, vêm dos quantis    │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌─ 4. aproximador.f(x) ───────────────── wnn/redes.py ─────┐
 │      bits → endereços (N RAMs) → soma das LUTs → 3 saídas│
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌─ 5. política ──────── agents/politica_continua.py ───────┐
 │      3 saídas → ação. TREINO: amostra. AVAL: média       │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌─ 6. recompensa ───────── campanha/recompensas.py ────────┐
 │      calculada FORA do ambiente (eq. 5-6 ou 7-2)         │
 └──────────────────────────┬───────────────────────────────┘
                            ▼
 ┌─ 7. Algoritmo 4.4 ──────── agents/ac_continuo.py ────────┐
 │      registrar(r) → buffer                               │
 │      quando |buffer| = m:                                │
 │         g     = Σ γᵏ r + γᵐ·v(x')      ← retorno m-passos│
 │         δ     = g − v(x)               ← a SURPRESA      │
 │         ator    += α·δ·∇log π(z|f)     ← eq. (4.11)      │
 │         crítico  = φ·v + g             ← eq. (4.8)       │
 └──────────────────────────────────────────────────────────┘
```

O passo 7 é o único ponto em que as duas redes são escritas, e a ordem lá tem
consequência: `δ` usa o valor **antes** de o crítico treinar. Inverter faria o
crítico aprender a resposta e se surpreender menos com ela — o sinal do ator
encolheria sem que nada acusasse.

## 3. Responsabilidade de cada arquivo

### Drivers — decidem O QUE treinar

| arquivo | responsabilidade |
|---|---|
| `spec06_busca.py` | grade de hiperparâmetros e corrida longa (`--modo grade\|aleatoria\|longa`). `--na 1,2,4` e/ou `--ns 12,24` treinam cada ponto com uma rede só, revezando a célula `(Na, Ns)` por episódio sobre o produto das listas, e validam/testam em cada célula. Paraleliza por processo, retoma do cache, grava `melhor`/`final`/`deploy` |
| `spec06_treino_multiagente.py` | célula `Na > 1`. **Parameter sharing**: uma rede, `Na` trajetórias de AC. Normaliza a recompensa por `Na`. `--config-de <json da grade>` usa o melhor ponto da grade daquele `Na` para o `T` pedido; `--ns 12,24,36` reveza o Ns por episódio (célula `a2s12-24-36`) |
| `spec06_campanha_fila.py` | fila da campanha com escolha livre de modelos × células × regimes (`--modelos`, `--celulas 2:12,24,36`); manifesto por lançamento com o checkpoint usado, e recusa reaproveitar CSV de outro checkpoint |

### Montagem — traduz configuração em objetos

| arquivo | responsabilidade |
|---|---|
| `spec06_treinar.py` | `_cfgs` (config de treino × avaliação — elas DIFEREM), `_ctx`, `avaliar`, `RECOMPENSAS`, `REPLICAS`, `corpus_de_observacoes` |

### Ambiente — a fronteira com a referência

| arquivo | responsabilidade |
|---|---|
| `refenv/loader.py` | localiza e carrega o submodule; nenhum outro módulo o importa |
| `refenv/runner.py` | `EnvConfig`, `make_env`, `run_episode`, semeadura. `LAMENZA_003` é a config canônica da tab. 7.1 |

### Codificação — floats viram bits

| arquivo | responsabilidade |
|---|---|
| `encoding/thermometer.py` | `TermometroQuantil`: ajusta limiares por quantis e binariza. **Estado treinado** — viaja com o modelo e entra no T1 |
| `encoding/corpus.py` | colhe as observações que calibram os limiares |
| `encoding/artigo.py` | codificadores literais da dissertação, para os testes diferenciais |

### Política e algoritmo

| arquivo | responsabilidade |
|---|---|
| `agents/base.py` | interface `Policy` congelada, `Observacao`, `Acao`, `ContextoEpisodio` — o contrato que os três braços cumprem |
| `agents/ac_continuo.py` | **o Algoritmo 4.4**: buffer de `m` passos, retorno, `δ`, e as duas atualizações. Também `APROXIMADORES` (a família selecionável por nome) |
| `agents/politica_continua.py` | `NormalProjetada`: 3 saídas → ação, e o `∇log π` que escala a atualização do ator |
| `agents/mlp_ac.py` | MLP pareada — o braço de controle que roda o MESMO laço |

### Redes — a memória

| arquivo | responsabilidade |
|---|---|
| `wnn/redes.py` | `RedeGradienteFuncional` (ator, §4.3.1 — só valores) e `RedeRegressaoEsquecimento` (crítico, §4.2.1 — valor + contador). Endereçamento, `salvar`/`carregar` |
| `wnn/ntuple.py` | variante densa, usada nos testes de identidade |

### Protocolo — o que torna a medição comparável

| arquivo | responsabilidade |
|---|---|
| `campanha/bancos.py` | bancos `val` e `test` **versionados**. Regenerá-los com outro NumPy mudaria os cenários com a mesma semente, e as comparações já registradas deixariam de valer sem erro aparente |
| `campanha/sementes.py` | os três papéis de semente (treino, avaliação, tupla) — separá-los é o que permite variar um sem contaminar os outros |
| `campanha/treino.py` | `ParadaAntecipada` (média móvel 5, paciência 8, mínimo 20 avaliações) |
| `campanha/recompensas.py` | as duas recompensas de Lamenza, calculadas fora do ambiente |
| `campanha/normalizacao.py` | `MeanStdFilter`, para o braço PPO |

### Saída

| arquivo | responsabilidade |
|---|---|
| `spec06_exportar_deploy.py` | `checkpoints/deploy`: só o ator, `float32`/`uint16`, podado 70 %, **verificado** contra o banco `val` com fallback |
| `custo/memoria.py` · `custo/operacoes.py` | T1 e T3 — contam bytes e operações a partir da rede real |

## 4. Artefatos que o treino produz

```
runs/<experimento>/<hash>/<semente>/
├── config.json          hiperparâmetros + hash + papéis de semente
├── env.json             commit, versões, hash dos bancos  ← procedência
├── metrics.jsonl        uma linha por avaliação intercalada (a curva)
├── eval_test.jsonl      uma linha por episódio do teste final
└── checkpoints/
    ├── melhor/          melhor validação — CONGELA CEDO, ver §9 do grid-search
    ├── final/           estado ao fim do treino — é o que models/ publica
    └── deploy/          só o ator, f32/u16, podado e verificado
```

## 5. As três separações que sustentam o resto

**Treino × avaliação.** `_cfgs` devolve duas configs: o treino usa
`end_when_all_collected=False` (dá experiência pós-conclusão); a avaliação usa
`True`, que é o protocolo do `visualize_performance.py` de referência. Usar uma
no lugar da outra muda a métrica sem erro nenhum.

**`val` × `test`.** Toda decisão — parada antecipada, escolha de checkpoint,
fração de poda — consome `val`. O `test` só aparece no número que vai ao
relatório. Decisão que toca o teste transforma estimativa em ajuste.

**Codificador fora do aproximador.** Os três braços recebem a MESMA observação e
cada um a prepara à sua maneira (§4.8). É a única assimetria permitida — e por
isso os limiares do termômetro entram na conta de memória da WNN.
