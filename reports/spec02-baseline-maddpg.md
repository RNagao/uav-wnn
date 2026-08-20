# Baseline MADDPG — tamanho e latência dos pesos publicados

Reprodução:

```bash
uv sync --group baseline                                       # ~200 MB, só para ler os .cleanrl_model
uv run --group baseline python scripts/spec02_baseline_maddpg.py
```

---

## 1. O achado: não precisamos treinar o MADDPG

O submodule de referência (`vendor/gradys-data-collection`, tag `dissertation`)
traz mais do que o cenário. Traz **o treinamento e os pesos**:

| item | onde |
|---|---|
| código de treino MADDPG, base CleanRL | `scenario/003-collaboration/main.py` |
| **9 checkpoints treinados**, `Na ∈ {2,4,8} × Ns ∈ {12,24,36}` | `runs/results/*/…cleanrl_model`, 45 MB |
| harness de avaliação | `scenario/003-collaboration/test.py` |
| logs do TensorBoard das corridas dele | `events.out.tfevents…` |

Carregar e medir levou minutos. Retreinar seriam **~7 dias de GPU por célula**,
pelo número declarado na própria dissertação de Lamenza.

## 2. Medição

Ator `obs → 256 → 256 → 2`, ReLU nas ocultas, `tanh` na saída. Crítico
`512 → 512 → 1`, centralizado. Larguras são os defaults do `main.py`
(linhas 108–109) — não palpite: é o que gerou estes pesos.

| célula | obs | params do ator | ator fp32 | mediana | p99 | mult. fp |
|---|---|---|---|---|---|---|
| `a_2-s_{12,24,36}` | 27 | 73 474 | **287,0 KB** | 36,5 µs | 155,4 µs | 72 960 |
| `a_4-s_{12,24,36}` | 31 | 74 498 | **291,0 KB** | 35,0 µs | 115,1 µs | 73 984 |
| `a_8-s_{12,24,36}` | 39 | 76 546 | **299,0 KB** | 35,3 µs | 106,8 µs | 76 032 |

*(latência: PyTorch em CPU, 1 thread, batch 1, fp32, 8 000 repetições após 2 000 de aquecimento)*

Crítico, para a coluna ator+crítico: 331 265 params (`Na=4`), **1 294 KB** — mais
de 4× o ator. Ele recebe observações e ações de **todos** os drones, que é o que
faz o MADDPG ser MADDPG. Não entra na razão de tamanho da §5.5, que compara
atores, mas precisa estar medido.

### Duas validações cruzadas que saíram de graça

**A fórmula da observação do PRD está certa.** `D = 2(Na − 1) + 25` prevê
27/31/39, e as formas de `fc1.weight` nos checkpoints dão 27/31/39. Se a nossa
leitura da eq. 5-1 de Lamenza estivesse errada, a largura de entrada não fecharia.

**`N*_s = 12` é fixo.** `obs_dim` não muda com `Ns ∈ {12,24,36}` — a observação
sempre traz os 12 sensores mais próximos, com zero-padding quando há menos. É o
que o PRD §5.2 afirma, agora confirmado nos pesos.

## 3. O que isto muda

**A pendência P4 fecha por medição.** Ela pedia "extrair do repositório a
arquitetura do ator; sem ela, a razão de tamanho contra a referência de Lamenza
não passa de estimativa". A estimativa registrada era *"padrão CleanRL,
256×256, ≈74 000 params, ≈298 KB"*. **O palpite estava certo** — e agora a razão
da §5.5 deixa de sair marcada como estimativa:

| | tamanho | razão contra a WNN de 133,9 KB |
|---|---|---|
| WNN, chaves ordenadas | 133,9 KB | — |
| **ator do MADDPG** (`Na=4`) | **291,0 KB** | **0,46×** |
| MLP pareada 25-32-32-2 | 7,6 KB | 17,5× |

A leitura permanece a mesma que a SPEC-03 já reportava: **a WNN é menor que o
ator do MADDPG e muito maior que a MLP pareada.** O que muda é que metade da
razão dupla deixou de ser estimativa.

**A decisão D7 se estreita, não cai.** Ela diz *"MADDPG não é reproduzido"*, com
a justificativa de que *"reproduzi-lo exigiria replicar exatamente simulador,
observação e recompensa; qualquer divergência invalidaria a comparação
silenciosamente"*.

Essa justificativa **não se aplica ao que foi feito aqui**: não reproduzimos
nada. Carregamos os pesos dele, no ambiente dele, que é o nosso submodule. O
risco de divergência que a D7 temia não existe. O que continua valendo é
**não retreinar** — e isso é o que custaria os 7 dias por célula.

## 4. Três ressalvas, e as duas primeiras vão para o relatório final

**1. Não há checkpoint para `Na = 1`.** As células publicadas começam em 2
agentes. A Etapa 1 do nosso plano começa em `Na = 1`, e ali **não existe baseline
MADDPG** — a comparação com Lamenza só é possível de `Na = 2` em diante. Para
`Na = 1` as referências continuam sendo a heurística e a MLP pareada.

**2. Os microssegundos não são comparação justa de modelo.** São PyTorch em CPU,
com sobrecarga de arcabouço e de Python; a nossa WNN em NumPy carrega sobrecarga
análoga. Medir tempo de parede favorece quem tem a biblioteca mais otimizada, não
o modelo mais barato. **A comparação independente de arcabouço é a contagem de
operações (T3)**, e é exatamente por isso que a pendência **T-03.6** (paridade de
arcabouço) existe — ela sobe de prioridade agora que a latência do MADDPG é
mensurável.

O contraste que se sustenta é o estrutural: **~73 000 multiplicações em ponto
flutuante por decisão, contra zero no nosso aproximador.**

**3. Tempo de conclusão não sai daqui.** Exige rodar a política no ambiente. O
`test.py` do submodule já faz isso — mede `completion_time`,
`avg_collection_time` e `all_collected` — e é tarefa da SPEC-02, não deste
script. Quando rodar, será a comparação mais forte possível: a política dele e a
nossa, no mesmo simulador, no mesmo harness de avaliação.

## 5. Sobre a dependência nova

`torch` entra como **grupo opcional** `baseline`, não como dependência padrão.
São ~200 MB para ler nove arquivos, e o `.cleanrl_model` é um `torch.save` — não
há como abri-lo sem torch. A instalação padrão segue sem ele, e o script falha
com instrução em vez de `ImportError`.

Os testes têm dois níveis: `tests/test_baseline_maddpg.py` confere o artefato
congelado contra as fórmulas do PRD **sem torch e sem submodule** (é o que roda na
suíte padrão), e relê os checkpoints quando os dois estão presentes.
