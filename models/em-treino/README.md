# Modelos em treino (parciais — continuar em outra máquina)

Checkpoints que **não fecharam** e ficam aqui para serem retomados/concluídos
noutra máquina. Não são modelos de deploy — os prontos estão em
[`../`](../README.md) (`a1s12`, `a2s24`, `a4s24`).

## `ntupla_dissertacao-a8s24` — Na=8, Ns=24 (parcial, SR ~0%)

Melhor validação da escalada (parou antes de fechar: clusters/gaps, ~cob 0,83,
SR ~0). Mesma config parcial de sempre (passo 8e-4, σ 0,35, T 32, n 12, φ 0,85,
m 8). Traz `ator`, `critico` e `termometro` (limiares treinados) — o estado
completo para continuar.

### Continuar de onde parou (retoma o parcial)

Na outra máquina, após clonar o repo e instalar (`uv sync`):

```bash
uv run python scripts/spec06_treino_multiagente.py \
  --na 8 --ns 24 --b-max 10000000 \
  --retomar models/em-treino/ntupla_dissertacao-a8s24 \
  --saida runs/spec06-multi/ntupla_dissertacao-a8s24
```

`--retomar` carrega ator/crítico/termômetro daqui e segue treinando; grava o
novo `melhor`/`final` em `--saida` (`runs/` é gitignored). Ao fechar, copie o
`melhor` de volta para `models/ntupla_dissertacao-a8s24/` e mova a entrada para o
README principal.

### Alternativa: recomeçar do zero

O parcial está num vale ruim (SR 0). Retomar pode ficar preso nele; um **fresh
run** com outra semente às vezes escapa. Para recomeçar, é só **omitir**
`--retomar` (a config está toda no driver):

```bash
uv run python scripts/spec06_treino_multiagente.py --na 8 --ns 24 --b-max 10000000
```

### Contexto

Na=2 e Na=4 fecharam (SR 91% e **100%**); Na=8 é o limite provável do **crítico
descentralizado** (Alg. 4.4) vs o CTDE do MADDPG. Concluir o Na=8 (retomando ou
fresh, idealmente com mais orçamento/sementes) fecha a escalada 2/4/8. Ver
[`../../reports/relatorio-geral.md`](../../reports/relatorio-geral.md).
