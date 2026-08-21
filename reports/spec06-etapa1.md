# SPEC-06 — Etapa 1: estado do portão G1

**Situação honesta: 4 dos 10 critérios fechados, a infraestrutura pronta e validada
de ponta a ponta, e o portão G1 dependendo de tempo de parede que não cabe numa
sessão.** Este relatório diz exatamente o que está medido, o que está construído, e
o comando que fecha o resto.

| # | critério | estado |
|---|---|---|
| 1 | P2, vazão, `B_max` | **FECHADO** — `reports/spec06-orcamento.md` |
| 2 | 1a reportada, `Na ∈ {1,2,4}` | infraestrutura pronta; **campanha em orçamento reduzido** |
| 3 | 1b reportada | infraestrutura pronta; não executada |
| 4 | tabela pareada WNN vs MLP | infraestrutura pronta; depende de 2 e 3 |
| 5 | varreduras | não executadas |
| 6 | tabela de custo §5.5 | parcial (T1/T3 medidos na SPEC-03 e T-06.0) |
| 7 | `R ≥ 3` sem descarte | **implementado e ativo** |
| 8 | veredito de H1 | depende de 2-4 |
| 9 | `make repro` | agregação por `eval_test.jsonl` implementada |
| 10 | RNF-07 | **FECHADO** — `tests/test_rnf07.py` |

Além disso: **P5 fechada** (o horizonte permissivo do §7.1) e a interação
`max_seconds_stalled` × RF-02 **medida**.

## O que está medido

### P5 — o horizonte permissivo, e o piso é que vale

Lamenza não publica o valor; a T-06.1 fixa `3 × CT_p99(guloso)`, múltiplos de 100 s,
piso 500, teto 2000. Medido:

| célula | `CT_p99(guloso)` | `3×` | horizonte | SR do guloso |
|---|---|---|---|---|
| 1a12s | 43,0 s | 129 | **500 s** (piso) | **100,0 %** |
| 2a12s | 43,8 s | 131 | **500 s** (piso) | **100,0 %** |
| 4a12s | 39,7 s | 119 | **500 s** (piso) | **100,0 %** |

O `3×` fica muito abaixo do piso, então o horizonte permissivo **é** os 500 s da
tab. 7.1 — a definição operacional não introduz parâmetro novo, que é o melhor
desfecho possível para uma pendência de valor não publicado. E a verificação
funcional passa: o horizonte não é o gargalo.

### `max_seconds_stalled` × RF-02 — 30,5 s, previsão confirmada

A T-06.1 mandava confirmar contra o código antes de 1a rodar. Medido: **30,5 s
exatos, com variância zero** (min = max) nas três células — os 30 s do parâmetro mais
um passo de decisão de 0,5 s. São 30,5 de 64,5 s de episódio, ou **47 % de
experiência pós-conclusão**, o que sustenta a atribuição de ganho grande que Lamenza
faz no §5.

Duas medições anteriores deram **0,0 s**, e as duas eram erro meu do mesmo tipo:
primeiro por usar `phase="eval"` (que encerra na coleta), depois porque
`run_episode` quebra na **primeira `info` não vazia** — verbatim do
`visualize_performance.py`. Ele é um runner de **avaliação** e estruturalmente não vê
o período pós-conclusão. Zero por construção é o modo de falha mais perigoso de uma
medição: parece resultado.

## Três correções ao protocolo, todas descobertas por execução

A §4 é pré-registrada, então cada correção é registrada com a medição que a motivou.

### 1. A eq. 7-2 não existe no ambiente

A T-06.1 exige a recompensa `R_t = (Σ visited)/Ns` do §7.1. O ambiente implementa
`punish` (eq. 5-6) e o incremento — não a 7-2. Calculá-la **fora** não é modificar o
problema: a recompensa de aprendizado é escolha do agente e o ambiente expõe
`sensors_collected`.

E a diferença entre as duas **não é uma constante**: a 7-2 é não negativa, então um
episódio mais longo acumula **mais** retorno — ela não penaliza demora. É exatamente
o que o §7.1 descreve (*"reduces the urgency to complete the scenario as quickly as
possible"*). Confundi-las faria a réplica do §7.1 medir o problema do §7.2 com
horizonte generoso, que é outra coisa.

### 2. A parada antecipada da §4.4 dispara antes de o treino começar

O critério — *"a média móvel não melhora mais que 1 % ao longo de 8 avaliações"* —
dispara também com a métrica **achatada no zero no início**. Medido: numa corrida de
fumaça declarou *"convergiu em 1950 passos"* de 6000, com `SR = 0` do começo ao fim.

Isso confunde *"platô depois de aprender"* com *"ainda não começou"* e libera
orçamento de quem mais precisa dele. Corrigido com uma guarda de **metade das
avaliações** antes de a parada poder disparar — a leitura mínima defensável do
critério, que detecta platô **após** aprendizado.

### 3. `SR` e `CT_cens` saturam, e selecionar por elas é selecionar ao acaso

**A caminhada aleatória também dá `SR = 0,0 %`**, e `CT_cens` fica em 500 s para
todo mundo. Com as duas achatadas:

- a seleção de checkpoint da §4.4 escolhe **ao acaso**;
- o critério de falha catastrófica (*"não supera a aleatória com IC separado"*)
  dispara para **qualquer** corrida sub-treinada, e deixa de distinguir "falhou" de
  "não terminou de treinar".

Separadas: a **seleção** usa `cobertura` (fração média de sensores coletados), que
tem sinal desde o primeiro passo, é o próprio objetivo da eq. 7-2 e é monótona em
relação a `SR` (`cobertura = 1 ⟺ sucesso`). O **reporte** continua `SR`, como a
T-06.1 manda. A §4.4 escolhe a métrica de validação e nada obriga que ela seja a de
reporte.

## A asserção da SPEC-05 §4.2 pagou por si

A T-06.1 previu o risco:

> *"A recompensa 7-2 é não negativa e monotonicamente não decrescente… com `γ = 0,99`
> e horizonte permissivo, `V` se aproxima de `≈ 100 × fração_visitada`… os alvos de TD
> são grandes desde o começo. **Normalizar o alvo de valor por `(1−γ)`**… precisa ser
> decidido e registrado antes da campanha, não descoberto no meio dela."*

Medido: **sem a normalização, a asserção aborta no passo 998** com
`passo 0,6742 excede c_max=0,5`. A §4.3 classifica isso como **bug, não resultado**,
e a correção — `escala = (1−γ) = 0,01` — está registrada em código antes da campanha,
com o fator `1/(1−γ) = 100` que a previsão antecipou. Não é alteração do MDP: um
fator escalar positivo na recompensa não muda a política ótima.

Vale registrar o encadeamento: a asserção foi escrita na SPEC-05 por causa de uma
divergência medida no protótipo, e cobrou o seu preço na primeira corrida da SPEC-06,
no caso exato que a SPEC-06 havia previsto.

## O que a infraestrutura entrega

`src/uavwnn/campanha/`, pré-registrado conforme a §4:

- **§4.1** os três papéis de semente, com a **D-06.1** (`seed_tuple` amarrada a
  `seed_train`, porque o ladrilhamento é parte do modelo) e a corrida de controle;
- **§4.2** bancos `test` (500) e `val` (100), disjuntos **por construção** e não por
  filtragem, com hash sobre a lista **ordenada** — a ordem faz parte do pareamento;
- **§4.4** parada antecipada com média móvel de 5, paciência de 8, melhora mínima de
  1 %, checkpoint de melhor validação, e a definição operacional de falha
  catastrófica;
- **§4.7** um diretório por corrida com `config.json` (hash da configuração
  **resolvida**), `env.json` (commit, versões, CPU, hash dos bancos),
  `metrics.jsonl` por avaliação intercalada com **ocupação das LUTs** (RNF-08),
  `checkpoints/{melhor,final}`, e `eval_test.jsonl` **por episódio** — sem o qual o
  IC da diferença não é reconstruível.

Validado de ponta a ponta: treino, avaliação intercalada, parada, seleção de
checkpoint, avaliação final no `test` e detecção de falha catastrófica.

## O que falta, e o custo

O `B_max` medido em T-06.0 é **19,5 M passos por configuração-semente**. A 848
passos/s isso é **6,4 h por corrida**. A 1a do critério 2 são 3 células × 5 sementes
= 15 corridas = **96 h de CPU**; com os dois braços (WNN e MLP), **192 h**. Em 12
núcleos, ~16 h de parede por braço.

Isso não cabe numa sessão, e o que roda aqui é uma campanha de **orçamento
reduzido** — `B_max = 400 000`, ou **2 % do medido** — com `R = 3`. Ela existe para
validar o arnês e dar sinal, **não** para fechar o critério 2: com 2 % do orçamento,
`SR = 0 %` é afirmação sobre o orçamento, não sobre o método.

```bash
# o que fecha o critério 2 no orçamento medido
uv run python scripts/spec06_treinar.py --replica 1a --aproximador ntupla \
    --celulas 1:12,2:12,4:12 --sementes 5 --b-max 19500000
uv run python scripts/spec06_treinar.py --replica 1a --aproximador mlp \
    --celulas 1:12,2:12,4:12 --sementes 5 --b-max 19500000
```

**O portão G1 não é declarado.** Declará-lo com 2 % do orçamento seria afirmar mais
do que se mediu — o mesmo erro que a §4.6 proíbe para o IC de nível 1.

## Reprodução

```bash
uv run python scripts/spec06_orcamento.py --episodios 3 --episodios-p2 40
uv run python scripts/spec06_bancos.py --celulas 1:12,2:12,4:12
uv run python scripts/spec06_treinar.py --replica 1a --celulas 1:12 --sementes 5
```

Artefatos: `reports/spec06-orcamento.json`, `reports/spec06-bancos.json`,
`reports/spec06-1a-*.json`, e um diretório por corrida em `runs/spec06-1a/`.
