# SPEC-06 — configurações vencedoras por célula (Na) e por T

Registro vivo: a cada grade que fecha, entra aqui o melhor ponto de **T = 16** e de
**T = 32** para aquele `Na`. É daqui que o treino longo parte, e é isto que o
relatório cita ao dizer "a configuração do modelo de `Na = k`".

A métrica de seleção é a da grade: **cobertura média entre sementes** (3) e, nas
grades com `--ns` em lista, **média entre as células de Ns**. `SR` e abortos ficam
registrados porque a grade de `Na = 1` mostrou que o 1º por cobertura pode viver
num nível de passo que não sustenta o próprio ótimo no orçamento longo.

Todas as grades: `ntupla_dissertacao`, réplica 1b, 200 mil passos por corrida,
`m = 8`, sem normalização, nível `desempate` (passo × σ × T × n × φ). O gêmeo
legível por máquina é `spec06-vencedores.json`.

---

## Na = 1 · Ns = 12

Grade: `reports/spec06-desempate.json` (720 corridas, 2 modelos; 120 pontos da
dissertação). Relatório: `spec06-grid-search.md` §9.

| T | id | passo | σ | n | φ | N | cobertura | SR | abortos | top 3 (ids) |
|---|---|---|---|---|---|---|---|---|---|---|
| **32** | 67 | 8e-4 | 0,35 | 12 | 0,85 | 67 | 50,1 % | 21,3 % | 0/3 | 67 · 31 · 42 |
| **16** | 61 | 8e-4 | 0,35 | 12 | 0,85 | 34 | 46,0 % | 0,7 % | 0/3 | 61 · 36 · 25 |

Treinados a 50 M (`models/T32/a1s12`, `models/T16/a1s12`) com exatamente estas
configs — a T-06.4. Os dois empatam em SR no orçamento longo (93 %, duas sementes
a 6,75 M); ver `spec06-grid-search.md` §9.

## Na = 2 · Ns = 12 / 24 / 36 (revezado)

Grade: `reports/spec06-busca-a2s12-24-36.json` (360 corridas, 120 pontos, 8,7 h,
138 abortos). Fechou em 2026-10-10.

| T | id | passo | σ | n | φ | N | cobertura | SR | abortos | a2s12 | a2s24 | a2s36 | top 3 (ids) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **32** | 90 | 1,5e-3 | 0,35 | 12 | 0,70 | 72 | 82,9 % | 51,0 % | 1/3 | 74 / 54 | 86 / 53 | 88 / 46 | 90 · 68 · 56 |
| **16** | 61 | 8e-4 | 0,35 | 12 | 0,85 | 36 | 71,7 % | 17,1 % | 0/3 | 58 / 17 | 75 / 18 | 82 / 16 | 61 · 62 · 50 |

Nas células, o par é cobertura / SR em %.

**Ressalva no T = 32.** O 1º (id 90) está em `passo = 1,5e-3`, nível que abortou
53 de 72 corridas nesta grade e 12 de 12 na de `Na = 1`; o 2º (id 68: `8e-4 ·
σ 0,35 · φ 0,99`, 81,3 % cob, SR 43,2 %, **0/3 abortos**) fica a 1,6 pp.

**Treino longo (lançado em 2026-10-14):** `T = 16` com o id 61; `T = 32` com o
**id 68**, não o 1º. O primeiro lançamento do `T = 32` caiu na asserção de
estabilidade a 1,08 M passos (`passo 0,51 > c_max 0,5`, pico isolado, antes do
primeiro checkpoint); relançado com `--c-max none`, o mesmo botão usado no
`T32-a2s24` publicado, que caiu igual a 40 M. A curva de validação é quem vigia
divergência. Corridas em `runs/spec06-50M/T{16,32}-a2s12-24-36`. O `T = 16`
também caiu na asserção, a 29,05 M (`passo 0,5037`), e foi retomado do `ultimo`
(28,75 M) com `--c-max none`.

**Resultado dos 50 M (fechados em 2026-10-16), banco de teste, 100 sementes por Ns:**

| T | checkpoint | cobertura | SR | CT | a2s12 | a2s24 | a2s36 |
|---|---|---|---|---|---|---|---|
| 16 | `final` (50 M) | 95,5 % | 85,3 % | 109,5 s | 95 / 93 | 94 / 78 | 98 / 85 |
| 16 | `melhor` (18,75 M) | 95,1 % | 89,7 % | 89,6 s | 95 / 93 | 93 / 86 | 97 / 90 |
| 32 | `final` (50 M) | 89,6 % | 72,3 % | 173,3 s | 86 / 76 | 89 / 67 | 94 / 74 |
| 32 | `melhor` (1,25 M) | 96,1 % | 77,0 % | 142,7 s | 95 / 80 | 95 / 74 | 98 / 77 |

**O `T = 16` ganha do `T = 32` em `Na = 2`**, por 13 pp de SR no `final`, com
metade das RAMs e deploy podado de 259 KB (contra 1 567 KB sem poda no `T = 32`,
cuja poda de 70 % derrubou o SR de 78 para 66 % e foi recusada).

**Alerta no `T = 32`:** o melhor ponto da curva de validação é o PRIMEIRO
(1,25 M); dali em diante a cobertura oscilou entre 78 e 90 % e nunca voltou. O
`final` está abaixo do `melhor` em cobertura e SR — o oposto do que as células
publicadas mostraram (`models/README.md`), onde o `final` sempre ganhava. Com a
asserção desligada, esta é a corrida em que a regra "promover o `final`" precisa
ser olhada antes de aplicada.

Leitura dos eixos (média marginal das 360 corridas): `n = 12` 43 % contra
`n = 24` 9 %; `passo` pica em 8e-4 e 3e-3 aborta 72/72; `σ = 0,35` melhor e mais
estável que 0,20 (53 contra 85 abortos); `φ` indiferente (26 % nos três níveis);
`T = 32` ganha de 16 por ~10 pp de cobertura e 2–3× em SR.

## Na = 4 · Ns = 12 / 24 / 36 (revezado)

Grade: `reports/spec06-busca-a4s12-24-36.json` (360 corridas, 120 pontos, 20,9 h,
80 abortos). Fechou em 2026-10-11.

| T | id | passo | σ | n | φ | N | cobertura | SR | abortos | a4s12 | a4s24 | a4s36 | top 3 (ids) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **32** | 92 | 1,5e-3 | 0,35 | 12 | 0,99 | 83 | 96,6 % | 75,9 % | 0/3 | 95 / 77 | 97 / 76 | 98 / 75 | 92 · 90 · 55 |
| **16** | 86 | 1,5e-3 | 0,35 | 12 | 0,99 | 42 | 93,3 % | 63,4 % | 0/3 | 89 / 63 | 95 / 64 | 96 / 64 | 86 · 84 · 85 |

Nas células, o par é cobertura / SR em %.

Os dois vencedores têm a MESMA config fora de `T`: `passo 1,5e-3 · σ 0,35 · n 12 ·
φ 0,99`, sem abortos nas 3 sementes. Diferente de `Na = 1` e `Na = 2`, aqui
`passo = 1,5e-3` não é território morto: 19 abortos em 72 (contra 53/72 em
`Na = 2` e 12/12 em `Na = 1`), e os pontos com `σ = 0,35` nesse nível não
abortaram. A recompensa dividida por `Na` encolhe o sinal por drone, o que
explica o passo maior caber.

Leitura dos eixos (média marginal das 360 corridas): `n = 12` 63 % contra
`n = 24` 20 %; `passo` empata 8e-4 e 1,5e-3 (52 %) e 3e-3 aborta 61/72;
`σ = 0,35` 45 % contra 37 % (25 contra 55 abortos); `φ = 0,99` é o melhor nos
vencedores mas o pior na média (37 %) — interage com o passo; `T = 32` ganha de
16 por 3 pp de cobertura e 12 pp de SR, margem menor que em `Na = 2`.

**Treino longo lançado** (`T = 32` em 2026-10-15, `T = 16` em 2026-10-16), ambos
pelo `--config-de` (ids 92 e 86) e com `--c-max none` desde o início — depois de
as duas corridas de `Na = 2` caírem em picos isolados de 1–2 % acima do teto.
Corridas em `runs/spec06-50M/T{16,32}-a4s12-24-36`.

**As duas COLAPSARAM** (constatado em 2026-10-17), e a decisão de desligar a
asserção foi errada para este `passo`:

| T | pico da validação | colapso | depois | parou |
|---|---|---|---|---|
| 16 | 2,5 M: 96,7 % · SR 83 % | 6,25 M → 4,2 % | 27 M de passos a 3,5 % | 33,7 M, parada antecipada leu "platô" |
| 32 | 1,25 M: 98,7 % · SR 93 % | 12,5 M → 4,3 % | 18 M de passos a 3,5 % | 30,7 M, interrompida à mão |

3,5 % de cobertura é o nível da caminhada aleatória: a política divergiu e não
voltou. É o que `c_max = 0,5` teria abortado na hora. Os `melhor` (2,5 M e
1,25 M) estão em disco e são bons — o `T = 16` dá 97,8 % / SR 86 % no teste —,
mas são redes de início de treino, não os 50 M que a comparação com Lamenza
pede. O `final` do `T = 16` e o deploy gerado dele NÃO servem.

Lição: a grade a 200 mil passos seleciona um `passo` que o treino longo não
sustenta — em `Na = 1` isso já tinha aparecido no 2º colocado (`grid-search.md`
§9). O treinador ganhou uma **guarda de colapso** (3 avaliações seguidas abaixo
de ¼ do melhor encerram a corrida) para isso não custar dias de novo.

### Autópsia do colapso (2026-10-17)

Comparação dos checkpoints `melhor` e `ultimo` e rollout determinístico de cada
um em `a4s24` (5 sementes):

| | T16 `melhor` (2,5 M) | T16 `ultimo` (33,7 M) | T32 `melhor` (1,25 M) | T32 `ultimo` (30 M) |
|---|---|---|---|---|
| ator, p99 de \|(f₀,f₁)\| | 5,7 | **16,0** | 3,3 | **18,8** |
| ator, máx de \|(f₀,f₁)\| | 35 | **423** | 24 | **444** |
| crítico, mediana de V | +1 467 | **+53** | +1 084 | **+141** |
| crítico, p1 de V | +11 | **−858** | −28 | **−852** |
| velocidade pedida no rollout | 0,98 (97 % a 1,0) | **0,00 (100 % a 0)** | 0,90 (71 % a 1,0) | **0,00 (100 % a 0)** |
| f médio no rollout | (−0,45, 0,18, **+4,0**) | (−6,0, 30,0, **−26,9**) | (−0,15, −0,06, **+1,5**) | (4,8, 1,2, **−15,9**) |
| cobertura · passos/ep | 0,94 · 98 | 0,01 · **61** | 1,00 · 79 | 0,01 · **61** |

**Mecanismo.** A terceira saída do ator é a velocidade (`s = clip(f₂, 0, 1)`).
Na rede colapsada, `f₂` vale −16 a −27 em TODOS os estados visitados: velocidade
zero, os quatro drones param onde nasceram, nenhum sensor novo é coletado, o
`environment.py` trunca por `stalled` aos 30 s (61 passos), e a cobertura é só o
que estava embaixo do ponto de partida. É um poço sem saída: parado, a
recompensa é negativa a cada passo, o retorno é muito negativo, o crítico
acompanha (p1 de V: de +11 para −858), e o `δ` enorme e negativo empurra `f₂`
ainda mais para baixo — nos mesmos poucos estados, porque parado o agente não
visita outros (`ocupadas` quase não cresce depois do colapso). Os valores da LUT
estouram (máximo de 35 para 423): foram atualizações de dezenas de unidades em
`f`, exatamente o que `c_max = 0,5` aborta.

**Por que estes dois e não os de `Na = 2`.** `passo` e `φ` juntos. Com
`φ = 0,99` o crítico da eq. (4.8) quase não esquece: a mediana de V passa de
1 000 (contra recompensas por passo da ordem de 1 a 10), e `δ = G − V` herda essa
escala — o passo efetivo do ator é `∝ passo·δ`. Em `Na = 2`, `φ = 0,85` com
`8e-4` (T16) atravessou 50 M sem incidente; `φ = 0,99` com `8e-4` (T32) degradou
sem colapsar; aqui `φ = 0,99` com `1,5e-3` colapsou duas vezes. A grade a 200 mil
passos premia a agressividade (ids 92 e 86 são 1º e 3º) antes de ela cobrar.

**Retreino (lançado em 2026-10-17):** o melhor ponto de cada `T` na grade DEPOIS
de excluir as três causas do colapso — `φ = 0,99`, `passo > 8e-4` e `σ = 0,20`
(σ está no denominador do passo efetivo e abortou 55/180 contra 25/180) —, com a
asserção LIGADA (`c_max = 0,5`) e a guarda de colapso:

| T | id | passo | σ | n | φ | grade |
|---|---|---|---|---|---|---|
| 32 | 67 | 8e-4 | 0,35 | 12 | 0,85 | 84,0 % · SR 28 % · 11º |
| 16 | 60 | 8e-4 | 0,35 | 12 | 0,70 | 82,8 % · SR 8,6 % · 13º |

O `T = 32` coincide com a config da T-06.4, que treinou os `a4s24` publicados até
50 M com SR 90 %. No `T = 16` o id 60 (φ 0,70) fica 2,7 pp acima do id 61 (φ 0,85,
a T-06.4) — diferença dentro do ruído de 3 sementes; a escolha pelo melhor da
grade filtrada foi do usuário.

**`T = 16` concluído em 2026-10-19, sem colapso:** parou por platô da cobertura
a 33,75 M passos (parada antecipada da §4.4; a asserção `c_max` nunca
disparou). Validação final 96,7 % / SR 92,3 %; melhor validação 98,7 %. No
teste (checkpoint `melhor`, 100 sementes por Ns): **cobertura 97,6 %, SR 94 %,
CT 67,1 s** — a2s12 96,5 / 93, a4s24 97,8 / 95, a4s36 98,6 / 93. Deploy podado
a 70 % com 288 KB, SR 88 → 92 % na validação em Ns = 36. É o melhor modelo de
`Na = 4` até aqui, e supera o `a4s24` publicado (SR 90 %) com um regime de `Ns`
mais difícil.

**`T = 32` (id 67) concluído em 2026-10-20, sem colapso:** parou por platô a
33,75 M passos, a asserção nunca disparou. Validação final 97,7 % / SR 95 %;
melhor validação 99,3 %. No teste (checkpoint `melhor`, 100 sementes por Ns):
**cobertura 98,8 %, SR 96 %, CT 49,1 s** — a4s12 99,8 / 98, a4s24 96,7 / 91,
a4s36 99,9 / 98. A poda de 70 % derrubou o SR de 97 para 93 % na validação e foi
recusada: deploy sem poda, 1 833 KB.

**Na=4, os dois fechados:** o `T = 32` ganha em SR (96 contra 94 %) e CT (49 contra
67 s), o `T = 16` ganha em tamanho por 6× (288 KB podado contra 1 833 KB sem poda).
Mesmo passo (8e-4), φ diferente (0,85 e 0,70), os dois pararam por platô a 33,75 M
— o mesmo passo, porque a parada antecipada conta avaliações e as duas corridas
avaliam no mesmo ritmo. A alternativa que mantém o `passo` da grade,
`1,5e-3 · φ 0,85` (ids 91 e 85, 7º e 8º), fica registrada caso esta não chegue
ao desempenho dos colapsados antes do colapso (SR 83–93 % a 1,25–2,5 M). Pastas
dos colapsados: `runs/spec06-50M/T{16,32}-a4s12-24-36.colapso-p1.5e-3-phi0.99`.

## Na = 8 · Ns = 12 / 24 / 36 (revezado)

Grade: `reports/spec06-busca-a8s12-24-36.json` (360 corridas, 120 pontos, ~51 h
somando os três trechos — dois reinícios da máquina no meio, retomados com
`--retomar` —, 35 abortos). Fechou em 2026-10-14.

| T | id | passo | σ | n | φ | N | cobertura | SR | abortos | a8s12 | a8s24 | a8s36 | top 3 (ids) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **32** | 80 | 1,5e-3 | 0,20 | 12 | 0,99 | 104 | 96,4 % | 56,1 % | **2/3** | 94 / 58 | 97 / 56 | 98 / 55 | 80 · 116 · 115 |
| **16** | 110 | 3e-3 | 0,35 | 12 | 0,99 | 52 | 97,3 % | 70,4 % | 0/3 | 97 / 72 | 97 / 69 | 98 / 70 | 110 · 108 · 109 |

Nas células, o par é cobertura / SR em %.

**Aqui o `T = 16` ganha do `T = 32`**, em cobertura (97,3 contra 96,4 %), em SR
(70,4 contra 56,1 %) e em estabilidade (0/3 contra 2/3 abortos) — e com metade
das RAMs. É a primeira célula em que isso acontece; em `Na = 2` e `Na = 4` o
`T = 32` ganhava. O 1º por SR da grade inteira é o id 110.

**Ressalva no T = 32.** O 1º (id 80) abortou em 2 de 3 sementes e está em
`σ = 0,20`, o nível que concentra 34 dos 35 abortos da grade. As alternativas:
id 116 (`3e-3 · σ 0,35 · φ 0,99`, 95,6 %, SR 53,1 %, 1/3) e id 115 (`3e-3 · σ 0,35
· φ 0,85`, 93,3 %, SR 43,6 %, **0/3**). A escolha para o treino longo é decisão a
registrar aqui; com dois abortos em três, o id 80 não deveria ser a padrão.

O passo ótimo continua subindo com `Na`: 8e-4 em `Na = 1`, 1,5e-3 em `Na = 2` e
`4`, 3e-3 em `Na = 8` para o `T = 16`. É o efeito da recompensa dividida por
`Na` — o sinal por drone encolhe e o passo maior cabe. `3e-3`, que abortava
72/72 em `Na = 2`, aqui aborta 30/72 e vence em `T = 16`.

Leitura dos eixos (média marginal das 360 corridas): `n = 12` 75 % contra
`n = 24` 34 %; `passo` cresce até 1,5e-3 (64 %) e 3e-3 fica em 58 % com 30/72
abortos; `σ = 0,35` empata com 0,20 em cobertura (55 contra 54 %) mas aborta 1
contra 34; `φ = 0,99` é o pior na média (50 %) e o melhor nos vencedores, como em
`Na = 4`; `T = 32` e `16` empatam na média (56 contra 53 %).

**Treino longo (2026-10-19/20 → 2026-10-25), concluído sem colapso nos dois T.**
Config escolhida: `passo 1,5e-3 · σ 0,35 · n 12 · φ 0,85` (ids 91 e 85) — o análogo
do 8e-4 que funcionou em `Na = 4`, porque a recompensa é dividida por `Na` e o
salto do ator por unidade de recompensa é `α/Na`; os 1ºs da grade (3e-3 com
φ 0,99; σ 0,20 com 2/3 abortos) repetiriam a receita do colapso de `Na = 4`.
Asserção ligada e guarda de colapso; nenhuma disparou. As duas corridas foram
retomadas a 7,5 e 8,75 M depois da correção do vazamento de memória
(`ac_continuo`, 2026-10-20).

| T | parou | validação final | melhor val | teste (`melhor`): cob / SR / CT | a8s12 | a8s24 | a8s36 | deploy 70 % |
|---|---|---|---|---|---|---|---|---|
| 16 | 50 M (orçamento) | 97,4 % / 88 % | 98,5 % a 36,3 M | **97,8 % / 93 % / 72 s** | 97 / 95 | 98 / 93 | 99 / 92 | 308 KB, SR 93 → 96 % |
| 32 | 42,5 M (platô) | 98,5 % / 90 % | 99,2 % a 23,8 M | **99,2 % / 89 % / 85 s** | 99 / 88 | 100 / 91 | 99 / 88 | 654 KB, SR 84 → 83 % |

Em `Na = 8` o `T = 16` ganha em SR (93 contra 89 %) e CT, com menos da metade do
tamanho; o `T = 32` ganha em cobertura (99,2 contra 97,8 %). O SR em `Na = 8`
oscila mais de avaliação para avaliação (71–92 %) porque exige que os oito drones
zerem a coleta no mesmo episódio.

## Resumo: o vencedor muda com Na

| Na | T32 | T16 |
|---|---|---|
| 1 | 8e-4 · σ 0,35 · φ 0,85 (50 % · SR 21 %) | 8e-4 · σ 0,35 · φ 0,85 (46 % · SR 1 %) |
| 2 | 1,5e-3 · σ 0,35 · φ 0,70 (83 % · SR 51 % · 1/3) | 8e-4 · σ 0,35 · φ 0,85 (72 % · SR 17 %) |
| 4 | 1,5e-3 · σ 0,35 · φ 0,99 (97 % · SR 76 %) | 1,5e-3 · σ 0,35 · φ 0,99 (93 % · SR 63 %) |
| 8 | 1,5e-3 · σ 0,20 · φ 0,99 (96 % · SR 56 % · 2/3) | 3e-3 · σ 0,35 · φ 0,99 (97 % · SR 70 %) |

`n = 12` em todos. A grade a 200 mil passos mede o início do treino; o orçamento
longo já inverteu posições antes (`spec06-grid-search.md` §9), então os números
acima são o ponto de partida, não o veredito.

## Os oito modelos treinados a 50 M (o veredito, no teste, checkpoint `melhor`)

| Na | T | config treinada | cobertura | SR | CT | deploy |
|---|---|---|---|---|---|---|
| 1 | 16 | 8e-4 · 0,35 · 12 · 0,85 | 94,6 % | 93 % | 70 s | `models/deploy/T16/a1s12` (`final`, 50 M) |
| 1 | 32 | 8e-4 · 0,35 · 12 · 0,85 | 98,9 % | 98 % | 49 s | `models/deploy/T32/a1s12` (`final`, 50 M) |
| 2 | 16 | 8e-4 · 0,35 · 12 · 0,85 | 95,1 % | 90 % | 90 s | 259 KB (70 %) |
| 2 | 32 | 8e-4 · 0,35 · 12 · 0,99 | 96,1 % | 77 % | 143 s | 1 567 KB (sem poda) |
| 4 | 16 | 8e-4 · 0,35 · 12 · 0,70 | 97,6 % | 94 % | 67 s | 288 KB (70 %) |
| 4 | 32 | 8e-4 · 0,35 · 12 · 0,85 | 98,8 % | 96 % | 49 s | 1 833 KB (sem poda) |
| 8 | 16 | 1,5e-3 · 0,35 · 12 · 0,85 | 97,8 % | 93 % | 72 s | 308 KB (70 %) |
| 8 | 32 | 1,5e-3 · 0,35 · 12 · 0,85 | 99,2 % | 89 % | 85 s | 654 KB (70 %) |

`Na = 1` é o modelo promovido (`final` dos 50 M, `metrica_test_final` da
procedência em `models/deploy`), em `Ns = 12`; as demais são os 50 M deste ciclo
com o checkpoint `melhor`, 100 sementes de teste por Ns, média das três células. A config é a mesma em todas fora de φ (0,70–0,99) e do passo em `Na = 8`,
que é 8e-4 na escala `α/Na`. Promoção para `models/` pendente de
`spec06_exportar_deploy.py --promover` célula a célula; o `T32` de `Na = 2` pede
decisão antes (ver o alerta na seção dele).

## Campanha comparável dos modelos novos (2026-10-26)

`spec06_campanha_novos.sh`, rótulos `wnn-v2-T16`/`wnn-v2-T32`, origem `deploy`
(o podado gerado do `final`; em `Na = 2` `T32` sem poda), cada modelo só nas
células do seu `Na`, MADDPG publicado como baseline (joga em casa em cada célula).
100 runs × 200 episódios por célula e regime; 81 corridas, 72 novas, 8 h 40 com 10
em paralelo, zero falhas. Atestados RF-05: 18, todos aprovados — `q_w = 4` em 17 e
`q_w = 5` no `T32` de `a2s36` (grade fina 4–8 depois de reprovar com a grade padrão).
CSVs em `reports/campanha/wnn-v2-*.csv`; manifesto por corrida em `logs/`.

**SR em % (média de 100 runs) · CT em s** — colunas: `wnn-v2-T16` | `wnn-v2-T32` | `maddpg`

| célula | limpo | sem-kf (σ 10) | com-kf (σ 10) |
|---|---|---|---|
| a2s12 | 88,9 · 83 \| 77,0 · 143 \| 100 · 18 | 95,9 · 54 \| 92,8 · 76 \| 100 · 19 | 91,8 · 70 \| 86,0 · 103 \| 100 · 19 |
| a2s24 | 78,3 · 144 \| 69,5 · 189 \| 99,4 · 30 | 91,2 · 91 \| 88,7 · 111 \| 100 · 30 | 84,6 · 116 \| 79,3 · 146 \| 99,8 · 29 |
| a2s36 | 77,8 · 153 \| 69,8 · 193 \| 99,6 · 35 | 89,2 · 109 \| 88,3 · 121 \| 100 · 36 | 83,2 · 129 \| 80,6 · 147 \| 99,8 · 35 |
| a4s12 | 94,8 · 55 \| 96,6 · 44 \| 100 · 10 | 97,8 · 44 \| 98,3 · 38 \| 100 · 11 | 96,1 · 50 \| 96,8 · 43 \| 100 · 10 |
| a4s24 | 94,6 · 66 \| 92,9 · 72 \| 100 · 14 | 96,9 · 60 \| 94,4 · 71 \| 100 · 15 | 95,5 · 62 \| 94,0 · 68 \| 100 · 14 |
| a4s36 | 94,5 · 70 \| 93,9 · 71 \| 100 · 16 | 97,1 · 63 \| 95,2 · 71 \| 100 · 18 | 95,6 · 65 \| 94,3 · 70 \| 100 · 16 |
| a8s12 | 92,5 · 67 \| 91,0 · 69 \| 100 · 15 | 96,1 · 52 \| 94,9 · 52 \| 100 · 16 | 91,1 · 74 \| 88,8 · 79 \| 100 · 15 |
| a8s24 | 93,9 · 70 \| 86,5 · 101 \| 100 · 10 | 95,6 · 68 \| 89,8 · 91 \| 100 · 10 | 92,0 · 81 \| 84,7 · 110 \| 100 · 10 |
| a8s36 | 92,5 · 82 \| 85,7 · 109 \| 100 · 13 | 92,9 · 87 \| 88,9 · 101 \| 100 · 14 | 90,0 · 95 \| 84,3 · 116 \| 100 · 13 |

Três leituras, a confirmar na análise:

- **O MADDPG domina** em todas as células e regimes: SR 99–100 % e CT 3 a 6×
  menor. A WNN chega perto em SR nas células de `Na = 4` (93–98 %) e fica mais
  longe em `Na = 2` (70–89 % no limpo), onde o `T32` do `final` degradado pesa.
- **Ruído sem filtro MELHORA a WNN a σ = 10** (sem-kf > limpo em todas as 18
  células, +0,4 a +19 pp de SR, CT menor). É o padrão da memória de ruído do
  projeto: as falhas da WNN são estagnação, e o ruído por passo tira o drone do
  estado parado. O KF (`com-kf`) devolve parte disso ao remover o ruído: fica
  entre os dois. **A σ = 25 o efeito se inverte em `Na = 4` e `8`** — ver a
  tabela abaixo.
- **`T16 ≥ T32`** em quase todas as células, com 2 a 4× menos memória; a exceção é
  `a4s12`, onde o `T32` ganha por 2 pp.

**σ = 25 (medido em 2026-10-29; 48 corridas, 9 h, 6 em paralelo; MADDPG e
publicados em `a2` já tinham σ = 25 de 2026-09-09, mesma convenção — o vizinho
entra com a posição verdadeira sob ruído em TODAS as campanhas; a correção
"posição divulgada" (f263c77) foi mesclada e revertida em 2026-10-28 por decisão
do usuário, e fica como trabalho futuro). SR % · CT s:**

| célula | T16 σ25 sem-kf | T32 σ25 sem-kf | MADDPG σ25 sem-kf | T16 σ25 com-kf | T32 σ25 com-kf | MADDPG σ25 com-kf |
|---|---|---|---|---|---|---|
| a2s12 | 91,6 · 94 s | 85,4 · 131 s | 99,9 · 29 s | 90,8 · 76 s | 84,2 · 112 s | 99,9 · 21 s |
| a2s24 | 88,4 · 136 s | 79,2 · 183 s | 100 · 43 s | 84,2 · 121 s | 76,9 · 159 s | 99,8 · 30 s |
| a2s36 | 89,5 · 144 s | 79,4 · 193 s | 100 · 54 s | 83,3 · 132 s | 78,2 · 161 s | 99,7 · 38 s |
| a4s12 | 90,4 · 95 s | 91,3 · 87 s | 100 · 16 s | 96,1 · 50 s | 96,7 · 44 s | 100 · 11 s |
| a4s24 | 84,0 · 143 s | 74,9 · 182 s | 100 · 24 s | 94,8 · 67 s | 93,6 · 71 s | 100 · 15 s |
| a4s36 | 84,0 · 149 s | 75,8 · 184 s | 100 · 28 s | 94,5 · 73 s | 93,8 · 74 s | 100 · 17 s |
| a8s12 | 93,0 · 77 s | 94,1 · 66 s | 100 · 20 s | 89,7 · 80 s | 86,1 · 92 s | 100 · 16 s |
| a8s24 | 83,1 · 145 s | 76,5 · 173 s | 100 · 14 s | 90,1 · 91 s | 82,7 · 120 s | 100 · 10 s |
| a8s36 | 72,3 · 200 s | 65,1 · 232 s | 100 · 19 s | 87,1 · 111 s | 82,3 · 127 s | 100 · 14 s |

Leitura: a σ = 25 **sem filtro** a WNN perde em `Na = 4` (SR −4 a −18 pp, CT +40
a +113 s) e `Na = 8` (−10 a −21 pp em s24/s36, CT +72 a +123 s), e só ganha em
`Na = 2` (+3 a +12 pp). A dose de ruído passou do ponto: o erro de 25 m na própria
posição desordena os sensores mais próximos e custa mais do que o destravamento
rende, e pesa mais com mais drones. **Com o KF** a σ = 25 a WNN volta ao nível do
limpo em `Na = 4` (±1 pp) e fica 3 a 5 pp abaixo em `Na = 8`; em `Na = 2` segue
acima do limpo. O MADDPG mantém SR 100 %; só o CT sobe sem filtro (+5 a +19 s) e o
filtro o devolve (+0 a +2 s). Em `Na = 2` os publicados continuam à frente dos
novos a σ = 25 (exceção: T16 novo 91,6 × 90,0 em `a2s12` sem filtro). CSVs:
`reports/campanha/*-sigma25-*.csv`.

**Alerta em `Na = 2`: os modelos novos perdem para os publicados** (`wnn-T16`/
`wnn-T32`, treinados em `a2s24` fixo com a MESMA config da T-06.4, campanha
anterior nas mesmas células e sementes):

| célula · regime | T16 publicado | T16 novo | T32 publicado | T32 novo |
|---|---|---|---|---|
| a2s12 · limpo | 90,5 · 78 s | 88,9 · 83 s | 89,8 · 80 s | 77,0 · 143 s |
| a2s24 · limpo | 91,8 · 81 s | 78,3 · 144 s | 94,3 · 68 s | 69,5 · 189 s |
| a2s36 · limpo | 92,8 · 82 s | 77,8 · 153 s | 96,4 · 62 s | 69,8 · 193 s |
| a2s24 · sem-kf | 98,3 · 58 s | 91,2 · 91 s | 98,1 · 57 s | 88,7 · 111 s |
| a2s24 · com-kf | 96,6 · 60 s | 84,6 · 116 s | 96,6 · 59 s | 79,3 · 146 s |

O publicado ganha até em `a2s12` e `a2s36`, onde ele TRANSFERE e o novo joga em
casa. As diferenças entre as duas gerações em `Na = 2`: revezamento de `Ns` no
treino, `--c-max none` (os dois novos), o `T32` com φ 0,99 (degradou após 1,25 M)
e o `deploy` gerado do `final` (no `T16` novo o `final` tem SR 85 % contra 90 %
do `melhor`). O diagnóstico barato é repetir a campanha de `Na = 2` com
`ORIGEM=melhor` (`ROT=wnn-v2m`): se o `melhor` recupera os ~13 pp, o problema é
o `final`; se não, é o regime de treino revezado para `Na = 2`. Em `Na = 4` e `8`
não há publicado revezado para comparar; o `a4s24` publicado (T16, SR 90 % no
teste) fica abaixo do novo (94–96 %).

## Pico de RAM e latência dos modelos novos × MADDPG (2026-10-26, máquina ociosa)

`spec03_latencia_memoria.py --ckpt-padrao <caminho> --sufixo=-v2`, um processo novo
por modelo, mesma observação para todos, Ns = 24 por Na; pico = VmHWM do processo que
carrega o ator e decide, média ± dp de 30 processos; latência = menor mediana de
8×1000 decisões (p99 global). Artefatos `reports/spec03-*-a{2,4,8}s24-v2.*`; os
publicados de a2s24 remedidos no mesmo estado em `-ref`.

| célula | modelo | latência mediana | p99 | pico de RAM | arcabouço | acréscimo da rede | entradas |
|---|---|---|---|---|---|---|---|
| a2s24 | WNN T16 novo | 22,3 µs | 58 | 45,9 ± 0,1 MB | 31,4 | 14,4 MB | 20 833 |
| a2s24 | WNN T32 novo | 24,9 µs | 55 | 73,4 ± 0,1 MB | 31,6 | 41,8 MB | 134 013 |
| a2s24 | MADDPG | 48,6 µs | 176 | 506,1 ± 0,1 MB | 496,0 | 10,0 MB | 73 478 |
| a2s24 | WNN T16 publicado | 22,3 µs | 55 | 46,2 ± 0,1 MB | 31,5 | 14,7 MB | 22 453 |
| a2s24 | WNN T32 publicado | 24,9 µs | 63 | 53,6 ± 0,1 MB | 31,6 | 22,0 MB | 41 476 |
| a4s24 | WNN T16 novo | 23,4 µs | 64 | 46,6 ± 0,1 MB | 31,5 | 15,1 MB | 23 234 |
| a4s24 | WNN T32 novo | 25,4 µs | 63 | 79,4 ± 0,1 MB | 31,5 | 47,9 MB | 157 388 |
| a4s24 | MADDPG | 48,0 µs | 162 | 506,1 ± 0,1 MB | 496,1 | 10,0 MB | 74 502 |
| a8s24 | WNN T16 novo | 23,4 µs | 60 | 48,0 ± 0,2 MB | 31,6 | 16,4 MB | 24 728 |
| a8s24 | WNN T32 novo | 27,3 µs | 68 | 59,4 ± 0,2 MB | 31,3 | 28,1 MB | 53 032 |
| a8s24 | MADDPG | 48,9 µs | 186 | 506,1 ± 0,1 MB | 496,1 | 10,0 MB | 76 550 |

A WNN decide em 22–27 µs contra ~49 µs do MADDPG (2×), com pico de RAM 6 a 11×
menor; 496 MB do pico do MADDPG são o PyTorch. O acréscimo da rede é maior na WNN
(14–48 MB: o load constrói a rede de dicionários antes de achatá-la em LUT) que no
MADDPG (10 MB). Os números batem com a medição canônica de a2s24 do deck do artigo
(`spec06-merito.json`: 47/52 MB, 22/25/49 µs). O `T32` novo de a2s24 ocupa 3,2× mais
entradas que o publicado (três Ns no treino) e por isso 73 contra 54 MB de pico.

## Coordenação: a lacuna de CT medida pela separação entre drones (2026-10-28)

Coluna `separacao_media_m` dos CSVs da campanha (distância média entre os drones
ao longo do episódio), regime sem ruído, média de 100 runs:

| célula | WNN T16 novo | WNN T32 novo | MADDPG | WNN / MADDPG |
|---|---|---|---|---|
| a2s12 | 72 m | 76 m | 93 m | 0,78–0,82 |
| a2s24 | 89 m | 99 m | 101 m | 0,89–0,98 |
| a2s36 | 99 m | 96 m | 117 m | 0,83–0,85 |
| a4s12 | 55 m | 53 m | 80 m | 0,66–0,68 |
| a4s24 | 70 m | 62 m | 94 m | 0,65–0,75 |
| a4s36 | 79 m | 72 m | 95 m | 0,75–0,83 |
| a8s12 | 37 m | 38 m | 77 m | 0,49 |
| a8s24 | 45 m | 45 m | 74 m | 0,61 |
| a8s36 | 50 m | 51 m | 88 m | 0,56–0,58 |

Os drones da WNN voam mais perto uns dos outros em todas as 9 células, e a razão
cai com `Na`: ~0,85 em 2 drones, ~0,7 em 4, ~0,55 em 8. É a assinatura de drones
que competem pelos mesmos sensores em vez de dividir a área, e explica a lacuna de
CT (3 a 6×) com cobertura quase igual.

**Mecanismo.** O ator vê os vizinhos nos dois métodos (observação idêntica). O que
difere é o crítico, que só existe no treino: o do MADDPG é centralizado (CTDE),
recebe observações e ações de todos os drones e faz a atribuição de crédito; o
nosso `V(x_i)` recebe só a observação do próprio drone e a recompensa do time
÷ Na — não distingue "o time coletou" de "eu coletei", e o δ do drone redundante
sai positivo.

**Correção implementada (2026-10-28), ainda sem treino longo:** crítico
centralizado n-tuple em `src/uavwnn/agents/ctde.py`, ligado no treinador por
`--critico centralizado`:

| modo | crítico | vantagem do drone `i` | o que dá |
|---|---|---|---|
| `--contrafactual 0` | `V_c(s)` sobre `[obs_1..obs_Na]` | `δ = G − V_c(s)`, igual para todos | vê o time → menos variância; sem crédito por drone |
| `--contrafactual K` | `Q_c(s, a)` sobre `[obs_1..obs_Na, ação_1..ação_Na]` | `A_i = G − (1/K) Σ_k Q_c(s, (a_{−i}, ã_i^k))`, `ã ~ π_i` | crédito por drone (COMA) — o análogo do `∂Q/∂a_i` do MADDPG |

`G − Q_c(s, a)` **não** é usado como vantagem: a base dependeria da própria ação
e o gradiente sumiria com o crítico convergindo (a classe recusa `K = 0` com
crítico de ação). O ator, a observação, o deploy (`*ator.npz` + termômetro) e o
problema de Lamenza não mudam; o crítico central vai só nos checkpoints de
treino (`critico-central.npz`). Recompensa segue ÷ Na. Custo medido no smoke
`a2s24`: 114 passos/s com K = 4 contra 126–139 sem (10–20 % mais lento).

*Guard rails* (`tests/test_ctde.py`, 8 testes): entrada = `Na·(bits_obs +
bits_ação)` na ordem dos drones e a troca da ação do drone `i` só mexe na fatia
dele; ator e deploy inalterados; ida e volta do checkpoint; K × tipo de crítico
coerentes; e o **eval de crédito**: num time de 2 drones em que a recompensa é
a velocidade do drone 0, com `K = 8` a vantagem do drone irrelevante cai a 0,55
da do drone que decide (dispersão 0,35), e com `V_c(s)` as duas são iguais. A
razão não chega a 0 porque a rede n-tuple prevê pela média das RAMs e explica só
~40–60 % do efeito da ação (estável de 1 200 a 6 000 episódios: viés, não
ruído). No treino, `metrics.jsonl` grava `separacao_m` por eval e, no CTDE,
`credito.dispersao_credito` (0 = todos recebem o mesmo sinal; no smoke de 5 000
passos ainda é ~0,01 porque o crítico está quase vazio — tem de subir ao longo
de uma corrida longa, senão o contrafactual não está agindo).

**Corrida de aceitação proposta (a decidir):** `a4s12-24-36`, T16, config id 60
da grade, 10 M passos, três braços — descentralizado, `--contrafactual 0` e
`--contrafactual 4` —, comparando na mesma avaliação `separacao_m`, CT e SR
contra a corrida de 50 M e o MADDPG (94 m em a4s24). Critério de sucesso: a
separação subir em direção à do MADDPG e o CT cair, com SR mantido. Segunda
frente: PPO sobre a rede n-tuple do anbubus (`vendor/ntn_neurocomputing_v0`),
estendendo a grade de 1 drone ao multiagente.

## Como atualizar

Com o JSON da grade em mãos:

    uv run python - <<'EOF'
    import json
    g = json.load(open("reports/spec06-busca-a4s12-24-36.json"))
    P = sorted(g["pontos"], key=lambda p: -p["cobertura"])
    for T in (16, 32):
        for p in [p for p in P if p["resolucao"] == T][:3]:
            print(T, p["id"], p["passo_efetivo"], p["sigma"], p["tupla"], p["critico"],
                  f"{p['cobertura']:.1%}", f"{p['sr']:.1%}", f"{p['abortos']}/{p['n']}",
                  p.get("por_celula"))
    EOF

Depois: linha na tabela do `Na`, entrada em `spec06-vencedores.json`, e o treino
longo lê a config com `spec06_treino_multiagente.py --config-de <json> --resolucao T`
(ou pelas flags, se a escolha não for o 1º por cobertura).
