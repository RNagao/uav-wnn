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
de ¼ do melhor encerram a corrida) para isso não custar dias de novo. Retreino
de `Na = 4`: asserção ligada e/ou `passo` um nível abaixo (8e-4), a decidir.

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

---

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
