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
σ 0,35 · φ 0,99`, 81,3 % cob, SR 43,2 %, **0/3 abortos**) fica a 1,6 pp. A escolha
para o treino longo é decisão a registrar aqui quando for feita.

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

## Na = 8 · Ns = 12 / 24 / 36

Grade ainda não rodada: `bash scripts/spec06_grade_celula.sh 8` (~2× o custo da
de `Na = 4`, que levou 21 h).

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
