# Campanha comparável — MADDPG × WNN T=16 × WNN T=32

100 runs × 200 episódios por célula. Todos os modelos veem as **mesmas**
sementes, e no regime com ruído o mesmo episódio cai no mesmo degrau —
a comparação é pareada em tudo.

`CT_cens` conta o fracasso como o horizonte (500 s); `CT_succ` só os que
completaram. Reportar só o segundo premiaria quem desiste dos difíceis.


## Desempenho por cenário

| Na | Ns | ruído | modelo | CT_cens | CT_succ | SR |
|---|---|---|---|---|---|---|
| 1 | 12 | sem ruído | wnn-T16 | 60.4 s | 37.4 s | 95.0% |
| 1 | 12 | sem ruído | wnn-T32 | 90.4 s | 39.2 s | 88.9% |
| 1 | 12 | σ=10 · SEM KF | wnn-T16 | 46.3 s | 43.5 s | 99.4% |
| 1 | 12 | σ=10 · SEM KF | wnn-T32 | 55.5 s | 46.1 s | 97.9% |
| 1 | 12 | σ=10 · COM KF | wnn-T16 | 45.4 s | 38.5 s | 98.5% |
| 1 | 12 | σ=10 · COM KF | wnn-T32 | 64.2 s | 40.3 s | 94.8% |
| 2 | 12 | sem ruído | maddpg | 18.4 s | 18.2 s | 100.0% |
| 2 | 12 | sem ruído | wnn-T16 | 133.7 s | 36.7 s | 79.1% |
| 2 | 12 | sem ruído | wnn-T32 | 98.3 s | 32.5 s | 85.9% |
| 2 | 12 | σ=10 · SEM KF | maddpg | 19.3 s | 19.3 s | 100.0% |
| 2 | 12 | σ=10 · SEM KF | wnn-T16 | 63.4 s | 42.4 s | 95.4% |
| 2 | 12 | σ=10 · SEM KF | wnn-T32 | 48.3 s | 36.2 s | 97.4% |
| 2 | 12 | σ=10 · COM KF | maddpg | 18.6 s | 18.5 s | 100.0% |
| 2 | 12 | σ=10 · COM KF | wnn-T16 | 84.8 s | 38.4 s | 89.9% |
| 2 | 12 | σ=10 · COM KF | wnn-T32 | 75.4 s | 33.1 s | 90.9% |
| 2 | 24 | sem ruído | maddpg | 30.3 s | 27.7 s | 99.4% |
| 2 | 24 | sem ruído | wnn-T16 | 86.8 s | 45.1 s | 90.8% |
| 2 | 24 | sem ruído | wnn-T32 | 69.7 s | 41.4 s | 93.8% |
| 2 | 24 | σ=10 · SEM KF | maddpg | 29.6 s | 29.6 s | 100.0% |
| 2 | 24 | σ=10 · SEM KF | wnn-T16 | 57.0 s | 50.9 s | 98.6% |
| 2 | 24 | σ=10 · SEM KF | wnn-T32 | 51.0 s | 46.4 s | 99.0% |
| 2 | 24 | σ=10 · COM KF | maddpg | 28.9 s | 28.0 s | 99.8% |
| 2 | 24 | σ=10 · COM KF | wnn-T16 | 68.0 s | 46.6 s | 95.3% |
| 2 | 24 | σ=10 · COM KF | wnn-T32 | 64.4 s | 42.3 s | 95.2% |
| 2 | 36 | sem ruído | maddpg | 35.4 s | 33.7 s | 99.6% |
| 2 | 36 | sem ruído | wnn-T16 | 74.5 s | 50.3 s | 94.6% |
| 2 | 36 | sem ruído | wnn-T32 | 67.1 s | 46.7 s | 95.5% |
| 2 | 36 | σ=10 · SEM KF | maddpg | 35.8 s | 35.8 s | 100.0% |
| 2 | 36 | σ=10 · SEM KF | wnn-T16 | 61.1 s | 56.7 s | 99.0% |
| 2 | 36 | σ=10 · SEM KF | wnn-T32 | 55.7 s | 52.9 s | 99.4% |
| 2 | 36 | σ=10 · COM KF | maddpg | 35.2 s | 34.3 s | 99.8% |
| 2 | 36 | σ=10 · COM KF | wnn-T16 | 62.2 s | 51.8 s | 97.7% |
| 2 | 36 | σ=10 · COM KF | wnn-T32 | 58.8 s | 47.7 s | 97.5% |

## O KF ajuda?

Diferença de tempo, pareada pelas mesmas sementes. **Negativo = o KF é
mais rápido.** IC 95 % por bootstrap pareado.

| Na | Ns | modelo | n | sem KF | com KF | Δ tempo | IC 95 % | SR sem→com |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | wnn-T16 | 20,000 | 46.35 s | 45.45 s | -0.90 s ** | [-1.74, -0.01] | 99.4% → 98.5% |
| 1 | 12 | wnn-T32 | 20,000 | 55.51 s | 64.22 s | +8.72 s ** | [+6.96, +10.30] | 97.9% → 94.8% |
| 2 | 12 | maddpg | 20,000 | 19.28 s | 18.62 s | -0.66 s ** | [-0.76, -0.54] | 100.0% → 100.0% |
| 2 | 12 | wnn-T16 | 20,000 | 63.38 s | 84.84 s | +21.45 s ** | [+19.38, +23.54] | 95.4% → 89.9% |
| 2 | 12 | wnn-T32 | 20,000 | 48.28 s | 75.37 s | +27.09 s ** | [+25.31, +29.14] | 97.4% → 90.9% |
| 2 | 24 | maddpg | 20,000 | 29.64 s | 28.90 s | -0.74 s ** | [-1.01, -0.44] | 100.0% → 99.8% |
| 2 | 24 | wnn-T16 | 20,000 | 57.05 s | 68.01 s | +10.96 s ** | [+9.47, +12.42] | 98.6% → 95.3% |
| 2 | 24 | wnn-T32 | 20,000 | 51.03 s | 64.43 s | +13.40 s ** | [+11.91, +14.91] | 99.0% → 95.2% |
| 2 | 36 | maddpg | 20,000 | 35.82 s | 35.18 s | -0.64 s ** | [-0.95, -0.34] | 100.0% → 99.8% |
| 2 | 36 | wnn-T16 | 20,000 | 61.07 s | 62.17 s | +1.11 s ** | [+0.02, +2.29] | 99.0% → 97.7% |
| 2 | 36 | wnn-T32 | 20,000 | 55.71 s | 58.85 s | +3.13 s ** | [+2.15, +4.25] | 99.4% → 97.5% |

`**` = o IC não contém zero.

## O ruído ajuda? (dithering)

Diferença entre o ruído bruto e o teto limpo. **SR positivo = o ruído
melhora**, que é o achado da SPEC-08.

| Na | Ns | modelo | limpo | σ=10 sem KF | Δ SR |
|---|---|---|---|---|---|
| 1 | 12 | wnn-T16 | 95.0% | 99.4% | **+4.3 pp** |
| 1 | 12 | wnn-T32 | 88.9% | 97.9% | **+9.1 pp** |
| 2 | 12 | maddpg | 100.0% | 100.0% | **+0.0 pp** |
| 2 | 12 | wnn-T16 | 79.1% | 95.4% | **+16.3 pp** |
| 2 | 12 | wnn-T32 | 85.9% | 97.4% | **+11.5 pp** |
| 2 | 24 | maddpg | 99.4% | 100.0% | **+0.6 pp** |
| 2 | 24 | wnn-T16 | 90.8% | 98.6% | **+7.8 pp** |
| 2 | 24 | wnn-T32 | 93.8% | 99.0% | **+5.2 pp** |
| 2 | 36 | maddpg | 99.6% | 100.0% | **+0.4 pp** |
| 2 | 36 | wnn-T16 | 94.6% | 99.0% | **+4.4 pp** |
| 2 | 36 | wnn-T32 | 95.5% | 99.4% | **+3.9 pp** |

## Métricas por cenário — média ± desvio SOBRE AS RUNS

Cada run é uma réplica da campanha de avaliação (200 episódios). O desvio
entre runs mede **quanto o número publicado variaria numa nova execução** —
não quanto um episódio varia, que é outra coisa e muito maior.

| Na | Ns | ruído | modelo | CT (s) | SR (%) | coleta (s) | separação (m) |
|---|---|---|---|---|---|---|---|
| 1 | 12 | sem ruído | wnn-T16 | 60.39 ± 6.70 | 95.03 ± 1.44 | 34.48 ± 5.35 | — |
| 1 | 12 | sem ruído | wnn-T32 | 90.42 ± 9.27 | 88.88 ± 2.01 | 48.09 ± 6.13 | — |
| 1 | 12 | σ=10 · SEM KF | wnn-T16 | 46.35 ± 2.84 | 99.38 ± 0.63 | 21.68 ± 1.64 | — |
| 1 | 12 | σ=10 · SEM KF | wnn-T32 | 55.51 ± 4.47 | 97.94 ± 0.99 | 23.65 ± 1.87 | — |
| 1 | 12 | σ=10 · COM KF | wnn-T16 | 45.45 ± 3.83 | 98.49 ± 0.84 | 22.81 ± 2.95 | — |
| 1 | 12 | σ=10 · COM KF | wnn-T32 | 64.22 ± 7.77 | 94.80 ± 1.69 | 32.22 ± 4.73 | — |
| 2 | 12 | sem ruído | maddpg | 18.44 ± 0.71 | 99.95 ± 0.15 | 9.79 ± 0.31 | 92.84 ± 13.24 |
| 2 | 12 | sem ruído | wnn-T16 | 133.68 ± 13.25 | 79.08 ± 2.86 | 77.42 ± 8.28 | 71.41 ± 36.64 |
| 2 | 12 | sem ruído | wnn-T32 | 98.29 ± 11.33 | 85.92 ± 2.42 | 59.39 ± 7.76 | 57.46 ± 28.16 |
| 2 | 12 | σ=10 · SEM KF | maddpg | 19.28 ± 0.19 | 100.00 ± 0.00 | 10.36 ± 0.12 | 93.91 ± 14.61 |
| 2 | 12 | σ=10 · SEM KF | wnn-T16 | 63.38 ± 6.62 | 95.41 ± 1.45 | 33.63 ± 4.34 | 81.46 ± 31.30 |
| 2 | 12 | σ=10 · SEM KF | wnn-T32 | 48.28 ± 5.75 | 97.40 ± 1.21 | 25.40 ± 3.57 | 59.48 ± 25.44 |
| 2 | 12 | σ=10 · COM KF | maddpg | 18.62 ± 0.61 | 99.97 ± 0.12 | 9.92 ± 0.16 | 93.52 ± 14.49 |
| 2 | 12 | σ=10 · COM KF | wnn-T16 | 84.84 ± 9.93 | 89.95 ± 2.13 | 47.16 ± 6.17 | 71.99 ± 33.85 |
| 2 | 12 | σ=10 · COM KF | wnn-T32 | 75.37 ± 9.74 | 90.94 ± 2.09 | 42.40 ± 5.95 | 57.26 ± 29.06 |
| 2 | 24 | sem ruído | maddpg | 30.32 ± 2.37 | 99.44 ± 0.50 | 14.36 ± 1.51 | 100.75 ± 15.36 |
| 2 | 24 | sem ruído | wnn-T16 | 86.82 ± 8.90 | 90.83 ± 1.94 | 32.29 ± 3.49 | 101.79 ± 25.66 |
| 2 | 24 | sem ruído | wnn-T32 | 69.73 ± 8.11 | 93.82 ± 1.75 | 28.24 ± 3.42 | 81.24 ± 28.51 |
| 2 | 24 | σ=10 · SEM KF | maddpg | 29.64 ± 0.27 | 100.00 ± 0.00 | 13.94 ± 0.16 | 102.89 ± 14.21 |
| 2 | 24 | σ=10 · SEM KF | wnn-T16 | 57.05 ± 3.77 | 98.63 ± 0.87 | 21.83 ± 1.33 | 105.52 ± 23.55 |
| 2 | 24 | σ=10 · SEM KF | wnn-T32 | 51.03 ± 3.31 | 98.99 ± 0.71 | 20.37 ± 1.11 | 79.40 ± 24.61 |
| 2 | 24 | σ=10 · COM KF | maddpg | 28.90 ± 1.32 | 99.81 ± 0.28 | 13.54 ± 0.84 | 102.24 ± 15.32 |
| 2 | 24 | σ=10 · COM KF | wnn-T16 | 68.01 ± 7.26 | 95.28 ± 1.59 | 25.78 ± 2.65 | 99.60 ± 27.67 |
| 2 | 24 | σ=10 · COM KF | wnn-T32 | 64.43 ± 6.91 | 95.17 ± 1.52 | 24.25 ± 2.31 | 77.70 ± 28.20 |
| 2 | 36 | sem ruído | maddpg | 35.36 ± 2.19 | 99.64 ± 0.46 | 14.85 ± 0.78 | 116.58 ± 19.21 |
| 2 | 36 | sem ruído | wnn-T16 | 74.53 ± 7.14 | 94.60 ± 1.59 | 23.91 ± 2.10 | 102.29 ± 23.09 |
| 2 | 36 | sem ruído | wnn-T32 | 67.06 ± 6.27 | 95.50 ± 1.37 | 22.65 ± 1.91 | 84.39 ± 24.24 |
| 2 | 36 | σ=10 · SEM KF | maddpg | 35.82 ± 0.41 | 100.00 ± 0.05 | 15.49 ± 0.18 | 119.20 ± 15.78 |
| 2 | 36 | σ=10 · SEM KF | wnn-T16 | 61.07 ± 2.95 | 99.02 ± 0.66 | 21.04 ± 0.68 | 106.84 ± 22.57 |
| 2 | 36 | σ=10 · SEM KF | wnn-T32 | 55.71 ± 2.60 | 99.38 ± 0.58 | 20.04 ± 0.62 | 86.69 ± 24.84 |
| 2 | 36 | σ=10 · COM KF | maddpg | 35.18 ± 1.55 | 99.81 ± 0.31 | 14.93 ± 0.56 | 117.50 ± 20.38 |
| 2 | 36 | σ=10 · COM KF | wnn-T16 | 62.17 ± 4.31 | 97.68 ± 0.96 | 20.75 ± 1.10 | 102.71 ± 24.12 |
| 2 | 36 | σ=10 · COM KF | wnn-T32 | 58.85 ± 5.34 | 97.52 ± 1.17 | 19.65 ± 1.16 | 84.67 ± 22.44 |

### Custo — constante por modelo e célula

| Na | Ns | modelo | params do ator | bytes (fp32) | mult. fp | latência mediana (µs) |
|---|---|---|---|---|---|---|
| 1 | 12 | wnn-T16 | 10,965 | 299,298 | 4 | 102.1 ± 28.9 |
| 1 | 12 | wnn-T32 | 23,440 | 628,844 | 4 | 104.8 ± 30.0 |
| 2 | 12 | maddpg | 73,474 | 293,896 | 72,960 | 60.7 ± 12.5 |
| 2 | 12 | wnn-T16 | 21,472 | 574,072 | 4 | 120.1 ± 35.1 |
| 2 | 12 | wnn-T32 | 38,900 | 1,039,656 | 4 | 129.7 ± 37.7 |
| 2 | 24 | maddpg | 73,474 | 293,896 | 72,960 | 54.6 ± 14.5 |
| 2 | 24 | wnn-T16 | 21,472 | 574,072 | 4 | 129.7 ± 35.9 |
| 2 | 24 | wnn-T32 | 38,900 | 1,039,656 | 4 | 129.3 ± 37.5 |
| 2 | 36 | maddpg | 73,474 | 293,896 | 72,960 | 65.3 ± 12.5 |
| 2 | 36 | wnn-T16 | 21,472 | 574,072 | 4 | 127.6 ± 36.7 |
| 2 | 36 | wnn-T32 | 38,900 | 1,039,656 | 4 | 136.7 ± 39.4 |

### O que o desvio diz

**A dispersão do MADDPG é uma ordem de grandeza menor.** Em `a2s12` limpo ele dá
`18,44 ± 0,71 s` contra `133,68 ± 13,25 s` do `wnn-T16` — 3,8 % de coeficiente de
variação contra 9,9 %. Ele não é só mais rápido; é mais **previsível**.

**O ruído reduz a dispersão das WNN.** No `wnn-T16 a1s12`, o CT vai de
`60,39 ± 6,70` (limpo) para `46,35 ± 2,84` (σ=10 sem KF) — média e desvio caem
juntos. Coerente com o dithering: o que gerava a cauda longa eram os episódios
travados, e o ruído os elimina.

**A separação entre drones tem desvio enorme** (±25 a ±37 m sobre médias de 57 a
118 m). Ela é medida num único episódio por run — o comentário do
`spec02_maddpg_campanha.py` já dizia que é "~0,5 % de custo" adicional, e o preço
dessa economia é justamente esta variância. Não use essa coluna para comparar
modelos sem aumentar o n.

**A latência da campanha não compara os braços de forma útil.** A coluna mede a WNN
(NumPy) contra o MADDPG (PyTorch) — arcabouços diferentes — e o número do MADDPG
usa a medida antiga/assimétrica. A comparação honesta está no `spec06-merito.md`:
sob o protocolo da SPEC-03 (um processo por modelo, mesma observação, menor mediana
de 8×1000), a WNN é **~2× mais rápida** que o ator MADDPG no relógio (22–25 µs ×
49 µs, célula a2s24).

---

## Leitura

### 1. O MADDPG domina em tempo, e não é perto

Nas células comuns (`Na=2`), ele conclui em **18–36 s** contra **48–134 s** das
WNN. Em SR os três chegam perto sob ruído (99–100 %), mas o tempo separa por
fator 2–4×.

Duas ressalvas. A primeira é do lado do baseline: os tensorboard logs mostram que
o MADDPG treinou com **50 M** de passos contra os **6,75 M** das WNN — **7,4×
mais**. E 80 % desse orçamento foi gasto reduzindo o tempo DEPOIS que o SR já
saturara, o que explica boa parte da vantagem em `CT`. A segunda é que a
comparação é **cross-método** (CTDE contra crítico descentralizado on-policy).
Detalhes em [`../docs/teoria/maddpg-treinamento.md`](../docs/teoria/maddpg-treinamento.md).

### 2. O KF só ajuda quem não trava

| modelo | efeito do KF |
|---|---|
| **MADDPG** | **−0,64 a −0,74 s** nas três células, IC sem zero. **Ajuda** |
| **WNN** | **+1,1 a +27,1 s**, e SR cai 1,3 a 6,5 pp. **Atrapalha** |

O sinal é oposto e consistente. O mecanismo é o da SPEC-08: a WNN **empaca** em
parte dos cenários, o ruído a destrava, e o filtro **remove o ruído junto com o
ganho**. O MADDPG já opera em 100 % de SR — não tem o que destravar, então
filtrar é ganho líquido.

O caso extremo é `wnn-T32 a2s12`: o KF custa **+27 s** e −6,5 pp.

**Magnitude importa.** O ganho do MADDPG é 0,7 s em ~30 s — **2,4 %**. É
significativo com 20 mil episódios pareados, mas pequeno. IC que exclui zero com
n desse tamanho detecta efeito que pode não importar na prática.

### 3. O ruído ajuda TODAS as WNN, e quase nada o MADDPG

| | Δ SR ao trocar limpo por ruído bruto |
|---|---|
| WNN | **+3,9 a +16,3 pp** |
| MADDPG | +0,0 a +0,6 pp |

E o ganho é maior onde o SR limpo é pior — `wnn-T16 a2s12` vai de 79,1 % para
95,4 %. É o que o mecanismo prevê: quanto mais a política trava, mais há para
destravar.

### 4. T=16 × T=32 — a inversão com o número de drones

| célula | T=16 | T=32 | quem ganha |
|---|---|---|---|
| Na=1 Ns=12 limpo | **95,0 %** | 88,9 % | **T=16** por 6,1 pp |
| Na=2 Ns=12 limpo | 79,1 % | **85,9 %** | T=32 por 6,8 pp |
| Na=2 Ns=24 limpo | 90,8 % | **93,8 %** | T=32 por 3,0 pp |
| Na=2 Ns=36 limpo | 94,6 % | **95,5 %** | T=32 por 0,9 pp |

O `T=16` ganha em 1 drone e perde nas três células de 2 drones. Com **uma
semente de treino de cada lado**, isto não é veredito sobre arquitetura — a
seção 2 do `spec08-t16-vs-t32-ruido.md` mostra que a semente sozinha valeu 10,8
pp. Mas o padrão é consistente nas quatro células e merece as ≥5 sementes que o
protocolo pede.

## Ressalvas de método

**Transferência.** Nas células `Ns ∈ {12, 36}` as WNN foram treinadas em `Ns=24`
e o MADDPG tem checkpoint por célula — ele joga em casa, elas transferem. A
coluna `transferencia` nos CSVs marca isso.

**Uma semente de treino.** Todos os vereditos são condicionais ao modelo treinado;
o protocolo pede ≥5 para declarar hipótese.

**Não há `Na = 1` no MADDPG.** Lamenza não publica single-drone, então a célula
mais difícil da grade não tem baseline — ali a referência é o guloso (SR 100 %,
CT ≈ 34 s).

**O artefato avaliado é o podado.** As WNN rodaram o `models/deploy/` — só o
ator, `float32`/`uint16`, podado a 70 %. Tamanho e desempenho vêm do mesmo
artefato, de propósito.

## Reprodução

```bash
bash scripts/spec06_campanha_fila.sh          # 33 corridas
uv run python scripts/spec06_campanha_tabela.py --md
```
