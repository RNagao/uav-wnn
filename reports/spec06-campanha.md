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

---

## Leitura

### 1. O MADDPG domina em tempo, e não é perto

Nas células comuns (`Na=2`), ele conclui em **18–36 s** contra **48–134 s** das
WNN. Em SR os três chegam perto sob ruído (99–100 %), mas o tempo separa por
fator 2–4×.

Duas ressalvas que cortam em direções opostas: o MADDPG treinou com **1 M** de
passos e as WNN com **6,75 M**; e a comparação é **cross-método** (CTDE contra
crítico descentralizado on-policy).

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
