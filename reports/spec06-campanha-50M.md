# Campanha comparável — MADDPG × WNN T=16 × WNN T=32

100 runs × 200 episódios por célula. Todos os modelos veem as **mesmas**
sementes, e no regime com ruído o mesmo episódio cai no mesmo degrau —
a comparação é pareada em tudo.

`CT_cens` conta o fracasso como o horizonte (500 s); `CT_succ` só os que
completaram. Reportar só o segundo premiaria quem desiste dos difíceis.


## Desempenho por cenário

| Na | Ns | ruído | modelo | CT_cens | CT_succ | SR |
|---|---|---|---|---|---|---|
| 1 | 12 | sem ruído | wnn-T16 | 51.9 s | 37.4 s | 96.9% |
| 1 | 12 | sem ruído | wnn-T32 | 57.5 s | 38.8 s | 95.9% |
| 1 | 12 | σ=10 · SEM KF | wnn-T16 | 46.6 s | 44.0 s | 99.4% |
| 1 | 12 | σ=10 · SEM KF | wnn-T32 | 59.1 s | 47.1 s | 97.4% |
| 1 | 12 | σ=10 · COM KF | wnn-T16 | 42.1 s | 38.5 s | 99.2% |
| 1 | 12 | σ=10 · COM KF | wnn-T32 | 48.3 s | 40.1 s | 98.2% |
| 2 | 12 | sem ruído | maddpg | 18.4 s | 18.2 s | 100.0% |
| 2 | 12 | sem ruído | wnn-T16 | 78.3 s | 33.8 s | 90.5% |
| 2 | 12 | sem ruído | wnn-T32 | 80.3 s | 32.9 s | 89.8% |
| 2 | 12 | σ=10 · SEM KF | maddpg | 19.3 s | 19.3 s | 100.0% |
| 2 | 12 | σ=10 · SEM KF | wnn-T16 | 46.7 s | 38.9 s | 98.3% |
| 2 | 12 | σ=10 · SEM KF | wnn-T32 | 45.9 s | 37.7 s | 98.2% |
| 2 | 12 | σ=10 · COM KF | maddpg | 18.6 s | 18.5 s | 100.0% |
| 2 | 12 | σ=10 · COM KF | wnn-T16 | 54.1 s | 34.8 s | 95.9% |
| 2 | 12 | σ=10 · COM KF | wnn-T32 | 58.0 s | 34.3 s | 94.9% |
| 2 | 24 | sem ruído | maddpg | 30.3 s | 27.7 s | 99.4% |
| 2 | 24 | sem ruído | wnn-T16 | 80.7 s | 43.4 s | 91.8% |
| 2 | 24 | sem ruído | wnn-T32 | 68.0 s | 42.0 s | 94.3% |
| 2 | 24 | σ=10 · SEM KF | maddpg | 29.6 s | 29.6 s | 100.0% |
| 2 | 24 | σ=10 · SEM KF | wnn-T16 | 57.7 s | 49.9 s | 98.3% |
| 2 | 24 | σ=10 · SEM KF | wnn-T32 | 57.0 s | 48.6 s | 98.1% |
| 2 | 24 | σ=10 · COM KF | maddpg | 28.9 s | 28.0 s | 99.8% |
| 2 | 24 | σ=10 · COM KF | wnn-T16 | 59.9 s | 44.6 s | 96.6% |
| 2 | 24 | σ=10 · COM KF | wnn-T32 | 58.8 s | 43.1 s | 96.6% |
| 2 | 36 | sem ruído | maddpg | 35.4 s | 33.7 s | 99.6% |
| 2 | 36 | sem ruído | wnn-T16 | 81.5 s | 48.9 s | 92.8% |
| 2 | 36 | sem ruído | wnn-T32 | 62.2 s | 45.9 s | 96.4% |
| 2 | 36 | σ=10 · SEM KF | maddpg | 35.8 s | 35.8 s | 100.0% |
| 2 | 36 | σ=10 · SEM KF | wnn-T16 | 66.1 s | 56.8 s | 97.9% |
| 2 | 36 | σ=10 · SEM KF | wnn-T32 | 61.2 s | 53.6 s | 98.3% |
| 2 | 36 | σ=10 · COM KF | maddpg | 35.2 s | 34.3 s | 99.8% |
| 2 | 36 | σ=10 · COM KF | wnn-T16 | 62.1 s | 50.2 s | 97.4% |
| 2 | 36 | σ=10 · COM KF | wnn-T32 | 57.2 s | 46.9 s | 97.7% |

## O KF ajuda?

Diferença de tempo, pareada pelas mesmas sementes. **Negativo = o KF é
mais rápido.** IC 95 % por bootstrap pareado.

| Na | Ns | modelo | n | sem KF | com KF | Δ tempo | IC 95 % | SR sem→com |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | wnn-T16 | 20,000 | 46.55 s | 42.12 s | -4.43 s ** | [-5.17, -3.72] | 99.4% → 99.2% |
| 1 | 12 | wnn-T32 | 20,000 | 59.09 s | 48.25 s | -10.84 s ** | [-12.05, -9.54] | 97.4% → 98.2% |
| 2 | 12 | maddpg | 20,000 | 19.28 s | 18.62 s | -0.66 s ** | [-0.76, -0.54] | 100.0% → 100.0% |
| 2 | 12 | wnn-T16 | 20,000 | 46.70 s | 54.08 s | +7.38 s ** | [+6.01, +8.79] | 98.3% → 95.9% |
| 2 | 12 | wnn-T32 | 20,000 | 45.90 s | 58.03 s | +12.13 s ** | [+10.64, +13.74] | 98.2% → 94.9% |
| 2 | 24 | maddpg | 20,000 | 29.64 s | 28.90 s | -0.74 s ** | [-1.01, -0.44] | 100.0% → 99.8% |
| 2 | 24 | wnn-T16 | 20,000 | 57.69 s | 59.91 s | +2.22 s ** | [+0.87, +3.59] | 98.3% → 96.6% |
| 2 | 24 | wnn-T32 | 20,000 | 56.96 s | 58.79 s | +1.84 s ** | [+0.37, +3.27] | 98.1% → 96.6% |
| 2 | 36 | maddpg | 20,000 | 35.82 s | 35.18 s | -0.64 s ** | [-0.95, -0.34] | 100.0% → 99.8% |
| 2 | 36 | wnn-T16 | 20,000 | 66.10 s | 62.10 s | -4.00 s ** | [-5.36, -2.70] | 97.9% → 97.4% |
| 2 | 36 | wnn-T32 | 20,000 | 61.23 s | 57.19 s | -4.04 s ** | [-5.25, -2.72] | 98.3% → 97.7% |

`**` = o IC não contém zero.

## O ruído ajuda? (dithering)

Diferença entre o ruído bruto e o teto limpo. **SR positivo = o ruído
melhora**, que é o achado da SPEC-08.

| Na | Ns | modelo | limpo | σ=10 sem KF | Δ SR |
|---|---|---|---|---|---|
| 1 | 12 | wnn-T16 | 96.9% | 99.4% | **+2.6 pp** |
| 1 | 12 | wnn-T32 | 95.9% | 97.4% | **+1.4 pp** |
| 2 | 12 | maddpg | 100.0% | 100.0% | **+0.0 pp** |
| 2 | 12 | wnn-T16 | 90.5% | 98.3% | **+7.8 pp** |
| 2 | 12 | wnn-T32 | 89.8% | 98.2% | **+8.4 pp** |
| 2 | 24 | maddpg | 99.4% | 100.0% | **+0.6 pp** |
| 2 | 24 | wnn-T16 | 91.8% | 98.3% | **+6.4 pp** |
| 2 | 24 | wnn-T32 | 94.3% | 98.1% | **+3.8 pp** |
| 2 | 36 | maddpg | 99.6% | 100.0% | **+0.4 pp** |
| 2 | 36 | wnn-T16 | 92.8% | 97.9% | **+5.1 pp** |
| 2 | 36 | wnn-T32 | 96.4% | 98.3% | **+1.9 pp** |
