# Como o MADDPG foi treinado — evidência dos logs

Investigação sobre o treino do baseline, feita a partir dos **tensorboard logs**
que acompanham os checkpoints publicados — não dos defaults do `main.py`.

Complementa [`maddpg.md`](maddpg.md), que descreve a arquitetura e o algoritmo.

---

## 1. A correção que abre este documento

O `main.py` declara `total_timesteps: int = 1000000`. **Não foi esse o valor
usado.** Os logs mostram:

```
a_2-s_12   49 999 999 passos    a_4-s_12   49 999 999    a_8-s_12   49 999 999
a_2-s_24   49 999 999           a_4-s_24   49 999 999    a_8-s_24   49 999 999
a_2-s_36   49 999 999           a_4-s_36   49 999 999    a_8-s_36   49 999 999
```

**50 milhões de passos em todas as nove células.** O `1 000 000` é o default do
`dataclass`, sobrescrito na linha de comando.

> Isto inverte uma comparação que eu vinha reportando. Eu dizia "o MADDPG treinou
> com 1 M e as WNN com 6,75 M" — a razão real é **50 M contra 6,75 M**, ou seja,
> **o baseline treinou 7,4× MAIS**. O `relatorio-geral.md` já registrava os 50 M;
> o erro estava em `maddpg.md`, e está corrigido.

---

## 2. Onde os logs estão

Ao lado de cada checkpoint:

```
vendor/gradys-data-collection/scenario/003-collaboration/runs/results/
└── results__a_2-s_24__1__1723034081/
    ├── a_2-s_24-checkpoint4999.cleanrl_model      ← os pesos que usamos
    └── events.out.tfevents.1723034081.vm116.cloud.inf.puc-rio.br...
```

O hostname (`vm116.cloud.inf.puc-rio.br`) e o carimbo Unix (`1723034081` =
07/08/2024) datam a corrida. O `checkpoint4999` é o índice do último salvamento,
a cada 10 mil passos.

**17 séries registradas** — 8 de treino a cada 10 mil passos (4 997 pontos), 6 de
avaliação a cada 100 mil (500 pontos), e 3 de perdas.

---

## 3. A curva de aprendizado — célula `a_2-s_24`

```
       passos       SR       CT
      100 000     0,0%   244,9 s
    1 000 000     0,0%   170,2 s     ← onde o default teria PARADO
    5 000 000     0,5%   110,3 s
   10 000 000    84,0%    39,1 s
   20 000 000   100,0%    22,4 s
   30 000 000   100,0%    21,1 s
   50 000 000    99,5%    19,0 s
```

**Em 1 M de passos o SR era ZERO.** Se a corrida tivesse parado no default, o
baseline não existiria — e a linha dos 5 M ainda dá 0,5 %.

### Quando cada marco foi cruzado (e mantido)

| marco | passo | fração do orçamento |
|---|---|---|
| SR ≥ 50 % | 9,3 M | 18,6 % |
| SR ≥ 90 % | 10,2 M | 20,4 % |
| SR ≥ 95 % | 10,5 M | 21,0 % |
| SR ≥ 99 % | 10,7 M | 21,4 % |
| SR = 100 % | 26,3 M | 52,6 % |

**A transição é abrupta.** Entre 9,3 M e 10,7 M — 2,8 % do orçamento — o SR vai
de 50 % para 99 %. Antes disso, nada; depois, refinamento.

### O tempo continuou caindo muito depois do SR saturar

A partir de 7,3 M, com o SR já perto do teto, o `CT` foi de **95,6 s para 19,0 s
— 80 % de redução**. Os 40 M de passos finais não compraram taxa de sucesso;
compraram **velocidade**.

É o que explica a vantagem de tempo do baseline na nossa campanha: ele não é
melhor por saber resolver, é melhor por ter passado 80 % do orçamento otimizando
o *como*.

---

## 4. O custo de parede

```
SPS: mediana 40 passos/s   (início 359 → fim 33)
```

A vazão **cai 10×** ao longo do treino — o *replay buffer* enche (10⁶) e a
amostragem fica cara.

| | |
|---|---|
| 50 M passos a 40 passos/s | **347 h ≈ 14,5 dias** por célula |
| as nove células | **≈ 3 125 h ≈ 130 dias de máquina** |

Rodadas numa VM da PUC-Rio, presumivelmente em paralelo.

> Para comparar: a nossa corrida longa da WNN fez 6,75 M em ~7 h (256 passos/s).
> O baseline é **6,4× mais lento por passo** e usou **7,4× mais passos**.

---

## 5. As perdas — e um sinal que merece nota

| série | 1 M | 10 M | 50 M |
|---|---|---|---|
| `qf1_loss` | 277,6 | 767,6 | **63,3** |
| `qf1_values` | 119,5 | 394,4 | **1 216,6** |
| `actor_loss` | −120,7 | −396,4 | **−1 215,1** |

**O `qf1_loss` sobe antes de cair** — pico em 10 M, exatamente onde o SR
transiciona. O crítico erra mais enquanto a política muda rápido, e converge
depois (63,3 ao fim, 4,4× abaixo do valor de 1 M).

**`qf1_values` cresce 10× e não estabiliza.** O valor estimado vai de 119 a
1 217, e `actor_loss = −qf1_values` por construção. Crescimento sustentado de Q
costuma indicar **superestimação** — o modo de falha clássico do DDPG, que o TD3
ataca com crítico duplo. Aqui há um só (`qf1`), sem par.

Isto **não invalida o baseline**: o desempenho medido é real, e vem da política,
não do valor estimado. Mas é uma característica do treino que vale registrar —
e explica por que `actor_loss` diverge sem que o SR piore.

---

## 6. O horizonte encolheu junto

```
passo  1 000 000: episódio dura 48,7 s
passo 10 000 000: episódio dura 47,3 s
passo 49 990 000: episódio dura 26,3 s
```

Como o treino usa `end_when_all_collected=False`, a duração do episódio é o
horizonte inteiro até a coleta terminar. Encolher de 48,7 s para 26,3 s é a mesma
melhora de `CT` vista de outro ângulo.

---

## 7. O que isto muda nas nossas comparações

**A ressalva de orçamento inverte de lado.** Eu vinha escrevendo que o MADDPG
teve menos treino; ele teve **7,4× mais**. Quando a WNN perde em tempo, parte da
diferença é orçamento — e igualar exigiria 50 M de passos na WNN, que a
6,75 M/7 h seriam ~52 h por célula.

**A comparação de SR fica mais favorável à WNN do que parecia.** Sob ruído as WNN
chegam a 97–99,4 % contra 100 % do MADDPG, com 13 % do orçamento de treino.

**A vantagem de tempo do baseline é explicada.** 80 % do orçamento dele foi gasto
reduzindo `CT` depois que o SR já estava resolvido. Não é uma diferença de
capacidade de resolver a tarefa; é de refinamento pago em passos.

---

## 8. Reprodução

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = "vendor/gradys-data-collection/scenario/003-collaboration/runs/results"
acc = EventAccumulator(f"{d}/results__a_2-s_24__1__1723034081/events.out.tfevents...")
acc.Reload()
acc.Scalars("eval/all_collected_rate")     # a curva de SR
```

`tensorboard` não é dependência do projeto — use `uv run --with tensorboard`.
