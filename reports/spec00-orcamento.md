# SPEC-00 T-00.10 — Orçamento de simulação e fidelidade do integrador (P2)

Máquina: WSL2, um núcleo, `execution_logging=False`. Política determinística (`GreedyPolicy`), sementes pareadas.

---

## 1. Fidelidade: `update_rate` — e o resultado contraria o código de referência

**Pergunta:** o passo do integrador de mobilidade muda o resultado?

**Armadilha encontrada primeiro.** O handler de mobilidade reagenda por **soma repetida** (`current_time + update_rate`), enquanto a barreira de época soma `interval`. Com `update_rate = 0,1` — que não é exato em binário — os dois relógios **derivam** e o número de tiques por época passa a oscilar. Medido: a sequência de deslocamentos por época foi **4,5 / 9,0 / 6,0 / 7,5 m** em vez de 7,5 constante. A média fica certa, a dinâmica não.

Por isso a varredura abaixo usa **apenas frações diádicas** (`interval / 2^k`), e `ScenarioConfig` passou a **recusar** valores não diádicos (teste 26).

Célula 4×24, 200 episódios pareados, referência fina `interval/64 = 0,0078125 s` (média 37,75 s):

| `update_rate` | `interval/2^k` | média (s) | viés (s) | IC95 da diferença pareada | KS *p* | s/ep |
|---|---|---|---|---|---|---|
| 0,250000 | /2 | 40,72 | **+2,97** | [+1,77, +4,17] | 0,000 | 0,088 |
| 0,125000 | /4 | 39,68 | +1,93 | [+1,15, +2,70] | 0,088 | 0,107 |
| 0,062500 | /8 | 38,78 | +1,02 | [+0,32, +1,73] | 0,628 | 0,146 |
| **0,031250** | **/16** | **38,21** | **+0,46** | **[−0,24, +1,16]** | **0,965** | **0,227** |
| 0,015625 | /32 | 38,02 | +0,27 | [−0,08, +0,62] | 1,000 | 0,383 |

**O código de referência usa `interval/2`**, que introduz viés sistemático de **+2,97 s (+7,9 %)** no tempo de conclusão, com IC bem longe de zero.

**Mecanismo do viés.** O heartbeat é emitido em `0,99 × interval` após a ação. Com integrador grosseiro, a posição do nó nesse instante está um tique atrasada — o broadcast sai de um ponto que o UAV já deixou, e coletas se perdem. Quanto mais grosseiro o integrador, mais coletas perdidas, mais longo o episódio.

**Decisão:** adotamos `interval/16 = 0,031250 s`, o valor mais barato cujo IC95 da diferença pareada **contém zero**. Custa 2,6× a referência.

Isto **não altera o MDP** — observação, ação, recompensa e terminação são idênticos; é resolução numérica. Mas altera valores absolutos em ~8 % em relação aos publicados, e por isso a comparação rigorosa da SPEC-06 deve ser por **razão contra o guloso reimplementado**, não por segundos absolutos. `update_rate=0.25` continua disponível para reproduzir o comportamento exato da referência.

---

## 2. Custo: grade completa

`update_rate = interval/16`, 20 episódios por célula, guloso.

| Na | Ns | eventos/ep | s/ep | decisões | sucesso | t_conc (s) | **h/núcleo por 50 M passos** |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 27 872 | 0,078 | 123,2 | 1,00 | 31,1 | 8,8 |
| 1 | 24 | 62 382 | 0,165 | 149,0 | 1,00 | 44,0 | 15,4 |
| 1 | 36 | 100 235 | 0,258 | 164,1 | 1,00 | 51,5 | **21,8** |
| 2 | 12 | 29 767 | 0,096 | 121,2 | 1,00 | 30,1 | 5,5 |
| 2 | 24 | 63 173 | 0,185 | 144,1 | 1,00 | 41,5 | 8,9 |
| 2 | 36 | 95 174 | 0,266 | 150,8 | 1,00 | 44,9 | 12,2 |
| 4 | 12 | 35 348 | 0,133 | 121,7 | 1,00 | 30,3 | 3,8 |
| 4 | 24 | 67 298 | 0,226 | 139,2 | 1,00 | 39,1 | 5,6 |
| 4 | 36 | 97 277 | 0,308 | 143,9 | 1,00 | 41,5 | 7,4 |
| 8 | 12 | 47 542 | 0,205 | 117,9 | 1,00 | 28,4 | 3,0 |
| 8 | 24 | 79 479 | 0,315 | 133,5 | 1,00 | 36,2 | 4,1 |
| 8 | 36 | 109 618 | 0,401 | 139,2 | 1,00 | 39,1 | 5,0 |

**P2 fechada: 3,0 a 21,8 h por núcleo** para o orçamento de 50 M passos de Lamenza. O PRD §6.1 estimava 8–30 h; a estimativa se sustenta, agora medida.

**Correção ao PRD e à SPEC-00.** Ambos supunham que a célula cara seria `Na = 8`. Medido:

- **Por episódio**, o custo é dominado por `Ns`: de `Ns=12` a `Ns=36` os eventos crescem 3,6× (Na=1); de `Na=1` a `Na=8` crescem 1,7× (Ns=12). A spec estava certa aqui.
- **Por campanha de treino**, a célula cara é `Na = 1, Ns = 36` (21,8 h), porque o contador de passos é `decisões × Na` — menos agentes significam menos passos por segundo simulado. `Na = 8, Ns = 36` custa apenas 5,0 h.

Consequência prática: **a Etapa 1, que roda `Na = 1`, é a mais cara em horas de treino de todo o projeto**, ao contrário do que a linha de corte do PRD §8 supunha ao listar `Na = 8` entre os primeiros cortes.

---

## 3. Critério de saída 5 da SPEC-00

Um episódio completo ponta a ponta com `GreedyPolicy` em **todas as 12 células** da grade `Na ∈ {1,2,4,8} × Ns ∈ {12,24,36}`: **satisfeito**, com taxa de sucesso 1,00 em todas.
