# SPEC-06 — Varredura de hiperparâmetros: o que foi medido, o que foi retirado

Registro da varredura que precede a T-06.5. O objetivo era um só: **escolher entre os
dois braços WNN** — `ntupla_dissertacao` (rede de gradiente funcional §4.3.1 + crítico
de regressão com esquecimento §4.2.1) e `ntupla_anbubus` (as duas redes de gradiente
funcional, convenção de soma) — em `Na = 1`, `Ns = 12`, sem ruído.

Três grades rodaram. **Duas conclusões minhas foram retiradas no caminho**, e as duas
pelo mesmo motivo: um canto do espaço de busca foi tomado pelo espaço todo. Isso está
registrado aqui porque é o achado metodológico mais reaproveitável do documento.

---

## 1. As três grades

| grade | eixos | pontos | corridas | `B` | parede |
|---|---|---|---|---|---|
| dois eixos | `α` × `σ` | 18 | 54 | 200 k | 1,6 h |
| `decisao` | `passo` × `σ` × `normalizar` | 18 | 108 | 75 k | 2,8 h |
| `desempate` | `passo` × `σ` × `T` × `n` × crítico | 120 | 720 | 200 k | ~6 h |

Os pontos são **compartilhados** pelos dois modelos: cada configuração tem par exato
no outro braço, com as mesmas sementes. A comparação é pareada por configuração, não
apenas por orçamento.

Duas coisas diferem entre braços, e as duas por necessidade:

- **a taxa**, convertida por amostra para a convenção de cada família;
- **o eixo do crítico**, que não pode ser o mesmo: o crítico da dissertação
  **acumula** retorno pela eq. (4.8) e o hiperparâmetro dele é `φ`; o do `anbubus` é
  treinado por descida e o hiperparâmetro dele é a taxa.

---

## 2. O que governa o passo efetivo

Derivado do código, não suposto. O passo do ator é `atualizar(x, α·δ·∇log π)`, e sob a
convenção de média `Δf = passo` exatamente (eq. 4.12). Com a normal projetada,
`∇log π = (z − f)/σ²`, e `(z − f)` tem magnitude `~σ`. Logo:

```
Δf = α · δ · (z − f)/σ²   ≈   α · δ / σ
```

Quatro fatores, e **`σ` está no denominador**: metade do `σ` dobra o passo.

Isso reinterpretou o primeiro achado da campanha. A grade de dois eixos mediu que
`σ = 0,10` dava 3,3 % de cobertura — nível do caminhante aleatório — em todas as seis
taxas, e eu li como *exploração estrangulada*. **Errado:** as 18 corridas de
`σ = 0,10` **abortaram** pela assertiva de estabilidade, todas. Não faltou exploração;
o passo dobrou e divergiu.

### Confundimento declarado

`α` e `σ` estão **confundidos** em todas as três grades: variar `σ` muda a amplitude da
exploração *e* o tamanho do passo. Isso é tolerável para a pergunta "qual modelo
ganha", porque o confundimento é idêntico nos dois braços — mesma política, mesmo `σ`
— e portanto não favorece nenhum. **Não** é tolerável para concluir qual `σ` é melhor.

A reparametrização que desconfunde é sortear `α' = α/σ` — o passo efetivo — em vez de
`α`. Não foi feita: entraria como mudança de desenho no meio da campanha. Fica para a
descida coordenada da T-06.4.

---

## 3. As duas conclusões retiradas

### 3.1 "A dissertação ganha por +2,1 pp" — retirada

Sobre 32 pontos pareados de uma grade interrompida, a dissertação vencia 21/32 com
Δ = +2,1 pp e IC95 % `[+0,7; +3,7]`, excluindo zero. Estatisticamente sólido, e ainda
assim inválido como veredito: aquelas corridas cobriam **um único nível** dos dois
eixos dominantes — `passo = 2e-4` e `σ = 0,20`.

A grade `decisao` mostrou por quê. Existe **interação**: os dois modelos têm escalas de
passo ótimas diferentes.

| `passo` | dissertação | anbubus | quem ganha |
|---|---|---|---|
| 2,0e-4 | 14,2 % | 7,7 % | dissertação, nos 6 pontos |
| 5,0e-4 | **18,0 %** | 15,9 % | dissertação, 5 de 6 |
| 1,5e-3 | 11,8 % | **18,5 %** | anbubus, 5 de 6 |

`passo = 2e-4` é exatamente o único nível onde a dissertação domina os seis pontos. O
resultado media a vantagem dela no canto que a favorece.

Sobre os 18 pontos da `decisao`, o veredito é **Δ = +0,7 pp, IC `[-2,9; +3,8]`** —
inclui zero. A grade não decidiu.

### 3.2 "O anbubus quer passo maior" — retirada

Da tabela acima, o anbubus subia monotonicamente (7,7 → 15,9 → 18,5 %) e ainda estava
subindo no topo da faixa. Estendi o `desempate` até `3e-3` por causa disso.

**Era viés de sobrevivência.** Em `1,5e-3` o anbubus já morria 8/18 na `decisao`, e a
média dele vinha dos sobreviventes. A 200 mil passos — 2,7× mais oportunidade de
estourar a assertiva — os dois níveis de topo morrem quase por completo:

| `passo` | abortos dissertação | abortos anbubus |
|---|---|---|
| 3,0e-4 | 0/6 | 0/8 |
| 5,0e-4 | 1/11 | 0/11 |
| 8,0e-4 | 7/11 | 0/8 |
| 1,5e-3 | **12/12** | **8/10** |
| 3,0e-3 | **10/10** | **8/8** |

Dois níveis de cinco são território morto, ou 288 das 720 corridas. O custo é baixo —
um aborto termina em ~1 min contra ~10,5 min de uma corrida real — mas o **desenho**
estava errado, e a origem do erro foi ler médias contaminadas por mortalidade
desigual.

### Regra que sai disso

**Média com mortalidade desigual não é média comparável.** Quando um braço morre 13/18
e o outro 8/18, as mortas entram com a cobertura do momento da morte e arrastam a
média do braço mais mortal. Reportar `falhas/R` junto de qualquer média — a spec já
manda isso (§4.5, "estabilidade de treino é métrica de primeira classe"); eu deixei de
aplicar na leitura.

---

## 4. O que a `decisao` estabeleceu

Separando por regime de estabilidade, o quadro é nítido:

| regime | Δ (dissertação − anbubus) | IC95 % | vitórias |
|---|---|---|---|
| **estável** (`passo` ≤ 5e-4) | **+4,3 pp** | **[+2,0; +6,3]** | **11/12** |
| instável (`passo` = 1,5e-3) | −6,7 pp | [−12,3; −1,3] | 1/6 |

E o instável não é comparável, pela regra da §3.

**A normalização móvel foi refutada.** Ela *piora* os dois — dissertação 15,8 → 13,5 %,
anbubus 14,9 → 13,2 %. A hipótese de que a estagnação do anbubus era artefato do nosso
codificador está descartada; era escala de passo. O eixo saiu do `desempate`, o que
cortou a grade pela metade e pagou pelo triplo de orçamento e pela volta de `T` e `n`.

**A assertiva está calibrada de forma justa entre os braços.** Verifiquei: o
`ganho_do_passo` multiplica por `N` na convenção de soma, então `c_max` mede a mudança
em `f` nos dois. A mortalidade maior da dissertação em passo grande é real, e tem
explicação mecânica — o crítico dela acumula pela eq. (4.8), o que produz picos de `δ`
maiores, e é `δ·∇log π` que a assertiva mede.

**O esquecimento não estabiliza.** Hipótese testada e não sustentada: `φ = 0,70` aborta
6/8, pior que `φ = 0,85` (15/26).

---

## 5. O orçamento da varredura, e por que 200 mil

Na `decisao`, a 75 mil passos, o **SR foi 0 % nas 108 corridas**. Ninguém completava um
episódio. Ela mediu "quem chega mais longe antes de desistir", não quem resolve a
tarefa — e a métrica de seleção teve de ser `cobertura`, porque `SR` e `CT_cens`
saturam (o caminhante aleatório também dá SR = 0 %).

A 200 mil, o `desempate` já produziu **SR = 64,0 %** e cobertura 76,2 %. A subida de
orçamento foi o que tornou a grade informativa.

Medições de custo que sustentam o dimensionamento:

- **13 min/corrida** a 200 mil passos, 12 processos, `T = 16`/`n = 12` (`N = 34`).
- A configuração mais caro/mais barata difere por **1,9×**, não pelos 18× que eu havia
  reportado (ver §6.1). O custo é linear em `N = ⌈obs·T/n⌉`.
- 12 processos entregam ~**1,9× a vazão de um**, em 6 núcleos físicos. O gargalo é
  banda de memória — a consulta em dicionário é *pointer chasing*. É o mesmo motivo
  pelo qual o hyperthreading ajudou **+23 %**, e pelo qual reduzir para 6 processos
  **piorou** (1320 contra 1624 corridas/h).
- A ordenação dos melhores pontos se preserva de 100 k para 200 k (ρ = 0,964, zero
  movimento de posto); a 50 k, ρ = 0,941 mas **só 2 dos 3 melhores sobrevivem**.

---

## 6. Defeitos de instrumentação encontrados

### 6.1 Autocontenção de bloco falseava o custo por configuração

`resolucao` é o terceiro eixo do produto cartesiano, então `T` alternava a cada 144
corridas e os 12 workers ficavam quase sempre na **mesma** configuração. O tempo por
corrida media autocontenção do bloco, não o custo daquele ponto: `T = 32` pareceu
**18× mais lento** que `T = 16`, quando isolado é **1,7×**. Eu havia apresentado "cortar
`T = 32` economiza 58 h" e tive de retirar.

Pior que a medição errada: um parcial lido no meio do caminho fica enviesado para o
bloco corrente, e a leitura comparativa deixa de valer.

**Correção:** embaralhamento com semente fixa (preserva reprodutibilidade).

### 6.2 Modelo no laço externo tornava os parciais não comparativos

Com o modelo no laço externo, a primeira rede rodava inteira antes de a segunda
começar — medido: 15 corridas concluídas, 15 da mesma rede. `chunksize` não resolve,
porque muda o lote e não a ordem. **Correção:** modelo no laço interno.

### 6.3 Parada antecipada impossível

`intervalo = B/20` colidia com a guarda `MINIMO_AVALIACOES = 20`: a parada só poderia
disparar quando o orçamento já tivesse acabado. Medido: nenhuma das 54 corridas parou
antes. **Correção:** `B/40`.

### 6.4 Renomeação silenciosa

A substituição de `"ntupla"` **com aspas** não casa dentro de `"ntupla,ntupla_funcional"`,
então o default de `--modelos` ficou apontando para uma chave inexistente e **todas** as
corridas daquele braço abortaram com `KeyError`. **Correção:** validar por `sys.exit` em
não-casamento, em vez de confiar que `ast.parse` detecta — ele não detecta.

### 6.5 A ETA sobe, e é esperado

Correlação +0,43 entre ordem de conclusão e duração. Com 12 workers, num instante
qualquer os concluídos são os rápidos e os lentos ainda ocupam núcleos; a média sobe
por lote (3,0 → 4,0 → 5,1 → 6,8 min) e converge. Não é degradação da máquina —
verificado o clock e a carga.

---

## 7. Retomada

`--retomar` pula corridas já concluídas em vez de truncar o registro de progresso. Uma
grade de 6 a 13 h **vai** ser interrompida.

Três filtros, cada um evitando uma contaminação diferente:

1. **Orçamento.** Registro treinado por outro `b_sweep` é outra corrida — o
   `config_hash` da §4.7 já separa as pastas por isso. Registros sem o campo são
   **recusados**, não presumidos: presumir misturaria corridas de 75 mil com as de 200
   mil e o comparativo cairia por terra.
2. **Pertinência.** Chave que não está nas tarefas desta grade sobrou de outra.
3. **Duplicidade.** Retomada de retomada: vale a última.

Corridas **abortadas contam como concluídas** e não são refeitas — o aborto é
determinação, não falha de infraestrutura: mesma semente, mesmo ponto, mesmo estouro.

O `id` (índice do produto cartesiano) **não** entra na chave, porque muda com o nível
da grade: a mesma configuração é `id=6` em `decisao` e `id=25` em `desempate`. Com ele
na chave, dois níveis que compartilham configuração e orçamento refariam o trabalho.

---

## 8. O que esta varredura NÃO fecha

A SPEC-06 tem `Na = 1` fechado; o **portão G1 segue não declarado**, e o que falta é
campanha, não varredura:

- **T-06.5**, a escalada `Na = 1 → 2 → 4 → 8` e `Ns ∈ {24, 36}`. É onde o MADDPG
  entra como baseline, porque os checkpoints publicados começam em 2 agentes.
- **T-06.6**, a tabela de custo T1/T2/T3, que o PRD §8 diz não poder cair.
- os critérios 2 a 6 e 8 a 9 do portão.

Três coisas que a varredura deixou medidas e não resolvidas:

**Os 720 pontos reportam o estado FINAL, não o de melhor validação.** A correção
entrou depois deles (`539b61e`), e o viés não é simétrico: quem aborta tem o estado
final no limite da divergência. Medido num caso, a diferença é de 14 pontos de SR.
Reavaliar os checkpoints `melhor` ficou barato (0,10 s por load) e é o próximo passo —
o Δ de +5,0 pp do regime estável é limite inferior.

**`α` e `σ` seguem confundidos** (`Δf ~ α·δ/σ`). Tolerável para comparar modelos, por
ser simétrico entre os braços; não tolerável para concluir qual `σ` é melhor.

**O recorte de `δ` nunca foi testado.** A própria mensagem da assertiva sugere
*"reduza η ou recorte δ"*, e três das quatro maiores perdas por aborto eram anbubus
com 42 a 61 % de cobertura — uma morta por estourar `c_max` em 0,00. Sob recorte,
aquele nível pode ser outra história.

---

## 9. O MELHOR MODELO TREINADO

    dissertação · passo 8,0e-4 · σ 0,35 · T = 16 · n = 12 (N = 34) · φ = 0,85 · m = 8
    semente 0 · 6 750 000 passos (convergiu, não esgotou o orçamento de 10 M)

    banco `test`, 100 episódios:   cobertura 98,5 %   SR 97,0 %
                                   CT_cens 52 s       CT_succ 38,2 s
    checkpoint: 2,6 MB · 400 bits de entrada · 34 RAMs

`runs/spec06-longa/spec06-busca-1b/8b328b1d6887/0/checkpoints/melhor`

Os quatro pontos da corrida longa, todos em `passo 8,0e-4 σ 0,35 φ 0,85`:

| T | semente | cobertura | SR | CT_cens | CT_succ | passos | checkpoint |
|---|---|---|---|---|---|---|---|
| **16** | **0** | **98,5 %** | **97,0 %** | 52 s | **38,2 s** | 6,75 M | 2,6 MB |
| 32 | 1 | 97,8 % | 95,0 % | 62 s | 38,5 s | 6,75 M | 5,9 MB |
| 32 | 0 | 95,4 % | 91,0 % | 81 s | 39,2 s | 5,9 M | 5,5 MB |
| 16 | 1 | 92,8 % | 89,0 % | 93 s | 42,5 s | 7,25 M | 2,8 MB |

Referências: guloso 100 % de SR com CT ≈ 34 s; caminhante aleatório 3,3 % de
cobertura e SR 0 %.

**O `CT_succ` de 38,2 s contra os 34 s do guloso é o número que mais diz.** Quando a
rede completa a coleta, ela leva 12 % mais tempo que a heurística que vê o mapa
inteiro — e ela decide por observação local, sem sinalização.

**`T = 16` ganha de `T = 32`, e isso importa para o custo.** Mesmo SR e metade do
checkpoint: 2,6 MB contra 5,9 MB, 400 bits contra 800, 34 RAMs contra 67. A grade a
200 mil passos apontava `T = 32` como melhor; a 6,75 M os dois empatam em desempenho
e `T = 16` ganha em tamanho. É uma inversão de orçamento, do mesmo tipo das outras
que esta campanha produziu.

**Não há MADDPG para comparar em `Na = 1`.** Os checkpoints publicados de Lamenza
começam em 2 agentes — registrado em `spec02-baseline-maddpg.md`. A comparação com o
baseline é da T-06.5, em `Na ≥ 2`; aqui a referência é o guloso.

---

## 10. As três campanhas e o que cada uma estabeleceu

| campanha | orçamento | corridas | o que fechou |
|---|---|---|---|
| grade `desempate` | 200 mil | 720 | qual **rede** funciona sob o Algoritmo 4.4 |
| corrida longa | até 10 M | 12 | qual **modelo** resolve a tarefa |
| grade PPO | 200 mil | 135 | qual **algoritmo** serve a rede do anbubus |

### O veredito entre as redes, sob o Algoritmo 4.4

| regime | Δ (dissertação − anbubus) | IC95 % | pontos |
|---|---|---|---|
| todos os 120 | +1,0 pp | [−1,0; +3,1] | 49/120 — inclui zero |
| **estável** (`passo` ≤ 8e-4) | **+5,0 pp** | **[+2,3; +7,8]** | 43/72 |
| instável (≥ 1,5e-3) | −4,9 pp | [−7,0; −3,2] | 6/48 |

No regime em que as duas são estáveis a dissertação ganha com IC excluindo zero.
Sobre a grade toda inclui zero, e o regime instável **não é média comparável**: a
mortalidade é desigual (187/360 contra 137/360) e as abortadas entram com a cobertura
do momento da morte.

### O braço PPO, no mesmo orçamento

| | cobertura | SR |
|---|---|---|
| PPO não treinado | 10,8 % | 0,0 % |
| **anbubus sob PPO**, melhor ponto | **44,6 %** | **0,0 %** |
| anbubus sob Algoritmo 4.4 | 46,6 % | 10,3 % |
| dissertação sob Algoritmo 4.4 | 50,1 % | 21,3 % |

**A cobertura do PPO chega perto; o SR não sai de zero.** O melhor ponto dele recolhe
44,6 % dos sensores e **nunca** fecha um episódio; o segundo melhor com SR > 0 faz
3,3 % de média. Cobertura mede quanto se recolhe, SR mede se se recolhe tudo — e é o
SR que Lamenza reporta.

Isso responde à pergunta que motivou o braço: **não, o PPO não resolve a
instabilidade do anbubus por si.** Ele diverge em 22 de 124 corridas com `nan` na
distribuição de ação, taxa comparável à do Algoritmo 4.4 — o que derruba o argumento
que eu sustentei por várias mensagens, de que o `clip_param = 0,3` era a válvula que
faltava.

### Duas assimetrias entre os braços, declaradas

**σ não transfere.** No PPO ela é aprendida (`policy_output_dim = 2 × dim de ação`);
no Algoritmo 4.4 é fixa e varrida, pelas quatro razões da §4.4. O eixo simplesmente
não existe no PPO, então o espaço pareado tem 45 pontos contra 120.

**`n = 24` não cabe no modelo publicado.** A alocação do `NTNModel` é DENSA
(`n_nodes · 2^n`): 2²⁴ endereços por RAM dão 6,7 a 13,4 GB só no ator, numa máquina
de 11,9 GB — medido, 18 corridas mortas por SIGKILL. Nossa implementação é esparsa e
varreu `n = 24` sem esforço, medindo que ele colapsa (4,9 % contra 31,7 %). O braço
PPO não pode nem testar. A perda científica é pequena porque o nível que não cabe já
era ruim; a de pareamento é real.

E o `_keys` deles exige `n | 25·T`, porque o `conv1d(stride=n)` dá `floor(B/n)` chaves
enquanto `n_nodes` dá `ceil(B/n)`. Nenhuma combinação de `T ∈ {16, 32}` com
`n ∈ {12, 24}` roda.

---

## 11. Configuração provisória, para desenvolvimento em paralelo

Escolhida sobre a **parcial** do `desempate` (160 de 720 corridas) para destravar
trabalho em outra máquina. É provisória: a grade não fechou.

| eixo | valor | por quê |
|---|---|---|
| aproximador | `ntupla_dissertacao` | as 4 corridas com SR > 0 são todas dela |
| `passo_efetivo` | **5e-4** | o maior passo amplamente estável |
| `σ` | **0,20** | o ponto de 3 sementes com melhor cobertura |
| `T` / `n` | **32 / 12** (`N` = 67) | `n = 12` domina `n = 24` por 3 a 7× |
| `φ` | **0,85** | melhor cobertura das vivas |
| `m` | 8 | não varrido — limitação declarada |
| `normalizar` | desligado | medido: piora os dois braços |

Medido nesse ponto, 3 sementes, `B` = 200 mil: cobertura **49,3 %** (54,8 / 74,1 /
19,0), SR 12,7 %, **0/3 abortos**. Referências: caminhante aleatório 3,3 %; guloso
100 % de SR com CT ≈ 34 s.

### Por que 5e-4 e não 8e-4, que rende mais

Mortalidade. Na dissertação, por nível de passo:

| `passo` | abortos | cobertura das vivas |
|---|---|---|
| 3e-4 | 0/16 | 14,0 % |
| **5e-4** | **2/19** | **26,1 %** |
| 8e-4 | 9/15 | 37,7 % |
| 1,5e-3 | 15/15 | — |
| 3e-3 | 15/15 | — |

`8e-4` rende mais **entre as que sobrevivem**, e mata 60 %. Para desenvolvimento,
uma configuração que aborta 3 de 5 vezes não serve.

### A armadilha que isso remove

O padrão da `NormalProjetada` é `σ = 0,1`, e o treinador não o sobrescrevia. A
primeira grade mediu que **as 18 corridas com `σ = 0,10` abortaram**, todas. Não é
exploração insuficiente: o passo efetivo é `~α·δ/σ`, com `σ` no **denominador**,
então metade do `σ` dobra o passo. Quem clonasse o repositório e rodasse o
treinador teria divergência garantida.

Agora `scripts/spec06_treinar.py` fixa `PASSO_EFETIVO` e `SIGMA` explicitamente, com
`ap_taxa()` convertendo para a convenção de cada família (`passo` na média,
`passo/N` na soma). A configuração também está em
`reports/spec06-config-provisoria.json`, legível por máquina.

**Ressalva:** a variância entre sementes nesse ponto é alta — 19,0 %, 54,8 %,
74,1 %. Uma única semente não é sinal.

---

## 12. Como reproduzir

```bash
# grade de decisão (2,8 h)
uv run python scripts/spec06_busca.py --modo grade --niveis decisao \
  --sementes 3 --b-sweep 75000 --processos 12 \
  --raiz runs/spec06-decisao --saida reports/spec06-decisao.json

# grade de desempate (~6 h)
uv run python scripts/spec06_busca.py --modo grade --niveis desempate \
  --sementes 3 --b-sweep 200000 --processos 12 --retomar \
  --raiz runs/spec06-desempate --saida reports/spec06-desempate.json

# parciais durante a execução
uv run python scripts/spec06_parciais.py reports/spec06-desempate.progresso.jsonl
```

Definir `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`: sem isso o BLAS abre threads por
processo e briga com o pool.
