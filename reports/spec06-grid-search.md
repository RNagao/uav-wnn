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

## 8. Onde a varredura está, e o que falta

Parcial do `desempate` (121 de 720 corridas):

| | melhor cobertura | melhor SR | configuração |
|---|---|---|---|
| dissertação | **76,2 %** | **64,0 %** | `passo` 8e-4, `σ` 0,35, `T` 32, `n` 12, `φ` 0,85 |
| anbubus | 27,0 % | 0,0 % | `passo` 3e-3, `σ` 0,35, `T` 32, `n` 12 |

Das 121 corridas, **4 têm SR > 0 e todas as 4 são da dissertação**. Todos os pontos
acima de 50 % de cobertura têm `n = 12`; `n = 24` não aparece.

Referências para leitura: caminhante aleatório 3,3 % de cobertura; guloso 100 % de SR
com `CT ≈ 34 s`.

**O que este documento NÃO conclui.** A grade está em 17 %. O veredito pareado sobre a
grade completa não está calculado, o portão G1 da SPEC-06 segue **não declarado**, e as
seis linhas de critério dele continuam abertas.

### Transferência para `Na > 1` e para os cenários com ruído

Pergunta levantada durante a varredura. A T-06.5 já pré-registra a resposta: a
escalada usa **a configuração vencedora da T-06.4**, sem re-sintonizar por célula.

Isso é o que preserva a comparação. Lamenza usa **um único conjunto** de
hiperparâmetros em todos os cenários (`3e-6` para ator e crítico, `γ = 0,99`,
`τ = 0,005`; o `benchmark.py` não sobrescreve). Re-sintonizar por cenário nos daria
vantagem de esforço que o baseline não teve — é o mesmo princípio do "pareamento do
orçamento de sintonia" da T-06.4.

Mas a T-06.5 registra **um gatilho de reconfirmação**, e a parcial acima sugere que ele
vai disparar: a observação cresce com `Na` (25 → 27 → 31 → 39 escalares), logo `B`
cresce, logo `N = ⌈B/n⌉` cresce. Todos os melhores pontos têm `n = 12`, e em
`Na = 8` com `T = 32` isso dá `N = 104` contra os 67 de hoje. O gatilho é a
**utilização de endereços** (T-01.4): se a ocupação por RAM cair muito, é capacidade
morta e `(n, N)` precisa ser reconfirmado.

Duas coisas transferem melhor do que parecem:

- **o termômetro quantil**, cujos limiares são reajustados por cenário a partir de um
  corpus colhido naquele cenário — então a mudança de distribuição do ruído de GPS é
  absorvida por construção;
- **`γ`**, que é fixado por autoridade e é o único número genuinamente compartilhado
  com Lamenza.

E uma que não transfere: `3e-6` do Lamenza **não** é comparável ao nosso `passo`. É
taxa nominal de Adam sobre rede densa 256×256, com replay e redes-alvo; o passo real é
normalizado pelo momento de segunda ordem. Nosso `passo_efetivo` é o deslocamento
direto em `f`. Só a nossa é interpretável como "quanto a saída se move".

---

## 9. Como reproduzir

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
