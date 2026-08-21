# Plano de testes de fidelidade ao artigo

Fonte: Katopodis, *Reinforcement Learning with Weightless Neural Networks*,
dissertação, PESC/COPPE-UFRJ, jan/2022 (`tmp/947841_compressed.pdf`).

Este documento existe porque tínhamos **três níveis de garantia diferentes** e
os tratávamos como um só:

| nível | o que prova | onde |
|---|---|---|
| **A — oráculo externo** | concordamos com uma implementação de referência | `test_diferencial*.py`, `test_refenv.py` |
| **B — identidade do artigo** | implementamos a **matemática** do texto | `test_identidades_artigo.py` |
| **C — coerência interna** | o código concorda consigo mesmo | grande parte de `test_custo.py` |

O nível A é forte mas insuficiente: se o Katopodis e nós tivéssemos o mesmo
mal-entendido, os dois concordariam. O nível B é o que fecha esse flanco, e era
o que faltava por inteiro.

---

## 1. Auditoria — toda equação numerada

Legenda: **✅** testada · **🟡** parcial · **⬜** não testada · **—** fora de escopo

### Capítulo 3 — o modelo original

| eq. | afirmação | status | teste |
|---|---|---|---|
| 3.15 | `V ← V + y`, `C ← C + 1` | ✅ | via 4.8 com `φ = 1` |
| 3.16 | `f(x) = ΣV / ΣC` | ✅ | `test_critico_sem_esquecimento_e_media_simples` |
| 3.17 | distância de tupla `δ = Σᵢ⟦τᵢ(x) ≠ τᵢ(z)⟧` | ✅ | `test_distancia_de_tupla_e_zero_sse_padroes_identicos` |
| 3.18 | indicador `M[i]ₜ(x)` | ✅ | usado por 3.19–3.21 |
| 3.19 | `V[i](x) = Σₜ yₜ M[i]ₜ(x)` | ✅ | `test_valor_e_contador_sao_somas_sobre_o_conjunto` |
| 3.20 | `C[i](x) = Σₜ M[i]ₜ(x)` | ✅ | idem |
| 3.21 | `Σᵢ M[i]ₜ(x) = N(1 − δ/N)` | ✅ | `test_soma_dos_indicadores_e_N_vezes_o_nucleo` |
| **3.22** | **a rede é um estimador de Nadaraya-Watson** | ✅ | `test_rede_de_regressao_e_estimador_de_nadaraya_watson` |
| 3.23 | termômetro, `>` estrito, limiares uniformes | ✅ | `test_termometro_do_artigo_segue_a_eq_3_23` |
| 3.24 | circular, `n_shift = ⌊(x−l)/(u−l)·R⌋` | ✅ | `test_circular_do_artigo_segue_a_eq_3_24` |
| §3.2.4 | resolução e tupla ↔ complexidade/generalização | ✅ | 3 testes, ver §3 abaixo |

### Capítulo 4 — as redes de RL

| eq. | afirmação | status | teste |
|---|---|---|---|
| 4.1–4.7 | limitações da NTRN (argumento, não algoritmo) | — | prosa |
| 4.8 | `v ← φv + y`, `c ← c + 1` | ✅ | oráculo + `test_critico_com_esquecimento_pesa_o_recente` |
| 4.9 | `f = Σv / Σs`, zero se `Σs = 0` | ✅ | oráculo + `test_critico_devolve_zero_onde_nada_foi_escrito` |
| 4.10 | `s = (1 − φ^c)/(1 − φ)`, ramo `φ = 1` | ✅ | oráculo com `φ ∈ {0,50; 0,85; 1,00}` |
| 4.11 | `v ← v + αΔ` | ✅ | oráculo + `test_ator_usa_media_e_o_passo_independe_de_N` |
| 4.12 | `f = (1/N) Σᵢ v[i]` — **média** | ✅ | idem |
| 4.13 | `v[i]ₜ(x) = Σⱼ αⱼ M[i]ⱼ(x)` | ✅ | `test_memoria_do_ator_e_soma_das_atualizacoes` |
| 4.14–4.15 | indicador e núcleo (repetem 3.18/3.21) | ✅ | via 3.21 |
| **4.16** | **a rede de gradiente funcional é máquina de núcleo** | ✅ | `test_rede_de_gradiente_funcional_e_maquina_de_nucleo` |
| 4.17 | `v[i]ₜ(x) = v[i]ₜ₋₁(x) + αₜΔₜ M[i]ₜ(x)` | ✅ | `test_atualizacao_so_toca_posicoes_enderecadas` |
| **4.19** | **`fₜ(x) = fₜ₋₁(x) + αₜΔₜ·k(xₜ,x)`** — a propagação pelo núcleo | ✅ | `test_uma_atualizacao_propaga_pelo_nucleo` |
| 4.20–4.21 | `k` proporcional às tuplas compartilhadas | ✅ | via 3.21 e o mapa |
| **4.22** | **mapa de características `φ(x)`, `N·2ⁿ` binário** | ✅ | `test_mapa_e_um_quente_por_neuronio_e_normalizado` |
| **4.23** | **`⟨φ(w),φ(x)⟩ = k(w,x)`** → núcleo positivo-definido | ✅ | `test_mapa_de_caracteristicas_reproduz_o_nucleo`, `test_matriz_de_gram_e_positiva_semidefinida` |
| 4.26 | funcional de risco do REINFORCE | — | não usamos REINFORCE |
| 4.27–4.30 | política sigmoide (ação binária) | — | ação contínua |
| 4.31–4.33 | política softmax (ação discreta) | — | ação contínua |
| 4.34 | gaussiana, `Σ` **predefinida** | ✅ | `test_sigma_e_covariancia_nao_desvio` |
| 4.35 | `∇_f ln f(a\|s) = Σ⁻¹(a − f(s))` | ✅ | `test_grad_log_e_sigma_inversa_vezes_o_residuo` |
| 4.36 | funcional de risco do Actor-Critic (vantagem) | ✅ | `test_vantagem_usa_o_valor_antes_do_treino_do_critico` |
| Alg. 4.4 | laço completo | 🟡 | invariantes + comportamento; **sem oráculo** |

### Capítulo 5 — resultados experimentais

| §  | afirmação | status |
|---|---|---|
| 5.2.3 | `φ = 0` → só a observação mais recente conta | ✅ `test_phi_zero_guarda_so_a_observacao_mais_recente` |
| 5.2.3 | `φ = 1` → comporta-se como a NTRN | ✅ via 3.22 |
| 5.2.2 | circular > simples em CartPole | 🟡 **replicado só em baixa resolução** — ver `reports/replica-katopodis-52.md` |
| Alg. 4.1 | SARSA de `m` passos com esquecimento | ✅ | `test_sarsa.py`, 11 testes |
| Alg. 4.2 | ε-greedy com desempate **uniforme** | ✅ | `test_desempate_entre_maximos_e_uniforme` |
| 5.2.1 | efeito conjunto de resolução e tupla | 🟡 §3.2.4 coberto; curvas não |

---

## 2. O que o nível B revelou

**Nenhuma discrepância.** As identidades 3.22 e 4.16 passaram de primeira, o
que era o desfecho esperado depois de o nível A dar desvio zero — mas não era
garantido, e agora está verificado em vez de presumido.

Três coisas que só apareceram por escrever estes testes:

1. **A diferença essencial entre as duas redes é a normalização.** (3.22) é
   média ponderada de núcleo; (4.16) é soma de funções de base. Está isolada em
   `test_as_duas_redes_diferem_exatamente_pela_normalizacao`, para que ninguém
   "simplifique" as duas para a mesma fórmula.
2. **Um termômetro de resolução `R` tem exatamente `R+1` palavras-código.** É a
   medida exata do aliasing que `R` impõe, e o teto que o teste agora afirma.
3. **O codificador circular é incompatível com a nossa convenção de padding.**
   Ele acende sempre `⌊R/2⌋` bits, logo **não existe padrão todo-zero**. As
   tarefas do artigo não têm slot vazio; a nossa tem. Documentado no módulo.

---

## 3. Fidelidade e o regime de `n` (pendência P14)

Os três testes de §3.2.4 dão conteúdo empírico à P14. Medido com `B = 96`,
120 amostras de treino, fração de consultas novas que endereçam **só** posições
nunca escritas — o "unable to generalize" do texto:

| tupla | fração sem resposta |
|---|---|
| 4 | < 0,10 |
| 8 | ↑ |
| 16 | ↑ |
| 32 | > 0,50 |

Monótono, como o texto prevê. E o caso degenerado de RAM única memoriza os 25
treinos exatamente e devolve o padrão para > 95 % do que não viu.

**A consequência para P14 é que copiar o `n` da dissertação seria erro** — mas
também que o nosso `n` não é o problema que eu supus. As dimensões de observação
estão no §5.1: CartPole 4, Lunar Lander 8, Mountain Car 2, Hopper 12 (3 juntas ×
posição e velocidade 2-D). Com `B = dim × resolução`:

| tarefa | dim | res | `B` | `n` | `n/B` | `N = B/n` |
|---|---|---|---|---|---|---|
| CartPole | 4 | 128 | 512 | 64 | 0,125 | 8 |
| Lunar Lander | 8 | 128 | 1024 | 64 | 0,062 | 16 |
| Mountain Car | 2 | 128 | 256 | 32 | 0,125 | 8 |
| Hopper | 12 | 32 | 384 | 16 | 0,042 | 24 |
| **nosso (SPEC-01)** | 25 | 8 | **200** | **12** | **0,060** | **17** |

Faixa do artigo: `n/B ∈ [0,042; 0,125]` e `N ∈ [8; 24]`. **Estamos dentro das
duas**, e quase exatamente sobre o Lunar Lander (`n/B = 0,062`, `N = 16`).

Isso **enfraquece a P14 como eu a formulei**. O `n = 12` não é obviamente
artefato do armazenamento denso: normalizado por `B`, está no mesmo regime do
artigo. O que difere é a decomposição — eles usam resolução alta (32–128) com
tupla grande; nós, resolução baixa (`T = 8`) com tupla pequena. Chegamos ao
mesmo `n/B` por outro caminho.

**A varredura de `n` continua valendo**, mas como refinamento e não como
correção de erro, e o eixo certo é `n/B`, não `n`.

---

## 4. Testes a criar nas etapas futuras

### Feitos nesta passagem

- **T-B1 — eqs. 4.17 a 4.23. ✅ Concluído.** As cinco identidades passaram,
  incluindo a positividade definida da matriz de Gram (60 × 60, autovalor
  mínimo > −1e−9). Com isso a justificativa de convergência do PRD está
  verificada de ponta a ponta: a rede **é** máquina de núcleo (4.16), o núcleo
  **é** positivo-definido (4.23), e a atualização **é** um passo de gradiente
  funcional que se propaga por ele (4.19).

- **T-B2 — §5.2.2. 🟡 Replicado parcialmente, e o resultado é desconfortável.**
  Exigiu implementar o Algoritmo 4.1 (`rl/sarsa.py`), porque é com ele que o
  capítulo 5 produziu os resultados. Em baixa resolução (`R=32, n=16`) o
  circular vence por **1,68×**, IC95% [+39,6; +56,1], em 20 de 20 sementes. **Na
  configuração declarada pelo artigo (`R=128, n=64`) dá empate** (0,96×, IC
  cruza zero). Ressalvas e interpretação em `reports/replica-katopodis-52.md`;
  a principal é que o nosso orçamento é ~1/50 do original.

### SPEC-02 — baselines e estatística

- **T-02.A — teste A/A.** Continua sendo a lacuna mais antiga do repositório.
  Todo IC já reportado pressupõe calibração nunca verificada. Não vem do
  artigo; vem de honestidade estatística.
- **T-02.B — cobertura nominal do bootstrap pareado** sob a distribuição de
  retornos real, não gaussiana.

### SPEC-06 — treino da Etapa 1

- **T-06.A — a busca de α precisa de faixa larga e várias sementes.** Já
  medido: o pior erro entre 5 sementes cresce **monotonicamente** com α (0,044
  em 0,002 contra 0,238 em 0,05) e em 0,05 uma semente diverge. Um teste de
  regressão sobre essa monotonicidade protege contra "ajustei α e melhorou" que
  na verdade é ruído de semente.
- **T-06.B — φ deve ter efeito não-monótono.** A §5.2.3 apresenta φ como
  compromisso: 0 descarta histórico, 1 ignora não-estacionariedade. Se a nossa
  varredura der monotonicidade, é sinal de que o problema não é
  suficientemente não-estacionário para o esquecimento importar — e isso é um
  resultado a reportar, não um bug.
- **T-06.C — o crítico deve estimar o valor na escala certa.** Já existe em
  miniatura (`test_o_critico_estima_o_valor_do_estado`, bandido de um passo);
  precisa da versão no ambiente real, contra retornos Monte Carlo medidos.

### SPEC-07/08 — ruído e filtro de Kalman

Estes **não têm respaldo no artigo** — são contribuição nossa. A referência de
fidelidade passa a ser a teoria de filtragem, não Katopodis:

- **T-07.A — consistência NEES/NIS por chi-quadrado.** Já previsto nas specs.
- **T-07.B — o KF degenera na identidade com ruído zero.** Caso-limite análogo
  ao `φ = 1` aqui: com `Q = R = 0` a estimativa tem de reproduzir a posição
  verdadeira, e o teste protege contra sinal trocado no ganho.
- **T-07.C — invariância do ranking de sensores.** A observação é ordenada por
  distância; com ruído pequeno o ranking deve mudar pouco. É onde o efeito do
  KF sobre a política aparece de fato.

### SPEC-09/10 — multiagente e relatório

- **T-09.A — a interseção de covariância é conservadora.** Propriedade
  matemática da CI, testável direto.
- **T-10.A — reprodutibilidade das tabelas do relatório** a partir dos
  artefatos, não de números colados à mão.

---

## 5. O que continua sem oráculo, e vai continuar

Três coisas, e o relatório final precisa declará-las em vez de deixá-las
implícitas:

1. **O laço do Algoritmo 4.4.** Validá-lo diferencialmente exigiria rodar o
   agente Julia inteiro, amarrado aos tipos de ambiente e buffer do
   `RLWorktable.jl`. Temos invariantes estruturais e dois testes de
   comportamento.
2. **A contagem de operações de T3.** Nada conta instruções executadas; a
   tabela de ciclos vem de manuais de arquitetura. A alegação "zero
   multiplicações em ponto flutuante" é verdadeira **sobre o algoritmo**.
3. **O termômetro quantil por slot.** É nosso, não do artigo, e portanto não
   tem contra o que ser conferido além das próprias propriedades. A SPEC-01
   mede que ele vence; nenhuma referência externa confirma.

---

## 6. Adendo SPEC-05 (2026-08-21) — o que os testes novos garantem, e em que nível

Três arquivos novos, e vale dizer o nível de garantia de cada um antes do
conteúdo, porque dois deles são de tipo diferente do resto deste plano.

| Arquivo | Nível | O que garante |
|---|---|---|
| `tests/test_pareamento.py` | **C** | encanamento simétrico entre os dois braços; nenhuma referência externa envolvida |
| `tests/test_barreira.py` | **B** | identidade aritmética do esquecimento, `(φ−1)(a₁−a₂)`, fechada em fórmula |
| `scripts/spec05_mountaincar.py` | **A** | oráculo **externo publicado** — os quatro números do Katopodis |

### O pareamento é nível C, e isso é uma limitação, não um detalhe

`test_pareamento.py` dirige os dois aproximadores pela **mesma fita** de
`(observação, recompensa, terminal)` e verifica que consomem as mesmas entradas na
mesma ordem, com o mesmo número de atualizações nos mesmos instantes.

O que ele **não** faz é comparar trajetórias no ambiente, e não pode: as duas
políticas emitem ações diferentes, então do segundo passo em diante veriam estados
diferentes. Exigir igualdade ali seria exigir que os dois aproximadores fossem o
mesmo — o oposto do que a comparação de H1 quer medir.

A assimetria que **permanece permitida** é o codificador: a n-tupla precisa de bits
e a MLP de floats. É a assimetria que a §4.8 declara ("os dois recebem a mesma
observação e cada um a prepara à sua maneira"), e o teste garante que ela é a
única.

Um teste merece destaque porque cobre o bug mais provável desta spec:
`test_o_ator_ve_o_estado_da_transicao_nao_o_seguinte`. Desalinhamento de um passo
entre `x` e a recompensa **não aparece em nenhuma média** e degrada os dois braços
de forma desigual — é exatamente o tipo de erro que sobreviveria a uma revisão de
código e inverteria a conclusão.

### A barreira é nível B, e o teste afirma uma desigualdade

`test_a_nao_comutatividade_e_exatamente_a_do_esquecimento` fecha a conta em
fórmula: duas escritas na mesma posição partindo de virgem dão `φ·a₁ + a₂` numa
ordem e `φ·a₂ + a₁` na outra, diferença `(φ−1)(a₁−a₂)`, que zera se `φ = 1` (e aí
recupera a eq. 3.16) ou `a₁ = a₂`.

`test_o_critico_NAO_e_invariante_a_ordem_sob_colisao` **afirma que a igualdade
falha**. É teste de sentido invertido de propósito: se ele um dia passar a falhar,
alguém mudou a semântica do esquecimento, e isso precisa ser decisão registrada e
não efeito colateral de refatoração.

### T-05.4(a) é o único nível A com oráculo *publicado* deste projeto

Os testes diferenciais contra o `Ramnet.jl` são nível A com oráculo **executável**
— rodamos o código do autor e comparamos bit a bit. O Mountain Car é diferente: o
oráculo é um **número publicado com IC**, e não há código para executar. Isso o
torna mais fraco como verificação (não localiza o erro) e mais forte como
validação (cobre o laço inteiro, que é justamente o que não tem oráculo executável
— item 1 da §5).

O controle negativo importa tanto quanto o positivo: a MLP deve ficar em ≈ 0, o
ótimo local de não mover o carro. **Uma MLP que resolva o Mountain Car é falha de
reprodução, não sucesso** — significaria que não é a MLP do autor, e a comparação
de H1 deixaria de ser a comparação dele.

### Acrescentar à §5 — uma quarta coisa sem oráculo

4. **O crédito temporal no brinquedo sequencial.** `toy/nav.py` não é resolvido por
   nenhum dos dois esquemas de política, embora a política ótima analítica faça
   7,86 passos com 100 % de sucesso. Descartados por medição: anelamento de `σ`,
   recompensa esparsa, e a maquinaria de atualização (o bandido com a **mesma**
   maquinaria aprende perfeitamente). O que resta não tem oráculo e fica como
   pendência TS5 — não bloqueia a SPEC-06, porque o portão foi decidido no bandido
   contextual, mas se o modo de falha transferir para o horizonte de 250 decisões
   do ambiente real, aparecerá lá como "não aprende" sem diagnóstico pronto.
