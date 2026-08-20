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
| 4.17–4.25 | conexão com descida de gradiente funcional implícita | ⬜ | ver §4, item **T-B1** |
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
| 5.2.2 | circular > simples em CartPole e Lunar Lander | ⬜ **T-B2** |
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

## 3. Fidelidade e o regime de `n` (pendência P11)

Os três testes de §3.2.4 dão conteúdo empírico à P11. Medido com `B = 96`,
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

**A consequência para P11 é que copiar o `n` da dissertação seria erro** — mas
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

Isso **enfraquece a P11 como eu a formulei**. O `n = 12` não é obviamente
artefato do armazenamento denso: normalizado por `B`, está no mesmo regime do
artigo. O que difere é a decomposição — eles usam resolução alta (32–128) com
tupla grande; nós, resolução baixa (`T = 8`) com tupla pequena. Chegamos ao
mesmo `n/B` por outro caminho.

**A varredura de `n` continua valendo**, mas como refinamento e não como
correção de erro, e o eixo certo é `n/B`, não `n`.

---

## 4. Testes a criar nas etapas futuras

### Ainda sobre o artigo (podem ser feitos já)

- **T-B1 — eqs. 4.17–4.25, descida de gradiente funcional implícita.**
  O texto prova que a atualização (4.11) *é* um passo de gradiente funcional
  numa máquina de núcleo. Testável comparando a nossa atualização com o passo
  calculado explicitamente sobre a base de núcleo. Fecha a última lacuna
  teórica do capítulo 4. *Custo: baixo. Valor: fecha a justificativa de
  convergência do PRD por inteiro.*

- **T-B2 — §5.2.2, circular vence o termômetro simples.**
  Agora que `CircularArtigo` existe, é reproduzível: mesma tarefa, mesmos
  hiperparâmetros, só o codificador muda. **Cuidado:** o resultado do artigo é
  em CartPole/Lunar Lander, não no nosso problema, e a nossa observação tem
  padding que o circular não trata. O teste vale como **replicação da
  dissertação**, não como escolha de projeto — o nosso codificador segue sendo
  o quantil por slot da SPEC-01.

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
