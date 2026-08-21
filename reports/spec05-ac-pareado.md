# SPEC-05 — Actor-Critic contínuo pareado

Reprodução: `uv run pytest tests/test_ac_continuo.py -q` (25 testes, ~5 s).

---

## 1. O pareamento de H1

Um laço só — o Algoritmo 4.4 — servido por dois aproximadores atrás da mesma
interface. Trocar `NTuplaAprox` por `MlpAprox` muda **apenas o aproximador**:
algoritmo, política, censura, bootstrap e tabela são os mesmos.

| | ator | crítico |
|---|---|---|
| **n-tuple** | rede de gradiente funcional (§4.3.1) | NTRN com esquecimento, `φ = 0,85` (§4.2.1) |
| **MLP** | `D → 32 → 32 → 3` | `D → 32 → 32 → 1` |

Os dois implementam a interface `Policy` congelada na SPEC-02, então entram no
harness ao lado dos baselines sem caminho especial. Há teste de que **a mesma
observação chega aos dois**, na mesma ordem — é o que torna a comparação sobre o
aproximador e não sobre a entrada.

## 2. A normal projetada resolve a C1

A ação de Lamenza é `⟨θ, s⟩` com `θ ∈ [0, 2π)`, e a gaussiana isotrópica de
Katopodis (eq. 4.34) não trata wrap-around. O ator emite `d = 3`,
`f = (u₁, u₂, μ_s)`, amostramos em **R³** e derivamos:

```
θ = atan2(z₂, z₁)      s = clip(z₃, 0, 1)
```

**O gradiente vive em R³, onde não existe wrap.** `∇_f log π = (z − f)/σ²` é a
pontuação gaussiana comum; a topologia circular fica inteiramente no `atan2`, que
é liso e sobrejetivo no círculo. Nenhuma correção de fase, nenhum caso especial
em `θ = 0`.

Verificado em teste: `z₂ = −10⁻⁶` dá `θ_norm = 0,99999`, e `z₂ = +10⁻⁶` dá
`0,00000` — opostos na coordenada, **vizinhos no círculo**.

Ganho de graça: `‖f₁:₂‖` controla a concentração do rumo, o análogo do parâmetro
da von Mises. Não é hiperparâmetro, e é o **diagnóstico obrigatório** da §4.3 —
distingue "não aprendeu" de "colapsou para determinístico cedo demais", dois
problemas com correções opostas.

**Consequência a registrar:** a MLP pareada é `D → 32 → 32 → 3`, e não `→ 2` como
diz a tabela do PRD §5.5. São +33 parâmetros — desprezível no total, mas o número
publicado tem de estar certo.

## 3. Amostrado ≠ executado

A regra que evita o bug mais provável da spec:

> A atualização usa **`z` bruto**. O ambiente recebe `(θ, s)` derivado e
> **recortado**. Nunca o contrário.

Recortar `z₃` e usar o valor recortado no resíduo viesa o gradiente e mata o
sinal justamente onde a política pede velocidade máxima — a região onde ela vai
passar a maior parte do tempo. Com `f₃ = 0,9`, `σ = 0,1` e `z₃ = 1,8`:

| caminho | `∇₃` |
|---|---|
| **certo** (bruto) | `(1,8 − 0,9)/0,01 = 90` |
| errado (recortado) | `(1,0 − 0,9)/0,01 = 10` |

**9× de diferença**, sempre subestimando. `amostrar` devolve os dois
explicitamente para que ninguém tenha como confundi-los.

## 4. `σ_s` fixa e anelada, e por que os valores do artigo não servem

Katopodis usa `σ² = 0,5` no Mountain Car, cuja ação vive em `[−1, 1]` — ou seja
`σ ≈ 0,71`, uns 35 % da faixa. **A nossa ação normalizada vive em `[0, 1]`**: o
mesmo `σ` produziria maioria de amostras recortadas e uma política efetivamente
de dois níveis, parado ou máximo.

**A grandeza transferível é `σ/faixa`, não `σ`.** Faixa adotada: `σ_s` de 0,25
anelado a 0,05 no brinquedo; a varredura da Etapa 1 usa `{0,05; 0,10; 0,20}` da
faixa.

Avaliação é determinística — protocolo de Lamenza §7, não escolha nossa. Há teste
de que `treinando = False` não atualiza parâmetro nenhum.

## 5. RNF-11: NumPy puro dos dois lados

Medir a latência de uma MLP `31→32→32→3` via PyTorch mede sobretudo o **despacho
do PyTorch**: são ~2 100 multiplicações, microssegundos de aritmética real sob
dezenas de microssegundos de dispatch. Como o T3 compara *aproximadores* e a
nossa WNN é NumPy, medir um lado em PyTorch inverteria a conclusão por artefato
de biblioteca.

Então inferência **e** retropropagação em NumPy, com Adam escrito à mão. Há teste
que lê o próprio fonte e falha se `torch` aparecer.

### O risco que isso cria, e as três guardas

Retropropagação à mão **erra em silêncio**, e o modo de falha é perverso: um
baseline artificialmente fraco **infla H1** — o erro empurra o resultado na
direção que nos favorece.

1. **Diferenças finitas** em cada parâmetro: pior erro relativo **7 × 10⁻¹⁰**.
2. **Máscara do ReLU**: teste com camada morta, que pega o erro clássico de usar
   a pré-ativação em vez da ativação.
3. **O baseline tem de aprender o brinquedo** — se a MLP não resolve o bandido, o
   gradiente está errado, não o problema é difícil.

## 6. Achado: a taxa do baseline precisa de varredura de verdade

Meu primeiro chute para a MLP foi `3e-3`, e o resultado parecia limpo:

| aproximador | rumo | veloc. |
|---|---|---|
| MLP (taxa 3e-3) | 4,8° | 0,136 |
| n-tuple (taxa 8e-3) | 0,9° | 0,009 |

Leitura tentadora: a WNN é 5× melhor. **Errada.** Varrendo as duas taxas:

| MLP | rumo | | n-tuple | rumo |
|---|---|---|---|---|
| 3e-5 | 15,6° | | 1e-3 | **0,4°** |
| **1e-4** | **0,8°** | | 3e-3 | 0,5° |
| 3e-4 | 1,4° | | 8e-3 | 0,8° |
| 1e-3 | 2,8° | | 2e-2 | 1,4° |
| 1e-2 | 53,1° | | 5e-2 | 4,7° |

**A taxa ótima da MLP é `1e-4` — o meu chute era 30× alto.** No ótimo de cada
lado o resultado é praticamente empate: 0,8° / 0,006 da MLP contra 0,4° / 0,012 do
n-tuple. O n-tuple é 2× melhor no rumo; a MLP é 2× melhor na velocidade.

**A curva da MLP é não-monótona com ótimo agudo** — 15,6° em 3e-5, 0,8° em 1e-4,
53° em 1e-2. Um palpite de ponto único erra em qualquer lugar dela.

É a lição da SPEC-03 repetindo: lá, descida simples deixava a MLP de clonagem em
23,3° contra 1,1° com Adam. **Dar ao baseline a sintonia certa é parte de não
construir espantalho**, e aqui o baseline é justamente quem pode refutar H1.

### Uma assimetria declarada, não escondida

Os dois têm semântica de passo diferente, e isso é do material de origem: no
n-tuple `α` é o deslocamento efetivo (convenção de média da eq. 4.12); na MLP o
passo é o que o Adam decide a partir do gradiente. Não há como igualar as duas
sem estragar uma. **Varremos a taxa de cada uma na sua própria escala e
reportamos o melhor de cada** — é o que torna a comparação sobre o aproximador.

## 7. Ressalva sobre este resultado

**O brinquedo tem 4 contextos discretos**, o que favorece a natureza quase tabular
do n-tuple. Nada aqui é claim sobre o problema real — é validação de que os dois
laços funcionam e que a comparação é justa. O veredito de H1 sai na SPEC-06, no
ambiente de referência, com IC pareado.

## 8. O que falta na SPEC-05

- **T-05.4** — validação nos benchmarks do apêndice A (Mountain Car, Hopper) com
  os hiperparâmetros verbatim. O brinquedo cobre o laço; os benchmarks cobrem a
  comparação com números publicados.
- **T-05.6** — varredura mínima no brinquedo já feita para a taxa; falta `σ` e `φ`.
- **T-05.7** — pontos de instrumentação para T1/T2/T3 já existem em
  `Aproximador.custo()`; falta ligar ao harness de custo da SPEC-03.
