# T-06.0 — orçamento de simulação, `B_max`, e o veredito sobre P2

**A primeira tarefa da SPEC-06, e ela bloqueia as outras.** Fecha com quatro
resultados, e o terceiro só existe porque a medição expôs um defeito nosso.

| | resultado |
|---|---|
| estimativa do PRD §6.1 (8-30 h para 50 M passos) | **CONFIRMADA** — 16,4 h medidos |
| `B_max` | **19,5 M** passos por configuração-semente |
| quem domina a vazão | **depende da célula, e há cruzamento** |
| P2 | **não é testável como posta** — e o que é testável **falha** |

## 1. Quem domina a vazão — e a resposta não é única

A spec previa a pergunta: *"se a simulação dominar, o tempo de parede de treino é
comum aos dois lados e não sustenta a alegação 'sem GPU'"*. Medido, **há
cruzamento**:

| `Na` | ambiente | codificação | aprendizado |
|---|---|---|---|
| 1 | 41-47 % | 0,3-0,8 % | **52-58 %** |
| 2 | 34-59 % | 0,2-0,5 % | 41-66 % |
| 4 | 47-69 % | 0,2-0,4 % | 31-53 % |
| 8 | **67-79 %** | 0,1-0,2 % | 21-32 % |

(24 células: `Na ∈ {1,2,4,8} × Ns ∈ {12,24,36}`, nos dois
`algorithm_iteration_interval`.)

**Na célula base da Etapa 1 (`Na = 1`) o aprendizado domina; em `Na = 8` a simulação
domina.** A consequência é que a alegação de tempo de parede **não pode ser feita em
geral** — ela vale onde o aprendizado domina e enfraquece onde a simulação domina.
Reportar um número só seria escolher a célula conveniente.

### A codificação é 0,2 % do relógio — e isso contradiz a contagem de operações

A SPEC-05 §4.8 mediu que **o codificador domina a contagem de operações** da WNN
(≈ 800 comparações contra 576 operações de bit e 48 consultas). Aqui ele é **0,2 %
do relógio**.

As duas medições estão certas e a contradição é informativa: o termômetro é NumPy
**vetorizado** (3,8 µs para 31 dims × 32 níveis), e a consulta na tabela esparsa é
um laço Python de dicionários (~0,4 µs **por RAM**). Fatores constantes muito
diferentes para contagens parecidas.

**Reportar só uma das duas enganaria**, em qualquer direção: a contagem de operações
sustenta a alegação estrutural (zero multiplicações em ponto flutuante), o relógio
mede a nossa implementação. A §5.5 do PRD já separa T2 de T3 exatamente por isso, e
este é o primeiro caso em que a separação muda a leitura.

## 2. Um defeito nosso, que a medição expôs e que mudou o orçamento

A primeira rodada deu **455 passos/s** e `B_max = 10,5 M` — **em cima da linha de
corte** de ~10 M, o que acionaria a ordem de corte do PRD §8. O perfil mostrou por
quê:

```
enderecos()   331 µs   de 337 µs do f()      ← 98 %
custo por RAM   ~5 µs, linear de 32 a 200 RAMs
```

Linear no número de RAMs a 5 µs cada é **despacho**, não aritmética — um laço Python
com uma chamada NumPy por RAM. E a contradição com T3 denunciava: são 576 operações
de bit; a 280 ns cada não é aritmética de máquina alguma.

Corrigido por vetorização (uma matriz `(N, n)` de índices e outra de potências,
tuplas curtas apontando para um bit sentinela sempre zero):

| | antes | depois | |
|---|---|---|---|
| `enderecos()` (`N = 67`) | 331 µs | **7,9 µs** | **42×** |
| `prever()` | 337 µs | **26 µs** | **13×** |
| vazão (Na=4, Ns=12) | 455 passos/s | **848 passos/s** | 1,9× |
| `B_max` | 10,5 M | **19,5 M** | 1,9× |

**Os 45 testes diferenciais contra o `Ramnet.jl` passam sem alteração**, que é a
verificação mais forte disponível de que a vetorização preservou a semântica.

Consequência de projeto: **a correção é o que torna a campanha pagável sem cortar a
varredura.** Com 10,5 M o PRD §8 mandaria cortar `n × N` primeiro; com 19,5 M a
grade cabe.

O que sobra do `prever()` são 12 µs em 67 RAMs — o laço de dicionários, ~0,4 µs por
RAM. A variante `LUTOrdenada` da SPEC-03 (chaves ordenadas por RAM) permitiria
`np.searchsorted` vetorizado e eliminaria também isso, mas troca a estrutura de
dados. Fica registrado como caminho, não feito.

## 3. Veredito sobre a estimativa do PRD §6.1

> 0,11 s por episódio de 120 s com 4 UAVs / 12 sensores, extrapolado para **8-30 h
> num núcleo** para 50 M passos — isto é, **≈ 460-1 700 passos/s**.

**CONFIRMADA.** Medido em `Na = 4, Ns = 12` com o caminho WNN completo:
**848 passos/s**, ou **16,4 h** num núcleo para 50 M passos — dentro da faixa, e
perto do meio dela.

Vale registrar que antes da vetorização eram 455 passos/s e 30,5 h, isto é, **fora
da faixa pela borda pessimista**. A estimativa do PRD estava certa; a nossa
implementação é que não estava.

## 4. `B_max` e o plano de campanha

```
B_max = v·W/(C·R) = 848 passos/s × (12 núcleos × 72 h) / (27 configs × 5 sementes)
      = 19,5 M passos por configuração-semente
```

**Acima da linha de ~10 M**, então a ordem de corte do PRD §8 **não** é acionada e a
varredura `T × n × N` cabe inteira. A conta e o plano ficam publicados aqui, antes
de rodar, como a tarefa exige.

## 5. P2 — não é testável como posta, e é achado

A T-06.0 pede *"fidelidade de `update_rate = 0,1` contra `0,01`"*. **`update_rate`
não é parâmetro livre.** O ambiente o deriva (`environment.py:473`):

```python
update_rate = self.algorithm_iteration_interval / 2
```

Variar `update_rate` exige variar `algorithm_iteration_interval`, o que muda
**também a frequência de decisão**. As duas são inseparáveis sem editar o ambiente,
e editar o ambiente é proibido pela regra do projeto. Um teste que variasse as duas
juntas e atribuísse a diferença ao integrador de mobilidade estaria errado por
confusão de variáveis.

**O que é testável, e responde a pergunta prática:** o efeito do par
`(aii, update_rate = aii/2)` sobre o tempo de conclusão. Guloso sobre os mesmos
cenários, 40 episódios, pareado:

| `Na` | `Ns` | CT @ 0,5 | CT @ 0,1 | Δ | IC95 % | \|Δ\|/média | KS p | aceita? |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 34,2 s | 31,1 s | −3,08 | [−3,39; −2,70] | 9,0 % | 0,097 | **NÃO** |
| 8 | 36 | 47,4 s | 40,1 s | −7,27 | [−10,75; −3,80] | 15,4 % | 0,014 | **NÃO** |

**Falha nos dois**, e por margem larga: o critério pedia IC contendo zero **e**
`|Δ| < 1 %`; o IC exclui zero e o efeito é 9-15 %. No `Na = 8` o KS também rejeita
(p = 0,014).

### O que isso diz sobre a fig. 7.3 publicada

A SPEC-00 registrou que o `visualize_performance.py` avalia o **MADDPG a
`aii = 0,5`** e o **guloso a `aii = 0,1`** — cinco vezes mais reativo, variável não
controlada. Agora está quantificado: **o intervalo mais fino dá ao guloso 9-15 % de
vantagem em tempo de conclusão.**

O sinal é o que a SPEC-00 supôs (favorece o guloso, logo a vantagem do MADDPG sobre
ele é conservadora), mas a magnitude não era conhecida. Quando citarmos aqueles
números, a ressalva passa a ter tamanho.

## Reprodução

```bash
uv run python scripts/spec06_orcamento.py --episodios 3 --episodios-p2 40
```

Artefato: `reports/spec06-orcamento.json`. Hardware declarado por RNF-13: 12
núcleos, um processo, sem GPU.
