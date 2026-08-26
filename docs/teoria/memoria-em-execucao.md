# Memória em execução — o que o T1 modela e o que a RAM cobra

Investigação sobre o custo **real** da WNN carregada, contra o custo **modelado**
pelo harness da SPEC-03. A diferença é grande, e a causa não está na rede.

Complementa [`../reports/spec06-memoria-wnn.md`](../reports/spec06-memoria-wnn.md),
que trata do tamanho do artefato; aqui é a memória residente.

---

## 1. A medição que abriu a investigação

Modelo `T=32, Na=2, Ns=24` carregado (ator + crítico, 259 328 entradas):

```
T1 modelado (variante `ordenado`) :   2,98 MB
RSS medido                        :  94,59 MB
                                     ────────
                                       31,8×
```

O T1 não está errado — ele modela **os bytes de uma implementação em C**, e é a
figura certa para "quanto isto ocuparia num microcontrolador". Mas ninguém tinha
medido o que a implementação **em Python** cobra.

---

## 2. Onde os 95 MB vão

A memória é `list[dict[int, ndarray]]` — um dicionário por RAM. Custo de **uma**
entrada do ator (3 × `float64`):

| componente | bytes |
|---|---|
| dados úteis (3 × f64) | **24** |
| cabeçalho do objeto `ndarray` | **136** |
| a chave `int`, boxed | 28 |
| slot no dicionário (amortizado) | 37 |
| **total** | **201** |

**8,4× de overhead**, e nada disso é dado. O `ndarray` sozinho custa 5,7× o que
guarda: para 24 bytes de conteúdo, o NumPy carrega ponteiro de dados, forma,
passos, dtype, flags e contagem de referências.

Projetando: 259 328 × 201 B = **52 MB**. Medi 95 MB — o resto é fragmentação do
alocador e folga de crescimento dos dicionários.

> **A rede não é o problema.** 3,1 MB de dados carregam 95 MB de RAM por causa da
> granularidade: 259 mil objetos Python minúsculos, cada um com seu cabeçalho.

---

## 3. A alternativa: LUT plana

É a variante `ordenado` do T1, implementada de verdade — três arrays contíguos em
vez de 72 dicionários:

```python
chaves   int32   (occ,)      # chave GLOBAL = ram·2ⁿ + endereço, ORDENADA
valores  float32 (occ, d)
desloc   int32   (N,)        # ram·2ⁿ, pré-computado
```

A chave global é o truque: em vez de 72 tabelas separadas, **uma** tabela
ordenada. Com `N = 72` e `n = 12`, a chave máxima é `72 · 4096 = 294 912` — cabe
em `int32` com folga.

A consulta vira **uma chamada vetorizada**:

```python
g  = self.desloc + end            # as N chaves globais de uma vez
j  = np.searchsorted(self.chaves, g)      # UMA busca binária vetorizada
ok = self.chaves[j] == g                  # quais existem
return self.valores[j[ok]].sum(axis=0) / self.n_ram
```

Sem laço Python. O `searchsorted` do NumPy faz as 72 buscas em código compilado.

---

## 4. Medido — e ganha nos dois eixos

Ator de `T=32, Na=2, Ns=24`, 129 664 entradas:

| variante | bytes | erro rel. | latência |
|---|---|---|---|
| `dict[int, ndarray]` (hoje) | **25 452 KB** | — | **93,0 µs** |
| plana `int64`/`f32` | 2 533 KB | 4,9e-08 | 11,7 µs |
| **plana `int32`/`f32`** | **2 026 KB** | 4,9e-08 | **12,0 µs** |
| plana `int32`/`f64` | 3 546 KB | **0** | 21,4 µs |

**12,6× menos memória e 7,8× mais rápida** na variante recomendada.

O ganho de velocidade é contraintuitivo — busca binária é `O(log n)` e o dict é
`O(1)`. Mas o dict paga **por RAM**: 72 buscas em Python, cada uma com boxing da
chave, hash e desreferência de objeto. A plana faz uma chamada compilada sobre
memória contígua, e a localidade de cache faz o resto.

### O erro de 4,9e-08 é o `float32`

A variante `int32/f64` dá **erro zero** por 1,5 KB a mais por milhar de entradas.
Como já verifiquei que o `float32` preserva SR e CT idênticos (`models/deploy/`),
a escolha é livre — mas quem quiser bit-a-bit tem a opção.

---

## 5. O que isto NÃO resolve

**O T1 publicado não muda.** Ele já reporta a variante `ordenado`, que é
exatamente esta estrutura: 1 778 KB modelados contra 2 026 KB medidos — a
diferença são as chaves em `int32` em vez do `uint16` que o modelo assume, e o
vetor de deslocamentos.

**A desvantagem contra o MADDPG continua.** O ator dele são 287 KB de pesos
densos; o nosso são ~2 MB de tabela esparsa. A razão estrutural — a WNN guarda
5,3× mais números — não muda com layout.

**O treino não pode usar isto.** A LUT plana é **imutável**: inserir uma chave
nova exigiria reordenar o array. O `dict` existe porque o treino escreve em
posições novas o tempo todo. Esta estrutura serve para **inferência**, que é
onde o `models/deploy/` vive.

---

## 6. Outras avenidas, medidas ou descartadas

| ideia | veredito |
|---|---|
| `array` da stdlib em vez de `ndarray` | não resolve — o overhead é do objeto, e `array` também é um |
| `__slots__` numa classe de entrada | pioraria: acrescenta um objeto por entrada |
| `gc.freeze()` após carregar | ajuda o *fork* (as páginas ficam limpas), não o RSS deste processo |
| desligar o GC durante o carregamento | acelera o `carregar`, não muda o pico |
| `numpy.memmap` para as LUTs | viável, e transformaria RSS em cache de página — só faz sentido se a memória for o gargalo real |
| poda pós-treino | **já aplicada** — corta 70 % das entradas, e é ortogonal a isto |

A poda e a LUT plana **se compõem**: 70 % menos entradas × 12,6× menos bytes por
entrada.

---

## 7. Recomendação

Aplicar a LUT plana **só no caminho de inferência**, sem tocar no treino:

* `models/deploy/` já é imutável por construção — nada escreve nele;
* a campanha carrega esses artefatos e roda milhões de episódios;
* o ganho é duplo, e a equivalência numérica está medida.

Não recomendo trocar a estrutura do treino. O `dict` é a escolha certa para
escrita esparsa e imprevisível, e a memória do treino é artefato de trabalho —
não é o que a tese reporta.

---

## 8. Reprodução

```python
# RSS real de um modelo carregado
def rss():
    return int([l for l in open('/proc/self/status')
                if l.startswith('VmRSS')][0].split()[1]) / 1024
```

Os protótipos e as medições desta investigação estão no histórico da sessão; a
estrutura da §3 é autocontida e cabe em 20 linhas.
