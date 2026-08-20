"""Codificadores **literais** da dissertação — §3.2.3, p.25-26, eqs. (3.23)/(3.24).

Katopodis (2022), `tmp/947841_compressed.pdf`. Páginas impressas; some 11 para
achar no PDF. A **fig. 3.3 (p.26)** desenha os dois esquemas com resolução de 5
bits e é a melhor referência visual para entender a diferença: o termômetro
acende um número **variável** de bits, o circular um número **fixo**, e a figura
mostra explicitamente o wrap-around do circular.

POR QUE O ESQUEMA DE CODIFICAÇÃO IMPORTA — §3.2.3, p.25

    "n-tuple networks measure the similarity between binary inputs in terms of
     the tuple distance (3.17), which could be regarded as a generalized Hamming
     distance. Therefore, in order to preserve the similarity between inputs, an
     encoding scheme should map similar inputs to binary patterns with a small
     tuple distance between them."

    O contraexemplo do texto é IEEE 754: dois floats com pequena distância de
    tupla podem representar números distantes, porque depende de onde os bits
    diferentes caem (sinal, expoente ou mantissa).

    Para binarizar vetores, "it only takes encoding its components individually
    and stacking the resulting patterns together" — é o que `codificar` faz.

O COMPROMISSO DE §3.2.4, p.27

    Resolução menor → mais aliasing → mais generalização, menos complexidade.
    "all values belonging to a given subinterval are mapped to a identical binary
    pattern. The extent of this aliasing of inputs is regulated by the encoding
    parameter R."

    Consequência exata, verificada em teste: um termômetro de resolução `R` tem
    **`R + 1` palavras-código** (`0000`, `1000`, `1100`, `1110`, `1111` para
    `R = 4`). Esse é o teto do número de padrões distintos.


Estes existem como **referência de fidelidade**, não como substitutos. O
codificador de produção do projeto é o `TermometroQuantil`, que a SPEC-01 mediu
vencendo o linear global por 4,3° contra 7,6°. O que falta é poder responder
"nosso termômetro difere do artigo em quê, exatamente?" com código em vez de
prosa — e ter o codificador circular disponível, que as tabelas do apêndice A
usam em CartPole e Hopper e que nós não tínhamos.

Duas diferenças em relação ao `TermometroGlobal`, ambas deliberadas do nosso
lado:

**Limiares.** A eq. (3.23) divide `[l, u]` em `R` subintervalos **uniformes**.
Os nossos vêm dos dados (quantis, por slot). A eq. (3.23) é o caso E0/E1.

**Comparação estrita.** A eq. (3.23) usa `x > l + (i−1)(u−l)/R`; nós usamos
`>=`. Com `i = 1` o limiar é exatamente `l`, então pelo artigo o mínimo do
intervalo codifica **todo-zero**, e pelo nosso ele acende um bit. A diferença
só aparece nos valores exatamente sobre um limiar, mas é justamente onde vive a
nossa propriedade de padding — por isso está testada, não presumida.

AVISO SOBRE O CIRCULAR E O PADDING. O codificador circular acende **sempre**
`⌊R/2⌋ bits`, por construção. Logo **não existe padrão todo-zero**, e a
convenção que dispensa o bit de validade na SPEC-01 não sobrevive a ele. As
tarefas do artigo (CartPole, Hopper) não têm slot vazio; a nossa tem. Usar o
circular na observação do Lamenza exige tratar o padding explicitamente.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class TermometroArtigo:
    """Eq. (3.23), §3.2.3 p.25: `S_i = 1` se `x > l + (i−1)(u−l)/R`, `i ∈ [1,R]`.

    Limiares uniformes e comparação **estrita**, como no texto. `l` e `u` são
    fixos e informados — o artigo os toma como o intervalo de definição da
    variável, não como estatística do corpus.
    """

    def __init__(self, minimo: float, maximo: float, resolucao: int) -> None:
        if maximo <= minimo:
            raise ValueError(f"intervalo vazio: [{minimo}, {maximo}]")
        if resolucao < 1:
            raise ValueError(f"resolucao={resolucao} deve ser >= 1")
        self.minimo = float(minimo)
        self.maximo = float(maximo)
        self.resolucao = int(resolucao)
        # i = 1..R  ->  limiar_i = l + (i-1)(u-l)/R
        self.limiares = self.minimo + np.arange(resolucao) * (
            (self.maximo - self.minimo) / resolucao
        )

    def n_bits_para(self, n_dims: int) -> int:
        return n_dims * self.resolucao

    def codificar(self, obs: NDArray) -> NDArray[np.bool_]:
        """(M, D) -> (M, D·R). Empilha as dimensões, como diz §3.2.3."""
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        bits = obs[:, :, None] > self.limiares[None, None, :]   # estrito
        return bits.reshape(obs.shape[0], -1)


class CircularArtigo:
    """Eq. (3.24), §3.2.3 p.26: padrão base com `⌊R/2⌋` bits, deslocado.

    O texto descreve o esquema como procedimento de dois passos, e não por
    equação de bit: "First, start with a base pattern of R bits, where the
    first ⌊R/2⌋ ones are activated. Next, right-shift the pattern n_shift
    units". Só o deslocamento tem fórmula fechada, a (3.24).

        n_shift = ⌊(x − l)/(u − l) · R⌋

    O deslocamento é à direita **com wrap**: "the activated bits can be
    shifted beyond the edge of the pattern, wrapping around. Besides aliasing
    inputs that fall into the same subinterval, this circular behavior favors
    a **radial similarity**, as extreme values are mapped to similar
    patterns" (§3.2.3, p.26). É o que o termômetro não dá: nele `l` e `u` são
    os padrões mais distantes possíveis.

    Medido em `tests/test_identidades_artigo.py` com `R = 16`:
    `d(0,00 · 0,99) = 2` no circular contra `16` no termômetro.

    O apêndice A usa codificação circular em CartPole e Hopper, e a simples
    em Lunar Lander e Mountain Car. A §5.2.2 (p.56) reporta que a circular
    vence nas duas tarefas testadas; a nossa replicação parcial confirma só
    em baixa resolução — ver `reports/replica-katopodis-52.md`.

    Número de bits ativos **constante**, ao contrário do termômetro. Ver o
    aviso sobre padding no docstring do módulo.
    """

    def __init__(self, minimo: float, maximo: float, resolucao: int) -> None:
        if maximo <= minimo:
            raise ValueError(f"intervalo vazio: [{minimo}, {maximo}]")
        if resolucao < 2:
            raise ValueError(f"resolucao={resolucao} deve ser >= 2")
        self.minimo = float(minimo)
        self.maximo = float(maximo)
        self.resolucao = int(resolucao)
        self.base = np.arange(resolucao) < resolucao // 2

    @property
    def bits_ativos(self) -> int:
        """`⌊R/2⌋`, invariante do esquema."""
        return self.resolucao // 2

    def n_bits_para(self, n_dims: int) -> int:
        return n_dims * self.resolucao

    def deslocamento(self, x: NDArray) -> NDArray[np.int64]:
        """`n_shift` da eq. (3.24)."""
        frac = (np.asarray(x, dtype=np.float64) - self.minimo) / (
            self.maximo - self.minimo
        )
        return np.floor(frac * self.resolucao).astype(np.int64)

    def codificar(self, obs: NDArray) -> NDArray[np.bool_]:
        """(M, D) -> (M, D·R)."""
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        s = self.deslocamento(obs)[:, :, None]                  # (M, D, 1)
        # bit na posição i = base[(i − n_shift) mod R]: deslocamento à direita
        idx = (np.arange(self.resolucao)[None, None, :] - s) % self.resolucao
        return self.base[idx].reshape(obs.shape[0], -1)
