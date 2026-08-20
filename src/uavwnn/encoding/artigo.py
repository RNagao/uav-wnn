"""Codificadores **literais** da dissertação — §3.2.3, eqs. (3.23) e (3.24).

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
    """Eq. (3.23): `S_i = 1` se `x > l + (i−1)(u−l)/R`, para `i ∈ [1, R]`.

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
    """Eq. (3.24): padrão base com `⌊R/2⌋` bits, deslocado por `n_shift`.

        n_shift = ⌊(x − l)/(u − l) · R⌋

    O deslocamento é à direita **com wrap**, o que faz valores extremos do
    intervalo mapearem para padrões próximos — a "similaridade radial" de
    §3.2.3. É o que o termômetro não dá: nele `l` e `u` são os padrões mais
    distantes possíveis.

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
