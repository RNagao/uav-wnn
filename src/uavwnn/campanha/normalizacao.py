"""Normalização móvel da observação — o `MeanStdFilter` do RLlib, em NumPy.

POR QUE ISTO É UM EIXO DA GRADE, E NÃO UMA ESCOLHA

**Todas** as configurações do `anbubus` têm `"observation_filter": "MeanStdFilter"`,
sempre acompanhado de um termômetro de faixa **fixa** em `[-1,5; 1,5]`. Os dois
formam um par: ±1,5 só faz sentido como ±1,5 **desvios** de uma observação
padronizada.

A nossa portagem usava outro mecanismo — `TermometroQuantil` ajustado a um corpus,
sobre a observação crua. Não é erro, mas não é o pipeline deles, e isso deixou uma
ambiguidade concreta: o braço `ntupla_anbubus` **estagnou** em ~37 % de cobertura
(ganhou +0,8 pontos ao dobrar o orçamento, contra +14 do outro braço). Duas
explicações, com ações opostas:

    saturou            mostrou o que tem; mais orçamento não muda
    codificador errado a estagnação é artefato nosso, não propriedade dela

O eixo separa as duas.

A DIFERENÇA REAL NÃO É "NORMALIZAR", É **ADAPTAR-SE À DERIVA**

O termômetro quantil já se adapta à distribuição — mas **uma vez**, no corpus de
ajuste, e depois congela. O `MeanStdFilter` é estatística **móvel**: acompanha a
distribuição enquanto ela muda.

E ela muda. À medida que a política melhora, os estados que ela visita mudam de
distribuição — um codificador congelado num corpus de caminhada aleatória fica
descalibrado justamente quando o agente começa a acertar. É a mesma
não-estacionariedade que o `φ` trata do lado do crítico, agora do lado da entrada.

POR QUE O EIXO É "PIPELINE" E NÃO "NORMALIZAÇÃO ISOLADA"

Isolar seria mais limpo, e não é possível:

- um termômetro **quantil** ajustado em dados crus, aplicado a dados
  **normalizados**, teria limiares que não correspondem a nada;
- um termômetro de **faixa fixa** `[-1,5; 1,5]` aplicado a dados **crus** saturaria
  quase tudo — a observação de Lamenza vive em `[-1; 1,5]`, com `-1` de padding.

Então cada lado do eixo carrega o par coerente:

    desligado   TermometroQuantil ajustado ao corpus cru      (nosso)
    ligado      normalização móvel + termômetro de faixa fixa (deles)

A confusão entre "normalizar" e "faixa fixa vs quantil" fica **declarada** em vez de
escondida. Se o eixo mover o resultado, o próximo passo é decompô-lo.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FAIXA_PADRONIZADA = (-1.5, 1.5)
"""A faixa do termômetro que acompanha a normalização, `min`/`max` de **todas** as
configurações do `anbubus`. São ±1,5 desvios, que cobrem ~87 % da massa de uma normal
— o resto satura, e saturar a cauda é o comportamento pretendido."""


class NormalizadorMovel:
    """Média e variância móveis por dimensão, por Welford.

    Welford e não `soma/n` acumulada: com milhões de passos, a variância pela soma
    dos quadrados perde precisão catastroficamente quando a média é grande em
    relação ao desvio. É o mesmo motivo pelo qual o RLlib usa a forma incremental.

    **A estatística faz parte do modelo treinado.** Ela é atualizada durante o treino
    e precisa ser **congelada** na avaliação — senão o agente se calibra usando os
    episódios de avaliação, o que é vazamento. `congelar()` faz isso, e o
    `salvar`/`carregar` a leva no checkpoint.
    """

    def __init__(self, d: int, eps: float = 1e-8) -> None:
        self.d = d
        self.eps = eps
        self.n = 0
        self.media = np.zeros(d)
        self._m2 = np.zeros(d)
        self.congelado = False

    def atualizar(self, x: NDArray) -> None:
        if self.congelado:
            return
        x = np.asarray(x, float).ravel()
        self.n += 1
        delta = x - self.media
        self.media += delta / self.n
        self._m2 += delta * (x - self.media)

    @property
    def desvio(self) -> NDArray:
        if self.n < 2:
            return np.ones(self.d)
        return np.sqrt(np.maximum(self._m2 / (self.n - 1), self.eps))

    def aplicar(self, x: NDArray, atualizar: bool = True) -> NDArray:
        """Padroniza. `atualizar=False` na avaliação, para não vazar."""
        if atualizar:
            self.atualizar(x)
        return (np.asarray(x, float).ravel() - self.media) / self.desvio

    def congelar(self) -> "NormalizadorMovel":
        self.congelado = True
        return self

    def descongelar(self) -> "NormalizadorMovel":
        self.congelado = False
        return self

    # -- persistência: a estatística é estado treinado ----------------------

    def salvar(self, caminho) -> None:
        np.savez_compressed(caminho, media=self.media, m2=self._m2,
                            n=np.array(self.n), d=np.array(self.d))

    @classmethod
    def carregar(cls, caminho) -> "NormalizadorMovel":
        z = np.load(caminho, allow_pickle=False)
        o = cls(int(z["d"]))
        o.media, o._m2, o.n = z["media"], z["m2"], int(z["n"])
        return o.congelar()
