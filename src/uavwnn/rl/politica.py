"""Parametrização gaussiana da política — dissertação §4.3.3, eqs. (4.34)–(4.35).

    f(a|s) = (1/η) exp(−½ (f(s) − a)ᵀ Σ⁻¹ (f(s) − a))              (4.34)
    ∇_f ln f(a|s) = Σ⁻¹ (a − f(s)) · k(s, ·)                        (4.35)

A rede emite a **média** da distribuição; a dispersão não é aprendida. A eq.
(4.34) é explícita: *"Σ is a **predefined** covariance matrix"*, e o apêndice A
lista Σ como hiperparâmetro fixo (0,3 a 0,5 nas tarefas contínuas). É a
diferença de fundo para o `NTNModel` em PyTorch, onde o RLlib aprende `log_std`
como saída da rede.

Nas tarefas multivariadas a dissertação usa **identidade escalada** — matriz
diagonal com todos os elementos não-nulos iguais (§5.1.3). É o que
`escala_covariancia` representa, e por isso Σ⁻¹ é um escalar aqui em vez de uma
inversão de matriz.

DIVERGÊNCIA CONHECIDA (registrada como P10). O `ContinousActionPolicy.jl` do
Katopodis amostra `Σ·z + f(s)`, tratando Σ como **desvio-padrão**, mas usa
`inv(Σ)` no gradiente, tratando-o como **variância**. Se Σ é a covariância,
como diz (4.34), a amostragem correta é `√Σ·z`. Seguimos o artigo; a alternativa
fica disponível em `escala_como_desvio=True` para reproduzir o Julia.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PoliticaGaussiana:
    """Política gaussiana de covariância predefinida sobre uma rede n-tuple."""

    def __init__(
        self,
        rede,
        escala_covariancia: float,
        escala_como_desvio: bool = False,
    ) -> None:
        if escala_covariancia <= 0:
            raise ValueError(f"escala_covariancia={escala_covariancia} deve ser > 0")
        self.rede = rede
        self.escala = float(escala_covariancia)
        self.escala_como_desvio = escala_como_desvio

    @property
    def desvio(self) -> float:
        """Desvio-padrão usado na amostragem.

        Pelo artigo, Σ é covariância, então o desvio é `√Σ`. Com
        `escala_como_desvio` reproduzimos a escolha do código Julia.
        """
        return self.escala if self.escala_como_desvio else np.sqrt(self.escala)

    @property
    def inv_covariancia(self) -> float:
        """Σ⁻¹ da eq. (4.35). Escalar, porque Σ é identidade escalada."""
        return 1.0 / self.escala

    def media(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """`f(s)` — a média da gaussiana."""
        return self.rede.prever(bits)[0]

    def amostrar(
        self, bits: NDArray[np.bool_], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Ação amostrada de `N(f(s), Σ)`."""
        mu = self.media(bits)
        return mu + self.desvio * rng.standard_normal(len(mu))

    def grad_log(
        self, bits: NDArray[np.bool_], acao: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Eq. (4.35): `Σ⁻¹ (a − f(s))`.

        O fator `k(s, ·)` da equação é a função de base do núcleo — na rede
        n-tuple ele é implícito: escrever nas `N` posições endereçadas por `s`
        **é** aplicar a base. Por isso `atualizar` recebe só este vetor.
        """
        return self.inv_covariancia * (np.asarray(acao, float) - self.media(bits))
