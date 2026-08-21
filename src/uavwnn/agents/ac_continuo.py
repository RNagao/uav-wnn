"""Actor-Critic contínuo como `Policy` — SPEC-05 T-05.1, T-05.3 e §4.8.

**O pareamento de H1 mora aqui.** Um laço só — o Algoritmo 4.4 — servido por dois
aproximadores atrás da mesma interface. Trocar `NTuplaAprox` por `MlpAprox` muda
*apenas* o aproximador; algoritmo, política, censura, bootstrap e tabela são os
mesmos. Se a comparação não for assim, ela mede duas coisas ao mesmo tempo.

    NTuplaAprox   rede de gradiente funcional (§4.3.1) + NTRN com esquecimento (§4.2.1)
    MlpAprox      D → 32 → 32 → 3  e  D → 32 → 32 → 1, NumPy puro, Adam

UMA ASSIMETRIA QUE PRECISA SER DECLARADA, NÃO ESCONDIDA

Os dois têm **semântica de passo diferente**, e isso é do material de origem:

- No n-tuple, `α` é a taxa e o passo efetivo é exatamente `α·δ·∇log π` — a
  convenção de média da eq. (4.12) garante `Δf = passo`.
- Na MLP, o passo é o que o **Adam** decide a partir do gradiente. Adam
  normaliza por momento de segunda ordem, então a taxa nominal não é o
  deslocamento.

Não há como igualar as duas sem estragar uma delas. O que fazemos é **varrer a
taxa de cada uma na sua própria escala** e reportar o melhor de cada, que é o que
torna a comparação sobre o aproximador e não sobre a sintonia. Dar ao baseline um
otimizador pior seria construir espantalho — a SPEC-03 já pagou essa lição, onde
descida simples deixava a MLP em 23,3° contra 1,1° com Adam.

O CRÍTICO DOS DOIS LADOS

No n-tuple é a NTRN com esquecimento (`φ = 0,85`, o melhor nas quatro tarefas de
Katopodis), que **acumula** `G` pela eq. (4.8) e não tem taxa de aprendizado. Na
MLP é regressão para `G` por Adam. São mecanismos diferentes para o mesmo papel, e
a spec §4.5 registra a escolha: seguimos o autor em cada lado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .base import Acao, ContextoEpisodio, Observacao
from .mlp_ac import MLP, Adam
from .politica_continua import D_ATOR, NormalProjetada


class Aproximador(Protocol):
    """O que o laço precisa saber. Nada além disto."""

    def f(self, x: NDArray) -> NDArray: ...
    """Saída do ator: `(u₁, u₂, μ_s)`."""

    def v(self, x: NDArray) -> float: ...
    """Estimativa de valor do crítico."""

    def ascender_ator(self, x: NDArray, direcao: NDArray) -> None: ...
    """Sobe o ator na `direcao` dada no espaço de saída (sem taxa — cada
    aproximador aplica a sua)."""

    def regredir_critico(self, x: NDArray, alvo: float) -> None: ...
    """Aproxima o crítico de `alvo`. No n-tuple é acumulação (eq. 4.8), na MLP é
    um passo de Adam sobre perda quadrática."""

    def custo(self) -> dict[str, Any]: ...
    """Instrumentação de T1/T3 — parâmetros, bytes, multiplicações fp."""


# --------------------------------------------------------------------------


@dataclass
class MlpAprox:
    """Ator e crítico MLP, NumPy puro (RNF-11)."""

    n_bits_ou_dim: int
    oculta: int = 32
    taxa_ator: float = 3e-4
    taxa_critico: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        self.ator = MLP(self.n_bits_ou_dim, D_ATOR, self.oculta, seed=self.seed)
        self.critico = MLP(self.n_bits_ou_dim, 1, self.oculta, seed=self.seed + 1)
        self._opt_a = Adam(taxa=self.taxa_ator)
        self._opt_c = Adam(taxa=self.taxa_critico)

    def f(self, x):
        return self.ator.prever(x)

    def v(self, x) -> float:
        return float(self.critico.prever(x)[0])

    def ascender_ator(self, x, direcao) -> None:
        # Adam DESCE, então o gradiente da perda é o negativo da direção de
        # ascensão. Trocar o sinal aqui faz a política fugir do que funciona —
        # e o teste do bandido pega.
        self._opt_a.passo(self.ator.parametros(),
                          self.ator.gradientes(x, -np.atleast_2d(direcao)))

    def regredir_critico(self, x, alvo: float) -> None:
        erro = self.v(x) - float(alvo)
        self._opt_c.passo(self.critico.parametros(),
                          self.critico.gradientes(x, np.array([[erro]])))

    def custo(self) -> dict[str, Any]:
        return {
            "familia": "mlp",
            "params_ator": self.ator.n_parametros(),
            "bytes_ator_fp32": self.ator.bytes_fp32(),
            "params_critico": self.critico.n_parametros(),
            "mult_fp_ator": self.ator.multiplicacoes_fp(),
            "arquitetura": f"{self.n_bits_ou_dim}→{self.oculta}→{self.oculta}→{D_ATOR}",
        }


@dataclass
class NTuplaAprox:
    """Ator de gradiente funcional (§4.3.1) e crítico NTRN com esquecimento
    (§4.2.1) — as duas redes da dissertação, validadas contra o `Ramnet.jl`."""

    n_bits: int
    tamanho_tupla: int = 12
    phi: float = 0.85
    """`φ = 0,85` foi o melhor nas quatro tarefas do apêndice A."""
    taxa_ator: float = 2e-3
    seed: int = 0

    def __post_init__(self) -> None:
        from ..wnn import RedeGradienteFuncional, RedeRegressaoEsquecimento

        self.ator = RedeGradienteFuncional(
            self.n_bits, self.tamanho_tupla, n_saidas=D_ATOR, seed=self.seed)
        self.critico = RedeRegressaoEsquecimento(
            self.n_bits, self.tamanho_tupla, n_saidas=1, phi=self.phi,
            seed=self.seed)

    def f(self, x):
        return self.ator.prever(np.atleast_2d(x))[0]

    def v(self, x) -> float:
        return float(self.critico.prever(np.atleast_2d(x))[0, 0])

    def ascender_ator(self, x, direcao) -> None:
        # Convenção de MÉDIA (eq. 4.12): `Δf = passo` exatamente, então a taxa
        # multiplica a direção e nada mais é preciso.
        self.ator.atualizar(np.atleast_2d(x), self.taxa_ator * np.asarray(direcao))

    def regredir_critico(self, x, alvo: float) -> None:
        # ACUMULA `G` pela eq. (4.8) — não é passo de gradiente para o alvo. É a
        # média ponderada da (4.9) que converte acumulação em estimativa.
        self.critico.treinar(np.atleast_2d(x), np.array([float(alvo)]))

    def custo(self) -> dict[str, Any]:
        return {
            "familia": "n-tuple",
            "ocupadas_ator": self.ator.entradas_ocupadas(),
            "ocupadas_critico": self.critico.entradas_ocupadas(),
            "n_ram": self.ator.n_ram,
            "tamanho_tupla": self.tamanho_tupla,
            "mult_fp_ator": 0,
            "phi": self.phi,
        }


# --------------------------------------------------------------------------


@dataclass
class AtorCriticoContinuo:
    """Algoritmo 4.4 como `Policy`, para qualquer aproximador.

    Implementa a interface congelada da SPEC-02, então entra no harness ao lado
    dos baselines sem caminho especial.
    """

    aproximador: Aproximador
    codificar: Any
    """`vetor da observação -> entrada do aproximador`.

    Para a MLP é normalização afim; para a WNN é o termômetro quantil. Manter
    fora do aproximador é o que permite o pareamento — os dois recebem a **mesma
    observação** e cada um a prepara à sua maneira.
    """

    id: str = "ac_continuo"
    usa_estado_verdadeiro: bool = False
    politica: NormalProjetada = field(default_factory=NormalProjetada)
    passos: int = 8
    desconto: float = 0.99
    treinando: bool = False
    """**Avaliação é determinística** (Lamenza §7). Com `treinando = False` a
    ação é a média, sem ruído — e nenhuma atualização acontece."""

    def __post_init__(self) -> None:
        self._buffer: list[tuple[NDArray, NDArray, float]] = []
        self._rng: np.random.Generator | None = None
        self._concentracoes: list[float] = []

    # -- Policy -------------------------------------------------------------

    def reset(self, ctx: ContextoEpisodio, rng: np.random.Generator) -> None:
        self._ctx = ctx
        self._rng = rng
        self._buffer.clear()

    def act(self, obs: Observacao) -> Acao:
        x = self.codificar(obs.vetor)
        f = self.aproximador.f(x)
        self._concentracoes.append(NormalProjetada.concentracao(f))

        if not self.treinando:
            return self.politica.media(f)          # avaliação: sem ruído

        assert self._rng is not None, "reset() antes de act()"
        acao, z = self.politica.amostrar(f, self._rng)
        # Guarda `z` BRUTO, nunca a ação recortada. Ver §4.3.
        self._ultimo = (x, z)
        return acao

    def metadados(self) -> dict[str, Any]:
        return {
            "deterministica": not self.treinando,
            "sigma": self.politica.sigma_corrente,
            "passos_m": self.passos,
            "desconto": self.desconto,
            "concentracao_media": (float(np.mean(self._concentracoes))
                                   if self._concentracoes else float("nan")),
            **self.aproximador.custo(),
        }

    # -- treino: Algoritmo 4.4 ---------------------------------------------

    def registrar(self, recompensa: float) -> None:
        """Fecha a transição com a recompensa recebida. Chamar depois de `act`."""
        x, z = self._ultimo
        self._buffer.append((x, z, float(recompensa)))

    def _retorno(self, bootstrap: float) -> float:
        g = bootstrap
        for _, _, r in reversed(self._buffer):
            g = r + self.desconto * g
        return g

    def _aplicar(self, x, z, g: float) -> None:
        # δ = G − V(s), com o valor ANTES do treino do crítico (§4.3.4)
        delta = g - self.aproximador.v(x)
        f = self.aproximador.f(x)
        self.aproximador.ascender_ator(x, delta * self.politica.grad_log(f, z))
        self.aproximador.regredir_critico(x, g)
        self.politica.contar_atualizacao()

    def passo_intermediario(self, obs_atual: Observacao) -> None:
        if len(self._buffer) < self.passos:
            return
        x_atual = self.codificar(obs_atual.vetor)
        g = self._retorno(self.aproximador.v(x_atual))
        x, z, _ = self._buffer.pop(0)
        self._aplicar(x, z, g)

    def fim_de_episodio(self) -> None:
        g = 0.0
        for x, z, r in reversed(self._buffer):
            g = r + self.desconto * g
            self._aplicar(x, z, g)
        self._buffer.clear()
