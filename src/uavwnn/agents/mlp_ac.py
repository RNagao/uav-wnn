"""Baseline MLP Actor-Critic pareado — SPEC-05 §4.8 e T-05.3.

**O outro lado da comparação de H1.** Mesmo algoritmo (4.4), mesma política
(normal projetada), mesmo harness, mesma censura, mesmo bootstrap — trocando
**apenas o aproximador**. Se a comparação não for assim, ela mede duas coisas ao
mesmo tempo e não decide nada.

    ator     D → 32 → 32 → 3     `(u₁, u₂, μ_s)`
    crítico  D → 32 → 32 → 1     `V(s)`

Duas camadas de 32 unidades ReLU é a arquitetura dos baselines de Katopodis
(fig. 5.11), não escolha nossa. E a saída é **3**, não 2, porque a normal
projetada exige — a tabela do PRD §5.5 diz `→2` e precisa ser corrigida (+33
parâmetros, desprezível no total, mas o número publicado tem de estar certo).

RNF-11 — NUMPY PURO DOS DOIS LADOS, E POR QUÊ

Medir a latência de uma MLP `31→32→32→3` via PyTorch mede sobretudo o **despacho
do PyTorch**, não o modelo. Numa rede desse tamanho o custo por chamada é
dominado por overhead de framework: são ~2 300 multiplicações, algo como
microssegundos de aritmética real sob dezenas de microssegundos de dispatch.

Como o T3 do PRD compara *aproximadores* — e a nossa WNN é NumPy —, medir um lado
em PyTorch inverteria a conclusão por artefato de biblioteca. Daí a decisão:
**inferência E retropropagação em NumPy, sem PyTorch em nenhum caminho do
experimento.**

PyTorch aparece **só em teste**, como oráculo de equivalência do gradiente. É o
mesmo padrão dos testes diferenciais contra o `Ramnet.jl`: o oráculo valida, não
executa.

O RISCO QUE ISTO CRIA, E COMO É MITIGADO

Retropropagação escrita à mão **erra em silêncio**, e o modo de falha é perverso:
um baseline artificialmente fraco **infla H1**. Ou seja, o erro empurraria o
resultado na direção que nos favorece, que é exatamente o tipo de erro que não se
percebe.

Três guardas:

1. **Verificação por diferenças finitas** de cada parâmetro (`tests/test_mlp_ac.py`).
2. **Equivalência com PyTorch** dentro de tolerância, só em teste.
3. **O baseline tem de aprender o brinquedo** — se a MLP não resolve o bandido
   contextual, o gradiente está errado, não o problema é difícil.

ADAM, E NÃO DESCIDA SIMPLES

A SPEC-03 já pagou essa lição: com descida simples e 400 épocas a MLP de clonagem
dava 23,3°, e reportar isso teria "provado" a superioridade da WNN. Era subtreino.
Com Adam converge a 1,1°. **Dar ao baseline o melhor otimizador razoável é parte
de não construir espantalho** — e aqui o baseline é quem pode refutar H1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class Adam:
    """Adam por tensor, com os defaults canônicos.

    Escrito aqui em vez de importado porque a RNF-11 proíbe PyTorch no caminho do
    experimento, e é 15 linhas.
    """

    taxa: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m: list = field(default_factory=list, init=False)
    _v: list = field(default_factory=list, init=False)
    _t: int = field(default=0, init=False)

    def passo(self, params: list[NDArray], grads: Sequence[NDArray]) -> None:
        """Atualiza `params` **no lugar**."""
        if not self._m:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        c1 = 1.0 - self.beta1 ** self._t
        c2 = 1.0 - self.beta2 ** self._t
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            p -= self.taxa * (self._m[i] / c1) / (np.sqrt(self._v[i] / c2) + self.eps)


class MLP:
    """`entrada → oculta → oculta → saida`, ReLU nas ocultas, linear na saída.

    Inicialização He (`√(2/n_entrada)`), que é a apropriada para ReLU — Xavier
    subestimaria a escala e a rede começaria quase morta.
    """

    def __init__(self, entrada: int, saida: int, oculta: int = 32,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        dims = [entrada, oculta, oculta, saida]
        self.dims = dims
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b))
                  for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]

    # -- inferência ---------------------------------------------------------

    def prever(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """`(D,)` ou `(M, D)` → `(saida,)` ou `(M, saida)`."""
        unidim = np.asarray(x).ndim == 1
        h = np.atleast_2d(np.asarray(x, float))
        for i in range(len(self.W) - 1):
            h = np.maximum(0.0, h @ self.W[i] + self.b[i])
        out = h @ self.W[-1] + self.b[-1]
        return out[0] if unidim else out

    def _passe_frente(self, x: NDArray) -> tuple[NDArray, list[NDArray]]:
        """Guarda as ativações para a retropropagação."""
        h = np.atleast_2d(np.asarray(x, float))
        ativacoes = [h]
        for i in range(len(self.W) - 1):
            h = np.maximum(0.0, h @ self.W[i] + self.b[i])
            ativacoes.append(h)
        return h @ self.W[-1] + self.b[-1], ativacoes

    # -- retropropagação ----------------------------------------------------

    def gradientes(
        self, x: NDArray[np.float64], grad_saida: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Gradientes de `W` e `b`, dado `∂L/∂saida`.

        Recebe o gradiente **da saída**, não uma perda — é o que permite usar a
        mesma MLP como ator (onde `∂L/∂f` vem da política) e como crítico (onde
        vem da perda quadrática), sem duplicar código.

        Ordem devolvida: `[W₀, b₀, W₁, b₁, W₂, b₂]`, a mesma de `parametros()`.
        """
        saida, ativacoes = self._passe_frente(x)
        g = np.atleast_2d(np.asarray(grad_saida, float))
        n_cam = len(self.W)

        gW: list[NDArray | None] = [None] * n_cam
        gb: list[NDArray | None] = [None] * n_cam
        for i in range(n_cam - 1, -1, -1):
            gW[i] = ativacoes[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                # derivada do ReLU: passa onde a ativação foi positiva
                g = (g @ self.W[i].T) * (ativacoes[i] > 0)
        fora: list[NDArray] = []
        for i in range(n_cam):
            fora += [gW[i], gb[i]]
        return fora

    # -- parâmetros ---------------------------------------------------------

    def parametros(self) -> list[NDArray[np.float64]]:
        """Vistas mutáveis, na ordem que `gradientes` devolve. O Adam atualiza
        no lugar, então isto não pode copiar."""
        fora: list[NDArray] = []
        for W, b in zip(self.W, self.b):
            fora += [W, b]
        return fora

    def n_parametros(self) -> int:
        """Pesos **e** vieses — a conta que vai para a tabela de T1."""
        return sum(p.size for p in self.parametros())

    def bytes_fp32(self) -> int:
        return self.n_parametros() * 4

    def multiplicacoes_fp(self) -> int:
        """Por decisão, no aproximador. A comparação central de T3: a WNN faz
        **zero**."""
        return sum(a * b for a, b in zip(self.dims[:-1], self.dims[1:]))
