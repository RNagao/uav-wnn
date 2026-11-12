"""CTDE para o Algoritmo 4.4: crítico CENTRALIZADO n-tuple, ator inalterado.

O QUE O MADDPG DE LAMENZA FAZ, e o que se espelha aqui

O `Critic` do `main.py` de referência recebe `cat(obs de TODOS os drones, ações de
TODOS os drones)` — `fc1 = Linear(obs·Na + ação·Na, 256)` — e devolve `Q(s, a)`.
Cada ator aprende por `∂Q/∂a_i`: o crítico, vendo o que os outros fizeram, atribui
a cada drone a parte da recompensa que depende da AÇÃO DELE. É treino
centralizado com execução descentralizada (CTDE): no voo só o ator existe.

O nosso Algoritmo 4.4 tem um crítico `V(x_i)` que vê só a observação do próprio
drone e recebe a recompensa do time ÷ Na. Ele não distingue "o time coletou" de
"eu coletei": o δ do drone redundante sai positivo. A campanha mediu a
consequência — os drones da WNN voam a 0,5–0,85 da separação do MADDPG, competindo
pelos mesmos sensores (`reports/spec06-vencedores.md`, "Coordenação").

O QUE ESTE MÓDULO MUDA, e o que NÃO muda

    MUDA (só no treino)                      NÃO MUDA
    ------------------------------------     ------------------------------------
    crítico: Q_c(s, a) n-tuple sobre os      ator: a MESMA rede de gradiente
    bits de [obs_1..obs_Na, ação_1..ação_Na] funcional, mesma observação, mesma
                                             política normal projetada
    δ: G − Q_c(s, a), m passos, bootstrap    deploy: só ator + termômetro; a
    em Q_c(s', a') (on-policy, SARSA-like)   campanha e a medição de RAM/latência
                                             leem exatamente o mesmo artefato
    opcional: vantagem CONTRAFACTUAL por     problema de Lamenza: observação, ação,
    drone (COMA-lite), ver abaixo            recompensa, cenário e métricas intactos

A VANTAGEM CONTRAFACTUAL, que é o análogo do `∂Q/∂a_i`

Com δ compartilhado, todos os drones recebem o mesmo sinal — o crítico central
reduz a variância (vê o que os outros fizeram) mas não dá crédito por drone. O
crédito vem do contrafactual de COMA (Foerster et al., 2018): a vantagem do drone
`i` é o retorno observado menos o que o crítico espera se SÓ o drone `i` tivesse
agido de outro jeito, segundo a própria política:

    A_i = G − (1/K) Σ_k Q_c(s, (a_{−i}, ã_i^k)),     ã_i^k ~ π_i(· | x_i)

`K` consultas extras ao crítico por drone e por passo — consultas n-tuple custam
microssegundos, então K = 4–8 é barato.

POR QUE `K = 0` NÃO PODE USAR `Q_c(s, a)` COMO BASE (a armadilha que o teste pega).
`G − Q_c(s, a)` tem média ZERO para toda ação quando o crítico acerta — a base
depende da própria ação, e o gradiente da política some à medida que o crítico
converge; sobra ruído. A base de um gradiente de política tem de ser independente
da ação do drone que a recebe: `V(s)` ou a marginal contrafactual `b_i`. Então:

    K = 0 → crítico de ESTADO conjunto V_c(s) = Q_c sobre [obs_1..obs_Na], sem
            bits de ação; δ = G − V_c(s), o mesmo para todos (variância menor que
            o descentralizado — vê o time — mas sem crédito por drone)
    K > 0 → crítico de AÇÃO conjunta Q_c(s, a) e A_i = G − b_i (não biased: b_i
            não depende de a_i; é o estimador do COMA com o retorno observado
            no lugar de Q_c(s, a), porque a rede n-tuple subajusta e devolve 0
            em endereço nunca visitado — usar Q_c(s, a) no primeiro termo
            empurraria o ator para a ação já visitada)

Em ambos o bootstrap de m passos é o crítico no (s', a') corrente, on-policy.

ESCALA DA RECOMPENSA. Mantém-se a recompensa ÷ Na do treinador (a rede do ator é
compartilhada; o sinal total por passo fica invariante a Na). O crítico central
aprende o retorno nessa mesma escala, então δ e c_max continuam comparáveis com as
corridas descentralizadas.

CODIFICAÇÃO DA AÇÃO. A ação executada `(θ/2π, s)` ∈ [0, 1]² é binarizada por um
termômetro de faixa FIXA (`TermometroArtigo(0, 1, R)`), não por quantis: a faixa
da ação é conhecida e não é estatística do corpus. Usa-se a ação EXECUTADA
(recortada), como o MADDPG, e não o `z` bruto — o crítico avalia o que aconteceu.
O termômetro é monotônico (`s > k/R`): velocidade 0 → nenhum bit, 1 → todos.

QUANTO CRÉDITO O CRÍTICO n-tuple CONSEGUE ISOLAR (medido, `tests/test_ctde.py`).
A predição da rede é a MÉDIA das RAMs (eq. 4.12); as RAMs cujas tuplas não
contêm bits da ação do drone `i` devolvem a média geral, e as que contêm só
alguns bits veem uma versão grosseira dela. Num brinquedo em que a recompensa é
exatamente a velocidade do drone 0 e o drone 1 é irrelevante, a razão
|A_1|/|A_0| fica em 0,55–0,65 com `n = 12` (0,40 com `n = 16`), estável de 1 200 a
6 000 episódios — é viés de aproximação, não ruído. Ou seja: o contrafactual
isola ~40–60 % do efeito da ação; o restante continua compartilhado. É
estritamente melhor que o `V(x_i)` descentralizado (0 %), e `dispersao_credito`
em `estatisticas()` mede isso ao vivo no treino.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from ..encoding.artigo import TermometroArtigo
from ..wnn import RedeRegressaoEsquecimento
from .politica_continua import NormalProjetada

D_ACAO = 2
"""`(θ_norm, s_norm)`, a ação executada — `Acao.como_array()`."""


class CriticoCentral:
    """`Q_c(s, a)`: regressão n-tuple com esquecimento sobre os bits conjuntos.

    `bits(xs, acoes)` monta a entrada na ORDEM dos drones (índice do agente, como
    o `cat` de Lamenza — a observação de cada um já carrega o próprio índice).
    """

    def __init__(self, n_bits_obs: int, na: int, tamanho_tupla: int = 12,
                 phi: float = 0.85, seed: int = 0, resolucao_acao: int = 16,
                 com_acoes: bool = True) -> None:
        if na < 1:
            raise ValueError(f"na={na}")
        self.na = int(na)
        self.n_bits_obs = int(n_bits_obs)
        self.com_acoes = bool(com_acoes)
        self.termo_acao = TermometroArtigo(0.0, 1.0, resolucao_acao)
        self.n_bits_acao = self.termo_acao.n_bits_para(D_ACAO) if self.com_acoes else 0
        self.n_bits = self.na * (self.n_bits_obs + self.n_bits_acao)
        self.rede = RedeRegressaoEsquecimento(self.n_bits, tamanho_tupla, n_saidas=1,
                                              phi=phi, seed=seed)
        self.tamanho_tupla, self.phi, self.seed = int(tamanho_tupla), float(phi), int(seed)

    # -- entrada -----------------------------------------------------------

    def _bits_acao(self, acoes) -> NDArray[np.bool_]:
        a = np.clip(np.asarray(acoes, dtype=float).reshape(-1, D_ACAO), 0.0, 1.0)
        return self.termo_acao.codificar(a).ravel()

    def bits(self, xs: list[NDArray[np.bool_]], acoes: list[NDArray] | None = None) -> NDArray[np.bool_]:
        """Bits conjuntos `[x_1, …, x_Na, termo(a_1), …, termo(a_Na)]` (ou só as
        observações, no crítico de estado)."""
        if len(xs) != self.na:
            raise ValueError(f"esperava {self.na} drones, veio {len(xs)} obs")
        partes = [np.asarray(x, dtype=bool).ravel() for x in xs]
        for x in partes:
            if x.size != self.n_bits_obs:
                raise ValueError(f"obs com {x.size} bits, esperado {self.n_bits_obs}")
        if self.com_acoes:
            if acoes is None or len(acoes) != self.na:
                raise ValueError(f"esperava {self.na} ações")
            partes.append(self._bits_acao(acoes))
        return np.concatenate(partes)

    def bits_substituindo(self, bits: NDArray[np.bool_], i: int, acao_i: NDArray) -> NDArray[np.bool_]:
        """Os mesmos bits com a ação do drone `i` trocada — o contrafactual."""
        if not self.com_acoes:
            raise RuntimeError("crítico de estado não tem bits de ação para substituir")
        ini = self.na * self.n_bits_obs + i * self.n_bits_acao
        novo = bits.copy()
        novo[ini:ini + self.n_bits_acao] = self._bits_acao(acao_i)
        return novo

    # -- predição e treino ---------------------------------------------------

    def q(self, bits: NDArray[np.bool_]) -> float:
        return float(self.rede.prever(np.atleast_2d(bits))[0, 0])

    def treinar(self, bits: NDArray[np.bool_], alvo: float) -> None:
        self.rede.treinar(np.atleast_2d(bits), np.array([float(alvo)]))

    def entradas_ocupadas(self) -> int:
        return int(self.rede.entradas_ocupadas())

    # -- persistência (só checkpoints de treino; o deploy não o leva) --------

    NOME = "critico-central.npz"

    def salvar(self, diretorio) -> None:
        d = Path(diretorio)
        d.mkdir(parents=True, exist_ok=True)
        self.rede.salvar(d / self.NOME)

    @classmethod
    def carregar(cls, diretorio, n_bits_obs: int, na: int, resolucao_acao: int = 16,
                 com_acoes: bool = True) -> "CriticoCentral":
        d = Path(diretorio)
        rede = RedeRegressaoEsquecimento.carregar(d / cls.NOME)
        obj = cls(n_bits_obs, na, rede.tamanho_tupla, getattr(rede, "phi", 0.85), 0,
                  resolucao_acao, com_acoes)
        if rede.n_bits != obj.n_bits:
            raise ValueError(f"crítico central gravado com {rede.n_bits} bits; esta célula pede "
                             f"{obj.n_bits} ({'com' if com_acoes else 'sem'} ações)")
        obj.rede = rede
        return obj


class TimeAtorCritico:
    """Ator compartilhado + `Na` trajetórias, com UM crítico central.

    Substitui as `Na` instâncias de `AtorCriticoContinuo` do treinador. A
    disciplina de chamada é a mesma: `agir(obs) → registrar(r) → agir(obs') …` e
    `fim_de_episodio()`. A diferença é que `agir` recebe as observações de TODOS
    os drones de uma vez, porque o crítico precisa do estado conjunto e da ação
    conjunta — inclusive para o bootstrap `Q_c(s', a')`, que só existe depois de
    `a'` ser escolhida. Por isso a atualização da transição mais antiga acontece
    DENTRO de `agir`, e não num `passo_intermediario` separado.
    """

    def __init__(self, aproximador: Any, codificar: Callable, critico: CriticoCentral,
                 na: int, sigma: float, passos: int = 8, desconto: float = 0.99,
                 contrafactual: int = 0) -> None:
        self.ap = aproximador                      # o ator vive em `ap.ator`
        self.codificar = codificar
        self.critico = critico
        self.na = int(na)
        self.politicas = [NormalProjetada(sigma, None) for _ in range(self.na)]
        self.passos, self.desconto = int(passos), float(desconto)
        self.K = int(contrafactual)
        if (self.K > 0) != critico.com_acoes:
            raise ValueError("K > 0 exige crítico COM ações (Q_c(s, a)); K = 0 exige crítico de "
                             "ESTADO (V_c(s)) — G − Q_c(s, a) não é vantagem válida (ver módulo)")
        self._rngs: list[np.random.Generator] = []
        self._soma_abs = np.zeros(self.na)     # Σ|A_i|, por drone
        self._soma_disp = 0.0                  # Σ std_i(A_i) — 0 quando o δ é compartilhado
        self._n_stats = 0
        self._buffer: list[tuple] = []           # (xs, zs, fs, bits_sa, r)
        self._pendente: tuple | None = None
        self.ultimas_vantagens: NDArray[np.float64] = np.zeros(self.na)
        self.n_atualizacoes = 0

    # -- diagnóstico do crédito (guard rail de treino) -----------------------

    def estatisticas(self) -> dict:
        """`abs_vantagem` média por drone e `dispersao_credito` =
        E[std_i(A_i)] / E[mean_i|A_i|]: 0 quando todos recebem o mesmo sinal
        (K = 0, ou crítico que não distingue a ação de cada um); cresce quando
        o crítico atribui crédito por drone. Zera ao ler."""
        n = max(1, self._n_stats)
        abs_media = self._soma_abs / n
        disp = float(self._soma_disp / n / (abs_media.mean() + 1e-12))
        self._soma_abs[:] = 0.0; self._soma_disp = 0.0; self._n_stats = 0
        return {"abs_vantagem": abs_media.tolist(), "dispersao_credito": disp}

    def reset(self, rngs: list[np.random.Generator]) -> None:
        if len(rngs) != self.na:
            raise ValueError("um gerador por drone")
        self._rngs = list(rngs)
        self._buffer.clear()
        self._pendente = None

    def agir(self, obs_por_drone: list) -> list[NDArray[np.float64]]:
        xs = [self.codificar(np.asarray(o, float)) for o in obs_por_drone]
        fs = [self.ap.f(x) for x in xs]
        zs, acoes = [], []
        for i, f in enumerate(fs):
            acao, z = self.politicas[i].amostrar(f, self._rngs[i])
            zs.append(z)
            acoes.append(acao.como_array())
        bits_sa = self.critico.bits(xs, acoes)
        # A transição mais antiga fecha agora: o bootstrap é Q_c do (s, a) corrente.
        if len(self._buffer) >= self.passos:
            self._aplicar(0, bootstrap=self.critico.q(bits_sa))
            self._buffer.pop(0)
        self._pendente = (xs, zs, fs, bits_sa)
        return acoes

    def registrar(self, recompensa: float) -> None:
        if self._pendente is None:
            raise RuntimeError("registrar() antes de agir()")
        self._buffer.append((*self._pendente, float(recompensa)))
        self._pendente = None

    def fim_de_episodio(self) -> None:
        # Sem bootstrap no fim: fecha da mais antiga para a mais nova, como o
        # `fim_de_episodio` do AtorCriticoContinuo (retorno de Monte Carlo do resto).
        while self._buffer:
            self._aplicar(0, bootstrap=0.0)
            self._buffer.pop(0)

    # -- Algoritmo 4.4 com crítico central --------------------------------

    def _retorno(self, idx: int, bootstrap: float) -> float:
        g = bootstrap
        for k in range(len(self._buffer) - 1, idx - 1, -1):
            g = self._buffer[k][4] + self.desconto * g
        return g

    def _aplicar(self, idx: int, bootstrap: float) -> None:
        xs, zs, fs, bits_sa, _ = self._buffer[idx]
        g = self._retorno(idx, bootstrap)
        # Tudo consultado ANTES de treinar o crítico neste (s, a) (§4.3.4).
        if self.K == 0:
            vantagens = np.full(self.na, g - self.critico.q(bits_sa))     # δ = G − V_c(s)
        else:
            vantagens = np.empty(self.na)
            for i in range(self.na):
                base = 0.0
                for _ in range(self.K):
                    acao_k, _z = self.politicas[i].amostrar(fs[i], self._rngs[i])
                    base += self.critico.q(self.critico.bits_substituindo(bits_sa, i, acao_k.como_array()))
                vantagens[i] = g - base / self.K                           # A_i = G − b_i
        self._soma_abs += np.abs(vantagens)
        self._soma_disp += float(np.std(vantagens))
        self._n_stats += 1
        for i in range(self.na):
            self.ap.ascender_ator(xs[i], vantagens[i] * self.politicas[i].grad_log(fs[i], zs[i]))
            self.politicas[i].contar_atualizacao()
        self.critico.treinar(bits_sa, g)
        self.ultimas_vantagens = vantagens
        self.n_atualizacoes += 1
