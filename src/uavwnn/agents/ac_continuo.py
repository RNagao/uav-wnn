"""Actor-Critic contínuo como `Policy` — SPEC-05 T-05.1, T-05.3 e §4.8.

**O pareamento de H1 mora aqui.** Um laço só — o Algoritmo 4.4 — servido por dois
aproximadores atrás da mesma interface. Trocar `NTuplaDissertacaoAprox` por `MlpAprox` muda
*apenas* o aproximador; algoritmo, política, censura, bootstrap e tabela são os
mesmos. Se a comparação não for assim, ela mede duas coisas ao mesmo tempo.

    NTuplaDissertacaoAprox   rede de gradiente funcional (§4.3.1) + NTRN com esquecimento (§4.2.1)
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

    def salvar(self, diretorio) -> None: ...
    """Grava o modelo treinado. Sem isto, uma campanha de mil runs teria de
    retreinar a cada run — o que é caro e, pior, mediria variância de treino
    misturada com variância de avaliação."""

    def custo(self) -> dict[str, Any]: ...
    """Instrumentação de T1/T3 — **vinda do harness da SPEC-03** (T-05.7).

    Os dois lados chamam `uavwnn.custo`, e não uma contagem própria. O motivo é
    concreto: duas implementações da mesma grandeza divergem em silêncio, e a
    que fica no relatório é a que alguém lembrou de atualizar. Aqui há uma só.
    """


# --------------------------------------------------------------------------


@dataclass
class MlpAprox:
    """Ator e crítico MLP, NumPy puro (RNF-11)."""

    n_bits_ou_dim: int
    oculta: int = 32
    d_saida: int = D_ATOR
    """Dimensão da saída do ator. `D_ATOR = 3` é a normal projetada de σ fixa; o
    PPO com σ aprendida pede mais (ver `rl/ppo.py`)."""
    taxa_ator: float = 3e-4
    taxa_critico: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        self.ator = MLP(self.n_bits_ou_dim, self.d_saida, self.oculta, seed=self.seed)
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

    def salvar(self, diretorio) -> None:
        from pathlib import Path
        d = Path(diretorio); d.mkdir(parents=True, exist_ok=True)
        for nome, rede in (("ator", self.ator), ("critico", self.critico)):
            np.savez_compressed(d / f"mlp-{nome}.npz",
                                **{str(i): p for i, p in enumerate(rede.parametros())})
        (d / "familia.txt").write_text("mlp")

    def carregar(self, diretorio) -> None:
        from pathlib import Path
        d = Path(diretorio)
        for nome, rede in (("ator", self.ator), ("critico", self.critico)):
            z = np.load(d / f"mlp-{nome}.npz")
            for i, p in enumerate(rede.parametros()):
                p[...] = z[str(i)]

    @property
    def camadas(self) -> list[int]:
        return [self.n_bits_ou_dim, self.oculta, self.oculta, self.d_saida]

    def custo(self) -> dict[str, Any]:
        """T1 e T3 pelo harness da SPEC-03. `custo_mlp` conta pesos **e vieses**;
        `operacoes_mlp` é reportada nas mesmas duas linhas do lado n-tupla, para
        que a comparação de `mult_fp_aproximador` seja entre iguais."""
        from ..custo import custo_mlp, operacoes_mlp

        mem = custo_mlp(self.camadas, rotulo="MLP pareada (ator)")
        so_rede = operacoes_mlp(self.camadas, incluir_amostragem=False)
        ops = operacoes_mlp(self.camadas)
        return {
            "familia": "mlp",
            "arquitetura": "→".join(str(c) for c in self.camadas),
            "params_ator": self.ator.n_parametros(),
            "params_critico": self.critico.n_parametros(),
            "bytes_persistente": mem.persistente,
            "bytes_residente": mem.residente,
            "mult_fp_aproximador": so_rede.mult_fp,
            "soma_fp_aproximador": so_rede.soma_fp,
            "mult_fp_caminho_completo": ops.mult_fp,    # o MESMO amostrador
            "detalhe_memoria": mem.detalhe,
        }


@dataclass
class _PrefixoCheckpoint:
    """Nome de arquivo do checkpoint derivado da FAMÍLIA, não fixo.

    As duas famílias n-tupla gravavam `ntupla-ator.npz` — o mesmo nome. Isso não
    causava erro porque cada corrida tem diretório próprio, mas tornava um
    checkpoint indistinguível do outro fora do contexto dele: carregar o do
    `anbubus` com a classe da dissertação leria a convenção errada (soma contra
    média), e o modelo prediria valores `N` vezes fora de escala **sem erro**.

    O `familia.txt` já registrava a procedência, mas depender de um arquivo
    auxiliar para saber o que o `.npz` é deixa a proteção fora do caminho de
    carregamento. O prefixo põe a informação no próprio nome.
    """

    def _prefixo_checkpoint(self) -> str:
        return self.custo()["familia"].replace("-", "_")


@dataclass
class NTuplaDissertacaoAprox(_PrefixoCheckpoint):
    """Ator de gradiente funcional (§4.3.1) e crítico NTRN com esquecimento
    (§4.2.1) — as duas redes da dissertação, validadas contra o `Ramnet.jl`."""

    n_bits: int
    tamanho_tupla: int = 12
    phi: float = 0.85
    """`φ = 0,85` foi o melhor nas quatro tarefas do apêndice A."""
    taxa_ator: float = 2e-3
    seed: int = 0
    d_saida: int = D_ATOR
    c_max: float | None = 0.5
    """Asserção de estabilidade da §4.2, **ligada aqui** e não na rede.

    É neste ponto que o gradiente de política entra na tabela, e é exatamente
    onde a divergência da RNF-01 foi medida (`η = 0,05` com `N = 48` divergindo
    para NaN no protótipo). O `0,5` vem do `c ≲ 0,5` de lá; a convenção de média
    da eq. (4.12) é o que torna o teto exato, porque somar `passo` às `N`
    posições endereçadas muda `f` em exatamente `passo`.

    `None` desliga, e desligar tem de ser decisão explícita de quem monta o
    agente — não o padrão."""
    variante_lut: str = "hash"
    """Estratégia de armazenamento, das três medidas na SPEC-03. `hash` é o
    padrão porque é a que o RNF-10 discute; `ordenado` mediu ~2× menos."""
    n_dims_codificador: int = 0
    limiares_por_dim: int = 0
    """Do codificador, que vive FORA do aproximador. Zero por padrão, e isso é
    uma lacuna declarada: os limiares do termômetro quantil (E2) são estado
    persistente, e omiti-los **infla a nossa vantagem**. Quem monta o agente com
    um codificador ajustado tem de preencher os dois."""

    def __post_init__(self) -> None:
        from ..wnn import RedeGradienteFuncional, RedeRegressaoEsquecimento

        self.ator = RedeGradienteFuncional(
            self.n_bits, self.tamanho_tupla, n_saidas=self.d_saida,
            seed=self.seed, c_max=self.c_max)
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

    def salvar(self, diretorio) -> None:
        from pathlib import Path
        d = Path(diretorio); d.mkdir(parents=True, exist_ok=True)
        prefixo = self._prefixo_checkpoint()
        self.ator.salvar(d / f"{prefixo}-ator.npz")
        self.critico.salvar(d / f"{prefixo}-critico.npz")
        (d / "familia.txt").write_text(self.custo()["familia"])

    def carregar(self, diretorio) -> None:
        """Substitui as redes pelas gravadas — inclusive o **mapeamento**.

        Reconstruir o mapeamento pela semente daria uma rede cujos endereços não
        correspondem aos valores gravados, e o erro seria silencioso: a política
        prediria zero em quase todo estado e pareceria "não aprendeu".
        """
        from pathlib import Path

        from ..wnn.redes import RedeGradienteFuncional
        d = Path(diretorio)
        prefixo = self._prefixo_checkpoint()
        # `carregar` devolve a CLASSE GRAVADA, então o crítico volta como
        # `RedeRegressaoEsquecimento` na família da dissertação e como
        # `RedeGradienteFuncional` na do anbubus, sem ramificação aqui.
        self.ator = RedeGradienteFuncional.carregar(d / f"{prefixo}-ator.npz")
        self.critico = RedeGradienteFuncional.carregar(d / f"{prefixo}-critico.npz")

    def custo(self) -> dict[str, Any]:
        """T1 e T3 pelo harness da SPEC-03, via a porta esparsa de T-05.7.

        A contagem sai em TRÊS linhas, e a separação não é cosmética — é o que o
        PRD §5.5 exige depois de a alegação ter sido reescrita:

        > A alegação verificável é sobre multiplicações, e sobre **o
        > aproximador** — o codificador e o amostrador entram na conta em linhas
        > próprias.

        `mult_fp_aproximador` é **zero**, e é a alegação central de H1. O caminho
        completo não é zero: o amostrador faz Box-Muller (`sqrt`, `log`, `sin`,
        `cos`) e um `atan2`, e isso é **comum aos dois braços** — atribuí-lo à
        n-tupla ou omiti-lo do lado da MLP falsearia a comparação nas duas
        direções. O codificador domina o resto, como a SPEC-01 T-01.7 previu.
        """
        from ..custo import custo_ntuple_de_rede, operacoes_ntuple

        mem = custo_ntuple_de_rede(
            self.ator, variante=self.variante_lut,
            n_dims=self.n_dims_codificador,
            limiares_por_dim=self.limiares_por_dim,
            rotulo="n-tuple dissertação (ator)")
        so_rede = operacoes_ntuple(
            self.ator.n_ram, self.tamanho_tupla, self.d_saida,
            incluir_codificador=False, incluir_amostragem=False)
        ops = operacoes_ntuple(
            self.ator.n_ram, self.tamanho_tupla, self.d_saida,
            n_dims=self.n_dims_codificador, n_niveis=self.limiares_por_dim,
            incluir_codificador=self.limiares_por_dim > 0)
        return {
            "familia": "ntupla_dissertacao",
            "n_ram": self.ator.n_ram,
            "tamanho_tupla": self.tamanho_tupla,
            "phi": self.phi,
            "variante_lut": self.variante_lut,
            "ocupadas_ator": self.ator.entradas_ocupadas(),
            "ocupadas_critico": self.critico.entradas_ocupadas(),
            "bytes_persistente": mem.persistente,
            "bytes_residente": mem.residente,
            "mult_fp_aproximador": so_rede.mult_fp,     # a alegação de H1: zero
            "soma_fp_aproximador": so_rede.soma_fp,
            "mult_fp_caminho_completo": ops.mult_fp,    # inclui o amostrador
            "comparacoes_caminho_completo": ops.comparacoes,
            "lookups_ator": ops.lookups,
            "op_bits_ator": ops.op_bits,
            "codificador_contado": self.limiares_por_dim > 0,
            "detalhe_memoria": mem.detalhe,
        }


@dataclass
class NTuplaAnbubusAprox(_PrefixoCheckpoint):
    """**A segunda família WNN** — gradiente funcional nos DOIS lados, convenção
    de soma. Portada do `NTNModel` de `anbubus/ntn_neurocomputing_v0`.

    Existe para que a comparação contra o MADDPG tenha **duas** opções de rede sem
    peso, e não uma. E não é a nossa com outra roupa: difere em três pontos
    substantivos, todos verificados no código-fonte deles.

    O QUE VEM DELES, E POR QUE CADA COISA IMPORTA

    1. **Crítico da mesma família do ator.** Eles usam `EmbeddingBag` para os dois
       (`policy_mem` e `vf_mem`), treinados por autograd. A dissertação usa modelos
       **diferentes**: gradiente funcional no ator (§4.3.1) e regressão com
       esquecimento no crítico (§4.2.1) — o que foi o achado que fechou a pendência
       **C2**. A razão da divergência é o algoritmo: o PPO deles treina ator e
       crítico por gradiente, então um modelo só serve; o Algoritmo 4.4 acumula
       retorno no crítico pela eq. (4.8), que é outra operação.

       A §4.5 desta spec já previa esta variante: *"NTRN com esquecimento como
       primária (fidelidade a D6), variante funcional atrás de uma flag, ambas
       medidas em T-05.6"*. É esta a variante.

    2. **Convenção de soma**, `mode='sum'`. Ver `RedeGradienteFuncional.convencao`.
       A consequência prática é que **as taxas deles não são transferíveis** para a
       nossa convenção de média sem dividir por `N` — `taxa_equivalente` faz a
       conversão explícita.

    3. **Alocação densa.** `num_embeddings = n_nodes · 2^n` no código deles, o que
       impõe um teto duro em `n`: com `n = 32` seriam 4,3·10⁹ posições por RAM. É
       por isso que as configurações deles usam `tuple_size = 8`. Aqui a estrutura
       é esparsa como a nossa, e a diferença de custo é **medida** pelo harness da
       SPEC-03 em vez de assumida — `variante_lut="denso"` reproduz o custo deles.

    O QUE **NÃO** VEM DELES, E É DECISÃO DECLARADA

    O `NTNModel` emite `num_outputs`, que no PPO contínuo do RLlib é
    `2 · dim(ação)` — média **e `log σ`**, isto é, **variância aprendida** (o
    trecho comentado deles inicializando o segundo bloco em `log(0,5)` confirma).
    Nós mantemos `σ` fixa, pelas quatro razões da §4.4, e **a política é a mesma
    nas duas famílias de propósito**: trocar aproximador e política ao mesmo tempo
    mediria duas coisas. A variância aprendida fica registrada como a diferença
    remanescente, não importada.
    """

    n_bits: int
    tamanho_tupla: int = 8
    """Padrão 8, que é o que as configurações do repositório deles usam — e o teto
    prático da alocação densa que eles têm."""
    taxa_ator: float = 2e-4
    d_saida: int = D_ATOR
    taxa_critico: float = 2e-4
    """Na convenção de **soma**. Divida por `N` para comparar com as taxas do
    apêndice A do Katopodis, que estão na de média."""
    seed: int = 0
    c_max: float | None = 0.5
    variante_lut: str = "hash"
    """`denso` reproduz o custo de memória da implementação deles (`n_nodes · 2^n`
    posições alocadas), que é o que limita `n` lá."""
    n_dims_codificador: int = 0
    limiares_por_dim: int = 0

    def __post_init__(self) -> None:
        from ..wnn import RedeGradienteFuncional

        comum = dict(n_bits=self.n_bits, tamanho_tupla=self.tamanho_tupla,
                     c_max=self.c_max, convencao="soma")
        # Mesmo mapeamento nos dois, como no `NTNModel`: um `randperm` só, e as
        # duas `EmbeddingBag` endereçadas pelas MESMAS chaves.
        self.ator = RedeGradienteFuncional(n_saidas=self.d_saida, seed=self.seed,
                                           **comum)
        self.critico = RedeGradienteFuncional(n_saidas=1, seed=self.seed, **comum)

    def f(self, x):
        return self.ator.prever(np.atleast_2d(x))[0]

    def v(self, x) -> float:
        return float(self.critico.prever(np.atleast_2d(x))[0, 0])

    def ascender_ator(self, x, direcao) -> None:
        self.ator.atualizar(np.atleast_2d(x), self.taxa_ator * np.asarray(direcao))

    def regredir_critico(self, x, alvo: float) -> None:
        """Passo de gradiente, **não** acumulação.

        É aqui que a diferença de família aparece na aritmética: o crítico da
        dissertação faz `v ← φ·v + G` (eq. 4.8) e não tem taxa; este faz
        `Δ = −η(v − alvo)`, que é descida de gradiente sobre erro quadrático — o
        que o autograd do PPO deles calcularia.
        """
        erro = self.v(x) - float(alvo)
        self.critico.atualizar(np.atleast_2d(x), np.array([-self.taxa_critico * erro]))

    def salvar(self, diretorio) -> None:
        from pathlib import Path
        d = Path(diretorio); d.mkdir(parents=True, exist_ok=True)
        prefixo = self._prefixo_checkpoint()
        self.ator.salvar(d / f"{prefixo}-ator.npz")
        self.critico.salvar(d / f"{prefixo}-critico.npz")
        (d / "familia.txt").write_text(self.custo()["familia"])

    def carregar(self, diretorio) -> None:
        """Substitui as redes pelas gravadas — inclusive o **mapeamento**.

        Reconstruir o mapeamento pela semente daria uma rede cujos endereços não
        correspondem aos valores gravados, e o erro seria silencioso: a política
        prediria zero em quase todo estado e pareceria "não aprendeu".
        """
        from pathlib import Path

        from ..wnn.redes import RedeGradienteFuncional
        d = Path(diretorio)
        prefixo = self._prefixo_checkpoint()
        # `carregar` devolve a CLASSE GRAVADA, então o crítico volta como
        # `RedeRegressaoEsquecimento` na família da dissertação e como
        # `RedeGradienteFuncional` na do anbubus, sem ramificação aqui.
        self.ator = RedeGradienteFuncional.carregar(d / f"{prefixo}-ator.npz")
        self.critico = RedeGradienteFuncional.carregar(d / f"{prefixo}-critico.npz")

    def custo(self) -> dict[str, Any]:
        from ..custo import custo_ntuple_de_rede, operacoes_ntuple

        mem = custo_ntuple_de_rede(
            self.ator, variante=self.variante_lut,
            n_dims=self.n_dims_codificador,
            limiares_por_dim=self.limiares_por_dim,
            rotulo="n-tuple anbubus (ator)")
        so_rede = operacoes_ntuple(self.ator.n_ram, self.tamanho_tupla, D_ATOR,
                                   incluir_codificador=False,
                                   incluir_amostragem=False)
        ops = operacoes_ntuple(
            self.ator.n_ram, self.tamanho_tupla, D_ATOR,
            n_dims=self.n_dims_codificador, n_niveis=self.limiares_por_dim,
            incluir_codificador=self.limiares_por_dim > 0)
        return {
            "familia": "ntupla_anbubus",
            "procedencia": "anbubus/ntn_neurocomputing_v0 (NTNModel)",
            "convencao": "soma",
            "critico_mesma_familia": True,
            "n_ram": self.ator.n_ram,
            "tamanho_tupla": self.tamanho_tupla,
            "variante_lut": self.variante_lut,
            "ocupadas_ator": self.ator.entradas_ocupadas(),
            "ocupadas_critico": self.critico.entradas_ocupadas(),
            "bytes_persistente": mem.persistente,
            "bytes_residente": mem.residente,
            "mult_fp_aproximador": so_rede.mult_fp,
            "soma_fp_aproximador": so_rede.soma_fp,
            "mult_fp_caminho_completo": ops.mult_fp,
            "taxa_ator_em_media": self.ator.taxa_equivalente(self.taxa_ator, "media"),
            "detalhe_memoria": mem.detalhe,
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


# --------------------------------------------------------------------------


APROXIMADORES = {
    "mlp": MlpAprox,
    "ntupla_dissertacao": NTuplaDissertacaoAprox,
    "ntupla_anbubus": NTuplaAnbubusAprox,
}
"""As opções selecionáveis por nome, para a comparação contra o MADDPG.

Existe para que escolher aproximador seja **parâmetro**, e não edição de código —
que é o que garante que os três atravessem o MESMO laço do Algoritmo 4.4, a mesma
política, a mesma censura e o mesmo bootstrap. Trocar o laço junto com o
aproximador mediria duas coisas, e é o erro que a §4.8 nomeia.

    mlp                MLP densa, NumPy puro — o controle pareado do MÉTODO
    ntupla_dissertacao a nossa: gradiente funcional (§4.3.1) + NTRN com
                       esquecimento (§4.2.1). **Duas** famílias, convenção de
                       média, fidelidade à dissertação (pendência C2)
    ntupla_anbubus   portada do `NTNModel` de `anbubus/ntn_neurocomputing_v0`:
                       gradiente funcional nos dois lados, convenção de soma

O baseline do PROBLEMA é outro e não entra aqui: o MADDPG de Lamenza, medido dos
pesos publicados na SPEC-02 (256 × 256, 287-299 KB). Estes três são variações do
MÉTODO, e a comparação contra o MADDPG usa cada um deles como um braço.
"""
