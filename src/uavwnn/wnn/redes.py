"""As duas redes n-tuple da dissertação de Katopodis (2022).

FONTE PRIMÁRIA
    Rafael Fontella Katopodis. *Reinforcement Learning with Weightless Neural
    Networks*. Dissertação de mestrado, PESC/COPPE-UFRJ, janeiro de 2022, 87 p.
    Arquivo local: `tmp/947841_compressed.pdf`.

    Números de página abaixo são os **impressos** na dissertação. Para achar a
    página no PDF, some 11 (a numeração impressa começa depois da capa e do
    material pré-textual). Ex.: Algoritmo 4.4 está na p. 48 impressa = p. 59 do
    PDF.

MAPA DAS SEÇÕES QUE ESTE MÓDULO IMPLEMENTA

    §3.2.1  p.22  Redes discriminadoras; fig. 3.1 (neurônio RAM de 1 bit),
                  fig. 3.2 (multidiscriminador / WiSARD)
    §3.2.2  p.23  A rede de regressão n-tuple (NTRN) — eqs. (3.15) a (3.22)
    §3.2.4  p.27  Resolução, tamanho de tupla e o compromisso
                  complexidade/generalização
    §4.2.1  p.34  Rede de regressão **com esquecimento** — eqs. (4.8) a (4.10);
                  fig. 4.2 (p.35) ilustra a estimativa de densidade ponderada
    §4.2.2  p.35  SARSA de m passos; fig. 4.3 (p.36) mostra as DUAS formas de
                  modelar saída vetorial
    §4.3.1  p.38  Rede inspirada em **gradiente funcional** — eqs. (4.11) a
                  (4.25)

DUAS REDES, NÃO UMA — e é isto que fecha a pendência C2

    O PRD pedia "gradiente funcional **com** esquecimento". Essa combinação não
    existe na dissertação: são modelos distintos, para papéis distintos no
    Algoritmo 4.4 (p. 48).

    | papel   | seção  | classe aqui                  | guarda           | φ   |
    |---------|--------|------------------------------|------------------|-----|
    | ator    | §4.3.1 | `RedeGradienteFuncional`     | só valor         | não |
    | crítico | §4.2.1 | `RedeRegressaoEsquecimento`   | valor + contador | sim |

    O porquê está no laço do Algoritmo 4.4: o crítico recebe o retorno `G`,
    cujo alvo se desloca conforme a política melhora. É a distribuição mutável
    que o fator de esquecimento existe para acompanhar (§4.1.1, p.31, "The
    assumption of a fixed joint distribution"). O ator não tem alvo se movendo
    assim.

ARMAZENAMENTO ESPARSO — não é preferência de estilo

    O apêndice A usa tuplas de **16, 32 e 64 bits**. `2^64` posições por RAM não
    cabem em array denso. O `NTupleRegressor` de `ntuple.py` é denso e serve à
    SPEC-01/SPEC-03, mas fica preso a `n ≲ 16`; estas classes não.

    O `Ramnet.jl` do autor usa `Dict{UInt, ...}` por nó, com o mesmo teto de 64
    (`max_tuple_size = (UInt == UInt32) ? 32 : 64`).

MAPEAMENTO POR PARTIÇÃO

    `randperm(B)` fatiado em `⌈B/n⌉` tuplas — o `:uniform_random` do
    `Ramnet.jl`. Cada bit é lido **exatamente uma vez**, então a cobertura é 1,0
    por construção (fecha o nosso achado C4). Quando `B` não é múltiplo de `n` a
    última tupla é mais curta, e isso é fiel: o Julia usa `Iterators.partition`,
    que também devolve resto.

VALIDAÇÃO

    `tests/test_redes.py`         — desvio **zero** contra `Ramnet.jl` em 5
                                    configurações (dois papéis, φ ∈ {0,5; 0,85; 1,0})
    `tests/test_identidades_artigo.py` — as identidades do texto: (3.17)-(3.22),
                                    (4.13), (4.16)-(4.23), §3.2.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MAX_TUPLA = 64
"""Teto do tamanho de tupla.

Mesmo do `Ramnet.jl` (`max_tuple_size = (UInt == UInt32) ? 32 : 64`). O limite é
de representação: endereços vivem em `uint64`, então `n > 64` não cabe. As
tabelas do apêndice A vão até 64 exatamente por isso.
"""


def mapeamento_particionado(
    n_bits: int, tamanho_tupla: int, seed: int = 0
) -> list[NDArray[np.int64]]:
    """Partição dos `B` bits em `⌈B/n⌉` tuplas.

    Implementa o esquema `:uniform_random` do `Ramnet.jl`
    (`Partitioners/RandomPartitioner.jl`): sorteia uma permutação de `1..B` e a
    fatia em blocos de `n`. A dissertação descreve o sorteio das tuplas em
    §3.2.1 (p.22) sem nomear o esquema; o código do autor fixa este.

    **Cada bit é lido exatamente uma vez.** É o que dá cobertura 1,0 e o que
    torna `N = ⌈B/n⌉` determinado em vez de hiperparâmetro livre — diferente da
    amostragem por RAM do `NTupleRegressor`, que pode deixar bits sem leitura.

    Devolve **lista**, não matriz, porque a última tupla pode ser mais curta
    (mesma semântica do `Iterators.partition` do Julia). O `NTNModel` em PyTorch
    do repositório `anbubus` erra exatamente aqui: conta os nós com `ceil` mas a
    `conv1d` de endereçamento produz `floor`, e as duas discordam quando `B` não
    é múltiplo de `n`.
    """
    if tamanho_tupla < 1:
        raise ValueError(f"tamanho_tupla={tamanho_tupla} deve ser >= 1")
    if tamanho_tupla > MAX_TUPLA:
        raise ValueError(f"tamanho_tupla={tamanho_tupla} excede {MAX_TUPLA}")
    if tamanho_tupla > n_bits:
        raise ValueError(f"tamanho_tupla={tamanho_tupla} maior que n_bits={n_bits}")

    ordem = np.random.default_rng(seed).permutation(n_bits)
    return [ordem[i:i + tamanho_tupla] for i in range(0, n_bits, tamanho_tupla)]


class TreinoInstavel(RuntimeError):
    """Divergência detectada durante o treino — asserção obrigatória da §4.2.

    A RNF-01 do PRD nasceu de uma **medição**, não de teoria: no protótipo, `η =
    0,05` com `N = 48` divergiu para `NaN`. A spec §4.2 registra que *"o modo de
    falha medido no protótipo foi divergência para NaN; ele precisa de detecção,
    não de fé"* — esta exceção é essa detecção.

    A mensagem carrega o número do passo e o valor ofensor, porque uma divergência
    silenciosa numa varredura de hiperparâmetros vira um ponto ruim indistinguível
    de um ponto legitimamente ruim. Abortar é informação; `NaN` propagado não é.
    """


class _RedeNTupla:
    """Endereçamento comum às duas redes — §3.2.1 (p.22), fig. 3.1 e 3.2.

    Um discriminador é um conjunto de `N` neurônios RAM. Cada neurônio `i` lê um
    subconjunto fixo de `n` bits da entrada binária e usa esses bits como
    **endereço** numa tabela própria. Na notação da dissertação, `τ[i](x)` é "the
    address that `x` indexes in neuron `i`" (§3.2.2, p.23).

    A composição do endereço segue a convenção do `encode` do `Ramnet.jl`
    (`Encoders/RawBitsEncoder.jl`): `value += bit << (i-1)`, isto é, o `k`-ésimo
    bit da tupla pesa `2^k`.
    """

    def __init__(self, n_bits: int, tamanho_tupla: int, n_saidas: int = 1,
                 seed: int = 0, c_max: float | None = None) -> None:
        self.n_bits = n_bits
        self.c_max = c_max
        """Teto do passo em espaço de saída — asserção da §4.2, `None` desliga.

        PADRÃO `None`, E A RAZÃO NÃO É LENIÊNCIA. Esta classe é primitiva de uso
        geral, e serve de **oráculo numérico**: os testes diferenciais alimentam
        gradientes arbitrários para casar bit a bit com o `Ramnet.jl`, e os de
        identidade usam passo unitário para sondar a estrutura do núcleo (eq.
        4.16, 4.22). Ligar o teto aqui aborta 13 desses testes por sondas
        deliberadas de magnitude 1,0 — e a validação contra o Julia é a evidência
        mais forte que o projeto tem. Uma primitiva não policia quem a chama.

        Quem liga é `NTuplaAprox`, o ponto onde o gradiente de política de fato
        entra, que é exatamente onde a divergência da RNF-01 foi MEDIDA. Um teste
        garante que ele não seja desligado por descuido.

        A convenção de **média** da eq. (4.12) é o que torna este teto exato e não
        heurístico: somar `passo` às `N` posições endereçadas muda `f` em
        **exatamente** `passo`, porque `f = (1/N) Σ v[i]` e há `N` termos. Então
        `max|passo| ≤ c_max` É o `η·max|∂c/∂f| ≤ c_max` da spec, sem fator de `N`
        escondido. Na convenção de soma o mesmo teto valeria `c_max/N` e mudaria
        a cada ponto da varredura de `N` — o motivo (ii) da decisão da §4.2.

        `0,5` vem do `c ≲ 0,5` da RNF-01.
        """
        self._n_passos = 0
        self.tamanho_tupla = tamanho_tupla
        self.n_saidas = n_saidas
        self.mapeamento = mapeamento_particionado(n_bits, tamanho_tupla, seed)
        self.n_ram = len(self.mapeamento)
        # Potências de 2 por tupla. Mantida como lista para quem inspeciona
        # tupla a tupla; o caminho quente não a usa.
        self._potencias = [
            (np.uint64(1) << np.arange(len(t), dtype=np.uint64))
            for t in self.mapeamento
        ]
        self._montar_endereco_vetorizado()

    def _montar_endereco_vetorizado(self) -> None:
        """Pré-computa as duas matrizes que tornam `enderecos` uma chamada só.

        POR QUE ISTO EXISTE — medido em T-06.0, e o número é embaraçoso.

        A versão anterior fazia um laço Python sobre as RAMs, com uma chamada
        NumPy por RAM. O custo saiu **linear em `n_ram` a ~5 µs por RAM**, medido de
        32 a 200 RAMs — isto é, despacho, não aritmética. Com `N = 67` isso dava
        **331 µs dos 337 µs** de `prever()`, contra 8,6 µs da MLP pareada: o
        n-tuple parecia **39× mais lento**, exatamente o oposto da alegação de H1.

        E a contradição com a contagem de operações (T3) denunciava o problema: são
        576 operações de bit no endereçamento, que a 280 ns cada não são
        aritmética de máquina alguma.

        As tuplas curtas (a última pode ser) são preenchidas apontando para um bit
        **sentinela** acrescentado à entrada, sempre zero, com potência zero — o
        que o torna neutro na soma. Assim a matriz é retangular e o endereço sai de
        um `gather` mais um produto.
        """
        n_max = max(len(t) for t in self.mapeamento)
        # Coluna sentinela: índice `n_bits`, acrescentada em `enderecos`.
        self._tuplas = np.full((self.n_ram, n_max), self.n_bits, dtype=np.intp)
        self._pot_mat = np.zeros((self.n_ram, n_max), dtype=np.uint64)
        for i, tupla in enumerate(self.mapeamento):
            k = len(tupla)
            self._tuplas[i, :k] = tupla
            self._pot_mat[i, :k] = np.uint64(1) << np.arange(k, dtype=np.uint64)

    @property
    def cobertura(self) -> float:
        """Fração dos bits de entrada que alguma RAM lê.

        Sempre 1,0 aqui, por a partição ser exaustiva. Existe como propriedade
        porque o achado C4 do projeto era justamente que a amostragem por RAM do
        `NTupleRegressor` podia deixar bits sem leitura, e o valor precisa ser
        inspecionável nos dois casos.
        """
        return len(np.unique(np.concatenate(self.mapeamento))) / self.n_bits

    def enderecos(self, bits: NDArray[np.bool_]) -> NDArray[np.uint64]:
        """`τ[i](x)` para cada RAM — §3.2.2 (p.23). `(M, B) -> (M, N)`.

        `uint64` e não `int64` porque com `n = 64` o bit mais significativo pesa
        `2^63`, que estoura o inteiro com sinal. É o mesmo motivo de o
        `Ramnet.jl` usar `UInt` nas chaves dos dicionários.
        """
        if bits.ndim == 1:
            bits = bits[None, :]
        # Bit sentinela em zero, para as tuplas curtas apontarem sem efeito.
        estendido = np.zeros((bits.shape[0], self.n_bits + 1), dtype=np.uint64)
        estendido[:, :self.n_bits] = bits
        # Uma chamada, não `n_ram` chamadas: `(M, N, n) · (N, n) -> (M, N)`.
        return (estendido[:, self._tuplas] * self._pot_mat).sum(axis=2)

    @property
    def ganho_do_passo(self) -> float:
        """Quanto `f` muda ao somar `passo` às `N` posições endereçadas.

        `1` na convenção de média (é o que torna o teto da §4.2 exato) e `N` na de
        soma. Existe para que `c_max` signifique **a mesma coisa** nas duas — o
        limite é sobre a mudança em `f`, não sobre o número que se escreve na
        memória.
        """
        return 1.0 if getattr(self, "convencao", "media") == "media" else self.n_ram

    def taxa_equivalente(self, taxa: float, para: str) -> float:
        """Converte uma taxa entre as convenções, preservando o passo em `f`.

        Uma taxa `η` na convenção de soma produz o mesmo deslocamento que `η·N` na
        de média. É a conversão que a §4.2 diz ser necessária para usar os números
        do apêndice A do Katopodis — e a que a RNF-01 do PRD errou ao prescrever
        `η = c/N` supondo `‖φ‖² = N`.
        """
        atual = getattr(self, "convencao", "media")
        if para not in ("media", "soma"):
            raise ValueError(f"para={para!r}")
        if atual == para:
            return float(taxa)
        return float(taxa * self.n_ram) if para == "media" else float(taxa / self.n_ram)

    def remapear(self, mapeamento: list) -> None:
        """Troca o mapeamento e **remonta** as matrizes de endereçamento.

        Existe porque trocar `self.mapeamento` direto deixaria `_tuplas` e
        `_pot_mat` desatualizadas, e o endereçamento passaria a usar o mapeamento
        antigo — silenciosamente. Quem carrega checkpoint ou importa modelo de
        fora (o `NTNModel` do `anbubus`) passa por aqui.
        """
        self.mapeamento = list(mapeamento)
        self.n_ram = len(self.mapeamento)
        self._potencias = [(np.uint64(1) << np.arange(len(t), dtype=np.uint64))
                           for t in self.mapeamento]
        self._montar_endereco_vetorizado()

    def _verificar(self, passo: NDArray[np.float64]) -> None:
        """Asserção da §4.2, aplicada ao `δ·g` **efetivo** — não ao nominal.

        Verificar o passo efetivo e não `η` é o que importa: `η` pequeno com `δ`
        explodindo diverge igual, e é justamente o que acontece quando `σ → 0` faz
        o fator `σ⁻²` da eq. (4.35) crescer sem limite (spec §4.4, razão 3).
        """
        self._n_passos += 1
        if not np.all(np.isfinite(passo)):
            raise TreinoInstavel(
                f"passo não finito no passo de treino {self._n_passos}: "
                f"{passo!r}. Origem provável: δ·∇log π com σ pequena (o fator "
                f"σ⁻² da eq. 4.35 explode), ou alvo do crítico já contaminado."
            )
        if self.c_max is not None:
            # Multiplica pelo ganho para o teto valer sobre a mudança em `f`, e
            # não sobre o valor escrito — senão `c_max` significaria coisas
            # diferentes nas duas convenções.
            pico = float(np.max(np.abs(passo))) * self.ganho_do_passo
            if pico > self.c_max:
                raise TreinoInstavel(
                    f"passo {pico:.4g} excede c_max={self.c_max} no passo de "
                    f"treino {self._n_passos}. Pela convenção de média (eq. "
                    f"4.12) isto é a mudança direta em f, então reduza η ou "
                    f"recorte δ. Passe c_max=None para desligar a asserção."
                )

    # -- persistência -------------------------------------------------------

    def _config(self) -> dict:
        """Tudo que não é conteúdo de memória. Vai junto no arquivo.

        Guardar o **mapeamento** e não a semente é deliberado: `randperm` depende
        da implementação do gerador, e um `numpy` futuro pode sortear diferente
        para a mesma semente. Um checkpoint que muda de significado com a versão
        da biblioteca é pior que um checkpoint grande — e a SPEC-03 já mediu que o
        mapeamento é ~20 % do footprint, então o custo é conhecido, não surpresa.
        """
        return {
            "classe": type(self).__name__,
            "n_bits": self.n_bits,
            "tamanho_tupla": self.tamanho_tupla,
            "n_saidas": self.n_saidas,
            "c_max": self.c_max,
            "convencao": getattr(self, "convencao", "media"),
            "phi": getattr(self, "phi", None),
        }

    def salvar(self, caminho) -> None:
        """Grava num `.npz`, achatando a memória esparsa em vetores paralelos.

        Formato: para cada RAM, os endereços e os valores em arrays separados,
        mais um vetor de contagens que diz onde cada RAM começa. É o mesmo
        princípio da `LUTOrdenada` da SPEC-03 — chaves e valores lado a lado —, e
        deixa o arquivo legível por qualquer coisa que leia `npz`, sem `pickle`.
        """
        import json

        enderecos, valores, contadores, tamanhos = [], [], [], []
        for m in self.memoria:
            tamanhos.append(len(m))
            for chave, v in m.items():
                enderecos.append(chave)
                if isinstance(v, tuple):          # crítico: (valor, contador)
                    valores.append(v[0])
                    contadores.append(v[1])
                else:
                    valores.append(v)
        dados = {
            "config": np.array(json.dumps(self._config())),
            "mapeamento": np.concatenate(self.mapeamento).astype(np.int64),
            "tamanhos_tupla": np.array([len(t) for t in self.mapeamento], np.int64),
            "tamanhos_ram": np.array(tamanhos, np.int64),
            "enderecos": np.array(enderecos, np.uint64) if enderecos
                         else np.zeros(0, np.uint64),
            "valores": (np.stack(valores) if valores
                        else np.zeros((0, self.n_saidas))),
        }
        if contadores:
            dados["contadores"] = np.stack(contadores)
        np.savez_compressed(caminho, **dados)

    @classmethod
    def carregar(cls, caminho):
        """Reconstrói a rede. Devolve a **classe gravada**, não `cls`.

        Assim `RedeGradienteFuncional.carregar` sobre um arquivo de crítico
        devolve o crítico, em vez de montar silenciosamente a rede errada.
        """
        import json

        z = np.load(caminho, allow_pickle=False)
        cfg = json.loads(str(z["config"]))
        alvo = {c.__name__: c for c in (RedeGradienteFuncional,
                                        RedeRegressaoEsquecimento)}[cfg["classe"]]

        kwargs = dict(n_bits=cfg["n_bits"], tamanho_tupla=cfg["tamanho_tupla"],
                      n_saidas=cfg["n_saidas"], c_max=cfg["c_max"])
        if alvo is RedeGradienteFuncional:
            kwargs["convencao"] = cfg["convencao"]
        else:
            kwargs["phi"] = cfg["phi"]
        rede = alvo(**kwargs)

        # mapeamento gravado, não re-sorteado
        corte = np.cumsum(z["tamanhos_tupla"])[:-1]
        rede.mapeamento = np.split(z["mapeamento"], corte)
        rede.n_ram = len(rede.mapeamento)
        rede._potencias = [(np.uint64(1) << np.arange(len(t), dtype=np.uint64))
                           for t in rede.mapeamento]

        rede.memoria = [{} for _ in range(rede.n_ram)]
        tem_cont = "contadores" in z
        i = 0
        for r, n in enumerate(z["tamanhos_ram"]):
            for _ in range(int(n)):
                chave = int(z["enderecos"][i])
                if tem_cont:
                    rede.memoria[r][chave] = (z["valores"][i].copy(),
                                              z["contadores"][i].copy())
                else:
                    rede.memoria[r][chave] = z["valores"][i].copy()
                i += 1
        return rede

    def entradas_ocupadas(self) -> int:
        """Posições de memória efetivamente escritas, somadas sobre as RAMs.

        Alimenta a figura de mérito T1 do PRD §5.5 — o **ocupado**, que é
        conteúdo de informação, distinto do **alocado** (`N × 2^n`) e do
        **realizado** (o footprint da estrutura de dados). Ver
        `src/uavwnn/custo/memoria.py`.
        """
        return sum(len(m) for m in self.memoria)

    def reiniciar(self) -> None:
        """Equivalente ao `reset!` do `Ramnet.jl` (`empty!(node.memory)`).

        Posição ausente vale zero, então esvaziar os dicionários devolve a rede
        ao estado inicial `v[i]_0(x) = 0 ∀x`, que é a condição inicial que
        §4.3.1 (p.38) declara explicitamente.
        """
        for m in self.memoria:
            m.clear()


class RedeGradienteFuncional(_RedeNTupla):
    """O **ator** — §4.3.1, p.38. "Functional gradient-inspired n-tuple network".

    EQUAÇÕES IMPLEMENTADAS

        v[i]_t(x_t) = v[i]_{t-1}(x_t) + α_t Δ_t,   i = 1..N          (4.11)

        f_t(x) = (1/N) Σ_i v[i]_t(x)                                 (4.12)

        Δ_t = −∂c(x_t, y_t, z)/∂z  avaliado em  z = f_{t-1}(x_t)

    onde `c` é uma função de perda diferenciável. Diferente da rede de regressão
    de §3.2.2, este modelo **exige a escolha de uma métrica de erro**: "the new
    model requires the choice of an error metric, not having one implicitly"
    (§4.3.1, p.38).

    O QUE AS POSIÇÕES GUARDAM

        Só o valor. "the memory positions do not need to store access counters.
        They hold only values, that are either scalars or vectors, depending on
        the dimensionality of the target function" (§4.3.1, p.38).

        Consequência de custo: metade dos bytes por entrada em relação ao
        crítico. A contabilidade de T1 precisa das duas colunas.

    MÉDIA, NÃO SOMA — e isto tem consequência direta na taxa de aprendizado

        A eq. (4.12) traz o `1/N` **explícito**. Com `Δ = −η(f − y)`, a variação
        da saída num passo é exatamente `−η(f − y)`, **independente de `N`**
        (testado em `test_ator_usa_media_e_o_passo_independe_de_N`).

        Isto é o que permite varrer `N` sem confundir capacidade com passo, e é
        onde o `EmbeddingBag(mode='sum')` do repositório `anbubus` divergiu do
        artigo. A RNF-01 antiga do nosso PRD prescrevia `η = c/N` supondo
        `‖φ‖² = N`; a eq. (4.22) normaliza o mapa por `1/√N`, então `‖φ‖² = 1` e
        não há divisão por `N` a fazer. Regra retirada.

    POR QUE ISTO É UMA MÁQUINA DE NÚCLEO — §4.3.1, p.39-41

        v[i]_t(x) = Σ_j α_j Δ_j M[i]_j(x)                            (4.13)
        M[i]_j(x) = ⟦τ[i](x) = τ[i](x_j)⟧                            (4.14)
        Σ_i M[i]_t(x) = N·k(x, x_t),  k = 1 − δ(x,x_t)/N              (4.15)
        f_t(x) = Σ_j α_j Δ_j k(x, x_j)                                (4.16)

        A eq. (4.16) é a diferença essencial em relação à rede de regressão:
        aqui é **soma de funções de base**, sem normalização; lá (3.22) é
        **média ponderada** de núcleo. É o que permite a este modelo representar
        funções que o outro não representa.

        v[i]_t(x) = v[i]_{t-1}(x) + α_t Δ_t M[i]_t(x)                 (4.17)
        f_t(x) = f_{t-1}(x) + α_t Δ_t k(x_t, x)                       (4.19)

        A (4.19) é a forma precisa da generalização: uma atualização em `x_t`
        muda a saída em **qualquer** outro ponto proporcionalmente ao núcleo. E
        ela tem a forma de um passo de descida de gradiente funcional, o que é
        a conclusão do §4.3.1: "the system implicitly implements a kernel
        machine that learns according to functional gradient descent".

        A passagem de (4.18) para (4.19) depende de o núcleo ser
        positivo-definido — o texto diz que "hinges on" isso. A prova vem do
        mapa de características explícito (4.22) e da identidade
        `⟨φ(w), φ(x)⟩ = k(w,x)` (4.23). Tudo isso está testado em
        `tests/test_identidades_artigo.py`.

    SEM FATOR DE ESQUECIMENTO

        Não é omissão: o esquecimento é do modelo de §4.2.1, e a combinação
        "gradiente funcional com esquecimento" não existe na dissertação. Ver o
        docstring do módulo (pendência C2).
    """

    def __init__(self, n_bits: int, tamanho_tupla: int, n_saidas: int = 1,
                 seed: int = 0, c_max: float | None = None,
                 convencao: str = "media", eta_treino: float = 0.1) -> None:
        super().__init__(n_bits, tamanho_tupla, n_saidas, seed, c_max)
        if convencao not in ("media", "soma"):
            raise ValueError(f"convencao={convencao!r}, esperado 'media' ou 'soma'")
        self.eta_treino = float(eta_treino)
        """`η` padrão de `treinar`, para quem chama sem passá-lo.

        Existe porque o `_aplicar` do Algoritmo 4.4 chama `critico.treinar(bits,
        alvo)` sem `eta` — e quando o crítico é desta família (a variante do
        `anbubus`), o `η` **precisa** respeitar a convenção. Sob soma, `η = 0,1`
        muda `f` em `0,1·N`, que com `N = 8` é 0,8 por passo: quase divergente.
        """
        self.convencao = convencao
        """`media` (eq. 4.12, padrão) ou `soma` — as **duas convenções
        incompatíveis** que a SPEC-05 §4.2 discute, agora selecionáveis.

            media   `f(x) = (1/N) Σ_i v[i](x)`     somar `η·g` muda `f` em `η·g`
            soma    `f(x) = Σ_i v[i](x)`           …muda `f` em `η·N·g`

        A média é a da dissertação e é o padrão. A soma existe porque é o que o
        `NTNModel` do repositório `anbubus/ntn_neurocomputing_v0` faz —
        `EmbeddingBag(mode='sum')` — e sem ela não é possível comparar as duas
        famílias sem trocar duas coisas ao mesmo tempo.

        **Consequência que não é cosmética:** as taxas não são transferíveis entre
        as convenções sem dividir por `N`. A `taxa_equivalente` abaixo faz a
        conversão explícita, em vez de deixá-la como pegadinha."""
        # Um dicionário por RAM, espelhando `FunctionalNode{D}` do Ramnet.jl:
        #     struct FunctionalNode{D}; memory::Dict{UInt,Vector{Float64}}
        # Posição ausente vale zero — a condição inicial v[i]_0(x) = 0 da §4.3.1.
        self.memoria: list[dict[int, NDArray[np.float64]]] = [
            {} for _ in range(self.n_ram)
        ]

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Eq. (4.12): `f(x) = (1/N) Σ_i v[i](x)`. `(M, B) -> (M, d)`.

        A divisão é por `N = n_ram`, **não** pelo número de posições visitadas.
        Posições nunca escritas contribuem com zero e ainda assim entram no
        denominador — é o que a (4.12) diz, e é o que o
        `predict(d::FunctionalDiscriminator, ...)` do Ramnet.jl faz
        (`d.cum_weight ./ length(d.nodes)`).
        """
        end = self.enderecos(bits)
        saida = np.zeros((end.shape[0], self.n_saidas))
        for m, linha in enumerate(end):
            for i, e in enumerate(linha):
                v = self.memoria[i].get(int(e))
                if v is not None:
                    saida[m] += v
        return saida / self.n_ram if self.convencao == "media" else saida

    def atualizar(self, bits: NDArray[np.bool_], passo: NDArray[np.float64]) -> None:
        """Eq. (4.11) crua: soma `passo` às `N` posições endereçadas por `x`.

        `passo` é o termo `α_t Δ_t` inteiro, já escalado — quem o monta é o
        chamador, porque **o que ele é depende do algoritmo**:

        - regressão supervisionada: `α_t Δ_t = −η(f − y)`, ver `treinar` abaixo
        - Algoritmo 4.4 (p.48): `α · δ · ∇_f ln f(A|S)`, com `δ` a vantagem da
          eq. (4.36) e o gradiente da política vindo da (4.35)

        A mesma separação existe no Ramnet.jl: `train!(f, x, grad)` faz
        `value .+= -grad` e não sabe de onde `grad` veio; quem calcula é o
        `ContinousActionPolicy.update!`.

        As posições **não** endereçadas ficam intactas — é o conteúdo da eq.
        (4.17), que unifica os dois casos pelo indicador `M[i]_t`.
        """
        passo = np.atleast_1d(np.asarray(passo, dtype=np.float64))
        self._verificar(passo)
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            v = self.memoria[i].get(chave)
            if v is None:
                # Primeira escrita nesta posição: v_0 = 0, logo v_1 = passo.
                self.memoria[i][chave] = passo.copy()
            else:
                v += passo

    def treinar(self, bits: NDArray[np.bool_], alvo: NDArray[np.float64],
                eta: float | None = None, epocas: int = 1,
                rng: np.random.Generator | None = None) -> None:
        """Regressão supervisionada com perda quadrática.

        Instancia a eq. (4.11) com `c(x,y,z) = ½(z − y)²`, logo
        `Δ_t = −∂c/∂z = −(f − y)` e o passo é `−η(f − y)`.

        **O Algoritmo 4.4 não passa por aqui** — ele chama `atualizar`
        diretamente, com o gradiente da política. Este método existe para os
        proxies de clonagem da SPEC-01 e para o teste diferencial contra o
        `NTupleRegressor` denso.

        A ordem de visita é uma permutação por época, como o `epochs` do
        `FunctionalOptimizer` do Ramnet.jl.
        """
        eta = self.eta_treino if eta is None else float(eta)
        # Aceita `(B,)` além de `(M, B)`. O `_aplicar` do Algoritmo 4.4 passa uma
        # observação só, e sem isto a variante de crítico funcional (`anbubus`)
        # estoura no `enderecos` com um `IndexError` obscuro.
        bits = np.atleast_2d(bits)
        alvo = np.asarray(alvo, dtype=np.float64)
        if alvo.ndim == 1:
            alvo = alvo[:, None]
        rng = rng or np.random.default_rng(0)
        for _ in range(epocas):
            for m in rng.permutation(len(bits)):
                x = bits[m:m + 1]
                # f_{t-1}(x_t): a predição ANTES da atualização, como a (4.11)
                # exige — Δ_t é avaliado em f_{t-1}, não em f_t.
                f = self.prever(x)[0]
                self.atualizar(x, -eta * (f - alvo[m]))


class RedeRegressaoEsquecimento(_RedeNTupla):
    """O **crítico** — §4.2.1, p.34. "Forgetting n-tuple regression network".

    EQUAÇÕES IMPLEMENTADAS

        v[i]_{t+1}(x_{t+1}) = φ · v[i]_t(x_{t+1}) + y_{t+1}           (4.8)
        c[i]_{t+1}(x_{t+1}) = c[i]_t(x_{t+1}) + 1

        f_t(x) = Σ_i v[i]_t(x) / Σ_i s[i]_t(x)   se Σ s ≠ 0, senão 0  (4.9)

        s[i]_t(x) = (1 − φ^c[i]_t(x)) / (1 − φ)  se c > 0 e φ ≠ 1     (4.10)
                  = c[i]_t(x)                    caso contrário

    A arquitetura é a mesma da rede de regressão original de §3.2.2 (p.23) —
    "consisting of a RAM discriminator where memory positions store a value and
    a counter". O que muda são a regra de atualização e a função de saída.

    POR QUE O DENOMINADOR NÃO É A SOMA DOS CONTADORES

        Na rede original a saída é `Σ V / Σ C` (eq. 3.16), uma média simples: o
        denominador conta quantas observações contribuíram. Com esquecimento as
        observações têm **pesos diferentes**, então a média tem de ser ponderada
        e o denominador passa a ser a **soma dos pesos**. A eq. (4.10) é a forma
        fechada dessa soma — os primeiros `c` termos de uma série geométrica de
        razão `φ`. Nas palavras do texto: "Fortunately, this is easily
        obtainable, since it is simply the sum of the first terms of a geometric
        series".

        Errar aqui produz números plausíveis: somar contadores em vez de pesos,
        ou aplicar `φ` na ordem trocada, dá saídas na escala certa e erradas. É
        por isso que este modelo é validado contra o `Ramnet.jl` com três
        valores de `φ`, não só por propriedades.

    OS DOIS EXTREMOS DE `φ` — §5.2.3, p.58

        "[φ] ranges from 0, where only the most recent observation to address a
        position influence it, to 1, where all samples to address a position
        influence its content equally and the system behaves exactly as an
        n-tuple regression network."

        `φ = 1` cai no ramo `c[i](x)` da (4.10). Não é conveniência numérica: é
        o que evita a divisão por zero de `(1 − 1^c)/(1 − 1)` **e** o que faz a
        rede degenerar exatamente na (3.16). Ambos testados.

        O apêndice A usa `φ = 0,85` em todas as quatro tarefas.

    POR QUE ESTA REDE EXISTE — §4.1.1, p.31

        A rede original supõe distribuição conjunta fixa ("The assumption of a
        fixed joint distribution"). Em RL o alvo se desloca: o crítico é
        alimentado com retornos que mudam conforme a política melhora. A fig. 4.2
        (p.35) ilustra o efeito — estimativa de densidade em que amostras antigas
        têm peso multiplicado por 0,1 a cada lote novo, o que permite acompanhar
        a distribuição subjacente.

    CUSTO: O DOBRO POR ENTRADA

        Valor **e** contador por posição, ambos vetores de dimensão `d`. O ator
        guarda só o valor. A tabela de T1 precisa das duas colunas — reportar só
        o ator subestimaria o modelo completo por ~2×.
    """

    def __init__(self, n_bits: int, tamanho_tupla: int, n_saidas: int = 1,
                 phi: float = 1.0, seed: int = 0,
                 c_max: float | None = None) -> None:
        # §4.2.1: "the forgetting factor, denoted by φ and defined in the [0,1]
        # interval".
        if not 0.0 <= phi <= 1.0:
            raise ValueError(f"phi={phi} fora de [0, 1]")
        # `c_max=None` por padrão, e isso é deliberado: o crítico **acumula** `G`
        # pela eq. (4.8) e não tem taxa de aprendizado, então um alvo grande é
        # legítimo — retorno grande é retorno grande. Limitar aqui rejeitaria
        # dado bom. O que ainda se verifica é finitude, que nunca é legítima.
        super().__init__(n_bits, tamanho_tupla, n_saidas, seed, c_max)
        self.phi = float(phi)
        # Espelha `RegressionNodeMemoryElement{D}` do Ramnet.jl, que guarda
        # `count::Vector{Int}` e `value::Vector{Float64}` — ambos de comprimento
        # D. O contador é POR COMPONENTE, e isso não é detalhe: o Algoritmo 4.1
        # (p.37) atualiza só `v[i](S_code, A)`, a componente da ação tomada, e um
        # contador compartilhado degradaria a média ponderada da (4.9) para as
        # ações menos visitadas. Ver `treinar_componente`.
        self.memoria: list[dict[int, tuple[NDArray, NDArray]]] = [
            {} for _ in range(self.n_ram)
        ]

    def _pesos(self, contador: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eq. (4.10): o peso acumulado de uma posição, dado seu contador.

        Soma dos `c` primeiros termos da série geométrica de razão `φ`. O ramo
        `φ = 1` devolve o próprio contador, recuperando a (3.16).
        """
        if self.phi == 1.0:
            return contador
        return (1.0 - self.phi ** contador) / (1.0 - self.phi)

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Eq. (4.9): `f(x) = Σ_i v[i](x) / Σ_i s[i](x)`. `(M, B) -> (M, d)`.

        O ramo `Σ s = 0 → 0` é explícito na equação, não uma salvaguarda nossa.
        Ele cobre o caso em que nenhuma das `N` posições endereçadas foi escrita
        — situação comum no início do treino e, com tuplas grandes, também para
        entradas novas (§3.2.4, p.27: "the model is unable to generalize").
        """
        end = self.enderecos(bits)
        saida = np.zeros((end.shape[0], self.n_saidas))
        for m, linha in enumerate(end):
            num = np.zeros(self.n_saidas)
            den = np.zeros(self.n_saidas)
            for i, e in enumerate(linha):
                item = self.memoria[i].get(int(e))
                if item is not None:
                    v, c = item
                    num += v
                    den += self._pesos(c)
            saida[m] = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        return saida

    def treinar(self, bits: NDArray[np.bool_], alvo: NDArray[np.float64]) -> None:
        """Eq. (4.8), aplicada a **todas** as componentes da saída.

        A ordem importa e é a do texto: "when a memory position is to be
        updated, its content is **first** multiplied by the forgetting factor and
        **then** incremented by the value associated with a new observation".
        Inverter as duas daria peso errado à observação nova.

        Para atualizar uma componente só — o que o Algoritmo 4.1 exige — use
        `treinar_componente`.
        """
        alvo = np.atleast_1d(np.asarray(alvo, dtype=np.float64))
        self._verificar(alvo)
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            item = self.memoria[i].get(chave)
            if item is None:
                # Posição virgem: v_0 = 0 e c_0 = 0, logo (4.8) dá
                # v_1 = φ·0 + y = y  e  c_1 = 1.
                self.memoria[i][chave] = (alvo.astype(np.float64).copy(),
                                          np.ones(self.n_saidas))
            else:
                v, c = item
                v *= self.phi          # primeiro o esquecimento...
                v += alvo              # ...depois a observação nova
                c += 1.0

    def treinar_componente(self, bits: NDArray[np.bool_], indice: int,
                           alvo: float) -> None:
        """Eq. (4.8) aplicada a **uma** componente — o que o Algoritmo 4.1 pede.

        O Algoritmo 4.1 (p.37) atualiza `v[i](S_code_τ, A_τ)`: só o
        discriminador ligado à ação tomada. O §4.2.2 (p.35) explica por que a
        função de valor-ação precisa disso — ela tem dois argumentos, mas a rede
        aceita um só padrão binário:

            "a much simpler approach is handling state and action separately.
             The separation could be understood as modeling a state-value
             function for each possible action."

        A fig. 4.3 (p.36) mostra as **duas realizações equivalentes**: (a) um
        discriminador com posições vetoriais, (b) vários discriminadores
        escalares. Usamos (a) — é o que `n_saidas` já dá. E é por isso que o
        contador é por componente: cada ação acumula suas próprias visitas.

        Note que **só o estado é codificado**: "only states require being
        encoded into a binary pattern".
        """
        if not 0 <= indice < self.n_saidas:
            raise ValueError(f"indice={indice} fora de [0, {self.n_saidas})")
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            item = self.memoria[i].get(chave)
            if item is None:
                v, c = np.zeros(self.n_saidas), np.zeros(self.n_saidas)
                v[indice] = float(alvo)
                c[indice] = 1.0
                self.memoria[i][chave] = (v, c)
            else:
                v, c = item
                v[indice] = self.phi * v[indice] + float(alvo)
                c[indice] += 1.0
