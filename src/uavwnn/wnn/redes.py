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


_TETO_DENSO_BYTES = 16 * 2**20
"""Teto para o armazenamento DENSO (bytes do bloco de valores, float64).

Para `n = 12` cada RAM tem só 4 096 posições, e o dicionário por entrada
custava 218 B (ator) e 361 B (crítico) para 24 e 12 B de dado útil — medido
por tracemalloc num checkpoint real: 15,9 MB e 26,4 MB de RAM para o que o
denso guarda em 3,5 MB e 1,7 MB, sem chave nenhuma e com indexação direta.

O teto existe porque a classe aceita `n` até 64 e `2^64` posições não são
alocáveis — é o achado da SPEC-03 que fez as redes nascerem esparsas. Denso
só quando cabe (16 MiB cobre todos os modelos do projeto, `n = 12`); acima
disso fica o dicionário de sempre, intocado. A troca é INVISÍVEL de fora:
`memoria` continua legível como lista de dicts, e todos os caminhos quentes
replicam as operações do modo esparso na MESMA ordem — a igualdade bit a bit
com o comportamento anterior é pinada em teste e verificada por treino de
referência com sementes fixas.
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

        Quem liga é `NTuplaDissertacaoAprox`, o ponto onde o gradiente de política de fato
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

    def _iniciar_armazenamento(self, com_contador: bool) -> None:
        """Denso quando cabe (ver `_TETO_DENSO_BYTES`); dicionários senão."""
        slots = 1 << self.tamanho_tupla
        self._densa = (self.n_ram * slots * self.n_saidas * 8
                       <= _TETO_DENSO_BYTES)
        if self._densa:
            self._val = np.zeros((self.n_ram, slots, self.n_saidas))
            self._ocup = np.zeros((self.n_ram, slots), dtype=bool)
            self._cnt = (np.zeros((self.n_ram, slots, self.n_saidas),
                                  dtype=np.uint32) if com_contador else None)
            self._memo = None
        else:
            self._memo = [{} for _ in range(self.n_ram)]
            self._cnt = None

    @property
    def memoria(self):
        """Lista de dicts, como sempre foi — reconstruída sob demanda no denso.

        No modo denso isto é uma VISTA (os arrays de valor são views nos blocos
        densos): ler é barato e coerente, mas ESCREVER num dict reconstruído se
        perde — quem precisa popular uma rede de fora atribui a lista inteira
        (`rede.memoria = dicts`), que o setter ingere no armazenamento certo.
        É o padrão que `carregar` e `spec06_importar_ppo` já usavam.
        """
        if not getattr(self, "_densa", False):
            return self._memo
        saida = []
        for i in range(self.n_ram):
            idx = np.flatnonzero(self._ocup[i])
            if self._cnt is None:
                saida.append({int(a): self._val[i, a] for a in idx})
            else:
                saida.append({int(a): (self._val[i, a], self._cnt[i, a])
                              for a in idx})
        return saida

    @memoria.setter
    def memoria(self, dicts) -> None:
        if not getattr(self, "_densa", False):
            self._memo = dicts               # mesmo objeto: mutação externa vale
            return
        self._val[:] = 0.0
        self._ocup[:] = False
        if self._cnt is not None:
            self._cnt[:] = 0
        for i, m in enumerate(dicts):
            for k, v in m.items():
                if isinstance(v, tuple):
                    self._val[i, k] = v[0]
                    self._cnt[i, k] = v[1]
                else:
                    self._val[i, k] = v
                self._ocup[i, k] = True

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
        if getattr(self, "_densa", False):
            # denso: entradas em ordem crescente de endereço, por RAM. No modo
            # esparso a ordem era a de inserção do dict — o CONTEÚDO é o mesmo,
            # e nada no projeto depende da ordem (prever soma por RAM, não por
            # entrada; a LUTPlana ordena as chaves ao nascer).
            for i in range(self.n_ram):
                idx = np.flatnonzero(self._ocup[i])
                tamanhos.append(int(idx.size))
                for a in idx:
                    enderecos.append(int(a))
                    valores.append(self._val[i, a])
                    if self._cnt is not None:
                        contadores.append(self._cnt[i, a])
        else:
            for m in self._memo:
                tamanhos.append(len(m))
                for chave, v in m.items():
                    enderecos.append(chave)
                    if isinstance(v, tuple):      # crítico: (valor, contador)
                        valores.append(v[0])
                        contadores.append(v[1])
                    else:
                        valores.append(v)
        # dtypes ESTREITOS, dimensionados pelo que o conteúdo exige — não pelo
        # que a aritmética interna usa. O endereço local à RAM tem
        # `tamanho_tupla` bits (12 -> máx 4095), e gravá-lo em uint64 fazia o
        # campo pesar 40 % do checkpoint do crítico para carregar zeros. O
        # `carregar` é agnóstico a dtype (os arrays viram índices/ints), então
        # arquivos antigos em uint64 continuam legíveis; a igualdade bit a bit
        # do round-trip é pinada em teste. Os VALORES ficam em float64 de
        # propósito: são estado de treino, e truncá-los mudaria a retomada.
        t_end = (np.uint16 if self.tamanho_tupla <= 16
                 else np.uint32 if self.tamanho_tupla <= 32 else np.uint64)
        t_map = np.uint16 if self.n_bits < 2**16 else np.int64
        dados = {
            "config": np.array(json.dumps(self._config())),
            "mapeamento": np.concatenate(self.mapeamento).astype(t_map),
            "tamanhos_tupla": np.array([len(t) for t in self.mapeamento],
                                       np.int32),
            "tamanhos_ram": np.array(tamanhos, np.int32),
            "enderecos": np.array(enderecos, t_end) if enderecos
                         else np.zeros(0, t_end),
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

        # `remapear` e NÃO atribuição direta, e a diferença era um defeito real.
        #
        # Atribuir `.mapeamento` deixa `_tuplas` e `_pot_mat` — as matrizes que o
        # `enderecos()` vetorizado de fato usa — com o mapa ALEATÓRIO do construtor.
        # A rede carregada então lê endereços errados, com os valores certos na
        # memória, e não levanta nada: `n_ram` coincide, as formas coincidem, só o
        # conteúdo está trocado.
        #
        # Medido antes da correção: a rede original predizia [1,004; 2,008; 3,012] e
        # a carregada [0,012; 0,024; 0,036]. O docstring de `remapear` já dizia
        # "quem carrega checkpoint passa por aqui" — e este caminho não passava.
        corte = np.cumsum(z["tamanhos_tupla"])[:-1]
        rede.remapear(np.split(z["mapeamento"], corte))

        # Os arrays saem do `NpzFile` UMA vez, fora do laço, e isso não é estilo.
        #
        # `NpzFile.__getitem__` relê e descomprime do zip a cada acesso — medido: 3,4
        # ms por acesso a um array de 70 950 elementos, sem cache. Acessar dentro do
        # laço fazia 70 950 leituras completas, ou seja trabalho quadrático: **4 min
        # só para `enderecos`**, mais o mesmo para `valores`. Um checkpoint de uma
        # corrida de 3,5 M de passos levava ~8 min para carregar, e o custo cresce
        # com o quadrado da ocupação.
        tamanhos = z["tamanhos_ram"]
        enderecos = z["enderecos"]
        valores = z["valores"]
        contadores = z["contadores"] if "contadores" in z else None

        # `rede` e não `cls`: o `carregar` devolve a CLASSE GRAVADA, que pode
        # não ser a que recebeu a chamada.
        tipo_cont = getattr(type(rede), "TIPO_CONTADOR", np.uint32)
        if rede._densa:
            i = 0
            for r, n in enumerate(tamanhos):
                for _ in range(int(n)):
                    chave = int(enderecos[i])
                    rede._val[r, chave] = valores[i]
                    if contadores is not None:
                        rede._cnt[r, chave] = contadores[i].astype(tipo_cont)
                    rede._ocup[r, chave] = True
                    i += 1
            return rede
        memo = [{} for _ in range(rede.n_ram)]
        i = 0
        for r, n in enumerate(tamanhos):
            for _ in range(int(n)):
                chave = int(enderecos[i])
                if contadores is not None:
                    memo[r][chave] = (valores[i].copy(),
                                      contadores[i].astype(tipo_cont))
                else:
                    memo[r][chave] = valores[i].copy()
                i += 1
        rede.memoria = memo
        return rede

    def entradas_ocupadas(self) -> int:
        """Posições de memória efetivamente escritas, somadas sobre as RAMs.

        Alimenta a figura de mérito T1 do PRD §5.5 — o **ocupado**, que é
        conteúdo de informação, distinto do **alocado** (`N × 2^n`) e do
        **realizado** (o footprint da estrutura de dados). Ver
        `src/uavwnn/custo/memoria.py`.
        """
        if getattr(self, "_densa", False):
            return int(self._ocup.sum())
        return sum(len(m) for m in self._memo)

    def reiniciar(self) -> None:
        """Equivalente ao `reset!` do `Ramnet.jl` (`empty!(node.memory)`).

        Posição ausente vale zero, então esvaziar os dicionários devolve a rede
        ao estado inicial `v[i]_0(x) = 0 ∀x`, que é a condição inicial que
        §4.3.1 (p.38) declara explicitamente.
        """
        if getattr(self, "_densa", False):
            self._val[:] = 0.0
            self._ocup[:] = False
            if self._cnt is not None:
                self._cnt[:] = 0
            return
        for m in self._memo:
            m.clear()


class LUTPlana(_RedeNTupla):
    """Ator IMUTÁVEL, para inferência: três arrays contíguos em vez de `N` dicts.

    POR QUE ELA EXISTE — e o número que a motivou

    O T1 modela os bytes de uma implementação em C. A implementação em Python
    cobra outra coisa, e ninguém tinha medido: um modelo de 2,98 MB modelados
    ocupava **94,6 MB de RSS**, 31,8×. A causa não é a rede — é granularidade.
    Uma entrada guarda 24 B e custa 201: o cabeçalho do `ndarray` sozinho são
    136 B (5,7× o conteúdo), mais 28 do `int` boxed e 37 do slot do dict. São
    259 mil objetos Python minúsculos, cada um com seu cabeçalho.

    A ESTRUTURA

        chaves   int32   (occ,)     chave GLOBAL = ram·2ⁿ + endereço, ORDENADA
        valores  float32 (occ, d)
        desloc   int32   (N,)       ram·2ⁿ, pré-computado

    A chave global é o ponto: em vez de `N` tabelas separadas, **uma** ordenada.
    Com `N = 72` e `n = 12` a chave máxima é 294 912 — cabe em `int32` com folga,
    e o construtor recusa se não couber.

    O GANHO, medido no ator de `T=32, Na=2, Ns=24` (129 664 entradas):

        dict[int, ndarray]   25 452 KB   93,0 µs
        LUTPlana int32/f32    2 026 KB   12,0 µs    12,6× menor · 7,8× mais rápida

    Ganha nos DOIS eixos, e a velocidade é contraintuitiva — busca binária
    `O(log n)` contra dict `O(1)`. A razão está medida: o dict paga **por RAM**
    (`N` buscas em Python, cada uma com boxing, hash e desreferência); esta faz
    UMA chamada compilada sobre memória contígua.

    POR QUE SÓ INFERÊNCIA

    É imutável por construção: as chaves estão ordenadas e a busca binária depende
    disso, então inserir exigiria reordenar o array inteiro. O treino escreve a
    cada transição — ~13 milhões de escritas numa corrida de 6,75 M passos, muitas
    em endereços nunca vistos. `atualizar` e `treinar` levantam aqui, de
    propósito: é melhor falhar alto que degradar em silêncio.
    """

    def __init__(self, rede: "RedeGradienteFuncional",
                 dtype_valor=np.float32) -> None:
        if not isinstance(rede, RedeGradienteFuncional):
            raise TypeError(f"esperava RedeGradienteFuncional, veio {type(rede).__name__}")
        self.n_bits = rede.n_bits
        self.tamanho_tupla = rede.tamanho_tupla
        self.n_saidas = rede.n_saidas
        self.n_ram = rede.n_ram
        self.convencao = rede.convencao
        self.c_max = rede.c_max
        self.mapeamento = rede.mapeamento
        self._potencias = rede._potencias
        self._tuplas = rede._tuplas
        self._pot_mat = rede._pot_mat
        self._n_passos = 0

        desloc = 1 << self.tamanho_tupla
        maximo = self.n_ram * desloc
        if maximo >= 2**31:
            raise ValueError(
                f"chave global máxima {maximo:,} não cabe em int32 "
                f"(n_ram={self.n_ram}, tupla={self.tamanho_tupla})")
        chaves, valores = [], []
        for i, m in enumerate(rede.memoria):
            for k in sorted(m):                       # ORDENADO por RAM, e as
                chaves.append(i * desloc + k)         # RAMs em ordem crescente
                valores.append(m[int(k)])             # → o vetor global já sai
        self.chaves = np.asarray(chaves, np.int32)    #   ordenado, sem sort
        self.valores = (np.asarray(valores, dtype_valor).reshape(-1, self.n_saidas)
                        if valores else np.zeros((0, self.n_saidas), dtype_valor))
        self._desloc = (np.arange(self.n_ram) * desloc).astype(np.int32)

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Eq. (4.12), com busca binária vetorizada. `(M, B) -> (M, d)`.

        Mesma semântica do dict: posição ausente contribui zero e ainda entra no
        denominador `N`.
        """
        end = self.enderecos(bits)
        saida = np.zeros((end.shape[0], self.n_saidas))
        if self.chaves.size:
            for m, linha in enumerate(end):
                g = self._desloc + linha.astype(np.int32)
                j = np.minimum(np.searchsorted(self.chaves, g), self.chaves.size - 1)
                ok = self.chaves[j] == g
                if ok.any():
                    saida[m] = self.valores[j[ok]].sum(axis=0, dtype=np.float64)
        return saida / self.n_ram if self.convencao == "media" else saida

    def consulta_do_vetor(self, dim_por_bit, limiar_por_bit):
        """Compila `v -> f(v)`: termômetro, endereçamento e busca numa closure.

        O caminho em três etapas — `codificar` (aloca os `B` bits), `enderecos`
        (aloca o estendido e a fatia `(1, N, n)`) e `prever` — gasta ~20 despachos
        NumPy por decisão, e para `M = 1` o despacho domina o trabalho. Aqui o bit
        nunca é materializado: para cada slot de tupla, junta-se A DIMENSÃO e O
        LIMIAR que o bit compararia, e o endereço sai de um único
        `(v[dim] >= lim) · pot`. Três despachos no lugar de doze.

        A SEMÂNTICA É A MESMA POR CONSTRUÇÃO, não por aproximação: o bit
        `b = d·T + t` do termômetro é `v[d] >= limiares[d, t]`, e é exatamente
        essa comparação que o gather reproduz — mesmos float64, mesmo `>=`. O
        sentinela (índice `n_bits`, para tuplas curtas) vira limiar `+inf`, e o
        peso dele em `_pot_mat` já é zero. Bit a bit igual a
        `prever(codificar(v))[0]`, pinado em teste.

        `dim_por_bit` e `limiar_por_bit` vêm de fora (do termômetro) porque esta
        classe não conhece codificação — e é assim que deve ficar.
        """
        dim = np.concatenate([np.asarray(dim_por_bit, np.intp),
                              [0]])[self._tuplas]
        lim = np.concatenate([np.asarray(limiar_por_bit, float),
                              [np.inf]])[self._tuplas]
        pot = self._pot_mat
        chaves, valores, desloc = self.chaves, self.valores, self._desloc
        tam, n_saidas = chaves.size, self.n_saidas
        divisor = self.n_ram if self.convencao == "media" else 1
        d_obs = int(np.asarray(dim_por_bit).max()) + 1

        def consulta(v):
            if v.size != d_obs:
                raise ValueError(f"observação com {v.size} entradas, "
                                 f"esperado {d_obs}")
            addr = ((v[dim] >= lim) * pot).sum(axis=1)
            saida = np.zeros(n_saidas)
            if tam:
                g = desloc + addr.astype(np.int32)
                j = np.minimum(np.searchsorted(chaves, g), tam - 1)
                ok = chaves[j] == g
                if ok.any():
                    saida = valores[j[ok]].sum(axis=0, dtype=np.float64)
            return saida / divisor

        return consulta

    @property
    def memoria(self):
        """Vista somente-leitura, para o harness de custo contar a ocupação.

        Reconstrói dicionários — é caro, e de propósito: quem precisar disto num
        caminho quente está usando a estrutura errada.
        """
        desloc = 1 << self.tamanho_tupla
        saida = [{} for _ in range(self.n_ram)]
        for k, v in zip(self.chaves.tolist(), self.valores):
            saida[k // desloc][k % desloc] = v
        return saida

    def _imutavel(self, *_a, **_k):
        raise TypeError(
            "LUTPlana é imutável — as chaves estão ordenadas e a busca binária "
            "depende disso. Para treinar, use RedeGradienteFuncional.")

    atualizar = treinar = treinar_componente = salvar = _imutavel


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
        # Espelha `FunctionalNode{D}` do Ramnet.jl (Dict{UInt,Vector{Float64}});
        # posição ausente vale zero — a condição inicial v[i]_0(x) = 0 da §4.3.1.
        # Denso quando cabe, dicts senão; ver `_TETO_DENSO_BYTES`.
        self._iniciar_armazenamento(com_contador=False)

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
        if self._densa:
            # A MESMA soma sequencial do modo esparso, com indexação direta no
            # lugar do dict.get — a ordem de acumulação é idêntica, e a
            # igualdade é bit a bit (pinada em teste).
            val, ocup = self._val, self._ocup
            for m, linha in enumerate(end):
                for i, e in enumerate(linha):
                    if ocup[i, e]:
                        saida[m] += val[i, e]
        else:
            for m, linha in enumerate(end):
                for i, e in enumerate(linha):
                    v = self._memo[i].get(int(e))
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
        if self._densa:
            for i, e in enumerate(self.enderecos(bits)[0]):
                if self._ocup[i, e]:
                    self._val[i, e] += passo
                else:
                    # Primeira escrita: v_0 = 0, logo v_1 = passo.
                    self._val[i, e] = passo
                    self._ocup[i, e] = True
            return
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            v = self._memo[i].get(chave)
            if v is None:
                # Primeira escrita nesta posição: v_0 = 0, logo v_1 = passo.
                self._memo[i][chave] = passo.copy()
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
        # (valor, contador) por posição; denso quando cabe, dicts senão.
        self._iniciar_armazenamento(com_contador=True)

    TIPO_CONTADOR = np.uint32
    """O contador é uma CONTAGEM DE VISITAS — inteiro por construção (`c += 1`).

    Estava em `float64` por conveniência (mantinha valor e contador no mesmo
    dtype), gastando 8 bytes onde 4 bastam: metade do custo do crítico, que
    guarda dois números por entrada.

    `uint32` e não `uint16`: o máximo medido num modelo real é **2 353 751**
    visitas na posição mais quente (p99 = 33 127), muito acima dos 65 535 do
    `uint16`. `uint32` cobre 4,29e9 — a ~100 passos/s, mais de um ano de treino
    contínuo na MESMA posição.

    Nenhuma mudança numérica: `_pesos` promove a float na exponenciação, e
    `1 − φ^c` com `c` inteiro dá exatamente o mesmo que com `c` float, porque o
    valor era inteiro nos dois casos.
    """

    def _pesos(self, contador: NDArray[np.integer]) -> NDArray[np.float64]:
        """Eq. (4.10): o peso acumulado de uma posição, dado seu contador.

        Soma dos `c` primeiros termos da série geométrica de razão `φ`. O ramo
        `φ = 1` devolve o próprio contador, recuperando a (3.16).
        """
        if self.phi == 1.0:
            # `float64` na saída mesmo com contador inteiro: quem chama divide
            # por este valor, e devolver inteiro faria a divisão truncar.
            return np.asarray(contador, dtype=np.float64)
        return (1.0 - self.phi ** np.asarray(contador, dtype=np.float64)) / (1.0 - self.phi)

    def prever(self, bits: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Eq. (4.9): `f(x) = Σ_i v[i](x) / Σ_i s[i](x)`. `(M, B) -> (M, d)`.

        O ramo `Σ s = 0 → 0` é explícito na equação, não uma salvaguarda nossa.
        Ele cobre o caso em que nenhuma das `N` posições endereçadas foi escrita
        — situação comum no início do treino e, com tuplas grandes, também para
        entradas novas (§3.2.4, p.27: "the model is unable to generalize").
        """
        # COLETA e depois SOMA, em vez de acumular por RAM — e o ganho foi medido.
        #
        # A versão anterior fazia, por posição encontrada, dois `+=` de array e uma
        # exponenciação escalar `phi ** contador`. O custo cresce com a TAXA DE
        # ACERTO, que cresce com a ocupação — e é isso que fazia o treino desacelerar
        # ao longo da corrida sem que contenção, GC ou localidade explicassem:
        #
        #     ocupadas       acertos/prever   ms/prever
        #            0            0/67           0,020
        #       20 000            4/67           0,072
        #      120 000           29/67           0,313
        #      200 000           51/67           0,480   (24× o vazio)
        #
        # O `prever` do ator foi vetorizado na T-06.0; este ficou com o laço, e o
        # perfil o mostrou consumindo 33 % do laço de treino, com `_pesos` chamado
        # 381 960 vezes em 3 000 passos.
        #
        # A ORDEM DA SOMA MUDA, e isso é decisão consciente: `np.sum` usa soma por
        # pares e o acumulador dava soma sequencial, então os últimos bits diferem.
        # Os testes diferenciais contra o Ramnet.jl passam com a tolerância que já
        # tinham; exigir bit a bit aqui seria fixar uma ordem de acumulação que a
        # eq. (4.9) não prescreve — ela é uma soma sobre `i`, sem ordem.
        end = self.enderecos(bits)
        saida = np.zeros((end.shape[0], self.n_saidas))
        densa = self._densa
        for m, linha in enumerate(end):
            vals, cnts = [], []
            if densa:
                for i, e in enumerate(linha):
                    if self._ocup[i, e]:
                        vals.append(self._val[i, e])
                        cnts.append(self._cnt[i, e])
            else:
                for i, e in enumerate(linha):
                    item = self._memo[i].get(int(e))
                    if item is not None:
                        vals.append(item[0])
                        cnts.append(item[1])
            if not vals:
                continue                      # eq. (4.9): `Σ s = 0 → 0`, explícito
            num = np.sum(vals, axis=0)
            den = np.sum(self._pesos(np.asarray(cnts)), axis=0)
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
        if self._densa:
            for i, e in enumerate(self.enderecos(bits)[0]):
                if self._ocup[i, e]:
                    v = self._val[i, e]
                    v *= self.phi      # primeiro o esquecimento...
                    v += alvo          # ...depois a observação nova
                    self._cnt[i, e] += 1
                else:
                    # Posição virgem: v_1 = φ·0 + y = y  e  c_1 = 1.
                    self._val[i, e] = alvo
                    self._cnt[i, e] = 1
                    self._ocup[i, e] = True
            return
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            item = self._memo[i].get(chave)
            if item is None:
                # Posição virgem: v_0 = 0 e c_0 = 0, logo (4.8) dá
                # v_1 = φ·0 + y = y  e  c_1 = 1.
                self._memo[i][chave] = (
                    alvo.astype(np.float64).copy(),
                    np.ones(self.n_saidas, dtype=self.TIPO_CONTADOR))
            else:
                v, c = item
                v *= self.phi          # primeiro o esquecimento...
                v += alvo              # ...depois a observação nova
                c += 1                 # inteiro: contagem de visitas

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
        if self._densa:
            for i, e in enumerate(self.enderecos(bits)[0]):
                if self._ocup[i, e]:
                    self._val[i, e, indice] = (self.phi * self._val[i, e, indice]
                                               + float(alvo))
                    self._cnt[i, e, indice] += 1
                else:
                    self._val[i, e, indice] = float(alvo)
                    self._cnt[i, e, indice] = 1
                    self._ocup[i, e] = True
            return
        for i, e in enumerate(self.enderecos(bits)[0]):
            chave = int(e)
            item = self._memo[i].get(chave)
            if item is None:
                v = np.zeros(self.n_saidas)
                c = np.zeros(self.n_saidas, dtype=self.TIPO_CONTADOR)
                v[indice] = float(alvo)
                c[indice] = 1
                self._memo[i][chave] = (v, c)
            else:
                v, c = item
                v[indice] = self.phi * v[indice] + float(alvo)
                c[indice] += 1
