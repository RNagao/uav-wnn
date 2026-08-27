"""T1 — modelo de custo de memória (SPEC-03, T-03.2).

Três números por modelo, e a distinção entre eles é o ponto da spec:

- **alocado**: `cabeças × N × 2ⁿ × bytes`. Cota superior teórica. É o número que
  costuma ser reportado, e é uma mentira **a favor** da hipótese.
- **ocupado**: entradas distintas efetivamente endereçadas × bytes. É conteúdo
  de informação, não footprint.
- **realizado**: o footprint da estrutura de dados de fato usada. **É este que
  responde "cabe no microcontrolador"**, e é o que decide a comparação com a
  MLP pareada.

E duas colunas, sempre (§4.2): **persistente** (flash) e **residente** (RAM).
O mapeamento das tuplas é uma permutação de PRNG — uma seed de 4 B substitui
~1 KB no persistente, mas **não** no residente. Usar a seed para encolher a
coluna residente é proibido: é justamente a coluna que responde à pergunta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Variante = Literal["hash", "ordenado", "denso"]


def _bytes_indice(n_valores: int) -> int:
    """Bytes por índice, arredondado para byte inteiro."""
    return max(1, math.ceil(math.log2(max(2, n_valores)) / 8))


@dataclass
class CustoMemoria:
    """Custo de um modelo, em bytes. Nada de KB ou razões — o gerador de
    tabelas deriva isso (T-03.9); aqui só se grava o bruto."""

    rotulo: str
    persistente: int
    residente: int
    detalhe: dict[str, int] = field(default_factory=dict)
    """Decomposição obrigatória: LUTs, mapeamento de tuplas, limiares,
    parâmetros da distribuição, buffers. Omitir qualquer um infla a vantagem."""

    def __post_init__(self) -> None:
        # A soma tem de fechar: fórmula que não bate com as partes é bug.
        soma = sum(v for k, v in self.detalhe.items() if not k.startswith("buffer"))
        if soma and abs(soma - self.persistente) > 1:
            raise ValueError(
                f"{self.rotulo}: detalhe soma {soma} B mas persistente é "
                f"{self.persistente} B"
            )


def ocupacao_por_ram(pesos: NDArray) -> NDArray[np.int64]:
    """Entradas distintas efetivamente escritas, **por RAM**.

    Por RAM e não no total, porque o custo de todas as três variantes depende
    da distribuição entre RAMs, não só do agregado — desbalanceamento paga
    arredondamento em cada uma.
    """
    # pesos: (cabeças, N, 2^n) — uma posição conta se qualquer cabeça a escreveu
    tocado = (pesos != 0).any(axis=0)
    return tocado.sum(axis=1).astype(np.int64)


def _bytes_luts(
    occ_por_ram: NDArray[np.int64],
    n_ram: int,
    tamanho_tupla: int,
    n_cabecas: int,
    bytes_valor: int,
    variante: Variante,
    alpha: float = 0.75,
    bytes_contador: int = 0,
) -> int:
    """Footprint das LUTs sob cada estratégia de armazenamento.

    `bytes_contador` cobre a rede de regressão com esquecimento (§4.2.1), que
    guarda **valor E contador** por componente — o ator (§4.3.1) guarda só o
    valor ("the memory positions do not need to store access counters").

    Sem este termo o crítico saía ~40 % subestimado. Não afetou o T1 publicado,
    que compara ATORES, mas qualquer conta de "ator + crítico" pelo harness
    estava errada.
    """
    b_chave = 2 if tamanho_tupla <= 16 else 4
    occ = int(occ_por_ram.sum())
    b_entrada = n_cabecas * (bytes_valor + bytes_contador)

    if variante == "denso":
        # V-c: previsibilidade máxima, memória máxima. É a mitigação do RNF-10
        # (cauda do hash) e o limite superior.
        return n_ram * (1 << tamanho_tupla) * b_entrada

    if variante == "ordenado":
        # V-b: chaves ordenadas + busca binária. Sem folga de carga; O(log occ)
        # com cauda determinística.
        b_offset = 4
        return occ * (b_chave + b_entrada) + n_ram * b_offset

    # V-a: hash aberto por RAM. Paga fator de carga E arredondamento para
    # potência de 2 — é a variante que a conta ilustrativa da spec usava.
    total = 0
    for oc in occ_por_ram:
        if oc == 0:
            continue
        cap = 1 << math.ceil(math.log2(max(1, oc / alpha)))
        total += cap * (b_chave + b_entrada)                 # tabela
        total += math.ceil(cap / 8)                          # bitmap de ocupação
    total += n_ram * 4                                       # cabeçalhos
    return total


def custo_ntuple(
    pesos: NDArray,
    n_bits: int,
    tamanho_tupla: int,
    n_ram: int,
    variante: Variante = "hash",
    bytes_valor: int = 4,
    limiares_por_dim: int = 0,
    n_dims: int = 0,
    bytes_limiar: int = 4,
    mapeamento_por_seed: bool = False,
    rotulo: str = "n-tuple",
) -> CustoMemoria:
    """Custo do ator n-tuple.

    `limiares_por_dim` é `T` quando o codificador é E1/E2/E3 (limiares
    ajustados aos dados) e **0** quando é E0 (binning uniforme, reconstruível
    de dois números). A SPEC-01 escolheu E2, então este item conta.
    """
    n_cabecas = pesos.shape[0]
    occ_ram = ocupacao_por_ram(pesos)

    luts = _bytes_luts(occ_ram, n_ram, tamanho_tupla, n_cabecas, bytes_valor, variante)

    # Mapeamento das tuplas: N × n índices de bit. Mais de 20 % do total numa
    # configuração típica — omitir infla a vantagem.
    b_idx = _bytes_indice(n_bits)
    mapeamento = n_ram * tamanho_tupla * b_idx
    mapeamento_persistente = 4 if mapeamento_por_seed else mapeamento

    limiares = n_dims * limiares_por_dim * bytes_limiar
    distribuicao = 8  # sigma^2 da política, hiperparâmetro

    persistente = luts + mapeamento_persistente + limiares + distribuicao

    # Residente inclui os buffers de trabalho e o mapeamento SEMPRE expandido.
    buf_bits = math.ceil(n_bits / 8)
    buf_end = n_ram * 4
    buf_acum = n_cabecas * 8
    residente = luts + mapeamento + limiares + distribuicao + buf_bits + buf_end + buf_acum

    return CustoMemoria(
        rotulo=rotulo,
        persistente=persistente,
        residente=residente,
        detalhe={
            "luts": luts,
            "mapeamento_tuplas": mapeamento_persistente,
            "limiares_codificador": limiares,
            "params_distribuicao": distribuicao,
            "buffer_bits": buf_bits,
            "buffer_enderecos": buf_end,
            "buffer_acumuladores": buf_acum,
        },
    )


def custo_mlp(
    camadas: list[int],
    bytes_peso: int = 4,
    rotulo: str = "MLP",
) -> CustoMemoria:
    """Custo de uma MLP densa. `camadas = [D, 32, 32, d]`.

    Conta pesos **e vieses**, por camada. `D` varia com `Na`
    (`D = 2(Na−1) + 25`), então a MLP pareada tem tamanho diferente por célula
    e a razão precisa vir sempre com a configuração ao lado (RNF-06).
    """
    total = 0
    detalhe = {}
    for i, (entrada, saida) in enumerate(zip(camadas[:-1], camadas[1:])):
        n = entrada * saida + saida
        detalhe[f"camada_{i}"] = n * bytes_peso
        total += n * bytes_peso

    # Residente: pesos + ativações intermediárias (a maior camada, duas vezes)
    maior = max(camadas[1:-1]) if len(camadas) > 2 else camadas[-1]
    buffers = 2 * maior * 8
    return CustoMemoria(
        rotulo=rotulo,
        persistente=total,
        residente=total + buffers,
        detalhe={**detalhe, "buffer_ativacoes": buffers},
    )


def ocupacao_esparsa(memoria: list[dict], n_ram: int) -> NDArray[np.int64]:
    """Ocupação por RAM lida direto da memória **esparsa** — SPEC-05 T-05.7.

    O `ocupacao_por_ram` acima exige o array denso `(cabeças, N, 2^n)`, e a
    SPEC-03 mediu que ele é **impossível de alocar** para tuplas de 16 a 64:
    `2^64` posições não existem. As redes da SPEC-04 guardam
    `list[dict[endereço, valor]]` exatamente por isso, e esta função é a ponte —
    a mesma grandeza, sem materializar o que não cabe.

    Uma posição conta **uma vez**, não uma por cabeça: a decisão de arquitetura
    da SPEC-05 §4.8 é um discriminador com posições **vetoriais** (Katopodis fig.
    4.3a), então um endereço guarda `d` valores sob uma chave só. Contar por
    cabeça superestimaria o realizado em ~2 chaves por entrada.
    """
    if len(memoria) != n_ram:
        raise ValueError(f"memória com {len(memoria)} RAMs, esperado {n_ram}")
    return np.array([len(m) for m in memoria], dtype=np.int64)


def custo_ntuple_de_rede(
    rede,
    variante: Variante = "hash",
    bytes_valor: int = 4,
    limiares_por_dim: int = 0,
    n_dims: int = 0,
    bytes_limiar: int = 4,
    mapeamento_por_seed: bool = False,
    rotulo: str = "n-tuple",
) -> CustoMemoria:
    """`custo_ntuple` aplicado a uma rede da SPEC-04 — a ligação de T-05.7.

    Existe para que `Aproximador.custo()` devolva os números **do harness da
    SPEC-03**, e não uma segunda contagem paralela que pode divergir dela em
    silêncio. Toda a aritmética de bytes continua a viver em `_bytes_luts`; aqui
    só se lê a ocupação da estrutura esparsa.
    """
    occ = ocupacao_esparsa(rede.memoria, rede.n_ram)
    # O contador é `uint32` (4 B) e existe só na rede com esquecimento; a
    # presença é detectada pelo atributo `phi`, que só ela tem.
    b_cont = 4 if hasattr(rede, "phi") else 0
    luts = _bytes_luts(occ, rede.n_ram, rede.tamanho_tupla, rede.n_saidas,
                       bytes_valor, variante, bytes_contador=b_cont)

    b_idx = _bytes_indice(rede.n_bits)
    mapeamento = rede.n_ram * rede.tamanho_tupla * b_idx
    mapeamento_persistente = 4 if mapeamento_por_seed else mapeamento
    limiares = n_dims * limiares_por_dim * bytes_limiar
    distribuicao = 8

    buf_bits = math.ceil(rede.n_bits / 8)
    buf_end = rede.n_ram * 4
    buf_acum = rede.n_saidas * 8

    return CustoMemoria(
        rotulo=rotulo,
        persistente=luts + mapeamento_persistente + limiares + distribuicao,
        residente=(luts + mapeamento + limiares + distribuicao
                   + buf_bits + buf_end + buf_acum),
        detalhe={
            "luts": luts,
            "mapeamento_tuplas": mapeamento_persistente,
            "limiares_codificador": limiares,
            "params_distribuicao": distribuicao,
            "buffer_bits": buf_bits,
            "buffer_enderecos": buf_end,
            "buffer_acumuladores": buf_acum,
        },
    )
