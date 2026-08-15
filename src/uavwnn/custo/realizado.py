"""T1 — validação cruzada: constrói as estruturas de verdade e MEDE.

`memoria.py` calcula o footprint por **fórmula**. Uma fórmula que ninguém
confere contra a realidade é uma estimativa com cara de medição, e o veredito
de T1 (WNN 17x maior que a MLP pareada) é forte demais para descansar nisso.

Aqui as três variantes de armazenamento são **materializadas**: chaves, valores,
bitmaps e cabeçalhos viram `bytes` de verdade, serializados num blob plano sem
`pickle` — o que um firmware gravaria em flash. O tamanho medido é
`len(blob)`, não uma conta.

E o passo que dá valor à medição: cada estrutura **responde consultas**. Antes
de medir, `verificar()` compara a predição da estrutura esparsa com a do
`NTupleRegressor` denso. Uma estrutura que ocupa poucos bytes porque perdeu
entradas seria detectada aqui, não na tabela final.

Diferença esperada contra o denso: os valores são gravados em `float32` e o
regressor treina em `float64`. A verificação usa tolerância de `float32`, o que
de quebra valida a premissa `bytes_valor=4` da fórmula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .memoria import _bytes_indice, ocupacao_por_ram


@dataclass
class MedidaRealizada:
    """Bytes efetivamente medidos, mais o resultado da verificação funcional."""

    rotulo: str
    persistente: int
    """`len(blob)` — o que iria para a flash."""

    residente: int
    """Soma dos `nbytes` das estruturas vivas na inferência, mais buffers.

    Exclui o overhead de objeto do Python (cabeçalho de `ndarray`, refcount):
    numa implementação em C ele não existiria, e incluí-lo mediria o CPython e
    não o modelo.
    """

    detalhe: dict[str, int]
    erro_max_vs_denso: float
    """Maior desvio absoluto entre a predição da estrutura e a do denso."""


def _hash_pos(chave: int, cap: int) -> int:
    """Multiplicativo de Knuth, mascarado — `cap` é potência de 2."""
    return (chave * 2654435761) & (cap - 1)


class LUTOrdenada:
    """V-b: chaves ordenadas por RAM + busca binária.

    Sem folga de carga e com cauda determinística `O(log occ)`. A SPEC-03 previu
    que ganha do hash nas duas dimensões; esta classe é o que torna a previsão
    verificável.
    """

    variante = "ordenado"

    def __init__(self, pesos: NDArray, bytes_valor: int = 4) -> None:
        n_cab, n_ram, _ = pesos.shape
        self.n_cab, self.n_ram = n_cab, n_ram
        tocado = (pesos != 0).any(axis=0)
        self.b_chave = 2 if pesos.shape[2] <= (1 << 16) else 4
        dt_chave = np.uint16 if self.b_chave == 2 else np.uint32

        chaves, valores, self.offsets = [], [], np.zeros(n_ram + 1, dtype=np.uint32)
        for i in range(n_ram):
            (k,) = np.nonzero(tocado[i])          # np.nonzero já devolve ordenado
            chaves.append(k.astype(dt_chave))
            valores.append(pesos[:, i, k].T.astype(np.float32))
            self.offsets[i + 1] = self.offsets[i] + len(k)

        self.chaves = np.concatenate(chaves) if chaves else np.zeros(0, dt_chave)
        self.valores = (np.concatenate(valores) if valores
                        else np.zeros((0, n_cab), np.float32))

    def consultar(self, ram: int, chave: int) -> NDArray:
        ini, fim = int(self.offsets[ram]), int(self.offsets[ram + 1])
        j = int(np.searchsorted(self.chaves[ini:fim], chave))
        if j < fim - ini and self.chaves[ini + j] == chave:
            return self.valores[ini + j]
        return np.zeros(self.n_cab, np.float32)    # ausente = nunca escrita = 0

    def blob(self) -> bytes:
        # offsets guarda n_ram+1 entradas por conveniência de indexação; o
        # último é redundante (é o total), então a fórmula conta n_ram e o blob
        # gravado também.
        return (self.offsets[:-1].tobytes() + self.chaves.tobytes()
                + self.valores.tobytes())

    def nbytes(self) -> int:
        return (self.offsets[:-1].nbytes + self.chaves.nbytes
                + self.valores.nbytes)


class LUTHash:
    """V-a: hash aberto por RAM, sondagem linear.

    Paga fator de carga **e** arredondamento da capacidade para potência de 2.
    """

    variante = "hash"

    def __init__(self, pesos: NDArray, alpha: float = 0.75) -> None:
        n_cab, n_ram, _ = pesos.shape
        self.n_cab, self.n_ram, self.alpha = n_cab, n_ram, alpha
        tocado = (pesos != 0).any(axis=0)
        self.b_chave = 2 if pesos.shape[2] <= (1 << 16) else 4
        dt_chave = np.uint16 if self.b_chave == 2 else np.uint32

        self.caps, self.tab_chave, self.tab_valor, self.bitmaps = [], [], [], []
        for i in range(n_ram):
            (ks,) = np.nonzero(tocado[i])
            oc = len(ks)
            cap = 1 << math.ceil(math.log2(max(1, oc / alpha))) if oc else 0
            self.caps.append(cap)
            tc = np.zeros(cap, dt_chave)
            tv = np.zeros((cap, n_cab), np.float32)
            bm = np.zeros(math.ceil(cap / 8) if cap else 0, np.uint8)
            for k in ks:                            # inserção com sondagem linear
                p = _hash_pos(int(k), cap)
                while bm[p >> 3] >> (p & 7) & 1:
                    p = (p + 1) & (cap - 1)
                tc[p] = k
                tv[p] = pesos[:, i, k].astype(np.float32)
                bm[p >> 3] |= 1 << (p & 7)
            self.tab_chave.append(tc)
            self.tab_valor.append(tv)
            self.bitmaps.append(bm)

    def consultar(self, ram: int, chave: int) -> NDArray:
        cap = self.caps[ram]
        if cap == 0:
            return np.zeros(self.n_cab, np.float32)
        tc, bm = self.tab_chave[ram], self.bitmaps[ram]
        p = _hash_pos(chave, cap)
        while bm[p >> 3] >> (p & 7) & 1:
            if tc[p] == chave:
                return self.tab_valor[ram][p]
            p = (p + 1) & (cap - 1)
        return np.zeros(self.n_cab, np.float32)

    def blob(self) -> bytes:
        partes = [np.asarray(self.caps, np.uint32).tobytes()]   # cabeçalhos
        for tc, tv, bm in zip(self.tab_chave, self.tab_valor, self.bitmaps):
            partes += [tc.tobytes(), tv.tobytes(), bm.tobytes()]
        return b"".join(partes)

    def nbytes(self) -> int:
        return len(self.blob())


class LUTDensa:
    """V-c: o array inteiro. Previsibilidade máxima, memória máxima."""

    variante = "denso"

    def __init__(self, pesos: NDArray) -> None:
        self.n_cab = pesos.shape[0]
        self.tabela = pesos.astype(np.float32)

    def consultar(self, ram: int, chave: int) -> NDArray:
        return self.tabela[:, ram, chave]

    def blob(self) -> bytes:
        return self.tabela.tobytes()

    def nbytes(self) -> int:
        return self.tabela.nbytes


_LUTS = {"ordenado": LUTOrdenada, "hash": LUTHash, "denso": LUTDensa}


def medir_realizado(
    net,
    variante: str = "ordenado",
    limiares_por_dim: int = 0,
    n_dims: int = 0,
    bytes_limiar: int = 4,
    bits_verificacao: NDArray | None = None,
    rotulo: str = "",
) -> MedidaRealizada:
    """Materializa, verifica contra o denso, e mede.

    `bits_verificacao` são observações binarizadas de verdade; sem elas a
    verificação é pulada e `erro_max_vs_denso` volta `nan` — o que a tabela do
    relatório precisa mostrar como lacuna, não esconder.
    """
    lut = _LUTS[variante](net.pesos)

    # -- verificação funcional: a estrutura reproduz o denso? ----------------
    erro = float("nan")
    if bits_verificacao is not None and len(bits_verificacao):
        ends = net.enderecos(bits_verificacao)
        esperado = net.prever(bits_verificacao)
        obtido = np.empty_like(esperado)
        for m, linha in enumerate(ends):
            acum = np.zeros(net.n_saidas, np.float64)
            for i, e in enumerate(linha):
                acum += lut.consultar(i, int(e))
            obtido[m] = acum / net.n_ram
        erro = float(np.abs(obtido - esperado).max())
        # float32 nos valores, float64 no denso: o desvio tolerado é o do
        # arredondamento, e nada além dele.
        tol = 1e-6 * max(1.0, float(np.abs(esperado).max()))
        if erro > tol:
            raise AssertionError(
                f"{variante}: estrutura diverge do denso em {erro:.3e} "
                f"(tolerância {tol:.3e}) — a medição de tamanho não vale"
            )

    # -- partes comuns às três variantes ------------------------------------
    b_idx = _bytes_indice(net.n_bits)
    mapeamento = np.asarray(net.mapeamento,
                            np.uint8 if b_idx == 1 else np.uint16).tobytes()
    limiares = np.zeros(n_dims * limiares_por_dim,
                        np.float32 if bytes_limiar == 4 else np.float64).tobytes()
    distribuicao = np.zeros(2, np.float32).tobytes()   # sigma^2 da política

    blob = lut.blob() + mapeamento + limiares + distribuicao

    buf_bits = math.ceil(net.n_bits / 8)
    buf_end = net.n_ram * 4
    buf_acum = net.n_saidas * 8

    return MedidaRealizada(
        rotulo=rotulo or f"WNN {variante} (medido)",
        persistente=len(blob),
        residente=lut.nbytes() + len(mapeamento) + len(limiares)
        + len(distribuicao) + buf_bits + buf_end + buf_acum,
        detalhe={
            "luts": len(lut.blob()),
            "mapeamento_tuplas": len(mapeamento),
            "limiares_codificador": len(limiares),
            "params_distribuicao": len(distribuicao),
            "buffer_bits": buf_bits,
            "buffer_enderecos": buf_end,
            "buffer_acumuladores": buf_acum,
        },
        erro_max_vs_denso=erro,
    )


def medir_mlp(camadas: list[int], rotulo: str = "MLP (medido)") -> MedidaRealizada:
    """A MLP pareada, materializada em `float32` e serializada.

    A comparação só é honesta se os dois lados forem medidos pela mesma régua:
    a WNN grava valores `float32`, então a MLP também.
    """
    detalhe, partes = {}, []
    rng = np.random.default_rng(0)
    for i, (ent, sai) in enumerate(zip(camadas[:-1], camadas[1:])):
        W = rng.normal(size=(ent, sai)).astype(np.float32)
        b = rng.normal(size=sai).astype(np.float32)
        partes += [W.tobytes(), b.tobytes()]
        detalhe[f"camada_{i}"] = W.nbytes + b.nbytes
    blob = b"".join(partes)
    maior = max(camadas[1:-1]) if len(camadas) > 2 else camadas[-1]
    buffers = 2 * maior * 8
    return MedidaRealizada(
        rotulo=rotulo,
        persistente=len(blob),
        residente=len(blob) + buffers,
        detalhe={**detalhe, "buffer_ativacoes": buffers},
        erro_max_vs_denso=0.0,
    )
