"""T3 — contagem de operações por decisão (SPEC-03, T-03.7).

A figura **independente de hardware**, e onde a diferença entre os dois
aproximadores é estrutural em vez de incidental. Tempo de parede depende de
implementação; contagem de operações não.

A alegação verificável do PRD §5.5 é precisa: **zero multiplicações em ponto
flutuante no aproximador**. Não "zero MACs no caminho de inferência", que é
falso como escrito — a acumulação usa somas em ponto flutuante e a amostragem
gaussiana usa transcendentais. Por isso as categorias abaixo são separadas, e
o codificador e o amostrador aparecem em linhas próprias.

Categoria à parte para **divisões**: num microcontrolador sem FPU uma divisão
custa mais que uma multiplicação, e o KF da Etapa 2 usa uma por ciclo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CustoOperacoes:
    """Operações por decisão, por tipo. Sempre inteiros."""

    mult_fp: int = 0
    soma_fp: int = 0
    div_fp: int = 0
    transcendental_fp: int = 0
    """`exp`, `log`, `sqrt`, `sin`, `cos`, `atan2` — cada uma dezenas de ciclos."""

    comparacoes: int = 0
    """Onde o termômetro vive: `v >= limiar`, uma por bit."""

    lookups: int = 0
    soma_int: int = 0
    op_bits: int = 0

    def __add__(self, outro: "CustoOperacoes") -> "CustoOperacoes":
        return CustoOperacoes(**{
            k: getattr(self, k) + getattr(outro, k) for k in asdict(self)
        })

    @property
    def total(self) -> int:
        return sum(asdict(self).values())

    def como_linha(self) -> dict[str, int]:
        return {**asdict(self), "total": self.total}


def operacoes_codificador(n_dims: int, n_niveis: int) -> CustoOperacoes:
    """Termômetro: uma comparação por bit de saída.

    Isto é do **codificador**, não da rede. A SPEC-05 mediu que ele domina o
    lado WNN na contagem total — parte da vantagem é dele, e o relatório
    precisa dizer isso em vez de creditar tudo ao aproximador.
    """
    return CustoOperacoes(comparacoes=n_dims * n_niveis)


def operacoes_ntuple(
    n_ram: int,
    tamanho_tupla: int,
    n_cabecas: int,
    n_dims: int = 0,
    n_niveis: int = 0,
    incluir_codificador: bool = True,
    incluir_amostragem: bool = True,
) -> CustoOperacoes:
    """Inferência da rede n-tuple.

    Endereçamento: cada RAM monta um endereço de `n` bits — deslocamentos e
    ORs, mais uma soma inteira. Acumulação: `N` somas em ponto flutuante por
    cabeça, e **uma divisão** por cabeça pela convenção de média.

    **Nenhuma multiplicação em ponto flutuante.** É a alegação central, e ela
    aparece aqui como zero literal.
    """
    end = CustoOperacoes(
        op_bits=n_ram * tamanho_tupla * 2,   # shift + or por bit
        soma_int=n_ram * tamanho_tupla,
        lookups=n_ram * n_cabecas,
    )
    acum = CustoOperacoes(
        soma_fp=n_ram * n_cabecas,
        div_fp=n_cabecas,                    # a média da convenção
    )
    total = end + acum
    if incluir_codificador:
        total = total + operacoes_codificador(n_dims, n_niveis)
    if incluir_amostragem:
        total = total + operacoes_amostragem()
    return total


def operacoes_mlp(
    camadas: list[int],
    incluir_amostragem: bool = True,
) -> CustoOperacoes:
    """Inferência de MLP densa com ReLU.

    Uma multiplicação e uma soma em ponto flutuante por peso, mais o viés, mais
    uma comparação por unidade oculta (o ReLU).
    """
    mult = soma = comp = 0
    for i, (entrada, saida) in enumerate(zip(camadas[:-1], camadas[1:])):
        mult += entrada * saida
        soma += entrada * saida + saida
        if i < len(camadas) - 2:
            comp += saida                    # ReLU
    base = CustoOperacoes(mult_fp=mult, soma_fp=soma, comparacoes=comp)
    return base + operacoes_amostragem() if incluir_amostragem else base


def operacoes_amostragem() -> CustoOperacoes:
    """Amostragem da ação: normal projetada mais escala da velocidade.

    Box–Muller para duas gaussianas (`sqrt`, `log`, `sin`, `cos`) e um `atan2`
    para recuperar o rumo. **Comum aos dois aproximadores**, então não entra na
    diferença — mas precisa aparecer, senão a soma não fecha e a alegação de
    "sem transcendentais" ficaria falsa.
    """
    return CustoOperacoes(
        transcendental_fp=5,
        mult_fp=4,
        soma_fp=2,
    )


# Ciclos por operação, por classe de alvo (T-03.8). Ordens de grandeza de
# manuais de arquitetura, não medições nossas — servem para mostrar que a
# vantagem da WNN é CONDICIONAL à classe de hardware, não absoluta.
CICLOS = {
    "x86-64 (com FPU e SIMD)": dict(
        mult_fp=1, soma_fp=1, div_fp=14, transcendental_fp=40,
        comparacoes=1, lookups=4, soma_int=1, op_bits=1),
    "Cortex-M4F (com FPU)": dict(
        mult_fp=3, soma_fp=3, div_fp=14, transcendental_fp=100,
        comparacoes=1, lookups=2, soma_int=1, op_bits=1),
    "Cortex-M0 (SEM FPU)": dict(
        mult_fp=70, soma_fp=55, div_fp=250, transcendental_fp=800,
        comparacoes=1, lookups=2, soma_int=1, op_bits=1),
}


def ciclos(custo: "CustoOperacoes", alvo: str) -> int:
    """Ciclos estimados de uma decisão, no alvo dado."""
    tabela = CICLOS[alvo]
    return sum(getattr(custo, k) * c for k, c in tabela.items())
