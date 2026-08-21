"""Bancos de episódios `test` e `val` — §4.2, e o horizonte permissivo (P5).

    test   500 episódios (Lamenza §7.2)   métrica final. NUNCA usado para
                                          selecionar checkpoint, hiperparâmetro
                                          ou configuração
    val    100 episódios, DISJUNTOS       avaliação intercalada, parada
                                          antecipada, seleção, varreduras

Lamenza avalia **durante** o treino e reporta o modelo carregado dessas avaliações
(§7, p36) — o que é, tecnicamente, seleção sobre o conjunto de reporte. Nós
separamos e **dizemos que separamos**: é diferença de protocolo a favor do rigor,
não do resultado.

O QUE É UM "EPISÓDIO" NO BANCO, E POR QUE É SÓ UMA SEMENTE

O `reset()` do ambiente de referência sorteia posições com o `random` da stdlib e
**ignora** o argumento `seed` que sua assinatura declara — a SPEC-00 mediu isso. A
única forma de fixar um cenário sem tocar no ambiente é `random.seed(s)` imediatamente
antes do `reset()`.

Então um episódio do banco **é** um inteiro, e o banco é uma lista de inteiros. Isso
tem uma vantagem: o pareamento da §4.2 (todo método vê os mesmos 500 cenários, na
mesma ordem) sai de graça, e o hash do banco é o hash da lista.

A disjunção `test ∩ val = ∅` é verificada por teste automatizado, porque uma
interseção silenciosa transformaria a seleção de checkpoint em seleção sobre o
conjunto de reporte — exatamente o que separamos para evitar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_TEST = 500
"""§7.2 de Lamenza."""
N_VAL = 100
"""§4.2. Menor que `test` porque roda a cada avaliação intercalada — 40 vezes por
corrida — e o custo entra no `B_max`."""

RAIZ_BANCOS = Path("runs/bancos")


@dataclass(frozen=True)
class Banco:
    """Uma lista de sementes de cenário, com hash."""

    nome: str
    na: int
    ns: int
    sementes: tuple[int, ...]

    @property
    def hash(self) -> str:
        """Identifica o banco em cada corrida (§4.7, `env.json`).

        Sobre a **lista ordenada**, não sobre o conjunto: a ordem faz parte do
        pareamento, e dois bancos com os mesmos cenários em ordem diferente dariam
        o mesmo IC pareado mas curvas de avaliação intercalada diferentes.
        """
        crua = json.dumps({"nome": self.nome, "na": self.na, "ns": self.ns,
                           "sementes": list(self.sementes)}, sort_keys=True)
        return hashlib.sha256(crua.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.sementes)


@dataclass(frozen=True)
class BancoPar:
    """`test` e `val` de uma célula, com a disjunção garantida na construção."""

    test: Banco
    val: Banco

    def __post_init__(self) -> None:
        comum = set(self.test.sementes) & set(self.val.sementes)
        if comum:
            raise ValueError(
                f"test ∩ val = {sorted(comum)[:5]}… — a disjunção da §4.2 é o que "
                f"impede que a seleção de checkpoint vire seleção sobre o conjunto "
                f"de reporte")

    def como_dict(self) -> dict:
        return {"test": {"n": len(self.test), "hash": self.test.hash},
                "val": {"n": len(self.val), "hash": self.val.hash}}


def gerar_bancos(na: int, ns: int, raiz_eval: int = 20260901,
                 n_test: int = N_TEST, n_val: int = N_VAL) -> BancoPar:
    """Gera o par para uma célula, de forma determinística e disjunta.

    As sementes de cenário são `uint32`: o `reset()` de referência usa
    `random.seed`/`np.random.seed`, e o gerador legado do NumPy **exige** semente
    abaixo de `2³²`. A SPEC-02 pagou esse bug uma vez.

    `test` e `val` saem de faixas **separadas por construção** em vez de por
    filtragem: sortear e depois remover colisões daria bancos cujo tamanho depende
    do sorteio.
    """
    g = np.random.default_rng([raiz_eval, na, ns])
    # Faixas disjuntas: `test` no primeiro bilhão, `val` no segundo.
    test = g.choice(1_000_000_000, size=n_test, replace=False).astype(np.uint32)
    val = (g.choice(1_000_000_000, size=n_val, replace=False)
           + 1_000_000_000).astype(np.uint32)
    return BancoPar(
        test=Banco("test", na, ns, tuple(int(x) for x in test)),
        val=Banco("val", na, ns, tuple(int(x) for x in val)),
    )


def carregar_bancos(na: int, ns: int, raiz: Path | None = None,
                    **kw) -> BancoPar:
    """Do disco se existir, senão gera e grava.

    Gravar importa: os bancos são **versionados** (§4.2), e um banco regenerado por
    uma versão diferente do NumPy daria cenários diferentes com a mesma semente —
    e as comparações já registradas deixariam de valer sem nenhum erro aparecer.
    """
    dir_ = (raiz or RAIZ_BANCOS)
    arq = dir_ / f"bancos-a{na}s{ns}.json"
    if arq.exists():
        d = json.loads(arq.read_text())
        par = BancoPar(
            test=Banco("test", na, ns, tuple(d["test"]["sementes"])),
            val=Banco("val", na, ns, tuple(d["val"]["sementes"])),
        )
        for nome, banco in (("test", par.test), ("val", par.val)):
            if banco.hash != d[nome]["hash"]:
                raise ValueError(
                    f"hash do banco {nome} de a{na}s{ns} não confere: "
                    f"{banco.hash} != {d[nome]['hash']}. O arquivo foi editado ou "
                    f"gerado por outra versão — não use.")
        return par

    par = gerar_bancos(na, ns, **kw)
    dir_.mkdir(parents=True, exist_ok=True)
    arq.write_text(json.dumps({
        "na": na, "ns": ns,
        "test": {"n": len(par.test), "hash": par.test.hash,
                 "sementes": list(par.test.sementes)},
        "val": {"n": len(par.val), "hash": par.val.hash,
                "sementes": list(par.val.sementes)},
    }, indent=1))
    return par


# ---------------------------------------------------------------------------


def horizonte_permissivo(ct_guloso: np.ndarray, piso: float = 500.0,
                         teto: float = 2000.0, fator: float = 3.0) -> float:
    """P5 — `max_episode_length` do §7.1, por definição operacional.

    Lamenza não publica o valor; diz só que é *"permissivo"*. A T-06.1 fixa:

        max_episode_length_1a = 3 × CT_p99(guloso)

    arredondado para cima em múltiplos de 100 s, com piso de 500 s (o valor da
    tab. 7.1) e teto de 2 000 s (limite de custo).

    A justificativa é funcional e verificável: *"permissivo"* significa, no texto,
    que **o horizonte não é o gargalo** — e a verificação é que a taxa de sucesso do
    guloso sob esse horizonte seja ≥ 99 %. Se não for, sobe até ser.
    """
    p99 = float(np.percentile(np.asarray(ct_guloso, float), 99))
    bruto = fator * p99
    return float(min(teto, max(piso, np.ceil(bruto / 100.0) * 100.0)))
