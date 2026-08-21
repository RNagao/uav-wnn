"""Converte a tabela treinada pelo PPO/RLlib para o nosso formato NumPy.

O segundo passo do par. O primeiro, `spec06_ppo_anbubus.py`, roda num ambiente
efêmero com `ray` e `torch` e grava arrays crus; este roda no ambiente do projeto
e produz o checkpoint que `spec06_campanha_wnn.py` consome.

    uv run python scripts/spec06_importar_ppo.py bruto.npz reports/ckpt/ppo-a2s12

POR QUE DOIS PASSOS, E POR QUE ISSO É BOM

`ray[rllib]` fixa `gymnasium==1.2.2` no máximo e o projeto exige `>=1.3.0`: são
incompatíveis no mesmo ambiente. Medido, não suposto.

O incômodo coincide com o que a **RNF-11** já exigia: torch no treino, NumPy puro
na medição — porque latência de rede pequena via PyTorch mede despacho, não modelo.
Aqui a fronteira é um arquivo, o que é mais defensável que uma flag.

A CONVERSÃO SÓ É LEGÍTIMA POR CAUSA DE UM TESTE

`tests/test_anbubus.py` prova que a `EmbeddingBag(mode='sum')` deles e o nosso
`RedeGradienteFuncional(convencao="soma")` são **numericamente a mesma coisa**:
endereçamento idêntico **exato** (são inteiros) e saída a 2·10⁻⁵ (float32 contra
float64). Sem esse teste, isto seria um salto de fé com cara de engenharia.

O QUE A CONVERSÃO PRESERVA, E O QUE ELA DESCARTA

    preserva   o mapeamento (gravado, nunca re-sorteado), os valores das posições
               escritas, o tamanho de tupla e a faixa do termômetro
    descarta   as posições que ficaram em ZERO — que é a esparsificação, e é
               exatamente o achado que a SPEC-03 mediu: o `anbubus` aloca
               `n_nodes · 2^n` densamente, e a fração de fato usada é pequena

Descartar zeros é seguro porque a condição inicial da §4.3.1 é `v[i]_0(x) = 0`:
posição ausente **vale** zero na nossa estrutura. O relatório ganha de graça a
razão ocupado/alocado, que é o número de T1 que distingue as duas estratégias de
armazenamento.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uavwnn.wnn.redes import RedeGradienteFuncional


def _rede_de(peso: np.ndarray, mapeamento: np.ndarray, tupla: int,
             n_bits: int) -> tuple[RedeGradienteFuncional, int]:
    """Monta a nossa rede a partir de uma `EmbeddingBag` achatada.

    O `peso` é `(n_ram · 2^n, d)` com as RAMs **empilhadas** na primeira dimensão
    — é o `mem_offsets = 2^n · arange(n_nodes)` do `NTNModel`. Desempilhar é
    fatiar em blocos de `2^n`.
    """
    d = int(peso.shape[1])
    rede = RedeGradienteFuncional(n_bits, tupla, n_saidas=d, seed=0, c_max=None,
                                  convencao="soma")
    # Mapeamento DELES, fatiado como as nossas tuplas. Nunca re-sorteado: o
    # `torch.randperm` não coincide com o do NumPy, e uma rede cujos endereços não
    # correspondem aos valores falha em silêncio.
    #
    # Via `remapear` e não por atribuição direta: a rede mantém matrizes de
    # endereçamento pré-computadas, e trocar `mapeamento` sem remontá-las deixaria o
    # endereçamento usando o mapeamento ANTIGO — em silêncio. Foi o que seis testes
    # pegaram quando a vetorização entrou.
    rede.remapear([mapeamento[i:i + tupla]
                   for i in range(0, len(mapeamento), tupla)])

    base = 2 ** tupla
    esperado = rede.n_ram * base
    if peso.shape[0] != esperado:
        raise ValueError(
            f"a tabela tem {peso.shape[0]} posições, mas {rede.n_ram} RAMs × "
            f"2^{tupla} = {esperado}. O mapeamento e os pesos não vêm do mesmo "
            f"modelo")

    rede.memoria = [{} for _ in range(rede.n_ram)]
    ocupadas = 0
    for r in range(rede.n_ram):
        bloco = peso[r * base:(r + 1) * base]
        for e in np.flatnonzero(np.abs(bloco).sum(axis=1) > 0):
            rede.memoria[r][int(e)] = bloco[e].astype(np.float64).copy()
            ocupadas += 1
    return rede, ocupadas


def _verificar(rede: RedeGradienteFuncional, peso: np.ndarray, tupla: int,
               n_amostras: int = 200, seed: int = 0) -> float:
    """Reconstrói a saída deles **em NumPy** e compara com a nossa.

    Não é redundante com o teste diferencial: aquele prova que as duas
    implementações concordam em geral; isto prova que **esta conversão** não
    perdeu nada — mapeamento trocado, bloco deslocado, dtype errado.

    Refaz o caminho do `NTNModel` sem torch: endereço por `Σ bit_k·2^k`, deslocado
    por `2^n · r`, soma sobre as RAMs.
    """
    rng = np.random.default_rng(seed)
    bits = rng.random((n_amostras, rede.n_bits)) > 0.5
    base = 2 ** tupla

    nosso = rede.prever(bits)
    deles = np.zeros_like(nosso)
    for m in range(n_amostras):
        for r, tup in enumerate(rede.mapeamento):
            end = int((bits[m, tup].astype(np.uint64)
                       * (np.uint64(1) << np.arange(len(tup), dtype=np.uint64))).sum())
            deles[m] += peso[r * base + end]
    return float(np.abs(nosso - deles).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bruto", help=".npz gravado por spec06_ppo_anbubus.py --bruto")
    ap.add_argument("destino", help="diretório do checkpoint (para a campanha)")
    ap.add_argument("--tolerancia", type=float, default=1e-5,
                    help="desvio máximo aceito na verificação da conversão")
    a = ap.parse_args()

    z = np.load(a.bruto, allow_pickle=False)
    tupla = int(z["tamanho_tupla"])
    n_bits = int(z["n_bits"])
    mapeamento = z["mapeamento"].astype(np.int64)
    celula = tuple(int(x) for x in z["celula"])
    faixa = tuple(float(x) for x in z["faixa"])
    resolucao = int(z["resolucao"])

    destino = Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    info: dict = {"origem": str(a.bruto), "celula": list(celula),
                  "tamanho_tupla": tupla, "n_bits": n_bits,
                  "faixa_obs": list(faixa), "resolucao": resolucao,
                  "treinador": "PPO (Ray RLlib)", "modelo": "NTNModel",
                  "convencao": "soma"}

    print(f"convertendo · célula {celula[0]}a{celula[1]}s · {n_bits} bits · "
          f"tupla {tupla}")
    for nome, chave in (("ator", "peso_ator"), ("critico", "peso_critico")):
        peso = z[chave]
        rede, ocupadas = _rede_de(peso, mapeamento, tupla, n_bits)
        desvio = _verificar(rede, peso, tupla)
        if desvio > a.tolerancia:
            sys.exit(f"ERRO: conversão do {nome} divergiu por {desvio:.3e} "
                     f"(tolerância {a.tolerancia:.0e}). O checkpoint NÃO foi "
                     f"gravado — um modelo silenciosamente errado é pior que "
                     f"nenhum modelo.")
        rede.salvar(destino / f"ntupla-{nome}.npz")
        alocadas = int(peso.shape[0])
        info[nome] = {"n_saidas": int(peso.shape[1]), "n_ram": rede.n_ram,
                      "ocupadas": ocupadas, "alocadas": alocadas,
                      "fracao_usada": ocupadas / alocadas,
                      "desvio_max_vs_torch": desvio}
        print(f"  {nome:>7}: {ocupadas:>7}/{alocadas:>7} posições "
              f"({ocupadas / alocadas:>6.2%} da alocação densa)  "
              f"desvio {desvio:.2e}")

    # O termômetro deles é E0 — bins uniformes na faixa medida. Grava-se para o
    # checkpoint ser autocontido, no formato que `TermometroQuantil.carregar` lê.
    lim = np.linspace(faixa[0], faixa[1], resolucao, endpoint=False)
    d_obs = n_bits // resolucao
    np.savez_compressed(destino / "termometro.npz",
                        limiares=np.tile(lim, (d_obs, 1)),
                        n_niveis=np.array(resolucao))
    (destino / "familia.txt").write_text("n-tuple-funcional (PPO/RLlib, anbubus)")
    (destino / "procedencia.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False))

    print(f"\ncheckpoint: {destino}")
    print(f"  use com: uv run python scripts/spec06_campanha_wnn.py "
          f"--aproximador ntupla_funcional --checkpoint {destino} "
          f"--celulas {celula[0]}:{celula[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
