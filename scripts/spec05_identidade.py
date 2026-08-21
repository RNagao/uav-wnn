"""T-05.5 — compartilhamento de parâmetros e o canal de identidade.

Uma rede n-tuple **só** para todos os UAVs (§4.7). A observação de Lamenza (eq.
5-1) carrega o índice `k` do agente, então em princípio a rede pode diferenciar
comportamento. A pergunta é se o canal **sobrevive à binarização**: o índice é
`Na` bits num padrão de `B ≈ 800`, e cada RAM lê só `n` deles.

TRÊS MEDIÇÕES, DAS QUAIS A PRIMEIRA CORRIGE A PRÓPRIA SPEC

1. Cobertura do canal: `p = 1 − (1 − R·Na/B)^n` teórico contra o medido.
2. Efeito de `R` sobre a diferenciação: dispersão dos rumos escolhidos por
   agentes que veem **a mesma observação exceto o índice**.
3. Determinismo da barreira: mesma semente, duas ordens de evento → tabelas
   idênticas bit a bit.

A FÓRMULA DA SPEC SUPÕE O MAPEAMENTO ERRADO

`p = 1 − (1 − R·Na/B)^n` é a probabilidade de uma RAM ler ao menos um bit de
índice **se cada RAM sorteasse seus `n` bits com reposição**. É o que o
`NTupleRegressor` do protótipo fazia — e foi a origem do achado **C4**, em que
alguns bits ficavam sem nenhuma RAM lendo.

O nosso `mapeamento_particionado` **particiona**: `randperm` fatiado, cobertura
1,0 por construção. Aí a contagem certa é **hipergeométrica**, não binomial:

    p_hiper = 1 − C(B − K, n) / C(B, n),   K = R·Na

Amostragem sem reposição contra com reposição. A diferença é pequena quando
`K, n ≪ B`, e o experimento mede o quanto — mas a fórmula da spec fica registrada
como aproximação, não como identidade.

O BRINQUEDO EXIGE DIFERENCIAÇÃO, EM VEZ DE SÓ PERMITI-LA

`Na` agentes, 4 sensores, **observação idêntica** para todos exceto o índice. O
rumo correto do agente `k` é o do sensor `k` — então a única forma de acertar é
**usar o índice**. Se o canal não passa pela binarização, o erro fica no patamar
de quem escolhe um rumo médio, e isso é mensurável sem ambiguidade.

Usa o bandido contextual de T-05.4(b), e não o `nav` sequencial: aquele tem
pendência aberta de crédito temporal que afetaria as duas condições e mascararia
o efeito de `R`.

Uso:
    uv run python scripts/spec05_identidade.py
    uv run python scripts/spec05_identidade.py --tentativas 40000 --sementes 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from uavwnn.agents.politica_continua import D_ATOR, NormalProjetada
from uavwnn.encoding.artigo import TermometroArtigo
from uavwnn.eval import ArvoreSementes, bootstrap_pareado
from uavwnn.wnn.redes import RedeGradienteFuncional, RedeRegressaoEsquecimento

TAMANHO = 200.0
N_SENSORES = 4
RESOLUCAO = 32
TAMANHO_TUPLA = 12
SEED_AVAL = 4242


def p_binomial(R: int, na: int, n_bits: int, tupla: int) -> float:
    """`p = 1 − (1 − R·Na/B)^n` — a fórmula da spec, com reposição."""
    return 1.0 - (1.0 - R * na / n_bits) ** tupla


def p_hipergeometrica(R: int, na: int, n_bits: int, tupla: int) -> float:
    """`1 − C(B−K, n)/C(B, n)` — sem reposição, que é o nosso mapeamento."""
    K = R * na
    if n_bits - K < tupla:
        return 1.0
    return 1.0 - math.comb(n_bits - K, tupla) / math.comb(n_bits, tupla)


class Codificador:
    """Termômetro nas coordenadas + índice one-hot replicado `R` vezes.

    A replicação é a mitigação (i) da §4.7: repetir o one-hot `R` vezes eleva a
    probabilidade de uma RAM qualquer ler o canal de `1−(1−Na/B)^n` para
    `1−(1−R·Na/B)^n`. Custa `R·Na` bits, que é barato perto dos ~800 do padrão.
    """

    def __init__(self, na: int, R: int) -> None:
        self.na, self.R = na, R
        self.termo = TermometroArtigo(-TAMANHO, TAMANHO, RESOLUCAO)
        self.bits_coord = 2 * N_SENSORES * self.termo.n_bits_para(1)
        self.bits_indice = R * na
        self.n_bits = self.bits_coord + self.bits_indice
        # Fatia onde o canal de identidade vive — usada na medição de cobertura.
        self.fatia_indice = np.arange(self.bits_coord, self.n_bits)

    def __call__(self, rel: NDArray, k: int) -> NDArray[np.bool_]:
        coord = np.concatenate([self.termo.codificar(np.array([v]))
                               for v in np.asarray(rel, float).ravel()]).ravel()
        idx = np.zeros(self.bits_indice, bool)
        for r in range(self.R):
            idx[r * self.na + k] = True
        return np.concatenate([coord, idx])


def cobertura_medida(cod: Codificador, seed: int) -> float:
    """Fração das RAMs que leem **ao menos um** bit do canal de identidade."""
    rede = RedeGradienteFuncional(cod.n_bits, TAMANHO_TUPLA, n_saidas=D_ATOR,
                                  seed=seed)
    alvo = set(cod.fatia_indice.tolist())
    tocam = sum(1 for tupla in rede.mapeamento if alvo & set(tupla.tolist()))
    return tocam / rede.n_ram


def treinar(na: int, R: int, tentativas: int, seed: int, sigma: float = 0.2,
            taxa: float = 2e-3) -> dict:
    """Uma rede compartilhada por `na` agentes. Devolve erro e dispersão."""
    cod = Codificador(na, R)
    ator = RedeGradienteFuncional(cod.n_bits, TAMANHO_TUPLA, n_saidas=D_ATOR,
                                  seed=seed, c_max=0.5)
    critico = RedeRegressaoEsquecimento(cod.n_bits, TAMANHO_TUPLA, n_saidas=1,
                                        phi=0.85, seed=seed)
    pol = NormalProjetada(sigma, None)
    rng = np.random.default_rng(seed)

    def cenario(g: np.random.Generator):
        """4 sensores em ângulos sorteados; **a mesma observação para todos**."""
        angs = g.uniform(0, 2 * np.pi, N_SENSORES)
        raios = g.uniform(60.0, 90.0, N_SENSORES)
        rel = np.stack([raios * np.cos(angs), raios * np.sin(angs)], axis=1)
        return rel.ravel(), angs

    for _ in range(tentativas):
        rel, angs = cenario(rng)
        for k in range(na):
            bits = cod(rel, k)
            f = ator.prever(bits[None, :])[0]
            acao, z = pol.amostrar(f, rng)
            # O rumo correto do agente k é o do sensor k: sem usar o índice,
            # nenhuma política acerta mais de uma das `na` respostas.
            r = float(np.cos(acao.theta_norm * 2 * np.pi - angs[k % N_SENSORES]))
            v = float(critico.prever(bits[None, :])[0, 0])
            ator.atualizar(bits, taxa * (r - v) * pol.grad_log(f, z))
            critico.treinar(bits[None, :], np.array([r]))

    # -- avaliação: erro por agente e dispersão entre agentes ---------------
    aval = np.random.default_rng(SEED_AVAL)
    erros, dispersoes = [], []
    for _ in range(200):
        rel, angs = cenario(aval)
        rumos = []
        for k in range(na):
            f = ator.prever(cod(rel, k)[None, :])[0]
            th = pol.media(f).theta_norm * 2 * np.pi
            rumos.append(th)
            alvo = angs[k % N_SENSORES]
            erros.append(abs(np.mod(th - alvo + np.pi, 2 * np.pi) - np.pi))
        # Dispersão circular entre agentes na MESMA observação: `1 − ‖média
        # vetorial‖`. Zero significa rumos idênticos — canal de identidade mudo.
        u = np.exp(1j * np.array(rumos))
        dispersoes.append(1.0 - float(abs(u.mean())))

    return {"erro_graus": float(np.degrees(np.mean(erros))),
            "erro_por_alvo": np.degrees(np.array(erros)).tolist(),
            "dispersao": float(np.mean(dispersoes)),
            "n_bits": cod.n_bits, "n_ram": ator.n_ram,
            "cobertura": cobertura_medida(cod, seed)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tentativas", type=int, default=15000)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--saida", default="reports/spec05-identidade.json")
    a = ap.parse_args()

    art: dict = {"config": vars(a), "cobertura": [], "diferenciacao": []}
    globals()["_art"] = art

    # -- 1. cobertura do canal ---------------------------------------------
    print("=" * 78)
    print("1. COBERTURA DO CANAL DE IDENTIDADE")
    print("=" * 78)
    print(f"{'Na':>3} {'R':>3} {'B':>5} {'N':>4} {'p binom':>9} {'p hiper':>9} "
          f"{'medido':>8} {'medido−hiper':>13}")
    for na in (2, 4):
        for R in (1, 4, 16):
            cod = Codificador(na, R)
            rede = RedeGradienteFuncional(cod.n_bits, TAMANHO_TUPLA, seed=0)
            pb = p_binomial(R, na, cod.n_bits, TAMANHO_TUPLA)
            ph = p_hipergeometrica(R, na, cod.n_bits, TAMANHO_TUPLA)
            med = float(np.mean([cobertura_medida(cod, s) for s in range(20)]))
            linha = {"na": na, "R": R, "n_bits": cod.n_bits, "n_ram": rede.n_ram,
                     "p_binomial": pb, "p_hipergeometrica": ph, "medido": med}
            art["cobertura"].append(linha)
            print(f"{na:>3} {R:>3} {cod.n_bits:>5} {rede.n_ram:>4} {pb:>9.4f} "
                  f"{ph:>9.4f} {med:>8.4f} {med - ph:>+13.4f}")
    print("\nA hipergeométrica é a conta certa para o nosso mapeamento "
          "particionado;\na binomial da spec supõe amostragem com reposição por "
          "RAM (o que gerou C4).")

    # -- 2. diferenciação ---------------------------------------------------
    print("\n" + "=" * 78)
    print("2. EFEITO DE R SOBRE A DIFERENCIAÇÃO DE COMPORTAMENTO")
    print("=" * 78)
    print("mesma observação, índice diferente: o rumo correto do agente k é o "
          "do sensor k")
    print(f"\n{'Na':>3} {'R':>3} {'cobertura':>10} {'erro':>8} {'dispersão':>10}"
          f" {'do cego':>9}")
    por_config: dict = {}
    # `R = 0` é o CONTROLE CEGO: nenhum bit de índice. É a melhor política
    # possível sem saber qual sensor é o seu, e sem ele os outros números não têm
    # escala — "48°" pode ser bom ou desastroso. A dispersão sai exatamente zero
    # por construção, o que também serve de verificação do próprio experimento.
    cego: dict[int, float] = {}
    for na in (2, 4):
        rs = [treinar(na, 0, a.tentativas, s) for s in range(a.sementes)]
        cego[na] = float(np.mean([r["erro_graus"] for r in rs]))
        disp = float(np.mean([r["dispersao"] for r in rs]))
        art["diferenciacao"].append({"na": na, "R": 0, "cobertura": 0.0,
                                     "erro_graus": cego[na], "dispersao": disp,
                                     "cego": True})
        print(f"{na:>3} {0:>3} {0.0:>10.3f} {cego[na]:>7.1f}° {disp:>10.3f} "
              f"{'(cego)':>9}")
    for na in (2, 4):
        for R in (1, 4, 16):
            rs = [treinar(na, R, a.tentativas, s) for s in range(a.sementes)]
            erro = float(np.mean([r["erro_graus"] for r in rs]))
            disp = float(np.mean([r["dispersao"] for r in rs]))
            cob = float(np.mean([r["cobertura"] for r in rs]))
            por_config[(na, R)] = rs
            # Fração do erro do cego que o canal elimina — a grandeza
            # comparável entre `Na` diferentes, já que o cego piora com `Na`.
            recup = (cego[na] - erro) / cego[na]
            art["diferenciacao"].append({"na": na, "R": R, "cobertura": cob,
                                         "erro_graus": erro, "dispersao": disp,
                                         "fracao_do_cego_eliminada": recup})
            print(f"{na:>3} {R:>3} {cob:>10.3f} {erro:>7.1f}° {disp:>10.3f} "
                  f"{recup:>8.0%}")

    # `R = 16` contra `R = 1`, pareado por alvo de avaliação
    arv = ArvoreSementes(20260821)
    print()
    for na in (2, 4):
        e16 = np.mean([r["erro_por_alvo"] for r in por_config[(na, 16)]], axis=0)
        e1 = np.mean([r["erro_por_alvo"] for r in por_config[(na, 1)]], axis=0)
        d = bootstrap_pareado(e16, e1, arv.bootstrap(f"ident-na{na}"))
        veredito = ("R ajuda" if d.conclusivo and d.delta_media < 0
                    else "R piora" if d.conclusivo else "sem efeito detectável")
        print(f"Na={na}  R=16 − R=1: Δ = {d.delta_media:+.2f}°  "
              f"IC95% [{d.delta_ic_lo:+.2f}, {d.delta_ic_hi:+.2f}]  "
              f"MDE {d.mde_95:.2f}°  -> {veredito}")
        art.setdefault("comparacoes", []).append(
            {"na": na, "R16_menos_R1": d.como_linha(), "veredito": veredito})

    print("\nUm efeito NULO aqui é resultado a reportar, não fracasso: seria a")
    print("mitigação (iii) da §4.7 — a coordenação de Lamenza emergiria de")
    print("posições relativas, e não de identidade.")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nartefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
