"""T-05.4(b) — normal projetada contra gaussiana simples. **O portão da SPEC-05.**

O critério de saída é explícito e falsifica a nossa própria decisão:

> C2 (normal projetada) estritamente superior a C0 (gaussiana simples) no
> brinquedo, com IC separado. **Se C2 não vencer, a §4.3 estava errada e a decisão
> precisa ser revista antes da SPEC-06 — não depois.**

Nenhum benchmark clássico tem ação **angular**, então sem um ambiente próprio esta
decisão seria indefensável.

TRÊS INSTRUMENTOS, E POR QUE PRECISOU DE TRÊS

A primeira tentativa usou só o brinquedo sequencial `toy/nav.py` e deu resultado
inconclusivo com **os dois braços piores que rumo aleatório** (150° de erro, 3-8 %
de sucesso). Diagnosticado: com sucesso quase nulo o retorno vale ≈ -60 qualquer
que seja o rumo, então o gradiente não carrega informação direcional. Aquilo media
exploração sob recompensa esparsa — pergunta diferente da que o portão faz.

A correção foi decompor em três, do mais isolado ao mais completo:

    A. bandido fixo        um alvo, um estado. Sem crédito temporal, sem
                           generalização. Isola o MECANISMO de falha do C0.
    B. bandido contextual  alvo sorteado, exige generalizar entre estados.
                           **É O PORTÃO** — é o teste mais forte em que os dois
                           braços demonstravelmente aprendem.
    C. nav sequencial      fim a fim, com crédito temporal. Reportado, mas NÃO
                           conta para o veredito: os dois braços falham nele por
                           um motivo comum, e comparar dois fracassos não informa.

A regra que isso obedece: **um braço que não aprende não é evidência sobre o
outro.** Só (B) satisfaz a condição, então só (B) decide.

O MODO DE FALHA DO C0, medido em (A)

Uma política que quer rumo perto de `θ_norm = 0` tem metade da massa recortada na
fronteira. Pior: se `f₁` cai abaixo de 0 com margem, **toda** amostra recorta para
a mesma ação, a recompensa deixa de variar, e `(r − b) → 0`. É zona absorvente —
o gradiente morre e a política não tem como sair. Em (A) isso aparece como o C0
resolvendo **só** o alvo em 0°, onde a fronteira já aponta por acidente.

CONTRA ESPANTALHO — quatro concessões ao C0

1. Varredura de `σ` própria. `σ` significa coisas diferentes nos dois: no C0 é
   desvio em `θ_norm` (σ = 0,25 são 90° de ruído angular); no C2 é desvio no vetor
   de R³, e o ruído angular resultante depende de `‖f₁:₂‖` — o C2 pode ENCOLHER o
   próprio ruído crescendo a norma. Varrer só a taxa seria injusto.
2. Variante **centrada**: saída deslocada para 0,5, o centro da caixa de ação. É o
   que qualquer implementação competente faria, e sem isso valeria a objeção
   "você inicializou mal o baseline". Medido: melhora a média do C0 de 0,18 para
   0,31 em (A), e o erro de 46,5° para 27,0° em (B). Fica a variante melhor.
3. Varredura de taxa própria.
4. `d = 2` contra `d = 3` do C2 — uma cabeça de menos por posição de memória. A
   comparação favorece o C0 em custo, o que torna uma vitória do C2 mais forte.

MLP dos dois lados, de propósito: o que está sob teste é a **parametrização da
política**, não o aproximador. Trocar os dois ao mesmo tempo mediria duas coisas.

Uso:
    uv run python scripts/spec05_circular.py
    uv run python scripts/spec05_circular.py --sementes 8 --tentativas 40000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from uavwnn.agents.mlp_ac import MLP, Adam
from uavwnn.agents.politica_continua import GaussianaSimples, NormalProjetada
from uavwnn.eval import ArvoreSementes, bootstrap_pareado
from uavwnn.toy import NavAngular

TAMANHO = 200.0
SIGMAS = (0.05, 0.10, 0.20)
TAXAS = (1e-3, 3e-3)
SEED_AVAL = 999
"""Semente fixa da avaliação: os alvos de teste são **os mesmos** para os dois
esquemas, e é isso que autoriza o bootstrap pareado."""


def _politica(esquema: str, sigma: float):
    """`σ` fixa, sem anelamento.

    O anelamento foi desligado depois de uma medição: `decaimento` conta
    **transições**, não episódios, então `decaimento = 800` num brinquedo de 40
    passos colapsava `σ` em ~20 episódios. Aqui não é preciso — a avaliação já é
    determinística (usa a média), então ruído residual no treino não contamina o
    número reportado.
    """
    return (NormalProjetada(sigma, None) if esquema == "C2"
            else GaussianaSimples(sigma, None))


def _rede(esquema: str, d_obs: int, seed: int, centrar: bool):
    d = 3 if esquema == "C2" else 2
    ator = MLP(d_obs, d, 32, seed=seed)
    critico = MLP(d_obs, 1, 32, seed=seed + 7919)
    # Deslocamento para o centro da caixa de ação. Ver concessão 2 no topo.
    off = np.full(d, 0.5) if centrar else np.zeros(d)
    return ator, critico, off


def _erro_angular(theta_norm: float, alvo: float) -> float:
    """Erro de rumo em radianos, no menor arco. `θ_norm ∈ [0,1] → [0, 2π)`."""
    th = float(theta_norm) * 2 * np.pi
    return float(abs(np.mod(th - alvo + np.pi, 2 * np.pi) - np.pi))


# ---------------------------------------------------------------------------
# A. bandido fixo — isola o mecanismo
# ---------------------------------------------------------------------------


def bandido_fixo(esquema: str, alvo: float, sigma: float, taxa: float,
                 tentativas: int, seed: int, centrar: bool) -> float:
    """Um estado, um alvo, `r = cos(θ − θ*)`. Devolve a recompensa final média.

    Sem crédito temporal e sem generalização: se um esquema falha **aqui**, a
    falha é da parametrização e de nada mais. Linha de base é a média móvel das
    últimas 200 recompensas — mais simples que um crítico, e suficiente sem
    estado.
    """
    pol = _politica(esquema, sigma)
    ator, _, off = _rede(esquema, 1, seed, centrar)
    opt = Adam(taxa=taxa)
    rng = np.random.default_rng(seed)
    x = np.array([1.0])
    hist: list[float] = []
    for _ in range(tentativas):
        f = ator.prever(x) + off
        acao, z = pol.amostrar(f, rng)
        r = float(np.cos(acao.theta_norm * 2 * np.pi - alvo))
        hist.append(r)
        b = float(np.mean(hist[-200:]))
        opt.passo(ator.parametros(),
                  ator.gradientes(x, -np.atleast_2d((r - b) * pol.grad_log(f, z))))
    return float(np.mean(hist[-300:]))


# ---------------------------------------------------------------------------
# B. bandido contextual — O PORTÃO
# ---------------------------------------------------------------------------


def bandido_contextual(esquema: str, sigma: float, taxa: float, tentativas: int,
                       seed: int, centrar: bool, n_aval: int = 400) -> NDArray:
    """Alvo sorteado por tentativa; devolve o vetor de erros na avaliação.

    A observação usa **a mesma normalização do ambiente de referência**,
    `(p − s + S)/(2S)`, e o mesmo recentramento afim que o `nav` — o brinquedo
    exercita o caminho de decodificação do problema real. Sem crédito temporal:
    um passo por tentativa, crítico dependente de estado como linha de base.
    """
    pol = _politica(esquema, sigma)
    ator, critico, off = _rede(esquema, 2, seed, centrar)
    opt_a, opt_c = Adam(taxa=taxa), Adam(taxa=taxa)
    rng = np.random.default_rng(seed)

    def observar(sensor: NDArray) -> NDArray:
        # Recentrar é pré-processamento do aproximador: o ambiente devolve algo em
        # ~[0,28; 0,72], faixa estreita e deslocada, ruim para MLP com init He.
        return (((-sensor + TAMANHO) / (2 * TAMANHO)) - 0.5) * 4.0

    def sortear(g: np.random.Generator) -> tuple[NDArray, float]:
        ang = g.uniform(0.0, 2 * np.pi)
        raio = g.uniform(60.0, 90.0)
        return np.array([raio * np.cos(ang), raio * np.sin(ang)]), float(ang)

    for _ in range(tentativas):
        sensor, ang = sortear(rng)
        x = observar(sensor)
        f = ator.prever(x) + off
        acao, z = pol.amostrar(f, rng)
        r = float(np.cos(acao.theta_norm * 2 * np.pi - ang))
        v = float(critico.prever(x)[0])
        opt_a.passo(ator.parametros(),
                    ator.gradientes(x, -np.atleast_2d((r - v) * pol.grad_log(f, z))))
        opt_c.passo(critico.parametros(), critico.gradientes(x, np.array([[v - r]])))

    aval = np.random.default_rng(SEED_AVAL)
    erros = np.empty(n_aval)
    for i in range(n_aval):
        sensor, ang = sortear(aval)
        f = ator.prever(observar(sensor)) + off
        erros[i] = _erro_angular(pol.media(f).theta_norm, ang)   # avaliação: a média
    return np.degrees(erros)


# ---------------------------------------------------------------------------
# C. nav sequencial — reportado, não decide
# ---------------------------------------------------------------------------


def nav_sequencial(esquema: str, sigma: float, taxa: float, episodios: int,
                   seed: int, centrar: bool, desconto: float = 0.99) -> dict:
    """Fim a fim com crédito temporal. Vantagem normalizada por episódio."""
    pol = _politica(esquema, sigma)
    ator, critico, off = _rede(esquema, 2, seed, centrar)
    opt_a, opt_c = Adam(taxa=taxa), Adam(taxa=taxa)
    env = NavAngular()
    rng = np.random.default_rng(seed)
    cod = lambda o: (np.asarray(o, float) - 0.5) * 4.0

    for _ in range(episodios):
        o = env.reset(rng)
        trilha = []
        while True:
            f = ator.prever(cod(o)) + off
            acao, z = pol.amostrar(f, rng)
            o2, r, fim = env.passo(acao.theta_norm, acao.s_norm)
            trilha.append((cod(o), z, r))
            o = o2
            if fim:
                break
        g, retornos = 0.0, []
        for _, _, r in reversed(trilha):
            g = r + desconto * g
            retornos.append(g)
        gs = np.array(retornos[::-1])
        adv = (gs - gs.mean()) / (gs.std() + 1e-8)
        for i, (x, z, _) in enumerate(trilha):
            v = float(critico.prever(x)[0])
            f = ator.prever(x) + off
            opt_a.passo(ator.parametros(),
                        ator.gradientes(x, -np.atleast_2d(adv[i] * pol.grad_log(f, z))))
            opt_c.passo(critico.parametros(),
                        critico.gradientes(x, np.array([[v - gs[i]]])))

    aval = np.random.default_rng(SEED_AVAL)
    passos, erros, sucessos = [], [], []
    for _ in range(200):
        o = env.reset(aval)
        e = []
        while True:
            f = ator.prever(cod(o)) + off
            a = pol.media(f)
            e.append(_erro_angular(a.theta_norm, env.rumo_otimo))
            o, _, fim = env.passo(a.theta_norm, a.s_norm)
            if fim:
                break
        passos.append(env.passos if env.coletado else env.max_passos)
        erros.append(float(np.median(e)))
        sucessos.append(bool(env.coletado))
    return {"passos": float(np.mean(passos)),
            "erro_graus": float(np.degrees(np.mean(erros))),
            "sucesso": float(np.mean(sucessos))}


# ---------------------------------------------------------------------------


BRACOS = (("C0", False, "C0 cru"), ("C0", True, "C0 centrado"), ("C2", False, "C2"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--tentativas", type=int, default=20000)
    ap.add_argument("--episodios", type=int, default=800)
    ap.add_argument("--pular-nav", action="store_true",
                    help="pula (C), que não conta para o veredito")
    ap.add_argument("--saida", default="reports/spec05-circular.json")
    a = ap.parse_args()

    art: dict = {"config": vars(a), "sigmas": list(SIGMAS), "taxas": list(TAXAS)}

    # -- A ------------------------------------------------------------------
    print("=" * 74)
    print("A. BANDIDO FIXO — isola o mecanismo de falha (não é o portão)")
    print("=" * 74)
    alvos = np.radians(np.arange(0, 360, 30))
    print(f"{'braço':>14} {'σ':>5} {'r média':>8} {'r pior':>8} {'>0,95':>7}")
    art["A"] = {}
    for esq, ctr, rot in BRACOS:
        melhor = None
        for sig in SIGMAS:
            v = np.array([bandido_fixo(esq, alv, sig, 3e-3, a.tentativas // 5, s, ctr)
                          for alv in alvos for s in range(a.sementes)])
            print(f"{rot:>14} {sig:>5.2f} {v.mean():>8.3f} {v.min():>8.3f} "
                  f"{np.mean(v > 0.95):>6.0%}")
            if melhor is None or v.mean() > melhor[1]:
                melhor = (sig, float(v.mean()), float(v.min()),
                          float(np.mean(v > 0.95)))
        art["A"][rot] = {"sigma": melhor[0], "r_media": melhor[1],
                         "r_pior": melhor[2], "frac_acima_095": melhor[3]}
    print("\nO C0 resolve só o alvo em 0°, onde a fronteira do recorte já aponta.")
    print("Com f₁ < 0 toda amostra recorta na mesma ação: a recompensa não varia,")
    print("(r − b) → 0, e o gradiente morre. Zona absorvente.")

    # -- B: O PORTÃO --------------------------------------------------------
    print("\n" + "=" * 74)
    print("B. BANDIDO CONTEXTUAL — **O PORTÃO**")
    print("=" * 74)
    print(f"{'braço':>14} {'σ':>5} {'taxa':>7} {'erro méd':>9} {'mediana':>8} "
          f"{'p90':>7} {'<15°':>6}")
    melhores: dict[str, tuple] = {}
    art["B"] = {"varredura": []}
    for esq, ctr, rot in BRACOS:
        for sig in SIGMAS:
            for tx in TAXAS:
                per_semente = [bandido_contextual(esq, sig, tx, a.tentativas, s, ctr)
                               for s in range(a.sementes)]
                e = np.concatenate(per_semente)
                linha = {"braco": rot, "sigma": sig, "taxa": tx,
                         "erro_medio": float(e.mean()),
                         "mediana": float(np.median(e)),
                         "p90": float(np.percentile(e, 90)),
                         "frac_sub15": float(np.mean(e < 15.0))}
                art["B"]["varredura"].append(linha)
                print(f"{rot:>14} {sig:>5.2f} {tx:>7.0e} {e.mean():>8.1f}° "
                      f"{np.median(e):>7.1f}° {np.percentile(e, 90):>6.1f}° "
                      f"{np.mean(e < 15.0):>5.0%}")
                # média POR ALVO entre sementes: preserva o pareamento por alvo
                por_alvo = np.mean(np.stack(per_semente), axis=0)
                if rot not in melhores or e.mean() < melhores[rot][1]:
                    melhores[rot] = ((sig, tx), float(e.mean()), por_alvo)

    # O C0 entra na comparação com a **melhor** das suas duas variantes.
    c0 = min(("C0 cru", "C0 centrado"), key=lambda r: melhores[r][1])
    (s0, t0), m0, v0 = melhores[c0]
    (s2, t2), m2, v2 = melhores["C2"]
    print(f"\nmelhor C0: {c0}, σ={s0}, taxa={t0:.0e} → {m0:.1f}° de erro médio")
    print(f"melhor C2: σ={s2}, taxa={t2:.0e} → {m2:.1f}° de erro médio")

    arv = ArvoreSementes(20260809)
    d = bootstrap_pareado(v2, v0, arv.bootstrap("spec05-circular-B"))
    print(f"\nC2 − C0 (erro angular em graus, menor é melhor), pareado por alvo")
    print(f"  n = {d.n_pareado}   Δ = {d.delta_media:+.2f}°  "
          f"IC95% [{d.delta_ic_lo:+.2f}, {d.delta_ic_hi:+.2f}]  MDE {d.mde_95:.2f}°")
    print(f"  taxa de vitória do C2: {d.taxa_vitoria_a:.1%} "
          f"[{d.tv_ic_lo:.1%}, {d.tv_ic_hi:.1%}]   Wilcoxon p = {d.p_wilcoxon:.2e}")
    art["B"]["melhor_C0"] = {"variante": c0, "sigma": s0, "taxa": t0, "erro": m0}
    art["B"]["melhor_C2"] = {"sigma": s2, "taxa": t2, "erro": m2}
    art["B"]["comparacao"] = d.como_linha()

    venceu = d.conclusivo and d.delta_media < 0

    # -- C ------------------------------------------------------------------
    if not a.pular_nav:
        print("\n" + "=" * 74)
        print("C. NAV SEQUENCIAL — reportado, NÃO decide")
        print("=" * 74)
        art["C"] = []
        print(f"{'braço':>14} {'σ':>5} {'passos':>7} {'erro':>8} {'sucesso':>8}")
        for esq, ctr, rot in BRACOS:
            for sig in (0.10, 0.20):
                r = [nav_sequencial(esq, sig, 3e-3, a.episodios, s, ctr)
                     for s in range(a.sementes)]
                agg = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
                art["C"].append({"braco": rot, "sigma": sig, **agg})
                print(f"{rot:>14} {sig:>5.2f} {agg['passos']:>7.2f} "
                      f"{agg['erro_graus']:>7.1f}° {agg['sucesso']:>7.1%}")
        print("\nO ótimo analítico faz 7,86 passos com 100 % de sucesso, então o")
        print("brinquedo é resolvível — o que falha é o crédito temporal, e falha")
        print("nos DOIS braços. Comparar dois fracassos não informa sobre a §4.3,")
        print("então (C) fica registrado como pendência e fora do veredito.")

    # -- veredito -----------------------------------------------------------
    print("\n" + "=" * 74)
    if venceu:
        print("PORTÃO ABERTO: C2 estritamente melhor que C0 em (B), IC separado.")
        print("  -> a decisão da §4.3 (normal projetada) se sustenta.")
        print("  -> (A) explica o mecanismo: zona absorvente no recorte.")
    elif d.contem_zero:
        print("EMPATE em (B): o IC contém zero.")
        print("  -> a §4.3 NÃO está demonstrada. Antes da SPEC-06, decidir entre")
        print("     aumentar o poder do teste ou revisar a decisão.")
    else:
        print("C0 VENCEU em (B): a §4.3 estava errada.")
        print("  -> revisar a parametrização da política ANTES da SPEC-06.")
    print("=" * 74)

    art["veredito"] = "C2" if venceu else ("empate" if d.contem_zero else "C0")
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nartefato: {a.saida}")
    return 0 if venceu else 1


if __name__ == "__main__":
    sys.exit(main())
