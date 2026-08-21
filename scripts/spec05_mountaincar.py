"""T-05.4(a) — reprodução do Katopodis no `MountainCarContinuous-v0`.

Critério de saída 1 da SPEC-05:

> Os quatro números de Katopodis reproduzidos dentro dos ICs publicados, **ou a
> discrepância diagnosticada e registrada por escrito**. Sem isto, nenhum
> resultado negativo da SPEC-06 é atribuível ao problema.

    alvo publicado   n-tuple AC   93,3  (92,9; 93,8)
                     AC com MLP    0,02 (−0,03; −0,02)

O CONTROLE NEGATIVO É TÃO INFORMATIVO QUANTO O POSITIVO

O `0,02` da MLP não é "a MLP falhou": é o **ótimo local de não mover o carro**. A
recompensa do Mountain Car penaliza a ação ao quadrado, então ficar parado rende
~0 e é um mínimo forte. Se a nossa MLP *resolver* a tarefa, ela não é a MLP dele
e a comparação de H1 deixa de ser a comparação dele. Uma MLP boa demais aqui é
falha de reprodução, não sucesso.

HIPERPARÂMETROS, DAS TABELAS A.3 E A.6, VERBATIM

    n-tuple AC (A.3)   m = 100, n = 32, codificação Simple, resolução 128,
                       φ = 0,85, α = 8·10⁻⁴, γ = 1,0, Σ = 0,5
    AC com MLP (A.6)   m = 500, α = 10⁻⁵, γ = 0,95, Σ = 0,4

Três coisas que estas tabelas resolvem e que valem registro:

1. `γ = 1,0` no n-tuple e `0,95` na MLP. **Não é erro de transcrição** — são
   linhas de tabelas diferentes, e Katopodis usou desconto diferente em cada
   algoritmo. Igualar "para ser justo" seria deixar de reproduzir.
2. `Σ` é **covariância** (eq. 4.34), então o desvio é `√Σ`. `PoliticaGaussiana`
   trata isso, e a alternativa (`Σ` como desvio, que é o que o
   `ContinousActionPolicy.jl` faz) é a pendência P13 — fica atrás de uma flag e é
   medida aqui, porque este é o único lugar onde há número publicado para
   decidir.
3. A pergunta aberta 3 da spec — *"a variância de Katopodis para o Hopper é 0,2
   (§5.3) ou 0,3 (tab. A.3)?"* — fica **respondida na parte factual**: a tabela
   A.3 diz **0,3**. A inconsistência é entre o texto da §5.3 e o apêndice, não
   dentro do apêndice. Resolver qual das duas reproduz o número publicado exigiria
   rodar o Hopper, que precisa de MuJoCo e fica fora do orçamento desta spec (a
   T-05.4(a) já a trata como condicional: *"e, se o orçamento permitir, Hopper"*).

Uso:
    uv run python scripts/spec05_mountaincar.py
    uv run python scripts/spec05_mountaincar.py --episodios 3000 --sementes 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uavwnn.agents.mlp_ac import MLP, Adam
from uavwnn.encoding.artigo import TermometroArtigo
from uavwnn.eval import ArvoreSementes, bootstrap_media
from uavwnn.rl.actor_critic import AtorCriticoNTupla
from uavwnn.rl.politica import PoliticaGaussiana
from uavwnn.wnn.redes import RedeGradienteFuncional, RedeRegressaoEsquecimento

# -- alvos publicados (tab. 5.x de Katopodis) -------------------------------
ALVOS = {
    "n-tuple AC": (93.3, 92.9, 93.8),
    "AC com MLP": (0.02, -0.03, -0.02),
}

# -- tabela A.3, linha Mountaincar ------------------------------------------
A3 = dict(passos=100, tamanho_tupla=32, resolucao=128, phi=0.85,
          taxa=8e-4, desconto=1.0, covariancia=0.5)
# -- tabela A.6, linha Mountaincar ------------------------------------------
A6 = dict(passos=500, taxa=1e-5, desconto=0.95, covariancia=0.4)

# `MountainCarContinuous-v0`: obs = (posição, velocidade), ação escalar em [-1,1].
LIMITES = ((-1.2, 0.6), (-0.07, 0.07))
LIMITE_ACAO = 1.0


def _env():
    import gymnasium as gym
    return gym.make("MountainCarContinuous-v0")


def _codificador():
    """Termômetro *Simple* da eq. (3.23), resolução 128, **por dimensão**.

    Cada dimensão tem faixa própria — posição em [−1,2; 0,6] e velocidade em
    [−0,07; 0,07]. Um termômetro único sobre as duas gastaria quase todos os bits
    da velocidade num platô, porque as escalas diferem por 13×.
    """
    termos = [TermometroArtigo(lo, hi, A3["resolucao"]) for lo, hi in LIMITES]

    def cod(obs):
        o = np.asarray(obs, float).ravel()
        return np.concatenate([t.codificar(np.array([v]))
                               for t, v in zip(termos, o)]).ravel()
    return cod, sum(t.n_bits_para(1) for t in termos)


def rodar_ntupla(episodios: int, seed: int, sigma_como_desvio: bool = False) -> list[float]:
    cod, n_bits = _codificador()
    ator = RedeGradienteFuncional(n_bits, A3["tamanho_tupla"], n_saidas=1,
                                  seed=seed, c_max=None)
    critico = RedeRegressaoEsquecimento(n_bits, A3["tamanho_tupla"], n_saidas=1,
                                        phi=A3["phi"], seed=seed)
    agente = AtorCriticoNTupla(
        politica=PoliticaGaussiana(ator, A3["covariancia"],
                                   escala_como_desvio=sigma_como_desvio),
        critico=critico, passos=A3["passos"], desconto=A3["desconto"],
        taxa=A3["taxa"], rng=np.random.default_rng(seed))

    env = _env()
    retornos, chegadas = [], []
    for ep in range(episodios):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        agente.reiniciar()
        total, chegou = 0.0, False
        while True:
            bits = cod(obs)
            acao = agente.agir(bits)
            obs2, r, term, trunc, _ = env.step(np.clip(acao, -LIMITE_ACAO, LIMITE_ACAO))
            # `acao` BRUTA na atualização, recortada só para o ambiente — a mesma
            # regra da §4.3 da SPEC-05, e aqui ela vale igual.
            agente.registrar(bits, acao, float(r))
            total += float(r)
            obs = obs2
            chegou = chegou or bool(term)
            if term or trunc:
                agente.fim_de_episodio()
                break
            agente.passo_intermediario(cod(obs))
        retornos.append(total)
        chegadas.append(chegou)
    env.close()
    return retornos, chegadas


def rodar_mlp(episodios: int, seed: int) -> list[float]:
    """MLP `2→32→32→1`, Adam, hiperparâmetros da A.6.

    Duas camadas ocultas de 32 com ReLU é a arquitetura da §5.4 de Katopodis, e a
    mesma do baseline pareado da SPEC-05 §4.8 — só a saída muda, porque a ação
    aqui é escalar e não `⟨θ, s⟩`.
    """
    ator = MLP(2, 1, 32, seed=seed)
    critico = MLP(2, 1, 32, seed=seed + 7919)
    opt_a, opt_c = Adam(taxa=A6["taxa"]), Adam(taxa=A6["taxa"])
    desvio = float(np.sqrt(A6["covariancia"]))     # Σ é covariância (eq. 4.34)
    inv_cov = 1.0 / A6["covariancia"]

    env = _env()
    rng = np.random.default_rng(seed)
    retornos, chegadas = [], []
    for ep in range(episodios):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        trilha, total, chegou = [], 0.0, False
        while True:
            x = np.asarray(obs, float)
            mu = ator.prever(x)
            acao = mu + desvio * rng.standard_normal(1)
            obs2, r, term, trunc, _ = env.step(np.clip(acao, -LIMITE_ACAO, LIMITE_ACAO))
            trilha.append((x, acao, float(r)))
            total += float(r)
            obs = obs2
            chegou = chegou or bool(term)
            if term or trunc:
                break
        # retorno de `m` passos com bootstrap, como o Algoritmo 4.4
        g = 0.0
        for i, (x, acao, r) in enumerate(reversed(trilha)):
            if i == A6["passos"]:
                g = float(critico.prever(x)[0])
            g = r + A6["desconto"] * g
            v = float(critico.prever(x)[0])
            delta = g - v
            grad = inv_cov * (acao - ator.prever(x))          # eq. (4.35)
            opt_a.passo(ator.parametros(),
                        ator.gradientes(x, -np.atleast_2d(delta * grad)))
            opt_c.passo(critico.parametros(),
                        critico.gradientes(x, np.array([[v - g]])))
        retornos.append(total)
        chegadas.append(chegou)
    env.close()
    return retornos, chegadas


def _veredito(rotulo: str, medido, arv: ArvoreSementes) -> dict:
    alvo, lo, hi = ALVOS[rotulo]
    m = bootstrap_media(medido, arv.bootstrap(f"mc-{rotulo}"))
    dentro = not (m.ic_hi < lo or m.ic_lo > hi)
    print(f"{rotulo:>12}  medido {m.media:>8.2f} [{m.ic_lo:>7.2f}, {m.ic_hi:>7.2f}]"
          f"   publicado {alvo:>6.2f} [{lo:>6.2f}, {hi:>6.2f}]   "
          f"{'SOBREPÕE' if dentro else 'DISCREPANTE'}")
    return {"rotulo": rotulo, "media": m.media, "ic": [m.ic_lo, m.ic_hi],
            "publicado": [alvo, lo, hi], "sobrepoe": bool(dentro)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=1500)
    ap.add_argument("--sementes", type=int, default=5)
    ap.add_argument("--avaliar-ultimos", type=int, default=100,
                    help="episódios finais que entram na média reportada")
    ap.add_argument("--p13", action="store_true",
                    help="trata Σ como desvio (ContinousActionPolicy.jl) e não "
                         "como covariância (eq. 4.34) — mede a pendência P13")
    ap.add_argument("--saida", default="reports/spec05-mountaincar.json")
    a = ap.parse_args()

    print(f"T-05.4(a) · MountainCarContinuous-v0 · {a.episodios} episódios "
          f"· {a.sementes} sementes")
    print(f"média sobre os últimos {a.avaliar_ultimos} episódios de cada semente")
    print(f"Σ tratada como {'DESVIO (P13)' if a.p13 else 'covariância (eq. 4.34)'}\n")

    arv = ArvoreSementes(20260821)
    art: dict = {"config": vars(a), "A3": A3, "A6": A6, "resultados": []}

    finais_nt, finais_mlp = [], []
    cheg_nt, cheg_mlp = [], []
    for s in range(a.sementes):
        rn, cn = rodar_ntupla(a.episodios, s, sigma_como_desvio=a.p13)
        rm, cm = rodar_mlp(a.episodios, s)
        finais_nt.extend(rn[-a.avaliar_ultimos:])
        finais_mlp.extend(rm[-a.avaliar_ultimos:])
        cheg_nt.extend(cn)
        cheg_mlp.extend(cm)
        print(f"  semente {s}: n-tuple {np.mean(rn[-a.avaliar_ultimos:]):>7.2f} "
              f"(chegou em {np.mean(cn):>5.1%} dos episódios)   "
              f"MLP {np.mean(rm[-a.avaliar_ultimos:]):>7.2f} "
              f"({np.mean(cm):>5.1%})")

    # A TAXA DE CHEGADA É O QUE TORNA UMA DISCREPÂNCIA DIAGNOSTICÁVEL.
    # "nunca alcançou a bandeira" e "alcançou e aprendeu mal" são falhas
    # diferentes com correções opostas: a primeira é exploração (Σ, m), a segunda
    # é atribuição de crédito (α, φ, γ). Sem este número, um retorno de -33 não
    # distingue as duas — e o critério 1 exige diagnóstico, não só o veredito.
    art["taxa_chegada"] = {"n-tuple": float(np.mean(cheg_nt)),
                           "mlp": float(np.mean(cheg_mlp))}

    print()
    art["resultados"].append(_veredito("n-tuple AC", finais_nt, arv))
    art["resultados"].append(_veredito("AC com MLP", finais_mlp, arv))

    print(f"\ntaxa de chegada à bandeira: n-tuple {np.mean(cheg_nt):.1%}, "
          f"MLP {np.mean(cheg_mlp):.1%}")
    if np.mean(cheg_nt) < 0.01:
        print("  -> a n-tupla praticamente NUNCA chega. Uma discrepância aqui é")
        print("     de EXPLORAÇÃO (Σ, m), não de atribuição de crédito.")
    print("\nO alvo da MLP (≈ 0) é o ótimo local de não mover o carro, e é")
    print("CONTROLE NEGATIVO: uma MLP que resolvesse a tarefa não seria a dele.")

    todos = all(r["sobrepoe"] for r in art["resultados"])
    art["hopper"] = {
        "rodado": False,
        "motivo": "exige MuJoCo; T-05.4(a) trata o Hopper como condicional ao "
                  "orçamento ('e, se o orçamento permitir, Hopper')",
        "pergunta_aberta_3": "a tabela A.3 diz covariância 0,3 no Hopper; a "
                             "inconsistência é com o texto da §5.3, não dentro "
                             "do apêndice. Qual reproduz o número publicado "
                             "fica sem resposta sem rodar o Hopper.",
    }
    art["veredito"] = "reproduzido" if todos else "discrepante"

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(art, indent=2, ensure_ascii=False))
    print(f"\nartefato: {a.saida}")
    if not todos:
        print("\nDISCREPÂNCIA — o critério 1 admite isto, mas exige diagnóstico")
        print("por escrito. Ele vai para reports/spec05-mountaincar.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
