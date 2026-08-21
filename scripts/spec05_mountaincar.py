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

# -- alvos publicados, tabela 5.8 (p.67) ------------------------------------
#
# ERRO DE TRANSCRIÇÃO CORRIGIDO. A tabela da SPEC-05 T-05.4(a) registra o alvo da
# MLP como `0,02 (−0,03; −0,02)` — um estimador pontual FORA do próprio intervalo,
# o que já denunciava o problema. A tabela 5.8 do original diz **−0,02**, e o sinal
# se perdeu na transcrição.
ALVOS = {
    "n-tuple AC": (93.3, 92.9, 93.8),
    "AC com MLP": (-0.02, -0.03, -0.02),
}

# -- orçamento do §5.4 (p.65), que a spec não registrava --------------------
#
# > "Each pair of algorithm and benchmark task is executed **10 times**, with
# > distinct random initializations, for **2000 episodes**, except for hopper".
#
# A primeira execução usou 1500 × 5 — 75 % dos episódios e 50 % das sementes. Não
# era reprodução do protocolo dele, e um resultado negativo ali não seria
# atribuível ao método.
EPISODIOS_PUBLICADO = 2000
SEMENTES_PUBLICADO = 10

# O protocolo de avaliação também sai da p.66, e por dedução em vez de citação
# direta. O texto explica o resultado da MLP assim:
#
# > "the MLP methods having optimized their agents to a local optimum of not
# > moving the car. In this way, they **never incur the penalty of expending
# > energy**, but also never attain the reward of reaching the flag."
#
# Com `Σ = 0,4` o ruído sozinho custa `−0,1 · 0,4 · 999 = −39,96` por episódio. Se
# o número reportado incluísse exploração, a MLP NÃO poderia "never incur the
# penalty" — logo o número é **sem ruído**, e a avaliação determinística da
# `_avaliar` é o protocolo certo. Confirmado por medição: com ruído medimos
# −31,86; sem ruído, −0,01 contra os −0,02 publicados.

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


def _avaliar(politica_media, env_fn, episodios: int, seed: int) -> tuple[list[float], float]:
    """**Avaliação determinística: sem ruído, usando a média.**

    Não é escolha nossa — é o protocolo, e vale para os dois braços:

    > Lamenza §7: *"during an evaluation the training is temporarily halted and
    > action noise is removed"*.

    ISTO CORRIGE UM ERRO MEU. A primeira versão deste script reportava os
    retornos de **treino**, com o ruído de exploração ligado, e por isso não podia
    reproduzir número algum: com `Σ = 0,4`, a MLP tem `σ = 0,632`, logo
    `E[a²] = 0,4`, e o custo de ação sozinho vale `−0,1 · 0,4 · 999 = −39,96`. O
    `0,02` publicado é **inatingível** sob ruído; ele exige média ≈ 0 **e** ruído
    removido. Medi −31,86 e chamei de discrepância — a discrepância era do
    protocolo, não do algoritmo.
    """
    env = env_fn()
    retornos, chegadas = [], 0
    for ep in range(episodios):
        obs, _ = env.reset(seed=900_000 + seed * 10_000 + ep)
        total = 0.0
        while True:
            acao = politica_media(obs)
            obs, r, term, trunc, _ = env.step(np.clip(acao, -LIMITE_ACAO, LIMITE_ACAO))
            total += float(r)
            if term or trunc:
                chegadas += bool(term)
                break
        retornos.append(total)
    env.close()
    return retornos, chegadas / episodios


def rodar_ntupla(episodios: int, seed: int, sigma_como_desvio: bool = False,
                 n_aval: int = 100, familia: str = "dissertacao") -> dict:
    cod, n_bits = _codificador()
    # `familia` seleciona QUAL rede sem peso, e a diferença não é cosmética:
    #
    #   dissertacao   ator de gradiente funcional (§4.3.1) + crítico NTRN com
    #                 esquecimento (§4.2.1). DUAS famílias, convenção de média —
    #                 é o que fechou a pendência C2.
    #   anbubus       gradiente funcional nos DOIS lados, convenção de soma. É o
    #                 `NTNModel` de `anbubus/ntn_neurocomputing_v0`, cujo PPO
    #                 treina ator e crítico por gradiente e portanto usa um
    #                 modelo só.
    #
    # O laço é o MESMO nos dois, e é isso que torna a comparação sobre a rede.
    if familia not in ("dissertacao", "anbubus"):
        raise ValueError(f"familia={familia!r}")
    soma = familia == "anbubus"
    conv = "soma" if soma else "media"

    ator = RedeGradienteFuncional(n_bits, A3["tamanho_tupla"], n_saidas=1,
                                  seed=seed, c_max=None, convencao=conv)
    if soma:
        # `eta` do crítico convertido: sob soma, `atualizar` muda `f` em `η·N`.
        # Sem dividir, `η = 0,1` daria 0,8 por passo com `N = 8`.
        critico = RedeGradienteFuncional(
            n_bits, A3["tamanho_tupla"], n_saidas=1, seed=seed, c_max=None,
            convencao="soma", eta_treino=0.1 / ator.n_ram)
    else:
        critico = RedeRegressaoEsquecimento(n_bits, A3["tamanho_tupla"],
                                            n_saidas=1, phi=A3["phi"], seed=seed)

    # A taxa da A.3 está na convenção de MÉDIA. Transferi-la crua para a de soma
    # multiplicaria o passo efetivo por `N` — o erro que a §4.2 nomeia.
    taxa = ator.taxa_equivalente(A3["taxa"], conv) if soma else A3["taxa"]
    agente = AtorCriticoNTupla(
        politica=PoliticaGaussiana(ator, A3["covariancia"],
                                   escala_como_desvio=sigma_como_desvio),
        critico=critico, passos=A3["passos"], desconto=A3["desconto"],
        taxa=taxa, rng=np.random.default_rng(seed))

    env = _env()
    retornos, chegadas = [], []
    for ep in range(episodios):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        agente.novo_episodio()
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

    # A média da gaussiana É a ação determinística (`mode` do
    # `ContinousActionPolicy.jl`): `f(s)`, sem somar ruído.
    aval, cheg_aval = _avaliar(
        lambda o: agente.politica.media(cod(o)), _env, n_aval, seed)
    return {"treino": retornos, "chegadas_treino": chegadas,
            "aval": aval, "chegada_aval": cheg_aval}


def rodar_mlp(episodios: int, seed: int, n_aval: int = 100) -> dict:
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

    aval, cheg_aval = _avaliar(lambda o: ator.prever(np.asarray(o, float)),
                               _env, n_aval, seed)
    return {"treino": retornos, "chegadas_treino": chegadas,
            "aval": aval, "chegada_aval": cheg_aval}


class _Tarefa:
    """Uma semente, os dois braços. Classe e não closure para ser picklável.

    As sementes são independentes por construção — cada uma instancia redes,
    `rng` e `env.reset` próprios —, então rodá-las em paralelo **não altera
    resultado algum**, só o relógio. É o que permite o orçamento publicado
    (2000 × 10) caber em tempo utilizável.
    """

    def __init__(self, episodios: int, n_aval: int, p13: bool,
                 familia: str = "dissertacao") -> None:
        self.episodios, self.n_aval, self.p13 = episodios, n_aval, p13
        self.familia = familia

    def __call__(self, seed: int):
        return (rodar_ntupla(self.episodios, seed, sigma_como_desvio=self.p13,
                             n_aval=self.n_aval, familia=self.familia),
                rodar_mlp(self.episodios, seed, n_aval=self.n_aval))


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
    ap.add_argument("--episodios", type=int, default=EPISODIOS_PUBLICADO)
    ap.add_argument("--sementes", type=int, default=SEMENTES_PUBLICADO)
    ap.add_argument("--processos", type=int, default=1,
                    help="sementes em paralelo. As sementes são independentes por "
                         "construção (cada uma tem rng e reset próprios), então "
                         "paralelizar NÃO altera resultado algum — só o relógio.")
    ap.add_argument("--avaliar-ultimos", type=int, default=100,
                    help="episódios finais de TREINO, para a coluna de treino")
    ap.add_argument("--n-aval", type=int, default=100,
                    help="episódios de avaliação DETERMINÍSTICA por semente — "
                         "é esta a coluna que se compara ao publicado")
    ap.add_argument("--familia", choices=("dissertacao", "anbubus"),
                    default="dissertacao",
                    help="qual rede sem peso: `dissertacao` (duas famílias, "
                         "convenção de média) ou `anbubus` (gradiente funcional "
                         "nos dois lados, convenção de soma)")
    ap.add_argument("--p13", action="store_true",
                    help="trata Σ como desvio (ContinousActionPolicy.jl) e não "
                         "como covariância (eq. 4.34) — mede a pendência P13")
    ap.add_argument("--saida", default="reports/spec05-mountaincar.json")
    a = ap.parse_args()

    print(f"T-05.4(a) · MountainCarContinuous-v0 · {a.episodios} episódios "
          f"· {a.sementes} sementes")
    print(f"AVAL = {a.n_aval} episódios determinísticos por semente, ruído "
          f"removido (Lamenza §7)")
    print(f"treino = últimos {a.avaliar_ultimos} episódios, com ruído — NÃO é o "
          f"que se compara ao publicado")
    print(f"Σ tratada como {'DESVIO (P13)' if a.p13 else 'covariância (eq. 4.34)'}")
    print(f"família WNN: {a.familia}\n")

    arv = ArvoreSementes(20260821)
    art: dict = {"config": vars(a), "familia": a.familia, "A3": A3, "A6": A6,
                 "resultados": []}

    aval_nt, aval_mlp = [], []
    trei_nt, trei_mlp = [], []
    cheg_nt, cheg_mlp = [], []
    ca_nt, ca_mlp = [], []
    print(f"{'':>11} {'n-tuple':>26}   {'MLP':>26}")
    print(f"{'semente':>11} {'treino':>8} {'AVAL':>8} {'cheg.av':>8}   "
          f"{'treino':>8} {'AVAL':>8} {'cheg.av':>8}")

    tarefa = _Tarefa(a.episodios, a.n_aval, a.p13, a.familia)
    if a.processos > 1:
        import multiprocessing as mp
        with mp.Pool(a.processos) as pool:
            pares = pool.map(tarefa, range(a.sementes))
    else:
        pares = [tarefa(s) for s in range(a.sementes)]

    for s, (rn, rm) in enumerate(pares):
        for r, trei, avl, cheg, ca in ((rn, trei_nt, aval_nt, cheg_nt, ca_nt),
                                       (rm, trei_mlp, aval_mlp, cheg_mlp, ca_mlp)):
            trei.extend(r["treino"][-a.avaliar_ultimos:])
            avl.extend(r["aval"])
            cheg.extend(r["chegadas_treino"])
            ca.append(r["chegada_aval"])
        print(f"{s:>11} {np.mean(rn['treino'][-a.avaliar_ultimos:]):>8.2f} "
              f"{np.mean(rn['aval']):>8.2f} {rn['chegada_aval']:>7.1%}   "
              f"{np.mean(rm['treino'][-a.avaliar_ultimos:]):>8.2f} "
              f"{np.mean(rm['aval']):>8.2f} {rm['chegada_aval']:>7.1%}")

    finais_nt, finais_mlp = aval_nt, aval_mlp   # o publicado é AVALIAÇÃO
    art["treino"] = {"n-tuple": float(np.mean(trei_nt)),
                     "mlp": float(np.mean(trei_mlp))}

    # A TAXA DE CHEGADA É O QUE TORNA UMA DISCREPÂNCIA DIAGNOSTICÁVEL.
    # "nunca alcançou a bandeira" e "alcançou e aprendeu mal" são falhas
    # diferentes com correções opostas: a primeira é exploração (Σ, m), a segunda
    # é atribuição de crédito (α, φ, γ). Sem este número, um retorno de -33 não
    # distingue as duas — e o critério 1 exige diagnóstico, não só o veredito.
    art["taxa_chegada"] = {"n-tuple_treino": float(np.mean(cheg_nt)),
                           "mlp_treino": float(np.mean(cheg_mlp)),
                           "n-tuple_aval": float(np.mean(ca_nt)),
                           "mlp_aval": float(np.mean(ca_mlp))}

    print()
    art["resultados"].append(_veredito("n-tuple AC", finais_nt, arv))
    art["resultados"].append(_veredito("AC com MLP", finais_mlp, arv))

    print(f"\ntaxa de chegada à bandeira — treino: n-tuple {np.mean(cheg_nt):.1%}, "
          f"MLP {np.mean(cheg_mlp):.1%}")
    print(f"                              avaliação: n-tuple {np.mean(ca_nt):.1%}, "
          f"MLP {np.mean(ca_mlp):.1%}")
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
