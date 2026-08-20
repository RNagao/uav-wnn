"""Teste A/A — calibração do bootstrap pareado. SPEC-02 T-02.9 §2.

**É o único teste que verifica de fato a cobertura do procedimento de intervalo.**
Um erro de reamostragem — reamostrar os dois métodos independentemente em vez de
aplicar o mesmo conjunto de índices — aparece aqui como cobertura errada e **em
nenhum outro lugar**. Passa em todo teste unitário óbvio.

A IDEIA

Comparar um método **contra ele mesmo**. A diferença verdadeira é zero por
construção, então um IC de 95 % bem calibrado deve conter zero em ~95 % das
execuções. Repetimos ~200 vezes e contamos.

    cobertura muito abaixo de 95 %  →  IC estreito demais, declararíamos
                                       diferenças que não existem
    cobertura muito acima de 95 %   →  IC largo demais, perderíamos diferenças
                                       reais (e o MDE reportado estaria otimista)

POR QUE ISTO IMPORTA RETROATIVAMENTE

Todo IC já reportado neste repositório passa por `bootstrap_pareado` ou pelo
mesmo procedimento escrito à mão: a validação 9/9 do guloso (SPEC-00), o `greedy`
vs `smart`, a comparação de custo (SPEC-03), a replicação de §5.2.2 e a validação
3/3 do MADDPG. Se a cobertura estiver errada, **todos eles precisam de revisão**.

O QUE ESTE SCRIPT COMPARA

Dois modos, e a diferença entre eles é o ponto:

- `--fonte sintetica` (padrão) — amostras de distribuições com a forma da nossa
  métrica: assimétrica e **censurada** em `max_episode_length`. Rápido, e é onde
  a censura é exercitada de propósito.
- `--fonte maddpg` — usa os tempos de conclusão reais da campanha
  (`reports/maddpg/campanha-a*s*.csv`), que é a distribuição que de fato
  medimos.

Também roda o **contra-exemplo**: o mesmo teste com reamostragem independente, que
é o bug que o A/A existe para pegar. Se o contra-exemplo passar, o teste não está
discriminando nada.

Uso:
    uv run python scripts/spec02_aa.py
    uv run python scripts/spec02_aa.py --execucoes 500 --fonte maddpg
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

from uavwnn.eval import MAX_EPISODE_LENGTH, ArvoreSementes, bootstrap_pareado
from uavwnn.eval.stats import B_CANONICO

NOMINAL = 0.95


def _par_sintetico(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Duas realizações da MESMA distribuição, nos MESMOS cenários.

    É o desenho que a T-02.9 pede — *"`random_uniforme` contra ela mesma com
    **sementes de política diferentes**"* — e é onde a primeira versão deste
    script errou: comparar o vetor contra si mesmo dá `Δ_b = 0` exatamente em
    toda réplica, largura de IC zero e cobertura de 100 %. Não testa nada.

    A estrutura correta separa as duas fontes de variância:

        efeito_cenario   COMPARTILHADO entre A e B  ← é o que o pareamento remove
        ruido_politica   INDEPENDENTE               ← é o que sobra na diferença

    Sem o efeito compartilhado não há pareamento a verificar; sem o ruído
    independente não há diferença a medir.

    A forma é a da nossa métrica: log-normal assimétrica, **censurada** em
    `max_episode_length`. A censura não é enfeite — `t_conc_pen` tem um átomo no
    extremo direito (D-02.2), e o bootstrap da média é sensível a átomos.
    """
    efeito = rng.lognormal(mean=np.log(30.0), sigma=0.55, size=n)
    ruido_a = rng.normal(0.0, 4.0, size=n)
    ruido_b = rng.normal(0.0, 4.0, size=n)
    a = np.minimum(np.maximum(efeito + ruido_a, 0.0), MAX_EPISODE_LENGTH)
    b = np.minimum(np.maximum(efeito + ruido_b, 0.0), MAX_EPISODE_LENGTH)
    return a, b


def _amostras_maddpg() -> list[tuple[np.ndarray, np.ndarray]]:
    """Par A/A construído da campanha real, célula por célula.

    O CSV guarda, por run, a **média** dos 200 episódios e o **desvio entre
    episódios**. Isso dá exatamente as duas componentes de que o A/A precisa:

        efeito_cenario  = a média da run          (compartilhado)
        ruido_politica  = desvio/√200             (independente, o erro da média)

    É modelo, e declarado como tal — mas as duas escalas são medidas, não
    inventadas.
    """
    fora = []
    for f in sorted(glob.glob("reports/maddpg/campanha-a*s*.csv")):
        with open(f, newline="") as fh:
            linhas = list(csv.DictReader(fh))
        if len(linhas) < 20:
            continue
        medias = np.array([float(l["completion_time_medio"]) for l in linhas])
        dp_ep = np.array([float(l["completion_time_std_episodios"]) for l in linhas])
        E = int(linhas[0]["episodios"])
        fora.append((medias, dp_ep / np.sqrt(E)))
    return fora


def cobertura(
    fonte: str,
    execucoes: int,
    n_episodios: int,
    B: int,
    raiz: int,
    independente: bool = False,
) -> dict:
    """Fração das execuções em que o IC da diferença A−A contém zero.

    `independente=True` reproduz o **bug**: reamostra os dois lados com índices
    diferentes. O efeito esperado é IC **largo demais** — a reamostragem
    independente reintroduz a variância entre cenários, que o pareamento
    cancelaria —, então a cobertura sobe acima do nominal. Serve de controle: se
    ela não se afastar de 95 %, o teste não está medindo nada.
    """
    arv = ArvoreSementes(raiz)
    rng = arv.bootstrap(f"aa-{fonte}-{'indep' if independente else 'pareado'}")
    reais = _amostras_maddpg() if fonte == "maddpg" else []
    if fonte == "maddpg" and not reais:
        sys.exit("sem CSVs da campanha; rode scripts/spec02_maddpg_campanha.py")

    contidos = 0
    larguras: list[float] = []
    for e in range(execucoes):
        if fonte == "maddpg":
            medias, erro = reais[e % len(reais)]
            k = rng.integers(0, medias.size, size=n_episodios)
            efeito, sd = medias[k], erro[k]
            a = efeito + rng.normal(0.0, sd)
            b = efeito + rng.normal(0.0, sd)
        else:
            a, b = _par_sintetico(rng, n_episodios)

        if independente:
            ia = rng.integers(0, n_episodios, size=(B, n_episodios))
            ib = rng.integers(0, n_episodios, size=(B, n_episodios))
            deltas = a[ia].mean(axis=1) - b[ib].mean(axis=1)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
        else:
            d = bootstrap_pareado(a, b, rng, B=B)
            lo, hi = d.delta_ic_lo, d.delta_ic_hi

        contidos += int(lo <= 0.0 <= hi)
        larguras.append(hi - lo)

    cob = contidos / execucoes
    # IC binomial de Wilson para a própria cobertura: com 200 execuções o erro
    # padrão é ~1,5 pp, então "94,5 %" e "95,5 %" não são distinguíveis e a
    # tolerância precisa refletir isso.
    z = 1.96
    centro = (contidos + z * z / 2) / (execucoes + z * z)
    meia = z * np.sqrt(cob * (1 - cob) / execucoes + z * z / (4 * execucoes**2)) / (
        1 + z * z / execucoes)
    return {
        "cobertura": cob,
        "ic_cobertura": [float(max(0.0, centro - meia)), float(min(1.0, centro + meia))],
        "largura_media_ic": float(np.mean(larguras)),
        "execucoes": execucoes,
        "n_episodios": n_episodios,
        "B": B,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execucoes", type=int, default=200,
                    help="~200 é o pedido pela T-02.9; erro padrão ~1,5 pp")
    ap.add_argument("--n-episodios", type=int, default=500,
                    help="tamanho da amostra pareada, como na avaliação")
    ap.add_argument("--B", type=int, default=B_CANONICO)
    ap.add_argument("--raiz", type=int, default=20260809)
    ap.add_argument("--fonte", choices=("sintetica", "maddpg", "ambas"),
                    default="ambas")
    ap.add_argument("--saida", default="reports/spec02-aa.json")
    a = ap.parse_args()

    fontes = ("sintetica", "maddpg") if a.fonte == "ambas" else (a.fonte,)
    print(f"Teste A/A · {a.execucoes} execuções × {a.n_episodios} episódios "
          f"× B={a.B} · nominal {NOMINAL:.0%}\n")
    print(f"{'fonte':>12} {'reamostragem':>14} {'cobertura':>11} "
          f"{'IC da cobertura':>18} {'largura do IC':>14}")

    artefato = {"config": vars(a), "nominal": NOMINAL, "resultados": {}}
    veredito_ok = True

    for fonte in fontes:
        for indep in (False, True):
            try:
                r = cobertura(fonte, a.execucoes, a.n_episodios, a.B, a.raiz, indep)
            except SystemExit as e:
                print(f"{fonte:>12} {'—':>14}   {e}")
                continue
            rotulo = "INDEPENDENTE" if indep else "pareada"
            lo, hi = r["ic_cobertura"]
            print(f"{fonte:>12} {rotulo:>14} {r['cobertura']:>10.1%} "
                  f"  [{lo:.1%}, {hi:.1%}] {r['largura_media_ic']:>13.3f}")
            artefato["resultados"][f"{fonte}_{'indep' if indep else 'pareado'}"] = r
            if not indep:
                # Calibrado se o IC da cobertura contém o nominal.
                if not (lo <= NOMINAL <= hi):
                    veredito_ok = False

    print()
    # O contra-exemplo precisa FALHAR, senão o teste não discrimina.
    contras = [v for k, v in artefato["resultados"].items() if k.endswith("indep")]
    contra_falha = all(not (v["ic_cobertura"][0] <= NOMINAL <= v["ic_cobertura"][1])
                       for v in contras) if contras else False

    if veredito_ok:
        print("CALIBRADO: o IC da cobertura contém 95 % em todas as fontes.")
        print("  -> os ICs pareados já reportados no projeto se sustentam.")
    else:
        print("DESCALIBRADO: a cobertura não bate o nominal.")
        print("  -> TODOS os ICs pareados do repositório precisam de revisão.")

    if contra_falha:
        print("\nO contra-exemplo (reamostragem independente) FALHA, como deve —")
        print("logo o teste está de fato discriminando, e não passando por acaso.")
    elif contras:
        print("\nATENÇÃO: o contra-exemplo passou. O teste não está discriminando;")
        print("aumente --execucoes ou --n-episodios antes de confiar no veredito.")

    artefato["veredito"] = {"calibrado": veredito_ok,
                           "contraexemplo_falha": contra_falha}
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.saida, "w") as f:
        json.dump(artefato, f, indent=2)
    print(f"\nartefato: {a.saida}")
    return 0 if (veredito_ok and contra_falha) else 1


if __name__ == "__main__":
    sys.exit(main())
