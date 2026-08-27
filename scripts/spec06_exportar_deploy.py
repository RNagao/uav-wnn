"""Exporta o artefato de IMPLANTAÇÃO de um modelo WNN.

O que está em `models/` é o checkpoint de TREINO: ator + crítico, endereços em
`uint64` e valores em `float64`. Para voar, nada disso é necessário:

  * o **crítico** só existe para treinar (ele estima "quão boa é esta situação"
    e nunca escolhe ação) — 32 % do pacote, inútil no deploy;
  * o **endereço** tem `tamanho_tupla` bits (12 → 4096 valores), então `uint16`
    basta e o `uint64` desperdiça 4×;
  * os **valores** cabem em `float32`, que é a precisão que o modelo de custo da
    SPEC-03 já assume ao reportar o T1.

Medido no ator de 2 drones: 4722 KB → 1773 KB, 2,7× menor.

O `float32` MUDA as predições nos últimos bits (erro relativo máximo 5,9e-08,
mediano 2,1e-08 — dentro do eps do `float32`, 1,2e-07). Por isso este script
**verifica** antes de declarar sucesso: compara as ações do ator nos dois
formatos sobre observações reais e recusa exportar se a diferença passar do
tolerado.

    uv run python scripts/spec06_exportar_deploy.py models/T32/a2s24 --na 2 --ns 24
    uv run python scripts/spec06_exportar_deploy.py --todos
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")

RAIZ = Path(__file__).resolve().parents[1]
TOL_REL = 1e-6
"""Tolerância na AÇÃO, não no peso. `float32` erra ~6e-08 por valor; a ação passa
por uma soma de `N` termos e uma divisão, então o erro se acumula um pouco. 1e-6
dá três ordens de folga sobre o erro por valor e ainda é 10⁴ vezes menor que a
menor diferença de ação que o ambiente distingue."""


def exportar(origem: Path, destino: Path, poda: float = 0.0) -> dict:
    """Grava só o ator, com `uint16`/`float32`. Devolve o resumo de tamanhos.

    `poda` remove a fração pedida das entradas de menor |v|. A base empírica
    (500 sementes, banco de teste): na convenção de média cada entrada contribui
    `v/N` à saída, então a cauda de |v| pequeno quase não move a ação — 50 % da
    massa está em 6,3 % das entradas. Medido: podar 70 % custa 0,8 pp de SR no
    T16 (96,8→96,0 %) e nada no T32; 90 % custa 3 pp no T16. O produto continua
    verificado POR COMPORTAMENTO antes de publicado.
    """
    fonte = next(origem.glob("*ator.npz"))
    z = np.load(fonte, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    n = int(cfg["tamanho_tupla"])
    if n > 16:
        raise SystemExit(f"tamanho_tupla={n} não cabe em uint16 — ajuste o dtype")

    end = z["enderecos"]
    if end.size and int(end.max()) >= (1 << 16):
        raise SystemExit(f"endereço máximo {int(end.max())} não cabe em uint16")

    destino.mkdir(parents=True, exist_ok=True)
    saida = {k: z[k] for k in z.files}
    if poda > 0:
        # Os vetores são paralelos por RAM (`tamanhos_ram` diz onde cada uma
        # começa); a poda é GLOBAL por |v|, então refazemos os três coerentes.
        d = int(cfg["n_saidas"])
        vals = z["valores"].reshape(-1, d)
        tam = z["tamanhos_ram"]
        ram_de = np.repeat(np.arange(tam.size), tam)
        mag = np.abs(vals).max(axis=1)
        manter = np.ones(mag.size, bool)
        manter[np.argsort(mag)[: int(mag.size * poda)]] = False
        saida["enderecos"] = end[manter]
        # MANTÉM a forma (entradas, d): o `carregar` fatia `valores[j]` como o
        # vetor da entrada `j` — achatar para 1-D fazia cada fatia virar um
        # escalar, a rede lia lixo, e o SR ia a zero (medido: 97,5 % → 0,0 %).
        saida["valores"] = vals[manter]
        saida["tamanhos_ram"] = np.bincount(
            ram_de[manter], minlength=tam.size).astype(np.int64)
        end = saida["enderecos"]
    saida["enderecos"] = end.astype(np.uint16)
    saida["valores"] = np.asarray(saida["valores"]).astype(np.float32)
    # `mapeamento` indexa bits de entrada (n_bits ≤ 1248 aqui) — uint16 basta.
    saida["mapeamento"] = z["mapeamento"].astype(np.uint16)
    alvo = destino / fonte.name
    np.savez_compressed(alvo, **saida)
    for extra in ("termometro.npz", "familia.txt", "procedencia.json", "config.json"):
        f = origem / extra
        if f.exists():
            (destino / extra).write_bytes(f.read_bytes())

    antes = sum(f.stat().st_size for f in origem.iterdir() if f.is_file())
    depois = sum(f.stat().st_size for f in destino.iterdir() if f.is_file())
    return {"origem_kb": antes / 1024, "deploy_kb": depois / 1024,
            "reducao": antes / max(depois, 1)}


def verificar(origem: Path, destino: Path, na: int, ns: int, n_obs: int = 400) -> dict:
    """Compara as AÇÕES dos dois formatos sobre observações reais do ambiente.

    Observações reais, não vetores aleatórios: o custo e o resultado do WNN
    dependem de QUAIS endereços são tocados, e ruído gaussiano tocaria endereços
    vazios — mediria o caso trivial em que os dois formatos concordam por serem
    ambos zero.
    """
    import logging
    from spec06_campanha_wnn import montar
    from uavwnn.refenv import LAMENZA_003, make_env, prune_logging_handlers

    cfg = LAMENZA_003(na, ns, phase="eval")
    H = list(logging.getLogger().handlers)
    p64, _ = montar("ntupla_dissertacao", cfg, origem)
    p32, _ = montar("ntupla_dissertacao", cfg, destino)

    difs, vistos = [], 0
    for s in range(50):
        random.seed(s); np.random.seed(s)
        env = make_env(cfg); obs, _ = env.reset(); prune_logging_handlers(H)
        for _ in range(60):
            for a in env.agents:
                x = obs[a]
                d = np.abs(np.asarray(p64(x), float) - np.asarray(p32(x), float))
                difs.append(d.max()); vistos += 1
            if vistos >= n_obs:
                break
            obs, _, term, trunc, infos = env.step(
                {a: p64(obs[a]) for a in env.agents})
            if (infos and "avg_reward" in next(iter(infos.values()))) \
               or any(term.values()) or any(trunc.values()):
                break
        try:
            env.close()
        except Exception:                                    # pragma: no cover
            pass
        if vistos >= n_obs:
            break
    d = np.array(difs)
    return {"n_obs": int(d.size), "dif_max": float(d.max()),
            "dif_mediana": float(np.median(d)), "ok": bool(d.max() <= TOL_REL)}


TOL_SR_PP = 2.0
"""Queda de SR tolerada num export PODADO, em pontos percentuais. Com 200
sementes o ruído binomial é ~±2,8 pp em SR 90 %; abaixo de 2 pp a queda não é
distinguível do ruído, acima é sinal de poda excessiva."""


def verificar_comportamento(origem: Path, destino: Path, na: int, ns: int,
                            n_sementes: int, sementes=None) -> dict:
    """SR dos dois artefatos — a verificação de um export PODADO, cuja ação muda
    por construção.

    `sementes` explícitas têm precedência; sem elas, usa o banco de TESTE (o
    caso da CLI, pós-hoc). Os treinadores passam o banco de VAL: a poda é uma
    decisão sobre o modelo, e decisões não podem consumir o teste.
    """
    import logging
    import random
    from spec06_campanha_wnn import montar
    from uavwnn.campanha import carregar_bancos
    from uavwnn.refenv import LAMENZA_003, make_env, prune_logging_handlers

    cfg = LAMENZA_003(na, ns, phase="eval")
    H = list(logging.getLogger().handlers)
    if sementes is None:
        sementes = carregar_bancos(na, ns).test.sementes[:n_sementes]

    def sr_de(caminho):
        pol, _ = montar("ntupla_dissertacao", cfg, caminho)
        oks = []
        for s in sementes:
            random.seed(int(s)); np.random.seed(int(s))
            env = make_env(cfg); obs, _ = env.reset()
            prune_logging_handlers(H)
            ok = False
            for _ in range(int(cfg.max_episode_length
                                / cfg.algorithm_iteration_interval) + 2):
                obs, _, term, trunc, infos = env.step(
                    {a: pol(obs[a]) for a in env.agents})
                if infos and "avg_reward" in next(iter(infos.values())):
                    ok = bool(next(iter(infos.values())).get("all_collected", False))
                    break
                if any(term.values()) or any(trunc.values()):
                    break
            oks.append(ok)
            try:
                env.close()
            except Exception:                                # pragma: no cover
                pass
        return float(np.mean(oks))

    cheio, podado = sr_de(origem), sr_de(destino)
    return {"sr_cheio": cheio, "sr_podado": podado,
            "ok": (cheio - podado) * 100 <= TOL_SR_PP}


PODA_PADRAO = 0.7
"""Fração padrão do pós-treino. Medido com 500 sementes: 70 % custa 0,8 pp de SR
na política forte (dentro do ruído) e nada na fraca; 90 % já custa 3 pp. Ver
`reports/spec06-memoria-wnn.md`."""


def deploy_pos_treino(final: Path, na: int, ns: int, sementes,
                      poda: float = PODA_PADRAO) -> dict:
    """O passo padrão pós-treino: grava `<checkpoints>/deploy` podado e VERIFICADO.

    Três invariantes:
      * o checkpoint de treino fica INTACTO — de rede podada não se retoma
        treino, e o crítico segue só no `final`/`melhor`;
      * a verificação usa as `sementes` dadas (o banco de VAL do treinador) —
        a poda é decisão sobre o modelo e não pode consumir o teste;
      * se o SR cair além de `TOL_SR_PP`, o deploy é regravado SEM poda
        (só f32/u16) — o passo nunca piora o artefato publicado, e o motivo
        fica no dicionário devolvido (vai para a procedência).
    """
    destino = final.parent / "deploy"
    r = exportar(final, destino, poda)
    v = verificar_comportamento(final, destino, na, ns, 0, sementes=sementes)
    if not v["ok"]:
        r = exportar(final, destino, 0.0)
        return {**r, **v, "poda": 0.0,
                "motivo": f"poda {poda:.0%} derrubou SR "
                          f"{v['sr_cheio']:.1%}→{v['sr_podado']:.1%}; deploy sem poda"}
    return {**r, **v, "poda": poda}


# ---------------------------- promoção dos 50 M -----------------------------

ALVO_PASSOS = 50_000_000
"""Decisão de projeto (2026-08-26): o checkpoint promovido é o `final` da
corrida de 50 M — sem seleção pelo melhor ponto da curva de validação, que
oscila SR 0,90–0,99 num platô desde ~13 M e faria de qualquer escolha um
sorteio da mesma banda. O `final` existe em todas as corridas e não
reintroduz seleção sobre validação."""


def promover(cel: str) -> None:
    """Instala o `final` de `runs/spec06-50M/<cel>` em `models/`.

    Na ordem: recusa corrida incompleta (promover parcial por engano foi o que
    já misturou orçamentos numa comparação); avalia o `final` no `test` e grava
    `metrica_test_final` — o `metrica_test` do treinador é do MELHOR (fix
    539b61e), não do artefato promovido; arquiva o 6,75 M em `models/6M75/`;
    instala cheio + podado com a procedência junto.
    """
    import shutil

    t_res = int(cel.split("-")[0][1:])
    a = cel.split("-")[1]
    na, ns = int(a[1:a.index("s")]), int(a[a.index("s") + 1:])
    run = RAIZ / "runs" / "spec06-50M" / cel
    if not (run / "procedencia.json").exists():
        raise SystemExit(f"{cel}: sem procedencia.json — a corrida ainda está "
                         f"em curso (o treinador só o escreve ao terminar)")
    proc = json.loads((run / "procedencia.json").read_text())
    if int(proc.get("passos", 0)) < ALVO_PASSOS:
        raise SystemExit(f"{cel}: só {proc.get('passos', 0):,} passos — "
                         f"corrida incompleta NÃO é promovida")

    final = run / "checkpoints" / "final"
    deploy = run / "checkpoints" / "deploy"
    for d in (final, deploy):
        if not d.exists():
            raise SystemExit(f"{cel}: falta {d}")

    for raiz in (RAIZ / "models", RAIZ / "models" / "deploy"):
        arq = RAIZ / "models" / "6M75" / (raiz / f"T{t_res}" / f"a{na}s{ns}"
                                          ).relative_to(RAIZ / "models")
        if arq.exists():
            raise SystemExit(f"{cel}: {arq.relative_to(RAIZ)} já existe — "
                             f"promovido antes; não sobrescrevo")

    from spec06_treino_multiagente import _cfgs, _ctx, avaliar_multi, construir
    from uavwnn.campanha import carregar_bancos
    print(f"[{cel}] avaliando o final no test (100 sementes)…", flush=True)
    cfg_tr, cfg_av = _cfgs("1b", na, ns)
    ap, termo, _ = construir(cfg_tr, na, t_res=t_res, retomar=final)
    test = carregar_bancos(na, ns).test.sementes[:100]
    m = avaliar_multi(cfg_av, ap, termo, _ctx(cfg_av), test)
    proc["metrica_test_final"] = {k: (float(v) if isinstance(v, float) else v)
                                  for k, v in m.items()}
    proc["promovido_de"] = str(run.relative_to(RAIZ))
    print(f"[{cel}] final@test: {proc['metrica_test_final']}", flush=True)

    for raiz, novo in ((RAIZ / "models", final),
                       (RAIZ / "models" / "deploy", deploy)):
        atual = raiz / f"T{t_res}" / f"a{na}s{ns}"
        arquivo = RAIZ / "models" / "6M75" / atual.relative_to(RAIZ / "models")
        if atual.exists():
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            if arquivo.exists():
                raise SystemExit(f"{cel}: {arquivo} já existe — não sobrescrevo")
            shutil.move(str(atual), str(arquivo))
            print(f"[{cel}] arquivado: {atual.relative_to(RAIZ)} -> "
                  f"{arquivo.relative_to(RAIZ)}", flush=True)
        shutil.copytree(novo, atual)
        (atual / "procedencia.json").write_text(json.dumps(proc, indent=2))
        print(f"[{cel}] instalado: {atual.relative_to(RAIZ)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origem", nargs="?", type=Path)
    ap.add_argument("--na", type=int); ap.add_argument("--ns", type=int)
    ap.add_argument("--todos", action="store_true",
                    help="exporta models/T*/a*s* para models/deploy/")
    ap.add_argument("--destino", type=Path, default=None)
    ap.add_argument("--poda", type=float, default=0.0,
                    help="fração das entradas de menor |v| a REMOVER do ator "
                         "(0.7 = remove 70%%). Muda a política de propósito, então "
                         "a verificação passa a ser por COMPORTAMENTO (SR no banco "
                         "de teste), não por igualdade de ação.")
    ap.add_argument("--sementes-verif", type=int, default=200)
    ap.add_argument("--promover", nargs="+", metavar="CEL", default=None,
                    help="promove corridas de 50 M (ex.: T16-a1s12 T32-a2s24): "
                         "instala o `final` em models/ e arquiva o 6,75 M")
    a = ap.parse_args()
    if a.promover:
        for cel in a.promover:
            promover(cel)
        return 0

    if a.todos:
        alvos = []
        for d in sorted((RAIZ / "models").glob("T*/a*s*")):
            if "em-treino" in str(d) or not any(d.glob("*ator.npz")):
                continue
            m = d.name  # aNsM
            na = int(m.split("s")[0][1:]); ns = int(m.split("s")[1])
            alvos.append((d, RAIZ / "models" / "deploy" / d.parent.name / d.name, na, ns))
    else:
        if a.origem is None or a.na is None or a.ns is None:
            sys.exit("informe origem --na --ns, ou use --todos")
        dst = a.destino or (RAIZ / "models" / "deploy" / a.origem.parent.name / a.origem.name)
        alvos = [(a.origem, dst, a.na, a.ns)]

    falhou = False
    if a.poda:
        print(f"{'modelo':<22}{'treino':>10}{'deploy':>10}{'redução':>9}"
              f"{'SR cheio':>10}{'SR podado':>11}  veredito")
    else:
        print(f"{'modelo':<22}{'treino':>10}{'deploy':>10}{'redução':>9}"
              f"{'dif. máx':>12}  veredito")
    for origem, destino, na, ns in alvos:
        t = exportar(origem, destino, a.poda)
        rot = f"{origem.parent.name}/{origem.name}"
        if a.poda:
            v = verificar_comportamento(origem, destino, na, ns, a.sementes_verif)
            if not v["ok"]:
                # FALLBACK, como no pós-treino: um artefato reprovado não pode
                # ficar no disco parecendo publicável. Regrava sem poda — o
                # export nunca piora o que seria publicado.
                t = exportar(origem, destino, 0.0)
            print(f"{rot:<22}{t['origem_kb']:>8.0f}K{t['deploy_kb']:>9.0f}K"
                  f"{t['reducao']:>8.1f}×{v['sr_cheio']:>9.1%}{v['sr_podado']:>10.1%}  "
                  f"{'OK' if v['ok'] else f'CAIU >{TOL_SR_PP:.0f}pp → deploy SEM poda'}")
        else:
            v = verificar(origem, destino, na, ns)
            falhou |= not v["ok"]
            print(f"{rot:<22}{t['origem_kb']:>8.0f}K{t['deploy_kb']:>9.0f}K"
                  f"{t['reducao']:>8.1f}×{v['dif_max']:>12.2e}  "
                  f"{'OK' if v['ok'] else 'FALHOU (> ' + f'{TOL_REL:.0e}' + ')'}")
    if falhou and not a.poda:
        print("\nalgum export mudou a ação além do tolerado — NÃO publique estes.")
        return 1
    if a.poda:
        print(f"\n-> {len(alvos)} modelos PODADOS ({a.poda:.0%}) — SR verificado em "
              f"{a.sementes_verif} sementes, queda ≤ {TOL_SR_PP:.0f} pp")
    else:
        print(f"\n-> models/deploy/  ({len(alvos)} modelos, ação idêntica até {TOL_REL:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
