"""Grade do braço **PPO** — a rede do `anbubus` com o algoritmo dela, `Na = 1`.

POR QUE ESTA GRADE EXISTE

As 720 corridas do `spec06_busca.py` compararam as duas **redes** sob o **Algoritmo
4.4**, que é o algoritmo da dissertação. A rede do `anbubus` é um aproximador de
função para PPO: o `experiment.py` de todas as oito tarefas publicadas roda
`tune.run("PPO", …)`. Rodá-la sob o Algoritmo 4.4 a deixa sem nenhum dos mecanismos
que a mantinham estável:

    recorte do objetivo substituto   `clip_param = 0,3`; o Alg 4.4 não tem limite
    GAE(λ)                          contra o nosso `δ = G − V` de `m` passos
    atualização em lote             ~8 passos de gradiente por 1000 do ambiente,
                                    contra 1 por transição — 125× a mais
    σ APRENDIDA                     `policy_output_dim = 2×dim_ação`, média e log σ;
                                    nós usamos σ fixa, varrida como hiperparâmetro

Isso explica o resultado que motivou esta grade: no orçamento de 10 M, **as seis
corridas do anbubus abortaram**, inclusive a que tinha 0/3 abortos a 200 mil passos.
Mas o ponto "seguro" dela saiu de 19,2 % para **77,6 % de cobertura e SR 55 %** antes
de divergir. A rede aprende; ela não sustenta. Uma σ aprendida é justamente a válvula
que falta: quando o gradiente cresce, o PPO pode AUMENTAR σ, o que encolhe
`∇log π = (z − f)/σ²`. Com σ fixa não há válvula.

O ESPAÇO, E POR QUE ESTES EIXOS

Os valores são os publicados, não inventados. Das oito configurações, as duas de ação
CONTÍNUA são a referência do nosso problema:

    tarefa    lr     γ     λ    lote  minilote  n   T    vf_clip
    hopper    4e-4   0,98  —    1000  512       8   32   3000
    reacher   7e-5   0,99  0,9  1000  128       10  300  (padrão 10)

`lr` varre os três valores publicados — 7e-5, 4e-4 e 3e-3 (as duas discretas) —, uma
faixa de 43×, que é a maior dispersão que eles próprios reportam.

`T` varre 32 e 300, os das duas contínuas. Com `obs_dim = 25`, isso é 800 contra
7 500 bits, e com `n = 10` são 80 contra 750 RAMs. Diferença de capacidade de ~10×.

`vf_clip_param` é eixo, e a razão foi MEDIDA: o reacher usa 10, e os nossos retornos
sob a eq. 5-6 valem ~-59 (faixa -77 a -41, oito episódios com política aleatória).
Um recorte em ±10 mutilaria o alvo do crítico. O hopper usa 3000 porque tem retornos
grandes — é o análogo do nosso caso, e é isso que o eixo testa.

FIXOS, nos valores do reacher: `n = 10`, `λ = 0,9`, minilote 128, lote 1000,
`num_epochs = 1` (invariante nas oito), `MeanStdFilter` (invariante nas oito) e
`γ = 0,99` — que é o desconto que Lamenza fixa por autoridade e coincide com o do
reacher, então não há conflito a resolver.

    3 × 2 × 2 = 12 pontos × 2 sementes = 24 corridas

ORÇAMENTO: 1 000 000 de passos, que é o `timesteps_total` das oito configurações
publicadas. Não é escolha nossa.

A SELEÇÃO É PELA MESMA MÉTRICA DOS OUTROS BRAÇOS

`--sementes-aval` faz cada corrida terminar com avaliação **determinística** no banco
`val`, devolvendo cobertura, SR e CT. Sem isso a grade selecionaria por retorno de
treino — com ruído de exploração ligado — enquanto os braços do Algoritmo 4.4
selecionam por cobertura determinística, e a comparação mediria dois critérios de
seleção em vez de dois métodos.

DOIS AMBIENTES, E NÃO POR ESCOLHA

`ray[rllib]` fixa `gymnasium <= 1.2.2` e o projeto exige `>= 1.3.0`. Este driver roda
no ambiente do projeto e lança cada corrida como **subprocesso** num ambiente
efêmero. É a mesma divisão que a RNF-11 já pedia.

    uv run python scripts/spec06_grid_ppo.py --sementes 2 --passos 1000000 --processos 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

# --- o espaço ---------------------------------------------------------------

LR = (7e-5, 4e-4, 3e-3)
"""Os três valores publicados: reacher, hopper e as duas discretas."""

RESOLUCAO = (32, 300)
"""`T` do hopper e do reacher. Com `obs_dim = 25`: 800 e 7 500 bits."""

VF_CLIP = (10.0, 3000.0)
"""Padrão do RLlib contra o do hopper. Os nossos retornos (~-59) excedem ±10."""

FIXOS = dict(referencia="reacher", tupla=10, gamma=0.99)
"""`n = 10`, `λ = 0,9`, minilote 128 e lote 1000 vêm da base `reacher`; `γ = 0,99` é
o de Lamenza e coincide com o dela."""


def pontos() -> list[dict]:
    """Produto cartesiano dos três eixos."""
    return [{"id": i, "lr": lr, "resolucao": T, "vf_clip": vf}
            for i, (lr, T, vf) in enumerate(itertools.product(LR, RESOLUCAO, VF_CLIP))]


def _comando(p: dict, seed: int, a, bruto: Path, saida: Path) -> list[str]:
    """A invocação do ambiente efêmero. `--isolated` para não herdar o do projeto."""
    return ["uv", "run", "--isolated", "--with", "ray[rllib]", "--with", "torch",
            "--with", "pettingzoo", "python",
            str(RAIZ / "scripts" / "spec06_ppo_anbubus.py"),
            "--celula", f"{a.na}:{a.ns}",
            "--passos", str(a.passos),
            "--seed", str(seed),
            "--referencia", FIXOS["referencia"],
            "--lr", repr(p["lr"]),
            "--gamma", repr(FIXOS["gamma"]),
            "--tupla", str(FIXOS["tupla"]),
            "--resolucao", str(p["resolucao"]),
            "--vf-clip", repr(p["vf_clip"]),
            "--ray-cpus", str(a.ray_cpus),
            "--sementes-aval", str(a.sementes_aval),
            "--bruto", str(bruto),
            "--saida", str(saida)]


def uma(tarefa: tuple) -> dict:
    """Roda um ponto e devolve a linha do registro. Nunca levanta.

    Um subprocesso que falha é **dado**, não interrupção: a grade tem de registrar
    qual configuração não roda e seguir. Levantar aqui perderia as outras corridas do
    lote por causa de uma.
    """
    p, seed, a = tarefa
    rot = f"lr{p['lr']:.0e}-T{p['resolucao']}-vf{int(p['vf_clip'])}-s{seed}"
    dest = Path(a.raiz) / rot
    dest.mkdir(parents=True, exist_ok=True)
    bruto, saida = dest / "bruto.npz", dest / "artefato.json"
    t0 = time.perf_counter()
    proc = subprocess.run(_comando(p, seed, a, bruto, saida),
                          cwd=str(RAIZ), capture_output=True, text=True)
    linha = {**{k: v for k, v in p.items()}, "seed": seed, "rotulo": rot,
             "passos": a.passos, "parede_s": time.perf_counter() - t0,
             "dir": str(dest)}
    if proc.returncode != 0 or not saida.exists():
        # As últimas linhas do stderr, não todas: o RLlib despeja centenas de avisos
        # de depreciação e o que interessa é a exceção final.
        cauda = "\n".join(proc.stderr.strip().splitlines()[-6:])
        return {**linha, "falhou": f"código {proc.returncode}", "stderr": cauda,
                "cobertura": 0.0, "sr": 0.0}
    art = json.loads(saida.read_text())
    aval = art.get("avaliacao") or {}
    hist = art.get("historico") or []
    return {**linha, "falhou": None,
            "cobertura": aval.get("cobertura", 0.0), "sr": aval.get("sr", 0.0),
            "ct_cens": aval.get("ct_cens"), "ct_succ": aval.get("ct_succ"),
            "retorno_final": hist[-1]["retorno"] if hist else None,
            "hiperparametros": art.get("hiperparametros")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--na", type=int, default=1)
    ap.add_argument("--ns", type=int, default=12)
    ap.add_argument("--sementes", type=int, default=2)
    ap.add_argument("--passos", type=int, default=1_000_000,
                    help="`timesteps_total` das oito configurações publicadas")
    ap.add_argument("--n-val", type=int, default=20,
                    help="episódios do banco val na avaliação determinística")
    ap.add_argument("--processos", type=int, default=4,
                    help="corridas simultâneas; cada uma reserva --ray-cpus")
    ap.add_argument("--ray-cpus", type=int, default=2)
    ap.add_argument("--raiz", default="runs/spec06-ppo")
    ap.add_argument("--saida", default="reports/spec06-ppo.json")
    ap.add_argument("--retomar", action="store_true")
    a = ap.parse_args()

    # As sementes do banco `val` num arquivo, porque o ambiente efêmero não importa
    # `uavwnn`. É o MESMO banco que os braços do Algoritmo 4.4 usam para selecionar.
    from uavwnn.campanha import carregar_bancos
    bancos = carregar_bancos(a.na, a.ns)
    a.sementes_aval = Path(a.raiz) / "sementes-val.txt"
    a.sementes_aval.parent.mkdir(parents=True, exist_ok=True)
    a.sementes_aval.write_text(
        "\n".join(str(int(s)) for s in bancos.val.sementes[:a.n_val]) + "\n")

    tarefas = [(p, sd, a) for p in pontos() for sd in range(a.sementes)]
    progresso = Path(a.saida).with_suffix(".progresso.jsonl")
    progresso.parent.mkdir(parents=True, exist_ok=True)

    res = []
    if a.retomar and progresso.exists():
        vistos = set()
        for l in progresso.read_text().splitlines():
            if not l.strip():
                continue
            try:
                x = json.loads(l)
            except json.JSONDecodeError:
                continue          # linha cortada por um Ctrl-C no meio da escrita
            if x.get("passos") == a.passos:
                res.append(x)
                vistos.add(x["rotulo"])
        antes = len(tarefas)
        tarefas = [t for t in tarefas
                   if f"lr{t[0]['lr']:.0e}-T{t[0]['resolucao']}-"
                      f"vf{int(t[0]['vf_clip'])}-s{t[1]}" not in vistos]
        print(f"RETOMANDO: {len(res)} de {antes} aproveitadas · {len(tarefas)} a fazer")
    else:
        progresso.write_text("")

    print(f"grade PPO · rede do anbubus com o algoritmo dela · Na={a.na} Ns={a.ns}")
    print(f"{len(pontos())} pontos × {a.sementes} sementes = "
          f"{len(pontos()) * a.sementes} corridas · {a.passos:,} passos")
    print(f"eixos: lr{LR} × T{RESOLUCAO} × vf_clip{VF_CLIP}")
    print(f"fixos: {FIXOS} · λ=0,9 · minilote 128 · lote 1000 · MeanStdFilter")
    print(f"{a.processos} simultâneas × {a.ray_cpus} CPUs cada\n")
    if not tarefas:
        print("nada a fazer.")

    t0 = time.perf_counter()
    # Threads e não processos: o trabalho está nos SUBPROCESSOS, então o GIL não
    # atrapalha — cada thread só espera um `subprocess.run`.
    with ThreadPoolExecutor(max_workers=a.processos) as pool:
        for i, x in enumerate(pool.map(uma, tarefas), 1):
            res.append(x)
            with open(progresso, "a") as f:
                f.write(json.dumps(x) + "\n")
            estado = (f"FALHOU {x['falhou']}" if x["falhou"]
                      else f"cobertura {x['cobertura']:.1%} SR {x['sr']:.1%}")
            print(f"  [{i:>3}/{len(tarefas)}] {x['rotulo']:<28} {estado} "
                  f"· {x['parede_s'] / 60:.1f} min", flush=True)

    parede = time.perf_counter() - t0
    Path(a.saida).write_text(json.dumps(
        {"espaco": {"lr": LR, "resolucao": RESOLUCAO, "vf_clip": VF_CLIP,
                    "fixos": FIXOS},
         "config": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(a).items()},
         "parede_s": parede, "corridas": res}, indent=2, ensure_ascii=False))

    vivas = [x for x in res if not x["falhou"]]
    print(f"\nparede: {parede / 3600:.1f} h · {len(vivas)}/{len(res)} sem falha")
    if vivas:
        melhor = max(vivas, key=lambda x: x["cobertura"])
        print(f"melhor: {melhor['rotulo']} -> cobertura {melhor['cobertura']:.1%} "
              f"SR {melhor['sr']:.1%}")
        print("referências: aleatório 3,3 % de cobertura · guloso SR 100 %, CT ≈ 34 s")
        print("Alg 4.4: dissertação 50,1 % cob / SR 21,3 % · anbubus 46,6 % / 10,3 %")
    print(f"artefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
