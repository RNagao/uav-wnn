"""Fila da campanha comparável com escolha LIVRE de modelos × células × regimes.

A `spec06_campanha_fila.sh` codifica UMA campanha — MADDPG × WNN T16 × WNN T32 em
`Na = 1` e `Na = 2`, com a WNN treinada em `Ns = 24` — e é isso que a torna
reprodutível: ela continua existindo, intacta, com o supervisor que a acompanha.

Esta fila existe para o passo seguinte: um modelo por `Na` (grade própria, treino
longo revezando `Ns`), e a pergunta "este modelo, nestas células, nestes regimes".
Cada corrida é a MESMA `spec06_campanha_completa.py` de sempre — mesmas sementes,
mesmo CSV, mesmo cache por run —, só a montagem da fila muda.

MODELOS, e como cada um acha o seu checkpoint

    maddpg                 os checkpoints publicados de Lamenza (não há Na=1)
    wnn-T16 / wnn-T32      `<raiz-modelos>/T<T>/a<Na>s*`: a pasta EXATA da célula
                           se existir, senão a ÚNICA pasta daquele Na. Duas
                           candidatas (ex.: `a2s24` e `a2s12-24-36`) é ambiguidade,
                           e a fila recusa em vez de escolher — use a forma abaixo.
    <rótulo>=<caminho>     explícito; `{na}` no caminho é substituído pelo Na da
                           célula, ex.: `wnn-g2-T16=models/deploy/T16/a{na}s12-24-36`

O RÓTULO É A IDENTIDADE DO CSV, e o antigo não pode contaminar o novo

`<rótulo>-a<Na>s<Ns>-<regime>.csv` é onde a corrida grava e de onde ela RETOMA. Se
um modelo novo for lançado com o rótulo de um antigo, a corrida acha o cache cheio
e não roda nada — e a tabela final atribui ao modelo novo os números do velho, sem
erro nenhum. Por isso cada lançamento grava um `.manifest.json` ao lado do log com
o checkpoint usado, e a fila RECUSA relançar um CSV cujo manifesto aponta para
outro checkpoint. `--forcar` desliga a guarda, para quem sabe o que está fazendo.

`treinado_em_ns` vem da procedência do checkpoint (um inteiro, ou a lista do
revezamento, gravada como `12-24-36`); a célula conta como transferência quando o
seu `Ns` não está nessa lista.

    uv run python scripts/spec06_campanha_fila.py --listar
    uv run python scripts/spec06_campanha_fila.py --seco \\
        --modelos wnn-g2-T16=models/deploy/T16/a{na}s12-24-36,maddpg \\
        --celulas 2:12,24,36 --regimes limpo,sem-kf
    nohup uv run python scripts/spec06_campanha_fila.py --celulas 2:12,24,36 \\
        --paralelo 4 > reports/campanha/logs/_fila.log 2>&1 &
    uv run python scripts/spec06_campanha_fila.py --parar
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PGID = RAIZ / ".campanha-fila.pgid"
REGIMES = {
    # regime -> (ruido, degraus)  — os mesmos três da fila em shell
    "limpo": ("limpo", None),
    "sem-kf": ("sorteio", "1"),
    "com-kf": ("sorteio", "4"),
}


def _rel(p: Path) -> str:
    """Relativo à raiz do projeto quando está dentro dela; absoluto quando não —
    `--raiz-modelos` e `--gates` podem apontar para outro checkout."""
    try:
        return str(p.relative_to(RAIZ))
    except ValueError:
        return str(p)


def _sufixo(regime: str, sigmas: str) -> str:
    """O trecho do nome do CSV que `spec06_campanha_completa.py` produz."""
    if regime == "limpo":
        return "limpo"
    return f"sigma{sigmas.replace(',', '-')}-{regime}"


def _ler_lista(txt: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(v) for v in str(txt).replace("-", ",").split(",")
                               if v.strip()))


def _treinado_em(ckpt: Path) -> str:
    """`24` ou `12-24-36`, da procedência; senão, do nome da pasta (`a2s12-24-36`)."""
    proc = ckpt / "procedencia.json"
    if proc.exists():
        ns = json.loads(proc.read_text()).get("config", {}).get("ns")
        if ns is not None:
            return "-".join(map(str, ns)) if isinstance(ns, list) else str(ns)
    nome = ckpt.name
    return nome[nome.index("s") + 1:] if nome.startswith("a") and "s" in nome else "?"


def _checkpoint_wnn(raiz: Path, t: int, na: int, ns: int) -> Path | str:
    """A pasta do modelo para a célula, ou a razão de não haver uma."""
    base = raiz / f"T{t}"
    exata = base / f"a{na}s{ns}"
    if exata.is_dir():
        return exata
    cands = sorted(p for p in base.glob(f"a{na}s*")
                   if p.is_dir() and "em-treino" not in p.name)
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return f"sem modelo em {_rel(base)}/a{na}s*"
    return ("ambíguo: " + ", ".join(p.name for p in cands)
            + " — escolha com <rótulo>=<caminho>")


@dataclass
class Corrida:
    rotulo: str
    modelo: str                 # "maddpg" | "wnn"
    na: int
    ns: int
    regime: str
    checkpoint: str | None      # caminho, ou None (maddpg)
    treinado_em_ns: str | None
    gate: str | None
    csv: str
    log: str
    manifesto: str
    bloqueio: str | None        # por que não roda, se não roda

    @property
    def transferencia(self) -> bool:
        return (self.treinado_em_ns is not None and self.treinado_em_ns != "?"
                and self.ns not in _ler_lista(self.treinado_em_ns))


def montar_fila(a) -> list[Corrida]:
    celulas = []
    for item in a.celulas:
        na_txt, ns_txt = item.split(":")
        for ns in _ler_lista(ns_txt):
            celulas.append((int(na_txt), ns))
    regimes = [r.strip() for r in a.regimes.split(",") if r.strip()]
    for r in regimes:
        if r not in REGIMES:
            sys.exit(f"regime desconhecido: {r} (use {', '.join(REGIMES)})")
    fila = []
    for regime in regimes:
        suf = _sufixo(regime, a.sigmas)
        for na, ns in celulas:
            for spec in a.modelos.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                bloqueio = ckpt = None
                if spec == "maddpg":
                    rot, mod = "maddpg", "maddpg"
                    if na == 1:
                        continue        # Lamenza não publica single-drone
                elif "=" in spec:
                    rot, caminho = spec.split("=", 1)
                    mod = "wnn"
                    ckpt = Path(caminho.format(na=na))
                    if not ckpt.is_absolute():
                        ckpt = RAIZ / ckpt
                    if not ckpt.is_dir():
                        bloqueio = f"sem checkpoint {_rel(ckpt)}"
                elif spec.startswith("wnn-T"):
                    rot, mod = spec, "wnn"
                    r = _checkpoint_wnn(Path(a.raiz_modelos) if Path(a.raiz_modelos).is_absolute()
                                        else RAIZ / a.raiz_modelos, int(spec[5:]), na, ns)
                    if isinstance(r, str):
                        bloqueio = r
                    else:
                        ckpt = r
                else:
                    sys.exit(f"modelo desconhecido: {spec!r}")
                gate = Path(a.gates) / f"gate_rf05-{rot}-a{na}s{ns}.json"
                if not gate.is_absolute():
                    gate = RAIZ / gate
                if regime == "com-kf" and not gate.exists() and bloqueio is None:
                    bloqueio = f"sem atestado {_rel(gate)}"
                treinado = (_treinado_em(ckpt) if ckpt is not None
                            else (str(ns) if mod == "maddpg" else None))
                nome = f"{rot}-a{na}s{ns}-{suf}"
                fila.append(Corrida(
                    rotulo=rot, modelo=mod, na=na, ns=ns, regime=regime,
                    checkpoint=_rel(ckpt) if ckpt else None,
                    treinado_em_ns=treinado,
                    gate=_rel(gate) if regime != "limpo" else None,
                    csv=str(Path(a.dir_saida) / f"{nome}.csv"),
                    log=str(Path(a.dir_saida) / "logs" / f"{nome}.log"),
                    manifesto=str(Path(a.dir_saida) / "logs" / f"{nome}.manifest.json"),
                    bloqueio=bloqueio))
    return fila


def runs_feitas(csv: Path) -> int:
    if not csv.exists():
        return 0
    with csv.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def conflito(c: Corrida) -> str | None:
    """O CSV já tem dono diferente? Lê o manifesto do lançamento anterior."""
    m = RAIZ / c.manifesto
    if not m.exists() or not (RAIZ / c.csv).exists():
        return None
    antes = json.loads(m.read_text()).get("checkpoint")
    if antes != c.checkpoint:
        return f"CSV já pertence a {antes or 'maddpg'} (manifesto); use --forcar"
    return None


def comando(c: Corrida, a) -> list[str]:
    cmd = ["nice", "-n", str(a.nice), "uv", "run"]
    if c.modelo == "maddpg":
        cmd += ["--group", "baseline"]
    cmd += ["python", "scripts/spec06_campanha_completa.py",
            "--modelo", c.modelo, "--rotulo", c.rotulo,
            "--na", str(c.na), "--ns", str(c.ns),
            "--ruido", REGIMES[c.regime][0],
            "--runs", str(a.runs), "--episodios", str(a.episodios),
            "--dir-saida", str(a.dir_saida)]
    if c.treinado_em_ns is not None:
        cmd += ["--treinado-em-ns", c.treinado_em_ns]
    if c.checkpoint is not None:
        cmd += ["--checkpoint", c.checkpoint]
    if REGIMES[c.regime][1] is not None:
        cmd += ["--gate", c.gate, "--sigmas", a.sigmas, "--degraus", REGIMES[c.regime][1]]
    return cmd


def rodar(c: Corrida, a) -> tuple[Corrida, int]:
    log = RAIZ / c.log
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = comando(c, a)
    (RAIZ / c.manifesto).write_text(json.dumps({
        **asdict(c), "transferencia": c.transferencia, "comando": cmd,
        "lancado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "procedencia": (json.loads((RAIZ / c.checkpoint / "procedencia.json").read_text())
                        if c.checkpoint and (RAIZ / c.checkpoint / "procedencia.json").exists()
                        else None)}, indent=2, ensure_ascii=False))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    with log.open("a") as fh:
        fh.write(f"\n=== fila: {time.strftime('%Y-%m-%d %H:%M:%S')} · "
                 f"checkpoint {c.checkpoint} · {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=RAIZ, env=env)
    return c, r.returncode


def listar(fila: list[Corrida], a) -> None:
    for c in fila:
        feitas = runs_feitas(RAIZ / c.csv)
        # O conflito vem ANTES de "pronta": um CSV completo de OUTRO checkpoint é
        # exatamente o caso em que a tabela final atribuiria números velhos ao
        # modelo novo sem erro nenhum.
        st = ("PULA (" + c.bloqueio + ")" if c.bloqueio else
              "CONFLITO (" + conflito(c) + ")" if conflito(c) and not a.forcar else
              "PRONTA" if feitas >= a.runs else
              "roda")
        extra = " ·transf" if c.modelo == "wnn" and c.transferencia else ""
        ck = f"  [{c.checkpoint}]" if c.checkpoint else ""
        print(f"{c.rotulo:<14} Na={c.na:<2} Ns={c.ns:<3} {c.regime:<7} "
              f"{feitas:>3}/{a.runs}  {st}{extra}{ck}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", default="wnn-T16,wnn-T32,maddpg",
                    help="lista separada por vírgula — ver o cabeçalho")
    ap.add_argument("--celulas", nargs="+", default=["1:12", "2:12,24,36"],
                    help="`Na:Ns,Ns,...` por item, ex.: 2:12,24,36 4:12,24,36")
    ap.add_argument("--regimes", default="limpo,sem-kf,com-kf")
    ap.add_argument("--sigmas", default="10")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--episodios", type=int, default=200)
    ap.add_argument("--raiz-modelos", default="models/deploy",
                    help="de onde `wnn-T16`/`wnn-T32` resolvem o checkpoint; "
                         "`models/deploy` porque o artefato reportado em tamanho "
                         "é o podado, e desempenho tem de sair do MESMO artefato")
    ap.add_argument("--gates", default="results/etapa2")
    ap.add_argument("--dir-saida", default="reports/campanha")
    ap.add_argument("--paralelo", type=int, default=6)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--forcar", action="store_true",
                    help="ignora a guarda do manifesto (CSV de outro checkpoint)")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--seco", action="store_true", help="só imprime os comandos")
    ap.add_argument("--parar", action="store_true",
                    help="encerra a fila E as corridas lançadas por ela")
    a = ap.parse_args()

    if a.parar:
        if PGID.exists():
            pg = int(PGID.read_text().strip())
            try:
                os.killpg(pg, signal.SIGTERM)
                print(f"grupo {pg} encerrado (fila + corridas)")
            except ProcessLookupError:
                print(f"grupo {pg} já não existe")
            PGID.unlink()
        else:
            print(f"nenhuma fila registrada em {PGID.name}")
        return 0

    fila = montar_fila(a)
    if a.listar:
        listar(fila, a)
        return 0
    pend = []
    for c in fila:
        if c.bloqueio:
            print(f"PULADA: {c.rotulo} a{c.na}s{c.ns} {c.regime} — {c.bloqueio}")
        elif conflito(c) and not a.forcar:
            print(f"RECUSADA: {c.rotulo} a{c.na}s{c.ns} {c.regime} — {conflito(c)}")
        elif runs_feitas(RAIZ / c.csv) >= a.runs:
            continue
        else:
            pend.append(c)
    if a.seco:
        for c in pend:
            print(" ".join(comando(c, a)))
        print(f"\n{len(pend)} corridas a lançar, {a.paralelo} em paralelo")
        return 0
    if not pend:
        print("nada a fazer — a fila está completa.")
        return 0

    # Grupo de processos próprio, para que `--parar` derrube fila e corridas de
    # uma vez: matar só o PID da fila deixaria as corridas em voo órfãs.
    os.setpgrp()
    PGID.write_text(str(os.getpgrp()))
    print(f"{len(pend)} corridas · {a.paralelo} em paralelo · pgid {os.getpgrp()}",
          flush=True)
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=a.paralelo) as pool:
            for c, rc in pool.map(lambda c: rodar(c, a), pend):
                dec = (time.perf_counter() - t0) / 60
                print(f"  [{dec:6.1f} min] {c.rotulo} a{c.na}s{c.ns} {c.regime}: "
                      f"{'ok' if rc == 0 else f'código {rc}'} · {c.log}", flush=True)
    finally:
        if PGID.exists():
            PGID.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
