"""O laço de treino da campanha — §4.4 (parada) e §4.7 (artefatos).

Um treinador que serve às tarefas T-06.1 a T-06.5 sem ramificação por tarefa: o que
muda entre elas é **configuração**, não código. Ramificar por tarefa faria cada
réplica ter o seu laço, e a comparação entre elas mediria os laços.

§4.4 — O QUE "CONVERGIU" SIGNIFICA, OPERACIONALMENTE

    avaliação intercalada    a cada `B_max/40` passos, sobre o banco `val`, com o
                             treino congelado e SEM ruído de exploração
    convergiu                a média móvel da métrica de validação sobre as últimas
                             5 avaliações não melhora mais que 1 % relativo ao
                             longo de 8 avaliações consecutivas
    checkpoint reportado     o de MELHOR validação, não o último
    falha catastrófica       a métrica de validação não supera a caminhada
                             aleatória com IC bootstrap 95 % separado

A última é a definição operacional do fenômeno do §7.3 de Lamenza, e é o que separa
"o método não funciona" de "essa semente não funcionou". Por isso **nenhuma corrida
é descartada** (§4.3): uma corrida que falha catastroficamente é dado.

DUAS ARMADILHAS QUE ESTE MÓDULO EVITA POR CONSTRUÇÃO

1. **Avaliar no banco `test`.** O `test` entra aqui **só** no fim, uma vez, depois
   de o checkpoint estar escolhido. A seleção usa `val`. Lamenza avalia durante o
   treino e reporta o modelo carregado dessas avaliações (§7, p36) — o que é
   seleção sobre o conjunto de reporte. Nós separamos, e o `avaliar_test` só é
   chamado pelo orquestrador, nunca pelo laço.

2. **Avaliar com ruído.** A política é avaliada com a **média** da gaussiana — o
   análogo do *"action noise is removed"* de Lamenza. Um treino que avalia com ruído
   subestima a própria política, e o efeito cresce com `σ`.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .bancos import BancoPar
from .sementes import PapeisDeSemente

FRACAO_AVALIACOES = 40
"""§4.4: avaliação intercalada a cada `B_max/40` passos."""
JANELA_MEDIA = 5
"""Média móvel sobre as últimas 5 avaliações."""
PACIENCIA = 8
"""Avaliações consecutivas sem melhora de 1 % para declarar convergência."""
MELHORA_MINIMA = 0.01
MINIMO_AVALIACOES = 20
"""Metade das avaliações antes de a parada antecipada poder disparar.

CORREÇÃO AO CRITÉRIO DA §4.4, e ela veio de uma medição.

O critério como escrito — *"a média móvel não melhora mais que 1 % ao longo de 8
avaliações consecutivas"* — dispara também quando a métrica está **achatada no zero
no início do treino**. Medido: numa corrida de fumaça, `SR` ficou em 0,0 % e a parada
declarou "convergiu em 1950 passos" de um orçamento de 6000. Isso confunde *"platô
depois de aprender"* com *"ainda não começou"*, e libera orçamento de uma corrida que
não aprendeu nada — o oposto do que a parada existe para fazer.

A guarda é a leitura mínima defensável do critério: ele detecta **platô após
aprendizado**, então exige que haja treino suficiente para que "após" signifique
algo. Metade das avaliações é a metade do orçamento, o que mantém o ganho de liberar
a segunda metade quando a corrida de fato estabilizou."""


@dataclass
class Avaliacao:
    """Uma linha do `metrics.parquet` da §4.7."""

    passo: int
    recompensa_media: float
    sr: float
    ct_cens: float
    ct_succ: float
    cobertura: float = 0.0
    """Fração média de sensores coletados. É a métrica de SELEÇÃO de 1a, porque `SR`
    e `CT_cens` saturam (a caminhada aleatória dá SR = 0 %) e selecionar por métrica
    achatada é selecionar ao acaso."""
    ocupadas: int = 0
    """RNF-08: entradas ocupadas nas LUTs. Cresce com o treino, e a curva é o que
    verifica saturação na T-06.6."""
    parede_s: float = 0.0
    passos_por_s: float = 0.0

    def como_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("passo", "recompensa_media", "sr", "ct_cens", "ct_succ",
                 "cobertura", "ocupadas", "parede_s", "passos_por_s")}


@dataclass
class ParadaAntecipada:
    """§4.4 — a média móvel de 5, com paciência de 8 e melhora mínima de 1 %.

    `maior_e_melhor` porque a métrica de validação de 1a é `SR` (maior é melhor) e
    de 1b é `CT_cens` (menor é melhor). Deixar isso implícito faria a parada
    antecipada disparar imediatamente numa das duas.
    """

    maior_e_melhor: bool = True
    historico: list[float] = field(default_factory=list)
    _sem_melhora: int = 0
    _melhor_media: float | None = None

    def registrar(self, valor: float) -> bool:
        """Devolve `True` quando convergiu."""
        self.historico.append(float(valor))
        if len(self.historico) < JANELA_MEDIA:
            return False
        if len(self.historico) < MINIMO_AVALIACOES:
            # Acumula a estatística mas NÃO pode parar ainda — ver
            # `MINIMO_AVALIACOES`. Atualizar o melhor aqui é o que impede que a
            # guarda apenas atrase a mesma decisão errada.
            media_p = float(np.mean(self.historico[-JANELA_MEDIA:]))
            if self._melhor_media is None:
                self._melhor_media = media_p
            elif ((media_p > self._melhor_media) if self.maior_e_melhor
                  else (media_p < self._melhor_media)):
                self._melhor_media = media_p
            return False
        media = float(np.mean(self.historico[-JANELA_MEDIA:]))
        if self._melhor_media is None:
            self._melhor_media = media
            return False
        ref = abs(self._melhor_media) or 1.0
        ganho = ((media - self._melhor_media) if self.maior_e_melhor
                 else (self._melhor_media - media)) / ref
        if ganho > MELHORA_MINIMA:
            self._melhor_media = media
            self._sem_melhora = 0
        else:
            self._sem_melhora += 1
        return self._sem_melhora >= PACIENCIA


def ambiente_de_execucao(bancos: BancoPar) -> dict:
    """O `env.json` da §4.7: commit, versões, CPU, hash dos bancos."""
    def _git(*args) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # pragma: no cover
            return "?"

    import numpy
    return {
        "commit": _git("rev-parse", "HEAD"),
        "sujo": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "plataforma": platform.platform(),
        "processador": platform.processor() or platform.machine(),
        "bancos": bancos.como_dict(),
    }


@dataclass
class Corrida:
    """Uma corrida: uma configuração, uma `seed_train`, um diretório.

    O diretório é `runs/<experimento>/<config_hash>/<seed_train>/`, como a §4.7
    manda. O `config_hash` sai da configuração **resolvida**, não da linha de
    comando — duas invocações diferentes que resolvem para a mesma configuração têm
    de cair no mesmo diretório, senão o cache não funciona e o registro duplica.
    """

    experimento: str
    config: dict
    sementes: PapeisDeSemente
    bancos: BancoPar
    raiz: Path = Path("runs")

    def __post_init__(self) -> None:
        import hashlib
        crua = json.dumps(self.config, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(crua.encode()).hexdigest()[:12]
        self.dir = (self.raiz / self.experimento / self.config_hash
                    / str(self.sementes.seed_train))
        self.avaliacoes: list[Avaliacao] = []

    def preparar(self) -> None:
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(
            json.dumps({"config": self.config, "config_hash": self.config_hash,
                        "sementes": self.sementes.como_dict(),
                        "experimento": self.experimento},
                       indent=2, default=str))
        (self.dir / "env.json").write_text(
            json.dumps(ambiente_de_execucao(self.bancos), indent=2))

    def registrar(self, aval: Avaliacao) -> None:
        self.avaliacoes.append(aval)
        # JSONL e não parquet: `pyarrow` não é dependência do projeto, e o que a
        # §4.7 exige é o registro por avaliação, não o formato. A agregação lê os
        # dois igual.
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(aval.como_dict()) + "\n")

    def gravar_eval_test(self, por_episodio: list[dict]) -> None:
        """O insumo do bootstrap pareado — **por episódio**, não agregado.

        Sem isto o IC da diferença não é reconstruível, e a §4.7 o exige por isso.
        Agregar aqui economizaria disco e destruiria a única coisa que permite
        recalcular o IC sem re-treinar.
        """
        with open(self.dir / "eval_test.jsonl", "w") as f:
            for linha in por_episodio:
                f.write(json.dumps(linha) + "\n")


def treinar(
    corrida: Corrida,
    passo_de_treino: Callable[[int], dict],
    avaliar_val: Callable[[], dict],
    b_max: int,
    salvar_checkpoint: Callable[[Path], None],
    metrica: str = "sr",
    maior_e_melhor: bool = True,
    ocupadas: Callable[[], int] = lambda: 0,
    verboso: bool = True,
) -> dict:
    """O laço da §4.4. Devolve o resumo da corrida.

    Recebe as três operações por injeção — passo de treino, avaliação, checkpoint —
    porque o laço é o **mesmo** para WNN e MLP, e é isso que torna a comparação de
    H1 sobre o aproximador. Um laço por família mediria os laços.
    """
    corrida.preparar()
    intervalo = max(1, b_max // FRACAO_AVALIACOES)
    parada = ParadaAntecipada(maior_e_melhor=maior_e_melhor)
    melhor = None
    t0 = time.perf_counter()
    passos = 0

    while passos < b_max:
        info = passo_de_treino(min(intervalo, b_max - passos))
        passos += int(info.get("passos", intervalo))

        m = avaliar_val()
        parede = time.perf_counter() - t0
        aval = Avaliacao(
            passo=passos, recompensa_media=float(m.get("recompensa_media", 0.0)),
            sr=float(m["sr"]), ct_cens=float(m["ct_cens"]),
            ct_succ=float(m["ct_succ"]), cobertura=float(m.get("cobertura", 0.0)),
            ocupadas=int(ocupadas()),
            parede_s=parede, passos_por_s=passos / parede if parede else 0.0)
        corrida.registrar(aval)

        valor = float(m[metrica])
        melhora = (melhor is None
                   or (valor > melhor if maior_e_melhor else valor < melhor))
        if melhora:
            melhor = valor
            # Checkpoint de MELHOR validação, não o último (§4.4). A regra é fixada
            # antes da campanha, e gravar aqui é o que a torna efetiva.
            salvar_checkpoint(corrida.dir / "checkpoints" / "melhor")
        if verboso:
            print(f"    {passos:>9} passos  {metrica}={valor:<8.4f} "
                  f"SR={m['sr']:.1%} cob={m.get('cobertura', 0):.1%} "
                  f"CT={m['ct_cens']:.1f} "
                  f"ocup={aval.ocupadas:<7} {aval.passos_por_s:.0f} p/s"
                  f"{'  *' if melhora else ''}")

        if parada.registrar(valor):
            if verboso:
                print(f"    convergiu em {passos} passos "
                      f"({len(parada.historico)} avaliações)")
            break

    salvar_checkpoint(corrida.dir / "checkpoints" / "final")
    return {"passos": passos, "avaliacoes": len(corrida.avaliacoes),
            "melhor_val": melhor, "convergiu": passos < b_max,
            "parede_s": time.perf_counter() - t0,
            "historico_val": parada.historico}
