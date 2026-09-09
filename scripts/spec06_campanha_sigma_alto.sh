#!/bin/bash
# Extensão EXPLORATÓRIA da campanha comparável para ruído ALTO — σ ∈ {25, 50}.
#
# POR QUE UM SCRIPT À PARTE, E NÃO SIGMAS=25 NA FILA. A grade de σ é pré-registrada
# e CONGELADA em {1,3,5,10} (`configs/etapa2/experimento.json`, hasheado em
# `results/etapa2/analysis_plan.sha256`). σ=25 e 50 estão FORA dela. Tratar como
# extensão exploratória — script separado, resultado rotulado como pós-hoc — não
# toca no arquivo congelado, não invalida o hash nem o atestado do gate. Se um dia
# virar confirmatório, aí sim se emenda o pré-registro.
#
# O QUE MUDA (E O QUE NÃO). Nada no filtro precisa de resintonia: o KF é criado
# com `sigma_gps = σ` e `R = σ²·I`, então o R acompanha o ruído sozinho; o `q_w`
# do degrau 4 modela descasamento de ATUAÇÃO (não o GPS), é invariante ao σ, e já
# está atestado na âncora σ=5 — por isso os gates de σ=10 são REUSADOS aqui. O
# NEES é adimensional para filtro casado, logo a consistência atestada carrega.
#
# ESCOPO. Só Na=2 (Ns ∈ {12,24,36}), modelos MADDPG + WNN T16 + WNN T32, regimes
# `sem-kf` (degrau 1, medida crua) e `com-kf` (degrau 4, KF+controle). O `limpo`
# independe de σ e já foi medido — não se repete. Cada regime roda com um único
# (degrau, σ), então todo episódio cai na MESMA condição e o pareamento
# MADDPG×WNN por semente é exato (`condicao_do_episodio` depende do tamanho da
# escada, que aqui é 1).
#
# SAÍDA. `reports/campanha/{rotulo}-a2s{ns}-sigma{σ}-{sem-kf|com-kf}.csv`, no mesmo
# formato dos de σ=10 — some direto no `scripts/campanha_resumo.py`.
#
#     bash scripts/spec06_campanha_sigma_alto.sh              # roda σ=25 e 50
#     SIGMAS_LISTA=25 bash scripts/spec06_campanha_sigma_alto.sh   # só σ=25
#     RUNS=2 EPISODIOS=10 bash scripts/spec06_campanha_sigma_alto.sh --listar  # smoke
#     bash scripts/spec06_campanha_sigma_alto.sh --parar     # para tudo (fila+corridas)
#     bash scripts/spec06_campanha_sigma_alto.sh --listar    # mostra a fila e o estado
#
# NICE. Corridas com prioridade rebaixada (`nice -n 10`) para não roubar núcleo de
# um treino single-thread que esteja rodando. PARALELO=6 deixa folga em 12 núcleos.
set -u
cd "$(dirname "$0")/.."
RAIZ=$(pwd)
PARALELO=${PARALELO:-6}
NS_TREINO=24                       # as redes WNN multiagente foram treinadas em Ns=24
G=results/etapa2
RUNS=${RUNS:-100}
EPISODIOS=${EPISODIOS:-200}
PGID_FILE="$RAIZ/.campanha-sigma-alto.pgid"

ckpt_wnn() { echo "models/deploy/T$1/a2s${NS_TREINO}"; }   # $1=T
gate_de()  { echo "$G/gate_rf05-$1-a2s$2.json"; }          # $1=rótulo  $2=Ns

# Só Na=2; só sem-kf e com-kf (limpo independe de σ). Campos:
# MOD|ROT|NS|REG|CKPT|GATE|TRNS
fila_na2() {
  for REG in sem-kf com-kf; do
    for NS in 12 24 36; do
      echo "maddpg|maddpg|$NS|$REG|-|$(gate_de maddpg $NS)|$NS"
      for T in 16 32; do
        echo "wnn|wnn-T$T|$NS|$REG|$(ckpt_wnn $T)|$(gate_de wnn-T$T $NS)|$NS_TREINO"
      done
    done
  done
}

uma() {
  IFS='|' read -r MOD ROT NS REG CKPT GATE TRNS <<< "$1"
  cd "$RAIZ"
  if [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ]; then
    echo "PULADA: $ROT a2s$NS $REG σ$SIGMAS — sem checkpoint $CKPT"; return 0
  fi
  if [ "$REG" = "com-kf" ] && [ ! -f "$GATE" ]; then
    echo "PULADA: $ROT a2s$NS $REG σ$SIGMAS — sem atestado $GATE"; return 0
  fi
  mkdir -p reports/campanha/logs
  local DEG
  case "$REG" in sem-kf) DEG=1 ;; com-kf) DEG=4 ;; esac
  local SUF="sigma${SIGMAS}-${REG}"
  local LOG="reports/campanha/logs/${ROT}-a2s${NS}-${SUF}.log"
  local ARGS=(--modelo "$MOD" --rotulo "$ROT" --na 2 --ns "$NS" --ruido sorteio
              --runs "$RUNS" --episodios "$EPISODIOS" --treinado-em-ns "$TRNS"
              --dir-saida reports/campanha --gate "$GATE" --sigmas "$SIGMAS" --degraus "$DEG")
  [ "$CKPT" != "-" ] && ARGS+=(--checkpoint "$CKPT")
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  local NICE=${NICE:-10}
  if [ "$MOD" = "maddpg" ]; then
    nice -n "$NICE" uv run --group baseline python \
      scripts/spec06_campanha_completa.py "${ARGS[@]}" > "$LOG" 2>&1 \
      && echo "OK: $ROT a2s$NS $SUF" || echo "FALHOU: $ROT a2s$NS $SUF (ver $LOG)"
  else
    nice -n "$NICE" uv run python \
      scripts/spec06_campanha_completa.py "${ARGS[@]}" > "$LOG" 2>&1 \
      && echo "OK: $ROT a2s$NS $SUF" || echo "FALHOU: $ROT a2s$NS $SUF (ver $LOG)"
  fi
}
export -f uma
export RAIZ RUNS EPISODIOS NICE

# ---- parar: derruba as corridas desta extensão (marcadas por --rotulo/--dir) ----
if [ "${1:-}" = "--parar" ]; then
  pkill -f "spec06_campanha_completa.py" && echo "corridas encerradas" \
    || echo "nenhuma corrida em andamento"
  exit 0
fi

# ---- listar: fila + progresso, sem rodar ----
if [ "${1:-}" = "--listar" ]; then
  for SIGMAS in ${SIGMAS_LISTA:-25 50}; do
    fila_na2 | while IFS='|' read -r MOD ROT NS REG CKPT GATE TRNS; do
      SUF="sigma${SIGMAS}-${REG}"
      CSV="reports/campanha/${ROT}-a2s${NS}-${SUF}.csv"
      F=0; [ -f "$CSV" ] && F=$(( $(wc -l < "$CSV") - 1 ))
      ST=roda
      [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ] && ST="PULA (sem checkpoint)"
      [ "$REG" = "com-kf" ] && [ ! -f "$GATE" ] && ST="PULA (sem atestado)"
      [ "$F" -ge "$RUNS" ] && ST=PRONTA
      printf '%-9s Na=2 Ns=%-3s %-16s %3s/%s  %s\n' "$ROT" "$NS" "$SUF" "$F" "$RUNS" "$ST"
    done
  done
  exit 0
fi

# ---- rodar: um pool xargs por σ, em FOREGROUND (bloqueia), σ em sequência ----
# Sem setsid: o script fica vivo até o fim, então roda bem em background como UMA
# tarefa. Para parar, use `--parar` (pkill nas corridas) ou mate a tarefa.
for SIGMAS in ${SIGMAS_LISTA:-25 50}; do
  export SIGMAS
  echo "=== σ=$SIGMAS · $(fila_na2 | wc -l) células · PARALELO=$PARALELO · runs=$RUNS ep=$EPISODIOS ==="
  fila_na2 | xargs -d '\n' -P "$PARALELO" -I{} bash -c 'uma "$@"' _ {}
done
echo "=== fim: σ ∈ {${SIGMAS_LISTA:-25 50}} ==="
