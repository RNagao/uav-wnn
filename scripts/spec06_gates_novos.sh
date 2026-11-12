#!/bin/bash
# Atestados RF-05 (spec08_gate_rf05.py) dos modelos NOVOS, um por modelo e célula,
# cada modelo só nas células do SEU Na — o pré-requisito do regime `com-kf` da
# `spec06_campanha_novos.sh`. Mesmos argumentos da `spec06_gates_fila.sh` (a dos
# modelos publicados), só muda de onde vem o checkpoint e o rótulo.
#
# O atestado vale sob a mobilidade DAQUELA política NAQUELA célula (T-08.1), por
# isso é um por (modelo, Na, Ns): 2 modelos × 9 células = 18. ~100 episódios
# cada; o MADDPG já tem os seus nas 9 células.
#
#     bash scripts/spec06_gates_novos.sh --listar
#     nohup bash scripts/spec06_gates_novos.sh > reports/campanha/logs/_gates-novos.log 2>&1 &
#     NAS=2 PARALELO=2 bash scripts/spec06_gates_novos.sh
#
# ORIGEM/ROT como na campanha: o atestado é do MESMO checkpoint que a campanha vai
# usar — atestar o `melhor` e rodar o `deploy` seria medir o filtro sob outra
# mobilidade.
set -u
cd "$(dirname "$0")/.."
RAIZ=$(pwd)
ORIGEM=${ORIGEM:-deploy}
NAS=${NAS:-2,4,8}
NSS=${NSS:-12,24,36}
ROT=${ROT:-wnn-v2}
PARALELO=${PARALELO:-4}
EPS=${EPS:-100}
RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}
RUNS=${RUNS:-spec06-50M}      # pasta de runs (ex.: spec06-50M-ctde para o crítico centralizado)
G="$RAIZ_DADOS/results/etapa2"

ckpt() {   # $1=T $2=Na
  case "$ORIGEM" in
    models) echo "$RAIZ_DADOS/models/deploy/T$1/a$2s12-24-36" ;;
    *)      echo "$RAIZ_DADOS/runs/$RUNS/T$1-a$2s12-24-36/checkpoints/$ORIGEM" ;;
  esac
}
fila() {
  for NA in ${NAS//,/ }; do for NS in ${NSS//,/ }; do for T in 16 32; do
    echo "$ROT-T$T|$NA|$NS|$(ckpt $T $NA)"
  done; done; done
}
um() {
  IFS='|' read -r R NA NS CKPT <<< "$1"
  cd "$RAIZ"
  local OUT="$G/gate_rf05-${R}-a${NA}s${NS}.json"
  [ -f "$OUT" ] && { echo "já existe: $OUT"; return 0; }
  [ ! -d "$CKPT" ] && { echo "PULADO $R a${NA}s${NS}: sem $CKPT"; return 0; }
  mkdir -p "$RAIZ_DADOS/reports/campanha/logs"
  local LOG="$RAIZ_DADOS/reports/campanha/logs/gate-${R}-a${NA}s${NS}.log"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  uv run python scripts/spec08_gate_rf05.py --checkpoint "$CKPT" --aproximador ntupla_dissertacao \
    --na "$NA" --ns "$NS" --episodios "$EPS" --q-w-grade 0,0.25,1,4,8,16,32,64 \
    --saida "$OUT" > "$LOG" 2>&1
  grep -qE "=== APROVADO ===" "$LOG" \
    && echo "OK  $R a${NA}s${NS} · $(grep -oE 'melhor q_w=[0-9.]+' "$LOG" | tail -1)" \
    || echo "REPROVADO  $R a${NA}s${NS} — ver $LOG"
}
export -f um; export RAIZ EPS G RAIZ_DADOS
if [ "${1:-}" = "--listar" ]; then
  fila | while IFS='|' read -r R NA NS CKPT; do
    OUT="$G/gate_rf05-${R}-a${NA}s${NS}.json"
    printf '%-11s Na=%-2s Ns=%-3s  %s  [%s]\n' "$R" "$NA" "$NS" \
      "$([ -f "$OUT" ] && echo "existe" || echo "FALTA ")" "$CKPT"
  done; exit 0
fi
fila | xargs -d '\n' -P "$PARALELO" -I{} bash -c 'um "$@"' _ {}
