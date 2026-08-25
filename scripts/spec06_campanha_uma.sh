#!/bin/bash
# Lança UMA corrida da campanha, em segundo plano e com prioridade rebaixada.
#
# Existe para o escalonamento manual: a fila inteira derruba o treino do
# Algoritmo 4.4 pela METADE (medido: 105 → 49 passos/s com 8 corridas), e a
# causa é contenção de CACHE, não de CPU — `nice` não ajuda contra isso. Então
# a forma segura de aproveitar a máquina é subir de uma em uma, medindo o treino
# entre cada subida.
#
#     bash scripts/spec06_campanha_uma.sh wnn-T16 2 24 limpo
#     bash scripts/spec06_campanha_uma.sh maddpg 4 36 com-kf
set -eu
cd "$(dirname "$0")/.."
ROT=$1; NA=$2; NS=$3; REG=$4
SIGMAS=${SIGMAS:-10}; NICE=${NICE:-10}; NS_TREINO=24
G=results/etapa2

case "$ROT" in
  maddpg)  MOD=maddpg; CKPT=- ;;
  wnn-T16) MOD=wnn; CKPT=$([ "$NA" = 1 ] && echo models/T16/a1s12 || echo models/T16/a${NA}s${NS_TREINO}) ;;
  wnn-T32) MOD=wnn; CKPT=$([ "$NA" = 1 ] && echo models/T32/a1s12 || echo models/T32/a${NA}s${NS_TREINO}) ;;
  *) echo "rótulo desconhecido: $ROT"; exit 1 ;;
esac
case "$REG" in
  limpo)  SUF=limpo;                        RUIDO=limpo;   DEG="" ;;
  sem-kf) SUF="sigma${SIGMAS//,/-}-sem-kf"; RUIDO=sorteio; DEG=1  ;;
  com-kf) SUF="sigma${SIGMAS//,/-}-com-kf"; RUIDO=sorteio; DEG=4  ;;
  *) echo "regime desconhecido: $REG (limpo|sem-kf|com-kf)"; exit 1 ;;
esac
TRNS=$([ "$MOD" = maddpg ] && echo "$NS" || { [ "$NA" = 1 ] && echo 12 || echo $NS_TREINO; })
GATE="$G/gate_rf05-${ROT}-a${NA}s${NS}.json"
[ "$CKPT" != "-" ] && [ ! -d "$CKPT" ] && { echo "sem checkpoint: $CKPT"; exit 1; }
[ -n "$DEG" ] && [ ! -f "$GATE" ] && { echo "sem atestado: $GATE"; exit 1; }

mkdir -p reports/campanha/logs
LOG="reports/campanha/logs/${ROT}-a${NA}s${NS}-${SUF}.log"
ARGS=(--modelo "$MOD" --rotulo "$ROT" --na "$NA" --ns "$NS" --ruido "$RUIDO"
      --runs 100 --episodios 200 --treinado-em-ns "$TRNS" --dir-saida reports/campanha)
[ "$CKPT" != "-" ] && ARGS+=(--checkpoint "$CKPT")
[ -n "$DEG" ] && ARGS+=(--gate "$GATE" --sigmas "$SIGMAS" --degraus "$DEG")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EXTRA=$([ "$MOD" = maddpg ] && echo "--group baseline" || echo "")
# shellcheck disable=SC2086
nohup nice -n "$NICE" uv run $EXTRA python scripts/spec06_campanha_completa.py \
  "${ARGS[@]}" >> "$LOG" 2>&1 &
echo "lançada: $ROT a${NA}s${NS} $SUF (pid $!) → $LOG"
