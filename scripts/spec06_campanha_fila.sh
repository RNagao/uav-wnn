#!/bin/bash
# Fila da campanha comparável — 6 corridas em paralelo, retomável.
#
# Cada corrida grava o CSV run a run e relê o cache ao reiniciar: matar e
# relançar não perde nada além da run em voo. É por isso que a fila é um
# arquivo e não um laço no Python — interromper tem de ser barato.
#
#     bash scripts/spec06_campanha_fila.sh            # roda a fila
#     bash scripts/spec06_campanha_fila.sh --listar   # só mostra o que falta
#
# O regime de ruído é σ = 10 (o maior da escada), degraus 1–4 sorteados por
# episódio. O regime `limpo` mede o degrau 0 com os 20 mil episódios inteiros.
set -u
cd "$(dirname "$0")/.."
RAIZ=$(pwd)
PARALELO=${PARALELO:-6}
SIGMAS=${SIGMAS:-10}
G=results/etapa2
T16=models/T16
T32=models/T32

# modelo|rótulo|Na|Ns|regime|checkpoint|gate
FILA=$(cat <<EOF
wnn|wnn-T16|1|12|limpo|$T16/a1s12|$G/gate_rf05-T16final.json
wnn|wnn-T16|1|12|sorteio|$T16/a1s12|$G/gate_rf05-T16final.json
wnn|wnn-T32|1|12|limpo|runs/spec06-longa/spec06-busca-1b/15b49e642647/0/checkpoints/final|$G/gate_rf05-T32final.json
wnn|wnn-T32|1|12|sorteio|runs/spec06-longa/spec06-busca-1b/15b49e642647/0/checkpoints/final|$G/gate_rf05-T32final.json
wnn|wnn-T16|2|24|limpo|$T16/a2s24|$G/gate_rf05-wnn-T16-a2s24.json
wnn|wnn-T16|2|24|sorteio|$T16/a2s24|$G/gate_rf05-wnn-T16-a2s24.json
wnn|wnn-T16|4|24|limpo|$T16/a4s24|$G/gate_rf05-wnn-T16-a4s24.json
wnn|wnn-T16|4|24|sorteio|$T16/a4s24|$G/gate_rf05-wnn-T16-a4s24.json
wnn|wnn-T16|8|24|limpo|$T16/a8s24|$G/gate_rf05-wnn-T16-a8s24.json
wnn|wnn-T16|8|24|sorteio|$T16/a8s24|$G/gate_rf05-wnn-T16-a8s24.json
wnn|wnn-T32|2|24|limpo|$T32/a2s24|$G/gate_rf05-wnn-T32-a2s24.json
wnn|wnn-T32|2|24|sorteio|$T32/a2s24|$G/gate_rf05-wnn-T32-a2s24.json
wnn|wnn-T32|4|24|limpo|$T32/a4s24|$G/gate_rf05-wnn-T32-a4s24.json
wnn|wnn-T32|4|24|sorteio|$T32/a4s24|$G/gate_rf05-wnn-T32-a4s24.json
maddpg|maddpg|2|24|limpo|-|$G/gate_rf05-maddpg-a2s24.json
maddpg|maddpg|2|24|sorteio|-|$G/gate_rf05-maddpg-a2s24.json
maddpg|maddpg|4|24|limpo|-|$G/gate_rf05-maddpg-a4s24.json
maddpg|maddpg|4|24|sorteio|-|$G/gate_rf05-maddpg-a4s24.json
maddpg|maddpg|8|24|limpo|-|$G/gate_rf05-maddpg-a8s24.json
maddpg|maddpg|8|24|sorteio|-|$G/gate_rf05-maddpg-a8s24.json
EOF
)

uma() {
  IFS='|' read -r MOD ROT NA NS REG CKPT GATE <<< "$1"
  cd "$RAIZ"
  # Sem o checkpoint (T=32 multiagente ainda treinando) a corrida é pulada, e
  # dizer isso é melhor que morrer no meio da fila.
  if [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ]; then
    echo "PULADA: $ROT a${NA}s${NS} $REG — sem $CKPT"; return 0
  fi
  if [ "$REG" = "sorteio" ] && [ ! -f "$GATE" ]; then
    echo "PULADA: $ROT a${NA}s${NS} $REG — sem atestado $GATE"; return 0
  fi
  mkdir -p reports/campanha/logs
  local SUF; SUF=$([ "$REG" = "limpo" ] && echo limpo || echo "sigma${SIGMAS//,/-}")
  local LOG="reports/campanha/logs/${ROT}-a${NA}s${NS}-${SUF}.log"
  local ARGS=(--modelo "$MOD" --rotulo "$ROT" --na "$NA" --ns "$NS" --ruido "$REG"
              --runs 100 --episodios 200 --dir-saida reports/campanha)
  [ "$CKPT" != "-" ] && ARGS+=(--checkpoint "$CKPT")
  [ "$REG" = "sorteio" ] && ARGS+=(--gate "$GATE" --sigmas "$SIGMAS" --degraus 1,2,3,4)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  if [ "$MOD" = "maddpg" ]; then
    uv run --group baseline python scripts/spec06_campanha_completa.py "${ARGS[@]}" > "$LOG" 2>&1
  else
    uv run python scripts/spec06_campanha_completa.py "${ARGS[@]}" > "$LOG" 2>&1
  fi
}
export -f uma
export RAIZ SIGMAS

if [ "${1:-}" = "--listar" ]; then
  echo "$FILA" | nl -w3 -s'  '; exit 0
fi
echo "$FILA" | xargs -d '\n' -P "$PARALELO" -I{} bash -c 'uma "$@"' _ {}
