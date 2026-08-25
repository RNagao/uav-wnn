#!/bin/bash
# Atestados RF-05 que a campanha exige — um por (modelo, Na, Ns).
#
# O atestado NÃO é formalidade nem constante do projeto: a consistência do filtro
# vale sob a mobilidade da política avaliada NAQUELE cenário (T-08.1), e os
# valores medidos variam por ordem de grandeza — q_w foi de 1,0 (MADDPG a2s24) a
# 16,0 (WNN T16 a8s24). Reusar o q_w de outra célula mede o filtro errado.
#
# A grade de q_w é estendida (…,16,64) porque a padrão {0;0,25;1;4} deixa
# políticas agressivas otimistas: o T=16 semente 1 reprovou nela.
#
#     bash scripts/spec06_gates_fila.sh            # gera os que faltam
#     bash scripts/spec06_gates_fila.sh --listar
#     PARALELO=3 bash scripts/spec06_gates_fila.sh
set -u
cd "$(dirname "$0")/.."
RAIZ=$(pwd); PARALELO=${PARALELO:-4}; EPS=${EPS:-100}
G=results/etapa2; NS_TREINO=24

fila() {
  for T in 16 32; do echo "wnn-T$T|1|12|models/T$T/a1s12"; done
  for NA in 2 4 8; do for NS in 12 24 36; do
    echo "maddpg|$NA|$NS|-"
    for T in 16 32; do echo "wnn-T$T|$NA|$NS|models/T$T/a${NA}s${NS_TREINO}"; done
  done; done
}

um() {
  IFS='|' read -r ROT NA NS CKPT <<< "$1"
  cd "$RAIZ"
  local OUT="results/etapa2/gate_rf05-${ROT}-a${NA}s${NS}.json"
  [ -f "$OUT" ] && { echo "já existe: $OUT"; return 0; }
  [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ] && { echo "PULADO $ROT a${NA}s${NS}: sem $CKPT"; return 0; }
  mkdir -p reports/campanha/logs
  local LOG="reports/campanha/logs/gate-${ROT}-a${NA}s${NS}.log"
  local ARGS=(--na "$NA" --ns "$NS" --episodios "$EPS" --q-w-grade 0,0.25,1,4,8,16,32,64
              --saida "$OUT")
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  if [ "$ROT" = maddpg ]; then
    uv run --group baseline python scripts/spec08_gate_rf05.py --maddpg "${ARGS[@]}" > "$LOG" 2>&1
  else
    uv run python scripts/spec08_gate_rf05.py --checkpoint "$CKPT" \
      --aproximador ntupla_dissertacao "${ARGS[@]}" > "$LOG" 2>&1
  fi
  grep -qE "=== APROVADO ===" "$LOG" \
    && echo "OK  $ROT a${NA}s${NS} · $(grep -oE 'melhor q_w=[0-9.]+' "$LOG" | tail -1)" \
    || echo "REPROVADO  $ROT a${NA}s${NS} — ver $LOG"
}
export -f um; export RAIZ EPS

if [ "${1:-}" = "--listar" ]; then
  fila | while IFS='|' read -r ROT NA NS CKPT; do
    OUT="$G/gate_rf05-${ROT}-a${NA}s${NS}.json"
    printf '%-9s Na=%-2s Ns=%-3s  %s\n' "$ROT" "$NA" "$NS" \
      "$([ -f "$OUT" ] && echo "existe" || echo "FALTA")"
  done; exit 0
fi
fila | xargs -d '\n' -P "$PARALELO" -I{} bash -c 'um "$@"' _ {}
