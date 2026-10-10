#!/bin/bash
# Grade `desempate` de UM Na, com Ns revezando 12/24/36 — saída isolada por Na.
#
# Existe porque a linha de comando equivalente tem ~400 caracteres e já chegou
# truncada ao processo duas vezes (uma sem a variável de ambiente, outra sem
# `--raiz`/`--saida`/redirecionamento), e cada vez a grade gravou em cima do
# lugar errado. Um script não se parte ao colar.
#
# Tudo vai para o CHECKOUT PRINCIPAL (runs/ e reports/ de $RAIZ_DADOS), mesmo
# quando rodado de um worktree: os worktrees são descartáveis, os dados não.
#
#     bash scripts/spec06_grade_celula.sh 4            # Na=4, Ns=12,24,36
#     NS=12,24 bash scripts/spec06_grade_celula.sh 2   # outra lista de Ns
#     bash scripts/spec06_grade_celula.sh 4 --retomar  # continua a mesma grade
#     tail -f reports/spec06-busca-a4s12-24-36.log     # acompanhar
#     pkill -f "spec06_busca.py --modo grade --niveis desempate --na 4"
set -eu
cd "$(dirname "$0")/.."
NA=${1:?uso: $0 NA [--retomar]}; shift || true
NS=${NS:-12,24,36}
NIVEL=${NIVEL:-desempate}
B=${B:-200000}
PROCESSOS=${PROCESSOS:-12}
RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}
ROT="a${NA}s${NS//,/-}"
LOG="$RAIZ_DADOS/reports/spec06-busca-$ROT.log"

if pgrep -f "spec06_busca.py --modo grade --niveis $NIVEL --na $NA " >/dev/null; then
  echo "já há uma grade de Na=$NA rodando:"; pgrep -af "spec06_busca.py.*--na $NA " | head -1; exit 1
fi
mkdir -p "$RAIZ_DADOS/reports" "$RAIZ_DADOS/runs"
nohup uv run python scripts/spec06_busca.py \
  --modo grade --niveis "$NIVEL" --na "$NA" --ns "$NS" \
  --b-sweep "$B" --sementes 3 --processos "$PROCESSOS" \
  --raiz  "$RAIZ_DADOS/runs/spec06-$NIVEL-a$NA" \
  --saida "$RAIZ_DADOS/reports/spec06-busca-$ROT.json" "$@" \
  >> "$LOG" 2>&1 &
echo "lançada: grade $NIVEL Na=$NA Ns=$NS (pid $!) · log: $LOG"
sleep 15
if ! kill -0 "$!" 2>/dev/null; then echo "MORREU na inicialização — veja $LOG"; tail -5 "$LOG"; exit 1; fi
head -6 "$LOG" | grep -v "^nohup"
