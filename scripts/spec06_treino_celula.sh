#!/bin/bash
# Treino LONGO (50 M) de um modelo por célula: Na fixo, Ns revezando 12/24/36,
# config vinda da grade daquele Na (`--config-de`) para o T pedido.
#
# Mesmo motivo do `spec06_grade_celula.sh`: a linha equivalente é longa, já
# chegou truncada ao processo, e os caminhos têm de apontar para o CHECKOUT
# PRINCIPAL mesmo quando lançado de um worktree.
#
#     bash scripts/spec06_treino_celula.sh 2 16                 # Na=2, T=16, vencedor da grade
#     bash scripts/spec06_treino_celula.sh 2 32 --passo 8e-4 --sigma 0.35 --tupla 12 --phi 0.99 --m 8
#                                                               # flags explícitas têm precedência
#     bash scripts/spec06_treino_celula.sh 2 32 --retomar runs/spec06-50M/T32-a2s12-24-36/checkpoints/ultimo
#     tail -f reports/spec06-50M-T16-a2s12-24-36.log            # acompanhar
#     RUNS=spec06-50M-ctde bash scripts/spec06_treino_celula.sh 2 16 \
#         --critico centralizado --contrafactual 4             # outra pasta de runs
#
# Saída: runs/$RUNS/T<T>-a<Na>s12-24-36 (RUNS=spec06-50M por padrão) — o rótulo
# que `spec06_exportar_deploy.py --promover T<T>-a<Na>s12-24-36` espera. Log em
# reports/$RUNS-T<T>-a<Na>s12-24-36.log.
set -eu
cd "$(dirname "$0")/.."
NA=${1:?uso: $0 NA T [flags extras do spec06_treino_multiagente.py]}
T=${2:?uso: $0 NA T [flags extras]}; shift 2 || true
NS=${NS:-12,24,36}
B=${B:-50000000}
RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}
RUNS=${RUNS:-spec06-50M}
CEL="T${T}-a${NA}s${NS//,/-}"
GRADE="$RAIZ_DADOS/reports/spec06-busca-a${NA}s${NS//,/-}.json"
SAIDA="$RAIZ_DADOS/runs/$RUNS/$CEL"
LOG="$RAIZ_DADOS/reports/$RUNS-$CEL.log"

[ -f "$GRADE" ] || { echo "sem grade: $GRADE"; exit 1; }
if pgrep -f "spec06_treino_multiagente.py --na $NA --ns $NS --resolucao $T " >/dev/null; then
  echo "já há um treino de $CEL rodando:"; pgrep -af "spec06_treino_multiagente.py --na $NA --ns $NS --resolucao $T " | head -1; exit 1
fi
mkdir -p "$RAIZ_DADOS/runs/$RUNS" "$RAIZ_DADOS/reports"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup uv run python scripts/spec06_treino_multiagente.py \
  --na "$NA" --ns "$NS" --resolucao "$T" \
  --config-de "$GRADE" --b-max "$B" --val-n 100 \
  --saida "$SAIDA" "$@" \
  >> "$LOG" 2>&1 &
echo "lançado: treino $CEL (pid $!) · log: $LOG"
sleep 20
if ! kill -0 "$!" 2>/dev/null; then echo "MORREU na inicialização — veja $LOG"; tail -5 "$LOG"; exit 1; fi
grep -v "^nohup" "$LOG" | head -4
