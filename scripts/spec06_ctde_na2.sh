#!/bin/bash
# Na=2 com CRÍTICO CENTRALIZADO (CTDE), de ponta a ponta e sem intervenção:
#   1. treina T16 e T32 em paralelo, MESMAS configs dos modelos de Na=2 de
#      2026-10-15/16 (c_max desligado), só trocando o crítico
#      (`--critico centralizado --contrafactual 4`, vantagem por drone):
#        T16 → ponto 61 da grade a2s12-24-36 (passo 8e-4, σ 0,35, n 12, φ 0,85, m 8)
#        T32 → passo 8e-4 · φ 0,99 (EXTRA_T32), NÃO o topo da grade (ponto 90:
#              1,5e-3 · φ 0,70, 1/3 abortos) — a escolha registrada em
#              reports/spec06-vencedores.md para o modelo publicado de 2026-10-16;
#      se já houver um treino da célula em curso, ele é ADOTADO (pid vigiado);
#   2. espera os dois terminarem;
#   3. atestados RF-05 dos dois modelos nas 3 células de Na=2 (pré-requisito do
#      regime com-kf);
#   4. campanha de Na=2 (a2s12/24/36 × limpo, sem-kf, com-kf) com o rótulo
#      `wnn-ctde` — os CSVs do MADDPG e dos `wnn-v2` já existem nessas células.
#
#     nohup bash scripts/spec06_ctde_na2.sh > /home/rnagao/mestrado/fusao/reports/spec06-ctde-na2.log 2>&1 &
#     tail -n 3 -f /home/rnagao/mestrado/fusao/reports/spec06-ctde-na2.log \
#         /home/rnagao/mestrado/fusao/reports/spec06-50M-ctde-T16-a2s12-24-36.log \
#         /home/rnagao/mestrado/fusao/reports/spec06-50M-ctde-T32-a2s12-24-36.log
#
# Se um dos treinos morrer sem `checkpoints/deploy`, o gate pula o modelo
# ("PULADO") e a fila da campanha bloqueia as corridas dele ("sem modelo"); o
# outro segue. Tempo esperado: T16 ~26 h, T32 ~42 h (10–20 % acima dos 22 h e
# 35 h dos descentralizados), depois ~2 h de campanha.
set -u
cd "$(dirname "$0")/.."
export RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}
export RUNS=${RUNS:-spec06-50M-ctde}
ROT=${ROT:-wnn-ctde}
K=${K:-4}
TS=( ${TS:-16 32} )
EXTRA=${EXTRA:---c-max none}        # como nos treinos de Na=2 de 2026-10-15/16
EXTRA_T16=${EXTRA_T16:-}
EXTRA_T32=${EXTRA_T32:---passo 8e-4 --phi 0.99}

ts() { date '+%F %T'; }
PIDS=()
for T in "${TS[@]}"; do
  pid=$(pgrep -f "spec06_treino_multiagente.py --na 2 --ns 12,24,36 --resolucao $T " | head -1 || true)
  if [ -n "$pid" ]; then
    echo "[$(ts)] treino T$T Na=2 já em curso (pid $pid) — adotado"; PIDS+=("$pid"); continue
  fi
  por_t="EXTRA_T$T"
  echo "[$(ts)] lançando treino T$T Na=2 (CTDE, K=$K) em runs/$RUNS · flags: $EXTRA ${!por_t}"
  # shellcheck disable=SC2086
  saida=$(bash scripts/spec06_treino_celula.sh 2 "$T" --critico centralizado --contrafactual "$K" $EXTRA ${!por_t}) || {
    echo "[$(ts)] FALHOU ao lançar T$T:"; echo "$saida"; continue; }
  echo "$saida"
  pid=$(echo "$saida" | grep -oE 'pid [0-9]+' | head -1 | cut -d' ' -f2)
  [ -n "$pid" ] && PIDS+=("$pid")
done
[ ${#PIDS[@]} -eq 0 ] && { echo "[$(ts)] nenhum treino lançado — abortando"; exit 1; }

echo "[$(ts)] esperando pids ${PIDS[*]} (checagem a cada 10 min)"
while :; do
  vivos=0
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && vivos=$((vivos+1)); done
  [ "$vivos" -eq 0 ] && break
  sleep 600
done
for T in "${TS[@]}"; do
  d="$RAIZ_DADOS/runs/$RUNS/T$T-a2s12-24-36"
  if [ -d "$d/checkpoints/deploy" ]; then
    echo "[$(ts)] treino T$T terminou com deploy · $(grep -E '^Na=2' "$RAIZ_DADOS/reports/$RUNS-T$T-a2s12-24-36.log" | tail -1)"
  else
    echo "[$(ts)] treino T$T SEM checkpoints/deploy — ver $RAIZ_DADOS/reports/$RUNS-T$T-a2s12-24-36.log"
  fi
done

echo "[$(ts)] atestados RF-05 ($ROT, Na=2)"
NAS=2 PARALELO=2 ROT="$ROT" bash scripts/spec06_gates_novos.sh
echo "[$(ts)] campanha Na=2 ($ROT)"
NAS=2 ROT="$ROT" COM_MADDPG=0 bash scripts/spec06_campanha_novos.sh
echo "[$(ts)] FIM — CSVs em $RAIZ_DADOS/reports/campanha/$ROT-T*-a2s*.csv"
