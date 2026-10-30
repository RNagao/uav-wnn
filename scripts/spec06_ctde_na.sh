#!/bin/bash
# Um Na com CRÍTICO CENTRALIZADO (CTDE), de ponta a ponta e sem intervenção — a
# generalização do `spec06_ctde_na2.sh` (que fica como está, porque está rodando):
#   1. treina T16 e T32 do Na pedido em paralelo, MESMAS configs dos modelos
#      descentralizados daquele Na (ver tabela abaixo), só trocando o crítico
#      (`--critico centralizado --contrafactual K`, vantagem por drone); adota um
#      treino da célula já em curso em vez de duplicar;
#   2. espera os dois terminarem;
#   3. atestados RF-05 dos dois modelos nas 3 células do Na (pré-requisito do com-kf);
#   4. campanha do Na: limpo + σ 10 e 25 (sem-kf, com-kf), rótulo `wnn-ctde`.
#
# Configs dos modelos descentralizados de 50 M (procedência em runs/spec06-50M/*):
#   Na=2  T16 ponto 61 da grade (8e-4 · φ 0,85)   T32 8e-4 · φ 0,99 (escolha do registro)
#   Na=4  T16 8e-4 · φ 0,70 (id 60)               T32 8e-4 · φ 0,85  — ambos c_max off,
#         guarda de colapso ligada, pararam por platô em 33,8 M
#   Na=8  T16 e T32 1,5e-3 · φ 0,85 (ids 110 e 80 com flags explícitas); T16 foi a
#         50 M (125 passos/s), T32 parou por platô em 42,5 M (81 passos/s)
#   Outro Na: passe EXTRA_T16 e EXTRA_T32 explicitamente.
#
#     NA=4 nohup bash scripts/spec06_ctde_na.sh > /home/rnagao/mestrado/fusao/reports/spec06-ctde-na4.log 2>&1 &
#     tail -n 3 -f /home/rnagao/mestrado/fusao/reports/spec06-ctde-na4.log \
#         /home/rnagao/mestrado/fusao/reports/spec06-50M-ctde-T16-a4s12-24-36.log \
#         /home/rnagao/mestrado/fusao/reports/spec06-50M-ctde-T32-a4s12-24-36.log
set -u
cd "$(dirname "$0")/.."
export RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}
export RUNS=${RUNS:-spec06-50M-ctde}
NA=${NA:?defina NA (2, 4 ou 8)}
ROT=${ROT:-wnn-ctde}
K=${K:-4}
TS=( ${TS:-16 32} )
SIGMAS=${SIGMAS:-10 25}
EXTRA=${EXTRA:---c-max none}
case "$NA" in
  2) : "${EXTRA_T16:=}";                    : "${EXTRA_T32:=--passo 8e-4 --phi 0.99}" ;;
  4) : "${EXTRA_T16:=--passo 8e-4 --phi 0.7}"; : "${EXTRA_T32:=--passo 8e-4 --phi 0.85}" ;;
  8) : "${EXTRA_T16:=--passo 1.5e-3 --phi 0.85}"; : "${EXTRA_T32:=--passo 1.5e-3 --phi 0.85}" ;;
  *) : "${EXTRA_T16:?Na=$NA sem config padrão: defina EXTRA_T16}"; : "${EXTRA_T32:?defina EXTRA_T32}" ;;
esac
CEL="a${NA}s12-24-36"
ts() { date '+%F %T'; }

PIDS=()
for T in "${TS[@]}"; do
  pid=$(pgrep -f "spec06_treino_multiagente.py --na $NA --ns 12,24,36 --resolucao $T " | head -1 || true)
  if [ -n "$pid" ]; then
    echo "[$(ts)] treino T$T Na=$NA já em curso (pid $pid) — adotado"; PIDS+=("$pid"); continue
  fi
  por_t="EXTRA_T$T"
  echo "[$(ts)] lançando treino T$T Na=$NA (CTDE, K=$K) em runs/$RUNS · flags: $EXTRA ${!por_t}"
  # shellcheck disable=SC2086
  saida=$(bash scripts/spec06_treino_celula.sh "$NA" "$T" --critico centralizado --contrafactual "$K" $EXTRA ${!por_t}) || {
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
  d="$RAIZ_DADOS/runs/$RUNS/T$T-$CEL"
  if [ -d "$d/checkpoints/deploy" ]; then
    echo "[$(ts)] treino T$T terminou com deploy · $(grep -E "^Na=$NA" "$RAIZ_DADOS/reports/$RUNS-T$T-$CEL.log" | tail -1)"
  else
    echo "[$(ts)] treino T$T SEM checkpoints/deploy — ver $RAIZ_DADOS/reports/$RUNS-T$T-$CEL.log"
  fi
done

echo "[$(ts)] atestados RF-05 ($ROT, Na=$NA)"
NAS=$NA PARALELO=2 ROT="$ROT" bash scripts/spec06_gates_novos.sh
echo "[$(ts)] campanha Na=$NA ($ROT) · limpo"
NAS=$NA ROT="$ROT" COM_MADDPG=0 REGIMES=limpo bash scripts/spec06_campanha_novos.sh
for S in $SIGMAS; do
  echo "[$(ts)] campanha Na=$NA ($ROT) · σ=$S · sem-kf,com-kf"
  NAS=$NA ROT="$ROT" COM_MADDPG=0 REGIMES=sem-kf,com-kf SIGMAS=$S bash scripts/spec06_campanha_novos.sh
done
echo "[$(ts)] FIM — CSVs em $RAIZ_DADOS/reports/campanha/$ROT-T*-a${NA}s*.csv"
