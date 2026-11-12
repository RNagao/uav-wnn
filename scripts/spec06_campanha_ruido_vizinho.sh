#!/bin/bash
# Refaz TODAS as corridas sob ruído depois da mudança "vizinho recebe a posição
# divulgada" (commit f263c77, mesclado em 2026-10-28): até então o vizinho entrava
# na observação com a posição VERDADEIRA mesmo sob ruído de GPS. A mudança vive
# em `uavwnn/fusion/ambiente_ruidoso.py`, o mesmo envoltório que a campanha usa
# para o MADDPG e para a WNN — vale para todos os modelos igualmente.
#
# O regime `limpo` não muda (sem ruído, divulgada = verdadeira) e não é refeito.
# `Na = 1` não tem vizinho e também não é refeito. σ = 50 sai (decisão de
# 2026-10-28): só σ = 10 e 25, nos regimes sem-kf e com-kf.
#
#   1. arquiva os CSVs/logs/manifestos de ruído de a2/a4/a8 e os atestados RF-05
#      desses modelos em `_vizinho-exato-ate-2026-10-28/` (nada é apagado);
#   2. regenera os atestados RF-05 (o atestado vale sob a mobilidade da política
#      naquela célula, e a mobilidade muda com o que o vizinho vê): MADDPG nas 9
#      células, wnn-T16/T32 publicados em a2, wnn-v2 T16/T32 nas 9 células —
#      reprovado na grade padrão é refeito numa grade fina;
#   3. campanhas: wnn-v2 + MADDPG nas 9 células, e os publicados em a2, para
#      σ = 10 e σ = 25.
#
#     nohup bash scripts/spec06_campanha_ruido_vizinho.sh > /home/rnagao/mestrado/fusao/reports/campanha/logs/_ruido-vizinho.log 2>&1 &
#     tail -f /home/rnagao/mestrado/fusao/reports/campanha/logs/_ruido-vizinho.log
#
# ~132 corridas de 100 runs; a campanha de 81 corridas levou 8 h 40 com 6 em
# paralelo — estimar ~14 h, mais ~40 min de atestados.
set -u
cd "$(dirname "$0")/.."
RAIZ=$(pwd)
export RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}
SIGMAS=${SIGMAS:-10 25}
PARALELO=${PARALELO:-6}
C="$RAIZ_DADOS/reports/campanha"
G="$RAIZ_DADOS/results/etapa2"
ARQ="$C/_vizinho-exato-ate-2026-10-28"
ARQG="$G/_vizinho-exato-ate-2026-10-28"
ts() { date '+%F %T'; }

# -- 1. arquivar -----------------------------------------------------------
if [ ! -d "$ARQ" ]; then
  mkdir -p "$ARQ/logs" "$ARQG"
  n=0
  for f in "$C"/*-a[248]s[0-9]*-sigma*.csv; do [ -e "$f" ] && mv "$f" "$ARQ/" && n=$((n+1)); done
  for f in "$C"/logs/*-a[248]s[0-9]*-sigma*.log "$C"/logs/*-a[248]s[0-9]*-sigma*.manifest.json; do
    [ -e "$f" ] && mv "$f" "$ARQ/logs/" && n=$((n+1)); done
  for f in "$G"/gate_rf05-maddpg-a[248]s*.json "$G"/gate_rf05-wnn-T16-a2s*.json "$G"/gate_rf05-wnn-T32-a2s*.json \
           "$G"/gate_rf05-wnn-v2-T16-a[248]s*.json "$G"/gate_rf05-wnn-v2-T32-a[248]s*.json; do
    [ -e "$f" ] && mv "$f" "$ARQG/" && n=$((n+1)); done
  for f in "$C"/logs/gate-maddpg-a[248]s*.log "$C"/logs/gate-wnn-T16-a2s*.log "$C"/logs/gate-wnn-T32-a2s*.log \
           "$C"/logs/gate-wnn-v2-*.log; do
    [ -e "$f" ] && mv "$f" "$ARQ/logs/" && n=$((n+1)); done
  echo "[$(ts)] arquivados $n arquivos (vizinho exato) em $ARQ e $ARQG"
else
  echo "[$(ts)] arquivo $ARQ já existe — não arquivo de novo"
fi

# -- 2. atestados ----------------------------------------------------------
pub() {   # checkpoint do modelo PUBLICADO de Na=2: o podado (deploy) se tiver ator; senão o cheio
  local d="$RAIZ_DADOS/models/deploy/T$1/a2s24"
  ls "$d"/*ator.npz >/dev/null 2>&1 || d="$RAIZ_DADOS/models/T$1/a2s24"
  echo "$d"
}
gate() {   # $1=rótulo $2=Na $3=Ns $4=checkpoint|-   — grade padrão e, se reprovar, grade fina
  local R=$1 NA=$2 NS=$3 CKPT=$4
  local OUT="$G/gate_rf05-$R-a${NA}s${NS}.json" LOG="$C/logs/gate-$R-a${NA}s${NS}.log"
  [ -f "$OUT" ] && { echo "já existe: $OUT"; return 0; }
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  cd "$RAIZ"
  for GRADE in 0,0.25,1,4,8,16,32,64 0,0.25,0.5,1,2,3,4,5,6,8,12,16,32,64; do
    if [ "$CKPT" = "-" ]; then
      uv run --group baseline python scripts/spec08_gate_rf05.py --maddpg --na "$NA" --ns "$NS" \
        --episodios 100 --q-w-grade "$GRADE" --saida "$OUT" > "$LOG" 2>&1
    else
      uv run python scripts/spec08_gate_rf05.py --checkpoint "$CKPT" --aproximador ntupla_dissertacao \
        --na "$NA" --ns "$NS" --episodios 100 --q-w-grade "$GRADE" --saida "$OUT" > "$LOG" 2>&1
    fi
    if grep -qE "=== APROVADO ===" "$LOG"; then
      echo "OK  $R a${NA}s${NS} · $(grep -oE 'melhor q_w=[0-9.]+' "$LOG" | tail -1) · grade $GRADE"; return 0
    fi
    echo "reprovado na grade $GRADE: $R a${NA}s${NS} — tentando grade fina"
    rm -f "$OUT"
  done
  echo "REPROVADO  $R a${NA}s${NS} — ver $LOG"
}
export -f gate pub; export RAIZ RAIZ_DADOS G C
echo "[$(ts)] atestados RF-05 (vizinho divulgado)"
{
  for NA in 2 4 8; do for NS in 12 24 36; do
    echo "maddpg $NA $NS -"
    for T in 16 32; do echo "wnn-v2-T$T $NA $NS $RAIZ_DADOS/runs/spec06-50M/T$T-a${NA}s12-24-36/checkpoints/deploy"; done
  done; done
  for NS in 12 24 36; do for T in 16 32; do echo "wnn-T$T 2 $NS $(pub $T)"; done; done
} | xargs -d '\n' -P "$PARALELO" -I{} bash -c 'gate $1 $2 $3 $4' _ {}
echo "[$(ts)] atestados prontos: $(ls "$G"/gate_rf05-*-a[248]s*.json | wc -l) em $G"

# -- 3. campanhas ----------------------------------------------------------
for S in $SIGMAS; do
  echo "[$(ts)] campanha σ=$S · wnn-v2 + MADDPG · a2/a4/a8 · sem-kf,com-kf"
  NAS=2,4,8 REGIMES=sem-kf,com-kf SIGMAS=$S PARALELO=$PARALELO COM_MADDPG=1 \
    bash scripts/spec06_campanha_novos.sh
  echo "[$(ts)] campanha σ=$S · publicados wnn-T16/T32 · a2 · sem-kf,com-kf"
  uv run python scripts/spec06_campanha_fila.py \
    --modelos "wnn-T16=$(pub 16 | sed 's/a2s24/a{na}s24/'),wnn-T32=$(pub 32 | sed 's/a2s24/a{na}s24/')" \
    --celulas 2:12,24,36 --regimes sem-kf,com-kf --sigmas "$S" --paralelo "$PARALELO" \
    --gates "$G" --dir-saida "$C"
done
echo "[$(ts)] FIM — $(ls "$C"/*-sigma*.csv | grep -v episodios | wc -l) CSVs de ruído em $C"
