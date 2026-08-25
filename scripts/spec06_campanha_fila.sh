#!/bin/bash
# Fila da campanha comparável — MADDPG × WNN T16 × WNN T32.
#
# CÉLULAS: Na=1/Ns=12 (só WNN — Lamenza não publica single-drone) e as 9 células
# multiagente da tab. 7.1 (Na ∈ {2,4,8} × Ns ∈ {12,24,36}), que são exatamente as
# que o submodule publica checkpoint.
#
# TRANSFERÊNCIA: `d_obs` depende só de Na, não de Ns — a mesma rede WNN roda em
# qualquer Ns. Mas a WNN foi treinada em Ns=24 e o MADDPG tem checkpoint por
# célula, então em Ns=12 e Ns=36 o MADDPG joga em casa e a WNN transfere. O CSV
# marca isso em `treinado_em_ns`/`transferencia`; ignorar a coluna ao montar a
# tabela final produziria uma comparação injusta sem parecer.
#
# Cada corrida grava run a run e relê o cache ao reiniciar: matar e relançar não
# perde nada além da run em voo.
#
# REGIMES: `limpo` (sem ruído), `sem-kf` (σ=10, medida crua — o piso) e `com-kf`
# (σ=10, KF + entrada de controle — a proposta). Cada um é uma campanha inteira
# de 20 mil episódios no MESMO degrau, não um sorteio: a pergunta é "o KF ajuda
# nesta célula?", e para respondê-la os dois braços precisam do n cheio.
#
#     bash scripts/spec06_campanha_fila.sh            # roda
#     bash scripts/spec06_campanha_fila.sh --listar   # mostra a fila e o estado
#     PARALELO=3 bash scripts/spec06_campanha_fila.sh # deixa folga p/ outro treino
#
# NICE. As corridas rodam com prioridade rebaixada (`nice -n 10`). O treino do
# Algoritmo 4.4 é SINGLE-THREAD e satura um núcleo; sem o `nice`, N corridas
# disputando com ele em igualdade roubariam fatias de tempo e o treino perderia
# vazão. Rebaixadas, elas só pegam núcleo que o treino não quer — e como há 12
# núcleos e o treino usa 1, isso é quase o tempo todo.
set -u
cd "$(dirname "$0")/.."
RAIZ=$(pwd); PARALELO=${PARALELO:-6}; SIGMAS=${SIGMAS:-10}
G=results/etapa2
NS_TREINO=24          # o Ns em que as redes WNN multiagente foram treinadas

ckpt_wnn() {   # $1=T (16|32)  $2=Na
  if [ "$2" = 1 ]; then echo "models/T$1/a1s12"; else echo "models/T$1/a${2}s${NS_TREINO}"; fi
}
gate_de() {    # $1=rótulo  $2=Na  $3=Ns
  echo "$G/gate_rf05-$1-a${2}s${3}.json"
}

fila() {
  for REG in limpo sem-kf com-kf; do
    for T in 16 32; do
      echo "wnn|wnn-T$T|1|12|$REG|$(ckpt_wnn $T 1)|$(gate_de wnn-T$T 1 12)|12"
    done
    for NA in 2 4 8; do
      for NS in 12 24 36; do
        echo "maddpg|maddpg|$NA|$NS|$REG|-|$(gate_de maddpg $NA $NS)|$NS"
        for T in 16 32; do
          echo "wnn|wnn-T$T|$NA|$NS|$REG|$(ckpt_wnn $T $NA)|$(gate_de wnn-T$T $NA $NS)|$NS_TREINO"
        done
      done
    done
  done
}

uma() {
  IFS='|' read -r MOD ROT NA NS REG CKPT GATE TRNS <<< "$1"
  cd "$RAIZ"
  if [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ]; then
    echo "PULADA: $ROT a${NA}s${NS} $REG — sem $CKPT"; return 0
  fi
  # Só o `com-kf` depende do atestado: `sem-kf` é a medida crua, não há filtro
  # para atestar. Exigir gate nele bloquearia metade da comparação à toa.
  if [ "$REG" = "com-kf" ] && [ ! -f "$GATE" ]; then
    echo "PULADA: $ROT a${NA}s${NS} $REG — sem atestado $GATE"; return 0
  fi
  mkdir -p reports/campanha/logs
  local SUF DEG RUIDO
  case "$REG" in
    limpo)  SUF=limpo;                        RUIDO=limpo;   DEG="" ;;
    sem-kf) SUF="sigma${SIGMAS//,/-}-sem-kf"; RUIDO=sorteio; DEG=1  ;;
    com-kf) SUF="sigma${SIGMAS//,/-}-com-kf"; RUIDO=sorteio; DEG=4  ;;
  esac
  local LOG="reports/campanha/logs/${ROT}-a${NA}s${NS}-${SUF}.log"
  local ARGS=(--modelo "$MOD" --rotulo "$ROT" --na "$NA" --ns "$NS" --ruido "$RUIDO"
              --runs 100 --episodios 200 --treinado-em-ns "$TRNS"
              --dir-saida reports/campanha)
  [ "$CKPT" != "-" ] && ARGS+=(--checkpoint "$CKPT")
  [ -n "$DEG" ] && ARGS+=(--gate "$GATE" --sigmas "$SIGMAS" --degraus "$DEG")
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  local NICE=${NICE:-10}
  if [ "$MOD" = "maddpg" ]; then
    nice -n "$NICE" uv run --group baseline python \
      scripts/spec06_campanha_completa.py "${ARGS[@]}" > "$LOG" 2>&1
  else
    nice -n "$NICE" uv run python \
      scripts/spec06_campanha_completa.py "${ARGS[@]}" > "$LOG" 2>&1
  fi
}
export -f uma; export RAIZ SIGMAS NICE

if [ "${1:-}" = "--listar" ]; then
  fila | while IFS='|' read -r MOD ROT NA NS REG CKPT GATE TRNS; do
    case "$REG" in
      limpo)  SUF=limpo ;;
      sem-kf) SUF="sigma${SIGMAS//,/-}-sem-kf" ;;
      com-kf) SUF="sigma${SIGMAS//,/-}-com-kf" ;;
    esac
    CSV="reports/campanha/${ROT}-a${NA}s${NS}-${SUF}.csv"
    F=0; [ -f "$CSV" ] && F=$(( $(wc -l < "$CSV") - 1 ))
    ST=roda
    [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ] && ST="PULA (sem checkpoint)"
    [ "$REG" = "com-kf" ] && [ ! -f "$GATE" ] && ST="PULA (sem atestado)"
    [ "$F" -ge 100 ] && ST=PRONTA
    X=$([ "$MOD" = wnn ] && [ "$TRNS" != "$NS" ] && echo " ·transf" || echo "")
    printf '%-9s Na=%-2s Ns=%-3s %-8s %3s/100  %s%s\n' "$ROT" "$NA" "$NS" "$SUF" "$F" "$ST" "$X"
  done
  exit 0
fi
fila | xargs -d '\n' -P "$PARALELO" -I{} bash -c 'uma "$@"' _ {}
