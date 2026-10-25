#!/bin/bash
# Campanha dos modelos NOVOS (uma grade por Na, Ns revezado 12/24/36): cada
# modelo percorre SÓ as células do seu Na — o T16 e o T32 treinados com Na=2
# rodam em a2s12, a2s24 e a2s36; os de Na=4 em a4s*; os de Na=8 em a8s*.
#
# É um embrulho de `spec06_campanha_fila.py`: o `{na}` no caminho do modelo é
# substituído pelo Na de cada célula, então um lançamento cobre os três Na. O
# MADDPG entra como baseline nas mesmas células (checkpoints publicados de
# Lamenza, por célula). Cada corrida grava CSV + log + manifesto com o
# checkpoint usado, e a fila recusa reaproveitar um CSV de outro checkpoint.
#
# ORIGEM do modelo (variável de ambiente):
#   deploy  (padrão)  runs/spec06-50M/T<T>-a<Na>s12-24-36/checkpoints/deploy
#                     — o artefato podado/verificado que o treinador gravou a
#                     partir do `final` (em Na=2 T32 a poda foi recusada: f32 só)
#   melhor            .../checkpoints/melhor   — melhor validação (rede cheia)
#   final             .../checkpoints/final    — estado ao fim do treino
#   models            models/deploy/T<T>/a<Na>s12-24-36 — depois da promoção
#
#     bash scripts/spec06_campanha_novos.sh --listar            # o que vai rodar
#     bash scripts/spec06_campanha_novos.sh --seco              # só os comandos
#     nohup bash scripts/spec06_campanha_novos.sh > reports/campanha/logs/_novos.log 2>&1 &
#     NAS=2 REGIMES=limpo bash scripts/spec06_campanha_novos.sh  # um Na, um regime
#     ORIGEM=melhor ROT=wnn-v2m bash scripts/spec06_campanha_novos.sh --listar
#     bash scripts/spec06_campanha_novos.sh --parar
#
# O rótulo (ROT) é a identidade dos CSVs: `wnn-v2-T16-a4s24-limpo.csv`. Mudou o
# checkpoint, mude o rótulo — a guarda do manifesto recusa o contrário.
set -eu
cd "$(dirname "$0")/.."
ORIGEM=${ORIGEM:-deploy}
NAS=${NAS:-2,4,8}
NSS=${NSS:-12,24,36}
REGIMES=${REGIMES:-limpo,sem-kf,com-kf}
ROT=${ROT:-wnn-v2}
PARALELO=${PARALELO:-6}
SIGMAS=${SIGMAS:-10}
COM_MADDPG=${COM_MADDPG:-1}
RAIZ_DADOS=${RAIZ_DADOS:-/home/rnagao/mestrado/fusao}

case "$ORIGEM" in
  models)              PAD="$RAIZ_DADOS/models/deploy/T%s/a{na}s12-24-36" ;;
  deploy|melhor|final) PAD="$RAIZ_DADOS/runs/spec06-50M/T%s-a{na}s12-24-36/checkpoints/$ORIGEM" ;;
  *) echo "ORIGEM desconhecida: $ORIGEM (deploy|melhor|final|models)"; exit 1 ;;
esac
# shellcheck disable=SC2059  # o padrão tem só o %s do T; `{na}` fica literal
MODELOS="$ROT-T16=$(printf "$PAD" 16),$ROT-T32=$(printf "$PAD" 32)"
[ "$COM_MADDPG" = 1 ] && MODELOS="$MODELOS,maddpg"
CELULAS=()
for na in ${NAS//,/ }; do CELULAS+=("$na:$NSS"); done

exec uv run python scripts/spec06_campanha_fila.py \
  --modelos "$MODELOS" --celulas "${CELULAS[@]}" --regimes "$REGIMES" \
  --sigmas "$SIGMAS" --paralelo "$PARALELO" \
  --gates "$RAIZ_DADOS/results/etapa2" --dir-saida "$RAIZ_DADOS/reports/campanha" "$@"
