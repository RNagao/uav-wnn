#!/usr/bin/env bash
# Painel das retomadas até 50 M passos.
#
# Lê por MAIOR contagem de passos, não por `tail`: o log do T16-a1s12 tem uma
# linha órfã da corrida do zero que foi morta, gravada num deslocamento POSTERIOR
# ao ponto em que a retomada escreve. `tail -1` pegaria justamente ela.
cd "$(dirname "$0")/.."
printf '  %-11s %13s %7s %10s %9s %7s  %s\n' célula passos pct ep passos/s ETA proc
for f in reports/spec06-50M-T*.log; do
  n=$(basename "$f" .log | sed 's/spec06-50M-//')
  vivo=não
  for d in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in *python\ scripts/spec06_treino_multiagente*"--saida runs/spec06-50M/$n"*) vivo=sim ;; esac
  done
  tr -d '\0' < "$f" \
  | grep -oE '\[ *[0-9.]+%\] [0-9,]+/[0-9,]+ · [0-9,]+ ep · [0-9]+ passos/s · ETA [0-9.]+ h' \
  | awk -v n="$n" -v vivo="$vivo" '
      { p=$0; sub(/^.*\] /,"",p); sub(/\/.*/,"",p); q=p; gsub(/,/,"",q)
        if (q+0>m) { m=q+0; passos=p
          pct=$0;  sub(/^\[ */,"",pct); sub(/%.*/,"%",pct)
          ep=$0;   sub(/^.*· /,"",ep);  ep=$0; match($0,/· [0-9,]+ ep/); ep=substr($0,RSTART+2,RLENGTH-5)
          tx=$0;   match($0,/[0-9]+ passos\/s/); tx=substr($0,RSTART,RLENGTH-9)
          eta=$0;  match($0,/ETA [0-9.]+ h/);   eta=substr($0,RSTART+4,RLENGTH-4) } }
      END { if (m) printf "  %-11s %13s %7s %10s %9s %7s  %s\n", n, passos, pct, ep, tx, eta, vivo
            else    printf "  %-11s %13s %7s %10s %9s %7s  %s\n", n, "—","—","—","—","—", vivo }'
done
echo
echo "  última avaliação no val:"
for f in reports/spec06-50M-T*.log; do
  n=$(basename "$f" .log | sed 's/spec06-50M-//')
  e=$(tr -d '\0' < "$f" | grep 'eval @' | tail -1)
  printf '    %-11s %s\n' "$n" "${e:-(nenhuma ainda — a 1ª sai a cada 1,25 M passos)}"
done
