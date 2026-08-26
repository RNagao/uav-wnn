#!/usr/bin/env bash
# Painel das retomadas até 50 M passos: progresso + curva de validação.
#
# As métricas saem do `metrics.jsonl` de cada corrida, não do log: o log é prosa
# e já mostrou que se corrompe quando duas corridas dividem o arquivo.
# A coluna `proc` confere /proc/*/cmdline, não o log — um treino morto deixa o
# log intacto e continuaria parecendo vivo.
cd "$(dirname "$0")/.."

printf '  %-11s %13s %6s %9s %8s %7s  %s\n' célula passos pct ep passos/s ETA proc
for f in reports/spec06-50M-T*.log; do
  n=$(basename "$f" .log | sed 's/spec06-50M-//')
  vivo=não
  for d in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in *python\ scripts/spec06_treino_multiagente*"--saida runs/spec06-50M/$n "*) vivo=sim ;; esac
  done
  tr -d '\0' < "$f" \
  | grep -oE '\[ *[0-9.]+%\] [0-9,]+/[0-9,]+ · [0-9,]+ ep · [0-9]+ passos/s · ETA [0-9.]+ h' \
  | awk -v n="$n" -v v="$vivo" '
      { q=$0; sub(/^.*\] /,"",q); sub(/\/.*/,"",q); r=q; gsub(/,/,"",r)
        if (r+0>m) { m=r+0; P=q
          pc=$0; sub(/^\[ */,"",pc); sub(/%.*/,"%",pc)
          match($0,/· [0-9,]+ ep/); E=substr($0,RSTART+2,RLENGTH-5)
          match($0,/[0-9]+ passos\/s/); T=substr($0,RSTART,RLENGTH-9)
          match($0,/ETA [0-9.]+ h/); A=substr($0,RSTART+4,RLENGTH-4) } }
      END { if (m) printf "  %-11s %13s %6s %9s %8s %7s  %s\n", n,P,pc,E,T,A,v
            else    printf "  %-11s %13s %6s %9s %8s %7s  %s\n", n,"—","—","—","—","—",v }'
done

echo
echo "  validação (100 sementes do banco val, política sem ruído de exploração):"
printf '    %-11s %11s %7s %7s %9s %10s\n' célula 'eval @' cob SR 'CT (s)' ocupadas
for j in runs/spec06-50M/*/metrics.jsonl; do
  [ -f "$j" ] || continue
  n=$(basename "$(dirname "$j")")
  tail -1 "$j" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(f"    {"'"$n"'":<11} {d["passo"]:>11,} {d["cobertura"]:>7.3f} {d["sr"]:>7.3f} '\
'{d["ct_cens"]:>9.1f} {d["ocupadas"]:>10,}")'
done
[ -n "$(ls runs/spec06-50M/*/metrics.jsonl 2>/dev/null)" ] || \
  echo "    (nenhuma ainda — a 1ª avaliação sai a cada 1,25 M passos)"
