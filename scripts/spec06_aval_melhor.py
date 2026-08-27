"""Avalia os checkpoints `melhor` da campanha longa no banco de teste (100 eps).

Existe porque a corrida longa rodou com o código anterior a `539b61e`: a avaliação
final mediu a REDE VIVA no fim do treino, não o checkpoint de melhor validação.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
from spec06_treinar import REPLICAS, _cfgs, _ctx, avaliar, corpus_de_observacoes
from uavwnn.agents import APROXIMADORES
from uavwnn.agents.ac_continuo import AtorCriticoContinuo
from uavwnn.agents.politica_continua import NormalProjetada
from uavwnn.campanha import carregar_bancos
from uavwnn.encoding import TermometroQuantil

RAIZ = Path("runs/spec06-longa/spec06-busca-1b")
ALVOS = [("15b49e642647", "8e-4/0,35/T32"),
         ("8b328b1d6887", "8e-4/0,35/T16"),
         ("e11c9f45b54b", "5e-4/0,20/T32")]

test = carregar_bancos(1, 12).test.sementes[:100]
cfg_tr, cfg_av = _cfgs("1b", 1, 12)
ctx, ctx_av = _ctx(cfg_tr), _ctx(cfg_av)
out = []
for h, rot in ALVOS:
    for s in (0, 1):
        d = RAIZ / h / str(s)
        dm = d / "checkpoints" / "melhor"
        if not dm.exists():
            print(f"{rot} s{s}: sem checkpoint melhor"); continue
        c = json.load(open(d / "config.json"))["config"]
        termo = TermometroQuantil.carregar(dm / "termometro.npz")
        ap = APROXIMADORES[c["aproximador"]](
            n_bits=termo.n_bits, tamanho_tupla=c["tupla"], phi=c["phi"],
            taxa_ator=c["taxa_ator"], seed=0, d_saida=3,
            n_dims_codificador=cfg_tr.observation_size,
            limiares_por_dim=c["resolucao"])
        ap.ator = type(ap.ator).carregar(next(dm.glob("*ator.npz")))
        ap.critico = type(ap.critico).carregar(next(dm.glob("*critico.npz")))
        pol = AtorCriticoContinuo(
            aproximador=ap,
            codificar=lambda v, _t=termo: _t.codificar(np.asarray(v, float)[None, :])[0],
            politica=NormalProjetada(c["sigma"], None), passos=c["passos_m"],
            desconto=0.99, id="aval", treinando=False)
        m = avaliar(cfg_av, pol, ctx_av, test)
        # passo em que o `melhor` foi gravado
        M = [json.loads(l) for l in open(d / "metrics.jsonl")]
        b = max(M, key=lambda x: x["cobertura"])
        r = dict(config=rot, seed=s, hash=h, passo_melhor=b["passo"],
                 cobertura=m["cobertura"], sr=m["sr"],
                 ct_cens=m["ct_cens"], ct_succ=m["ct_succ"])
        out.append(r)
        print(f"{rot:<16} s{s}  melhor@{b['passo']:>9,}  "
              f"cob {m['cobertura']*100:5.1f}%  SR {m['sr']*100:5.1f}%  "
              f"CT_succ {m['ct_succ']:5.1f}s  CT_cens {m['ct_cens']:6.1f}s", flush=True)
json.dump(out, open("reports/spec06-longa-melhor-ckpt.json", "w"), indent=2)
print("\n-> reports/spec06-longa-melhor-ckpt.json")
