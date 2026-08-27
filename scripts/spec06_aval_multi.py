"""Avalia `melhor` e `final` de uma célula multiagente no banco de teste."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
from spec06_treino_multiagente import avaliar_multi, D_SAIDA, N_TUPLA, PHI, PASSO
from spec06_treinar import _cfgs, _ctx
from uavwnn.agents import APROXIMADORES
from uavwnn.campanha import carregar_bancos
from uavwnn.encoding import TermometroQuantil

d = Path(sys.argv[1]); na, ns = int(sys.argv[2]), int(sys.argv[3])
proc = json.load(open(d / "procedencia.json"))
t_res = proc["resolucao"]
cfg_tr, cfg_av = _cfgs("1b", na, ns); ctx_av = _ctx(cfg_av)
test = carregar_bancos(na, ns).test.sementes[:100]
for qual in ("melhor", "final"):
    dc = d / "checkpoints" / qual
    if not dc.exists():
        print(f"{qual}: ausente"); continue
    termo = TermometroQuantil.carregar(dc / "termometro.npz")
    ap = APROXIMADORES["ntupla_dissertacao"](
        n_bits=termo.n_bits, tamanho_tupla=N_TUPLA, phi=PHI, taxa_ator=PASSO,
        seed=0, d_saida=D_SAIDA, n_dims_codificador=cfg_tr.observation_size,
        limiares_por_dim=t_res)
    ap.ator = type(ap.ator).carregar(next(dc.glob("*ator.npz")))
    ap.critico = type(ap.critico).carregar(next(dc.glob("*critico.npz")))
    m = avaliar_multi(cfg_av, ap, termo, ctx_av, test)
    print(f"{qual:<7} cob {m['cobertura']*100:5.1f}%  SR {m['sr']*100:5.1f}%  "
          f"CT_cens {m['ct_cens']:6.1f}s  (n={m['n']})")
