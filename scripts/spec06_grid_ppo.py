"""Grade do braço **PPO** — a rede do `anbubus` com o algoritmo dela, `Na = 1`.

POR QUE ESTA GRADE EXISTE

As 720 corridas do `spec06_busca.py` compararam as duas **redes** sob o **Algoritmo
4.4**, que é o algoritmo da dissertação. A rede do `anbubus` é um aproximador de
função para PPO: o `experiment.py` de todas as oito tarefas publicadas roda
`tune.run("PPO", …)`. Rodá-la sob o Algoritmo 4.4 a deixa sem nenhum dos mecanismos
que a mantinham estável:

    recorte do objetivo substituto   `clip_param = 0,3`; o Alg 4.4 não tem limite
    GAE(λ)                          contra o nosso `δ = G − V` de `m` passos
    atualização em lote             ~8 passos de gradiente por 1000 do ambiente,
                                    contra 1 por transição — 125× a mais
    σ APRENDIDA                     `policy_output_dim = 2×dim_ação`, média e log σ;
                                    nós usamos σ fixa, varrida como hiperparâmetro

Isso explica o resultado que motivou esta grade: no orçamento de 10 M, **as seis
corridas do anbubus abortaram**, inclusive a que tinha 0/3 abortos a 200 mil passos.
Mas o ponto "seguro" dela saiu de 19,2 % para **77,6 % de cobertura e SR 55 %** antes
de divergir. A rede aprende; ela não sustenta. Uma σ aprendida é justamente a válvula
que falta: quando o gradiente cresce, o PPO pode AUMENTAR σ, o que encolhe
`∇log π = (z − f)/σ²`. Com σ fixa não há válvula.

O ESPAÇO, E POR QUE ESTES EIXOS

Os valores são os publicados, não inventados. Das oito configurações, as duas de ação
CONTÍNUA são a referência do nosso problema:

    tarefa    lr     γ     λ    lote  minilote  n   T    vf_clip
    hopper    4e-4   0,98  —    1000  512       8   32   3000
    reacher   7e-5   0,99  0,9  1000  128       10  300  (padrão 10)

O ESPAÇO É PAREADO COM O DA GRADE `desempate`, e o mapeamento está no bloco de
constantes: `T`, `n` e a razão do crítico transferem IDÊNTICOS; `lr` transfere só a
cardinalidade (Adam sobre a tabela não é somar `passo` às `N` posições); e `σ` NÃO
TRANSFERE, porque no PPO ela é aprendida.

    5 × 3 × 1 × 3 = 45 pontos × 3 sementes = 135 corridas

Dois eixos do desempate NÃO couberam, e por limites do modelo publicado:

    T ∈ {16, 32}   o `conv1d(stride=n)` do `NTNModel` exige `n | 25·T`, e nenhuma
                   combinação com `n ∈ {12, 24}` satisfaz. Viraram {24, 48, 300}.
    n = 24         a alocação é DENSA (`n_nodes · 2^n`): 6,7 a 13,4 GB só no ator,
                   numa máquina de 11,9. Medido: 18 corridas mortas por SIGKILL.

A nossa implementação é esparsa e varreu os dois sem esforço. É assimetria real
entre os braços, e a perda científica é pequena só porque `n = 24` já foi medido
como ruim lá (4,9 % contra 31,7 % de `n = 12`).

ORÇAMENTO: 200 000 passos, o MESMO da grade `desempate`. É menos que o
`timesteps_total = 1 000 000` das oito configurações publicadas, e a escolha é
deliberada: comparar 1 M contra 200 mil mediria orçamento, não método.

A assimetria que sobra tem de ir no relatório: no mesmo orçamento de passos de
ambiente o PPO faz ~8 passos de gradiente por 1000 e o Algoritmo 4.4 faz 1000 —
**125× menos atualizações**. Igualar passos de ambiente não iguala aprendizado, e
essa diferença é intrínseca à comparação, não escolha nossa.

A SELEÇÃO É PELA MESMA MÉTRICA DOS OUTROS BRAÇOS

`--sementes-aval` faz cada corrida terminar com avaliação **determinística** no banco
`val`, devolvendo cobertura, SR e CT. Sem isso a grade selecionaria por retorno de
treino — com ruído de exploração ligado — enquanto os braços do Algoritmo 4.4
selecionam por cobertura determinística, e a comparação mediria dois critérios de
seleção em vez de dois métodos.

DOIS AMBIENTES, E NÃO POR ESCOLHA

`ray[rllib]` fixa `gymnasium <= 1.2.2` e o projeto exige `>= 1.3.0`. Este driver roda
no ambiente do projeto e lança cada corrida como **subprocesso** num ambiente
efêmero. É a mesma divisão que a RNF-11 já pedia.

    uv run python scripts/spec06_grid_ppo.py --sementes 2 --passos 1000000 --processos 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

# --- o espaço ---------------------------------------------------------------

# --- o espaço PAREADO com a grade do Algoritmo 4.4 --------------------------
#
# A primeira grade PPO tinha 12 pontos contra os 120 da grade `desempate`, e isso
# enviesava a comparação: o PPO teve um décimo das chances de achar um bom canto, e
# o eixo que mais importou para ele (`T`) tinha o vencedor no EXTREMO da faixa — 300
# de dois níveis —, então não havia como saber se era ótimo ou apenas o maior
# testado.
#
# O MAPEAMENTO DOS EIXOS, e o que não transfere
#
#   Alg 4.4 (anbubus)              PPO                   transfere?
#   ----------------------------   -------------------   --------------------------
#   resolucao T   (16, 32)         resolucao             SIM, idêntico
#   tupla n       (12, 24)         tupla                 SIM, idêntico
#   razão crítico (0,25; 1; 4)     vf_loss_coeff         SIM, mesmos valores — nos
#                                                        dois o número diz quanto o
#                                                        crítico se move em relação
#                                                        ao ator
#   passo_efetivo (5 níveis)       lr (5 níveis)         só a ESTRUTURA: Adam sobre
#                                                        a tabela não é o mesmo que
#                                                        somar `passo` às N posições
#   sigma         (0,20; 0,35)     —                     NÃO EXISTE: σ é aprendida
#                                                        (`policy_output_dim` = 2×
#                                                        dim de ação)
#   passos_m      (8)              lambda_ = 0,9         análogo distante, fixo
#   normalizar    (False)          MeanStdFilter         difere por construção: o
#                                                        termômetro deles tem faixa
#                                                        fixa e EXIGE normalização
#
# Sem o eixo `σ`, o pareado é `5 × 2 × 2 × 3 = 60` pontos contra os 120 de lá — a
# metade, e a metade que falta é exatamente o eixo que o PPO não tem. É o mais
# próximo de "os mesmos hiperparâmetros" que a diferença de algoritmo permite.

LR = (7e-5, 1.5e-4, 3e-4, 7e-4, 1.5e-3)
"""Cinco níveis geométricos, mesma CARDINALIDADE do `passo_efetivo` de lá.

Começa em 7e-5, o valor publicado do reacher, e para em 1,5e-3: a primeira grade
mediu que **`3e-3` diverge nas seis configurações** com `nan` na distribuição de
ação, e `4e-4` também diverge quando `T = 300`. Estender além disso gastaria
sementes mapeando território morto que já está mapeado.

A faixa cobre 21×; a de lá cobria 10×. Não são comparáveis em valor absoluto —
`lr` é taxa de Adam sobre os pesos da `EmbeddingBag`, e `passo_efetivo` é o
deslocamento direto em `f`. O que se pareia é a resolução da varredura."""

RESOLUCAO = (24, 48, 300)
"""`24` e `48`, e NÃO os `16` e `32` do desempate — por um limite do código deles.

O `_keys` do `NTNModel` usa `F.conv1d(..., stride=tuple_size)`, que produz apenas
janelas completas, ou seja `floor(B/n)` chaves. Mas `n_nodes` é `ceil(B/n)`. Quando
`n` não divide `B`, os dois discordam e o `keys += self.mem_offsets` estoura:

    RuntimeError: The size of tensor a (33) must match the size of tensor b (34)

Com `obs_dim = 25`, `B = 25·T`, e a tabela de divisibilidade é:

        T      B    n=12    n=24
       16    400   33≠34   16≠17     <- impossível
       32    800   66≠67   33≠34     <- impossível
       24    600     OK      OK
       48   1200     OK      OK
      300   7500     OK    312≠313   <- só com n=12

Ou seja: **nenhuma combinação de `T ∈ {16, 32}` com `n ∈ {12, 24}` roda.** A primeira
grade PPO funcionou porque usava `n = 10`, e 10 divide 400, 800 e 7500.

Entre parear `T` e parear `n`, escolho `n`: foi o eixo dominante no desempate, onde
`n = 24` colapsou para 4,9 % de cobertura contra 31,7 % de `n = 12`. Os `T` viram os
vizinhos divisíveis mais próximos — `B = 600` e `1200`, contra `400` e `800` de lá.

É desvio declarado, e a alternativa seria pior: preencher a entrada até um múltiplo
modificaria o modelo publicado, que é justamente o que não se faz aqui."""

TUPLA = (12,)
"""Só `12` — e o `24` da grade `desempate` **não cabe na máquina**.

A alocação do `NTNModel` é DENSA: `num_embeddings = n_nodes · 2^n`. Com `n = 24` são
2²⁴ = 16 777 216 endereços por RAM, e só o ator (4 saídas, float32) pede:

    T = 24  ->  N = 25  ->   6,7 GB
    T = 48  ->  N = 50  ->  13,4 GB

numa máquina de 11,9 GB. Medido: **18 corridas mortas com código 137**, que é SIGKILL
do matador de memória do sistema.

ISSO É ASSIMETRIA REAL ENTRE OS BRAÇOS, e vai no relatório. A nossa implementação é
ESPARSA — dicionário por RAM, só os endereços visitados —, então a grade do Algoritmo
4.4 varreu `n = 24` sem esforço e mediu que ele COLAPSA: 4,9 % de cobertura contra
31,7 % de `n = 12`. O braço PPO não pode nem testar.

A perda científica é pequena justamente por isso: o nível que não cabe é o que já
sabemos ser ruim. A perda de PAREAMENTO é real e fica declarada."""

VF_LOSS_COEFF = (0.25, 1.0, 4.0)
"""Os MESMOS valores da razão `taxa_crítico/taxa_ator` do anbubus sob o Algoritmo
4.4. Em ambos, o número é quanto o crítico se move em relação ao ator: lá por taxas
distintas, aqui pelo peso da perda de valor no objetivo do PPO."""

FIXOS = dict(referencia="reacher", gamma=0.99, vf_clip=3000.0)
"""`λ = 0,9`, minilote 128 e lote 1000 vêm da base `reacher`; `γ = 0,99` é o de
Lamenza e coincide com ela.

`vf_clip = 3000` sai de medição, não da base: os nossos retornos sob a eq. 5-6 valem
~-59, e o `10` do reacher recortaria o alvo do crítico. A primeira grade confirmou o
sinal — `3000` rendeu +2,6 pontos de cobertura — então ele sai de eixo e vira fixo,
liberando cardinalidade para os eixos que pareiam."""


def pontos() -> list[dict]:
    """Produto cartesiano dos quatro eixos pareados."""
    fora, i = [], 0
    for lr, T, n, vf in itertools.product(LR, RESOLUCAO, TUPLA, VF_LOSS_COEFF):
        # A guarda que o modelo deles não tem: sem `n | B`, o `conv1d` dá `floor` e
        # o `n_nodes` dá `ceil`, e o `keys += mem_offsets` estoura por forma. Melhor
        # nunca gerar o ponto do que gastar 12 s de ray para descobrir.
        if (25 * T) % n:
            continue
        # SEGUNDA guarda: a alocação é densa (`n_nodes · 2^n`), e `n = 24` pede 6,7 a
        # 13,4 GB numa máquina de 11,9. Custou 18 corridas mortas por SIGKILL do
        # matador de memória antes de eu perceber. Melhor não gerar o ponto.
        if ((25 * T) // n) * (2 ** n) * 4 * 4 / 1e9 > 4.0:
            continue
        fora.append({"id": i, "lr": lr, "resolucao": T, "tupla": n,
                     "vf_loss_coeff": vf})
        i += 1
    return fora


def rotulo(p: dict, seed: int) -> str:
    """Identidade de uma corrida — UMA função, usada pelo despacho E pela retomada.

    Existe porque as duas montavam a string separadamente, e elas desviaram: quando
    `vf_clip` saiu do ponto e `vf_loss_coeff` entrou, o despacho acompanhou e a
    retomada não, e o `--retomar` estourou com `KeyError: 'vf_clip'`.

    Duplicar identidade é o tipo de erro que só aparece quando alguém retoma — ou
    seja, no pior momento.
    """
    return (f"lr{p['lr']:.1e}-T{p['resolucao']}-n{p['tupla']}"
            f"-vfc{p['vf_loss_coeff']:g}-s{seed}")


def _comando(p: dict, seed: int, a, bruto: Path, saida: Path) -> list[str]:
    """A invocação do ambiente efêmero. `--isolated` para não herdar o do projeto."""
    return ["uv", "run", "--isolated", "--with", "ray[rllib]", "--with", "torch",
            "--with", "pettingzoo", "python", "-u",
            str(RAIZ / "scripts" / "spec06_ppo_anbubus.py"),
            "--celula", f"{a.na}:{a.ns}",
            "--passos", str(a.passos),
            "--seed", str(seed),
            "--referencia", FIXOS["referencia"],
            "--lr", repr(p["lr"]),
            "--gamma", repr(FIXOS["gamma"]),
            "--tupla", str(p["tupla"]),
            "--resolucao", str(p["resolucao"]),
            "--vf-clip", repr(FIXOS["vf_clip"]),
            "--vf-loss-coeff", repr(p["vf_loss_coeff"]),
            "--ray-cpus", str(a.ray_cpus),
            # figuras de mérito 10 vezes por corrida, no log do ponto
            "--aval-cada", str(max(1, a.passos // 10)),
            "--n-aval-parcial", "8",
            "--sementes-aval", str(a.sementes_aval),
            "--bruto", str(bruto),
            "--saida", str(saida)]


def uma(tarefa: tuple) -> dict:
    """Roda um ponto e devolve a linha do registro. Nunca levanta.

    Um subprocesso que falha é **dado**, não interrupção: a grade tem de registrar
    qual configuração não roda e seguir. Levantar aqui perderia as outras corridas do
    lote por causa de uma.
    """
    p, seed, a = tarefa
    rot = rotulo(p, seed)
    dest = Path(a.raiz) / rot
    dest.mkdir(parents=True, exist_ok=True)
    bruto, saida = dest / "bruto.npz", dest / "artefato.json"
    # A saída de cada ponto vai para um LOG PRÓPRIO, em vez de `capture_output`.
    #
    # Com `capture_output=True` o driver engolia tudo até o subprocesso terminar, e
    # numa corrida de 200 mil passos isso é uma hora de silêncio — sem como saber se
    # o ponto está andando, travado ou apenas lento. O `spec06_ppo_anbubus.py`
    # imprime por lote (passos, %, ETA, figuras de mérito) e essa informação estava
    # sendo jogada fora. Foi o que me deixou 56 min sem saber que a projeção de 10
    # min por corrida estava errada por 6×.
    #
    # `buffering=1` no arquivo e `-u` no subprocesso para a linha aparecer na hora.
    # LIMITA AS THREADS DO TORCH, e é isso que faz a grade caber na máquina.
    #
    # `--ray-cpus` limita o que o **ray** reserva, não o que o **torch** usa. Medido:
    # um ponto isolado consome **585 % de CPU** — quase 6 núcleos — porque o torch
    # abre threads intra-operação por conta própria. Com 4 pontos simultâneos isso
    # são 24 threads em 12 núcleos, e o resultado não é contenção linear: é debandada
    # de troca de contexto.
    #
    # Os números que mostram o tamanho do problema: isolado, um ponto faz 200 mil
    # passos em ~5,6 min. Quatro simultâneos sem o limite não fecharam **nenhum em
    # 56 min** — 10× pior do que o 2× que a simples divisão de CPU explicaria.
    ambiente = dict(os.environ,
                    OMP_NUM_THREADS=str(a.ray_cpus),
                    OPENBLAS_NUM_THREADS=str(a.ray_cpus),
                    MKL_NUM_THREADS=str(a.ray_cpus),
                    TORCH_NUM_THREADS=str(a.ray_cpus))
    t0 = time.perf_counter()
    log = dest / "saida.log"
    with open(log, "w", buffering=1) as fh:
        proc = subprocess.run(_comando(p, seed, a, bruto, saida),
                              cwd=str(RAIZ), stdout=fh, env=ambiente,
                              stderr=subprocess.STDOUT, text=True)
    linha = {**{k: v for k, v in p.items()}, "seed": seed, "rotulo": rot,
             "passos": a.passos, "parede_s": time.perf_counter() - t0,
             "dir": str(dest)}
    if proc.returncode != 0 or not saida.exists():
        # As últimas linhas do stderr, não todas: o RLlib despeja centenas de avisos
        # de depreciação e o que interessa é a exceção final.
        cauda = "\n".join(log.read_text().strip().splitlines()[-6:])
        return {**linha, "falhou": f"código {proc.returncode}", "stderr": cauda,
                "cobertura": 0.0, "sr": 0.0}
    art = json.loads(saida.read_text())
    aval = art.get("avaliacao") or {}
    hist = art.get("historico") or []
    return {**linha, "falhou": None,
            "cobertura": aval.get("cobertura", 0.0), "sr": aval.get("sr", 0.0),
            "ct_cens": aval.get("ct_cens"), "ct_succ": aval.get("ct_succ"),
            "retorno_final": hist[-1]["retorno"] if hist else None,
            "hiperparametros": art.get("hiperparametros")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--na", type=int, default=1)
    ap.add_argument("--ns", type=int, default=12)
    ap.add_argument("--sementes", type=int, default=2)
    ap.add_argument("--passos", type=int, default=1_000_000,
                    help="`timesteps_total` das oito configurações publicadas")
    ap.add_argument("--n-val", type=int, default=20,
                    help="episódios do banco val na avaliação determinística")
    ap.add_argument("--processos", type=int, default=4,
                    help="corridas simultâneas; cada uma reserva --ray-cpus")
    ap.add_argument("--ray-cpus", type=int, default=2)
    ap.add_argument("--raiz", default="runs/spec06-ppo")
    ap.add_argument("--saida", default="reports/spec06-ppo.json")
    ap.add_argument("--retomar", action="store_true")
    a = ap.parse_args()

    # As sementes do banco `val` num arquivo, porque o ambiente efêmero não importa
    # `uavwnn`. É o MESMO banco que os braços do Algoritmo 4.4 usam para selecionar.
    from uavwnn.campanha import carregar_bancos
    bancos = carregar_bancos(a.na, a.ns)
    a.sementes_aval = Path(a.raiz) / "sementes-val.txt"
    a.sementes_aval.parent.mkdir(parents=True, exist_ok=True)
    a.sementes_aval.write_text(
        "\n".join(str(int(s)) for s in bancos.val.sementes[:a.n_val]) + "\n")

    tarefas = [(p, sd, a) for p in pontos() for sd in range(a.sementes)]
    progresso = Path(a.saida).with_suffix(".progresso.jsonl")
    progresso.parent.mkdir(parents=True, exist_ok=True)

    res = []
    if a.retomar and progresso.exists():
        vistos = set()
        for l in progresso.read_text().splitlines():
            if not l.strip():
                continue
            try:
                x = json.loads(l)
            except json.JSONDecodeError:
                continue          # linha cortada por um Ctrl-C no meio da escrita
            if x.get("passos") == a.passos:
                res.append(x)
                vistos.add(x["rotulo"])
        antes = len(tarefas)
        tarefas = [t for t in tarefas
                   if rotulo(t[0], t[1]) not in vistos]
        print(f"RETOMANDO: {len(res)} de {antes} aproveitadas · {len(tarefas)} a fazer")
    else:
        progresso.write_text("")

    print(f"grade PPO · rede do anbubus com o algoritmo dela · Na={a.na} Ns={a.ns}")
    print(f"{len(pontos())} pontos × {a.sementes} sementes = "
          f"{len(pontos()) * a.sementes} corridas · {a.passos:,} passos")
    print(f"eixos: lr{LR} × T{RESOLUCAO} × n{TUPLA} × vf_loss_coeff{VF_LOSS_COEFF}")
    print(f"pareado com a grade `desempate` em n e no eixo de crítico; `T` usa os "
          f"vizinhos divisíveis (o conv1d do NTNModel exige n | 25·T)")
    print(f"fixos: {FIXOS} · λ=0,9 · minilote 128 · lote 1000 · MeanStdFilter")
    print(f"{a.processos} simultâneas × {a.ray_cpus} CPUs cada\n")
    if not tarefas:
        print("nada a fazer.")

    t0 = time.perf_counter()
    # Threads e não processos: o trabalho está nos SUBPROCESSOS, então o GIL não
    # atrapalha — cada thread só espera um `subprocess.run`.
    with ThreadPoolExecutor(max_workers=a.processos) as pool:
        for i, x in enumerate(pool.map(uma, tarefas), 1):
            res.append(x)
            with open(progresso, "a") as f:
                f.write(json.dumps(x) + "\n")
            estado = (f"FALHOU {x['falhou']}" if x["falhou"]
                      else f"cobertura {x['cobertura']:.1%} SR {x['sr']:.1%}")
            # ETA por ONDAS, não por corrida: os pontos rodam `--processos` a cada
            # vez, então o que falta é `ceil(restantes / processos)` ondas de
            # duração igual à média de um ponto. Dividir o total pela vazão daria
            # número menor e errado, porque a última onda pode estar incompleta.
            #
            # Usa a média das últimas 8 conclusões, não de todas: se a máquina
            # ganhar ou perder carga, a estimativa acompanha. Foi a lição das ETAs
            # que erramos na campanha do Algoritmo 4.4 usando média cumulativa.
            recentes = [y["parede_s"] for y in res[-8:] if y.get("parede_s")]
            eta = ""
            if recentes:
                medio = sum(recentes) / len(recentes)
                ondas = -(-(len(tarefas) - i) // a.processos)
                falta = ondas * medio
                fim = time.strftime("%H:%M", time.localtime(time.time() + falta))
                eta = (f" │ {medio / 60:.1f} min/ponto · {ondas} ondas · "
                       f"resta {int(falta) // 3600}h{int(falta) % 3600 // 60:02d}m "
                       f"(~{fim})")
            print(f"  [{i:>3}/{len(tarefas)}] {x['rotulo']:<28} {estado} "
                  f"· {x['parede_s'] / 60:.1f} min{eta}", flush=True)

    parede = time.perf_counter() - t0
    Path(a.saida).write_text(json.dumps(
        {"espaco": {"lr": LR, "resolucao": RESOLUCAO, "tupla": TUPLA,
                    "vf_loss_coeff": VF_LOSS_COEFF, "fixos": FIXOS},
         "config": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(a).items()},
         "parede_s": parede, "corridas": res}, indent=2, ensure_ascii=False))

    vivas = [x for x in res if not x["falhou"]]
    print(f"\nparede: {parede / 3600:.1f} h · {len(vivas)}/{len(res)} sem falha")
    if vivas:
        melhor = max(vivas, key=lambda x: x["cobertura"])
        print(f"melhor: {melhor['rotulo']} -> cobertura {melhor['cobertura']:.1%} "
              f"SR {melhor['sr']:.1%}")
        print("referências: aleatório 3,3 % de cobertura · guloso SR 100 %, CT ≈ 34 s")
        print("Alg 4.4: dissertação 50,1 % cob / SR 21,3 % · anbubus 46,6 % / 10,3 %")
    print(f"artefato: {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
