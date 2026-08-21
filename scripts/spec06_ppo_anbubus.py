"""Treina a rede n-tuple com o **PPO do RLlib**, usando o `NTNModel` publicado.

Nada de reimplementação: usa `vendor/ntn_neurocomputing_v0` (submodule do
repositório `anbubus/ntn_neurocomputing_v0`) como está, e o PPO do Ray RLlib —
que é o que o artigo diz ter usado:

> *"In this work, the policy gradient method used was Proximal Policy Optimization
> (PPO). … The implementation used was the one from RLlib."*
> — Katopodis, Lima e França, *Neurocomputing* 500 (2022)

É o **quarto braço** da comparação: MADDPG (baseline do problema), MLP, n-tuple
treinada pelo Algoritmo 4.4, e n-tuple treinada por PPO.

    uv run --group ppo python scripts/spec06_ppo_anbubus.py --celula 2:12 --passos 200000

O QUE NÃO RODA VERBATIM, E FOI MEDIDO

As configurações do repositório são de 2022 e **não executam** no Ray atual. Não é
defeito delas — é deriva de API —, mas quem for reproduzir precisa saber:

    tune.run("PPO", …)          depreciado; usa-se `PPOConfig`
    sgd_minibatch_size          renomeado para `minibatch_size`
    num_sgd_iter                renomeado para `num_epochs`
    ray.init(local_mode=True)   removido do Ray
    (nova pilha de API)         é preciso DESATIVÁ-LA explicitamente com
                                `api_stack(enable_rl_module_and_learner=False,
                                           enable_env_runner_and_connector_v2=False)`,
                                porque o `ModelCatalog`/`TorchModelV2` que o
                                `NTNModel` usa só existe na pilha antiga

Mais uma inconsistência interna do repositório: os imports usam **três** nomes de
pacote — `ntn_neurocomputing_v0`, `ntn_neurocomputing` e `ntn_neurcomputing` (com
typo). Só o primeiro está no caminho do modelo, e é por isso que o submodule tem de
se chamar exatamente assim.

DOIS AMBIENTES, E NÃO POR ESCOLHA

`ray[rllib]` fixa `gymnasium==1.2.2` no máximo; este projeto exige `>=1.3.0`. São
**incompatíveis no mesmo ambiente** — medido, não suposto. Então este script roda
num ambiente efêmero e **não importa `uavwnn`**:

    uv run --isolated --with "ray[rllib]" --with torch --with pettingzoo \
        python scripts/spec06_ppo_anbubus.py --celula 2:12 --passos 200000 \
        --bruto reports/ckpt/ppo-a2s12-bruto.npz

Ele carrega o ambiente de referência **por caminho**, do mesmo `vendor/` que o
`refenv` usa, e grava a tabela treinada em arrays crus. A conversão para o nosso
formato é o passo seguinte, no ambiente do projeto:

    uv run python scripts/spec06_importar_ppo.py \
        reports/ckpt/ppo-a2s12-bruto.npz reports/ckpt/ppo-a2s12

O incômodo tem um lado bom: é exatamente a divisão que a **RNF-11** já pedia. Ela
existe porque latência de rede pequena via PyTorch mede despacho, não modelo.

    treino      RLlib PPO + NTNModel (torch + ray, efêmero)
    medição     tabela convertida + NumPy puro (ambiente do projeto)

A conversão só é defensável porque `tests/test_anbubus.py` prova que as duas
implementações são numericamente a mesma — endereçamento **exato** e saída a
2·10⁻⁵. Sem esse teste, exportar seria um salto de fé.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "vendor"))

# Defaults do `benchmark.py` que gerou o §7.2 — os MESMOS de `refenv.EnvConfig`,
# replicados aqui porque este script não pode importar `uavwnn` (ver o cabeçalho).
# Se um deles divergir do `refenv`, o treino e a avaliação deixam de ser o mesmo
# problema, e um teste guarda essa igualdade (`tests/test_ppo_anbubus.py`).
ENV_TREINO = dict(
    algorithm_iteration_interval=0.5, scenario_size=100.0,
    max_episode_length=500.0, max_seconds_stalled=30,
    communication_range=20.0, state_num_closest_sensors=12,
    state_mode="relative", id_on_state=True,
    min_sensor_priority=1.0, max_sensor_priority=1.0,
    full_random_drone_position=False, reward="punish", speed_action=True,
    # `phase="train"`: o episódio NÃO termina ao coletar tudo. É o
    # `--no-end-when-all-collected` do benchmark.py, e é o que dá o "longer
    # duration positive reward signal" da §5 de Lamenza.
    end_when_all_collected=False,
)


def _carregar_ambiente():
    """Carrega `environment.py` do cenário 003 por caminho, sem tocar nele."""
    import importlib.util
    alvo = (RAIZ / "vendor" / "gradys-data-collection" / "scenario"
            / "003-collaboration" / "environment.py")
    if not alvo.exists():
        sys.exit(f"ambiente de referência ausente: {alvo}\n"
                 "rode: git submodule update --init vendor/gradys-data-collection")
    spec = importlib.util.spec_from_file_location("environment", alvo)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["environment"] = mod
    spec.loader.exec_module(mod)
    return mod


def _analisar_celula(txt: str) -> tuple[int, int]:
    na, ns = txt.split(":")
    return int(na), int(ns)

# Hiperparâmetros do repositório deles (`wnn/*/individual_run/experiment.py`),
# traduzidos para os nomes atuais do RLlib. Os valores são os deles.
ANBUBUS = dict(lr=4e-4, gamma=0.98, train_batch_size=1000, num_epochs=1,
               minibatch_size=512)
TUPLA_ANBUBUS = 8
RESOLUCAO_ANBUBUS = 32
FAIXA_OBS = (-1.0, 1.5)
"""Faixa REAL da observação, medida — e ela **não** é a declarada pelo ambiente.

O `GrADySEnvironment` declara `Box(-1.0, 1.0)` e emite valores até **1,50**. Não é
ruído numérico: a normalização é `(d + S)/(2S)` com `S = 100`, o que dá 1,5 quando
`d = 200` — isto é, os drones **saem** da caixa de 100 × 100 e a normalização supõe
que não saem. Medido em 400 passos, `Na ∈ {2, 8}`: faixa `[-1,0000; 1,5000]`.

Consequências, e nenhuma delas é opcional:

1. O `Thermometer(min=-1, max=1)` das configurações do `anbubus` **saturaria** tudo
   acima de 1,0, perdendo a distinção entre "longe" e "muito longe" exatamente na
   região em que ela importa para reaproximar o enxame.
2. O RLlib **valida** o espaço declarado e aborta a coleta. Nós **não corrigimos o
   ambiente** (é a regra do projeto: clonar o cenário 3 sem modificação), então o
   que se desliga é a validação — os dados seguem intactos.
3. O nosso `TermometroQuantil` (E2) se ajusta a um corpus e portanto absorve isto
   sozinho. O E0 de faixa fixa não absorveria, e é um argumento a mais para o E2
   que a SPEC-01 não tinha.
"""
"""`tuple_size = 8` e `resolution = 32` são os das configurações deles. O 8 não é
escolha estética: a alocação é **densa** (`n_nodes · 2^n` posições), então `n = 32`
seria 4,3·10⁹ por RAM."""


def _exigir_dependencias():
    try:
        import ray  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        sys.exit('ray e torch são necessários, e NÃO há grupo no pyproject '
                 '(conflito de gymnasium). Rode:\n\n'
                 '  uv run --isolated --with "ray[rllib]" --with torch '
                 '--with pettingzoo \\\n'
                 '      python scripts/spec06_ppo_anbubus.py ...\n')
    if not (RAIZ / "vendor" / "ntn_neurocomputing_v0" / "wnn" / "ntn_model.py").exists():
        sys.exit("submodule ausente: git submodule update --init "
                 "vendor/ntn_neurocomputing_v0")


def construir(celula: tuple[int, int], seed: int, passos_lote: int | None = None):
    """Monta o PPO do RLlib sobre o `NTNModel`, no ambiente de Lamenza."""
    import ray
    from ntn_neurocomputing_v0.wnn.ntn_model import NTNModel
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    na, ns = celula
    modulo = _carregar_ambiente()
    kw = dict(ENV_TREINO, num_drones=na, num_sensors=ns,
              state_num_closest_drones=na - 1)

    # O ambiente de referência é `ParallelEnv` do PettingZoo, então o wrapper do
    # RLlib serve sem adaptação. **Nada no ambiente é tocado** — é a mesma regra
    # da SPEC-00: clonar o cenário 3 sem modificação, para a comparação com o
    # baseline continuar válida.
    register_env("lamenza003", lambda _: ParallelPettingZooEnv(
        modulo.GrADySEnvironment(render_mode=None, **kw)))
    ModelCatalog.register_custom_model("ntn_model", NTNModel)

    treino = dict(ANBUBUS)
    if passos_lote:
        treino["train_batch_size"] = passos_lote
        treino["minibatch_size"] = min(treino["minibatch_size"], passos_lote)

    modelo = {
        "custom_model": "ntn_model",
        "custom_model_config": {
            "seed": seed,
            "tuple_size": TUPLA_ANBUBUS,
            # Faixa MEDIDA, não a declarada. Ver `FAIXA_OBS`. O -1 é o padding
            # da eq. 5-1, e o 1,5 é o drone que saiu da caixa do cenário.
            "encoding": {"enc_type": "thermometer",
                         "resolution": RESOLUCAO_ANBUBUS,
                         "min": FAIXA_OBS[0], "max": FAIXA_OBS[1]},
        },
    }
    conf = (PPOConfig()
            .environment("lamenza003")
            .framework("torch")
            # A pilha nova não tem ModelCatalog/TorchModelV2, que é o que o
            # NTNModel usa. Desativar é obrigatório, não preferência.
            .api_stack(enable_rl_module_and_learner=False,
                       enable_env_runner_and_connector_v2=False)
            .env_runners(num_env_runners=0)
            # O ambiente emite fora do espaço que ele declara (ver `FAIXA_OBS`), e
            # o pré-processador do RLlib aborta por isso. Desligá-lo é a escolha
            # certa: **não tocamos no ambiente**, e o modelo já binariza com a
            # faixa medida. Recortar a observação seria modificar o problema.
            .experimental(_disable_preprocessor_api=True)
            .debugging(seed=seed)
            .training(model=modelo, **treino))
    algo = conf.build_algo() if hasattr(conf, "build_algo") else conf.build()
    return algo, kw


def exportar_bruto(algo, destino: Path, celula: tuple[int, int]) -> dict:
    """Grava a tabela treinada em **arrays crus**, para conversão posterior.

    Cru e não no nosso formato porque este processo não tem `uavwnn` importável
    (ver o cabeçalho). O que sai é o suficiente e nada além: o mapeamento, os
    pesos das duas `EmbeddingBag`, e os números que definem o endereçamento.

    O **mapeamento vai gravado**, e não a semente: o `torch.randperm` deles não
    coincide com o `permutation` do NumPy, então reconstruí-lo daria uma rede cujos
    endereços não correspondem aos valores — erro silencioso, que apareceria como
    "a política não aprendeu".
    """
    modelo = algo.get_policy().model
    destino.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "mapeamento": modelo.mapping.detach().cpu().numpy().astype(np.int64),
        "peso_ator": modelo.policy_mem.weight.detach().cpu().numpy(),
        "peso_critico": modelo.vf_mem.weight.detach().cpu().numpy(),
        "tamanho_tupla": np.array(TUPLA_ANBUBUS),
        "resolucao": np.array(RESOLUCAO_ANBUBUS),
        "n_bits": np.array(int(modelo.input_dim)),
        "n_nodes": np.array(int(modelo.n_nodes)),
        "celula": np.array(celula, np.int64),
        "faixa": np.array(FAIXA_OBS),
    }
    np.savez_compressed(destino, **dados)
    ocup = int((np.abs(dados["peso_ator"]).sum(axis=1) > 0).sum())
    return {"arquivo": str(destino), "n_bits": int(modelo.input_dim),
            "n_ram": int(modelo.n_nodes), "d_ator": int(dados["peso_ator"].shape[1]),
            "ocupadas_ator": ocup, "alocadas": int(dados["peso_ator"].shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--celula", default="2:12", help="ex. '4:24'")
    ap.add_argument("--passos", type=int, default=100_000,
                    help="timesteps de treino. O `anbubus` usa 1 000 000")
    ap.add_argument("--lote", type=int, default=None,
                    help="sobrescreve train_batch_size (útil para fumaça)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bruto", default=None,
                    help=".npz com a tabela treinada, para "
                         "scripts/spec06_importar_ppo.py converter")
    ap.add_argument("--saida", default="reports/spec06-ppo-anbubus.json")
    a = ap.parse_args()

    _exigir_dependencias()
    warnings.filterwarnings("ignore")
    import ray

    celula = _analisar_celula(a.celula)
    ray.init(num_cpus=4, log_to_driver=False, include_dashboard=False)
    algo, kw_env = construir(celula, a.seed, a.lote)
    print(f"PPO (RLlib) + NTNModel · Na={celula[0]} Ns={celula[1]} · "
          f"{a.passos} passos")
    print(f"hiperparâmetros do anbubus: {ANBUBUS}")

    hist, vistos = [], 0
    while vistos < a.passos:
        r = algo.train()
        m = r.get("env_runners", r)
        vistos = r.get("num_env_steps_sampled_lifetime",
                       r.get("timesteps_total", vistos + ANBUBUS["train_batch_size"]))
        ret = m.get("episode_return_mean", m.get("episode_reward_mean"))
        hist.append({"passos": int(vistos),
                     "retorno": None if ret is None else float(ret)})
        print(f"  {vistos:>8} passos  retorno {ret}")

    artefato = {"celula": list(celula), "config": vars(a), "env": kw_env,
                "hiperparametros": ANBUBUS, "historico": hist}
    if a.bruto:
        artefato["bruto"] = exportar_bruto(algo, Path(a.bruto), celula)
        b = artefato["bruto"]
        print(f"\ntabela crua: {b['arquivo']}")
        print(f"  ocupadas/alocadas no ator: {b['ocupadas_ator']}/{b['alocadas']}"
              f"  ({b['n_ram']} RAMs, {b['n_bits']} bits)")
        print("  converta com: uv run python scripts/spec06_importar_ppo.py "
              f"{b['arquivo']} <destino>")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(artefato, indent=2, ensure_ascii=False))
    print(f"\nartefato: {a.saida}")
    algo.stop()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
