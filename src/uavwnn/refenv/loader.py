"""Localiza e carrega os módulos de referência do submodule.

Carregamento por caminho de arquivo, e não por `sys.path`, para que o
`environment.py` de referência não passe a poluir o espaço de importação do
projeto — e para deixar explícito, em quem lê, que aquele código é externo.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

REFERENCE_COMMIT = "be94ca7fbaa586a2ca03fc43e2de0315c7f0de43"
"""Tag `dissertation` do repositório Project-GrADyS/gradys-data-collection.

Fixado. `dissertation` é uma TAG, não um branch — o branch default (`main`) é
um estado posterior. Nos arquivos que usamos os dois coincidem hoje, mas
depender disso seria depender de um acidente.
"""

SPAWN_HALF_WIDTH = 2.0
"""Semi-lado da área de largada dos UAVs, em metros.

Fato do ambiente de referência, não parâmetro nosso: o `reset()` sorteia as
posições iniciais com `random.uniform(-2, 2)` em x e y — um QUADRADO de lado
4 m centrado na origem, não um disco. Exposto aqui só para instrumentação e
gráficos; nada no ambiente lê esta constante.
"""

SCENARIO = "scenario/003-collaboration"
"""O cenário que corresponde ao §7.2 da dissertação."""

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "gradys-data-collection"


class ReferenceNotAvailable(RuntimeError):
    """O submodule não foi inicializado."""


def reference_root() -> Path:
    """Raiz do repositório de referência.

    Pode ser sobrescrita por `UAVWNN_REFERENCE_ROOT`, útil em CI ou quando o
    submodule vive fora da árvore.
    """
    env = os.environ.get("UAVWNN_REFERENCE_ROOT")
    root = Path(env) if env else _DEFAULT_ROOT
    if not (root / SCENARIO / "environment.py").exists():
        raise ReferenceNotAvailable(
            f"código de referência não encontrado em {root}.\n"
            "Rode: git submodule update --init --recursive"
        )
    return root


def _load(name: str, path: Path) -> ModuleType:
    key = f"_uavwnn_ref_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ReferenceNotAvailable(f"não foi possível carregar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def load_environment() -> ModuleType:
    """Carrega `environment.py` do cenário 003-collaboration, sem alteração."""
    return _load("environment", reference_root() / SCENARIO / "environment.py")


def load_heuristics() -> ModuleType:
    """Carrega `heuristics.py`: guloso, `smart` e aleatório, sem alteração."""
    return _load("heuristics", reference_root() / SCENARIO / "heuristics.py")
