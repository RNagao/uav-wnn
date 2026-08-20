"""Torna `scripts/` importável pelos testes.

Os scripts de estudo não são um pacote instalado — são executáveis na raiz do
repositório. `test_diferencial.py` precisa das mesmas configurações e do mesmo
construtor de dados que `scripts/spec04_diferencial.py` usou para gerar as
fixtures; duplicá-los aqui abriria a porta para os dois lados divergirem em
silêncio, que é justamente o que o teste existe para detectar.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
