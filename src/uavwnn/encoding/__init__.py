"""Codificação binária da observação (SPEC-01)."""

from .corpus import Corpus, coletar_corpus
from .thermometer import (
    Codificador,
    TermometroGlobal,
    TermometroPorSlot,
    TermometroQuantil,
    PADDING,
)

__all__ = [
    "Codificador",
    "Corpus",
    "PADDING",
    "TermometroGlobal",
    "TermometroPorSlot",
    "TermometroQuantil",
    "coletar_corpus",
]
