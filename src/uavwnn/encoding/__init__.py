"""Codificação binária da observação (SPEC-01)."""

from .artigo import CircularArtigo, TermometroArtigo
from .corpus import Corpus, coletar_corpus
from .thermometer import (
    Codificador,
    TermometroGlobal,
    TermometroPorSlot,
    TermometroQuantil,
    PADDING,
)

__all__ = [
    "CircularArtigo",
    "Codificador",
    "Corpus",
    "PADDING",
    "TermometroArtigo",
    "TermometroGlobal",
    "TermometroPorSlot",
    "TermometroQuantil",
    "coletar_corpus",
]
