"""Contrato do núcleo (Motor).

Tudo fora de `core` conversa com o motor só por aqui. Trocar LightGBM por um
transformer ou por uma LLM ajustada é implementar este protocolo de novo.
"""
from __future__ import annotations

from typing import Protocol

from trader.domain import MarketView, Signal


class Engine(Protocol):
    name: str

    def decide(self, view: MarketView) -> Signal: ...
