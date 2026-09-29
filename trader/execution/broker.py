"""Contrato da corretora (Binance, MT5, Nelogica, simulação).

O resto do sistema não sabe em qual corretora está operando.
"""
from __future__ import annotations

from typing import Protocol

from trader.domain import Fill, Order


class Broker(Protocol):
    def submit(self, order: Order) -> Fill: ...

    def position(self, symbol: str) -> float: ...

    def equity(self) -> float: ...
