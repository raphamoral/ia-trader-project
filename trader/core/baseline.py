"""Motor de referência: cruzamento de médias móveis.

Não é para ganhar dinheiro; é a régua que qualquer agente evoluído precisa
superar, e o motor usado para testar o resto do fluxo.
"""
from __future__ import annotations

from dataclasses import dataclass

from trader.domain import MarketView, Side, Signal


@dataclass
class MovingAverageCross:
    fast: int = 20
    slow: int = 50
    name: str = "ma_cross"

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("exige 0 < fast < slow")

    def decide(self, view: MarketView) -> Signal:
        close = view.candles["close"]
        if len(close) < self.slow:
            return Signal(Side.FLAT, reason="histórico insuficiente")
        fast = close.iloc[-self.fast:].mean()
        slow = close.iloc[-self.slow:].mean()
        if fast > slow:
            return Signal(Side.BUY, confidence=0.5, size=1.0, reason=f"MA{self.fast} > MA{self.slow}")
        return Signal(Side.FLAT, reason=f"MA{self.fast} <= MA{self.slow}")
