"""Filtro de risco entre o sinal do motor e a ordem.

Fica fora da evolução: nenhum agente altera estes limites nem desliga o
kill switch. Os limites são congelados na criação.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.domain import Order, Side, Signal

_MIN_QTY = 1e-9


@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float = 0.2  # do patrimônio, por ativo
    daily_loss_limit: float = 0.03  # fração do patrimônio no início do dia (UTC)
    rebalance_band: float = 0.25  # já posicionado: só ajusta se o desvio passar desta fração do alvo

    def __post_init__(self) -> None:
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction deve estar em (0, 1]")
        if not 0 < self.daily_loss_limit < 1:
            raise ValueError("daily_loss_limit deve estar em (0, 1)")


class RiskGate:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self._killed = False
        self._day: pd.Timestamp | None = None
        self._day_start_equity = 0.0
        self._halted_today = False

    @property
    def killed(self) -> bool:
        return self._killed

    def kill(self) -> None:
        """Parada de emergência manual: zera posições e bloqueia até reiniciar."""
        self._killed = True

    def order_for(
        self, signal: Signal, symbol: str, price: float, equity: float, position: float, now: pd.Timestamp
    ) -> Order | None:
        """Traduz o sinal numa ordem que respeita os limites, ou None."""
        self._roll_day(now, equity)
        if equity <= self._day_start_equity * (1 - self.limits.daily_loss_limit):
            self._halted_today = True

        if self._killed or self._halted_today or signal.side is not Side.BUY:
            target = 0.0  # spot: vender ou ficar de fora significa zerar
        else:
            target = equity * self.limits.max_position_fraction * signal.size / price

        delta = target - position
        if abs(delta) < _MIN_QTY:
            return None
        if target > 0 and position > 0 and abs(delta) < target * self.limits.rebalance_band:
            return None  # variação de preço não é motivo para pagar taxa
        side = Side.BUY if delta > 0 else Side.SELL
        return Order(symbol=symbol, side=side, quantity=abs(delta))

    def _roll_day(self, now: pd.Timestamp, equity: float) -> None:
        day = now.normalize()
        if day != self._day:
            self._day = day
            self._day_start_equity = equity
            self._halted_today = False
