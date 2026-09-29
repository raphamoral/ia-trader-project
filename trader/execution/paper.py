"""Corretora simulada (spot, sem venda a descoberto), com taxa por lado."""
from __future__ import annotations

import pandas as pd

from trader.domain import Fill, Order, Side

BINANCE_SPOT_FEE = 0.001  # 0,1% por lado


class InsufficientFunds(Exception):
    pass


class PaperBroker:
    def __init__(self, cash: float, fee_rate: float = BINANCE_SPOT_FEE) -> None:
        self.cash = cash
        self.fee_rate = fee_rate
        self._positions: dict[str, float] = {}
        self._marks: dict[str, tuple[float, pd.Timestamp]] = {}
        self.fills: list[Fill] = []

    def mark(self, symbol: str, price: float, ts: pd.Timestamp) -> None:
        """Atualiza o preço de referência usado para executar e avaliar."""
        self._marks[symbol] = (price, ts)

    def position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def equity(self) -> float:
        held = sum(qty * self._marks[sym][0] for sym, qty in self._positions.items() if qty)
        return self.cash + held

    def submit(self, order: Order) -> Fill:
        if order.symbol not in self._marks:
            raise ValueError(f"sem preço para {order.symbol}; chame mark() antes")
        price, ts = self._marks[order.symbol]
        notional = order.quantity * price
        fee = notional * self.fee_rate
        held = self.position(order.symbol)

        if order.side is Side.BUY:
            if notional + fee > self.cash:
                raise InsufficientFunds(f"precisa {notional + fee:.2f}, tem {self.cash:.2f}")
            self.cash -= notional + fee
            self._positions[order.symbol] = held + order.quantity
        else:
            if order.quantity > held + 1e-12:
                raise InsufficientFunds(f"vendendo {order.quantity}, possui {held}")
            self.cash += notional - fee
            self._positions[order.symbol] = max(held - order.quantity, 0.0)

        fill = Fill(order=order, price=price, fee=fee, ts=ts)
        self.fills.append(fill)
        return fill
