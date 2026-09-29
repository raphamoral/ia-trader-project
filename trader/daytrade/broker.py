"""Corretora simulada de futuros lineares (USDT-M), margem isolada.

Uma posição por símbolo, modo one-way. Comprado = side +1, vendido = -1.
Na liquidação a margem isolada inteira é perdida (o resto vai para o fundo
de seguro da corretora), sem cobrança adicional.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BINANCE_USDM_TAKER = 0.0005  # 0,05% por lado, VIP 0
BINANCE_USDM_MAKER = 0.0002  # 0,02% para ordens limit que ficam no livro (conferido via ccxt em 2026-09-29)
BINANCE_BTC_MMR = 0.004  # margem de manutenção da 1ª faixa do BTCUSDT (premissa; confira as faixas)


class InsufficientMargin(Exception):
    pass


def liquidation_price(side: int, entry: float, leverage: float, mmr: float) -> float:
    """Preço em que margem + PnL não realizado = margem de manutenção.

    Comprado: q(P - E) + qE/L = mmr·P·q  →  P = E(1 - 1/L) / (1 - mmr)
    Vendido:  q(E - P) + qE/L = mmr·P·q  →  P = E(1 + 1/L) / (1 + mmr)
    """
    if side > 0:
        return entry * (1 - 1 / leverage) / (1 - mmr)
    return entry * (1 + 1 / leverage) / (1 + mmr)


@dataclass
class Position:
    side: int
    qty: float
    entry: float
    leverage: float
    margin: float
    stop: float
    liq: float
    take_profit: float | None = None

    def pnl(self, price: float) -> float:
        return self.side * self.qty * (price - self.entry)


@dataclass(frozen=True)
class Trade:
    ts: pd.Timestamp
    symbol: str
    action: str  # open | close | stop | take_profit | eod | liquidation
    side: int
    qty: float
    price: float
    fee: float
    pnl: float  # realizado (0 na abertura)


class FuturesPaperBroker:
    def __init__(
        self, cash: float, taker_fee: float = BINANCE_USDM_TAKER, mmr: float = BINANCE_BTC_MMR,
        maker_fee: float = BINANCE_USDM_MAKER,
    ) -> None:
        self.cash = cash  # saldo livre (fora das margens)
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.mmr = mmr
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.funding_paid = 0.0

    def position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def equity(self, marks: dict[str, float]) -> float:
        """Saldo + margens + PnL não realizado (nunca abaixo de zero por posição isolada)."""
        total = self.cash
        for sym, p in self.positions.items():
            total += max(p.margin + p.pnl(marks[sym]), 0.0)
        return total

    def open(
        self,
        symbol: str,
        side: int,
        qty: float,
        price: float,
        leverage: float,
        stop: float,
        ts: pd.Timestamp,
        take_profit: float | None = None,
        maker: bool = False,
    ) -> Position:
        if symbol in self.positions:
            raise ValueError(f"já existe posição em {symbol}")
        if side not in (1, -1) or qty <= 0 or leverage < 1:
            raise ValueError("side ±1, qty > 0 e leverage >= 1")
        liq = liquidation_price(side, price, leverage, self.mmr)
        if (side > 0 and not liq < stop < price) or (side < 0 and not price < stop < liq):
            raise ValueError(f"stop {stop:.2f} precisa ficar entre a entrada {price:.2f} e a liquidação {liq:.2f}")
        notional = qty * price
        margin = notional / leverage
        fee = notional * (self.maker_fee if maker else self.taker_fee)
        if margin + fee > self.cash:
            raise InsufficientMargin(f"precisa {margin + fee:.2f}, tem {self.cash:.2f}")
        self.cash -= margin + fee
        pos = Position(side, qty, price, leverage, margin, stop, liq, take_profit)
        self.positions[symbol] = pos
        self.trades.append(Trade(ts, symbol, "open", side, qty, price, fee, 0.0))
        return pos

    def add(self, symbol: str, qty: float, price: float, ts: pd.Timestamp, maker: bool = False) -> Position:
        """Aumenta a posição (grid): preço médio, margem e liquidação recalculados."""
        pos = self.positions[symbol]
        notional = qty * price
        margin = notional / pos.leverage
        fee = notional * (self.maker_fee if maker else self.taker_fee)
        if margin + fee > self.cash:
            raise InsufficientMargin(f"precisa {margin + fee:.2f}, tem {self.cash:.2f}")
        self.cash -= margin + fee
        total = pos.qty + qty
        pos.entry = (pos.entry * pos.qty + price * qty) / total
        pos.qty = total
        pos.margin += margin
        pos.liq = liquidation_price(pos.side, pos.entry, pos.leverage, self.mmr)
        self.trades.append(Trade(ts, symbol, "add", pos.side, qty, price, fee, 0.0))
        return pos

    def close(self, symbol: str, price: float, ts: pd.Timestamp, action: str = "close", maker: bool = False) -> float:
        """Fecha a posição. Retorna o PnL realizado líquido de taxa."""
        pos = self.positions.pop(symbol)
        if action == "liquidation":
            self.trades.append(Trade(ts, symbol, action, pos.side, pos.qty, price, 0.0, -pos.margin))
            return -pos.margin
        pnl = pos.pnl(price)
        fee = pos.qty * price * (self.maker_fee if maker else self.taker_fee)
        released = pos.margin + pnl
        if released < 0:  # gap além da liquidação: perda limitada à margem isolada
            released, pnl = 0.0, -pos.margin
        self.cash += released - fee
        self.trades.append(Trade(ts, symbol, action, pos.side, pos.qty, price, fee, pnl - fee))
        return pnl - fee

    def apply_funding(self, symbol: str, rate: float, mark: float) -> float:
        """Comprado paga e vendido recebe quando a taxa é positiva. Retorna o valor pago."""
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        paid = pos.side * pos.qty * mark * rate
        self.cash -= paid
        self.funding_paid += paid
        return paid

    @property
    def fees(self) -> float:
        return sum(t.fee for t in self.trades)
