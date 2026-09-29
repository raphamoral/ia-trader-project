"""Tipos do domínio compartilhados por todos os módulos.

Timestamps são sempre UTC. Um candle é identificado pelo instante de abertura
(`ts`) e só é visível para decisões depois de fechado (`ts + timeframe`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

_TIMEFRAME_UNITS = {"m": "min", "h": "h", "d": "D", "w": "W"}


def parse_timeframe(timeframe: str) -> pd.Timedelta:
    """Converte '1m', '15m', '1h', '1d', '1w' em Timedelta."""
    amount, unit = timeframe[:-1], timeframe[-1]
    if unit not in _TIMEFRAME_UNITS or not amount.isdigit():
        raise ValueError(f"timeframe inválido: {timeframe!r}")
    return pd.Timedelta(int(amount), _TIMEFRAME_UNITS[unit])


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    FLAT = "flat"


@dataclass(frozen=True)
class Signal:
    """Saída do núcleo: o que o motor quer, antes de passar pelo risco."""

    side: Side
    confidence: float = 0.0  # 0..1
    size: float = 0.0  # fração da posição máxima permitida, 0..1
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence deve estar em [0, 1]")
        if not 0.0 <= self.size <= 1.0:
            raise ValueError("size deve estar em [0, 1]")


@dataclass(frozen=True)
class MarketView:
    """Retrato do mercado num instante, sem nenhum dado do futuro."""

    symbol: str
    timeframe: pd.Timedelta
    as_of: pd.Timestamp
    candles: pd.DataFrame

    @classmethod
    def at(cls, symbol: str, timeframe: str, candles: pd.DataFrame, as_of: pd.Timestamp) -> MarketView:
        """Monta a visão contendo apenas candles já fechados em `as_of`."""
        tf = parse_timeframe(timeframe)
        closed = candles[candles["ts"] + tf <= as_of]
        return cls(symbol=symbol, timeframe=tf, as_of=as_of, candles=closed.reset_index(drop=True))

    @property
    def last_close(self) -> float:
        return float(self.candles["close"].iloc[-1])


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float

    def __post_init__(self) -> None:
        if self.side is Side.FLAT:
            raise ValueError("ordem precisa ser BUY ou SELL")
        if self.quantity <= 0:
            raise ValueError("quantity deve ser positiva")


@dataclass(frozen=True)
class Fill:
    order: Order
    price: float
    fee: float
    ts: pd.Timestamp
