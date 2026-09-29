"""Backtest barra a barra com o mesmo fluxo da operação real.

Decide no fechamento do candle i e executa na abertura do candle i+1, pelo
mesmo RiskGate e por uma corretora simulada com taxa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader.core.engine import Engine
from trader.domain import Fill, MarketView, Side, parse_timeframe
from trader.execution.paper import BINANCE_SPOT_FEE, InsufficientFunds, PaperBroker
from trader.risk.gate import RiskGate, RiskLimits

SIM_LIMITS = RiskLimits(max_position_fraction=0.95, daily_loss_limit=0.05)


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    equity: pd.Series  # patrimônio no fechamento de cada candle
    fills: list[Fill]

    @property
    def fees(self) -> float:
        return sum(f.fee for f in self.fills)

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_cash - 1) if len(self.equity) else 0.0

    @property
    def gross_return(self) -> float:
        """Retorno se não houvesse taxa (aproximado)."""
        return float((self.equity.iloc[-1] + self.fees) / self.initial_cash - 1) if len(self.equity) else 0.0

    @property
    def max_drawdown(self) -> float:
        if self.equity.empty:
            return 0.0
        curve = self.equity.to_numpy()
        peak = np.maximum.accumulate(curve)
        return float(np.max(1 - curve / peak))

    @property
    def n_trades(self) -> int:
        """Número de entradas (compras)."""
        return sum(1 for f in self.fills if f.order.side is Side.BUY)


def run_backtest(
    engine: Engine,
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    cash: float = 10_000.0,
    fee_rate: float = BINANCE_SPOT_FEE,
    limits: RiskLimits = SIM_LIMITS,
) -> BacktestResult:
    """Opera candles com start <= ts < end; os anteriores servem de aquecimento."""
    tf = parse_timeframe(timeframe)
    df = candles.reset_index(drop=True)
    ts = df["ts"]
    first = int(ts.searchsorted(start))
    last = int(ts.searchsorted(end)) if end is not None else len(df)
    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()
    warmup = getattr(engine, "warmup", None)

    broker = PaperBroker(cash=cash, fee_rate=fee_rate)
    gate = RiskGate(limits)
    pending = None
    curve_ts, curve = [], []

    for i in range(first, last):
        broker.mark(symbol, float(opens[i]), ts[i])
        if pending is not None:
            try:
                broker.submit(pending)
            except InsufficientFunds:
                pass  # gap de abertura maior que a folga de caixa: ordem não sai
            pending = None

        close_time = ts[i] + tf
        broker.mark(symbol, float(closes[i]), close_time)
        curve_ts.append(close_time)
        curve.append(broker.equity())
        if i == last - 1:
            break

        lo = 0 if warmup is None else max(0, i + 1 - warmup)
        view = MarketView(symbol=symbol, timeframe=tf, as_of=close_time, candles=df.iloc[lo : i + 1])
        signal = engine.decide(view)
        pending = gate.order_for(
            signal, symbol, float(closes[i]), broker.equity(), broker.position(symbol), close_time
        )

    return BacktestResult(initial_cash=cash, equity=pd.Series(curve, index=curve_ts), fills=broker.fills)


def walk_forward_windows(
    start: pd.Timestamp, end: pd.Timestamp, length: pd.Timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Janelas consecutivas [a, b) de tamanho fixo cobrindo start..end."""
    windows = []
    cursor = start
    while cursor + length <= end:
        windows.append((cursor, cursor + length))
        cursor += length
    return windows
