"""Coleta mês a mês: 1m tem ~43 mil candles/mês, e tudo de uma vez estoura a RAM.

    python -m trader.daytrade.collect_chunked BTC/USDT 1m 2024-01-01
"""
from __future__ import annotations

import gc
import sys

import pandas as pd

from trader.daytrade import data
from trader.storage.candles import CandleStore


def main(symbol: str, timeframe: str, since: str, root: str = "data") -> None:
    store = CandleStore(root)
    start = pd.Timestamp(since, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    cursor = start
    total = 0
    while cursor < now:
        chunk_end = min(cursor + pd.DateOffset(months=1), now)
        total += data.update_candles(store, symbol, timeframe, start, now=chunk_end)
        cursor = chunk_end
        gc.collect()
    print(f"{symbol} {timeframe}: +{total} candles")


if __name__ == "__main__":
    main(*sys.argv[1:])
