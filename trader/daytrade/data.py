"""Coleta de candles e funding dos futuros perpétuos USDT-M da Binance.

Os candles vão para o mesmo CandleStore, sob a exchange "binanceusdm".
O símbolo guardado é o curto ("BTC/USDT"); o ccxt usa "BTC/USDT:USDT".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from trader.domain import parse_timeframe
from trader.ingest.binance import OhlcvSource, fetch_candles
from trader.storage.candles import CandleStore

EXCHANGE = "binanceusdm"


def perp_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTC/USDT:USDT' (liquidado em USDT)."""
    return symbol if ":" in symbol else f"{symbol}:{symbol.split('/')[1]}"


def default_exchange():
    import ccxt

    return ccxt.binanceusdm({"enableRateLimit": True})


class _PerpSource:
    """Adapta o nome do símbolo: o store usa 'BTC/USDT', o ccxt 'BTC/USDT:USDT'."""

    def __init__(self, exchange: Any) -> None:
        self.exchange = exchange

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return self.exchange.fetch_ohlcv(perp_symbol(symbol), timeframe, since=since, limit=limit)


def update_candles(
    store: CandleStore,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    source: OhlcvSource | None = None,
    now: pd.Timestamp | None = None,
) -> int:
    last = store.last_ts(EXCHANGE, symbol, timeframe)
    since = last + parse_timeframe(timeframe) if last is not None else start
    candles = fetch_candles(source or _PerpSource(default_exchange()), symbol, timeframe, since, now, limit=1500)
    store.write(EXCHANGE, symbol, timeframe, candles)
    return len(candles)


def funding_path(root: Path | str, symbol: str) -> Path:
    return Path(root) / "funding" / EXCHANGE / f"{symbol.replace('/', '')}.parquet"


def update_funding(root: Path | str, symbol: str, start: pd.Timestamp, exchange: Any = None) -> int:
    """Histórico de taxas de funding (público). Colunas: ts, rate."""
    exchange = exchange or default_exchange()
    path = funding_path(root, symbol)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ts", "rate"])
    since = int(((old["ts"].max() + pd.Timedelta(seconds=1)) if len(old) else start).timestamp() * 1000)
    rows = []
    while True:
        batch = exchange.fetch_funding_rate_history(perp_symbol(symbol), since=since, limit=1000)
        if not batch:
            break
        rows.extend((r["timestamp"], r["fundingRate"]) for r in batch)
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= since or len(batch) < 1000:
            break
        since = nxt
    new = pd.DataFrame(rows, columns=["ts", "rate"])
    new["ts"] = pd.to_datetime(new["ts"], unit="ms", utc=True).dt.floor("min")
    both = pd.concat([old, new]) if len(old) else new
    both = both.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    both.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def read_funding(root: Path | str, symbol: str) -> pd.DataFrame:
    path = funding_path(root, symbol)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ts", "rate"])
