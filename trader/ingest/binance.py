"""Coleta de candles da Binance via ccxt (endpoint público, sem chave)."""
from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from trader.domain import CANDLE_COLUMNS, parse_timeframe
from trader.storage.candles import CandleStore

EXCHANGE = "binance"


class OhlcvSource(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None) -> list[list[Any]]: ...


def default_source() -> OhlcvSource:
    import ccxt

    return ccxt.binance({"enableRateLimit": True})


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def fetch_candles(
    source: OhlcvSource,
    symbol: str,
    timeframe: str,
    since: pd.Timestamp,
    now: pd.Timestamp | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Baixa candles fechados de `since` até `now`, paginando."""
    tf = parse_timeframe(timeframe)
    now = now or pd.Timestamp.now(tz="UTC")
    tf_ms = int(tf.total_seconds() * 1000)
    cursor, end = _to_ms(since), _to_ms(now)
    rows: list[list[Any]] = []
    while cursor < end:
        batch = source.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + tf_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[(df["ts"] >= since) & (df["ts"] + tf <= now)]  # descarta o candle ainda aberto
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def update(
    store: CandleStore,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    source: OhlcvSource | None = None,
    now: pd.Timestamp | None = None,
) -> int:
    """Continua a coleta a partir do último candle salvo. Retorna linhas novas."""
    last = store.last_ts(EXCHANGE, symbol, timeframe)
    since = last + parse_timeframe(timeframe) if last is not None else start
    candles = fetch_candles(source or default_source(), symbol, timeframe, since, now)
    store.write(EXCHANGE, symbol, timeframe, candles)
    return len(candles)
