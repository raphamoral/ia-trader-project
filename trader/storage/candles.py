"""Armazenamento de candles em Parquet, um arquivo por ano.

Layout: <root>/candles/<exchange>/<SYMBOL>/<timeframe>/<ano>.parquet
Escritas são idempotentes: regravar o mesmo período não duplica linhas.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from trader.domain import CANDLE_COLUMNS


class CandleStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, exchange: str, symbol: str, timeframe: str) -> Path:
        return self.root / "candles" / exchange / symbol.replace("/", "") / timeframe

    def write(self, exchange: str, symbol: str, timeframe: str, candles: pd.DataFrame) -> None:
        missing = set(CANDLE_COLUMNS) - set(candles.columns)
        if missing:
            raise ValueError(f"colunas faltando: {sorted(missing)}")
        if candles.empty:
            return
        folder = self._dir(exchange, symbol, timeframe)
        folder.mkdir(parents=True, exist_ok=True)
        for year, part in candles[CANDLE_COLUMNS].groupby(candles["ts"].dt.year):
            path = folder / f"{year}.parquet"
            if path.exists():
                part = pd.concat([pd.read_parquet(path), part])
            part = part.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
            tmp = path.with_suffix(".tmp")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, path)

    def read(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Candles com start <= ts < end, ordenados."""
        folder = self._dir(exchange, symbol, timeframe)
        files = sorted(folder.glob("*.parquet"))
        if start is not None:
            files = [f for f in files if int(f.stem) >= start.year]
        if end is not None:
            files = [f for f in files if int(f.stem) <= end.year]
        if not files:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if start is not None:
            df = df[df["ts"] >= start]
        if end is not None:
            df = df[df["ts"] < end]
        return df.reset_index(drop=True)

    def last_ts(self, exchange: str, symbol: str, timeframe: str) -> pd.Timestamp | None:
        files = sorted(self._dir(exchange, symbol, timeframe).glob("*.parquet"))
        if not files:
            return None
        return pd.read_parquet(files[-1], columns=["ts"])["ts"].max()
