"""Contexto de mercado ("saber quando"): features de regime, todas causais.

Cada valor na linha i usa só candles até i (fechados). O rótulo de regime
resume o contexto em quiet / normal / expanding / extreme, e é por ele que o
meta-controlador do genoma escolhe o setup ou decide ficar de fora.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from trader.daytrade.sessions import session_keys

REGIMES = ("quiet", "normal", "expanding", "extreme")
REGIME_DESCRIPTIONS = {
    "quiet": "ATR in the bottom 20% of its recent history: compressed market, small ranges",
    "normal": "ordinary volatility",
    "expanding": "current candle range >= 1.2x the max range of the last N candles, or ATR in the top 20%",
    "extreme": "range >= 2x the recent max range, or ATR in the top 5%: shock / news spike",
}
# limiares do rótulo
EXPANDING_RANGE, EXTREME_RANGE = 1.2, 2.0
EXPANDING_ATR_PCT, EXTREME_ATR_PCT, QUIET_ATR_PCT = 0.8, 0.95, 0.2


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def regime_features(
    candles: pd.DataFrame,
    range_lookback: int = 20,
    atr_period: int = 14,
    atr_lookback: int = 500,
    volume_lookback: int = 96,
    trend_period: int = 20,
    session_end: dt.time = dt.time(0, 0),
) -> pd.DataFrame:
    o, h, l, c, v = (candles[k] for k in ("open", "high", "low", "close", "volume"))
    rng = h - l
    atr = _atr(h, l, c, atr_period)
    key = session_keys(candles["ts"], session_end)
    close_time = candles["ts"] + (candles["ts"].diff().median() if len(candles) > 1 else pd.Timedelta(0))
    hour = close_time.dt.hour

    out = pd.DataFrame(index=candles.index)
    out["range_expansion"] = rng / rng.shift(1).rolling(range_lookback, min_periods=range_lookback).max()
    out["range_vs_atr"] = rng / atr.shift(1)
    out["atr_pct"] = atr.rolling(atr_lookback, min_periods=min(atr_lookback, 50)).rank(pct=True)
    vol_mean = v.rolling(volume_lookback, min_periods=20).mean()
    vol_std = v.rolling(volume_lookback, min_periods=20).std()
    out["volume_z"] = (v - vol_mean) / vol_std.replace(0, np.nan)
    path = c.diff().abs().rolling(trend_period, min_periods=trend_period).sum()
    out["efficiency_ratio"] = (c - c.shift(trend_period)).abs() / path.replace(0, np.nan)
    out["dist_session_high"] = c / h.groupby(key).cummax() - 1
    out["dist_session_low"] = c / l.groupby(key).cummin() - 1
    out["tod_bucket"] = (hour // 4).astype(float)  # 0..5, blocos de 4h em UTC
    out["session_bucket"] = np.select([hour < 8, hour < 13, hour < 21], [0.0, 1.0, 2.0], 3.0)  # Ásia/Europa/EUA/tarde
    out["gap"] = o / c.shift(1) - 1
    out["regime"] = label(out["range_expansion"], out["atr_pct"])
    return out


def label(range_expansion: pd.Series, atr_pct: pd.Series) -> pd.Series:
    re, ap = range_expansion.fillna(1.0), atr_pct.fillna(0.5)
    lab = np.select(
        [(re >= EXTREME_RANGE) | (ap >= EXTREME_ATR_PCT),
         (re >= EXPANDING_RANGE) | (ap >= EXPANDING_ATR_PCT),
         ap <= QUIET_ATR_PCT],
        ["extreme", "expanding", "quiet"],
        "normal",
    )
    return pd.Series(lab, index=range_expansion.index)
