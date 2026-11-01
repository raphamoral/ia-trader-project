"""Notícias como alarme de proteção em cima da tendência diária.

Base: pesos diários do comitê de tendência (decididos no fechamento, valem no dia seguinte).
Alarme (em candles de 4h, causal): sentimento das últimas 24h muito abaixo do normal dos
últimos 30 dias (z-score) junto com um pico de volume de notícias; opcionalmente, hack recente.
Com o alarme, a exposição é cortada por H candles. Só reduz: notícia boa não aumenta posição.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from trader.research.trend_daily import ensemble_weights
from trader.storage.candles import CandleStore

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
BARS_PER_DAY = 6
LOOKBACK = 30 * BARS_PER_DAY  # "normal" = últimos 30 dias


def load(store: CandleStore, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    hist = start - pd.Timedelta(days=260)  # aquecimento das médias de 200 dias
    return pd.DataFrame({s: store.read("binance", s, "4h", start=hist, end=end).set_index("ts")["close"]
                         for s in SYMBOLS}).dropna()


def zscore(x: pd.Series) -> pd.Series:
    mean = x.rolling(LOOKBACK, min_periods=LOOKBACK // 2).mean()
    std = x.rolling(LOOKBACK, min_periods=LOOKBACK // 2).std()
    return (x - mean) / std.replace(0, np.nan)


def alarm_series(news: pd.DataFrame, z_thr: float, count_thr: float, hack: bool) -> pd.Series:
    sent_z = zscore(news["news_sentiment_24h"])
    count_z = zscore(news["news_count_24h"])
    alarm = (sent_z <= -z_thr) & (count_z >= count_thr)
    if hack:
        alarm |= news["news_event_hack_security_minutes"] < 240
    return alarm.fillna(False)


def curve(closes4h: pd.DataFrame, base_daily: pd.DataFrame, alarm: pd.Series | None, cut: float, hold: int,
          fee: float = 0.001) -> tuple[pd.Series, int]:
    """Patrimônio em 4h. Pesos diários entram no dia seguinte; o alarme age no próximo candle de 4h."""
    w = base_daily.shift(1).reindex(closes4h.index, method="ffill").fillna(0.0)
    n_alarms = 0
    if alarm is not None:
        mult = pd.Series(1.0, index=closes4h.index)
        fired = alarm.reindex(closes4h.index).fillna(False).to_numpy()
        n_alarms = int(fired.sum())
        active = np.zeros(len(fired), dtype=bool)
        for i in np.flatnonzero(fired):
            active[i + 1 : i + 1 + hold] = True  # decide no fechamento, corta a partir do próximo candle
        mult[active] = cut
        w = w.mul(mult, axis=0)
    ret = closes4h.pct_change().fillna(0.0)
    r = (w * ret).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0.0) * fee
    return (1 + r).cumprod(), n_alarms


def monthly(c: pd.Series) -> pd.Series:
    return c.resample("ME").last().pct_change().dropna()


def summarize(c: pd.Series) -> dict[str, float]:
    m = monthly(c)
    dd = float((1 - c / c.cummax()).max())
    return {"total": c.iloc[-1] / c.iloc[0] - 1, "mean_m": m.mean(), "worst_m": m.min(), "max_dd": dd}


def run(store: CandleStore, news: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
        vol_target: float = 0.3) -> pd.DataFrame:
    closes4h = load(store, start, end)
    daily = closes4h.resample("1D").last().dropna()
    base = ensemble_weights(daily, vol_target)
    closes4h = closes4h[closes4h.index >= start]
    rows = []
    c0, _ = curve(closes4h, base, None, 1.0, 0)
    rows.append({"config": "tendência pura", "alarms": 0, **summarize(c0)})
    for z, cnt, cut, hold, hack in itertools.product((1.5, 2.0), (0.0, 1.0), (0.5, 0.0), (1, 6), (False, True)):
        alarm = alarm_series(news, z, cnt, hack)
        c, n = curve(closes4h, base, alarm, cut, hold)
        rows.append({"config": f"z<=-{z} vol>={cnt} corta p/{int(cut * 100)}% por {hold * 4}h{' +hack' if hack else ''}",
                     "alarms": n, **summarize(c)})
    return pd.DataFrame(rows)
