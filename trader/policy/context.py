"""Contexto que a LLM lê antes de escrever a política: mercado, notícias recentes e livros.

Tudo limitado (poucas linhas, poucos caracteres) para caber no prompt e na RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.research.trend_daily import DAYS_PER_YEAR, ENSEMBLE
from trader.storage.candles import CandleStore

CRYPTO_WORDS = ("bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "ripple", "binance", "crypto", "cripto",
                "stablecoin", "hack", "exploit", "sec ", "etf", "fed", "juros", "rates")


def market(store: CandleStore, symbol: str, now: pd.Timestamp) -> dict[str, Any]:
    out: dict[str, Any] = {}
    d = store.read("binance", symbol, "1d", start=now - pd.Timedelta(days=260))
    if len(d) > 200:
        c = d.set_index("ts")["close"]
        out["committee"] = round(float(np.mean([c.iloc[-1] > c.rolling(n).mean().iloc[-1] for n in ENSEMBLE])), 2)
        out["vol20_annual"] = round(float(c.pct_change().tail(20).std() * np.sqrt(DAYS_PER_YEAR)), 3)
        out["ret_7d"] = round(float(c.iloc[-1] / c.iloc[-8] - 1), 4)
    m = store.read("binanceusdm", symbol, "5m", start=now - pd.Timedelta(days=3))
    if len(m) > 300:
        m = m.set_index("ts")
        c = m["close"]
        out["price"] = float(c.iloc[-1])
        out["ret_1h"] = round(float(c.iloc[-1] / c.iloc[-13] - 1), 4)
        out["ret_24h"] = round(float(c.iloc[-1] / c.iloc[-289] - 1), 4)
        tr = (m["high"] - m["low"]) / c
        atr = tr.rolling(14).mean()
        out["atr_pct_5m"] = round(float((atr.tail(288) <= atr.iloc[-1]).mean()), 2)  # percentil do ATR no último dia
    return out


def headlines(data: Path, now: pd.Timestamp, hours: int = 6, limit: int = 15) -> list[dict[str, str]]:
    try:
        from trader.news.pipeline import read_items

        items = read_items(data, since=now - pd.Timedelta(hours=hours), until=now)
    except Exception:
        return []
    if items.empty:
        return []
    text = items["original_text"].fillna("").str.slice(0, 160)
    relevant = text.str.lower().apply(lambda t: any(w in t for w in CRYPTO_WORDS))
    items = items.assign(text=text)[relevant].sort_values("published_at").tail(limit)
    return [{"at": f"{r.published_at:%Y-%m-%d %H:%M}", "source": r.source, "text": r.text} for r in items.itertuples()]


def build_context(data: Path, symbols: list[str], now: pd.Timestamp | None = None) -> dict[str, Any]:
    now = now or pd.Timestamp.now(tz="UTC")
    store = CandleStore(data)
    ctx: dict[str, Any] = {"now_utc": f"{now:%Y-%m-%d %H:%M}",
                           "market": {s: market(store, s, now) for s in symbols},
                           "news_last_6h": headlines(data, now)}
    try:
        from trader.evolution.priors import standing_lessons

        ctx["book_rules"] = standing_lessons(data / "books" / "playbook.json", 6)
    except Exception:
        pass
    return ctx
