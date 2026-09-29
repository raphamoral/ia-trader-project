"""Provedores de features externas (notícias) e a junção causal com os candles.

Contrato comum: fn(symbol, timeframe, index) -> DataFrame, onde index é um
DatetimeIndex UTC, o resultado tem o mesmo índice, e a linha t só usa itens
visíveis em <= t. Colunas com prefixo news_ (confiáveis) ou black_ (não
confiáveis).

A junção usa o FECHAMENTO de cada candle como instante: a decisão tomada no
fechamento do candle i vê exatamente o que o provedor diz para esse instante.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from trader.domain import parse_timeframe

FeatureProvider = Callable[[str, str, pd.DatetimeIndex], pd.DataFrame]
PREFIXES = ("news_", "black_")


def attach_features(
    candles: pd.DataFrame, symbol: str, timeframe: str, providers: list[FeatureProvider] | None
) -> pd.DataFrame:
    """Candles + colunas dos provedores, alinhadas pelo fechamento de cada candle."""
    if not providers:
        return candles
    close_idx = pd.DatetimeIndex(candles["ts"] + parse_timeframe(timeframe))
    out = candles.reset_index(drop=True).copy()
    for provider in providers:
        feats = provider(symbol, timeframe, close_idx)
        if not feats.index.equals(close_idx):
            raise ValueError(f"{getattr(provider, '__name__', provider)} devolveu índice diferente")
        bad = [c for c in feats.columns if not c.startswith(PREFIXES)]
        if bad:
            raise ValueError(f"colunas sem prefixo news_/black_: {bad}")
        for col in feats.columns:
            out[col] = feats[col].to_numpy(dtype="float32")
    return out


def asof_items(items: pd.DataFrame, index: pd.DatetimeIndex, value_cols: list[str], half_life: pd.Timedelta,
               prefix: str) -> pd.DataFrame:
    """Utilitário causal: para cada t, média dos itens com visible_at <= t, com decaimento exponencial.

    `items`: colunas visible_at + value_cols. Também devolve <prefix>minutes_since_last.
    """
    out = pd.DataFrame(index=index)
    items = items.sort_values("visible_at")
    vis = items["visible_at"].to_numpy(dtype="datetime64[ns]")
    t = index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(vis, t, side="right")  # itens visíveis em <= t
    hl = half_life / pd.Timedelta(minutes=1)
    for col in value_cols:
        vals = items[col].to_numpy(dtype=float)
        res = np.zeros(len(t))
        for k, p in enumerate(pos):
            if p == 0:
                continue
            ages = (t[k] - vis[:p]) / np.timedelta64(1, "m")
            w = np.exp(-np.log(2) * ages / hl)
            w[ages > 10 * hl] = 0
            res[k] = float((w * vals[:p]).sum() / w.sum()) if w.sum() > 1e-9 else 0.0
        out[f"{prefix}{col}"] = res
    last = np.where(pos > 0, (t - vis[np.maximum(pos - 1, 0)]) / np.timedelta64(1, "m"), 1e4)
    out[f"{prefix}minutes_since_last"] = last
    return out


def stub_white(seed: int = 0, events_per_day: float = 3.0, scheduled_every_days: int = 14) -> FeatureProvider:
    """Notícias sintéticas confiáveis: sentimento aleatório, confiança, e eventos agendados periódicos."""

    def provider(symbol: str, timeframe: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        span_days = max((index[-1] - index[0]) / pd.Timedelta(days=1), 1.0)
        n = int(span_days * events_per_day) + 1
        vis = index[0] + pd.to_timedelta(np.sort(rng.uniform(0, span_days, n)), unit="D")
        items = pd.DataFrame({"visible_at": vis, "sentiment": rng.uniform(-1, 1, n), "confidence": rng.uniform(0, 1, n)})
        out = asof_items(items, index, ["sentiment", "confidence"], pd.Timedelta(hours=2), "news_")
        out["news_spread"] = 0.3
        # eventos agendados: minutos até o próximo (conhecidos de antemão, não é vazamento)
        epoch = index[0].normalize()
        sched = pd.date_range(epoch, index[-1] + pd.Timedelta(days=scheduled_every_days),
                              freq=f"{scheduled_every_days}D", tz=index.tz) + pd.Timedelta(hours=18)
        nxt = np.searchsorted(sched.to_numpy(), index.to_numpy(), side="right")
        out["news_minutes_to_next_event"] = np.minimum(
            (sched.to_numpy()[np.minimum(nxt, len(sched) - 1)] - index.to_numpy()) / np.timedelta64(1, "m"), 10080
        )
        return out

    provider.__name__ = f"stub_white_{seed}"
    return provider


def stub_black(seed: int = 0, weighted: bool = False) -> FeatureProvider:
    """Notícias sintéticas não confiáveis (mais ruído)."""

    def provider(symbol: str, timeframe: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(seed + 1000)
        span_days = max((index[-1] - index[0]) / pd.Timedelta(days=1), 1.0)
        n = int(span_days * 10) + 1
        vis = index[0] + pd.to_timedelta(np.sort(rng.uniform(0, span_days, n)), unit="D")
        items = pd.DataFrame({"visible_at": vis, "mean_sent": rng.uniform(-1, 1, n)})
        prefix = "black_rw_" if weighted else "black_"
        return asof_items(items, index, ["mean_sent"], pd.Timedelta(hours=1), prefix)

    provider.__name__ = f"stub_black_{'rw_' if weighted else ''}{seed}"
    return provider


def real_white(data: str = "data") -> FeatureProvider:
    from trader.news.features import news_features

    def provider(symbol: str, timeframe: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        return news_features(symbol, timeframe, index, data=data)

    provider.__name__ = "news_features"
    return provider


def real_black(weighted: bool, data: str = "data") -> FeatureProvider:
    from pathlib import Path

    from trader.blacknews.api import black_features

    def provider(symbol: str, timeframe: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        return black_features(symbol, timeframe, index, weighted=weighted, data=Path(data))

    provider.__name__ = f"black_features_{'rw' if weighted else 'raw'}"
    return provider


def news_view(frame: pd.DataFrame, black_weight: float) -> dict[str, np.ndarray]:
    """Resumo que os genes de notícia usam: sentimento, confiança, frescor, próximo evento."""
    n = len(frame)
    zeros = np.zeros(n)

    def col(*names: str, default: float = 0.0) -> np.ndarray:
        for name in names:
            if name in frame.columns:
                return frame[name].to_numpy(dtype=float)
        return np.full(n, default)

    white = col("news_sentiment", "news_sentiment_4h")
    conf = col("news_confidence", "news_confidence_4h")
    spread = col("news_spread", "news_spread_4h")
    black = col("black_rw_mean_sent", "black_mean_sent")
    sentiment = white + black_weight * black
    since = col("news_minutes_since_last", "news_minutes_since_last_event", default=1e4)
    fresh = since <= 60
    if "news_count_1h" in frame.columns:
        fresh = frame["news_count_1h"].to_numpy(dtype=float) > 0
    return {
        "sentiment": np.clip(sentiment, -2, 2),
        "confidence": np.clip(conf if conf.any() else zeros, 0, 1),
        "spread": np.clip(spread, 0, 5),
        "fresh": fresh,
        "to_event": col("news_minutes_to_next_event", default=1e4),
    }
