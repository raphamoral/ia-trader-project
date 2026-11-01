"""Tendência no diário com proteção contra quedas × comprar e segurar.

Regra (decidida no fechamento do dia, executada no dia seguinte):
  - comprado se fechamento > média de N dias; senão, em caixa;
  - tamanho = min(1, vol_alvo / vol_realizada_20d)  (opcional: menos exposição quando o mercado está nervoso).
Testa uma grade de N para não escolher o parâmetro "que deu certo" a dedo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.storage.candles import CandleStore

DAYS_PER_YEAR = 365


def daily_closes(store: CandleStore, symbol: str) -> pd.Series:
    c = store.read("binance", symbol, "4h")
    if c.empty:
        return pd.Series(dtype=float)
    return c.set_index("ts")["close"].resample("1D").last().dropna()


def backtest(close: pd.Series, n: int, fee: float = 0.001, vol_target: float | None = None) -> pd.Series:
    """Curva de patrimônio diária (começa em 1)."""
    ret = close.pct_change().fillna(0.0)
    signal = (close > close.rolling(n).mean()).astype(float)
    if vol_target:
        vol = ret.rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
        signal = signal * (vol_target / vol).clip(upper=1.0).fillna(0.0)
    position = signal.shift(1).fillna(0.0)  # decide hoje, executa amanhã
    turnover = position.diff().abs().fillna(position.abs())
    strat = position * ret - turnover * fee
    curve = (1 + strat).cumprod()
    return curve.iloc[n:] / curve.iloc[n]  # só depois do aquecimento da média


def stats(curve: pd.Series) -> dict[str, float]:
    years = len(curve) / DAYS_PER_YEAR
    total = curve.iloc[-1] - 1
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    dd = float((1 - curve / curve.cummax()).max())
    daily = curve.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(DAYS_PER_YEAR)) if daily.std() > 0 else 0.0
    monthly = curve.resample("ME").last().pct_change().dropna()
    return {"total": total, "cagr": cagr, "max_dd": dd, "calmar": cagr / dd if dd else 0.0, "sharpe": sharpe,
            "pos_months": float((monthly > 0).mean()) if len(monthly) else 0.0,
            "worst_month": float(monthly.min()) if len(monthly) else 0.0}


def compare(store: CandleStore, symbols: list[str], grid: list[int], vol_target: float | None, fee: float) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        close = daily_closes(store, sym)
        if len(close) < max(grid) + 60:
            continue
        start = close.index[max(grid)]  # mesmo período para todos os N e para comprar e segurar
        bh = (close[start:] / close[start:].iloc[0])
        rows.append({"symbol": sym, "rule": "comprar e segurar", **stats(bh)})
        for n in grid:
            curve = backtest(close, n, fee, vol_target)
            rows.append({"symbol": sym, "rule": f"tendência {n}d" + (" + vol" if vol_target else ""),
                         **stats(curve[start:] / curve[start:].iloc[0])})
    return pd.DataFrame(rows)


ENSEMBLE = (50, 100, 150, 200)


def ensemble_weights(closes: pd.DataFrame, vol_target: float | None, lookbacks=ENSEMBLE) -> pd.DataFrame:
    """Peso de cada moeda: média dos sinais de tendência (comitê), ajustada pela volatilidade, pesos iguais."""
    ret = closes.pct_change()
    signal = sum((closes > closes.rolling(n).mean()).astype(float) for n in lookbacks) / len(lookbacks)
    if vol_target:
        vol = ret.rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
        signal = signal * (vol_target / vol).clip(upper=1.0)
    return (signal / closes.shape[1]).fillna(0.0)


def portfolio_curve(closes: pd.DataFrame, weights: pd.DataFrame, fee: float = 0.001) -> pd.Series:
    """Decide no fechamento, executa no dia seguinte; paga taxa sobre o que muda de posição."""
    ret = closes.pct_change().fillna(0.0)
    w = weights.shift(1).fillna(0.0)
    r = (w * ret).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0.0) * fee
    return (1 + r).cumprod()


def monthly_report(curve: pd.Series) -> dict[str, float]:
    m = curve.resample("ME").last().pct_change().dropna()
    t = m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 1 and m.std() > 0 else 0.0
    return {"months": len(m), "mean": m.mean(), "pos": (m > 0).mean(), "worst": m.min(), "t": t, "series": m}
