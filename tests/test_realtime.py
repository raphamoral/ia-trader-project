import math

import numpy as np
import pandas as pd
import pytest

from trader.daytrade.broker import FuturesPaperBroker
from trader.daytrade.genome import _atr, _rsi
from trader.policy.executor import PolicyExecutor
from trader.policy.schema import parse_policy
from trader.realtime.engine import RealtimeEngine
from trader.realtime.flow import TradeFlow, book_features
from trader.realtime.indicators import Bar, Fibonacci, IndicatorSet, RollingExtreme
from trader.realtime.news_reflex import classify, usd_amount


def candles(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.005, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.005, n))
    ts = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": rng.uniform(1, 10, n)})


def run(df):
    ind = IndicatorSet()
    return pd.DataFrame([ind.update(Bar(r.ts, r.open, r.high, r.low, r.close, r.volume)) for r in df.itertuples()])


def test_incremental_matches_vectorized():
    df = candles()
    out = run(df)
    c = df["close"]
    pd.testing.assert_series_equal(out["sma50"], c.rolling(50).mean(), check_names=False)
    pd.testing.assert_series_equal(out["ema21"], c.ewm(span=21, adjust=False).mean(), check_names=False)
    pd.testing.assert_series_equal(out["rsi14"].iloc[1:], _rsi(c, 14).iloc[1:], check_names=False)
    pd.testing.assert_series_equal(out["atr14"], _atr(df["high"], df["low"], c, 14), check_names=False)
    bb = c.rolling(20).mean() + 2 * c.rolling(20).std()
    np.testing.assert_allclose(out["bb_up"].iloc[19:], bb.iloc[19:], rtol=1e-9)
    hi, lo = df["high"].rolling(14).max(), df["low"].rolling(14).min()
    np.testing.assert_allclose(out["stoch_k"].iloc[13:], (100 * (c - lo) / (hi - lo)).iloc[13:], rtol=1e-9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    np.testing.assert_allclose(out["macd_hist"], macd - macd.ewm(span=9, adjust=False).mean(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(out["donchian_hi"].iloc[19:], df["high"].rolling(20).max().iloc[19:])


def test_indicators_are_causal():
    df = candles()
    full = run(df)
    part = run(df.iloc[:250])
    pd.testing.assert_frame_equal(full.iloc[:250][part.columns], part)


def test_rolling_extreme():
    r = RollingExtreme(3, "max")
    vals = [r.update(x) for x in [1, 5, 2, 1, 0, 7]]
    assert math.isnan(vals[0]) and vals[2:] == [5, 5, 2, 7]


def test_fibonacci_confirms_swings_causally():
    f = Fibonacci(swing_pct=0.05)
    path = [(100, 99), (110, 105), (120, 115), (118, 112), (113, 108)]  # sobe a 120 e recua > 5%
    for h, l in path:
        f.update(h, l)
    assert [s for s in f.swings] == [(120, "top")]
    for h, l in [(110, 100), (106, 101), (112, 104)]:  # fundo em 100, sobe > 5%: confirmado
        f.update(h, l)
    lv = f.levels()
    assert lv["fib_from"] == 120 and lv["fib_to"] == 100
    assert lv["fib_0.618"] == pytest.approx(100 + 20 * 0.618)


def test_book_features():
    bids = [(100 - i * 0.1, 1.0) for i in range(20)]
    asks = [(100.1 + i * 0.1, 1.0) for i in range(20)]
    bids[3] = (bids[3][0], 50.0)
    f = book_features(0, bids, asks)
    assert f.spread_bps == pytest.approx(0.1 / 100.05 * 1e4)
    assert f.imbalance5 > 0.8 and f.bid_wall == bids[3][0] and f.ask_wall is None


def test_trade_flow_window():
    fl = TradeFlow(window_ms=1000, large_notional=1000)
    fl.update(0, 100, 5, False)  # compra agressora 500
    fl.update(500, 100, 20, True)  # venda agressora 2000 (grande)
    s = fl.snapshot()
    assert s["cvd"] == -15 and s["buy_usd_win"] == 500 and s["sell_usd_win"] == 2000 and s["large_trades"] == 1
    fl.update(1600, 100, 1, False)  # a compra de t=0 e a venda de t=500 saem da janela
    assert fl.snapshot()["sell_usd_win"] == 0 and fl.snapshot()["buy_usd_win"] == 100


def test_engine_builds_minutes_and_feeds_executor():
    ex = PolicyExecutor(FuturesPaperBroker(1000.0))
    ex.set_policy(parse_policy({"symbols": {"SOL/USDT": {"bias": "long", "stop": 0.02}}}, ["SOL/USDT"]),
                  pd.Timestamp(0, unit="ms", tz="UTC"))
    minutes = []
    eng = RealtimeEngine(["SOL/USDT"], ex, lambda s, r: minutes.append(r), log=lambda m: None)
    ex.on_signal(pd.Timestamp(0, unit="ms", tz="UTC"), "SOL/USDT", 1, "momentum", 100.0)

    def trade(ms, p):
        eng.handle({"stream": "solusdt@aggTrade", "data": {"T": ms, "p": str(p), "q": "1", "m": False}})

    eng.handle({"stream": "solusdt@depth20@100ms", "data": {"T": 0, "b": [["99.9", "3"]] * 6, "a": [["100.1", "1"]] * 6}})
    trade(1000, 100.0)
    trade(30_000, 101.0)
    trade(61_000, 99.0)  # fecha o minuto 0
    assert len(minutes) == 1 and minutes[0]["high"] == 101.0 and minutes[0]["imbalance5"] == pytest.approx(0.5)
    trade(62_000, 97.9)  # stop de 2%: o executor zera no tick
    assert ex.broker.position("SOL/USDT") is None


def test_news_reflex():
    e = classify("Bitget's $388M hack pushes Q3 crypto security losses past $1B")
    assert e.kind == "hack" and e.sentiment < 0 and e.severity == 0.9
    assert classify("NEAR Intents hit by $3.8 million exploit").severity == 0.5
    e = classify("SEC approves spot Solana ETF")
    assert e.symbols == ("SOL/USDT",) and e.sentiment > 0
    assert usd_amount("lost $1.2 billion and $50 million") == 1.2e9
