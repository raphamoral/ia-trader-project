import json
import random

import numpy as np
import pandas as pd
import pytest

from trader.daytrade.backtest import run_daytrade_backtest
from trader.daytrade.genome import DayGenome, MARKET_STRATEGIES, eligible, random_day_genome
from trader.daytrade.market_signals import _closes_on, crowding_features, xs_ranks

T0 = pd.Timestamp("2026-01-01", tz="UTC")
TF = pd.Timedelta("5min")


def candles(close, start=T0):
    close = np.asarray(close, dtype=float)
    ts = pd.date_range(start, periods=len(close), freq="5min", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": 1.0})


def test_xs_rank_orders_assets_and_is_causal():
    rng = np.random.default_rng(1)
    idx = pd.date_range(T0 + TF, periods=600, freq="5min", tz="UTC")
    closes = {f"A{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005 * (i - 2), 0.003, 600))), index=idx)
              for i in range(5)}
    full = xs_ranks("A4", closes, (12,))["mkt_xs_rank_12"]
    assert full.dropna().between(0, 1).all() and full.dropna().mean() > 0.6  # o de maior deriva fica no topo
    cut = 400
    part = xs_ranks("A4", {k: v.iloc[:cut] for k, v in closes.items()}, (12,))["mkt_xs_rank_12"]
    pd.testing.assert_series_equal(full.iloc[:cut], part, check_names=False)  # futuro não muda o passado


def test_closes_only_use_candles_already_closed():
    df = candles([100, 101, 102, 103])  # aberturas 00:00..00:15, fechamentos 00:05..00:20
    idx = pd.DatetimeIndex([T0 + pd.Timedelta("7min"), T0 + pd.Timedelta("10min")])
    got = _closes_on(idx, df, "5m")
    assert got.tolist() == [100.0, 101.0]  # às 00:07 só o candle de 00:00 fechou


def test_funding_visible_only_after_settlement_and_stale_oi_is_nan():
    funding = pd.DataFrame({"ts": [T0, T0 + pd.Timedelta("8h")], "rate": [0.0001, 0.0009]})
    idx = pd.DatetimeIndex([T0 + pd.Timedelta("7h55min"), T0 + pd.Timedelta("8h")])
    out = crowding_features(idx, funding, None)
    assert out["mkt_funding_bps"].round(6).tolist() == [1.0, 9.0]
    assert out["mkt_oi_z"].isna().all()
    vis = pd.date_range(T0, periods=500, freq="5min", tz="UTC")
    metrics = pd.DataFrame({"ts": vis - TF, "visible_at": vis, "oi_usd": np.linspace(1e9, 2e9, 500)})
    late = pd.DatetimeIndex([vis[-1] + pd.Timedelta("2h")])  # 2h sem dado novo
    assert crowding_features(late, funding, metrics)["mkt_oi_z"].isna().all()


def test_crowd_block_stops_entries_on_the_crowded_side():
    rng = np.random.default_rng(2)
    df = candles(100 * np.exp(np.cumsum(rng.normal(0, 0.004, 2000))))
    df["mkt_funding_bps"] = 30.0  # longs pagando muito o tempo todo
    base = DayGenome("momentum", {"lookback": 6, "threshold_bps": 20}, direction="long", max_trades_per_day=20)
    free = run_daytrade_backtest(base, df, "SOL/USDT", "5m", df["ts"].iloc[600], None, cash=180.0)
    blocked = run_daytrade_backtest(DayGenome.from_dict({**base.to_dict(), "crowd_block": 10.0}),
                                    df, "SOL/USDT", "5m", df["ts"].iloc[600], None, cash=180.0)
    assert len(free.trades) > 0 and len(blocked.trades) == 0


def test_market_strategies_stay_out_of_any_and_roundtrip():
    assert not MARKET_STRATEGIES & set(eligible("any"))
    assert set(eligible("cross_section")) == MARKET_STRATEGIES
    rng = random.Random(3)
    gs = [random_day_genome(rng, styles=["cross_section"], market=True) for _ in range(30)]
    assert any(g.crowd_block for g in gs) and {g.strategy for g in gs} <= MARKET_STRATEGIES
    g = gs[0]
    assert DayGenome.from_dict(json.loads(json.dumps(g.to_dict()))) == g


def test_creator_catalog_mentions_new_strategies():
    from trader.daytrade.creator import DayLLMCreator

    c = DayLLMCreator("SOL/USDT", "5m", {"capital": "R$1000", "max_lev": "3", "max_risk": "0.02", "daily_loss": "0.05",
                                          "min_order": "5", "goal": "consistency"})
    p = c.prompt({"cause": "fees", "life": {}, "last_window": {}})
    assert "xs_momentum" in p and "funding_reversal" in p


def test_provider_uses_history_before_the_requested_slice(tmp_path):
    from trader.daytrade.market_signals import xs_provider
    from trader.storage.candles import CandleStore

    store = CandleStore(tmp_path)
    rng = np.random.default_rng(4)
    for i, sym in enumerate(["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT"]):
        store.write("binanceusdm", sym, "5m", candles(100 * np.exp(np.cumsum(rng.normal(0.0002 * i, 0.003, 800)))))
    idx = pd.date_range(T0 + TF * 700, periods=20, freq="5min", tz="UTC")  # trecho curto, bem depois do início
    out = xs_provider(tmp_path, basket=("AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT"), lookbacks=(288,))(
        "DDD/USDT", "5m", idx)
    assert out["mkt_xs_rank_288"].notna().all()  # 288 candles de histórico vêm de antes do trecho
