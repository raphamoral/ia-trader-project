import json
import time

import numpy as np
import pandas as pd
import pytest

from trader.flowhist.download import depth_minutes, read_flow, save_day, trades_minutes
from trader.realtime.news_labels import clean_label, is_crypto, slug_text
from trader.realtime.news_model import NewsModel, agreement, sent_class
from trader.research.flow_edge import decision_rows, features, quintiles, targets, walk_forward


def test_trades_minutes_aggregates_aggression():
    df = pd.DataFrame({"agg_trade_id": [1, 2, 3, 4], "price": [100.0, 101.0, 99.0, 100.0],
                       "quantity": [1.0, 2.0, 1000.0, 1.0], "first_trade_id": 0, "last_trade_id": 0,
                       "transact_time": [0, 30_000, 61_000, 62_000], "is_buyer_maker": ["false", "true", "true", "false"]})
    m = trades_minutes(df, large_usd=50_000)
    assert len(m) == 2
    first = m.iloc[0]
    assert first["open"] == 100 and first["high"] == 101 and first["close"] == 101
    assert first["buy_usd"] == 100 and first["sell_usd"] == 202
    second = m.iloc[1]
    assert second["large_sell_usd"] == pytest.approx(99_000) and second["large_buy_usd"] == 0


def test_depth_minutes_pivots_levels():
    rows = []
    for ts in ("2026-01-01 00:00:06", "2026-01-01 00:00:31"):
        for pct, n in ((-0.2, 10), (0.2, 20), (-1, 100), (1, 50), (-5, 1000), (5, 900), (-2, 7)):
            rows.append({"timestamp": ts, "percentage": pct, "depth": 0, "notional": n})
    d = depth_minutes(pd.DataFrame(rows))
    assert list(d.columns) == sorted(d.columns, key=list(d.columns).index)
    assert d.iloc[0]["bid_02"] == 10 and d.iloc[0]["ask_1"] == 50 and "bid_2" not in d.columns


def test_save_and_read_flow_roundtrip(tmp_path):
    idx = pd.date_range("2026-03-01", periods=3, freq="1min", tz="UTC")
    day = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    save_day(tmp_path, "SOLUSDT", day)
    save_day(tmp_path, "SOLUSDT", day.assign(close=[1.0, 2.0, 4.0]))  # regravar o dia não duplica
    got = read_flow(tmp_path, "SOLUSDT")
    assert len(got) == 3 and got["close"].iloc[-1] == 4.0


def flow_frame(n=60 * 24 * 100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    buy, sell = rng.uniform(1e4, 2e4, n), rng.uniform(1e4, 2e4, n)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "n_trades": 10.0,
                         "buy_usd": buy, "sell_usd": sell, "large_buy_usd": 0.0, "large_sell_usd": 0.0,
                         "bid_02": 1e5, "ask_02": 1e5, "bid_1": 1e6, "ask_1": 1e6, "bid_5": 1e7, "ask_5": 1e7},
                        index=idx)


def test_flow_features_are_causal_and_targets_forward():
    m = flow_frame(5000)
    full = features(m)
    part = features(m.iloc[:3000])
    pd.testing.assert_frame_equal(full.iloc[:3000], part)
    t = targets(m)
    # entra no fechamento do minuto seguinte, sai 5 minutos depois
    assert t["fwd_5"].iloc[10] == pytest.approx((m["close"].iloc[16] / m["close"].iloc[11] - 1) * 1e4)


def test_no_edge_in_noise_after_costs():
    m = flow_frame()
    df = decision_rows(m)
    feats = [c for c in df.columns if not c.startswith("fwd_")]
    q = quintiles(df, ["aggr_15", "book_1"], 15)
    assert (q["q5-q1"].abs() < 3).all()  # ruído: nenhuma faixa paga 10 bps
    t = walk_forward(df, feats, 15, cost=10.0)
    assert t.empty or t["net"].mean() < 0  # sem informação, o custo come tudo


def test_planted_flow_signal_is_found():
    m = flow_frame()
    # fluxo comprador forte num minuto → preço sobe nos 15 seguintes (efeito plantado)
    rng = np.random.default_rng(1)
    shock = rng.random(len(m)) < 0.05
    m.loc[shock, "buy_usd"] *= 20
    lr = np.log(m["close"]).diff().fillna(0).to_numpy().copy()
    for i in np.flatnonzero(shock):
        lr[i + 2:i + 17] += 0.0004
    m["close"] = 100 * np.exp(np.cumsum(lr))
    df = decision_rows(m)
    q = quintiles(df, ["aggr_1"], 15)
    assert q["q5-q1"].iloc[0] > 3


def test_slug_and_filters():
    assert slug_text("https://x.com/markets/2025/01/02/bitcoin-falls-below-90k-as-etf-outflows-mount.html") == \
        "bitcoin falls below 90k as etf outflows mount"
    assert is_crypto("bitcoin falls") and not is_crypto("petrobras sobe")
    assert clean_label({"kind": "meteor", "symbols": ["BTC", "DOGE"], "sentiment": 3, "severity": -1}) == \
        {"relevant": False, "kind": "other", "symbols": ["BTC"], "sentiment": 1.0, "severity": 0.0}


def test_news_model_learns_and_is_fast():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        coin = rng.choice(["bitcoin", "solana", "ethereum"])
        sym = {"bitcoin": "BTC", "solana": "SOL", "ethereum": "ETH"}[coin]
        if i % 2:
            rows.append({"text": f"{coin} exchange hacked for {rng.integers(10, 900)} million drained", "relevant": True,
                         "kind": "hack", "symbols": [sym], "sentiment": -0.8, "severity": 0.9})
        else:
            rows.append({"text": f"{coin} price analysis weekly outlook {i}", "relevant": False, "kind": "other",
                         "symbols": [], "sentiment": 0.0, "severity": 0.0})
    df = pd.DataFrame(rows)
    m = NewsModel().fit(df)
    p = m.predict(["solana protocol hacked 300 million drained", "bitcoin weekly outlook analysis"])
    assert p[0]["relevant"] and p[0]["kind"] == "hack" and p[0]["sentiment"] < 0 and p[0]["symbols"] == ["SOL"]
    assert not p[1]["relevant"]
    t = time.perf_counter()
    for _ in range(50):
        m.predict(["ethereum bridge exploit"])
    assert (time.perf_counter() - t) / 50 < 0.01  # bem abaixo de 10 ms (alvo: ~1 ms)
    rep = agreement(m.predict(df["text"].tolist()), df)
    assert rep["relevante"] > 0.95 and rep["graves_pegas"] > 0.9
    assert sent_class(-0.5) == -1 and sent_class(0.1) == 0
