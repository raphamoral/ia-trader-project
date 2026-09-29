import random

import numpy as np
import pandas as pd
import pytest

from trader.daytrade.backtest import run_daytrade_backtest
from trader.daytrade.broker import BINANCE_USDM_MAKER, BINANCE_USDM_TAKER
from trader.daytrade.features import asof_items, attach_features, news_view, stub_black, stub_white
from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.genome import STYLES, DayGenome, eligible, mutate_day, random_day_genome, switch_style
from trader.daytrade.regime import REGIMES, regime_features

T0 = pd.Timestamp("2024-01-01", tz="UTC")
TF = pd.Timedelta(minutes=15)


def ohlc(o, h, l, c, start=T0, freq="15min", volume=None):
    n = len(c)
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
        "open": np.asarray(o, float), "high": np.asarray(h, float), "low": np.asarray(l, float),
        "close": np.asarray(c, float), "volume": np.ones(n) if volume is None else np.asarray(volume, float),
    })


def walk(n, seed=0, vol=0.004):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.concatenate([[100.0], c[:-1]]) * (1 + rng.normal(0, 0.001, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return ohlc(o, hi, lo, c, volume=rng.lognormal(0, 0.8, n))


# regime

def test_every_regime_feature_is_causal():
    df = walk(1500)
    full = regime_features(df, atr_lookback=200)
    for cut in (600, 1111):
        part = regime_features(df.iloc[:cut], atr_lookback=200)
        for col in full.columns:
            a, b = full[col].iloc[:cut].to_numpy(), part[col].to_numpy()
            if a.dtype == object:
                assert (a == b).all(), col
            else:
                np.testing.assert_allclose(a, b, equal_nan=True, err_msg=f"{col} olha o futuro")
    assert set(full["regime"].unique()) <= set(REGIMES)


def test_range_expansion_and_label():
    # 30 candles de amplitude 1, depois um de amplitude 3 → expansão 3x → extreme
    n = 31
    c = np.full(n, 100.0)
    h, l = c + 0.5, c - 0.5
    h[-1], l[-1] = 101.5, 98.5
    reg = regime_features(ohlc(c, h, l, c), range_lookback=20, atr_lookback=30)
    assert reg["range_expansion"].iloc[-1] == pytest.approx(3.0)
    assert reg["regime"].iloc[-1] == "extreme"


# ordens

def genome(**kw):
    base = dict(strategy="momentum", params={"lookback": 3, "threshold_bps": 10}, stop=0.02, risk=0.01)
    base.update(kw)
    return DayGenome(**base).normalized()


def test_stop_entry_fills_only_when_next_bar_trades_through():
    # gatilho = máxima do candle do sinal (101) + 0; próximo candle só encosta (máx 101): não executa
    o = [100, 100, 100, 100, 100]
    h = [100.5, 101, 101, 101.5, 101]
    l = [99.5, 99.5, 99.8, 99.9, 99.9]
    c = [100, 100.8, 100.5, 101.2, 100.9]
    df = ohlc(o, h, l, c)
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    g = genome(strategy="vol_expansion", params={"threshold10": 12, "lookback": 5}, entry_offset=0.0,
               slippage=0.001, order_expiry=3)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    opens = [t for t in res.trades if t.action == "open"]
    assert len(opens) == 1
    assert opens[0].ts == df["ts"].iloc[3]  # candle 2 só encostou; candle 3 passou
    assert opens[0].price == pytest.approx(101 * 1.001)
    assert res.stop_placed == 1 and res.stop_filled == 1


def test_stop_entry_never_fills_on_signal_bar():
    df = ohlc([100, 100, 100], [100.5, 105, 100.2], [99.5, 99.5, 99.8], [100, 104, 100])
    ev = np.full(3, np.nan)
    ev[1] = 1
    g = genome(strategy="vol_expansion", params={"threshold10": 12, "lookback": 5}, entry_offset=0.0)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    assert res.n_trades == 0  # o candle 2 não passou de 105


def test_limit_needs_trade_through_with_haircut_and_expires():
    # limite de compra = 100 × (1 − 0,001) = 99,9. Mínimas: 99,9 (encosta), 99,89 (passa < 2 bps), 99,85
    o = [100, 100, 100, 100, 100, 100]
    h = [100.2] * 6
    l = [99.95, 99.95, 99.9, 99.89, 99.85, 99.8]
    c = [100.0] * 6
    df = ohlc(o, h, l, c)
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    g = genome(style="mean_reversion", strategy="bb_revert", params={"period": 20, "k10": 20},
               entry_offset=0.001, order_expiry=5)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={}, queue_haircut=0.0002)
    opens = [t for t in res.trades if t.action == "open"]
    assert len(opens) == 1 and opens[0].ts == df["ts"].iloc[4]
    assert opens[0].price == pytest.approx(99.9)
    assert opens[0].fee == pytest.approx(opens[0].qty * 99.9 * BINANCE_USDM_MAKER)
    # expira: só 2 candles de validade, e nenhum passa o haircut antes disso
    g2 = genome(style="mean_reversion", strategy="bb_revert", params={"period": 20, "k10": 20},
                entry_offset=0.001, order_expiry=2)
    res2 = run_daytrade_backtest(g2, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={}, queue_haircut=0.0002)
    assert res2.n_trades == 0 and res2.limit_placed == 1 and res2.limit_filled == 0


def test_market_entry_pays_taker():
    df = walk(40)
    ev = np.full(len(df), np.nan)
    ev[5] = 1
    res = run_daytrade_backtest(genome(), df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    op = res.trades[0]
    assert op.fee == pytest.approx(op.qty * op.price * BINANCE_USDM_TAKER)


def test_trailing_stop_ratchets():
    c = [100, 100, 102, 104, 106, 103, 101]
    o = [100] + c[:-1]
    df = ohlc(o, [max(a, b) + 0.2 for a, b in zip(o, c)], [min(a, b) - 0.2 for a, b in zip(o, c)], c)
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    res = run_daytrade_backtest(genome(trail=0.02, stop=0.03), df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    stop_exit = [t for t in res.trades if t.action == "stop"]
    assert stop_exit and stop_exit[0].price == pytest.approx(106 * 0.98)


def test_grid_inventory_is_bounded():
    # preço desce em degraus abaixo da VWAP: o grid compra no máximo `levels` vezes
    c = list(np.linspace(100, 95, 60))
    df = ohlc(c, [x + 0.05 for x in c], [x - 0.3 for x in c], c)
    g = DayGenome("grid", {"step_bps": 30, "levels": 3}, style="grid", direction="long", stop=0.02, risk=0.01,
                  leverage=3).normalized()
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], funding={})
    assert any(t.action == "add" for t in res.trades)
    adds_per_episode = 0
    peak = 0
    for t in res.trades:
        if t.action == "open":
            adds_per_episode = 1
        elif t.action == "add":
            adds_per_episode += 1
        peak = max(peak, adds_per_episode)
    assert peak <= 3


# meta-controlador

def test_stand_aside_regime_blocks_entries():
    df = walk(400, seed=3)
    ev = np.ones(len(df))  # quer comprar sempre
    everything_off = {r: "stand_aside" for r in REGIMES}
    g = genome(regime_map=everything_off)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[50], events=None, funding={})
    assert res.n_trades == 0 and res.stand_aside_bars > 0
    assert sum(res.regime_bars.values()) == len(df) - 50


# notícias

def test_future_dated_item_cannot_affect_earlier_bar():
    idx = pd.date_range(T0, periods=10, freq="15min", tz="UTC")
    items = pd.DataFrame({"visible_at": [idx[6] + pd.Timedelta(seconds=1)], "sentiment": [1.0]})
    out = asof_items(items, idx, ["sentiment"], pd.Timedelta(hours=1), "news_")
    assert (out["news_sentiment"].iloc[:7] == 0).all()  # visível só depois de idx[6]
    assert out["news_sentiment"].iloc[7] == pytest.approx(1.0)


def test_attach_features_uses_bar_close_and_validates():
    df = walk(20)
    seen = {}

    def provider(symbol, timeframe, index):
        seen["index"] = index
        return pd.DataFrame({"news_x": np.arange(len(index), dtype=float)}, index=index)

    out = attach_features(df, "X", "15m", [provider])
    assert seen["index"][0] == df["ts"].iloc[0] + TF  # fechamento do candle
    assert "news_x" in out.columns

    def bad(symbol, timeframe, index):
        return pd.DataFrame({"x": 0.0}, index=index)

    with pytest.raises(ValueError):
        attach_features(df, "X", "15m", [bad])


def test_stub_providers_are_causal():
    idx = pd.date_range(T0, periods=500, freq="15min", tz="UTC")
    for prov in (stub_white(3), stub_black(3), stub_black(3, weighted=True)):
        full = prov("X", "15m", idx)
        # mesmo índice truncado não pode mudar o passado (itens gerados do mesmo seed e mesmo início)
        assert full.index.equals(idx)
        assert all(c.startswith(("news_", "black_")) for c in full.columns)


def test_event_blackout_and_news_gate():
    df = walk(200, seed=4)
    idx = pd.DatetimeIndex(df["ts"] + TF)
    df["news_minutes_to_next_event"] = 30.0  # sempre 30 min antes de um evento
    ev = np.full(len(df), np.nan)
    ev[60::20] = 1
    blocked = run_daytrade_backtest(genome(event_blackout=60), df, "X", "15m", df["ts"].iloc[50], events=ev, funding={})
    free = run_daytrade_backtest(genome(event_blackout=10), df, "X", "15m", df["ts"].iloc[50], events=ev, funding={})
    assert blocked.n_trades == 0 and blocked.blackout_bars > 0 and free.n_trades > 0
    df2 = walk(200, seed=4)
    df2["news_sentiment"] = -0.9  # sentimento muito negativo: bloqueia compras
    gated = run_daytrade_backtest(genome(news_gate=0.5), df2, "X", "15m", df2["ts"].iloc[50], events=ev, funding={})
    assert gated.n_trades == 0 and gated.blocked_by_news > 0


def test_black_weight_zero_ignores_black():
    frame = pd.DataFrame({"news_sentiment": [0.1, 0.1], "black_mean_sent": [-1.0, 1.0]})
    assert np.allclose(news_view(frame, 0.0)["sentiment"], [0.1, 0.1])
    assert np.allclose(news_view(frame, 0.5)["sentiment"], [-0.4, 0.6])


# estilos

def test_style_switch_keeps_strategy_eligible():
    rng = random.Random(0)
    for _ in range(100):
        g = random_day_genome(rng, styles=list(STYLES))
        s = rng.choice(list(STYLES))
        g2 = switch_style(g, s, rng)
        assert g2.style == s and g2.strategy in eligible(s)
        assert DayGenome.from_dict(g2.to_dict()) == g2
    assert STYLES["scalping"]["timeframe"] == "1m"
    assert random_day_genome(rng, style="scalping").order_type == "limit"
    assert random_day_genome(rng, strategy="vol_expansion", style="expansion").order_type == "stop"


# Criador e população com estilos

from trader.daytrade.creator import DayRuleCreator, diagnose_day
from trader.daytrade.evolution import DayConfig, DayPopulation
from trader.daytrade.market import SymbolSpec
from trader.evolution.fitness import SelectionRules
from trader.evolution.registry import Registry


def test_rule_creator_learns_to_stand_aside_where_it_lost():
    cr = DayRuleCreator()
    cr.styles = ["momentum"]
    g = DayGenome("momentum", {"lookback": 5, "threshold_bps": 30}, style="momentum")
    autopsy = {"cause": "drawdown", "life": {"pnl_by_regime": {"extreme": -5.0, "normal": 1.0}}, "last_window": {}}
    hits = 0
    for seed in range(20):
        new, why = cr.successor(g, autopsy, random.Random(seed))
        if new.regime_map and new.regime_map["extreme"] == "stand_aside":
            hits += 1
            assert "extreme" in why
    assert hits >= 8
    assert "extreme" in diagnose_day(autopsy)


def test_black_weight_forced_zero_when_not_allowed():
    cr = DayRuleCreator()
    cr.styles, cr.black_allowed = ["any"], False
    g = DayGenome("momentum", {"lookback": 5, "threshold_bps": 30}, black_weight=0.8)
    new, _ = cr.successor(g, {"cause": "negative_edge", "life": {}}, random.Random(1))
    assert new.black_weight == 0.0


def test_population_with_styles_uses_loader_for_1m():
    rng = np.random.default_rng(11)
    n15 = 96 * 20
    df15 = walk(n15, seed=11)
    calls = []

    def loader(timeframe, start, end):
        calls.append(timeframe)
        idx = pd.date_range(start.floor("min"), end, freq="1min", tz="UTC", inclusive="left")
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(idx))))
        o = np.concatenate([[c[0]], c[:-1]])
        return pd.DataFrame({"ts": idx, "open": o, "high": np.maximum(o, c) * 1.0005,
                             "low": np.minimum(o, c) * 0.9995, "close": c, "volume": rng.lognormal(0, 0.5, len(idx))})

    cfg = DayConfig(initial=6, max_slots=8, styles=("scalping", "momentum", "expansion"), cash=18.18,
                    spec=SymbolSpec(5.0, 0.01), rules=SelectionRules(min_trades=2, min_age_for_edge=2, max_age=3),
                    news=True)
    pop = DayPopulation(df15, "X", "15m", Registry(":memory:"), cfg, seed=3, loader=loader,
                        providers=[stub_white(0)])
    pop.seed()
    styles = {a.genome.style for a in pop.alive}
    for d in range(5, 17, 3):
        pop.step(T0 + pd.Timedelta(days=d), T0 + pd.Timedelta(days=d + 3))
    if "scalping" in styles:
        assert "1m" in calls
    assert any(k.startswith("style_born:") for k in pop.totals)
    assert sum(v for k, v in pop.totals.items() if k.startswith("regime_bars:")) > 0


def test_signal_check_candidates_and_price_only_strip():
    from trader.daytrade.signal_check import Example, candidates, strip_context

    g = DayGenome("momentum", {"lookback": 5, "threshold_bps": 30}, regime_map={r: "momentum" for r in REGIMES},
                  news_gate=0.5, black_weight=0.4)
    assert strip_context(g).regime_map is None and strip_context(g).black_weight == 0
    ex = Example(g, g, {"cause": "drawdown", "life": {"pnl_by_regime": {"extreme": -1.0}}, "last_window": {}},
                 [(T0, T0 + pd.Timedelta(days=15))], "k")
    c = candidates(ex, price_only=True)
    assert set(c) == {"successor", "rules", "parent", "baseline", "mutation0", "mutation1"}
    assert all(v.regime_map is None and v.news_gate is None for v in c.values())
    assert candidates(ex, price_only=True)["rules"] == c["rules"]  # determinístico por chave
