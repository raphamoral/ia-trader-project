import datetime as dt
import random

import numpy as np
import pandas as pd
import pytest

from trader.daytrade.backtest import run_daytrade_backtest
from trader.daytrade.broker import FuturesPaperBroker, InsufficientMargin, liquidation_price
from trader.daytrade.evolution import DayConfig, DayPopulation, DayRuleCreator
from trader.daytrade.gate import Close, DaySignal, DayTradeLimits, DayTradeRiskGate, Open
from trader.daytrade.genome import DAY_STRATEGIES, DayGenome, mutate_day, random_day_genome, signals
from trader.daytrade.backtest import goal_metrics
from trader.daytrade.creator import DayLLMCreator, diagnose_day
from trader.daytrade.market import SPECS, SymbolSpec, feasibility
from trader.daytrade.features import news_view, stub_white
from trader.evolution.fitness import SelectionRules
from trader.evolution.registry import Registry

T0 = pd.Timestamp("2024-01-01", tz="UTC")
TF = pd.Timedelta(minutes=15)


def bars(prices, start=T0, spread=0.0):
    """Candles de 15m: open = close anterior, high/low = ±spread em volta."""
    close = np.asarray(prices, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=len(close), freq="15min", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread), "close": close, "volume": 1.0,
    })


# corretora

def test_liquidation_price_long_and_short():
    # 10x, mmr 0,4%: comprado liquida ~9,6% abaixo, vendido ~9,6% acima
    assert liquidation_price(1, 100, 10, 0.004) == pytest.approx(100 * 0.9 / 0.996)
    assert liquidation_price(-1, 100, 10, 0.004) == pytest.approx(100 * 1.1 / 1.004)
    # na liquidação, margem + PnL = manutenção
    p = liquidation_price(1, 100, 5, 0.004)
    assert (p - 100) + 100 / 5 == pytest.approx(0.004 * p)


def test_short_profits_when_price_falls():
    b = FuturesPaperBroker(1000, taker_fee=0.0005)
    b.open("X", -1, 2, 100, leverage=2, stop=105, ts=T0)
    pnl = b.close("X", 90, T0)
    assert pnl == pytest.approx(2 * 10 - 2 * 90 * 0.0005)
    assert b.cash == pytest.approx(1000 - 200 * 0.0005 + pnl)


def test_margin_and_fee_are_reserved():
    b = FuturesPaperBroker(100, taker_fee=0.001)
    with pytest.raises(InsufficientMargin):
        b.open("X", 1, 3, 100, leverage=3, stop=95, ts=T0)  # margem 100 + taxa 0,3 > 100


def test_stop_must_be_before_liquidation():
    b = FuturesPaperBroker(1000)
    with pytest.raises(ValueError):
        b.open("X", 1, 1, 100, leverage=10, stop=85, ts=T0)  # liquidação ~90,4 vem antes


def test_funding_long_pays_short_receives():
    b = FuturesPaperBroker(1000, taker_fee=0)
    b.open("X", 1, 1, 100, leverage=1, stop=50, ts=T0)
    assert b.apply_funding("X", 0.001, 100) == pytest.approx(0.1)
    b.close("X", 100, T0)
    b.open("X", -1, 1, 100, leverage=1, stop=150, ts=T0)
    assert b.apply_funding("X", 0.001, 100) == pytest.approx(-0.1)
    assert b.funding_paid == pytest.approx(0.0)


def test_gap_beyond_liquidation_loses_only_margin():
    b = FuturesPaperBroker(1000, taker_fee=0)
    b.open("X", 1, 10, 100, leverage=5, stop=90, ts=T0)  # margem 200
    b.close("X", 50, T0)  # gap absurdo
    assert b.cash == pytest.approx(800)


# gate

LIM = DayTradeLimits(max_leverage=3, max_risk_per_trade=0.02)


def test_gate_caps_leverage_and_sizes_by_risk():
    g = DayTradeRiskGate(LIM)
    acts = g.actions(DaySignal(1, leverage=20, stop=0.001, risk=0.05), 100, 1000, 0, T0 + pd.Timedelta(hours=1), TF)
    (o,) = acts
    assert o.leverage == 3
    assert o.qty * 100 <= 1000 * 3  # nocional limitado pela alavancagem máxima
    # stop largo: tamanho vem do risco (2% de 1000 / 4% = 500 de nocional)
    (o,) = g.actions(DaySignal(-1, leverage=3, stop=0.04, risk=0.05), 100, 1000, 0, T0 + pd.Timedelta(hours=1), TF)
    assert o.side == -1 and o.qty * 100 == pytest.approx(500)


def test_gate_lowers_leverage_when_stop_would_pass_liquidation():
    g = DayTradeRiskGate(DayTradeLimits(max_leverage=10, max_stop=0.2))
    (o,) = g.actions(DaySignal(1, leverage=10, stop=0.15, risk=0.01), 100, 1000, 0, T0 + pd.Timedelta(hours=1), TF)
    assert 1 / o.leverage - 0.004 - 0.15 >= 0.005 - 1e-9
    assert o.leverage < 10


def test_gate_forces_flat_before_session_end_and_blocks_entries():
    g = DayTradeRiskGate(LIM)
    late = pd.Timestamp("2024-01-01 23:45", tz="UTC")  # o próximo candle fecha 00:00
    assert g.actions(DaySignal(None), 100, 1000, 1, late, TF) == [Close("eod")]
    assert g.actions(DaySignal(1), 100, 1000, 0, late, TF) == []
    assert g.actions(DaySignal(1), 100, 1000, 0, pd.Timestamp("2024-01-02 00:15", tz="UTC"), TF)


def test_gate_reverse_and_daily_loss():
    g = DayTradeRiskGate(LIM)
    now = T0 + pd.Timedelta(hours=2)
    acts = g.actions(DaySignal(-1), 100, 1000, 1, now, TF)
    assert isinstance(acts[0], Close) and isinstance(acts[1], Open) and acts[1].side == -1
    assert g.actions(DaySignal(1), 100, 940, 0, now + TF, TF) == []  # perdeu 6% > 5%
    g.kill()


# backtest

def test_backtest_stop_hit_intrabar_is_pessimistic():
    # sinal de compra no candle 1; no candle 3 a mínima fura o stop E a máxima passa do alvo
    prices = [100] * 4 + [100]
    df = bars(prices)
    df.loc[3, ["high", "low"]] = [110, 90]
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02, take_profit=2.0, risk=0.01)
    res = run_daytrade_backtest(g, df, "X", "15m", start=df["ts"].iloc[0], events=ev, funding={})
    actions = [t.action for t in res.trades]
    assert actions == ["open", "stop"]
    assert res.trades[1].price == pytest.approx(98)


def test_backtest_liquidates_on_gap():
    df = bars([100, 100, 100, 60, 60])
    df.loc[3, ["open", "high"]] = [60, 60]  # abre direto abaixo da liquidação (~67)
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, leverage=3, stop=0.02, risk=0.02)
    res = run_daytrade_backtest(g, df, "X", "15m", start=df["ts"].iloc[0], events=ev,
                                limits=DayTradeLimits(max_leverage=3), funding={})
    assert res.liquidations == 1
    assert res.total_return < 0


def test_backtest_flat_at_session_end():
    start = pd.Timestamp("2024-01-01 22:00", tz="UTC")
    df = bars(np.linspace(100, 101, 16), start=start)
    ev = np.full(len(df), np.nan)
    ev[0] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02)
    res = run_daytrade_backtest(g, df, "X", "15m", start=start, events=ev, funding={})
    eod = [t for t in res.trades if t.action == "eod"]
    assert len(eod) == 1 and eod[0].ts == pd.Timestamp("2024-01-01 23:45", tz="UTC")


def test_funding_charged_at_funding_hours():
    start = pd.Timestamp("2024-01-01 07:00", tz="UTC")
    df = bars([100] * 8, start=start)
    ev = np.full(len(df), np.nan)
    ev[0] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02, risk=0.02)
    f = {pd.Timestamp("2024-01-01 08:00", tz="UTC"): 0.01}
    res = run_daytrade_backtest(g, df, "X", "15m", start=start, events=ev, funding=f)
    assert res.funding > 0


# genoma e sinais

@pytest.mark.parametrize("strategy", sorted(DAY_STRATEGIES))
def test_every_strategy_is_causal_and_fires(strategy):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3000)))
    df = bars(close, spread=0.002)
    df["open"] = df["open"] * (1 + rng.normal(0, 0.001, len(df)))  # corpos variados para os padrões
    body_hi, body_lo = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
    df["high"] = body_hi * (1 + np.abs(rng.normal(0, 0.004, len(df))))  # pavios aleatórios
    df["low"] = body_lo * (1 - np.abs(rng.normal(0, 0.004, len(df))))
    df["volume"] = rng.lognormal(0, 0.8, len(df))
    if strategy in ("xs_momentum", "funding_reversal"):  # estratégias de mercado leem as colunas mkt_
        for lb in (12, 48, 288):
            df[f"mkt_xs_rank_{lb}"] = np.clip(0.5 + 0.5 * np.sin(np.arange(len(df)) / (7 + lb / 12)), 0, 1)
        df["mkt_funding_bps"] = np.repeat(rng.normal(0, 4, len(df) // 96 + 1), 96)[: len(df)]
        df["mkt_oi_z"] = rng.normal(0, 1.5, len(df))
    lo_hi = DAY_STRATEGIES[strategy]
    g = DayGenome(strategy, {k: lo + (hi - lo) // 8 for k, (lo, hi) in lo_hi.items()}, news_follow=0.3).normalized()
    idx = pd.DatetimeIndex(df["ts"] + TF)
    news_full = news_view(stub_white(1)(None, "15m", idx), 0.0)

    def run(n):
        news = {k: v[:n] for k, v in news_full.items()}
        return signals(df.iloc[:n], g, news=news)

    full = run(len(df))
    for cut in (1500, 2222):
        np.testing.assert_array_equal(full[:cut], run(cut), err_msg=f"{strategy} olha o futuro")
    assert np.isin(full[np.isfinite(full)], [-1, 0, 1]).all()
    if strategy != "grid":  # o grid é conduzido pelo backtest, não por eventos
        assert (np.abs(full) == 1).sum() > 0, f"{strategy} nunca gera entrada"


def test_direction_filter():
    rng = np.random.default_rng(1)
    df = bars(100 * np.exp(np.cumsum(rng.normal(0, 0.004, 500))))
    ev = signals(df, DayGenome("momentum", {"lookback": 4, "threshold_bps": 20}, direction="long"))
    assert not (ev == -1).any()


def test_genomes_stay_valid():
    rng = random.Random(0)
    for _ in range(200):
        g = mutate_day(random_day_genome(rng), rng, 1.0)
        assert DayGenome.from_dict(g.to_dict()) == g


def test_liquidation_successor_halves_leverage():
    g = DayGenome("orb", {"range_bars": 4}, leverage=4)
    s, why = DayRuleCreator().successor(g, {"cause": "liquidation"}, random.Random(0))
    assert s.leverage < 3 and "alavancagem" in why


def test_population_runs_and_keeps_slots():
    rng = np.random.default_rng(3)
    df = bars(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 96 * 40))), spread=0.001)
    reg = Registry(":memory:")
    cfg = DayConfig(initial=6, max_slots=10, rules=SelectionRules(min_trades=3, max_age=3))
    pop = DayPopulation(df, "X", "15m", reg, cfg, funding=None, seed=0)
    pop.seed()
    for d in range(3, 38, 5):
        r = pop.step(T0 + pd.Timedelta(days=d), T0 + pd.Timedelta(days=d + 5))
        assert 6 <= r["alive"] <= 10
    assert sum(reg.cause_counts().values()) > 0


# ordem mínima, ruína, autonomia diária, escalada, meta, auditoria


def test_btc_minimum_is_the_lot_not_the_notional():
    btc = SPECS["BTC/USDT"]
    assert btc.min_order_value(83_000) == pytest.approx(83.0)  # 0,001 BTC > 50 USDT
    assert btc.min_order_value(20_000) == pytest.approx(50.0, abs=20)  # aí a regra de 50 USDT domina
    ok, msg = feasibility("BTC/USDT", 100 / 5.5, 3, 83_000)
    assert not ok and "INVIÁVEL" in msg
    assert feasibility("SOL/USDT", 100 / 5.5, 3, 118)[0]


def test_gate_bumps_to_minimum_or_refuses():
    spec = SymbolSpec(5.0, 0.01)
    g = DayTradeRiskGate(DayTradeLimits(max_leverage=3, max_risk_per_trade=0.02), spec)
    now = T0 + pd.Timedelta(hours=1)
    # risco pediria 18*0,001/0,01 = 1,8 USDT < 5: sobe para 5 (risco 0,05 USDT ≤ 2% de 18)
    (o,) = g.actions(DaySignal(1, 3, 0.01, 0.001), 100, 18.0, 0, now, TF)
    assert o.qty * 100 == pytest.approx(5.0)
    # com 1 USDT nem 3x cobre a ordem mínima
    assert g.actions(DaySignal(1, 3, 0.01, 0.02), 100, 1.0, 0, now, TF) == []
    assert g.ruined(1.0, 100) and not g.ruined(18.0, 100)


def test_ruin_stops_the_account():
    df = bars([100.0] * 12 + [70.0] * 12)
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, leverage=3, stop=0.02, risk=0.1)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], cash=2.0, events=ev, funding={},
                                limits=DayTradeLimits(max_leverage=3, max_risk_per_trade=0.1, max_stop=0.3),
                                spec=SymbolSpec(5.0, 0.01))
    assert res.ruined and res.day_log[-1]["end"] == "ruin"
    assert len(res.equity) == len(df)


def test_agent_daily_stop_target_and_max_trades():
    # sobe forte: meta diária de 1% atingida, agente para no dia
    df = bars(np.linspace(100, 110, 40))
    ev = np.full(len(df), np.nan)
    ev[1::3] = 1
    ev[2::3] = 0
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02, risk=0.02, daily_target=0.01)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    assert res.target_days == 1 and res.day_log[0]["end"] == "daily_target"
    entries_after = [t for t in res.trades if t.action == "open" and t.ts > res.trades[-1].ts]
    assert not entries_after
    # no máximo 2 tentativas por dia
    g2 = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02, max_trades_per_day=2)
    res2 = run_daytrade_backtest(g2, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    assert res2.n_trades == 2 and res2.day_log[0]["attempts"] == 2


def test_agent_daily_stop_is_capped_by_gate():
    df = bars(np.linspace(100, 80, 60))
    ev = np.full(len(df), np.nan)
    ev[1::2] = 1
    ev[2::2] = 0
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02, risk=0.02, daily_stop=0.5)
    res = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={},
                                limits=DayTradeLimits(daily_loss_limit=0.05))
    assert res.equity.min() >= 10_000 * (1 - 0.05 * 0.9) - 10_000 * 0.02 * 1.5  # para no stop do agente (4,5%)
    assert res.day_log[0]["end"] == "daily_stop"


def test_escalation_after_wins_grows_size_up_to_gate():
    df = bars(np.linspace(100, 120, 80))
    ev = np.full(len(df), np.nan)
    ev[1::4] = 1
    ev[3::4] = 0
    base = dict(strategy="momentum", params={"lookback": 3, "threshold_bps": 10}, stop=0.01, risk=0.005)
    flat_ = run_daytrade_backtest(DayGenome(**base), df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    esc = run_daytrade_backtest(DayGenome(**base, after_win_mult=2.0), df, "X", "15m", df["ts"].iloc[0],
                                events=ev, funding={})
    opens_flat = [t.qty for t in flat_.trades if t.action == "open"]
    opens_esc = [t.qty for t in esc.trades if t.action == "open"]
    assert opens_esc[2] > opens_flat[2] * 1.5
    assert esc.day_log[0]["max_mult"] > 1 and esc.day_log[0]["max_risk"] <= DayTradeLimits().max_risk_per_trade


def test_goal_metrics():
    daily = np.array([1.0, -0.5, 0.2, 1.5])
    g = goal_metrics(daily, 100)
    assert g["hits"] == 2 and g["best_day"] == 1.5
    assert g["days_to_income"] > 0
    assert goal_metrics(np.array([-0.01, 0.0]), 100)["days_to_income"] == float("inf")
    # 10%/dia: R$1000/dia exige R$10.000 → ln(100)/ln(1,1) ≈ 48 dias
    assert goal_metrics(np.full(5, 0.1), 100)["days_to_income"] == pytest.approx(np.log(100) / np.log(1.1))


def test_delay_shifts_signals_one_bar():
    df = bars(np.linspace(100, 101, 20))
    ev = np.full(len(df), np.nan)
    ev[2] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.02)
    a = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    b = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={}, delay=1)
    assert b.trades[0].ts - a.trades[0].ts == TF


def test_day_llm_creator_parses_and_falls_back():
    import json

    answer = {"genome": {"strategy": "supertrend", "params": {"period": 10, "mult10": 30}, "direction": "long",
                         "leverage": 2, "stop": 0.01, "take_profit": None, "risk": 0.01, "cooldown": 0,
                         "max_trades_per_day": 3, "daily_stop": 0.03, "daily_target": 1.0, "max_hold": 0,
                         "after_win_mult": 1.5, "after_loss_mult": 0.7, "goal_pressure": 1.0}, "why": "x"}
    ctx = {"capital": "R$100", "max_lev": "3", "max_risk": "0.02", "daily_loss": "0.05", "min_order": "5", "goal": "2x"}
    ok = DayLLMCreator("SOL/USDT", "15m", ctx, post=lambda u, p, t: {"choices": [{"message": {"content": json.dumps(answer)}}]})
    g0 = DayGenome("orb", {"range_bars": 4})
    autopsy = {"cause": "ruin", "life": {"stop_day_share": 0.5}}
    g, why = ok.successor(g0, autopsy, random.Random(0))
    assert g.strategy == "supertrend" and g.after_win_mult == 1.5 and why.startswith("LLM")
    assert "supertrend" in ok.prompt(autopsy) and "RUINED" in diagnose_day(autopsy)

    def down(u, p, t):
        raise ConnectionError
    bad = DayLLMCreator("SOL/USDT", "15m", ctx, post=down)
    g, why = bad.successor(g0, autopsy, random.Random(0))
    assert bad.failures == 1 and "arruinado" in why


def test_goal_fitness_population_runs():
    rng = np.random.default_rng(5)
    df = bars(100 * np.exp(np.cumsum(rng.normal(0, 0.004, 96 * 30))), spread=0.002)
    reg = Registry(":memory:")
    cfg = DayConfig(initial=6, max_slots=10, rules=SelectionRules(min_trades=2, min_age_for_edge=2, max_age=3),
                    fitness="goal", cash=18.18, spec=SymbolSpec(5.0, 0.01))
    pop = DayPopulation(df, "X", "15m", reg, cfg, seed=1)
    pop.seed()
    for d in range(4, 29, 4):
        pop.step(T0 + pd.Timedelta(days=d), T0 + pd.Timedelta(days=d + 4))
    assert pop.totals["agent_days"] > 0
    assert "no_goal" in reg.cause_counts() or "ruin" in reg.cause_counts()


# escada e Kelly

from trader.daytrade.backtest import KellyCap
from trader.daytrade.evolution import LADDER_BRL, DayAgent, LadderRules, daily_curve_stats


def test_kelly_cap_shrinks_with_few_trades_and_needs_edge():
    k = KellyCap(fraction=0.5, prior_n=30, probe=0.002)
    assert k.cap([]) == 0.002  # sem histórico: só sondagem
    assert k.cap([-1.0] * 50) == 0.002  # sem vantagem: nada além da sondagem
    wins = [2.0, -1.0] * 5  # média 0,5R em 10 operações
    many = [2.0, -1.0] * 200
    assert k.cap(wins) < k.cap(many)  # poucas operações: encolhe
    # muitas operações: ~ meio Kelly de E[r]/E[r²] = 0,5 / 2,5 = 0,2 → 0,1
    assert k.cap(many) == pytest.approx(0.1, abs=0.01)


def test_backtest_respects_kelly_cap():
    df = bars(np.linspace(100, 101, 30))
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, leverage=3, stop=0.01, risk=0.02)
    free = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    capped = run_daytrade_backtest(g, df, "X", "15m", df["ts"].iloc[0], events=ev, funding={}, kelly=KellyCap())
    assert capped.trades[0].qty == pytest.approx(free.trades[0].qty * 0.002 / 0.02)


def test_daily_curve_stats():
    ret, dd = daily_curve_stats([0.1, -0.5, 0.2])
    assert ret == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert dd == pytest.approx(0.5)


def test_ladder_promotion_requires_consistency():
    reg = Registry(":memory:")
    df = bars(np.linspace(100, 101, 96 * 3))
    cfg = DayConfig(fitness="ladder", ladder=LadderRules(min_days=10, min_profitable_share=0.6, max_drawdown=0.1))
    pop = DayPopulation(df, "X", "15m", reg, cfg)
    a = DayAgent("x", DayGenome("orb", {"range_bars": 4}), "x")
    a.rung_daily, a.rung_base_daily = [0.01] * 12, [0.0] * 12
    assert pop._promotable(a)
    a.rung_daily = [0.01] * 5  # poucos dias
    assert not pop._promotable(a)
    a.rung_daily = [0.02, -0.01] * 6  # só 50% de dias positivos
    assert not pop._promotable(a)
    a.rung_daily = [0.01] * 12
    pop._promote(a)
    assert a.rung == 1 and a.rung_daily == [] and pop.cash_for(a) == pytest.approx(LADDER_BRL[1] / 5.5)


def test_ladder_population_runs():
    rng = np.random.default_rng(9)
    df = bars(100 * np.exp(np.cumsum(rng.normal(0, 0.003, 96 * 30))), spread=0.002)
    reg = Registry(":memory:")
    cfg = DayConfig(initial=6, max_slots=10, rules=SelectionRules(min_trades=2, min_age_for_edge=2, max_age=4),
                    fitness="ladder", cash=18.18, spec=SymbolSpec(5.0, 0.01),
                    ladder=LadderRules(min_days=8, min_profitable_share=0.3, max_drawdown=0.3))
    pop = DayPopulation(df, "X", "15m", reg, cfg, seed=2)
    pop.seed()
    for d in range(4, 29, 4):
        pop.step(T0 + pd.Timedelta(days=d), T0 + pd.Timedelta(days=d + 4))
    assert sum(pop.rung_census().values()) == len(pop.alive)
    assert all(a.r_history is not None for a in pop.alive)


def test_day_creator_prompt_includes_playbook_and_passages():
    from trader.daytrade.creator import DayLLMCreator

    c = DayLLMCreator("SOL/USDT", "5m", {"capital": "R$100", "max_lev": "3", "max_risk": "0.02", "daily_loss": "0.05",
                                          "min_order": "5", "goal": "consistency"},
                      lessons=lambda: ["Use RSI 14 with 30/70 [Livro p.1]"],
                      library=lambda q: [{"book": "Livro", "page": 7, "text": "cut losses early"}])
    p = c.prompt({"cause": "fees", "life": {}, "last_window": {}})
    assert "Playbook from trading books" in p and "RSI 14" in p and "[Livro p.7]" in p
