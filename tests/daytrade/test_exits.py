import random

import numpy as np
import pytest

from trader.daytrade.backtest import exit_stop, run_daytrade_backtest
from trader.daytrade.broker import BINANCE_USDM_TAKER
from trader.daytrade.genome import DayGenome, mutate_day, random_day_genome

from tests.daytrade.test_styles_regimes_news import genome, ohlc

FEE = 0.0005


def test_lock_waits_for_the_gain():
    g = genome(stop=0.02, lock_at=1.0, lock_keep=0.4)
    # ganho de 1,5 < 1R (2): nada muda
    assert exit_stop(g, 1, 100, 0.02, 101.5, 101.5, np.nan, np.nan, FEE, False) == (None, False)


def test_lock_keeps_part_of_the_gain():
    # o exemplo do usuário em escala: entrou a 100, chegou a 125 com R = 20 → garante 40% de 25 = 110
    g = DayGenome("momentum", {"lookback": 3, "threshold_bps": 10}, stop=0.05, lock_at=1.0, lock_keep=0.4)
    new, _ = exit_stop(g, 1, 100, 0.2, 125, 124, np.nan, np.nan, FEE, False)
    assert new == pytest.approx(110)
    # vendido: espelhado
    new, _ = exit_stop(g, -1, 100, 0.2, 75, 76, np.nan, np.nan, FEE, False)
    assert new == pytest.approx(90)


def test_lock_never_below_breakeven_with_fees():
    g = genome(stop=0.01, lock_at=1.0, lock_keep=0.0)
    new, _ = exit_stop(g, 1, 100, 0.01, 101, 101, np.nan, np.nan, FEE, False)
    assert new == pytest.approx(100 * (1 + 2 * FEE))


def test_runner_turns_on_only_in_profit_and_high_vol():
    g = genome(runner_vol=0.8, runner_atr=2.0)
    assert exit_stop(g, 1, 100, 0.02, 103, 103, 0.5, 0.5, FEE, False) == (None, False)  # vol normal
    assert exit_stop(g, 1, 100, 0.02, 100, 99, 0.5, 0.95, FEE, False) == (None, False)  # no prejuízo
    new, on = exit_stop(g, 1, 100, 0.005, 103, 103, 0.5, 0.95, FEE, False)
    assert on and new == pytest.approx(102)
    # depois de ligado continua mesmo se a volatilidade cair
    new, on = exit_stop(g, 1, 100, 0.005, 105, 105, 0.5, 0.3, FEE, True)
    assert on and new == pytest.approx(104)
    # ATR ainda pequeno no começo da explosão: a distância não fica menor que o stop inicial (2% de 100)
    new, _ = exit_stop(g, 1, 100, 0.02, 103, 103, 0.1, 0.95, FEE, False)
    assert new == pytest.approx(101)


def test_backtest_lock_turns_loser_into_small_win():
    # entra a 100 (stop 2% = 98, alvo 2,5R = 105), sobe a 102,5, devolve tudo até 99
    c = [100, 100, 101, 102.5, 101.5, 100.5, 99, 97]
    o = [100] + c[:-1]
    h = [max(a, b) + 0.1 for a, b in zip(o, c)]
    lo = [min(a, b) - 0.1 for a, b in zip(o, c)]
    df = ohlc(o, h, lo, c)
    ev = np.full(len(df), np.nan)
    ev[1] = 1
    base = dict(stop=0.02, take_profit=2.5, direction="long")
    plain = run_daytrade_backtest(genome(**base), df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    locked = run_daytrade_backtest(genome(**base, lock_at=1.0, lock_keep=0.4), df, "X", "15m",
                                   df["ts"].iloc[0], events=ev, funding={})
    exit_plain = [t for t in plain.trades if t.action != "open"][0]
    exit_locked = [t for t in locked.trades if t.action != "open"][0]
    assert exit_plain.pnl < 0
    assert exit_locked.action == "stop" and exit_locked.pnl > 0
    # stop travado = 100 + 0,4 × (102,6 − 100) ≈ 101,04
    assert exit_locked.price == pytest.approx(100 + 0.4 * 2.6, rel=1e-3)


def test_backtest_runner_drops_target():
    # volatilidade cresce a favor: com o alvo saria em 104; deixando correr, sai bem mais alto
    rng = np.random.default_rng(0)
    calm = list(100 + rng.normal(0, 0.05, 80).cumsum() * 0)  # parado: ATR baixo
    run = [100 + 1.5 * k for k in range(1, 14)]  # candles grandes a favor
    back = [run[-1] - 3, run[-1] - 8]
    c = calm + run + back
    o = [100] + c[:-1]
    h = [max(a, b) + (0.05 if i < 80 else 0.8) for i, (a, b) in enumerate(zip(o, c))]
    lo = [min(a, b) - (0.05 if i < 80 else 0.8) for i, (a, b) in enumerate(zip(o, c))]
    df = ohlc(o, h, lo, c)
    ev = np.full(len(df), np.nan)
    ev[79] = 1
    base = dict(stop=0.02, take_profit=2.0, direction="long", leverage=1)
    plain = run_daytrade_backtest(genome(**base), df, "X", "15m", df["ts"].iloc[0], events=ev, funding={})
    runner = run_daytrade_backtest(genome(**base, runner_vol=0.8, runner_atr=2.0), df, "X", "15m",
                                   df["ts"].iloc[0], events=ev, funding={})
    tp = [t for t in plain.trades if t.action == "take_profit"]
    out = [t for t in runner.trades if t.action != "open"]
    assert tp and not any(t.action == "take_profit" for t in out)
    assert out[0].pnl > tp[0].pnl


def test_exit_genes_roundtrip_and_mutate():
    rng = random.Random(1)
    for _ in range(50):
        g = mutate_day(random_day_genome(rng), rng, scale=0.6)
        assert DayGenome.from_dict(g.to_dict()) == g
        assert g.lock_at is None or 0.3 <= g.lock_at <= 5.0
        assert 0.0 <= g.lock_keep <= 0.9
        assert g.runner_vol is None or 0.5 <= g.runner_vol <= 0.99
