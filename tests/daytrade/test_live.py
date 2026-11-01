import numpy as np
import pandas as pd
import pytest

import trader.daytrade.live as live_mod
from trader.daytrade.creator import DayRuleCreator
from trader.daytrade.evolution import DayConfig, DayPopulation
from trader.daytrade.live import DayLiveConfig, DayLiveEvolution
from trader.daytrade.market import NO_MINIMUM
from trader.evolution.fitness import SelectionRules
from trader.evolution.registry import Registry
from trader.storage.candles import CandleStore

T0 = pd.Timestamp("2026-09-01", tz="UTC")


def candles(n, start=T0, freq="5min", seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.concatenate([[100.0], c[:-1]])
    return pd.DataFrame({"ts": pd.date_range(start, periods=n, freq=freq, tz="UTC"), "open": o,
                         "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
                         "volume": rng.lognormal(0, 0.5, n)})


def test_restore_roundtrip(tmp_path):
    df = {"X": candles(288 * 6), "Y": candles(288 * 6, seed=1)}
    cfg = DayConfig(initial=6, max_slots=8, styles=("momentum", "mean_reversion"), spec=NO_MINIMUM,
                    rules=SelectionRules(min_trades=2, min_age_for_edge=2, max_age=3))
    db = tmp_path / "live.sqlite"
    pop = DayPopulation(df, "X", "5m", Registry(db), cfg, seed=0)
    pop.seed()
    for h in range(0, 18, 3):
        pop.step(T0 + pd.Timedelta(days=2, hours=h), T0 + pd.Timedelta(days=2, hours=h + 3))
    again = DayPopulation(df, "X", "5m", Registry(db), cfg, seed=0)
    assert again.restore()
    assert [a.id for a in again.alive] == [a.id for a in pop.alive]
    assert [a.genome for a in again.alive] == [a.genome for a in pop.alive]
    assert [a.life.age for a in again.alive] == [a.life.age for a in pop.alive]
    assert [a.life.ret for a in again.alive] == pytest.approx([a.life.ret for a in pop.alive])
    assert again._window == pop._window and again._counter == pop._counter
    assert len(again.baseline_windows) == len(pop.baseline_windows)
    again.step(T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=3, hours=3))  # continua sem colidir ids


class FakeFeed:
    """Candles que 'chegam' até o instante now()."""

    def __init__(self, full):
        self.full, self.now = full, T0

    def fetch(self, store, symbol, timeframe, start):
        df = self.full[symbol]
        tf = pd.Timedelta(timeframe.replace("m", "min"))
        closed = df[df["ts"] + tf <= self.now]
        store.write("binanceusdm", symbol, timeframe, closed)
        return len(closed)


def test_only_closed_windows_run(tmp_path):
    full = {s: candles(288 * 3, start=T0 - pd.Timedelta(days=2), seed=i) for i, s in enumerate(("SOL/USDT", "XRP/USDT"))}
    feed = FakeFeed(full)
    feed.now = T0 + pd.Timedelta(hours=1)
    config = DayLiveConfig(style="day", symbol="SOL/USDT", extra_symbols=("XRP/USDT",), llm=False, population=4, slots=6)
    live = DayLiveEvolution(tmp_path, tmp_path / "l.sqlite", config, creator=DayRuleCreator(), fetch=feed.fetch,
                            now=lambda: feed.now, log=lambda m: None)
    live.start()
    assert pd.Timestamp(live.registry.meta()["next_start"]) == T0  # janelas de 3h alinhadas
    assert live.tick() == 0  # janela 00h-03h ainda aberta
    feed.now = T0 + pd.Timedelta(hours=3, minutes=2)
    # XRP atrasado: só chega até 02:55 → nada roda
    full["XRP/USDT"] = full["XRP/USDT"][full["XRP/USDT"]["ts"] < T0 + pd.Timedelta(hours=2, minutes=50)]
    CandleStore(tmp_path)  # noop
    assert live.tick() == 0
    full["XRP/USDT"] = candles(288 * 3, start=T0 - pd.Timedelta(days=2), seed=1)
    assert live.tick() == 1
    assert pd.Timestamp(live.registry.meta()["next_start"]) == T0 + pd.Timedelta(hours=3)
    assert live.tick() == 0  # nada novo


def test_no_real_orders_guard(tmp_path, monkeypatch):
    assert live_mod.REAL_ORDERS_ENABLED is False
    monkeypatch.setattr(live_mod, "REAL_ORDERS_ENABLED", True)
    with pytest.raises(AssertionError):
        DayLiveEvolution(tmp_path, tmp_path / "x.sqlite", DayLiveConfig(llm=False), creator=DayRuleCreator())
    import trader.daytrade as pkg
    import pkgutil

    for mod in pkgutil.iter_modules(pkg.__path__):
        src = open(f"{pkg.__path__[0]}/{mod.name}.py", encoding="utf-8").read()
        assert "create_order" not in src and "apiKey" not in src, mod.name  # nenhum caminho para ordem real
