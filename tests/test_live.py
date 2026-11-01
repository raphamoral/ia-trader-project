import pandas as pd

from trader.evolution.creator import RuleCreator
from trader.evolution.population import Population, PopulationConfig
from trader.evolution.registry import Registry
from trader.live import runner
from trader.live.runner import LiveConfig, LiveEvolution
from tests.test_evolution import random_walk

T0 = pd.Timestamp("2024-01-01", tz="UTC")


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def make_live(tmp_path, clock, df):
    """Simula a exchange: a cada tick só existem candles fechados até o relógio."""
    def fetch(store, symbol, timeframe, since):
        visible = df[df["ts"] + pd.Timedelta("15min") <= clock()]
        store.write("binance", symbol, timeframe, visible)
        return len(visible)

    cfg = LiveConfig(timeframe="15m", window="3h", history_days=30, workers=1)
    return LiveEvolution(tmp_path / "data", tmp_path / "live.sqlite", cfg, creator=RuleCreator(),
                         fetch=fetch, now=clock, log=lambda m: None)


def test_real_orders_are_disabled():
    assert runner.REAL_ORDERS_ENABLED is False


def test_live_runs_only_closed_windows_and_resumes(tmp_path):
    df = random_walk(4000, seed=3)
    df["ts"] = pd.date_range(T0, periods=len(df), freq="15min", tz="UTC")
    clock = Clock(df["ts"].iloc[3000])
    live = make_live(tmp_path, clock, df)
    live.start()
    assert live.tick() == 0  # nenhuma janela de 3h fechou ainda

    clock.t += pd.Timedelta("7h")
    assert live.tick() == 2  # duas janelas de 3h fecharam
    alive_before = {a.id for a in live.population.alive}

    # reinício do processo: retoma a mesma população e a próxima janela
    clock.t += pd.Timedelta("3h")
    again = make_live(tmp_path, clock, df)
    again.start()
    assert {a.id for a in again.population.alive} == alive_before
    assert all(a.life.age == 2 for a in again.population.alive if a.id in alive_before)
    assert again.tick() == 1


def test_population_restore_roundtrip():
    df = random_walk(1500, seed=5)
    reg = Registry(":memory:")
    pop = Population(df, "X", "4h", reg, RuleCreator(), PopulationConfig(initial=6, max_slots=8), seed=2)
    pop.seed()
    start = df["ts"].iloc[300]
    pop.step(start, start + pd.Timedelta(days=30))
    restored = Population(df, "X", "4h", reg, RuleCreator(), PopulationConfig(initial=6, max_slots=8))
    assert restored.restore()
    assert [a.id for a in restored.alive] == [a.id for a in pop.alive]
    assert [a.life.ret for a in restored.alive] == [a.life.ret for a in pop.alive]
    assert restored._window == 1 and len(restored.baseline_windows) == 1


def test_news_genes_act_only_with_features():
    from trader.domain import MarketView, Side
    from trader.evolution.genome import Genome, GenomeEngine

    df = random_walk(200, seed=9)
    df["close"] = [100 + i for i in range(len(df))]  # tendência de alta: ma_cross quer comprar
    g = Genome("ma_cross", {"fast": 5, "slow": 20}, news_gate=0.3, black_weight=0.5)
    as_of = df["ts"].iloc[-1] + pd.Timedelta("4h")
    plain = GenomeEngine(g).decide(MarketView("X", pd.Timedelta("4h"), as_of, df))
    assert plain.side is Side.BUY  # sem colunas de notícia, os genes são neutros

    bad = df.assign(news_sentiment=-0.2, black_rw_mean_sent=-0.4)  # combinado = -0.2 + 0.5*-0.4 = -0.4
    gated = GenomeEngine(g).decide(MarketView("X", pd.Timedelta("4h"), as_of, bad))
    assert gated.side is Side.FLAT

    no_black = GenomeEngine(Genome("ma_cross", {"fast": 5, "slow": 20}, news_gate=0.3)).decide(
        MarketView("X", pd.Timedelta("4h"), as_of, bad))
    assert no_black.side is Side.BUY  # -0.2 não passa do limite -0.3 sem o black


def test_old_genomes_load_with_neutral_news_genes():
    from trader.evolution.genome import Genome

    g = Genome.from_dict({"strategy": "breakout", "params": {"lookback": 20}})
    assert (g.news_gate, g.news_follow, g.event_blackout, g.black_weight, g.hack_blackout) == (None, None, 0, 0.0, 0)


def test_collector_schedules_each_task_on_its_own_interval():
    from trader.live.collector import TASKS, _fill, due

    now = pd.Timestamp("2026-09-29 12:00", tz="UTC")
    assert {t.name for t in due(TASKS, {}, now)} == {t.name for t in TASKS}  # primeira vez: tudo
    last = {t.name: str(now - pd.Timedelta(minutes=11)) for t in TASKS}
    names = {t.name for t in due(TASKS, last, now)}
    assert "white_rss" in names and "calendar" not in names  # 10 min venceu, 1 dia não
    assert _fill(["--deriv-since", "{yesterday}"], now) == ["--deriv-since", "2026-09-28"]


def test_mutation_can_switch_news_genes_on():
    import random as _r
    from trader.evolution.genome import Genome, mutate

    rng = _r.Random(1)
    g = Genome("ma_cross", {"fast": 5, "slow": 20})
    kids = [mutate(g, rng) for _ in range(200)]
    assert any(k.news_gate is not None or k.event_blackout or k.black_weight for k in kids)
    quiet = [mutate(g, _r.Random(i), news_toggle=0.0) for i in range(50)]
    assert all(k.news_gate is None and not k.black_weight for k in quiet)


def test_live_exits_after_persistent_errors(tmp_path, monkeypatch):
    import pytest as _pytest
    from trader.live import runner as R

    df = random_walk(400, seed=3)
    df["ts"] = pd.date_range(T0, periods=len(df), freq="15min", tz="UTC")
    live = make_live(tmp_path, Clock(df["ts"].iloc[300]), df)
    monkeypatch.setattr(live, "tick", lambda: (_ for _ in ()).throw(ImportError("módulo antigo")))
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    with _pytest.raises(SystemExit):
        live.run_forever(max_consecutive_errors=3)
