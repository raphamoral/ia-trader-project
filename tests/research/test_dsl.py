import numpy as np
import pandas as pd
import pytest

from trader.research.dsl import InvalidStrategy, compile_strategy

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")


def closes(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(IDX)))) for s in "ABC"}, index=IDX)


SPEC = {"entry": {"all": [{"trend_vote": [50, 100, 150], "ge": 2}, {"not": {"rsi": 14, "gt": 80}},
                          {"any": [{"momentum": 20, "gt": 0.0}, {"vol_pct": 120, "lt": 0.6}]}]},
        "size": {"vol_target": 0.3}}


def test_compiles_to_bounded_weights():
    w = compile_strategy(SPEC, closes())
    assert w.shape == (400, 3) and (w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-9).all()


def test_no_future_data_truncation_invariance():
    """O peso no dia t não muda se cortarmos os dados depois de t."""
    c = closes(1)
    full = compile_strategy(SPEC, c)
    for cut in (250, 320, 399):
        part = compile_strategy(SPEC, c.iloc[: cut + 1])
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)


@pytest.mark.parametrize("bad", [
    {"entry": {"sma_cross": [100, 50]}},
    {"entry": {"above_sma": 10000}},
    {"entry": {"import": "os"}},
    {"entry": {"rsi": 14, "lt": 300}},
    {"entry": {"all": []}},
    {"entry": {"all": [{"not": {"not": {"not": {"not": {"not": {"above_sma": 20}}}}}}]}},
])
def test_rejects_invalid_or_unsafe(bad):
    with pytest.raises((InvalidStrategy, ValueError, TypeError)):
        compile_strategy(bad, closes())


def test_trend_frac_reproduces_the_committee():
    from trader.research.trend_daily import ensemble_weights

    c = closes(2)
    dsl = compile_strategy({"entry": {"trend_frac": [50, 100, 150, 200]}, "size": {"vol_target": 0.3}}, c)
    pd.testing.assert_frame_equal(dsl, ensemble_weights(c, 0.3), check_names=False)


def test_search_loop_records_valid_and_invalid(tmp_path, monkeypatch):
    import json
    from trader.research import strategy_search as S

    monkeypatch.setattr(S, "load_closes", lambda data: closes(3))
    monkeypatch.setattr(S, "SELECT", (IDX[0], IDX[-1]))
    monkeypatch.setattr(S, "standing_lessons", lambda *a, **k: [])
    answer = {"candidates": [
        {"name": "boa", "why": "x", "strategy": {"entry": {"trend_frac": [20, 50]}, "size": {"vol_target": 0.3}}},
        {"name": "ruim", "why": "y", "strategy": {"entry": {"sma_cross": [50, 20]}}},
    ]}
    post = lambda url, payload, timeout: {"choices": [{"message": {"content": json.dumps(answer)}}]}
    db = tmp_path / "s.sqlite"
    S.run(2, 2, tmp_path, db, "http://x", "m", post=post, log=lambda m: None)
    import sqlite3
    rows = sqlite3.connect(db).execute("SELECT round, name, valid FROM trials ORDER BY id").fetchall()
    assert rows[0] == (0, "comitê de tendência", 1)
    assert (1, "boa", 1) in rows and (1, "ruim", 0) in rows and (2, "boa", 1) in rows
