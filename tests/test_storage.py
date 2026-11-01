import pandas as pd

from trader.storage.candles import CandleStore


def test_write_is_idempotent_and_splits_by_year(tmp_path, candles):
    store = CandleStore(tmp_path)
    df = candles("2023-12-31 22:00", 4)  # atravessa a virada do ano
    store.write("binance", "BTC/USDT", "1h", df)
    store.write("binance", "BTC/USDT", "1h", df)

    files = sorted(p.name for p in (tmp_path / "candles/binance/BTCUSDT/1h").glob("*.parquet"))
    assert files == ["2023.parquet", "2024.parquet"]
    out = store.read("binance", "BTC/USDT", "1h")
    assert len(out) == 4
    assert out["ts"].is_monotonic_increasing
    assert store.last_ts("binance", "BTC/USDT", "1h") == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_read_filters_range(tmp_path, candles):
    store = CandleStore(tmp_path)
    store.write("binance", "BTC/USDT", "1h", candles("2024-01-01", 10))
    out = store.read(
        "binance", "BTC/USDT", "1h",
        start=pd.Timestamp("2024-01-01 02:00", tz="UTC"),
        end=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
    )
    assert len(out) == 3


def test_read_empty(tmp_path):
    assert CandleStore(tmp_path).read("binance", "ETH/USDT", "1h").empty


def test_replace_with_retry_waits_for_locked_target(tmp_path, monkeypatch):
    import os

    from trader.storage import atomic

    calls = []
    real = os.replace

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("em uso por outro processo")
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    monkeypatch.setattr(atomic.time, "sleep", lambda s: None)
    (tmp_path / "a.tmp").write_text("novo")
    atomic.replace_with_retry(tmp_path / "a.tmp", tmp_path / "a.parquet")
    assert (tmp_path / "a.parquet").read_text() == "novo" and len(calls) == 3
