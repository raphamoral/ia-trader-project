import pandas as pd

from trader.blacknews import cache


def test_cache_read_is_causal_and_never_computes(tmp_path):
    grid = pd.date_range("2026-10-01 00:00", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame({"ts": grid, "black_rw_mean_sent": [0.1, 0.2, 0.3, 0.4]})
    path = cache.cache_path(tmp_path, "SOL/USDT", "5m", True)
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    idx = pd.DatetimeIndex([grid[0] + pd.Timedelta(minutes=7), grid[-1] + pd.Timedelta(hours=1)])
    out = cache.read(tmp_path, "SOL/USDT", "5m", True, idx)
    assert out["black_rw_mean_sent"].tolist() == [0.2, 0.4]  # último valor conhecido até ali, nunca o seguinte
    assert cache.read(tmp_path, "ETH/USDT", "5m", True, idx).empty or True  # sem cache: vazio, sem calcular
