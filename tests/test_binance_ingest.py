import pandas as pd

from trader.ingest import binance
from trader.storage.candles import CandleStore

HOUR_MS = 3_600_000
T0 = pd.Timestamp("2024-01-01", tz="UTC")


class FakeSource:
    """Simula a paginação da Binance: no máximo `limit` candles por chamada."""

    def __init__(self, count: int):
        start = int(T0.timestamp() * 1000)
        self.rows = [[start + i * HOUR_MS, 1, 1, 1, float(i), 1] for i in range(count)]
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        return [r for r in self.rows if r[0] >= since][:limit]


def test_paginates_and_drops_open_candle():
    source = FakeSource(25)
    now = T0 + pd.Timedelta(hours=24, minutes=30)  # candle das 24h ainda aberto
    df = binance.fetch_candles(source, "BTC/USDT", "1h", T0, now=now, limit=10)
    assert len(df) == 24
    assert source.calls >= 3
    assert df["ts"].iloc[-1] == T0 + pd.Timedelta(hours=23)


def test_update_resumes_from_last_candle(tmp_path):
    store = CandleStore(tmp_path)
    source = FakeSource(48)
    first = binance.update(store, "BTC/USDT", "1h", T0, source, now=T0 + pd.Timedelta(hours=10))
    second = binance.update(store, "BTC/USDT", "1h", T0, source, now=T0 + pd.Timedelta(hours=20))
    assert (first, second) == (10, 10)
    assert len(store.read("binance", "BTC/USDT", "1h")) == 20
