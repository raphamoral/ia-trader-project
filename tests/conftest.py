import pandas as pd
import pytest


def make_candles(start: str, periods: int, freq: str = "1h", price: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = [price + i for i in range(periods)]
    return pd.DataFrame(
        {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": [1.0] * periods}
    )


@pytest.fixture
def candles():
    return make_candles
