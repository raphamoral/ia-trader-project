import pandas as pd
import pytest

from trader.domain import MarketView, Order, Side, Signal, parse_timeframe


def test_parse_timeframe():
    assert parse_timeframe("15m") == pd.Timedelta(minutes=15)
    assert parse_timeframe("1h") == pd.Timedelta(hours=1)
    assert parse_timeframe("1d") == pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        parse_timeframe("1M")


def test_market_view_hides_candle_still_open(candles):
    df = candles("2024-01-01", 5)  # aberturas 00h..04h
    view = MarketView.at("BTC/USDT", "1h", df, pd.Timestamp("2024-01-01 03:30", tz="UTC"))
    # 02h fecha às 03h (visível); 03h fecharia às 04h (futuro)
    assert view.candles["ts"].max() == pd.Timestamp("2024-01-01 02:00", tz="UTC")
    assert view.last_close == 102.0


def test_signal_and_order_validation():
    with pytest.raises(ValueError):
        Signal(Side.BUY, size=1.5)
    with pytest.raises(ValueError):
        Order("BTC/USDT", Side.FLAT, 1.0)
    with pytest.raises(ValueError):
        Order("BTC/USDT", Side.BUY, 0.0)
