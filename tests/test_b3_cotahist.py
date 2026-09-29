import zipfile

import pandas as pd
import pytest

from trader.ingest import b3_cotahist
from trader.storage.candles import CandleStore

HEADER = "00COTAHIST.2023BOVESPA 20231228".ljust(245)
# linha real do COTAHIST_A2023 (BOVA11 em 02/01/2023)
BOVA = (
    "012023010214BOVA11      010ISHARES BOVACI  ATZ      R$  00000000104360000000010445000000001022900000"
    "0001028300000000102630000000010263000000001027057055000000000007559253000000077734410158000000000000"
    "009999123100000010000000000000BRBOVACTF003107"
)
OPTION = BOVA[:24] + "070" + BOVA[27:]  # mesmo registro, mercado de opções


def test_line_is_official_width():
    assert len(BOVA) == 245


def test_parses_prices_and_skips_other_markets():
    df = b3_cotahist.parse_lines([HEADER, BOVA, OPTION])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "BOVA11"
    assert row["ts"] == pd.Timestamp("2023-01-02", tz="UTC")
    assert (row["open"], row["high"], row["low"], row["close"]) == pytest.approx((104.36, 104.45, 102.29, 102.63))
    assert row["volume"] == 7559253


def test_quote_factor_divides_prices():
    per_thousand = BOVA[:210] + "0001000" + BOVA[217:]
    assert b3_cotahist.parse_lines([per_thousand]).iloc[0]["close"] == pytest.approx(0.10263)


def test_import_zip_into_store(tmp_path):
    z = tmp_path / "COTAHIST_A2023.ZIP"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("COTAHIST_A2023.TXT", "\r\n".join([HEADER, BOVA, OPTION]) + "\r\n")
    store = CandleStore(tmp_path / "data")
    counts, _ = b3_cotahist.import_files(store, [z], tickers=["BOVA11", "PETR4"])
    assert counts == {"BOVA11": 1}
    assert store.read("b3", "BOVA11", "1d")["close"].iloc[0] == pytest.approx(102.63)


def test_adjust_splits_detects_2_for_1():
    ts = pd.date_range("2024-04-12", periods=4, freq="D", tz="UTC")
    close = [40.0, 42.0, 21.3, 21.5]  # desdobramento 1:2 no terceiro dia
    df = pd.DataFrame({"ts": ts, "open": [40.0, 41.0, 21.1, 21.3], "high": close, "low": close,
                       "close": close, "volume": [100.0] * 4})
    adjusted, events = b3_cotahist.adjust_splits(df)
    assert events == [(ts[2], 0.5)]
    assert adjusted["close"].tolist() == pytest.approx([20.0, 21.0, 21.3, 21.5])
    assert adjusted["volume"].iloc[0] == 200


def test_adjust_splits_ignores_normal_moves():
    ts = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"ts": ts, "open": [10.0, 8.9, 9.0], "high": 11.0, "low": 8.0,
                       "close": [10.0, 9.0, 9.1], "volume": 1.0})
    assert b3_cotahist.adjust_splits(df)[1] == []
