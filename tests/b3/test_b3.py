import pandas as pd
import pytest

from trader.b3.contracts import active_contract, code, continuous, expiry
from trader.b3.costs import B3Risk, DEFAULT_COSTS


def test_codes_and_expiries():
    assert code("WIN", 2026, 10) == "WINV26" and code("WDO", 2026, 11) == "WDOX26"
    assert expiry("WIN", 2026, 10) == pd.Timestamp("2026-10-14")  # quarta mais próxima do dia 15
    assert expiry("WDO", 2026, 11) == pd.Timestamp("2026-11-02")  # 1º dia útil (sem feriados na lista)


def test_active_contract_rolls_before_expiry():
    assert active_contract("WIN", pd.Timestamp("2026-10-01")) == "WINV26"
    assert active_contract("WIN", pd.Timestamp("2026-10-12")) == "WINZ26"  # dentro da janela de troca
    assert active_contract("WDO", pd.Timestamp("2026-10-15")) == "WDOX26"
    assert active_contract("WDO", pd.Timestamp("2026-10-30")) == "WDOZ26"


def test_continuous_series_removes_roll_jump_but_keeps_moves():
    ts_old = pd.date_range("2026-10-08 13:00", periods=3, freq="1D", tz="UTC")
    ts_new = pd.date_range("2026-10-08 13:00", periods=6, freq="1D", tz="UTC")
    old = pd.DataFrame({"ts": ts_old, "open": 100.0, "high": 101.0, "low": 99.0, "close": [100.0, 101.0, 102.0], "volume": 1.0})
    new = pd.DataFrame({"ts": ts_new, "open": 110.0, "high": 111.0, "low": 109.0,
                        "close": [110.0, 111.0, 112.0, 113.0, 114.0, 115.0], "volume": 1.0})
    out = continuous({"WINV26": old, "WINZ26": new}, "WIN")
    assert set(out["contract"]) == {"WINV26", "WINZ26"}
    assert out["close"].diff().dropna().abs().max() <= 1.0 + 1e-9  # sem salto de 10 pontos na troca


def test_risk_blocks_contracts_that_could_go_negative_on_gap():
    risk = B3Risk(margin_per_contract=100.0, max_contracts=5, gap_scenario=0.03)
    # WIN a 130.000 pontos: gap de 3% = 3.900 pts × R$0,20 = R$780 por contrato
    assert risk.contracts_allowed("WIN", equity=1000.0, price=130000, wanted=5) == 1
    assert risk.contracts_allowed("WIN", equity=500.0, price=130000, wanted=1) == 0  # ficaria negativo num gap
    assert DEFAULT_COSTS["WIN"].per_side("WIN", 1) == pytest.approx(0.30 + 5 * 0.20)
    assert risk.in_session(pd.Timestamp("2026-10-01 13:00", tz="UTC"))  # 10:00 em Brasília
    assert not risk.in_session(pd.Timestamp("2026-10-01 21:30", tz="UTC"))  # 18:30
