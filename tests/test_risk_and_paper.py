import pandas as pd
import pytest

from trader.core.baseline import MovingAverageCross
from trader.domain import MarketView, Order, Side, Signal
from trader.execution.paper import InsufficientFunds, PaperBroker
from trader.risk.gate import RiskGate, RiskLimits

NOW = pd.Timestamp("2024-01-01 12:00", tz="UTC")
BUY = Signal(Side.BUY, confidence=0.5, size=1.0)


def test_paper_broker_charges_fee_both_sides():
    broker = PaperBroker(cash=1000.0, fee_rate=0.001)
    broker.mark("BTC/USDT", 100.0, NOW)
    broker.submit(Order("BTC/USDT", Side.BUY, 5))
    broker.submit(Order("BTC/USDT", Side.SELL, 5))
    assert broker.cash == pytest.approx(1000 - 0.5 - 0.5)  # 0,1% de 500 em cada lado
    assert broker.position("BTC/USDT") == 0


def test_paper_broker_rejects_short_and_overspend():
    broker = PaperBroker(cash=100.0)
    broker.mark("BTC/USDT", 100.0, NOW)
    with pytest.raises(InsufficientFunds):
        broker.submit(Order("BTC/USDT", Side.SELL, 1))
    with pytest.raises(InsufficientFunds):
        broker.submit(Order("BTC/USDT", Side.BUY, 1))  # 100 + taxa > 100


def test_gate_caps_position_size():
    gate = RiskGate(RiskLimits(max_position_fraction=0.2))
    order = gate.order_for(BUY, "BTC/USDT", price=100.0, equity=1000.0, position=0.0, now=NOW)
    assert order.side is Side.BUY
    assert order.quantity == pytest.approx(2.0)


def test_gate_flattens_after_daily_loss_and_stays_halted():
    gate = RiskGate(RiskLimits(daily_loss_limit=0.03))
    gate.order_for(BUY, "BTC/USDT", 100.0, equity=1000.0, position=0.0, now=NOW)
    later = NOW + pd.Timedelta(hours=1)
    order = gate.order_for(BUY, "BTC/USDT", 100.0, equity=960.0, position=2.0, now=later)
    assert order == Order("BTC/USDT", Side.SELL, 2.0)
    # mesmo recuperando, não volta a comprar no mesmo dia
    assert gate.order_for(BUY, "BTC/USDT", 100.0, 1000.0, 0.0, later + pd.Timedelta(hours=1)) is None
    # dia seguinte libera
    assert gate.order_for(BUY, "BTC/USDT", 100.0, 1000.0, 0.0, NOW + pd.Timedelta(days=1)) is not None


def test_kill_switch_blocks_everything():
    gate = RiskGate(RiskLimits())
    gate.kill()
    assert gate.order_for(BUY, "BTC/USDT", 100.0, 1000.0, 0.0, NOW) is None
    assert gate.order_for(BUY, "BTC/USDT", 100.0, 1000.0, 3.0, NOW).side is Side.SELL


def test_end_to_end_engine_gate_broker(candles):
    df = candles("2024-01-01", 60)  # preço sempre subindo
    as_of = df["ts"].iloc[-1] + pd.Timedelta(hours=1)
    view = MarketView.at("BTC/USDT", "1h", df, as_of)
    signal = MovingAverageCross(fast=5, slow=20).decide(view)
    assert signal.side is Side.BUY

    broker = PaperBroker(cash=1000.0)
    broker.mark("BTC/USDT", view.last_close, as_of)
    gate = RiskGate(RiskLimits(max_position_fraction=0.5))
    order = gate.order_for(signal, "BTC/USDT", view.last_close, broker.equity(), broker.position("BTC/USDT"), as_of)
    broker.submit(order)
    assert broker.position("BTC/USDT") == pytest.approx(500 / view.last_close)


def test_gate_does_not_churn_on_small_price_moves():
    gate = RiskGate(RiskLimits(max_position_fraction=0.2))
    # alvo = 1000 * 0.2 / 101 = 1.98; posição 2.0 → desvio de 1%, não opera
    assert gate.order_for(BUY, "BTC/USDT", 101.0, 1000.0, 2.0, NOW) is None
    # sinal pede metade do tamanho → desvio de 50%, ajusta
    half = Signal(Side.BUY, confidence=0.5, size=0.5)
    order = gate.order_for(half, "BTC/USDT", 100.0, 1000.0, 2.0, NOW)
    assert order.side is Side.SELL and order.quantity == pytest.approx(1.0)
