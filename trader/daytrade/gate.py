"""Filtro de risco do day trade. Fora da evolução, limites congelados.

Transforma a intenção do motor em ações concretas:
  - teto de alavancagem;
  - stop obrigatório, sempre antes do preço de liquidação;
  - tamanho pela perda máxima aceita por operação (risco ÷ distância do stop);
  - zeragem obrigatória antes do fim da sessão;
  - limite de perda diária e parada de emergência.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from trader.daytrade.broker import BINANCE_BTC_MMR
from trader.daytrade.market import NO_MINIMUM, SymbolSpec


@dataclass(frozen=True)
class DaySignal:
    """direction: +1 comprar, -1 vender a descoberto, 0 zerar, None manter como está."""

    direction: int | None
    leverage: float = 1.0
    stop: float = 0.01  # distância do stop, fração do preço
    risk: float = 0.01  # fração do patrimônio perdida se o stop for atingido
    take_profit: float | None = None  # alvo em múltiplos da distância do stop (R)


@dataclass(frozen=True)
class DayTradeLimits:
    max_leverage: float = 3.0
    max_risk_per_trade: float = 0.02
    min_stop: float = 0.001
    max_stop: float = 0.05
    daily_loss_limit: float = 0.05
    session_end: dt.time = dt.time(0, 0)  # UTC; cripto vira o dia às 00:00. B3: ~18:00 BRT
    mmr: float = BINANCE_BTC_MMR
    liq_buffer: float = 0.005  # folga mínima entre o stop e a liquidação

    def __post_init__(self) -> None:
        if self.max_leverage < 1:
            raise ValueError("max_leverage >= 1")
        if not 0 < self.max_risk_per_trade <= 0.1:
            raise ValueError("max_risk_per_trade em (0, 0.1]")
        if not 0 < self.min_stop < self.max_stop < 1:
            raise ValueError("0 < min_stop < max_stop < 1")


@dataclass(frozen=True)
class Open:
    side: int
    qty: float
    leverage: float
    stop: float  # distância; o preço do stop é fixado sobre o preço de execução
    take_profit: float | None = None  # distância do alvo, fração do preço


@dataclass(frozen=True)
class Close:
    reason: str  # signal | reverse | eod | daily_loss | kill


class DayTradeRiskGate:
    def __init__(self, limits: DayTradeLimits, spec: SymbolSpec = NO_MINIMUM) -> None:
        self.limits = limits
        self.spec = spec
        self._killed = False
        self._session: pd.Timestamp | None = None
        self._session_start_equity = 0.0
        self._halted = False

    def kill(self) -> None:
        self._killed = True

    @property
    def halted(self) -> bool:
        """Limite de perda diária do gate atingido nesta sessão."""
        return self._halted

    def ruined(self, equity: float, price: float) -> bool:
        """Sem capital para a menor ordem da corretora, mesmo no teto de alavancagem."""
        return equity * self.limits.max_leverage * 0.98 < self.spec.min_order_value(price)

    def session_end_after(self, now: pd.Timestamp) -> pd.Timestamp:
        end = now.normalize() + pd.Timedelta(hours=self.limits.session_end.hour, minutes=self.limits.session_end.minute)
        return end if end > now else end + pd.Timedelta(days=1)

    def actions(
        self,
        signal: DaySignal,
        price: float,
        equity: float,
        position_side: int,
        now: pd.Timestamp,
        timeframe: pd.Timedelta,
    ) -> list[Open | Close]:
        """now = fechamento do candle; as ações executam na abertura do próximo."""
        session_end = self.session_end_after(now)
        if session_end != self._session:
            self._session = session_end
            self._session_start_equity = equity
            self._halted = False
        if equity <= self._session_start_equity * (1 - self.limits.daily_loss_limit):
            self._halted = True

        # o próximo candle termina no fim da sessão ou depois: zera e não abre nada
        last_bar = now + timeframe >= session_end
        blocked = self._killed or self._halted or last_bar
        if blocked:
            if position_side == 0:
                return []
            reason = "kill" if self._killed else "daily_loss" if self._halted else "eod"
            return [Close(reason)]

        d = signal.direction
        if d is None or d == position_side:
            return []
        out: list[Open | Close] = []
        if position_side != 0:
            out.append(Close("signal" if d == 0 else "reverse"))
        if d != 0:
            opened = self._open(signal, d, price, equity)
            if opened:
                out.append(opened)
        return out

    def _open(self, signal: DaySignal, side: int, price: float, equity: float) -> Open | None:
        lim = self.limits
        stop = min(max(signal.stop, lim.min_stop), lim.max_stop)
        leverage = min(max(signal.leverage, 1.0), lim.max_leverage)
        # distância até a liquidação ≈ 1/L - mmr; o stop precisa ficar antes, com folga
        max_lev_for_stop = 1 / (stop + lim.mmr + lim.liq_buffer)
        leverage = max(1.0, min(leverage, max_lev_for_stop))
        if (1 / leverage - lim.mmr) - stop < lim.liq_buffer - 1e-9:
            return None  # stop largo demais até para 1x
        risk = min(max(signal.risk, 0.0), lim.max_risk_per_trade)
        cap = equity * leverage * 0.98  # 2% de folga para taxa
        notional = min(equity * risk / stop, cap)
        minimum = self.spec.min_order_value(price)
        if notional < minimum:
            # sobe até a ordem mínima só se couber na alavancagem e no risco máximo
            if minimum > cap or minimum * stop > equity * lim.max_risk_per_trade:
                return None
            notional = minimum
        qty = self.spec.round_qty(notional / price)
        if qty <= 0 or qty * price < minimum - 1e-9:
            return None
        tp = None if signal.take_profit is None else stop * signal.take_profit
        return Open(side=side, qty=qty, leverage=leverage, stop=stop, take_profit=tp)
