"""Livro de ofertas e fluxo de negócios, em tempo real e com memória fixa.

Livro: snapshots parciais (top 20 níveis, a cada 100 ms) → spread, desequilíbrio
entre compra e venda e paredes (níveis muito maiores que a mediana).
Fluxo: negócios agregados (aggTrade) → CVD (compras a mercado − vendas a mercado),
volume comprador/vendedor numa janela móvel e negócios grandes.

Nada de tick é gravado em disco: só as agregações (a cada candle) saem daqui.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BookFeatures:
    ts_ms: int
    bid: float
    ask: float
    spread_bps: float
    imbalance5: float  # (bid − ask) / (bid + ask) nas 5 melhores faixas, de −1 a 1
    imbalance20: float
    bid_wall: float | None  # preço da maior parede de compra (>= wall_mult × mediana), se houver
    ask_wall: float | None


def book_features(ts_ms: int, bids: list[tuple[float, float]], asks: list[tuple[float, float]],
                  wall_mult: float = 5.0) -> BookFeatures | None:
    """bids/asks: [(preço, quantidade)], melhores primeiro."""
    if not bids or not asks:
        return None
    bid, ask = bids[0][0], asks[0][0]
    mid = (bid + ask) / 2

    def imb(k: int) -> float:
        b = sum(q for _, q in bids[:k])
        a = sum(q for _, q in asks[:k])
        return (b - a) / (b + a) if b + a > 0 else 0.0

    def wall(levels: list[tuple[float, float]]) -> float | None:
        qs = [q for _, q in levels]
        if len(qs) < 5:
            return None
        med = statistics.median(qs)
        price, q = max(levels, key=lambda lv: lv[1])
        return price if med > 0 and q >= wall_mult * med else None

    return BookFeatures(ts_ms, bid, ask, (ask - bid) / mid * 1e4, imb(5), imb(20), wall(bids), wall(asks))


@dataclass
class TradeFlow:
    """Fluxo de agressão numa janela móvel de `window_ms` (memória limitada a `max_trades`)."""

    window_ms: int = 60_000
    large_notional: float = 100_000.0  # negócio grande em US$
    max_trades: int = 50_000
    cvd: float = 0.0  # acumulado desde o início (em quantidade)
    _win: deque = field(default_factory=deque)
    _buy: float = 0.0
    _sell: float = 0.0
    large: deque = field(default_factory=lambda: deque(maxlen=50))

    def update(self, ts_ms: int, price: float, qty: float, buyer_is_maker: bool) -> None:
        # buyer_is_maker = quem agrediu foi o vendedor
        signed = -qty if buyer_is_maker else qty
        self.cvd += signed
        notional = price * qty
        if signed > 0:
            self._buy += notional
        else:
            self._sell += notional
        self._win.append((ts_ms, signed > 0, notional))
        if notional >= self.large_notional:
            self.large.append((ts_ms, price, signed))
        cutoff = ts_ms - self.window_ms
        while self._win and (self._win[0][0] < cutoff or len(self._win) > self.max_trades):
            _, is_buy, n = self._win.popleft()
            if is_buy:
                self._buy -= n
            else:
                self._sell -= n

    def snapshot(self) -> dict[str, float]:
        total = self._buy + self._sell
        return {"cvd": self.cvd, "buy_usd_win": self._buy, "sell_usd_win": self._sell,
                "aggr_ratio": (self._buy - self._sell) / total if total > 0 else 0.0,
                "large_trades": float(len(self.large))}
