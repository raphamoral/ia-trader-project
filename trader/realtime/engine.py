"""Motor em tempo real: websocket da Binance (futuros) → candles, indicadores, livro, fluxo → executor.

Streams combinados por ativo, em duas conexões: <s>@aggTrade (cada negócio, em /market) e
<s>@depth20@100ms (top 20 do livro a cada 100 ms). Cada negócio vai ao executor (stops, trava, "deixa correr") e
monta o candle de 1 minuto; no fechamento, os indicadores se atualizam e a agregação
do minuto (sem ticks) fica disponível para gravar.

Reconecta sozinho (a Binance derruba conexões a cada 24h). Só papel.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from trader.realtime.flow import BookFeatures, TradeFlow, book_features
from trader.realtime.indicators import Bar, IndicatorSet

# a Binance separou os streams: negócios em /market, livro no caminho antigo
WS_TRADES = "wss://fstream.binance.com/market/stream?streams="
WS_BOOK = "wss://fstream.binance.com/stream?streams="


def stream_name(symbol: str) -> str:
    return symbol.split(":")[0].replace("/", "").lower()


@dataclass
class SymbolState:
    symbol: str
    tf_ms: int = 60_000
    indicators: IndicatorSet = field(default_factory=IndicatorSet)
    flow: TradeFlow = field(default_factory=TradeFlow)
    book: BookFeatures | None = None
    bar: list | None = None  # [início_ms, open, high, low, close, volume]
    minutes: deque = field(default_factory=lambda: deque(maxlen=1440))  # agregados do último dia
    last_price: float = float("nan")
    lag_ms: deque = field(default_factory=lambda: deque(maxlen=1000))  # atraso evento→processado

    def on_trade(self, ts_ms: int, price: float, qty: float, buyer_is_maker: bool) -> dict | None:
        """Atualiza o fluxo e o candle; devolve o agregado do minuto quando um candle fecha."""
        self.last_price = price
        self.flow.update(ts_ms, price, qty, buyer_is_maker)
        start = ts_ms - ts_ms % self.tf_ms
        closed = None
        if self.bar is not None and start > self.bar[0]:
            closed = self._close_bar()
        if self.bar is None or start > self.bar[0]:
            self.bar = [start, price, price, price, price, 0.0]
        b = self.bar
        b[2], b[3], b[4], b[5] = max(b[2], price), min(b[3], price), price, b[5] + qty
        return closed

    def _close_bar(self) -> dict:
        s, o, h, l, c, v = self.bar
        ts = pd.Timestamp(s, unit="ms", tz="UTC")
        ind = self.indicators.update(Bar(ts, o, h, l, c, v))
        row = {"ts": ts, "open": o, "high": h, "low": l, "volume": v, **ind, **self.flow.snapshot()}
        if self.book is not None:
            row |= {"spread_bps": self.book.spread_bps, "imbalance5": self.book.imbalance5,
                    "imbalance20": self.book.imbalance20, "bid_wall": self.book.bid_wall,
                    "ask_wall": self.book.ask_wall}
        self.minutes.append(row)
        return row


class RealtimeEngine:
    def __init__(self, symbols: list[str], executor=None, on_minute: Callable[[str, dict], None] | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.states = {stream_name(s): SymbolState(s) for s in symbols}
        self.executor, self.on_minute, self.log = executor, on_minute, log
        self.messages = 0
        self._stop = False

    def urls(self) -> list[str]:
        return [WS_TRADES + "/".join(f"{n}@aggTrade" for n in self.states),
                WS_BOOK + "/".join(f"{n}@depth20@100ms" for n in self.states)]

    def handle(self, msg: dict) -> None:
        """Processa uma mensagem do stream combinado (separado da rede para testar sem conexão)."""
        stream, data = msg.get("stream", ""), msg.get("data", {})
        name, _, kind = stream.partition("@")
        st = self.states.get(name)
        if st is None:
            return
        self.messages += 1
        if kind == "aggTrade":
            ts_ms, price, qty = int(data["T"]), float(data["p"]), float(data["q"])
            closed = st.on_trade(ts_ms, price, qty, bool(data["m"]))
            if self.executor is not None:
                ind = st.indicators.last
                self.executor.on_price(pd.Timestamp(ts_ms, unit="ms", tz="UTC"), st.symbol, price,
                                       ind.get("atr14", float("nan")), ind.get("atr_pct", float("nan")))
            if closed is not None and self.on_minute is not None:
                self.on_minute(st.symbol, closed)
            st.lag_ms.append(time.time() * 1000 - ts_ms)
        elif kind.startswith("depth"):
            bids = [(float(p), float(q)) for p, q in data.get("b", [])]
            asks = [(float(p), float(q)) for p, q in data.get("a", [])]
            st.book = book_features(int(data.get("T", data.get("E", 0))), bids, asks) or st.book

    async def run(self, seconds: float | None = None) -> None:
        deadline = None if seconds is None else time.monotonic() + seconds
        await asyncio.gather(*(self._connection(u, deadline) for u in self.urls()))

    async def _connection(self, url: str, deadline: float | None) -> None:
        import websockets

        backoff = 1.0
        while not self._stop and (deadline is None or time.monotonic() < deadline):
            try:
                async with websockets.connect(url, ping_interval=20, max_size=2**22, open_timeout=15) as ws:
                    self.log(f"conectado: {url.split('?')[0]} ({len(self.states)} ativos)")
                    backoff = 1.0
                    while not self._stop:
                        timeout = 60.0 if deadline is None else max(0.1, deadline - time.monotonic())
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            if deadline is not None and time.monotonic() >= deadline:
                                return
                            raise  # 60 s sem nada: reconecta
                        self.handle(json.loads(raw))
            except Exception as exc:  # rede caiu ou silêncio: reconecta
                if self._stop or (deadline is not None and time.monotonic() >= deadline):
                    return
                self.log(f"conexão caiu ({type(exc).__name__}); nova tentativa em {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def stop(self) -> None:
        self._stop = True
