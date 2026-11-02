"""Reconstrói o livro L2 da Binance a partir do snapshot e das atualizações diferenciais.

O protocolo só é confiável com a numeração de sequência: perder uma mensagem deixa um nível
errado para sempre, sem nenhum erro. Por isso a sincronização é uma máquina de estados:

    SEM_SNAPSHOT --snapshot casa com o buffer--> SINCRONIZADO
    SINCRONIZADO --buraco na sequência ou livro cruzado--> SEM_SNAPSHOT

Fora de SINCRONIZADO o livro não responde: um preço velho é pior do que nenhum preço.

Regras oficiais:
  futuros: primeiro evento com U <= lastUpdateId <= u; depois, pu == u do evento anterior.
  spot:    primeiro evento com U <= lastUpdateId + 1 <= u; depois, U == u anterior + 1.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


def to_units(value: str, step: str) -> int:
    """Converte texto exato em múltiplo inteiro de `step`. Recusa valor fora da grade."""
    units = Decimal(value) / Decimal(step)
    if units != units.to_integral_value():
        raise ValueError(f"{value} não é múltiplo de {step}")
    return int(units)


class SyncState(str, Enum):
    NEEDS_SNAPSHOT = "needs_snapshot"
    SYNCED = "synced"


class _Side:
    """Níveis de preço de um lado, com os preços em lista ordenada."""

    def __init__(self) -> None:
        self.qty: dict[int, int] = {}
        self.prices: list[int] = []

    def set(self, price: int, qty: int) -> None:
        if qty:
            if price not in self.qty:
                insort(self.prices, price)
            self.qty[price] = qty
        elif price in self.qty:
            del self.qty[price]
            del self.prices[bisect_left(self.prices, price)]

    def clear(self) -> None:
        self.qty.clear()
        self.prices.clear()


@dataclass
class DepthBook:
    tick: str
    lot: str
    venue: str = "futures"  # "futures" ou "spot"
    max_buffer: int = 20_000
    state: SyncState = SyncState.NEEDS_SNAPSHOT
    last_id: int | None = None
    gaps: int = 0
    crossed: int = 0
    applied: int = 0
    _awaiting_bridge: bool = False  # snapshot chegou antes do evento que o cobre
    _bids: _Side = field(default_factory=_Side)
    _asks: _Side = field(default_factory=_Side)
    _buffer: deque = field(default_factory=deque)

    # ---------------------------------------------------------------- consulta
    @property
    def ready(self) -> bool:
        return self.state is SyncState.SYNCED

    def best_bid(self) -> int | None:
        return self._bids.prices[-1] if self.ready and self._bids.prices else None

    def best_ask(self) -> int | None:
        return self._asks.prices[0] if self.ready and self._asks.prices else None

    def mid(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else (bid + ask) / 2

    def level(self, is_bid: bool, price: int) -> int:
        return (self._bids if is_bid else self._asks).qty.get(price, 0)

    def walk(self, is_buy: bool, qty: int) -> tuple[int, float | None]:
        """Executa `qty` contra o livro, sem alterá-lo: devolve (quantidade feita, preço médio)."""
        side = self._asks if is_buy else self._bids
        prices = side.prices if is_buy else reversed(side.prices)
        left, cost = qty, 0
        for price in prices:
            take = min(left, side.qty[price])
            cost += take * price
            left -= take
            if not left:
                break
        done = qty - left
        return done, (cost / done if done else None)

    # ---------------------------------------------------------------- protocolo
    def on_diff(self, event: dict) -> bool:
        """Aplica um evento. Devolve True se o livro mudou."""
        if not self.ready:
            self._remember(event)
            return False
        if event["u"] < self.last_id or (event["u"] == self.last_id and not self._awaiting_bridge):
            return False  # reentrega de algo já aplicado
        if self._awaiting_bridge:
            if not self._bridges(event, self.last_id) and not self._continues(event):
                self._desync(event, "gap")
                return False
            self._awaiting_bridge = False
        elif not self._continues(event):
            self._desync(event, "gap")
            return False
        self._apply(event)
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            self._desync(event, "crossed")
            return False
        return True

    def on_snapshot(self, snapshot: dict) -> bool:
        """Devolve True se sincronizou. False: o snapshot é velho demais, peça outro."""
        last = snapshot["lastUpdateId"]
        events = [e for e in self._buffer if e["u"] >= (last if self.venue == "futures" else last + 1)]
        if events and not self._bridges(events[0], last):
            return False
        self._bids.clear()
        self._asks.clear()
        for price, qty in snapshot["bids"]:
            self._bids.set(to_units(price, self.tick), to_units(qty, self.lot))
        for price, qty in snapshot["asks"]:
            self._asks.set(to_units(price, self.tick), to_units(qty, self.lot))
        self.last_id = last
        self.state = SyncState.SYNCED
        self._buffer.clear()
        self._awaiting_bridge = not events
        for i, event in enumerate(events):
            if not self.ready:
                self._remember(event)
            elif i == 0:
                self._apply(event)  # a ponte: `pu` dela é anterior ao snapshot, por definição
            else:
                self.on_diff(event)
        return self.ready

    def _bridges(self, event: dict, last: int) -> bool:
        if self.venue == "futures":
            return event["U"] <= last <= event["u"]
        return event["U"] <= last + 1 <= event["u"]

    def _continues(self, event: dict) -> bool:
        if self.venue == "futures":
            return event["pu"] == self.last_id
        return event["U"] == self.last_id + 1

    def _apply(self, event: dict) -> None:
        for price, qty in event["b"]:
            self._bids.set(to_units(price, self.tick), to_units(qty, self.lot))
        for price, qty in event["a"]:
            self._asks.set(to_units(price, self.tick), to_units(qty, self.lot))
        self.last_id = event["u"]
        self.applied += 1

    def _remember(self, event: dict) -> None:
        self._buffer.append(event)
        if len(self._buffer) > self.max_buffer:
            self._buffer.popleft()

    def _desync(self, event: dict, why: str) -> None:
        if why == "gap":
            self.gaps += 1
        else:
            self.crossed += 1
        self.state = SyncState.NEEDS_SNAPSHOT
        self.last_id = None
        self._awaiting_bridge = False
        self._bids.clear()
        self._asks.clear()
        self._remember(event)
