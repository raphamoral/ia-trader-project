"""Motor de casamento com prioridade preço-tempo, escrito do zero.

Preços e quantidades são inteiros (ticks e lotes): assim não há erro de ponto flutuante e
o estado é reproduzível byte a byte, o que o diário de comandos (`journal.py`) exige.

Cada nível é uma lista duplamente ligada: cancelar é O(1), sem varrer a fila. Os melhores
preços vêm de um heap com remoção preguiçosa, e níveis vazios são descartados ao chegar ao topo.
"""
from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"

    @property
    def opposite(self) -> Side:
        return Side.ASK if self is Side.BID else Side.BID


class Tif(str, Enum):
    GTC = "gtc"  # fica no livro até executar ou cancelar
    IOC = "ioc"  # executa o que der e cancela o resto
    FOK = "fok"  # executa tudo de uma vez ou nada
    POST_ONLY = "post_only"  # recusada se fosse executar: garante taxa maker


class Stp(str, Enum):
    """Prevenção de negócio consigo mesmo."""

    NONE = "none"
    CANCEL_NEWEST = "cancel_newest"  # cancela o resto da ordem que chegou
    CANCEL_OLDEST = "cancel_oldest"  # cancela a ordem que estava no livro e segue casando


@dataclass(slots=True)
class Resting:
    id: int
    owner: str
    side: Side
    price: int
    qty: int
    seq: int
    level: Level | None = None
    prev: Resting | None = None
    next: Resting | None = None


@dataclass(slots=True)
class Level:
    price: int
    total: int = 0
    count: int = 0
    head: Resting | None = None
    tail: Resting | None = None

    def append(self, order: Resting) -> None:
        order.level, order.prev, order.next = self, self.tail, None
        if self.tail is None:
            self.head = order
        else:
            self.tail.next = order
        self.tail = order
        self.total += order.qty
        self.count += 1

    def unlink(self, order: Resting) -> None:
        if order.prev is None:
            self.head = order.next
        else:
            order.prev.next = order.next
        if order.next is None:
            self.tail = order.prev
        else:
            order.next.prev = order.prev
        self.total -= order.qty
        self.count -= 1
        order.level = order.prev = order.next = None


@dataclass(frozen=True, slots=True)
class Fill:
    taker_id: int
    maker_id: int
    taker_owner: str
    maker_owner: str
    taker_side: Side
    price: int
    qty: int
    seq: int


@dataclass(slots=True)
class Result:
    accepted: bool
    reason: str = ""
    fills: list[Fill] = field(default_factory=list)
    filled: int = 0
    resting: int = 0
    canceled: int = 0


class LimitOrderBook:
    def __init__(self) -> None:
        self._levels: dict[Side, dict[int, Level]] = {Side.BID: {}, Side.ASK: {}}
        self._heaps: dict[Side, list[int]] = {Side.BID: [], Side.ASK: []}  # bid guarda −preço
        self._orders: dict[int, Resting] = {}
        self._seq = 0
        self._fill_seq = 0

    # ---------------------------------------------------------------- consulta
    def best(self, side: Side) -> int | None:
        heap, levels = self._heaps[side], self._levels[side]
        while heap:
            price = -heap[0] if side is Side.BID else heap[0]
            if price in levels:
                return price
            heapq.heappop(heap)  # nível esvaziado: remoção preguiçosa
        return None

    def depth(self, side: Side, n: int | None = None) -> list[tuple[int, int]]:
        prices = sorted(self._levels[side], reverse=side is Side.BID)
        return [(p, self._levels[side][p].total) for p in prices[:n]]

    def order(self, order_id: int) -> Resting | None:
        return self._orders.get(order_id)

    def queue_ahead(self, order_id: int) -> int:
        """Quantidade à frente da ordem na fila do seu preço."""
        node = self._orders[order_id].prev
        ahead = 0
        while node is not None:
            ahead += node.qty
            node = node.prev
        return ahead

    def __len__(self) -> int:
        return len(self._orders)

    # ---------------------------------------------------------------- comandos
    def submit(self, order_id: int, owner: str, side: Side, qty: int, price: int | None = None,
               tif: Tif = Tif.GTC, stp: Stp = Stp.NONE) -> Result:
        """`price=None` é ordem a mercado: nunca fica no livro."""
        if qty <= 0:
            return Result(False, "qty_invalida")
        if price is not None and price <= 0:
            return Result(False, "preco_invalido")
        if order_id in self._orders:
            return Result(False, "id_duplicado")
        if price is None and tif in (Tif.GTC, Tif.POST_ONLY):
            return Result(False, "mercado_exige_ioc_ou_fok")
        if tif is Tif.POST_ONLY and self._crosses(side, price):
            return Result(False, "post_only_cruzaria")
        if tif is Tif.FOK and self._fillable(side, price, qty, owner, stp) < qty:
            return Result(True, "fok_sem_liquidez", canceled=qty)

        self._seq += 1
        result = Result(True)
        remaining = self._match(order_id, owner, side, qty, price, stp, result)
        if remaining and tif in (Tif.GTC, Tif.POST_ONLY) and not result.reason:
            self._rest(order_id, owner, side, price, remaining)
            result.resting = remaining
        else:
            result.canceled = remaining
        return result

    def cancel(self, order_id: int) -> Result:
        order = self._orders.get(order_id)
        if order is None:
            return Result(False, "ordem_inexistente")
        qty = order.qty
        self._remove(order)
        return Result(True, canceled=qty)

    def modify(self, order_id: int, qty: int, price: int | None = None, stp: Stp = Stp.NONE) -> Result:
        """Reduzir no mesmo preço mantém a fila. Aumentar ou mudar o preço vai para o fim da fila."""
        order = self._orders.get(order_id)
        if order is None:
            return Result(False, "ordem_inexistente")
        if qty <= 0 or (price is not None and price <= 0):
            return Result(False, "qty_invalida" if qty <= 0 else "preco_invalido")
        new_price = order.price if price is None else price
        if new_price == order.price and qty <= order.qty:
            order.level.total -= order.qty - qty
            order.qty = qty
            return Result(True, resting=qty)
        owner, side = order.owner, order.side
        self._remove(order)
        return self.submit(order_id, owner, side, qty, new_price, Tif.GTC, stp)

    # ---------------------------------------------------------------- casamento
    def _crosses(self, side: Side, price: int | None) -> bool:
        best = self.best(side.opposite)
        if best is None:
            return False
        if price is None:
            return True
        return price >= best if side is Side.BID else price <= best

    def _fillable(self, side: Side, price: int | None, qty: int, owner: str, stp: Stp) -> int:
        """Quanto executaria agora, sem mexer no livro. Respeita a parada da autoprevenção."""
        total = 0
        opposite = side.opposite
        for level_price in sorted(self._levels[opposite], reverse=opposite is Side.BID):
            if price is not None and (level_price > price if side is Side.BID else level_price < price):
                break
            node = self._levels[opposite][level_price].head
            while node is not None:
                if node.owner == owner and stp is not Stp.NONE:
                    if stp is Stp.CANCEL_NEWEST:
                        return total
                else:
                    total += node.qty
                if total >= qty:
                    return total
                node = node.next
        return total

    def _match(self, taker_id: int, owner: str, side: Side, qty: int, price: int | None, stp: Stp,
               result: Result) -> int:
        opposite = side.opposite
        while qty and self._crosses(side, price):
            level = self._levels[opposite][self.best(opposite)]
            maker = level.head
            if maker.owner == owner and stp is not Stp.NONE:
                if stp is Stp.CANCEL_NEWEST:
                    result.reason = "stp_cancelou_nova"
                    break
                self._remove(maker)
                continue
            traded = min(qty, maker.qty)
            self._fill_seq += 1
            result.fills.append(Fill(taker_id, maker.id, owner, maker.owner, side, level.price, traded,
                                     self._fill_seq))
            result.filled += traded
            qty -= traded
            maker.qty -= traded
            level.total -= traded
            if maker.qty == 0:
                self._remove(maker)
        return qty

    def _rest(self, order_id: int, owner: str, side: Side, price: int, qty: int) -> None:
        level = self._levels[side].get(price)
        if level is None:
            level = self._levels[side][price] = Level(price)
            heapq.heappush(self._heaps[side], -price if side is Side.BID else price)
        order = Resting(order_id, owner, side, price, qty, self._seq)
        level.append(order)
        self._orders[order_id] = order

    def _remove(self, order: Resting) -> None:
        level = order.level
        level.unlink(order)
        del self._orders[order.id]
        if level.count == 0:
            del self._levels[order.side][level.price]

    # ---------------------------------------------------------------- estado
    def dump(self) -> dict:
        """Estado completo, em ordem canônica. A ordem das listas é a prioridade da fila."""
        sides = {}
        for side in Side:
            rows = []
            for price in sorted(self._levels[side]):
                node = self._levels[side][price].head
                while node is not None:
                    rows.append([node.id, node.owner, node.price, node.qty, node.seq])
                    node = node.next
            sides[side.value] = rows
        return {"seq": self._seq, "fill_seq": self._fill_seq, **sides}

    @classmethod
    def load(cls, state: dict) -> LimitOrderBook:
        book = cls()
        book._fill_seq = state["fill_seq"]
        for side in Side:
            for order_id, owner, price, qty, seq in state[side.value]:
                book._seq = seq
                book._rest(order_id, owner, side, price, qty)
        book._seq = state["seq"]
        return book

    def state_hash(self) -> str:
        return hashlib.sha256(repr(self.dump()).encode()).hexdigest()

    def check(self) -> None:
        """Invariantes do livro. Levanta AssertionError na primeira violação."""
        seen = 0
        for side in Side:
            for price, level in self._levels[side].items():
                assert level.count > 0, f"nível vazio em {price}"
                total, count, node, previous = 0, 0, level.head, None
                while node is not None:
                    assert node.qty > 0 and node.price == price and node.side is side and node.level is level
                    assert node.prev is previous, "lista ligada quebrada"
                    assert previous is None or previous.seq < node.seq, "fila fora de ordem"
                    assert self._orders[node.id] is node
                    total, count, previous, node = total + node.qty, count + 1, node, node.next
                assert level.tail is previous and total == level.total and count == level.count
                seen += count
        assert seen == len(self._orders), "ordem órfã"
        bid, ask = self.best(Side.BID), self.best(Side.ASK)
        assert bid is None or ask is None or bid < ask, "livro cruzado"
