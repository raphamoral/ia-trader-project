"""Simulador de execução: ordem limitada na fila × ordem a mercado, sobre dados gravados.

Pergunta central: esperar na fila como maker paga? A taxa menor pode ser engolida por três
custos que um backtest por candle não vê: a ordem não executa, executa só quando o preço vai
contra (seleção adversa), ou o tempo perdido obriga a cruzar o spread mais caro depois.

Nossas ordens são hipotéticas e pequenas: não alteram o livro nem os negócios gravados.
A latência vale nos dois sentidos da decisão: a ordem só chega `latency_ms` depois.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Iterable

from trader.lob.depth import DepthBook
from trader.lob.queue import QueuePosition


@dataclass(frozen=True)
class Costs:
    maker_bps: float = 2.0
    taker_bps: float = 5.0


@dataclass(frozen=True)
class Trade:
    ts: int
    price: int
    qty: int
    seller_aggressed: bool


@dataclass(frozen=True)
class Diff:
    ts: int
    event: dict


@dataclass(frozen=True)
class Snapshot:
    ts: int
    data: dict


@dataclass
class Passive:
    is_buy: bool
    qty: int
    placed_ts: int
    expires_ts: int
    arrival_mid: float
    queue: QueuePosition | None = None
    rejected: bool = False
    filled_ts: int | None = None


@dataclass
class Decision:
    """Uma decisão de execução medida pelos dois caminhos, sobre o mesmo instante."""

    is_buy: bool
    arrival_mid: float
    taker_cost_bps: float | None = None
    passive_cost_bps: float | None = None
    passive_filled: bool = False
    rejected: bool = False
    markout_bps: float | None = None  # só maker: movimento a nosso favor depois de executar


@dataclass
class Study:
    decisions: list[Decision] = field(default_factory=list)
    resyncs: int = 0

    def summary(self) -> dict:
        done = [d for d in self.decisions if d.taker_cost_bps is not None and d.passive_cost_bps is not None]
        diffs = [d.passive_cost_bps - d.taker_cost_bps for d in done]
        out = {"decisoes": len(done), "resyncs": self.resyncs}
        if not done:
            return out
        out["custo_taker_bps"] = mean(d.taker_cost_bps for d in done)
        out["custo_passivo_bps"] = mean(d.passive_cost_bps for d in done)
        out["taxa_execucao_passiva"] = mean(1.0 if d.passive_filled else 0.0 for d in done)
        out["diferenca_bps"] = mean(diffs)
        out["t_diferenca"] = None
        if len(diffs) > 1 and stdev(diffs) > 0:
            out["t_diferenca"] = mean(diffs) / (stdev(diffs) / math.sqrt(len(diffs)))
        marks = [d.markout_bps for d in done if d.markout_bps is not None]
        out["markout_maker_bps"] = mean(marks) if marks else None
        return out


def signed_cost_bps(is_buy: bool, price: float, mid: float, fee_bps: float) -> float:
    """Custo em relação ao meio de referência. Positivo é pior."""
    sign = 1.0 if is_buy else -1.0
    return sign * (price - mid) / mid * 1e4 + fee_bps


def run_study(events: Iterable[Trade | Diff | Snapshot], book: DepthBook, qty: int, every_ms: int = 30_000,
              patience_ms: int = 10_000, latency_ms: int = 100, markout_ms: int = 5_000,
              costs: Costs = Costs()) -> Study:
    """A cada `every_ms`, alternando compra e venda, mede a mesma decisão pelos dois caminhos:
    - agredir: cruza o livro ao chegar;
    - passivo: entra no melhor preço do próprio lado, espera `patience_ms` e, sem execução
      completa, cruza o livro pelo que faltar.
    `every_ms` deve ser maior que paciência + latência: as decisões não se sobrepõem."""
    if every_ms <= patience_ms + latency_ms + markout_ms:
        raise ValueError("every_ms precisa superar paciência + latência + markout")
    study = Study()
    next_decision: int | None = None
    is_buy = True
    pending: list[tuple[int, str, Decision, Passive | None]] = []
    last_gaps = 0

    def schedule(ts: int, kind: str, decision: Decision, passive: Passive | None = None) -> None:
        pending.append((ts, kind, decision, passive))
        pending.sort(key=lambda row: row[0])

    def run_pending(now: int) -> None:
        # estritamente antes: num empate de horário, o evento gravado vem primeiro (contra nós)
        while pending and pending[0][0] < now:
            _, kind, decision, passive = pending.pop(0)
            if not book.ready:
                continue  # sem livro confiável a medida é descartada, não inventada
            if kind == "take":
                _take(decision)
            elif kind == "place":
                _place(decision, passive)
                if not passive.rejected:
                    active.append((decision, passive))
            elif kind == "expire":
                _expire(decision, passive, now)
            elif kind == "mark":
                _mark(decision, passive)

    def _take(decision: Decision) -> None:
        done, price = book.walk(decision.is_buy, qty)
        if done == qty:
            decision.taker_cost_bps = signed_cost_bps(decision.is_buy, price, decision.arrival_mid, costs.taker_bps)

    def _place(decision: Decision, passive: Passive) -> None:
        price = book.best_bid() if decision.is_buy else book.best_ask()
        opposite = book.best_ask() if decision.is_buy else book.best_bid()
        if price is None or opposite is None:
            passive.rejected = decision.rejected = True
            return
        total = book.level(decision.is_buy, price)
        passive.queue = QueuePosition(decision.is_buy, price, qty, float(total), total)

    def _finish_passive(decision: Decision, passive: Passive, ts: int) -> None:
        queue = passive.queue
        cost = queue.filled * signed_cost_bps(decision.is_buy, queue.price, decision.arrival_mid, costs.maker_bps)
        left = queue.remaining
        if left:
            done, price = book.walk(decision.is_buy, left)
            if done != left:
                return
            cost += left * signed_cost_bps(decision.is_buy, price, decision.arrival_mid, costs.taker_bps)
        decision.passive_cost_bps = cost / qty
        decision.passive_filled = left == 0
        if left == 0:
            passive.filled_ts = ts
            schedule(ts + markout_ms, "mark", decision, passive)

    def _expire(decision: Decision, passive: Passive, ts: int) -> None:
        pair = (decision, passive)
        if pair in active:
            active.remove(pair)
            _finish_passive(decision, passive, ts)

    def _mark(decision: Decision, passive: Passive) -> None:
        mid = book.mid()
        if mid is not None:
            sign = 1.0 if decision.is_buy else -1.0
            decision.markout_bps = sign * (mid - passive.queue.price) / passive.queue.price * 1e4

    active: list[tuple[Decision, Passive]] = []

    for event in events:
        run_pending(event.ts)

        if isinstance(event, Snapshot):
            book.on_snapshot(event.data)
        elif isinstance(event, Diff):
            changed = book.on_diff(event.event)
            if book.gaps + book.crossed != last_gaps:
                last_gaps = book.gaps + book.crossed
                study.resyncs += 1
                active.clear()  # filas desconhecidas após perder o livro
            if changed:
                for decision, passive in list(active):
                    queue = passive.queue
                    queue.on_level(book.level(queue.is_bid, queue.price))
                    opposite = book.best_ask() if queue.is_bid else book.best_bid()
                    crossed = opposite is not None and (opposite <= queue.price if queue.is_bid
                                                        else opposite >= queue.price)
                    if crossed and queue.remaining:  # o outro lado passou por nós: fomos executados
                        queue.filled = queue.qty
                    if not queue.remaining:
                        active.remove((decision, passive))
                        _finish_passive(decision, passive, event.ts)
        else:
            for decision, passive in list(active):
                passive.queue.on_trade(event.price, event.qty, event.seller_aggressed)
                if not passive.queue.remaining:
                    active.remove((decision, passive))
                    _finish_passive(decision, passive, event.ts)

        if not book.ready:
            continue
        if next_decision is None:
            next_decision = event.ts
        if event.ts >= next_decision and not active and not pending:
            mid = book.mid()
            if mid is not None:
                decision = Decision(is_buy, mid)
                passive = Passive(is_buy, qty, event.ts, event.ts + latency_ms + patience_ms, mid)
                study.decisions.append(decision)
                schedule(event.ts + latency_ms, "take", decision)
                schedule(event.ts + latency_ms, "place", decision, passive)
                schedule(event.ts + latency_ms + patience_ms, "expire", decision, passive)
                is_buy = not is_buy
                next_decision = event.ts + every_ms
    return study
