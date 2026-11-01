"""Executor: aplica a política a cada evento, em microssegundos, sem chamar a LLM.

Eventos:
  on_price(ts, symbol, price, atr, atr_pct)  cada negócio/tick: stops, alvo, trava, "deixa correr"
  on_signal(ts, symbol, side, setup, price)  um agente quer entrar: a política aprova ou recusa
  on_news(ts, kind, symbols, sentiment, severity)  reações pré-aprovadas (zerar, reduzir, bloquear, apertar)

Só papel: usa o FuturesPaperBroker. Política vencida = nenhuma entrada nova (as
saídas continuam sendo cuidadas). Todo evento fica num diário em memória (limitado).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import pandas as pd

from trader.daytrade.backtest import exit_stop
from trader.daytrade.broker import FuturesPaperBroker, InsufficientMargin
from trader.daytrade.gate import DayTradeLimits
from trader.policy.schema import SAFE, Policy

REAL_ORDERS_ENABLED = False  # trava: este módulo só opera em papel


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    qty: float = 0.0


class PolicyExecutor:
    def __init__(self, broker: FuturesPaperBroker, limits: DayTradeLimits = DayTradeLimits(),
                 journal_size: int = 5000) -> None:
        assert not REAL_ORDERS_ENABLED
        self.broker, self.limits = broker, limits
        self.policy: Policy = SAFE
        self.issued_at: pd.Timestamp | None = None
        self.blocked_until: dict[str, pd.Timestamp] = {}
        self.marks: dict[str, float] = {}
        self._best: dict[str, float] = {}
        self._stop0: dict[str, float] = {}
        self._runner: dict[str, bool] = {}
        self._day: object = None
        self._day_start = broker.cash
        self.journal: deque = deque(maxlen=journal_size)

    # política

    def set_policy(self, policy: Policy, ts: pd.Timestamp) -> None:
        self.policy, self.issued_at = policy, ts
        self._log(ts, "policy", f"nova política ({len(policy.symbols)} ativos, {len(policy.reactions)} reações)")
        if policy.kill:
            for sym in list(self.broker.positions):
                self._close(ts, sym, "kill")

    def expired(self, ts: pd.Timestamp) -> bool:
        return self.issued_at is None or ts >= self.issued_at + pd.Timedelta(minutes=self.policy.valid_minutes)

    # eventos

    def on_price(self, ts: pd.Timestamp, symbol: str, price: float, atr: float = math.nan,
                 atr_pct: float = math.nan) -> str | None:
        self.marks[symbol] = price
        self._roll_day(ts)
        pos = self.broker.position(symbol)
        if pos is None:
            return None
        if (pos.side > 0 and price <= pos.liq) or (pos.side < 0 and price >= pos.liq):
            return self._close(ts, symbol, "liquidation", price)
        if (pos.side > 0 and price <= pos.stop) or (pos.side < 0 and price >= pos.stop):
            return self._close(ts, symbol, "stop", price)
        if pos.take_profit is not None and ((pos.side > 0 and price >= pos.take_profit)
                                            or (pos.side < 0 and price <= pos.take_profit)):
            return self._close(ts, symbol, "take_profit", price)
        sp = self.policy.for_symbol(symbol)
        best = self._best.get(symbol, pos.entry)
        best = max(best, price) if pos.side > 0 else min(best, price)
        self._best[symbol] = best
        new, runner = exit_stop(sp, pos.side, pos.entry, self._stop0.get(symbol, sp.stop), best, price, atr, atr_pct,
                                self.broker.taker_fee, self._runner.get(symbol, False))
        self._runner[symbol] = runner
        if runner and pos.take_profit is not None:
            pos.take_profit = None
            self._log(ts, "runner", f"{symbol}: alvo cancelado, deixa correr")
        if new is not None:
            stop = max(pos.stop, new) if pos.side > 0 else min(pos.stop, new)
            if stop != pos.stop:
                pos.stop = stop
        return None

    def on_signal(self, ts: pd.Timestamp, symbol: str, side: int, setup: str, price: float) -> Decision:
        self.marks[symbol] = price
        self._roll_day(ts)
        d = self._check(ts, symbol, side, setup)
        if d is not None:
            self._log(ts, "recusada", f"{symbol} {setup} {side:+d}: {d.reason}")
            return d
        sp = self.policy.for_symbol(symbol)
        equity = self.equity()
        qty = equity * sp.risk_per_trade / (price * sp.stop)
        lev = min(sp.max_leverage, self.limits.max_leverage)
        qty = min(qty, self.broker.cash * lev * 0.95 / price)  # cabe na margem livre
        stop = price * (1 - side * sp.stop)
        tp = None if sp.take_profit_r is None else price * (1 + side * sp.take_profit_r * sp.stop)
        try:
            self.broker.open(symbol, side, qty, price, lev, stop, ts, tp)
        except (InsufficientMargin, ValueError) as exc:
            self._log(ts, "recusada", f"{symbol}: {exc}")
            return Decision(False, f"corretora: {exc}")
        self._best[symbol], self._stop0[symbol], self._runner[symbol] = price, sp.stop, False
        self._log(ts, "entrada", f"{symbol} {setup} {side:+d} qty {qty:.6g} @ {price:.6g} stop {stop:.6g}")
        return Decision(True, "ok", qty)

    def on_news(self, ts: pd.Timestamp, kind: str, symbols: tuple[str, ...], sentiment: float,
                severity: float) -> list[str]:
        done = []
        for r in self.policy.reactions:
            if not r.matches(kind, symbols, sentiment, severity):
                continue
            targets = [s for s in (self.policy.symbols if "*" in r.symbols else r.symbols)]
            targets = sorted(set(targets) | ({s for s in self.broker.positions} if "*" in r.symbols else set()))
            for sym in targets:
                if r.action == "block_entries":
                    until = ts + pd.Timedelta(minutes=r.minutes)
                    self.blocked_until[sym] = max(self.blocked_until.get(sym, until), until)
                    done.append(f"bloqueia {sym} por {r.minutes} min")
                elif self.broker.position(sym) is None:
                    continue
                elif r.action == "flatten":
                    self._close(ts, sym, "news_flatten")
                    done.append(f"zera {sym}")
                elif r.action == "reduce":
                    self._reduce(ts, sym, r.factor)
                    done.append(f"reduz {sym} para {r.factor:.0%}")
                elif r.action == "tighten_stop":
                    pos, price = self.broker.position(sym), self.marks.get(sym)
                    if price is not None:
                        dist = self._stop0.get(sym, 0.01) * r.factor
                        new = price * (1 - pos.side * dist)
                        pos.stop = max(pos.stop, new) if pos.side > 0 else min(pos.stop, new)
                        done.append(f"aperta stop {sym}")
        if done:
            self._log(ts, "notícia", f"{kind} {symbols} sent {sentiment:+.2f} sev {severity:.2f}: " + "; ".join(done))
        return done

    # internos

    def equity(self) -> float:
        return self.broker.equity(self.marks)

    def _check(self, ts, symbol, side, setup) -> Decision | None:
        p = self.policy
        if p.kill:
            return Decision(False, "kill ligado")
        if self.expired(ts):
            return Decision(False, "política vencida")
        if not p.for_symbol(symbol).allows(side, setup):
            return Decision(False, f"política não permite ({p.for_symbol(symbol).bias}, setups {p.for_symbol(symbol).setups})")
        if ts < self.blocked_until.get(symbol, ts):
            return Decision(False, "bloqueado por notícia")
        if self.broker.position(symbol) is not None:
            return Decision(False, "já posicionado")
        if self.equity() <= self._day_start * (1 - p.daily_loss_limit):
            return Decision(False, "limite de perda do dia")
        return None

    def _roll_day(self, ts: pd.Timestamp) -> None:
        day = ts.normalize()
        if day != self._day:
            self._day, self._day_start = day, self.equity()

    def _close(self, ts, symbol, why, price=None) -> str:
        price = self.marks.get(symbol) if price is None else price
        pnl = self.broker.close(symbol, price, ts, why)
        for d in (self._best, self._stop0, self._runner):
            d.pop(symbol, None)
        self._log(ts, "saída", f"{symbol} {why} @ {price:.6g} pnl {pnl:+.2f}")
        return why

    def _reduce(self, ts, symbol, keep: float) -> None:
        pos = self.broker.position(symbol)
        if keep <= 0:
            self._close(ts, symbol, "news_reduce")
            return
        # fecha tudo e reabre a fração mantida, com o mesmo stop (o broker não tem fechamento parcial)
        side, qty, lev, stop, tp = pos.side, pos.qty * keep, pos.leverage, pos.stop, pos.take_profit
        best, stop0, runner = self._best.get(symbol), self._stop0.get(symbol), self._runner.get(symbol)
        price = self.marks[symbol]
        self._close(ts, symbol, "news_reduce")
        try:
            self.broker.open(symbol, side, qty, price, lev, stop, ts, tp)
            self._best[symbol], self._stop0[symbol], self._runner[symbol] = best or price, stop0 or 0.01, bool(runner)
        except (InsufficientMargin, ValueError):
            pass

    def _log(self, ts, kind, msg) -> None:
        self.journal.append((ts, kind, msg))
