"""Indicadores incrementais: cada candle fechado atualiza em O(1), sem recalcular a série.

Mesmos números da versão vetorizada (pandas), conferidos nos testes. Memória fixa:
só o necessário para a janela de cada indicador.

Fibonacci: topos e fundos confirmados por um zigue-zague causal (o ponto só vira
topo depois que o preço recua `swing_pct` dele). Níveis de retração do último
movimento confirmado: 23,6%, 38,2%, 50%, 61,8% e 78,6%.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)
NAN = math.nan


class SMA:
    def __init__(self, n: int) -> None:
        self.n, self.buf, self.total = n, deque(maxlen=n), 0.0

    def update(self, x: float) -> float:
        if len(self.buf) == self.n:
            self.total -= self.buf[0]
        self.buf.append(x)
        self.total += x
        return self.value

    @property
    def value(self) -> float:
        return self.total / self.n if len(self.buf) == self.n else NAN


class RollingStd:
    """Desvio padrão amostral (ddof=1, como o pandas) por somas; recalibra de tempos em tempos contra erro."""

    def __init__(self, n: int) -> None:
        self.n, self.buf, self.s, self.s2, self.k = n, deque(maxlen=n), 0.0, 0.0, 0

    def update(self, x: float) -> float:
        if len(self.buf) == self.n:
            old = self.buf[0]
            self.s -= old
            self.s2 -= old * old
        self.buf.append(x)
        self.s += x
        self.s2 += x * x
        self.k += 1
        if self.k % 1000 == 0:
            self.s, self.s2 = sum(self.buf), sum(v * v for v in self.buf)
        return self.value

    @property
    def value(self) -> float:
        if len(self.buf) < self.n:
            return NAN
        var = (self.s2 - self.s * self.s / self.n) / (self.n - 1)
        return math.sqrt(max(var, 0.0))


class EMA:
    """span=n, adjust=False (como pandas ewm(span=n, adjust=False))."""

    def __init__(self, n: int = 0, alpha: float | None = None) -> None:
        self.alpha = alpha if alpha is not None else 2 / (n + 1)
        self.value = NAN

    def update(self, x: float) -> float:
        self.value = x if self.value != self.value else self.value + self.alpha * (x - self.value)
        return self.value


class RSI:
    """Wilder (ewm alpha=1/n, adjust=False), igual ao _rsi do day trade."""

    def __init__(self, n: int = 14) -> None:
        self.up, self.down, self.prev = EMA(alpha=1 / n), EMA(alpha=1 / n), NAN

    def update(self, close: float) -> float:
        if self.prev == self.prev:
            d = close - self.prev
            self.up.update(max(d, 0.0))
            self.down.update(max(-d, 0.0))
        self.prev = close
        return self.value

    @property
    def value(self) -> float:
        u, d = self.up.value, self.down.value
        if u != u or d != d:
            return NAN
        return 100.0 if d == 0 else 100 - 100 / (1 + u / d)


class ATR:
    """Média simples do true range em n candles (como o _atr do day trade)."""

    def __init__(self, n: int = 14) -> None:
        self.sma, self.prev_close = SMA(n), NAN

    def update(self, high: float, low: float, close: float) -> float:
        pc = self.prev_close
        tr = high - low if pc != pc else max(high - low, abs(high - pc), abs(low - pc))
        self.prev_close = close
        return self.sma.update(tr)

    @property
    def value(self) -> float:
        return self.sma.value


class RollingExtreme:
    """Máximo ou mínimo de n valores em O(1) amortizado (fila monotônica)."""

    def __init__(self, n: int, kind: str = "max") -> None:
        self.n, self.i, self.q, self.sign = n, 0, deque(), 1 if kind == "max" else -1

    def update(self, x: float) -> float:
        while self.q and self.sign * self.q[-1][1] <= self.sign * x:
            self.q.pop()
        self.q.append((self.i, x))
        if self.q[0][0] <= self.i - self.n:
            self.q.popleft()
        self.i += 1
        return self.value

    @property
    def value(self) -> float:
        return self.q[0][1] if self.i >= self.n else NAN


class Percentile:
    """Percentil do último valor dentro da janela (para o ATR do "deixa correr")."""

    def __init__(self, n: int) -> None:
        self.buf = deque(maxlen=n)

    def update(self, x: float) -> float:
        if x == x:
            self.buf.append(x)
        return self.value

    @property
    def value(self) -> float:
        if len(self.buf) < 20:
            return NAN
        last = self.buf[-1]
        return sum(v <= last for v in self.buf) / len(self.buf)


class Fibonacci:
    """Zigue-zague causal e níveis de retração do último movimento confirmado."""

    def __init__(self, swing_pct: float = 0.02) -> None:
        self.pct = swing_pct
        self.direction = 0  # +1 subindo desde um fundo confirmado, -1 caindo desde um topo
        self.ext_price = NAN  # extremo do movimento atual (ainda não confirmado)
        self.last_pivot = NAN  # último pivô confirmado
        self.swings: deque = deque(maxlen=10)  # (preço do pivô, tipo) confirmados

    def update(self, high: float, low: float) -> None:
        if self.ext_price != self.ext_price:
            self.ext_price, self.last_pivot = high, low
            self.direction = 1
            return
        if self.direction >= 0:
            if high > self.ext_price:
                self.ext_price = high
            elif low <= self.ext_price * (1 - self.pct):  # recuou o bastante: o topo está confirmado
                self.swings.append((self.ext_price, "top"))
                self.last_pivot, self.ext_price, self.direction = self.ext_price, low, -1
        else:
            if low < self.ext_price:
                self.ext_price = low
            elif high >= self.ext_price * (1 + self.pct):
                self.swings.append((self.ext_price, "bottom"))
                self.last_pivot, self.ext_price, self.direction = self.ext_price, high, 1

    def levels(self) -> dict[str, float]:
        """Retrações do último movimento entre dois pivôs confirmados."""
        if len(self.swings) < 2:
            return {}
        (a, _), (b, kind_b) = self.swings[-2], self.swings[-1]
        # movimento a → b; retração mede de volta a partir de b
        return {f"fib_{lvl}": b - (b - a) * lvl for lvl in FIB_LEVELS} | {"fib_from": a, "fib_to": b,
                                                                          "fib_up": float(kind_b == "top")}


@dataclass
class Bar:
    ts: object
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorSet:
    """Todos os indicadores de um ativo/tempo gráfico, atualizados a cada candle fechado."""

    def __init__(self, swing_pct: float = 0.02) -> None:
        self.sma = {n: SMA(n) for n in (20, 50, 200)}
        self.ema = {n: EMA(n) for n in (9, 21)}
        self.macd_fast, self.macd_slow, self.macd_signal = EMA(12), EMA(26), EMA(9)
        self.rsi = RSI(14)
        self.atr = ATR(14)
        self.atr_pct = Percentile(288)
        self.bb_mid, self.bb_std = SMA(20), RollingStd(20)
        self.stoch_hi, self.stoch_lo, self.stoch_d = RollingExtreme(14, "max"), RollingExtreme(14, "min"), SMA(3)
        self.don_hi, self.don_lo = RollingExtreme(20, "max"), RollingExtreme(20, "min")
        self.obv, self.prev_close = 0.0, NAN
        self.vwap_pv, self.vwap_v, self.vwap_day = 0.0, 0.0, None
        self.fib = Fibonacci(swing_pct)
        self.last: dict[str, float] = {}

    def update(self, bar: Bar) -> dict[str, float]:
        c = bar.close
        out: dict[str, float] = {"close": c}
        for n, s in self.sma.items():
            out[f"sma{n}"] = s.update(c)
        for n, e in self.ema.items():
            out[f"ema{n}"] = e.update(c)
        macd = self.macd_fast.update(c) - self.macd_slow.update(c)
        out["macd"], out["macd_signal"] = macd, self.macd_signal.update(macd)
        out["macd_hist"] = macd - out["macd_signal"]
        out["rsi14"] = self.rsi.update(c)
        out["atr14"] = self.atr.update(bar.high, bar.low, c)
        out["atr_pct"] = self.atr_pct.update(out["atr14"])
        mid, sd = self.bb_mid.update(c), self.bb_std.update(c)
        out["bb_up"], out["bb_mid"], out["bb_low"] = mid + 2 * sd, mid, mid - 2 * sd
        hi, lo = self.stoch_hi.update(bar.high), self.stoch_lo.update(bar.low)
        k = 100 * (c - lo) / (hi - lo) if hi == hi and hi > lo else NAN
        out["stoch_k"] = k
        out["stoch_d"] = self.stoch_d.update(k) if k == k else NAN
        out["donchian_hi"], out["donchian_lo"] = self.don_hi.update(bar.high), self.don_lo.update(bar.low)
        if self.prev_close == self.prev_close:
            self.obv += bar.volume if c > self.prev_close else -bar.volume if c < self.prev_close else 0.0
        self.prev_close = c
        out["obv"] = self.obv
        day = getattr(bar.ts, "date", lambda: None)()
        if day != self.vwap_day:
            self.vwap_pv, self.vwap_v, self.vwap_day = 0.0, 0.0, day
        typical = (bar.high + bar.low + c) / 3
        self.vwap_pv += typical * bar.volume
        self.vwap_v += bar.volume
        out["vwap"] = self.vwap_pv / self.vwap_v if self.vwap_v > 0 else NAN
        self.fib.update(bar.high, bar.low)
        out.update(self.fib.levels())
        self.last = out
        return out
