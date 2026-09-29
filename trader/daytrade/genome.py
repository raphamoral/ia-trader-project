"""Genoma de day trade: estratégia gráfica + genes de autonomia diária.

Os sinais são vetorizados sobre a série, mas causais: o valor no candle i
só usa dados até i (há teste para cada estratégia).
Eventos: +1 entrar comprado, -1 entrar vendido, 0 zerar, NaN manter.

Autonomia diária (genes; o Criador pode mudar todos):
  max_trades_per_day  tentativas por sessão
  daily_stop          stop diário do próprio agente (o gate congelado é o teto)
  daily_target        meta do dia; atingida, para até a próxima sessão
  risk                fração do patrimônio arriscada até o stop, por operação

Escalada de risco (genes; sempre limitada pelos tetos congelados do gate):
  after_win_mult      multiplica o tamanho após uma operação vencedora no dia
                      (>1 = anti-martingale)
  after_loss_mult     multiplica o tamanho após uma perdedora (>1 = martingale)
  goal_pressure       quanto aumenta o risco quando está atrás da meta diária
                      no fim da sessão
"""
from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from trader.daytrade.sessions import session_keys

# estratégia → parâmetros inteiros e faixas; descrição usada no prompt da LLM
DAY_STRATEGIES: dict[str, dict[str, tuple[int, int]]] = {
    "momentum": {"lookback": (3, 48), "threshold_bps": (10, 300)},
    "orb": {"range_bars": (2, 24)},
    "vwap_revert": {"band_bps": (10, 300)},
    "ema_cross": {"fast": (3, 50), "slow": (8, 200)},
    "macd": {"fast": (4, 30), "slow": (10, 80), "signal": (3, 20)},
    "bb_breakout": {"period": (10, 60), "k10": (10, 35)},
    "bb_revert": {"period": (10, 60), "k10": (15, 35)},
    "donchian": {"period": (10, 120)},
    "supertrend": {"period": (5, 40), "mult10": (10, 50)},
    "rsi_divergence": {"period": (5, 30), "lookback": (5, 60)},
    "engulfing": {"trend": (10, 200), "min_body_bps": (1, 100)},
    "hammer": {"trend": (10, 200), "wick10": (15, 50)},
    "inside_bar": {"min_range_bps": (5, 200)},
    "pivot_breakout": {"buffer_bps": (0, 50)},
    "vol_expansion": {"threshold10": (11, 30), "lookback": (5, 60)},
    "expansion_fade": {"threshold10": (12, 35), "volume_z10": (10, 40)},
    "squeeze_release": {"squeeze_pct": (3, 30), "threshold10": (10, 25)},
    "trend_pullback": {"fast": (5, 30), "slow": (30, 200)},
    "news_event": {"min_abs_sent10": (2, 9)},
    "grid": {"step_bps": (10, 200), "levels": (1, 5)},
}
CATALOG: dict[str, str] = {
    "momentum": "enter in the direction of the N-bar return when it crosses ±threshold_bps; exit when it flips sign",
    "orb": "opening range breakout: high/low of the first range_bars candles of the session, enter on breakout",
    "vwap_revert": "fade moves of band_bps away from the session VWAP; exit when price crosses VWAP",
    "ema_cross": "fast EMA crossing slow EMA: long above, short below (trend following)",
    "macd": "MACD line (EMA fast - EMA slow) crossing its signal EMA (trend following)",
    "bb_breakout": "close breaking outside Bollinger bands (period, k10/10 std devs); exit at the middle band",
    "bb_revert": "close back inside after piercing a Bollinger band: fade the move; exit at the middle band",
    "donchian": "turtle breakout: close above the prior period-high goes long, below the prior period-low goes short",
    "supertrend": "ATR(period) trailing band with multiplier mult10/10; enter when the trend flips",
    "rsi_divergence": "new low of the last lookback bars with RSI(period) above its prior low (bullish), mirror for bearish",
    "engulfing": "engulfing candle against the trend (price vs SMA(trend)); body at least min_body_bps",
    "hammer": "hammer after a decline / shooting star after a rise; wick >= wick10/10 x body",
    "inside_bar": "inside bar; enter when price breaks the mother bar (range >= min_range_bps)",
    "pivot_breakout": "classic pivot from the previous session: break above R1 long, below S1 short (buffer_bps)",
    "vol_expansion": "candle range >= threshold10/10 x max range of the last lookback candles: STOP-ENTRY beyond that candle in its direction",
    "expansion_fade": "an expansion candle (>= threshold10/10) with climax volume (z >= volume_z10/10), and the next candle fails to extend: STOP-ENTRY against it",
    "squeeze_release": "ATR percentile <= squeeze_pct% on the previous candle, then range expansion >= threshold10/10: STOP-ENTRY in the break direction",
    "trend_pullback": "trend by EMA(fast) vs EMA(slow); enter when price pulls back to EMA(fast) and closes back in the trend direction",
    "news_event": "trade in the direction of fresh high-confidence news (|sentiment| >= min_abs_sent10/10 and confidence >= news_follow)",
    "grid": "bounded grid around the session VWAP: LIMIT entries every step_bps, at most `levels` add-ons, exit at VWAP; ruin still kills",
}

# estilo → timeframe, estratégias elegíveis, tipo de ordem de entrada
STYLES: dict[str, dict] = {
    "scalping": {"timeframe": "1m", "order": "limit", "max_hold": (2, 30),
                 "strategies": ["momentum", "bb_revert", "vwap_revert", "orb", "inside_bar", "vol_expansion"]},
    "momentum": {"timeframe": "15m", "order": "market", "max_hold": (0, 96),
                 "strategies": ["momentum", "donchian", "bb_breakout", "orb", "macd", "ema_cross"]},
    "mean_reversion": {"timeframe": "15m", "order": "limit", "max_hold": (0, 48),
                       "strategies": ["vwap_revert", "bb_revert", "rsi_divergence"]},
    "trend_pullback": {"timeframe": "15m", "order": "market", "max_hold": (0, 96),
                       "strategies": ["trend_pullback", "supertrend", "ema_cross"]},
    "session_breakout": {"timeframe": "15m", "order": "stop", "max_hold": (0, 96),
                         "strategies": ["orb", "pivot_breakout"]},
    "news": {"timeframe": "15m", "order": "market", "max_hold": (0, 32), "strategies": ["news_event"]},
    "fade": {"timeframe": "15m", "order": "stop", "max_hold": (0, 32),
             "strategies": ["expansion_fade", "hammer", "engulfing"]},
    "grid": {"timeframe": "15m", "order": "limit", "max_hold": (0, 96), "strategies": ["grid"]},
    "expansion": {"timeframe": "15m", "order": "stop", "max_hold": (0, 48),
                  "strategies": ["vol_expansion", "squeeze_release", "expansion_fade"]},
    "any": {"timeframe": "15m", "order": "market", "max_hold": (0, 96), "strategies": None},  # catálogo todo
}
STYLE_DESCRIPTIONS = {
    "scalping": "1m candles, tiny stops/targets, LIMIT (maker 0.02%) entries, holds minutes",
    "momentum": "intraday momentum/breakout on 15m",
    "mean_reversion": "range trading back to VWAP / Bollinger mean, LIMIT entries",
    "trend_pullback": "trend following, entering on pullbacks",
    "session_breakout": "opening range / pivot breakout with STOP-ENTRY orders",
    "news": "trades fresh news in the news direction",
    "fade": "reversal after exhausted extreme moves (STOP-ENTRY against)",
    "grid": "bounded limit-order grid around VWAP",
    "expansion": "volatility expansion / squeeze release with STOP-ENTRY orders",
    "any": "any strategy of the catalog on 15m",
}
STAND_ASIDE = "stand_aside"


def eligible(style: str) -> list[str]:
    return STYLES[style]["strategies"] or [k for k in DAY_STRATEGIES if k != "grid"]
DIRECTIONS = ("long", "short", "both")
LEVERAGE_RANGE = (1.0, 10.0)  # o gate corta no teto dele
STOP_RANGE = (0.002, 0.05)
TP_RANGE = (1.0, 5.0)
RISK_RANGE = (0.002, 0.10)  # o gate corta no teto dele
COOLDOWN_RANGE = (0, 48)
MAX_TRADES_RANGE = (1, 20)
DAILY_STOP_RANGE = (0.005, 0.5)  # o backtest corta abaixo do limite do gate
DAILY_TARGET_RANGE = (0.01, 3.0)  # 1.0 = +100% no dia
MAX_HOLD_RANGE = (0, 96)  # candles; 0 = sem limite
MULT_RANGE = (0.25, 3.0)
PRESSURE_RANGE = (0.0, 3.0)


def _clamp(v: float, b: tuple[float, float]) -> float:
    return min(max(v, b[0]), b[1])


@dataclass(frozen=True)
class DayGenome:
    strategy: str
    params: dict[str, int] = field(default_factory=dict)
    direction: str = "both"
    leverage: float = 1.0
    stop: float = 0.01
    take_profit: float | None = None  # em R (múltiplos do stop)
    risk: float = 0.01
    cooldown: int = 0
    max_trades_per_day: int = 20
    daily_stop: float = 0.5
    daily_target: float | None = None
    max_hold: int = 0
    after_win_mult: float = 1.0
    after_loss_mult: float = 1.0
    goal_pressure: float = 0.0
    # estilo, execução e saída
    style: str = "any"
    trail: float | None = None  # trailing stop, fração do preço
    entry_offset: float = 0.0005  # limit abaixo/acima do fechamento; stop-entry além do gatilho
    order_expiry: int = 3  # candles até a ordem de entrada não executada expirar
    slippage: float = 0.0002  # em ordens stop-entry
    # meta-controlador: regime → estratégia ou stand_aside (None = sempre `strategy`)
    regime_map: dict[str, str] | None = None
    # notícias
    news_gate: float | None = None  # bloqueia entrada contra sentimento mais forte que isso
    news_follow: float | None = None  # confiança mínima para operar a favor de notícia fresca
    event_blackout: int = 0  # minutos antes de evento agendado sem novas entradas
    news_size_mult: float = 0.0  # escala o risco pela confiança da notícia
    black_weight: float = 0.0  # quanto as features black_ pesam (0 no braço só-confiável)

    def __post_init__(self) -> None:
        if self.strategy not in DAY_STRATEGIES:
            raise ValueError(f"estratégia desconhecida: {self.strategy!r}")
        if set(self.params) != set(DAY_STRATEGIES[self.strategy]):
            raise ValueError(f"{self.strategy} exige {sorted(DAY_STRATEGIES[self.strategy])}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction deve ser uma de {DIRECTIONS}")
        if self.style not in STYLES:
            raise ValueError(f"estilo desconhecido: {self.style!r}")
        if self.regime_map is not None:
            from trader.daytrade.regime import REGIMES

            for k, v in self.regime_map.items():
                if k not in REGIMES or (v != STAND_ASIDE and v not in DAY_STRATEGIES):
                    raise ValueError(f"regime_map inválido: {k} → {v}")

    @property
    def timeframe(self) -> str:
        return STYLES[self.style]["timeframe"]

    @property
    def order_type(self) -> str:
        """Tipo da ordem de entrada: setups com gatilho usam stop-entry, grid usa limit."""
        if self.strategy in STOP_ENTRY_STRATEGIES:
            return "stop"
        if self.strategy == "grid":
            return "limit"
        return STYLES[self.style]["order"]

    def strategies_used(self) -> list[str]:
        if self.regime_map is None:
            return [self.strategy]
        return sorted({v for v in self.regime_map.values() if v != STAND_ASIDE} | {self.strategy})

    def normalized(self) -> DayGenome:
        b = DAY_STRATEGIES[self.strategy]
        p = {k: int(round(_clamp(v, b[k]))) for k, v in self.params.items()}
        for fast, slow in (("fast", "slow"),):
            if fast in p and slow in p and p[fast] >= p[slow]:
                p[slow] = min(p[fast] + 5, b[slow][1])
                p[fast] = min(p[fast], p[slow] - 1)
        return replace(
            self,
            params=p,
            leverage=round(_clamp(self.leverage, LEVERAGE_RANGE), 2),
            stop=round(_clamp(self.stop, STOP_RANGE), 5),
            take_profit=None if self.take_profit is None else round(_clamp(self.take_profit, TP_RANGE), 2),
            risk=round(_clamp(self.risk, RISK_RANGE), 4),
            cooldown=int(_clamp(self.cooldown, COOLDOWN_RANGE)),
            max_trades_per_day=int(round(_clamp(self.max_trades_per_day, MAX_TRADES_RANGE))),
            daily_stop=round(_clamp(self.daily_stop, DAILY_STOP_RANGE), 4),
            daily_target=None if self.daily_target is None else round(_clamp(self.daily_target, DAILY_TARGET_RANGE), 3),
            max_hold=int(round(_clamp(self.max_hold, MAX_HOLD_RANGE))),
            after_win_mult=round(_clamp(self.after_win_mult, MULT_RANGE), 3),
            after_loss_mult=round(_clamp(self.after_loss_mult, MULT_RANGE), 3),
            goal_pressure=round(_clamp(self.goal_pressure, PRESSURE_RANGE), 3),
            trail=None if self.trail is None else round(_clamp(self.trail, STOP_RANGE), 5),
            entry_offset=round(_clamp(self.entry_offset, (0.0, 0.01)), 5),
            order_expiry=int(round(_clamp(self.order_expiry, (1, 20)))),
            slippage=round(_clamp(self.slippage, (0.0, 0.005)), 5),
            news_gate=None if self.news_gate is None else round(_clamp(self.news_gate, (0.05, 2.0)), 3),
            news_follow=None if self.news_follow is None else round(_clamp(self.news_follow, (0.1, 1.0)), 3),
            event_blackout=int(round(_clamp(self.event_blackout, (0, 240)))),
            news_size_mult=round(_clamp(self.news_size_mult, (0.0, 2.0)), 3),
            black_weight=round(_clamp(self.black_weight, (0.0, 1.0)), 3),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DayGenome:
        def opt(key: str) -> float | None:
            return None if d.get(key) is None else float(d[key])

        return cls(
            strategy=d["strategy"],
            params={k: int(v) for k, v in d["params"].items()},
            direction=d.get("direction", "both"),
            leverage=float(d.get("leverage", 1.0)),
            stop=float(d.get("stop", 0.01)),
            take_profit=opt("take_profit"),
            risk=float(d.get("risk", 0.01)),
            cooldown=int(d.get("cooldown", 0)),
            max_trades_per_day=int(d.get("max_trades_per_day", 20)),
            daily_stop=float(d.get("daily_stop", 0.5)),
            daily_target=opt("daily_target"),
            max_hold=int(d.get("max_hold", 0)),
            after_win_mult=float(d.get("after_win_mult", 1.0)),
            after_loss_mult=float(d.get("after_loss_mult", 1.0)),
            goal_pressure=float(d.get("goal_pressure", 0.0)),
            style=d.get("style", "any"),
            trail=opt("trail"),
            entry_offset=float(d.get("entry_offset", 0.0005)),
            order_expiry=int(d.get("order_expiry", 3)),
            slippage=float(d.get("slippage", 0.0002)),
            regime_map=d.get("regime_map"),
            news_gate=opt("news_gate"),
            news_follow=opt("news_follow"),
            event_blackout=int(d.get("event_blackout", 0)),
            news_size_mult=float(d.get("news_size_mult", 0.0)),
            black_weight=float(d.get("black_weight", 0.0)),
        ).normalized()

    def warmup_bars(self, bars_per_day: int) -> int:
        return 2 * bars_per_day + 3 * max(self.params.values()) + 2


STOP_ENTRY_STRATEGIES = {"vol_expansion", "expansion_fade", "squeeze_release"}


def random_regime_map(rng: random.Random, style: str) -> dict[str, str]:
    from trader.daytrade.regime import REGIMES

    options = eligible(style) + [STAND_ASIDE]
    return {r: rng.choice(options) for r in REGIMES}


def random_day_genome(
    rng: random.Random, strategy: str | None = None, style: str | None = None, styles: list[str] | None = None,
    news: bool = False, black: bool = False,
) -> DayGenome:
    style = style or rng.choice(styles or sorted(STYLES))
    strategy = strategy if strategy in eligible(style) else rng.choice(eligible(style))
    lo, hi = STYLES[style]["max_hold"]
    scalp = style == "scalping"
    g = DayGenome(
        strategy=strategy,
        params={k: rng.randint(lo, hi) for k, (lo, hi) in DAY_STRATEGIES[strategy].items()},
        direction=rng.choice(DIRECTIONS),
        leverage=rng.uniform(1, 5),
        stop=rng.uniform(0.003, 0.02),
        take_profit=rng.choice([None, rng.uniform(1, 4)]),
        risk=rng.uniform(0.003, 0.03),
        cooldown=rng.choice([0, rng.randint(1, 16)]),
        max_trades_per_day=rng.randint(1, 10),
        daily_stop=rng.uniform(0.01, 0.05),
        daily_target=rng.choice([None, rng.uniform(0.02, 1.0)]),
        max_hold=rng.choice([0, rng.randint(4, 48)]),
        after_win_mult=rng.choice([1.0, rng.uniform(0.5, 2.0)]),
        after_loss_mult=rng.choice([1.0, rng.uniform(0.5, 2.0)]),
        goal_pressure=rng.choice([0.0, rng.uniform(0.0, 2.0)]),
        style=style,
        trail=rng.choice([None, rng.uniform(0.002, 0.02)]),
        entry_offset=rng.uniform(0.0, 0.002),
        order_expiry=rng.randint(1, 6),
        slippage=rng.uniform(0.0, 0.001),
        regime_map=rng.choice([None, random_regime_map(rng, style)]),
        news_gate=rng.choice([None, rng.uniform(0.2, 1.0)]) if news else None,
        news_follow=(rng.uniform(0.3, 0.9) if style == "news" else rng.choice([None, rng.uniform(0.3, 0.9)])) if news else None,
        event_blackout=rng.choice([0, rng.randint(15, 120)]) if news else 0,
        news_size_mult=rng.choice([0.0, rng.uniform(0.0, 1.0)]) if news else 0.0,
        black_weight=rng.uniform(0.0, 1.0) if black else 0.0,
    )
    if scalp:  # alvos e stops minúsculos, segura minutos
        g = replace(g, stop=rng.uniform(0.002, 0.006), take_profit=rng.uniform(1.0, 2.5), max_hold=rng.randint(lo, hi),
                    max_trades_per_day=rng.randint(5, 20))
    elif g.max_hold:
        g = replace(g, max_hold=rng.randint(max(lo, 4), max(hi, 5)))
    return g.normalized()


def switch_style(g: DayGenome, style: str, rng: random.Random) -> DayGenome:
    """Troca de estilo: escolhe uma estratégia elegível e refaz o mapa de regimes."""
    new = random_day_genome(rng, style=style, news=g.news_gate is not None or g.news_follow is not None,
                            black=g.black_weight > 0)
    return replace(g, style=style, strategy=new.strategy, params=new.params, regime_map=new.regime_map,
                   stop=new.stop if style == "scalping" else g.stop, max_hold=new.max_hold).normalized()


def mutate_day(g: DayGenome, rng: random.Random, scale: float = 0.15, styles: list[str] | None = None,
               black_allowed: bool = True) -> DayGenome:
    def j(v: float) -> float:
        return v * math.exp(rng.gauss(0, scale))

    def jo(v: float | None) -> float | None:
        return None if v is None else j(v)

    if styles and rng.random() < scale / 3:
        g = switch_style(g, rng.choice(styles), rng)
    regime_map = g.regime_map
    if regime_map is not None and rng.random() < scale:
        from trader.daytrade.regime import REGIMES

        regime_map = dict(regime_map)
        regime_map[rng.choice(REGIMES)] = rng.choice(eligible(g.style) + [STAND_ASIDE])
    direction = g.direction if rng.random() > scale / 2 else rng.choice(DIRECTIONS)
    g = replace(
        g,
        regime_map=regime_map,
        trail=jo(g.trail),
        entry_offset=g.entry_offset * math.exp(rng.gauss(0, scale)),
        order_expiry=round(j(g.order_expiry)),
        slippage=g.slippage,
        news_gate=jo(g.news_gate),
        news_follow=jo(g.news_follow),
        event_blackout=0 if g.event_blackout == 0 else round(j(g.event_blackout)),
        news_size_mult=g.news_size_mult * math.exp(rng.gauss(0, scale)),
        black_weight=min(1.0, max(0.0, g.black_weight + rng.gauss(0, scale / 2))) if black_allowed else 0.0,
    )
    return replace(
        g,
        params={k: j(v) for k, v in g.params.items()},
        direction=direction,
        leverage=j(g.leverage),
        stop=j(g.stop),
        take_profit=None if g.take_profit is None else j(g.take_profit),
        risk=j(g.risk),
        cooldown=round(j(g.cooldown + 1)) - 1,
        max_trades_per_day=round(j(g.max_trades_per_day)),
        daily_stop=j(g.daily_stop),
        daily_target=None if g.daily_target is None else j(g.daily_target),
        max_hold=0 if g.max_hold == 0 else round(j(g.max_hold)),
        after_win_mult=j(g.after_win_mult),
        after_loss_mult=j(g.after_loss_mult),
        goal_pressure=g.goal_pressure + rng.gauss(0, scale),
    ).normalized()


# indicadores causais


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _cross_up(x: pd.Series, level) -> pd.Series:
    prev = level.shift(1) if isinstance(level, pd.Series) else level
    return (x > level) & (x.shift(1) <= prev)


def _cross_down(x: pd.Series, level) -> pd.Series:
    prev = level.shift(1) if isinstance(level, pd.Series) else level
    return (x < level) & (x.shift(1) >= prev)


def _sign_flip(x: pd.Series) -> pd.Series:
    return (np.sign(x) != np.sign(x.shift(1))) & x.notna() & x.shift(1).notna()


def _supertrend_flips(h: pd.Series, l: pd.Series, c: pd.Series, n: int, mult: float) -> tuple[np.ndarray, np.ndarray]:
    atr = _atr(h, l, c, n).to_numpy()
    hl2 = ((h + l) / 2).to_numpy()
    close = c.to_numpy()
    upper, lower = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = upper.copy(), lower.copy()
    trend = np.zeros(len(close))
    for i in range(1, len(close)):
        if np.isnan(atr[i]):
            continue
        if not np.isnan(fu[i - 1]):
            fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        prev = trend[i - 1] or 1
        if close[i] > fu[i - 1] if not np.isnan(fu[i - 1]) else False:
            trend[i] = 1
        elif close[i] < fl[i - 1] if not np.isnan(fl[i - 1]) else False:
            trend[i] = -1
        else:
            trend[i] = prev
    t = pd.Series(trend)
    started = t.shift(1).fillna(0) != 0
    return ((t == 1) & (t.shift(1) == -1) & started).to_numpy(), ((t == -1) & (t.shift(1) == 1) & started).to_numpy()


def signals(
    candles: pd.DataFrame,
    g: DayGenome,
    session_end: dt.time = dt.time(0, 0),
    strategy: str | None = None,
    regime: pd.DataFrame | None = None,
    news: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Eventos por candle (NaN = manter). Causal.

    `strategy` sobrepõe a do genoma (usado pelo meta-controlador); os parâmetros
    vêm do genoma quando a estratégia é a dele, senão do ponto médio das faixas.
    """
    strategy = strategy or g.strategy
    o, c, h, l = candles["open"], candles["close"], candles["high"], candles["low"]
    key = session_keys(candles["ts"], session_end)
    n_in_session = candles.groupby(key).cumcount()
    ev = pd.Series(np.nan, index=candles.index)
    p = g.params if strategy == g.strategy else {k: (lo + hi) // 2 for k, (lo, hi) in DAY_STRATEGIES[strategy].items()}
    long_ = short = exit_ = None
    if regime is None and strategy in ("vol_expansion", "expansion_fade", "squeeze_release"):
        from trader.daytrade.regime import regime_features

        regime = regime_features(candles, session_end=session_end)

    match strategy:
        case "momentum":
            thr = p["threshold_bps"] / 1e4
            r = c / c.shift(p["lookback"]) - 1
            exit_ = _sign_flip(r)
            long_, short = _cross_up(r, thr), _cross_down(r, -thr)
        case "orb":
            n = p["range_bars"]
            in_range = n_in_session < n
            rh = h.where(in_range).groupby(key).cummax().groupby(key).ffill()
            rl = l.where(in_range).groupby(key).cummin().groupby(key).ffill()
            valid = n_in_session >= n
            long_ = valid & (c > rh) & (c.shift(1) <= rh)
            short = valid & (c < rl) & (c.shift(1) >= rl)
        case "vwap_revert":
            band = p["band_bps"] / 1e4
            typical = (h + l + c) / 3
            vol = candles["volume"].where(candles["volume"] > 0, 1e-9)
            vwap = (typical * vol).groupby(key).cumsum() / vol.groupby(key).cumsum()
            dev = c / vwap - 1
            valid = n_in_session >= 4
            exit_ = valid & (key == key.shift(1)) & (np.sign(dev) != np.sign(dev.shift(1)))
            long_, short = valid & _cross_down(dev, -band), valid & _cross_up(dev, band)
        case "ema_cross":
            diff = _ema(c, p["fast"]) - _ema(c, p["slow"])
            long_, short = _cross_up(diff, 0.0), _cross_down(diff, 0.0)
        case "macd":
            line = _ema(c, p["fast"]) - _ema(c, p["slow"])
            hist = line - _ema(line, p["signal"])
            long_, short = _cross_up(hist, 0.0), _cross_down(hist, 0.0)
        case "bb_breakout" | "bb_revert":
            mid = c.rolling(p["period"]).mean()
            sd = c.rolling(p["period"]).std()
            k = p["k10"] / 10
            up, lo = mid + k * sd, mid - k * sd
            exit_ = _sign_flip(c - mid)
            if strategy == "bb_breakout":
                long_, short = _cross_up(c, up), _cross_down(c, lo)
            else:  # furou a banda e voltou para dentro: aposta na volta à média
                long_, short = _cross_up(c, lo), _cross_down(c, up)
        case "donchian":
            n = p["period"]
            hi, lo = h.shift(1).rolling(n).max(), l.shift(1).rolling(n).min()
            long_, short = _cross_up(c, hi), _cross_down(c, lo)
        case "supertrend":
            up_flip, down_flip = _supertrend_flips(h, l, c, p["period"], p["mult10"] / 10)
            long_, short = pd.Series(up_flip, index=c.index), pd.Series(down_flip, index=c.index)
        case "rsi_divergence":
            rsi = _rsi(c, p["period"])
            n = p["lookback"]
            prev_low, prev_high = c.shift(1).rolling(n).min(), c.shift(1).rolling(n).max()
            rsi_low, rsi_high = rsi.shift(1).rolling(n).min(), rsi.shift(1).rolling(n).max()
            long_ = (c < prev_low) & (rsi > rsi_low)
            short = (c > prev_high) & (rsi < rsi_high)
        case "engulfing":
            sma = c.rolling(p["trend"]).mean()
            body = (c - o).abs() / c
            po, pc = o.shift(1), c.shift(1)
            big = body >= p["min_body_bps"] / 1e4
            bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)
            bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)
            long_ = big & bull & (pc < sma.shift(1))
            short = big & bear & (pc > sma.shift(1))
        case "hammer":
            sma = c.rolling(p["trend"]).mean()
            body = (c - o).abs().clip(lower=c * 1e-4)
            lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
            upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
            w = p["wick10"] / 10
            long_ = (lower_wick >= w * body) & (upper_wick <= body) & (c < sma)
            short = (upper_wick >= w * body) & (lower_wick <= body) & (c > sma)
        case "inside_bar":
            mh, ml = h.shift(2), l.shift(2)
            inside = (h.shift(1) < mh) & (l.shift(1) > ml) & ((mh - ml) / c.shift(2) >= p["min_range_bps"] / 1e4)
            long_ = inside & (c > mh)
            short = inside & (c < ml)
        case "pivot_breakout":
            day = pd.DataFrame({"h": h, "l": l, "c": c, "key": key}).groupby("key").agg(
                h=("h", "max"), l=("l", "min"), c=("c", "last")
            )
            prev = day.shift(1)  # só a sessão anterior, já encerrada
            pivot = (prev["h"] + prev["l"] + prev["c"]) / 3
            r1 = (2 * pivot - prev["l"]).reindex(key).to_numpy()
            s1 = (2 * pivot - prev["h"]).reindex(key).to_numpy()
            buf = p["buffer_bps"] / 1e4
            r1s, s1s = pd.Series(r1 * (1 + buf), index=c.index), pd.Series(s1 * (1 - buf), index=c.index)
            same = key == key.shift(1)
            long_, short = same & _cross_up(c, r1s), same & _cross_down(c, s1s)
        case "vol_expansion":
            rng_ = h - l
            exp = rng_ / rng_.shift(1).rolling(p["lookback"], min_periods=p["lookback"]).max()
            fired = exp >= p["threshold10"] / 10
            long_, short = fired & (c > o), fired & (c < o)
        case "expansion_fade":
            exp = regime["range_expansion"]
            climax = (exp.shift(1) >= p["threshold10"] / 10) & (regime["volume_z"].shift(1) >= p["volume_z10"] / 10)
            up_bar, down_bar = c.shift(1) > o.shift(1), c.shift(1) < o.shift(1)
            # o candle seguinte (i) não estende a expansão: entra contra
            short = climax & up_bar & (h <= h.shift(1))
            long_ = climax & down_bar & (l >= l.shift(1))
        case "squeeze_release":
            squeezed = regime["atr_pct"].shift(1) <= p["squeeze_pct"] / 100
            fired = squeezed & (regime["range_expansion"] >= p["threshold10"] / 10)
            long_, short = fired & (c > o), fired & (c < o)
        case "trend_pullback":
            ef, es = _ema(c, p["fast"]), _ema(c, p["slow"])
            up, down = ef > es, ef < es
            long_ = up & (l <= ef) & (c > ef) & (c > o)
            short = down & (h >= ef) & (c < ef) & (c < o)
            exit_ = _sign_flip(ef - es)
        case "news_event":
            if news is None:
                long_ = short = pd.Series(False, index=c.index)
            else:
                thr = p["min_abs_sent10"] / 10
                conf_min = g.news_follow if g.news_follow is not None else 0.5
                s_ = pd.Series(news["sentiment"], index=c.index)
                ok = pd.Series(news["fresh"] & (news["confidence"] >= conf_min), index=c.index)
                long_, short = ok & _cross_up(s_, thr), ok & _cross_down(s_, -thr)
        case "grid":
            long_ = short = pd.Series(False, index=c.index)  # o grid é conduzido pelo backtest
        case _:
            raise AssertionError(strategy)

    if exit_ is not None:
        ev[exit_.fillna(False).astype(bool)] = 0
    ev[long_.fillna(False).astype(bool)] = 1
    ev[short.fillna(False).astype(bool)] = -1
    if g.direction == "long":
        ev[ev == -1] = 0
    elif g.direction == "short":
        ev[ev == 1] = 0
    return ev.to_numpy(dtype=float)
