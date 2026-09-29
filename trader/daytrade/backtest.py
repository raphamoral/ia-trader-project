"""Backtest intradiário com alavancagem, tipos de ordem, regimes e notícias.

Regras de tempo (nunca decidir e executar no mesmo candle):
  - a decisão sai no fechamento do candle i;
  - ordem a mercado executa na abertura de i+1;
  - LIMIT (maker): executa se i+1 abrir além do preço, ou se o candle
    NEGOCIAR ATRAVÉS do preço (mínima estritamente abaixo, com o desconto de
    fila `queue_haircut`); expira após `order_expiry` candles;
  - STOP-ENTRY: executa quando o candle negocia através do gatilho, no
    gatilho + slippage (ou na abertura, se abrir além dele).
Dentro de cada candle, stop, alvo e liquidação são checados pela
máxima/mínima, de forma pessimista: se stop e alvo cabem no mesmo candle, vale
o stop; num candle em que a entrada aconteceu no meio, só o stop é checado.

Regras do próprio agente por sessão (genes): tentativas, stop diário, meta
diária, escalada de risco. Ruína: sem capital para a menor ordem; a conta
para ali.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trader.daytrade.broker import FuturesPaperBroker, InsufficientMargin, Trade
from trader.daytrade.features import news_view
from trader.daytrade.gate import Close, DaySignal, DayTradeLimits, DayTradeRiskGate, Open
from trader.daytrade.genome import STAND_ASIDE, DayGenome, signals
from trader.daytrade.market import NO_MINIMUM, SymbolSpec
from trader.daytrade.regime import REGIMES, regime_features
from trader.daytrade.sessions import session_keys
from trader.domain import parse_timeframe
from trader.evolution.fitness import risk_adjusted

DEFAULT_FUNDING_RATE = 0.0001  # 0,01% a cada 8h, usado só se não houver histórico
FUNDING_HOURS = (0, 8, 16)
GOAL_DAY = 1.0  # +100% no dia: a meta do usuário
QUEUE_HAIRCUT = 0.0002  # limit só executa se o preço passar 2 bps além (fila na frente)


@dataclass(frozen=True)
class KellyCap:
    """Fração de Kelly para o risco por operação, com encolhimento para poucas operações.

    Com resultados em R (r = lucro ÷ risco até o stop), arriscar a fração f do
    patrimônio rende f·r por operação; o f que maximiza o crescimento é
    aproximadamente E[r] / E[r²]. O encolhimento soma `prior_n` operações
    fictícias com média 0 e E[r²] = 1: sem histórico, o limite é zero e só
    resta o risco de sondagem `probe`.
    """

    fraction: float = 0.5  # meio Kelly
    prior_n: float = 30.0
    probe: float = 0.002  # risco mínimo para continuar aprendendo

    def full_kelly(self, r: list[float]) -> float:
        n = len(r)
        mean = sum(r) / (n + self.prior_n)
        second = (sum(x * x for x in r) + self.prior_n) / (n + self.prior_n)
        return max(0.0, mean / second)

    def cap(self, r: list[float]) -> float:
        return max(self.probe, self.fraction * self.full_kelly(r))


@dataclass(frozen=True)
class DayResult:
    initial_cash: float
    equity: pd.Series
    trades: list[Trade]
    fees: float
    funding: float
    daily: np.ndarray = field(default_factory=lambda: np.array([]))  # retorno de cada sessão
    ruined: bool = False
    stop_days: int = 0  # sessões encerradas pelo stop diário do agente
    target_days: int = 0  # sessões encerradas pela meta diária
    max_trade_days: int = 0  # sessões em que esgotou as tentativas
    day_log: list[dict] = field(default_factory=list)  # por sessão: risco, tentativas, como terminou
    r_multiples: list[float] = field(default_factory=list)  # resultado de cada operação em R
    limit_placed: int = 0
    limit_filled: int = 0
    stop_placed: int = 0
    stop_filled: int = 0
    regime_bars: dict[str, int] = field(default_factory=dict)  # candles em cada regime
    regime_pnl: dict[str, float] = field(default_factory=dict)  # PnL líquido por regime de entrada
    stand_aside_bars: int = 0  # candles em que o mapa mandou ficar de fora
    blocked_by_news: int = 0
    blackout_bars: int = 0

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_cash - 1) if len(self.equity) else 0.0

    @property
    def gross_pnl(self) -> float:
        """Resultado antes das taxas de corretagem (funding continua)."""
        return float(self.equity.iloc[-1] - self.initial_cash + self.fees) if len(self.equity) else 0.0

    @property
    def gross_return(self) -> float:
        return self.gross_pnl / self.initial_cash

    @property
    def max_drawdown(self) -> float:
        if self.equity.empty:
            return 0.0
        curve = self.equity.to_numpy()
        return float(np.max(1 - curve / np.maximum.accumulate(curve)))

    @property
    def score(self) -> float:
        return risk_adjusted(self.total_return, self.max_drawdown)

    def count(self, action: str) -> int:
        return sum(1 for t in self.trades if t.action == action)

    @property
    def n_trades(self) -> int:
        return self.count("open")

    @property
    def liquidations(self) -> int:
        return self.count("liquidation")


def funding_lookup(funding: pd.DataFrame | None) -> dict[pd.Timestamp, float] | None:
    if funding is None or funding.empty:
        return None
    return dict(zip(funding["ts"], funding["rate"]))


def _shift(a: np.ndarray, n: int, fill) -> np.ndarray:
    if not n:
        return a
    return np.concatenate([np.full(n, fill, dtype=a.dtype if a.dtype != bool else bool), a[:-n]])


@dataclass
class _Pending:
    kind: str  # market | limit | stop | add
    side: int
    qty: float
    leverage: float
    stop: float  # distância do stop
    take_profit: float | None
    price: float  # limite ou gatilho (NaN a mercado)
    expires: int  # último candle em que pode executar
    regime: str


def run_daytrade_backtest(
    genome: DayGenome,
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    cash: float = 10_000.0,
    limits: DayTradeLimits = DayTradeLimits(),
    funding: dict[pd.Timestamp, float] | None = None,
    events: np.ndarray | None = None,
    spec: SymbolSpec = NO_MINIMUM,
    delay: int = 0,
    kelly: KellyCap | None = None,
    prior_r: list[float] | None = None,
    queue_haircut: float = QUEUE_HAIRCUT,
) -> DayResult:
    """`candles` pode trazer colunas news_/black_ (ver features.attach_features).

    `events`: sobrepõe os sinais (testes). `delay`: atrasa sinais, regime e
    notícias em N candles (auditoria de vazamento). `kelly`: limita o risco por
    operação a uma fração de Kelly estimada só com operações já encerradas.
    """
    tf = parse_timeframe(timeframe)
    df = candles.reset_index(drop=True)
    ts = df["ts"]
    if df.empty or not pd.api.types.is_datetime64_any_dtype(ts):
        return DayResult(cash, pd.Series(dtype=float), [], 0.0, 0.0)  # sem dados nesta janela
    first = int(ts.searchsorted(start))
    last = int(ts.searchsorted(end)) if end is not None else len(df)
    if first >= last:
        return DayResult(cash, pd.Series(dtype=float), [], 0.0, 0.0)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    keys = session_keys(ts, limits.session_end).to_numpy()

    # contexto: regime, notícias, sinais de cada estratégia usada
    reg = regime_features(df, session_end=limits.session_end)
    labels = _shift(reg["regime"].to_numpy(dtype=object), delay, "normal")
    news = news_view(df, genome.black_weight)
    news = {k: _shift(np.asarray(v), delay, 0 if k != "to_event" else 1e4) for k, v in news.items()}
    if events is not None:
        ev_by = {genome.strategy: _shift(np.asarray(events, dtype=float), delay, np.nan)}
    else:
        ev_by = {
            s: _shift(signals(df, genome, limits.session_end, strategy=s, regime=reg, news=news), delay, np.nan)
            for s in genome.strategies_used()
        }
    rmap = genome.regime_map
    order_type = genome.order_type
    grid = genome.strategy == "grid" and rmap is None
    if grid:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].where(df["volume"] > 0, 1e-9)
        k = pd.Series(keys)
        vwap = ((typical * vol).groupby(k).cumsum() / vol.groupby(k).cumsum()).to_numpy()
    own_stop = min(genome.daily_stop, limits.daily_loss_limit * 0.9)  # nunca mais frouxo que o gate
    maker_exit = order_type == "limit"

    broker = FuturesPaperBroker(cash, mmr=limits.mmr)
    gate = DayTradeRiskGate(limits, spec)
    exits: list[Close] = []
    pending: _Pending | None = None
    last_entry = -(10**9)
    entry_bar = -1
    midbar_entry = -1  # candle em que a entrada aconteceu no meio (só o stop vale nele)
    entry_regime = "normal"
    grid_levels = 0
    curve_ts, curve = [], []
    session = None
    day_start = cash
    day_ends: list[float] = []
    day_starts: list[float] = []
    entries_today = 0
    done_today: str | bool = False
    stop_days = target_days = max_trade_days = 0
    ruined = False
    mult = 1.0
    seen_trades = 0
    r_hist: list[float] = list(prior_r or [])
    new_r: list[float] = []
    open_risk = 0.0
    day_log: list[dict] = []
    today: dict = {}
    stats = {"limit_placed": 0, "limit_filled": 0, "stop_placed": 0, "stop_filled": 0,
             "stand_aside": 0, "news_block": 0, "blackout": 0}
    regime_bars = {r: 0 for r in REGIMES}
    regime_pnl = {r: 0.0 for r in REGIMES}

    def new_day(key) -> dict:
        return {"session": key, "attempts": 0, "max_risk": 0.0, "max_mult": 1.0, "end": "session_end"}

    def fill(p: _Pending, price: float, i: int, t: pd.Timestamp, at_open: bool) -> None:
        nonlocal last_entry, entry_bar, entries_today, open_risk, entry_regime, grid_levels, midbar_entry
        maker = p.kind in ("limit", "add")
        if p.kind == "add":
            broker.add(symbol, p.qty, price, t, maker=True)
            open_risk += p.qty * price * p.stop
            grid_levels += 1
            return
        stop = price * (1 - p.side * p.stop)
        tp = None if p.take_profit is None else price * (1 + p.side * p.take_profit)
        if p.price == p.price and grid:  # grid: alvo fixo na VWAP
            tp = p.take_profit
        broker.open(symbol, p.side, p.qty, price, p.leverage, stop, t, tp, maker=maker)
        open_risk = p.qty * price * p.stop
        last_entry = entry_bar = i
        entry_regime = p.regime
        grid_levels = 1
        entries_today += 1
        today["attempts"] += 1
        if p.kind == "limit":
            stats["limit_filled"] += 1
        elif p.kind == "stop":
            stats["stop_filled"] += 1
        if not at_open:  # entrou no meio do candle: pessimista, só o stop vale neste candle
            midbar_entry = i
            pos = broker.position(symbol)
            if (pos.side > 0 and l[i] <= pos.stop) or (pos.side < 0 and h[i] >= pos.stop):
                broker.close(symbol, pos.stop, t, "stop")

    for i in range(first, last):
        t = ts[i]
        if keys[i] != session:
            if session is not None:
                day_starts.append(day_start)
                day_ends.append(curve[-1])
                today["ret"] = curve[-1] / day_start - 1
                day_log.append(today)
                day_start = curve[-1]
            session = keys[i]
            entries_today = 0
            done_today = False
            mult = 1.0
            today = new_day(session)

        # 1) saídas pendentes, a mercado na abertura
        for action in exits:
            if broker.position(symbol):
                kind = action.reason if action.reason in (
                    "eod", "daily_loss", "kill", "agent_stop", "agent_target", "max_hold") else "close"
                broker.close(symbol, float(o[i]), t, kind)
        exits = []

        # 2) entrada pendente
        if pending is not None:
            p = pending
            price = None
            at_open = False
            if p.kind == "market":
                price, at_open = float(o[i]), True
            elif p.kind in ("limit", "add"):
                if (p.side > 0 and o[i] <= p.price) or (p.side < 0 and o[i] >= p.price):
                    price, at_open = float(o[i]), True
                elif (p.side > 0 and l[i] < p.price * (1 - queue_haircut)) or (
                    p.side < 0 and h[i] > p.price * (1 + queue_haircut)
                ):
                    price = p.price
            elif p.kind == "stop":
                if (p.side > 0 and o[i] >= p.price) or (p.side < 0 and o[i] <= p.price):
                    price, at_open = float(o[i]) * (1 + p.side * genome.slippage), True
                elif (p.side > 0 and h[i] > p.price) or (p.side < 0 and l[i] < p.price):
                    price = p.price * (1 + p.side * genome.slippage)
            if price is not None:
                try:
                    if p.kind != "add" and broker.position(symbol) is None:
                        fill(p, price, i, t, at_open)
                    elif p.kind == "add" and broker.position(symbol) is not None:
                        fill(p, price, i, t, at_open)
                except (InsufficientMargin, ValueError):
                    pass
                pending = None
            elif i >= p.expires:
                pending = None  # expirou sem executar

        # 3) posição: funding e saídas dentro do candle
        pos = broker.position(symbol)
        if pos is not None and midbar_entry != i:
            if t.minute == 0 and t.hour in FUNDING_HOURS:
                rate = DEFAULT_FUNDING_RATE if funding is None else funding.get(t)
                if rate is not None:
                    broker.apply_funding(symbol, rate, float(o[i]))
            _intrabar(broker, symbol, pos, float(o[i]), float(h[i]), float(l[i]), t, maker_exit)

        # resultados encerrados: escalada, Kelly, PnL por regime
        for tr in broker.trades[seen_trades:]:
            if tr.action not in ("open", "add"):
                mult *= genome.after_win_mult if tr.pnl > 0 else genome.after_loss_mult
                mult = min(max(mult, 0.05), 20.0)
                regime_pnl[entry_regime] += tr.pnl
                if open_risk > 0:
                    r = tr.pnl / open_risk
                    r_hist.append(r)
                    new_r.append(r)
                pending = None if pending is not None and pending.kind == "add" else pending
        seen_trades = len(broker.trades)

        # 4) fechamento do candle: trailing stop, patrimônio, ruína
        pos = broker.position(symbol)
        if pos is not None and genome.trail:
            if pos.side > 0:
                pos.stop = max(pos.stop, float(c[i]) * (1 - genome.trail))
            else:
                pos.stop = min(pos.stop, float(c[i]) * (1 + genome.trail))
        close_time = t + tf
        equity = broker.equity({symbol: float(c[i])})
        curve_ts.append(close_time)
        curve.append(equity)
        regime_bars[labels[i]] += 1
        if i == last - 1:
            break
        if pos is None and pending is None and gate.ruined(equity, float(c[i])):
            ruined = True
            today["end"] = "ruin"
            rest = ts[i + 1 : last] + tf
            curve_ts.extend(rest)
            curve.extend([equity] * len(rest))
            break

        # 5) regras do próprio agente para o dia
        if not done_today:
            if equity <= day_start * (1 - own_stop):
                done_today, stop_days = "agent_stop", stop_days + 1
                today["end"] = "daily_stop"
            elif genome.daily_target is not None and equity >= day_start * (1 + genome.daily_target):
                done_today, target_days = "agent_target", target_days + 1
                today["end"] = "daily_target"
            elif entries_today >= genome.max_trades_per_day and pos is None:
                done_today, max_trade_days = "max_trades", max_trade_days + 1
                today["end"] = "max_trades"
        if gate.halted and today["end"] == "session_end":
            today["end"] = "gate_daily_loss"

        pressure = 1.0
        if genome.goal_pressure and genome.daily_target is not None:
            session_end = gate.session_end_after(close_time)
            elapsed = 1 - (session_end - close_time) / pd.Timedelta(days=1)
            goal_eq = day_start * (1 + genome.daily_target)
            if elapsed > 0.5 and equity < goal_eq:
                shortfall = min(1.0, (goal_eq - equity) / (goal_eq - day_start))
                pressure = 1 + genome.goal_pressure * (elapsed - 0.5) * 2 * shortfall
        escalation = mult * pressure

        # 6) meta-controlador: qual estratégia vale neste regime
        active = genome.strategy if rmap is None else rmap.get(labels[i], genome.strategy)
        if active == STAND_ASIDE:
            stats["stand_aside"] += 1
            e = np.nan
        else:
            e = ev_by[active][i] if active in ev_by else np.nan
        direction = None if math.isnan(e) else int(e)
        if grid and pos is None:
            direction = _grid_direction(genome, float(c[i]), float(vwap[i]))

        # notícias: bloqueio antes de evento, contra sentimento forte
        entering = direction in (1, -1) and (pos is None or direction != pos.side)
        if entering and genome.event_blackout and news["to_event"][i] <= genome.event_blackout:
            stats["blackout"] += 1
            direction = None if pos is None else direction
        if entering and direction in (1, -1) and genome.news_gate is not None:
            if news["sentiment"][i] * direction < -genome.news_gate:
                stats["news_block"] += 1
                direction = None if pos is None else 0
        if entering and direction in (1, -1) and i - last_entry < genome.cooldown:
            direction = None
        if direction in (1, -1) and entries_today >= genome.max_trades_per_day:
            direction = None if pos is None else 0 if direction != pos.side else None
        if pos is not None and genome.max_hold and i - entry_bar + 1 >= genome.max_hold:
            direction = 0

        risk = genome.risk * escalation
        if genome.news_size_mult:
            risk *= min(2.0, max(0.25, 1 + genome.news_size_mult * (news["confidence"][i] - 0.5)))
        if kelly is not None:
            risk = min(risk, kelly.cap(r_hist))
        lev = genome.leverage * max(escalation, 1.0)
        stop_dist = genome.stop
        tp = genome.take_profit
        if grid:
            stop_dist = genome.params["step_bps"] / 1e4 * (genome.params["levels"] + 1)
            risk /= genome.params["levels"]  # risco total do grid cheio ≈ risk
        sig = DaySignal(direction, lev, stop_dist, risk, tp)
        actions = gate.actions(sig, float(c[i]), equity, pos.side if pos else 0, close_time, tf)
        if done_today in ("agent_stop", "agent_target"):
            actions = [Close(done_today)] if pos is not None else []
        elif done_today == "max_trades":
            actions = [a for a in actions if isinstance(a, Close)]

        exits = [a for a in actions if isinstance(a, Close)]
        opens = [a for a in actions if isinstance(a, Open)]
        if exits or opens:
            pending = None  # nova decisão cancela a ordem que não executou
        if opens:
            a = opens[0]
            kind = "market" if order_type == "market" else order_type
            if kind == "limit":
                price = float(c[i]) * (1 - a.side * genome.entry_offset)
                if grid:
                    price = float(vwap[i]) * (1 - a.side * genome.params["step_bps"] / 1e4)
                stats["limit_placed"] += 1
            elif kind == "stop":
                price = (float(h[i]) if a.side > 0 else float(l[i])) * (1 + a.side * genome.entry_offset)
                stats["stop_placed"] += 1
            else:
                price = float("nan")
            tp_abs = float(vwap[i]) if grid else a.take_profit  # grid: alvo absoluto na VWAP
            pending = _Pending(kind, a.side, a.qty, a.leverage, a.stop, tp_abs, price,
                               i + 1 + (genome.order_expiry - 1 if kind != "market" else 0), labels[i])
            today["max_risk"] = max(today["max_risk"], min(risk, limits.max_risk_per_trade))
            today["max_mult"] = max(today["max_mult"], escalation)
        elif grid and pos is not None and pending is None and not done_today and grid_levels < genome.params["levels"]:
            step = genome.params["step_bps"] / 1e4
            nxt = pos.entry * (1 - pos.side * step)
            unit = pos.qty / grid_levels
            if spec.min_order_value(nxt) <= unit * nxt + 1e-9:
                pending = _Pending("add", pos.side, unit, pos.leverage, step, None, nxt, i + genome.order_expiry,
                                   labels[i])
                stats["limit_placed"] += 1

    if curve:
        day_starts.append(day_start)
        day_ends.append(curve[-1])
        today["ret"] = curve[-1] / day_start - 1
        day_log.append(today)
    daily = np.array(day_ends) / np.maximum(np.array(day_starts), 1e-12) - 1
    adds = sum(1 for t in broker.trades if t.action == "add")
    return DayResult(
        cash, pd.Series(curve, index=curve_ts), broker.trades, broker.fees, broker.funding_paid,
        daily, ruined, stop_days, target_days, max_trade_days, day_log, new_r,
        stats["limit_placed"], stats["limit_filled"] + adds, stats["stop_placed"], stats["stop_filled"],
        regime_bars, regime_pnl, stats["stand_aside"], stats["news_block"], stats["blackout"],
    )


def _grid_direction(g: DayGenome, close: float, anchor: float) -> int | None:
    """Grid: abaixo da VWAP compra, acima vende (respeitando a direção do genoma)."""
    step = g.params["step_bps"] / 1e4
    if close > anchor * (1 + step) and g.direction in ("short", "both"):
        return -1
    if close < anchor * (1 - step) and g.direction in ("long", "both"):
        return 1
    return None


def goal_metrics(daily: np.ndarray, capital_brl: float, per_day_brl: float = 1000.0) -> dict[str, float]:
    """Métricas da meta "dobrar por dia".

    days_to_income: dias até o capital render `per_day_brl` por dia, compondo à
    média geométrica diária observada. Infinito se a média não for positiva.
    """
    n = len(daily)
    if n == 0:
        return {"days": 0, "hits": 0, "hit_rate": 0.0, "best_day": 0.0, "geo_daily": 0.0, "days_to_income": math.inf}
    hits = int((daily >= GOAL_DAY).sum())
    geo = float(np.exp(np.mean(np.log1p(np.maximum(daily, -0.999999)))) - 1)
    if geo > 0:
        needed = per_day_brl / geo
        days_to = 0.0 if needed <= capital_brl else math.log(needed / capital_brl) / math.log1p(geo)
    else:
        days_to = math.inf
    return {
        "days": n, "hits": hits, "hit_rate": hits / n, "best_day": float(daily.max()),
        "geo_daily": geo, "days_to_income": days_to,
    }


def _intrabar(broker: FuturesPaperBroker, symbol: str, pos, o: float, h: float, l: float, t: pd.Timestamp,
              maker_tp: bool) -> None:
    if pos.side > 0:
        if o <= pos.liq:
            broker.close(symbol, o, t, "liquidation")
        elif o <= pos.stop:
            broker.close(symbol, o, t, "stop")
        elif l <= pos.stop:  # stop fica acima da liquidação: é atingido antes
            broker.close(symbol, pos.stop, t, "stop")
        elif pos.take_profit is not None and h >= pos.take_profit:
            broker.close(symbol, max(pos.take_profit, o), t, "take_profit", maker=maker_tp)
    else:
        if o >= pos.liq:
            broker.close(symbol, o, t, "liquidation")
        elif o >= pos.stop:
            broker.close(symbol, o, t, "stop")
        elif h >= pos.stop:
            broker.close(symbol, pos.stop, t, "stop")
        elif pos.take_profit is not None and l <= pos.take_profit:
            broker.close(symbol, min(pos.take_profit, o), t, "take_profit", maker=maker_tp)
