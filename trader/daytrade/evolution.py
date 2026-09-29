"""Evolução dos agentes de day trade.

Mesmo ciclo da evolução principal: falhou, morre e o Criador gera o sucessor
a partir da autópsia; prosperou, ganha vaga para ter 2. Liquidação e ruína
matam na hora, em qualquer modo.

Dois critérios de seleção (fitness):
  risk  retorno ÷ queda máxima (o mesmo da evolução principal)
  goal  meta do usuário: fração de dias com ≥ +100%; nunca dobrar mata
  ladder escada de capital R$100 → R$1k → R$10k → R$100k: sobe um degrau só
        depois de provar consistência no atual; risco limitado a meio Kelly
Cada janela começa com o capital do degrau (conta nova), então os retornos
diários de todas as janelas formam o histórico de dias do agente.
"""
from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trader.daytrade.backtest import GOAL_DAY, KellyCap, goal_metrics, run_daytrade_backtest
from trader.daytrade.creator import DayRuleCreator
from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.genome import DayGenome, random_day_genome
from trader.daytrade.market import NO_MINIMUM, SymbolSpec, brl_to_usdt
from trader.domain import parse_timeframe
from trader.evolution.fitness import (
    Life,
    SelectionRules,
    Verdict,
    WindowStats,
    _drawdown_context,
    judge,
    risk_adjusted,
)
from trader.evolution.registry import Registry

LIQUIDATION, RUIN, NO_GOAL = "liquidation", "ruin", "no_goal"
DAY_BASELINE = DayGenome(
    "momentum", {"lookback": 16, "threshold_bps": 50}, direction="both", leverage=1.0, stop=0.01, risk=0.01
)


@dataclass(frozen=True)
class DayWindow:
    stats: WindowStats
    liquidations: int
    ruined: bool
    daily: np.ndarray
    stop_days: int
    max_trade_days: int
    escalated_days: int  # dias com escalada de risco (>1x)
    escalated_ruin: int  # desses, quantos terminaram em ruína
    escalated_double: int  # desses, quantos dobraram
    r_multiples: tuple[float, ...] = ()
    fees: float = 0.0
    gross_pnl: float = 0.0
    limit_placed: int = 0
    limit_filled: int = 0
    regime_bars: tuple = ()  # ((regime, candles), ...)
    regime_pnl: tuple = ()  # ((regime, pnl), ...)
    stand_aside_bars: int = 0


def evaluate_day(
    genome: DayGenome,
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limits: DayTradeLimits,
    funding: dict | None,
    cash: float,
    spec: SymbolSpec,
    kelly: KellyCap | None = None,
    prior_r: tuple[float, ...] = (),
) -> DayWindow:
    res = run_daytrade_backtest(
        genome, candles, symbol, timeframe, start, end, cash=cash, limits=limits, funding=funding, spec=spec,
        kelly=kelly, prior_r=list(prior_r),
    )
    window = candles[(candles["ts"] >= start) & (candles["ts"] < end)].reset_index(drop=True)
    ctx = _drawdown_context(res.equity.to_numpy(), window)
    stats = WindowStats(start, res.total_return, res.gross_return, res.max_drawdown, res.n_trades, **ctx)
    esc = [d for d in res.day_log if d["max_mult"] > 1.0 + 1e-9]
    return DayWindow(
        stats, res.liquidations, res.ruined, res.daily, res.stop_days, res.max_trade_days,
        len(esc), sum(d["end"] == "ruin" for d in esc), sum(d.get("ret", 0) >= GOAL_DAY for d in esc),
        tuple(res.r_multiples), res.fees, res.gross_pnl, res.limit_placed, res.limit_filled,
        tuple(res.regime_bars.items()), tuple(res.regime_pnl.items()), res.stand_aside_bars,
    )


def _job(args: tuple) -> DayWindow:
    return evaluate_day(*args)


LADDER_BRL = (100.0, 1_000.0, 10_000.0, 100_000.0)


@dataclass(frozen=True)
class LadderRules:
    min_days: int = 60  # dias de pregão no degrau antes de poder subir
    min_profitable_share: float = 0.5  # fração de dias com lucro
    max_drawdown: float = 0.15  # queda máxima da curva diária no degrau
    kelly: KellyCap = KellyCap()


def daily_curve_stats(daily: list[float]) -> tuple[float, float]:
    """(retorno composto, queda máxima) de uma sequência de retornos diários."""
    if not daily:
        return 0.0, 0.0
    curve = np.cumprod(1 + np.asarray(daily))
    peak = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    return float(curve[-1] - 1), float(np.max(1 - curve / np.maximum(peak, 1.0)))


@dataclass
class DayAgent:
    id: str
    genome: DayGenome
    lineage: str
    life: Life = field(default_factory=Life)
    liquidations: int = 0
    daily: list[float] = field(default_factory=list)
    stop_days: int = 0
    max_trade_days: int = 0
    # escada
    rung: int = 0
    rung_daily: list[float] = field(default_factory=list)  # dias no degrau atual
    rung_base_daily: list[float] = field(default_factory=list)  # referência nos mesmos dias
    days_per_rung: dict[int, int] = field(default_factory=dict)
    r_history: list[float] = field(default_factory=list)  # operações encerradas, em R
    fees: float = 0.0
    gross_pnl: float = 0.0
    limit_placed: int = 0
    limit_filled: int = 0
    regime_pnl: dict[str, float] = field(default_factory=dict)

    def rung_stats(self) -> dict[str, float]:
        ret, dd = daily_curve_stats(self.rung_daily)
        b_ret, b_dd = daily_curve_stats(self.rung_base_daily)
        n = len(self.rung_daily)
        return {
            "rung": self.rung, "days": n,
            "profitable_share": (sum(d > 0 for d in self.rung_daily) / n) if n else 0.0,
            "ret": ret, "max_dd": dd, "score": risk_adjusted(ret, dd), "baseline_score": risk_adjusted(b_ret, b_dd),
        }

    def goal(self, capital_brl: float) -> dict[str, float]:
        return goal_metrics(np.asarray(self.daily), capital_brl)

    def goal_key(self, capital_brl: float) -> tuple[float, float, float]:
        g = self.goal(capital_brl)
        return (g["hit_rate"], g["best_day"], self.life.score)


@dataclass(frozen=True)
class DayConfig:
    initial: int = 16
    max_slots: int = 32
    max_lineage_share: float = 0.3
    rules: SelectionRules = SelectionRules(max_drawdown=0.20)
    limits: DayTradeLimits = DayTradeLimits()
    fitness: str = "risk"  # risk | goal
    cash: float = 10_000.0  # USDT
    capital_brl: float = 100.0  # só para as métricas da meta
    spec: SymbolSpec = NO_MINIMUM
    brl_rate: float = 5.5
    ladder: LadderRules = LadderRules()
    styles: tuple[str, ...] = ("any",)  # estilos que podem nascer/ser escolhidos
    news: bool = False  # genes de notícia ativos (há provedores)
    black: bool = False  # genes black_weight evoluem (braços WB/WB')


class DayPopulation:
    def __init__(
        self,
        candles: pd.DataFrame,
        symbol: str,
        timeframe: str,
        registry: Registry,
        config: DayConfig = DayConfig(),
        funding: dict | None = None,
        seed: int = 0,
        executor: Executor | None = None,
        creator: Any = None,
        loader: Any = None,
        providers: list | None = None,
    ) -> None:
        """`loader(timeframe, start, end)` busca candles de outro timeframe (ex.: 1m do scalping)
        por intervalo de datas, sem carregar tudo; `providers`: features de notícia."""
        if config.fitness not in ("risk", "goal", "ladder"):
            raise ValueError("fitness deve ser risk, goal ou ladder")
        self.candles, self.symbol, self.timeframe = candles, symbol, timeframe
        self.registry, self.config, self.funding = registry, config, funding
        self.rng = random.Random(seed)
        self.executor = executor
        self.creator = creator or DayRuleCreator()
        self.creator.styles = list(config.styles)
        self.creator.black_allowed = config.black
        self.loader = loader
        self.providers = providers or []
        self.alive: list[DayAgent] = []
        self.baseline_windows: list[WindowStats] = []
        self._counter = 0
        self._window = 0
        self.tf = parse_timeframe(timeframe)
        self.bars_per_day = int(pd.Timedelta(days=1) / self.tf)
        # totais da evolução inteira
        self.totals: Counter[str] = Counter()
        self.best_seen: DayAgent | None = None  # melhor agente já avaliado (vivo ou morto)
        self.promotions: list[tuple[str, int, int]] = []  # (agente, degrau novo, janela)

    def _born(self, genome: DayGenome, origin: str, parent: DayAgent | None = None, note: str = "") -> None:
        self._counter += 1
        agent_id = f"d{self._counter:05d}"
        lineage = parent.lineage if parent else agent_id
        gen = 0
        if parent is not None:
            gen = int(self.registry.db.execute("SELECT generation FROM agents WHERE id = ?", (parent.id,)).fetchone()[0]) + 1
        self.registry.birth(agent_id, genome, lineage, gen, self._window, origin, parent.id if parent else None, note)
        self.alive.append(DayAgent(agent_id, genome, lineage))
        self.totals[f"style_born:{genome.style}"] += 1

    def seed(self) -> None:
        for _ in range(self.config.initial):
            self._born(random_day_genome(self.rng, styles=list(self.config.styles), news=self.config.news,
                                         black=self.config.black), "seed")

    def cash_for(self, a: DayAgent | None) -> float:
        if self.config.fitness != "ladder" or a is None:
            return self.config.cash
        return brl_to_usdt(LADDER_BRL[a.rung], self.config.brl_rate)

    def window_data(self, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, warm_bars: int) -> pd.DataFrame:
        tf = parse_timeframe(timeframe)
        lo = start - tf * warm_bars
        if timeframe == self.timeframe or self.loader is None:
            c = self.candles
            data = c[(c["ts"] >= lo) & (c["ts"] < end)].reset_index(drop=True)
        else:
            data = self.loader(timeframe, lo, end).reset_index(drop=True)
        if self.providers and len(data):
            from trader.daytrade.features import attach_features

            data = attach_features(data, self.symbol, timeframe, self.providers)
        return data

    def _evaluate(self, agents: list[DayAgent | None], start: pd.Timestamp, end: pd.Timestamp) -> list[DayWindow]:
        """None = motor de referência. Agentes são agrupados pelo timeframe do estilo."""
        genomes = [DAY_BASELINE if a is None else a.genome for a in agents]
        cfg = self.config
        kelly = cfg.ladder.kelly if cfg.fitness == "ladder" else None
        by_tf: dict[str, list[int]] = {}
        for k, g in enumerate(genomes):
            by_tf.setdefault(g.timeframe, []).append(k)
        results: list[DayWindow | None] = [None] * len(genomes)
        for timeframe, idxs in by_tf.items():
            bpd = int(pd.Timedelta(days=1) / parse_timeframe(timeframe))
            warm = max(genomes[k].warmup_bars(bpd) for k in idxs)
            data = self.window_data(timeframe, start, end, warm)
            jobs = [
                (genomes[k], data, self.symbol, timeframe, start, end, cfg.limits, self.funding,
                 self.cash_for(agents[k]), cfg.spec,
                 None if agents[k] is None else kelly, () if agents[k] is None else tuple(agents[k].r_history[-500:]))
                for k in idxs
            ]
            out = [_job(j) for j in jobs] if self.executor is None else list(self.executor.map(_job, jobs))
            for k, r in zip(idxs, out):
                results[k] = r
            del data, jobs
        return results

    def _baseline_life(self, a: DayAgent) -> Life:
        return Life(self.baseline_windows[len(self.baseline_windows) - a.life.age :])

    def _autopsy(self, a: DayAgent, cause: str) -> dict[str, Any]:
        last = a.life.windows[-1] if a.life.windows else None
        g = a.goal(self.config.capital_brl)
        days = max(len(a.daily), 1)
        return {
            "cause": cause,
            "genome": a.genome.to_dict(),
            "life": {
                "age_windows": a.life.age, "trades": a.life.trades, "liquidations": a.liquidations,
                "net_return": round(a.life.ret, 4), "gross_return": round(a.life.gross_ret, 4),
                "worst_drawdown": round(a.life.worst_dd, 4),
                "benchmark_return": round(self._baseline_life(a).ret, 4),
                "days": g["days"], "doubled_days": g["hits"], "best_day": round(g["best_day"], 4),
                "stop_day_share": round(a.stop_days / days, 3),
                "max_trade_day_share": round(a.max_trade_days / days, 3),
                "style": a.genome.style,
                "fee_share_of_gross": round(a.fees / a.gross_pnl, 3) if a.gross_pnl > 0 else None,
                "maker_fill_rate": round(a.limit_filled / a.limit_placed, 3) if a.limit_placed else None,
                "pnl_by_regime": {k: round(v, 4) for k, v in a.regime_pnl.items() if v},
            },
            "last_window": {} if last is None else {
                "net_return": round(last.ret, 4), "max_drawdown": round(last.max_dd, 4), "trades": last.trades,
                "dd_high_vol": last.dd_high_vol, "median_range": round(last.median_range, 5),
            },
        }

    def _kill(self, a: DayAgent, cause: str) -> dict[str, Any]:
        self.totals[f"style_died:{a.genome.style}"] += 1
        autopsy = self._autopsy(a, cause)
        self.registry.death(a.id, self._window, cause, autopsy)
        self.alive.remove(a)
        return autopsy

    def _key(self, a: DayAgent):
        if self.config.fitness == "goal":
            return a.goal_key(self.config.capital_brl)
        if self.config.fitness == "ladder":
            return (a.rung, a.rung_stats()["score"], a.life.score)
        return (a.life.score,)

    def _verdict(self, a: DayAgent, r: DayWindow) -> tuple[str | None, bool]:
        """(causa da morte ou None, prosperou?)"""
        if r.liquidations:
            return LIQUIDATION, False
        if r.ruined:
            return RUIN, False
        if self.config.fitness == "risk":
            verdict, cause = judge(a.life, self._baseline_life(a), self.config.rules)
            return (cause.value if verdict is Verdict.DIE else None), verdict is Verdict.THRIVE
        if self.config.fitness == "ladder":
            verdict, cause = judge(a.life, self._baseline_life(a), self.config.rules)
            st, lr = a.rung_stats(), self.config.ladder
            if st["max_dd"] > lr.max_drawdown:
                return "drawdown", False
            if verdict is Verdict.DIE and cause.value != "stagnation":
                return cause.value, False
            if st["days"] >= 3 * lr.min_days and st["score"] <= st["baseline_score"]:
                return "stagnation", False  # muito tempo no degrau sem provar nada
            return None, self._promotable(a)
        rules = self.config.rules
        hits_total = sum(1 for d in a.daily if d >= GOAL_DAY)
        hits_now = int((r.daily >= GOAL_DAY).sum())
        if a.life.age >= rules.min_age_for_edge and hits_total == 0:
            return NO_GOAL, False
        if a.life.age >= rules.max_age and hits_now == 0:
            return "stagnation", False
        return None, hits_now > 0 and hits_total >= 2

    def _promotable(self, a: DayAgent) -> bool:
        st, lr = a.rung_stats(), self.config.ladder
        return (
            a.rung < len(LADDER_BRL) - 1
            and st["days"] >= lr.min_days
            and st["profitable_share"] >= lr.min_profitable_share
            and st["score"] > max(st["baseline_score"], 0.0)
            and st["max_dd"] <= lr.max_drawdown
        )

    def _promote(self, a: DayAgent) -> None:
        a.rung += 1
        a.rung_daily, a.rung_base_daily = [], []
        self.promotions.append((a.id, a.rung, self._window))
        self.totals[f"promotions_to_{a.rung}"] += 1

    def step(self, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        results = self._evaluate([*self.alive, None], start, end)
        self.baseline_windows.append(results[-1].stats)
        base_daily = results[-1].daily.tolist()
        for a, r in zip(self.alive, results[:-1]):
            a.rung_daily.extend(r.daily.tolist())
            a.rung_base_daily.extend(base_daily[: len(r.daily)])
            a.days_per_rung[a.rung] = a.days_per_rung.get(a.rung, 0) + len(r.daily)
            a.r_history.extend(r.r_multiples)
            a.fees += r.fees
            a.gross_pnl += r.gross_pnl
            a.limit_placed += r.limit_placed
            a.limit_filled += r.limit_filled
            for reg, pnl in r.regime_pnl:
                a.regime_pnl[reg] = a.regime_pnl.get(reg, 0.0) + pnl
            style = a.genome.style
            self.totals[f"style_days:{style}"] += len(r.daily)
            self.totals[f"style_fees:{style}"] += r.fees
            self.totals[f"style_gross:{style}"] += r.gross_pnl
            self.totals[f"style_limit_placed:{style}"] += r.limit_placed
            self.totals[f"style_limit_filled:{style}"] += r.limit_filled
            for reg, bars in r.regime_bars:
                self.totals[f"regime_bars:{reg}"] += bars
            for reg, pnl in r.regime_pnl:
                self.totals[f"regime_pnl:{reg}"] += pnl
            self.totals["stand_aside_bars"] += r.stand_aside_bars
            a.life.windows.append(r.stats)
            a.liquidations += r.liquidations
            a.daily.extend(r.daily.tolist())
            a.stop_days += r.stop_days
            a.max_trade_days += r.max_trade_days
            self.totals["agent_days"] += len(r.daily)
            self.totals["doubled_days"] += int((r.daily >= GOAL_DAY).sum())
            self.totals["ruins"] += int(r.ruined)
            self.totals["liquidations"] += r.liquidations
            self.totals["escalated_days"] += r.escalated_days
            self.totals["escalated_ruin"] += r.escalated_ruin
            self.totals["escalated_double"] += r.escalated_double

        deaths: Counter[str] = Counter()
        births: Counter[str] = Counter()
        thrivers = []
        evaluated = list(zip(self.alive, results[:-1]))
        for a, _ in evaluated:
            if a.life.age > 0 and (self.best_seen is None or self._key(a) > self._key(self.best_seen)):
                self.best_seen = a
        for a, r in evaluated:
            cause, thrive = self._verdict(a, r)
            if thrive:
                thrivers.append(a)
                if self.config.fitness == "ladder":
                    self._promote(a)
            if cause:
                autopsy = self._kill(a, cause)
                deaths[cause] += 1
                genome, note = self.creator.successor(a.genome, autopsy, self.rng)
                self._born(genome, "successor", a, note)
                births["successor"] += 1

        cap = max(2, int(self.config.max_slots * self.config.max_lineage_share))
        for a in sorted(thrivers, key=self._key, reverse=True):
            if a not in self.alive or sum(x.lineage == a.lineage for x in self.alive) + 1 > cap:
                continue
            if len(self.alive) >= self.config.max_slots:
                others = [x for x in self.alive if x is not a and x.life.age > 0]
                worst = min(others, key=self._key, default=None)
                if worst is None or self._key(worst) >= self._key(a):
                    continue
                self._kill(worst, "culled")
                deaths["culled"] += 1
            self._born(self.creator.child(a.genome, self.rng), "child", a, f"filho de {a.id}")
            births["child"] += 1

        self._window += 1
        return {"index": self._window - 1, "start": start, "alive": len(self.alive),
                "deaths": dict(deaths), "births": dict(births)}

    def champion(self) -> DayAgent | None:
        r = self.config.rules
        ok = [a for a in self.alive if a.life.age >= r.min_age_for_edge and a.life.trades >= r.min_trades]
        if not ok and self.config.fitness == "goal":
            # no modo meta todos morrem antes da idade mínima: usa o melhor já avaliado
            return self.best_seen
        return max(ok, key=self._key, default=None)

    def rung_census(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for a in self.alive:
            out[a.rung] = out.get(a.rung, 0) + 1
        return dict(sorted(out.items()))
