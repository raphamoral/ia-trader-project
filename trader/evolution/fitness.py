"""Avaliação de um agente numa janela e o veredito: morrer, viver ou prosperar."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from trader.backtest import run_backtest
from trader.execution.paper import BINANCE_SPOT_FEE
from trader.evolution.genome import Genome, GenomeEngine


DD_FLOOR = 0.02  # evita que quase não arriscar vire nota infinita


def risk_adjusted(ret: float, max_dd: float) -> float:
    """Retorno por unidade de queda máxima (tipo índice Calmar)."""
    return ret / max(max_dd, DD_FLOOR)


@dataclass(frozen=True)
class WindowStats:
    start: pd.Timestamp
    ret: float
    gross_ret: float
    max_dd: float
    trades: int
    # contexto do pior trecho de perda, usado na autópsia
    dd_high_vol: bool = False  # amplitude do mercado acima do normal durante a queda
    dd_market_down: bool = False  # o próprio ativo caiu forte durante a queda
    median_range: float = 0.0  # amplitude média típica da janela (fração do preço)
    # contexto de mercado da janela, usado na análise de sucesso
    market_ret: float = 0.0  # quanto o próprio ativo variou na janela
    news_sentiment: float | None = None  # sentimento médio das notícias (None sem dados de notícia)

    @property
    def score(self) -> float:
        return risk_adjusted(self.ret, self.max_dd)


def evaluate(
    genome: Genome, candles: pd.DataFrame, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
    fee_rate: float = BINANCE_SPOT_FEE,
) -> WindowStats:
    result = run_backtest(GenomeEngine(genome), candles, symbol, timeframe, start, end, fee_rate=fee_rate)
    window = candles[(candles["ts"] >= start) & (candles["ts"] < end)].reset_index(drop=True)
    stats = dict(
        start=start,
        ret=result.total_return,
        gross_ret=result.gross_return,
        max_dd=result.max_drawdown,
        trades=result.n_trades,
    )
    stats.update(_drawdown_context(result.equity.to_numpy(), window))
    if len(window):
        stats["market_ret"] = float(window["close"].iloc[-1] / window["open"].iloc[0] - 1)
        if "news_sentiment" in window.columns:
            stats["news_sentiment"] = float(window["news_sentiment"].fillna(0).mean())
    return WindowStats(**stats)


def combine(stats: list[WindowStats]) -> WindowStats:
    """Uma janela operada em vários ativos com o mesmo genoma: carteira de pesos iguais."""
    if len(stats) == 1:
        return stats[0]
    mean = lambda xs: sum(xs) / len(xs)
    news = [s.news_sentiment for s in stats if s.news_sentiment is not None]
    return WindowStats(
        start=stats[0].start,
        ret=mean([s.ret for s in stats]),
        gross_ret=mean([s.gross_ret for s in stats]),
        max_dd=mean([s.max_dd for s in stats]),  # aproximação da queda da carteira
        trades=sum(s.trades for s in stats),
        dd_high_vol=sum(s.dd_high_vol for s in stats) * 2 > len(stats),
        dd_market_down=sum(s.dd_market_down for s in stats) * 2 > len(stats),
        median_range=mean([s.median_range for s in stats]),
        market_ret=mean([s.market_ret for s in stats]),
        news_sentiment=mean(news) if news else None,
    )


def _drawdown_context(curve: np.ndarray, window: pd.DataFrame) -> dict:
    if len(curve) < 2 or len(window) != len(curve):
        return {}
    amp = ((window["high"] - window["low"]) / window["close"]).to_numpy()
    peak = np.maximum.accumulate(curve)
    trough = int(np.argmax(1 - curve / peak))
    top = int(np.argmax(curve[: trough + 1]))
    median = float(np.median(amp))
    if trough == top:
        return {"median_range": median}
    closes = window["close"].to_numpy()
    return {
        "dd_high_vol": bool(amp[top : trough + 1].mean() > 1.25 * median),
        "dd_market_down": bool(closes[trough] / closes[top] - 1 < -0.05),
        "median_range": median,
    }


@dataclass
class Life:
    """Histórico acumulado de um agente desde que nasceu."""

    windows: list[WindowStats] = field(default_factory=list)

    @property
    def age(self) -> int:
        return len(self.windows)

    @property
    def trades(self) -> int:
        return sum(w.trades for w in self.windows)

    @property
    def ret(self) -> float:
        return math.prod(1 + w.ret for w in self.windows) - 1

    @property
    def gross_ret(self) -> float:
        return math.prod(1 + w.gross_ret for w in self.windows) - 1

    @property
    def worst_dd(self) -> float:
        return max((w.max_dd for w in self.windows), default=0.0)

    @property
    def score(self) -> float:
        return risk_adjusted(self.ret, self.worst_dd)


class Verdict(str, Enum):
    DIE = "die"
    LIVE = "live"
    THRIVE = "thrive"


class Cause(str, Enum):
    DRAWDOWN = "drawdown"  # perdeu demais numa janela
    FEES = "fees"  # ganharia sem taxa, mas a taxa comeu o lucro (opera demais)
    NEGATIVE_EDGE = "negative_edge"  # perde mesmo antes da taxa
    STAGNATION = "stagnation"  # velho demais sem nunca superar a referência
    CULLED = "culled"  # o pior da população, removido para dar vaga a um vencedor


@dataclass(frozen=True)
class SelectionRules:
    max_drawdown: float = 0.15
    min_trades: int = 8  # abaixo disso, resultado ainda é sorte
    min_age_for_edge: int = 3  # janelas antes de julgar se há vantagem
    max_age: int = 10
    thrive_streak: int = 2  # janelas seguidas batendo a referência para ganhar vaga


def judge(life: Life, baseline: Life, rules: SelectionRules) -> tuple[Verdict, Cause | None]:
    """baseline: o motor de referência nas mesmas janelas da vida do agente.

    Comparações usam retorno ajustado ao risco, para não premiar quem só arriscou mais.
    """
    if life.windows[-1].max_dd > rules.max_drawdown:
        return Verdict.DIE, Cause.DRAWDOWN
    experienced = life.trades >= rules.min_trades
    if experienced and life.age >= rules.min_age_for_edge and life.ret <= 0:
        return Verdict.DIE, Cause.FEES if life.gross_ret > 0 else Cause.NEGATIVE_EDGE
    n = rules.thrive_streak
    streak = life.age >= n and all(
        w.score > b.score for w, b in zip(life.windows[-n:], baseline.windows[-n:])
    )
    if experienced and life.ret > 0 and life.score > baseline.score and streak:
        return Verdict.THRIVE, None
    if life.age >= rules.max_age:
        return Verdict.DIE, Cause.STAGNATION
    return Verdict.LIVE, None
