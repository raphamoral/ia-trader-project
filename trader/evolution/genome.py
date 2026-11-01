"""Genoma: a configuração completa de um agente, e o motor que ele gera.

O genoma é dado puro (vira JSON numa linha do registro). Nascer ou morrer é
inserir ou remover uma linha; nenhum código muda.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from trader.domain import MarketView, Side, Signal

# faixas válidas de cada parâmetro, por estratégia
STRATEGIES: dict[str, dict[str, tuple[int, int]]] = {
    "ma_cross": {"fast": (3, 60), "slow": (10, 250)},
    "breakout": {"lookback": (5, 150)},
    "rsi_revert": {"period": (4, 40), "low": (10, 45), "high": (50, 90)},
}
TREND_RANGE = (20, 300)
COOLDOWN_RANGE = (0, 60)
VOL_FILTER_RANGE = (0.002, 0.2)
SIZE_RANGE = (0.1, 1.0)
# genes de notícia: neutros por padrão (sem as colunas, o motor ignora todos)
NEWS_GATE_RANGE = (0.05, 0.9)
NEWS_FOLLOW_RANGE = (0.1, 0.95)
BLACKOUT_MIN_RANGE = (0, 240)
BLACK_WEIGHT_RANGE = (0.0, 1.0)
HACK_BLACKOUT_H_RANGE = (0, 48)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


@dataclass(frozen=True)
class Genome:
    strategy: str
    params: dict[str, int] = field(default_factory=dict)
    size: float = 1.0
    vol_filter: float | None = None  # não entra se a amplitude média (% do preço) passar disso
    trend_filter: int | None = None  # só compra acima da média de N candles
    cooldown: int = 0  # candles mínimos entre uma entrada e a próxima
    news_gate: float | None = None  # não compra se o sentimento combinado estiver abaixo de -news_gate
    news_follow: float | None = None  # compra mesmo sem sinal técnico se o sentimento passar de news_follow
    event_blackout: int = 0  # minutos antes de evento agendado (Copom, FOMC, payroll) sem novas entradas
    black_weight: float = 0.0  # peso do sentimento black no sentimento combinado
    hack_blackout: int = 0  # horas após incidente de segurança sem novas entradas

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"estratégia desconhecida: {self.strategy!r}")
        expected = set(STRATEGIES[self.strategy])
        if set(self.params) != expected:
            raise ValueError(f"{self.strategy} exige parâmetros {sorted(expected)}")

    def normalized(self) -> Genome:
        """Coloca todos os valores dentro das faixas e corrige relações entre parâmetros."""
        bounds = STRATEGIES[self.strategy]
        params = {k: int(round(_clamp(v, bounds[k]))) for k, v in self.params.items()}
        if self.strategy == "ma_cross" and params["fast"] >= params["slow"]:
            params["slow"] = min(params["fast"] + 5, bounds["slow"][1])
            params["fast"] = min(params["fast"], params["slow"] - 1)
        if self.strategy == "rsi_revert" and params["low"] >= params["high"]:
            params["low"], params["high"] = 30, 70
        return replace(
            self,
            params=params,
            size=round(_clamp(self.size, SIZE_RANGE), 3),
            vol_filter=None if self.vol_filter is None else round(_clamp(self.vol_filter, VOL_FILTER_RANGE), 5),
            trend_filter=None if self.trend_filter is None else int(_clamp(self.trend_filter, TREND_RANGE)),
            cooldown=int(_clamp(self.cooldown, COOLDOWN_RANGE)),
            news_gate=None if self.news_gate is None else round(_clamp(self.news_gate, NEWS_GATE_RANGE), 3),
            news_follow=None if self.news_follow is None else round(_clamp(self.news_follow, NEWS_FOLLOW_RANGE), 3),
            event_blackout=int(_clamp(self.event_blackout, BLACKOUT_MIN_RANGE)),
            black_weight=round(_clamp(self.black_weight, BLACK_WEIGHT_RANGE), 3),
            hack_blackout=int(_clamp(self.hack_blackout, HACK_BLACKOUT_H_RANGE)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        return cls(
            strategy=data["strategy"],
            params={k: int(v) for k, v in data["params"].items()},
            size=float(data.get("size", 1.0)),
            vol_filter=None if data.get("vol_filter") is None else float(data["vol_filter"]),
            trend_filter=None if data.get("trend_filter") is None else int(data["trend_filter"]),
            cooldown=int(data.get("cooldown", 0)),
            news_gate=None if data.get("news_gate") is None else float(data["news_gate"]),
            news_follow=None if data.get("news_follow") is None else float(data["news_follow"]),
            event_blackout=int(data.get("event_blackout") or 0),
            black_weight=float(data.get("black_weight") or 0.0),
            hack_blackout=int(data.get("hack_blackout") or 0),
        ).normalized()

    @property
    def warmup(self) -> int:
        need = max(self.params.values()) + 2
        if self.trend_filter:
            need = max(need, self.trend_filter + 1)
        return max(need, 16)


def random_genome(rng: random.Random, strategy: str | None = None, priors: dict | None = None,
                  prior_weight: float = 0.7) -> Genome:
    """Genoma aleatório. Com `priors` (faixas vindas dos livros), cada gene cai na faixa sugerida
    com probabilidade `prior_weight`, senão na faixa completa (para a evolução poder refutar)."""
    priors = priors or {}

    def draw(name: str, lo: float, hi: float, integer: bool) -> float:
        p = priors.get(name)
        if p and rng.random() < prior_weight:
            lo2, hi2 = max(lo, float(p["low"])), min(hi, float(p["high"]))
            if lo2 <= hi2:
                lo, hi = lo2, hi2
        return rng.randint(int(lo), int(hi)) if integer else rng.uniform(lo, hi)

    strategy = strategy or rng.choice(sorted(STRATEGIES))
    params = {k: draw(f"{strategy}.{k}", lo, hi, True) for k, (lo, hi) in STRATEGIES[strategy].items()}
    return Genome(
        strategy=strategy,
        params=params,
        size=draw("size", 0.3, 1.0, False),
        vol_filter=rng.choice([None, draw("vol_filter", 0.01, 0.06, False)]),
        trend_filter=rng.choice([None, draw("trend_filter", 50, 200, True)]),
        cooldown=rng.choice([0, 0, draw("cooldown", 1, 12, True)]),
        news_gate=rng.choice([None, rng.uniform(0.1, 0.6)]),
        news_follow=rng.choice([None, None, rng.uniform(0.3, 0.9)]),
        event_blackout=rng.choice([0, 0, rng.randint(15, 180)]),
        black_weight=rng.choice([0.0, rng.uniform(0.1, 0.8)]),
        hack_blackout=rng.choice([0, 0, rng.randint(2, 24)]),
    ).normalized()


def random_news_genes(rng: random.Random, black: bool = True) -> dict:
    """Valores aleatórios para os genes de notícia (black só quando permitido)."""
    genes = {
        "news_gate": rng.choice([None, rng.uniform(0.1, 0.6)]),
        "news_follow": rng.choice([None, rng.uniform(0.3, 0.9)]),
        "event_blackout": rng.choice([0, rng.randint(15, 180)]),
    }
    if black:
        genes["black_weight"] = rng.choice([0.0, rng.uniform(0.1, 0.8)])
        genes["hack_blackout"] = rng.choice([0, rng.randint(2, 24)])
    if not any(v for v in genes.values()):
        genes["news_gate"] = rng.uniform(0.1, 0.6)  # garante ao menos um gene ligado
    return genes


def mutate(genome: Genome, rng: random.Random, scale: float = 0.15, news_toggle: float = 0.15) -> Genome:
    """Pequena variação aleatória em cada gene (scale = desvio relativo).

    Genes de notícia também podem ser ligados/desligados (news_toggle), senão uma
    população que nasceu sem eles nunca os descobriria.
    """

    def jitter(value: float) -> float:
        return value * math.exp(rng.gauss(0, scale))

    fresh = random_news_genes(rng)

    def toggle(name: str, value, off):
        if rng.random() < news_toggle:
            return fresh[name] if value in (None, 0, 0.0) and fresh[name] not in (None, 0, 0.0) else off
        return value

    news_gate = toggle("news_gate", genome.news_gate, None)
    news_follow = toggle("news_follow", genome.news_follow, None)
    event_blackout = toggle("event_blackout", genome.event_blackout, 0)
    black_weight = toggle("black_weight", genome.black_weight, 0.0)
    hack_blackout = toggle("hack_blackout", genome.hack_blackout, 0)
    return replace(
        genome,
        params={k: jitter(v) for k, v in genome.params.items()},
        size=jitter(genome.size),
        vol_filter=None if genome.vol_filter is None else jitter(genome.vol_filter),
        trend_filter=None if genome.trend_filter is None else jitter(genome.trend_filter),
        cooldown=round(jitter(genome.cooldown + 1)) - 1,
        news_gate=None if news_gate is None else jitter(news_gate),
        news_follow=None if news_follow is None else jitter(news_follow),
        event_blackout=round(jitter(event_blackout + 1)) - 1 if event_blackout else 0,
        black_weight=jitter(black_weight) if black_weight else 0.0,
        hack_blackout=round(jitter(hack_blackout + 1)) - 1 if hack_blackout else 0,
    ).normalized()


# indicadores sobre arrays (só a cauda necessária)


def _sma(values: np.ndarray, n: int) -> float:
    return float(values[-n:].mean())


def _rsi(close: np.ndarray, period: int) -> float:
    diff = np.diff(close[-(period + 1) :])
    gain, loss = diff[diff > 0].sum(), -diff[diff < 0].sum()
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def range_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    """Amplitude média dos últimos n candles como fração do preço."""
    return float(((high[-n:] - low[-n:]) / close[-n:]).mean())


def _last(c, column: str, default: float) -> float:
    if column not in c.columns:
        return default
    value = c[column].iloc[-1]
    return default if value != value else float(value)  # NaN → padrão


def _news_context(c, g: Genome) -> dict[str, float]:
    """Contexto de notícias no último candle fechado; neutro se as colunas não existirem."""
    white = _last(c, "news_sentiment", 0.0)
    black = _last(c, "black_rw_mean_sent", _last(c, "black_mean_sent", 0.0))
    return {
        "sentiment": white + g.black_weight * black,
        "minutes_to_event": _last(c, "news_minutes_to_next_event", 1e9),
        "minutes_since_hack": _last(c, "black_sec_minutes_since_incident", 1e9),
    }


class GenomeEngine:
    """Motor gerado por um genoma. Guarda estado entre candles (posição, cooldown)."""

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self.name = f"{genome.strategy}:{genome.params}"
        self.warmup = genome.warmup
        self._long = False
        self._since_entry = 10**9

    def decide(self, view: MarketView) -> Signal:
        g = self.genome
        c = view.candles
        if len(c) < self.warmup:
            return Signal(Side.FLAT, reason="aquecendo")
        close = c["close"].to_numpy()
        high = c["high"].to_numpy()
        low = c["low"].to_numpy()
        self._since_entry += 1

        news = _news_context(c, g)
        want = self._strategy_wants_long(close, high, low)
        if not want and not self._long and g.news_follow is not None and news["sentiment"] >= g.news_follow:
            want = True  # notícia forte a favor abre posição sem sinal técnico
        if want and not self._long:
            if g.news_gate is not None and news["sentiment"] <= -g.news_gate:
                want = False
            elif g.event_blackout and news["minutes_to_event"] < g.event_blackout:
                want = False
            elif g.hack_blackout and news["minutes_since_hack"] < g.hack_blackout * 60:
                want = False
            elif g.trend_filter and close[-1] < _sma(close, g.trend_filter):
                want = False
            elif g.vol_filter is not None and range_pct(high, low, close) > g.vol_filter:
                want = False
            elif self._since_entry <= g.cooldown:
                want = False
            else:
                self._since_entry = 0

        self._long = want
        if want:
            return Signal(Side.BUY, confidence=0.5, size=g.size, reason=g.strategy)
        return Signal(Side.FLAT, reason=g.strategy)

    def _strategy_wants_long(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> bool:
        p = self.genome.params
        match self.genome.strategy:
            case "ma_cross":
                return _sma(close, p["fast"]) > _sma(close, p["slow"])
            case "breakout":
                n = p["lookback"]
                if not self._long:
                    return close[-1] > high[-n - 1 : -1].max()
                exit_n = max(n // 2, 2)
                return not close[-1] < low[-exit_n - 1 : -1].min()
            case "rsi_revert":
                rsi = _rsi(close, p["period"])
                return rsi < p["low"] if not self._long else rsi < p["high"]
        raise AssertionError(self.genome.strategy)
