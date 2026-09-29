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
        ).normalized()

    @property
    def warmup(self) -> int:
        need = max(self.params.values()) + 2
        if self.trend_filter:
            need = max(need, self.trend_filter + 1)
        return max(need, 16)


def random_genome(rng: random.Random, strategy: str | None = None) -> Genome:
    strategy = strategy or rng.choice(sorted(STRATEGIES))
    params = {k: rng.randint(lo, hi) for k, (lo, hi) in STRATEGIES[strategy].items()}
    return Genome(
        strategy=strategy,
        params=params,
        size=rng.uniform(0.3, 1.0),
        vol_filter=rng.choice([None, rng.uniform(0.01, 0.06)]),
        trend_filter=rng.choice([None, rng.randint(50, 200)]),
        cooldown=rng.choice([0, 0, rng.randint(1, 12)]),
    ).normalized()


def mutate(genome: Genome, rng: random.Random, scale: float = 0.15) -> Genome:
    """Pequena variação aleatória em cada gene (scale = desvio relativo)."""

    def jitter(value: float) -> float:
        return value * math.exp(rng.gauss(0, scale))

    return replace(
        genome,
        params={k: jitter(v) for k, v in genome.params.items()},
        size=jitter(genome.size),
        vol_filter=None if genome.vol_filter is None else jitter(genome.vol_filter),
        trend_filter=None if genome.trend_filter is None else jitter(genome.trend_filter),
        cooldown=round(jitter(genome.cooldown + 1)) - 1,
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

        want = self._strategy_wants_long(close, high, low)
        if want and not self._long:
            if g.trend_filter and close[-1] < _sma(close, g.trend_filter):
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
