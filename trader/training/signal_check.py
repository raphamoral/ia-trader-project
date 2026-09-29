"""Existe sinal para o Criador aprender?

Compara, nas mesmas autópsias, o sucessor real, a correção por regras, mutações
aleatórias, o pai continuando vivo e a referência, em várias janelas e vários
ativos. Se as correções dirigidas não baterem a mutação aleatória com folga,
treinar a LLM seria ensinar ruído.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from trader.training.creator_data import Example, _candles, candidates_for, life_over


@dataclass(frozen=True)
class Verdict:
    means: dict[str, float]
    rules_vs_random: float  # fração de autópsias em que regras > média das mutações
    ci_low: float  # IC 95% (bootstrap) da diferença média regras − mutação
    ci_high: float
    n: int

    @property
    def has_signal(self) -> bool:
        return self.ci_low > 0


def _covered(symbol: str, timeframe: str, ex: Example) -> bool:
    candles = _candles(symbol, timeframe)
    return not candles.empty and candles["ts"].iloc[0] <= ex.windows[0][0] - (ex.windows[0][1] - ex.windows[0][0])


def cross_asset_scores(job: tuple[Example, list[str]]) -> dict[str, float] | None:
    """Nota de cada candidato = média das notas nos ativos com dados na janela."""
    ex, symbols = job
    usable = [s for s in symbols if _covered(s, ex.timeframe, ex)]
    if not usable:
        return None
    out = {}
    for name, genome in candidates_for(ex).items():
        lives = [life_over(genome, s, ex.timeframe, ex.windows) for s in usable]
        out[name] = statistics.mean(life.score for life in lives)
    return out


def verdict(scores: list[dict[str, float]], iterations: int = 2000, seed: int = 0) -> Verdict:
    names = scores[0].keys()
    means = {n: statistics.mean(s[n] for s in scores) for n in names}
    mutation_keys = [k for k in names if k.startswith("mutation")]
    diffs = [s["rules"] - statistics.mean(s[k] for k in mutation_keys) for s in scores]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(iterations))
    return Verdict(
        means=means,
        rules_vs_random=sum(d > 0 for d in diffs) / len(diffs),
        ci_low=boots[int(0.025 * iterations)],
        ci_high=boots[int(0.975 * iterations)],
        n=len(diffs),
    )

