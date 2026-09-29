"""Auditoria de vazamento de futuro nas features de notícias.

truncation_audit: recalcula as features de uma amostra de instantes usando só os
itens visíveis até cada instante e compara com o cálculo sobre a base inteira.
Qualquer diferença = a feature usou algo do futuro. Deve dar zero.

shift_sensitivity: atrasa a visibilidade de todas as notícias em N e mede o
quanto as features mudam. Mostra quanto a estratégia depende da latência (se o
resultado piora muito com +5 min, a vantagem pode ser só velocidade irreal).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

FeatureFn = Callable[[pd.Series, pd.DataFrame], pd.DataFrame]  # (candle_ts, items) -> features


def truncation_audit(build: FeatureFn, candle_ts: pd.Series, items: pd.DataFrame, as_of_of: Callable, samples: int = 50, seed: int = 0) -> pd.DataFrame:
    """as_of_of(ts) -> instante de decisão do candle. Retorna as linhas divergentes (vazio = ok)."""
    full = build(candle_ts, items)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(candle_ts), size=min(samples, len(candle_ts)), replace=False)
    bad = []
    for i in sorted(picks):
        ts = candle_ts.iloc[i]
        cut = items[items["visible_at"] <= as_of_of(ts)]
        one = build(candle_ts.iloc[[i]], cut)
        diff = (one.iloc[0] - full.loc[ts]).abs()
        if (diff.fillna(0) > 1e-9).any():
            bad.append({"ts": ts, **diff[diff > 1e-9].to_dict()})
    return pd.DataFrame(bad)


def shift_sensitivity(build: FeatureFn, candle_ts: pd.Series, items: pd.DataFrame, shift: pd.Timedelta) -> pd.DataFrame:
    """Por coluna: fração de candles que mudam e maior diferença absoluta com a visibilidade atrasada."""
    base = build(candle_ts, items)
    delayed = items.assign(visible_at=items["visible_at"] + shift)
    moved = build(candle_ts, delayed)
    diff = (moved - base).abs()
    return pd.DataFrame({"changed_frac": (diff > 1e-9).mean(), "max_abs_diff": diff.max()})
