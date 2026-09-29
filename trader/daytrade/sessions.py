"""Sessões de negociação (dia operacional)."""
from __future__ import annotations

import datetime as dt

import pandas as pd


def session_keys(ts: pd.Series, session_end: dt.time) -> pd.Series:
    """Identificador da sessão de cada candle; a sessão começa no horário de fim da anterior."""
    offset = pd.Timedelta(hours=session_end.hour, minutes=session_end.minute)
    return (ts - offset).dt.floor("D")
