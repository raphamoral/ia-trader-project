"""Monta a base de notícias pronta para features: itens + rótulos + visibilidade + histórias."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from trader.news.enrich import Enricher
from trader.news.reputation import Reputation, ReputationConfig, direction_outcomes, story_outcomes
from trader.news.sources import TIERS
from trader.news.store import DEFAULT_LATENCY, NewsStore, visible_at
from trader.news.stories import assign_stories
from trader.storage.candles import CandleStore

CALENDAR_FILE = Path("news") / "calendar.parquet"

# séries de preço usadas para resolver acertos de direção: (exchange, símbolo, timeframe)
PRICE_SOURCES = {
    "BTC": [("binanceusdm", "BTC/USDT", "15m"), ("binance", "BTC/USDT", "4h")],
    "IBOV": [("b3", "BOVA11", "1d")],
    "PETR4": [("b3", "PETR4", "1d")],
    "VALE3": [("b3", "VALE3", "1d")],
}


COLLECTIONS = ("items", "gdelt_gkg")


def read_items(data: Path) -> pd.DataFrame:
    """Todas as coleções (RSS/DOC em items/, GDELT GKG em gdelt_gkg/), sem duplicatas por id."""
    frames = [f for c in COLLECTIONS if len(f := NewsStore(data, subdir=c).read_all())]
    if not frames:
        return NewsStore(data).read_all()
    items = pd.concat(frames, ignore_index=True)
    return items.sort_values("first_seen_at").drop_duplicates("id", keep="first").reset_index(drop=True)


def load_items(data: Path, enricher: Enricher, latency: pd.Timedelta = DEFAULT_LATENCY) -> pd.DataFrame:
    """Só usa rótulos já em cache (ou tom do GDELT / léxico): montar a base nunca chama a LLM."""
    items = read_items(data)
    if items.empty:
        return items
    items = enricher.enrich(items, use_llm=False)
    items["visible_at"] = visible_at(items, latency)
    return assign_stories(items)


def load_calendar(data: Path) -> pd.DataFrame | None:
    path = data / CALENDAR_FILE
    return pd.read_parquet(path) if path.exists() else None


def close_prices(data: Path, asset: str) -> pd.Series | None:
    """Fechamentos indexados pelo instante em que ficam conhecidos (fim do candle)."""
    from trader.domain import parse_timeframe

    store = CandleStore(data)
    for exchange, symbol, tf in PRICE_SOURCES.get(asset, []):
        c = store.read(exchange, symbol, tf)
        if len(c):
            return pd.Series(c["close"].to_numpy(), index=pd.DatetimeIndex(c["ts"] + parse_timeframe(tf)))
    return None


def build_reputation(data: Path, items: pd.DataFrame, horizon: pd.Timedelta, config: ReputationConfig = ReputationConfig()) -> Reputation:
    prices = {a: p for a in PRICE_SOURCES if (p := close_prices(data, a)) is not None}
    outcomes = pd.concat([direction_outcomes(items, prices, horizon), story_outcomes(items)], ignore_index=True)
    return Reputation(outcomes, tiers=dict(TIERS), config=config, items=items)


HIGH_IMPACT = {"monetary_policy", "macro_data", "hack_security", "regulation", "earnings", "flows_etf"}


def select_for_llm(items: pd.DataFrame, daily_cap: int, reputation: Reputation | None = None) -> pd.DataFrame:
    """Subconjunto do histórico que vale uma chamada de LLM, por dia de visibilidade.

    Prioridade: nível 1 > evento de alto impacto (pelos temas do GKG) > |tom| forte >
    reputação da fonte naquele dia. Só itens ainda sem rótulo da LLM. Uma notícia por história.
    """
    cand = items[items["label_source"] != "llm"]
    if "story_rank" in cand.columns:
        cand = cand[cand["story_rank"] == 0]
    if cand.empty:
        return cand
    score = (
        (cand["tier"] == 1).astype(float) * 4
        + cand["event_type"].isin(HIGH_IMPACT).astype(float) * 2
        + cand["sentiment"].abs().clip(0, 1)
    )
    if reputation is not None:
        day = cand["visible_at"].dt.floor("D")
        for d, idx in cand.groupby(day).groups.items():
            w = reputation.weights(d, cand.loc[idx, "domain"].tolist())
            score.loc[idx] += w.to_numpy()
    ranked = cand.assign(_score=score, _day=cand["visible_at"].dt.floor("D"))
    ranked = ranked.sort_values(["_day", "_score"], ascending=[True, False])
    return ranked.groupby("_day").head(daily_cap).drop(columns=["_score", "_day"])
