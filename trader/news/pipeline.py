"""Monta a base de notícias pronta para features: itens + rótulos + visibilidade + histórias."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from trader.news.enrich import Enricher
from trader.news.fulltext import apply_fulltext
from trader.news.reputation import Reputation, ReputationConfig, direction_outcomes, story_outcomes
from trader.news.sources import TIERS
from trader.news.store import DEFAULT_LATENCY, NewsStore, visible_at
from trader.news.stories import StoryState, assign_stories
from trader.storage.candles import CandleStore
from trader.storage.atomic import replace_with_retry, unique_tmp

CALENDAR_FILE = Path("news") / "calendar.parquet"

# séries de preço usadas para resolver acertos de direção: (exchange, símbolo, timeframe)
PRICE_SOURCES = {
    "BTC": [("binanceusdm", "BTC/USDT", "15m"), ("binance", "BTC/USDT", "4h")],
    "IBOV": [("b3", "BOVA11", "1d")],
    "PETR4": [("b3", "PETR4", "1d")],
    "VALE3": [("b3", "VALE3", "1d")],
}


COLLECTIONS = ("items", "gdelt_gkg")


def read_items(data: Path, since: pd.Timestamp | None = None, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Coleções (RSS/DOC em items/, GDELT GKG em gdelt_gkg/) sem duplicatas por id.

    Passe `since`/`until` sempre que possível: sem período, lê o histórico inteiro (milhões de linhas)."""
    frames = [f for c in COLLECTIONS if len(f := NewsStore(data, subdir=c).read_range(since, until))]
    if not frames:
        return NewsStore(data).read_range(since, until)
    items = pd.concat(frames, ignore_index=True)
    return items.sort_values("first_seen_at").drop_duplicates("id", keep="first").reset_index(drop=True)


STORY_WINDOW = pd.Timedelta(hours=24)
STORY_SEED = pd.Timedelta(hours=72)  # histórias do mês anterior que ainda podem receber itens
DERIVED_COLUMNS = [
    "id", "domain", "tier", "original_lang", "original_text", "translated_text", "query_assets", "assets",
    "sentiment", "confidence", "event_type", "label_source", "published_at", "visible_at", "story_id", "story_rank",
]


def _month_sources(data: Path, month: str) -> list[Path]:
    return [p for c in COLLECTIONS if (p := data / "news" / c / f"{month}.parquet").exists()]


def _stamp_of(paths: list[Path]) -> list:
    return [[str(p.name), p.parent.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]


def labels_stamp(data: Path) -> list:
    from trader.news.fulltext import labels_stamp as fulltext_stamp

    cache = data / "news" / "cache" / "labels.jsonl"
    return ([cache.stat().st_size] if cache.exists() else [0]) + fulltext_stamp(data)


def derived_month(data: Path, month: str, enricher: Enricher, latency: pd.Timedelta, prev: pd.DataFrame | None) -> pd.DataFrame:
    """Itens listados do mês (por mês de publicação), rotulados, com visibilidade e histórias. Cache em disco.

    O cache vale enquanto os arquivos do mês, o cache de rótulos e o mês anterior não mudarem.
    As histórias continuam as do mês anterior (itens visíveis nas últimas 72 h dele).
    """
    folder = data / "news" / "derived" / f"lat{int(latency.total_seconds())}"
    path, meta = folder / f"{month}.parquet", folder / f"{month}.json"
    prev_stamp = None if prev is None else prev.attrs.get("stamp_hash")
    stamp = {"src": _stamp_of(_month_sources(data, month)), "labels": labels_stamp(data), "prev": prev_stamp}
    if path.exists() and meta.exists() and json.loads(meta.read_text()) == stamp:
        df = pd.read_parquet(path)
        df.attrs["stamp_hash"] = _digest(stamp)
        return df
    frames = [pd.read_parquet(p) for p in _month_sources(data, month)]
    if not frames:
        empty = pd.DataFrame(columns=DERIVED_COLUMNS)
        empty.attrs["stamp_hash"] = _digest(stamp)
        return empty
    raw = pd.concat(frames, ignore_index=True)
    # fontes fora da lista ficam só no bruto: as features e a reputação não as usam
    raw = raw[raw["tier"] != 9].sort_values("first_seen_at").drop_duplicates("id", keep="first")
    raw = raw.reset_index(drop=True)
    if raw.empty:
        empty = pd.DataFrame(columns=DERIVED_COLUMNS)
        empty.attrs["stamp_hash"] = _digest(stamp)
        return empty
    labeled = enricher.enrich(raw, use_llm=False)
    labeled["visible_at"] = visible_at(labeled, latency)
    labeled = apply_fulltext(labeled, data)  # rótulo do texto completo só vale depois de existir
    state = StoryState(STORY_WINDOW, 0.5)
    if prev is not None and len(prev):
        tail = prev[(prev["visible_at"] >= labeled["visible_at"].min() - STORY_SEED) & (prev["tier"] != 9)]
        state.seed(tail)
    df = assign_stories(labeled, STORY_WINDOW, state=state)[DERIVED_COLUMNS]
    folder.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    df.to_parquet(tmp, index=False)
    replace_with_retry(tmp, path)
    meta.write_text(json.dumps(stamp))
    df.attrs["stamp_hash"] = _digest(stamp)
    return df


def _digest(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_items_range(
    data: Path, since: pd.Timestamp, until: pd.Timestamp, latency: pd.Timedelta = DEFAULT_LATENCY,
    enricher: Enricher | None = None,
) -> pd.DataFrame:
    """Itens visíveis em [since, until], lendo só os meses necessários (mais o anterior, para as histórias)."""
    enricher = enricher or Enricher(data)
    lo = (since - pd.Timedelta(days=1)).tz_convert(None).to_period("M")
    hi = until.tz_convert(None).to_period("M")
    months = [str(m) for m in pd.period_range(lo - 1, hi, freq="M")]
    frames, prev = [], None
    for m in months:
        prev = derived_month(data, m, enricher, latency, prev)
        frames.append(prev)
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=DERIVED_COLUMNS)
    items = pd.concat(frames, ignore_index=True)
    return items[(items["visible_at"] >= since) & (items["visible_at"] <= until)].reset_index(drop=True)


def range_stamp(data: Path, since: pd.Timestamp, until: pd.Timestamp) -> tuple:
    """Carimbo barato do período (só os arquivos dos meses envolvidos)."""
    lo = (since - pd.Timedelta(days=1)).tz_convert(None).to_period("M")
    hi = until.tz_convert(None).to_period("M")
    paths = [p for m in pd.period_range(lo - 1, hi, freq="M") for p in _month_sources(data, str(m))]
    return tuple(map(tuple, _stamp_of(paths))) + tuple(labels_stamp(data))


def load_items(data: Path, enricher: Enricher, latency: pd.Timedelta = DEFAULT_LATENCY,
               since: pd.Timestamp | None = None, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Só usa rótulos já em cache (ou tom do GDELT / léxico): montar a base nunca chama a LLM."""
    items = read_items(data, since, until)
    if items.empty:
        return items
    items = enricher.enrich(items, use_llm=False)
    items["visible_at"] = visible_at(items, latency)
    items = apply_fulltext(items, data)
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


# ------------------------------------------------------------------ reputação em fotos semanais

REP_LOOKBACK = pd.Timedelta(days=180)


def week_start(t: pd.Timestamp) -> pd.Timestamp:
    return t.floor("D") - pd.Timedelta(days=t.dayofweek)


def reputation_snapshots(
    data: Path, weeks: list[pd.Timestamp], horizon: pd.Timedelta = pd.Timedelta(hours=4),
    latency: pd.Timedelta = DEFAULT_LATENCY,
) -> pd.DataFrame:
    """Pesos (week, domain, weight) em cada início de semana, só com resultados resolvidos antes dela.

    Incremental: semanas já calculadas vêm do disco; só as que faltam são calculadas, cada uma
    olhando os últimos 180 dias (a reputação decai com meia-vida de 90 dias, então o resto pesa pouco).
    A semana corrente só é gravada quando termina (antes disso, é recalculada a cada chamada).
    """
    path = data / "news" / "reputation" / f"weights_{int(horizon.total_seconds())}s.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["week", "domain", "weight"])
    done = set(pd.to_datetime(have["week"], utc=True)) if len(have) else set()
    now_week = week_start(pd.Timestamp.now(tz="UTC"))
    new, current = [], []
    prices = {a: p for a in PRICE_SOURCES if (p := close_prices(data, a)) is not None}
    for w in sorted(set(weeks)):
        if w in done:
            continue
        items = load_items_range(data, w - REP_LOOKBACK, w, latency)
        items = items[items["tier"] != 9]
        if items.empty:
            snap = pd.DataFrame({"week": [w], "domain": [""], "weight": [0.0]})
        else:
            outcomes = pd.concat([direction_outcomes(items, prices, horizon), story_outcomes(items)], ignore_index=True)
            rep = Reputation(outcomes, tiers=dict(TIERS), items=items)
            domains = sorted(set(items["domain"]))
            snap = pd.DataFrame({"week": w, "domain": domains, "weight": rep.weights(w, domains).to_numpy()})
        (current if w >= now_week else new).append(snap)
    if new:
        have = pd.concat([have, *new], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = unique_tmp(path)
        have.to_parquet(tmp, index=False)
        replace_with_retry(tmp, path)
    out = pd.concat([have, *current], ignore_index=True) if current else have
    out["week"] = pd.to_datetime(out["week"], utc=True)
    return out[out["week"].isin(set(weeks))]


def default_weight(domain: str, config: ReputationConfig = ReputationConfig()) -> float:
    """Peso de uma fonte sem histórico: média do prior do nível, com o piso de exploração."""
    from trader.news.reputation import PRIORS
    from trader.news.sources import tier_of

    a, b = PRIORS.get(tier_of(domain), PRIORS[None])
    return max(a / (a + b), config.explore_floor)
