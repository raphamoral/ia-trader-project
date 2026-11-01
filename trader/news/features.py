"""Features de notícias por candle, calculadas no fechamento do candle (momento da decisão).

Para o candle que abre em ts com duração tf, as_of = ts + tf, e só entram itens
com visible_at <= as_of: o mesmo contrato da MarketView.

Colunas (w = janela, ex.: 1h, 4h, 24h):
  news_count_w     itens relevantes visíveis na janela
  story_count_w    histórias novas (primeiro item da história) na janela
  spread_w         itens por história (mesma notícia em várias fontes/idiomas)
  langs_w          idiomas distintos
  tier1_count_w    itens de fontes nível 1
  sentiment_w      média do sentimento ponderada por confiança x nível x reputação,
                   com fatia máxima por fonte
  shock_1h         itens fortes (|sentimento| e confiança >= 0,6) na última hora
  burst_1h         volume da última hora / média horária dos 7 dias anteriores
  minutes_to_next_event / minutes_since_last_event   calendário macro do ativo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.domain import parse_timeframe
from trader.news.calendar import RELEVANT
from trader.news.reputation import Reputation, cap_shares
from trader.news.sources import TIER_WEIGHT, UNLISTED_TIER

SYMBOL_ASSET = {"BTC/USDT": "BTC", "BTCUSDT": "BTC", "BOVA11": "IBOV", "PETR4": "PETR4", "VALE3": "VALE3", "USDBRL": "USDBRL"}
WINDOWS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "24h": pd.Timedelta(hours=24)}


def relevant(items: pd.DataFrame, asset: str, include_unlisted: bool = False) -> pd.DataFrame:
    """Itens sobre o ativo: pelos rótulos; sem rótulo de ativo, pela busca que os trouxe.

    Fontes fora da lista branca (tier=UNLISTED_TIER) ficam de fora, salvo include_unlisted.
    """
    if not include_unlisted and len(items):
        items = items[items["tier"] != UNLISTED_TIER]
    labeled = items["assets"].fillna("").str.split(",")
    queried = items["query_assets"].fillna("").str.split(",")
    mask = np.array([asset in lab if any(lab) else asset in q for lab, q in zip(labeled, queried)], dtype=bool)
    return items[mask].sort_values("visible_at").reset_index(drop=True)


def build_features(
    candle_ts: pd.Series | pd.DatetimeIndex,
    timeframe: str,
    items: pd.DataFrame,
    asset: str,
    calendar: pd.DataFrame | None = None,
    reputation: Reputation | None = None,
    max_share: float = 0.25,
) -> pd.DataFrame:
    """items precisa de: visible_at, story_id, story_rank, tier, domain, original_lang, sentiment, confidence, assets, query_assets."""
    tf = parse_timeframe(timeframe)
    as_of = pd.DatetimeIndex(candle_ts) + tf
    news = relevant(items, asset)
    vis = news["visible_at"].to_numpy(dtype="datetime64[ns]") if len(news) else np.array([], dtype="datetime64[ns]")
    rep_cache: dict[pd.Timestamp, pd.Series] = {}

    def rep_weights(t: pd.Timestamp, domains: list[str]) -> np.ndarray:
        if reputation is None:
            return np.ones(len(domains))
        day = t.floor("D")  # reputação do início do dia: resultados resolvidos antes de t
        if day not in rep_cache:
            rep_cache[day] = reputation.weights(day, sorted(set(news["domain"])))
        return rep_cache[day].reindex(domains).fillna(reputation.config.explore_floor).to_numpy()

    rows = []
    for t in as_of:
        t64 = t.to_datetime64()
        hi = np.searchsorted(vis, t64, side="right")
        row = {}
        for name, w in WINDOWS.items():
            lo = np.searchsorted(vis, (t - w).to_datetime64(), side="right")
            win = news.iloc[lo:hi]
            n = len(win)
            stories = int((win["story_rank"] == 0).sum()) if n else 0
            row[f"news_count_{name}"] = n
            row[f"story_count_{name}"] = stories
            row[f"spread_{name}"] = n / stories if stories else 0.0
            row[f"langs_{name}"] = win["original_lang"].nunique() if n else 0
            row[f"tier1_count_{name}"] = int((win["tier"] == 1).sum()) if n else 0
            row[f"sentiment_{name}"] = _weighted_sentiment(win, rep_weights(t, win["domain"].tolist()) if n else None, max_share)
        lo1 = np.searchsorted(vis, (t - pd.Timedelta(hours=1)).to_datetime64(), side="right")
        last_hour = news.iloc[lo1:hi]
        row["shock_1h"] = int(((last_hour["sentiment"].abs() >= 0.6) & (last_hour["confidence"] >= 0.6)).sum())
        lo7 = np.searchsorted(vis, (t - pd.Timedelta(days=7)).to_datetime64(), side="right")
        base = (lo1 - lo7) / (7 * 24 - 1)
        row["burst_1h"] = (hi - lo1) / base if base > 0 else 0.0
        rows.append(row)

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(candle_ts, name="ts"))
    if calendar is not None and len(calendar):
        events = calendar[calendar["event"].isin(RELEVANT.get(asset, set()))]["scheduled_at"].sort_values()
        ev = events.to_numpy(dtype="datetime64[ns]")
        at = as_of.tz_convert(None).to_numpy(dtype="datetime64[ns]")  # UTC sem fuso, como `ev`
        nxt = np.searchsorted(ev, at, side="right")
        prev = nxt - 1
        minute = np.timedelta64(1, "m")
        out["minutes_to_next_event"] = [(ev[i] - a) / minute if i < len(ev) else np.nan for i, a in zip(nxt, at)]
        out["minutes_since_last_event"] = [(a - ev[i]) / minute if i >= 0 else np.nan for i, a in zip(prev, at)]
    return out


def _weighted_sentiment(win: pd.DataFrame, rep: np.ndarray | None, max_share: float) -> float:
    if rep is None or len(win) == 0:
        return 0.0
    w = win["confidence"].to_numpy() * win["tier"].map(TIER_WEIGHT).fillna(0.3).to_numpy() * rep
    per_source = pd.Series(w, index=win["domain"].to_numpy()).groupby(level=0).sum()
    if per_source.sum() == 0:
        return 0.0
    capped = cap_shares(per_source, max_share) if len(per_source) > 1 else per_source
    scale = (capped / per_source).reindex(win["domain"].to_numpy()).fillna(0).to_numpy()
    w = w * scale
    return float((w * win["sentiment"].to_numpy()).sum() / w.sum()) if w.sum() else 0.0


# ---------------------------------------------------------------------------
# Contrato para os motores (day trade e outros):
#     news_features(symbol, timeframe, index) -> DataFrame com o mesmo índice
# A linha t usa só itens com visible_at <= t. Passe o instante da decisão
# (fechamento do candle). Passar a abertura também é seguro, só mais conservador.
# ---------------------------------------------------------------------------

import functools
from pathlib import Path

from trader.news.enrich import EVENT_TYPES

MAIN_WINDOW = {"intraday": "4h", "daily": "3d"}
SERVE_WINDOWS = {
    "intraday": {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "24h": pd.Timedelta(hours=24)},
    "daily": {"24h": pd.Timedelta(hours=24), "3d": pd.Timedelta(days=3), "7d": pd.Timedelta(days=7)},
}
RECENCY_CAP_MIN = 7 * 24 * 60  # "nenhum evento recente" vira 7 dias, não NaN


def _asset_of(symbol: str) -> str:
    return SYMBOL_ASSET.get(symbol, SYMBOL_ASSET.get(symbol.replace("/", ""), symbol))


@functools.lru_cache(maxsize=16)
def _prepared(data: str, asset: str, latency_min: float, use_reputation: bool, since: pd.Timestamp,
              until: pd.Timestamp, stamp: tuple) -> dict:
    """Arrays por item do ativo no período, ordenados por visibilidade, com somas acumuladas. Cacheado."""
    from trader.news.pipeline import default_weight, load_calendar, load_items_range, reputation_snapshots, week_start

    root = Path(data)
    latency = pd.Timedelta(minutes=latency_min)
    items = load_items_range(root, since, until, latency)
    news = relevant(items, asset) if len(items) else items
    n = len(news)
    vis = news["visible_at"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]") if n else np.array([], "datetime64[ns]")
    rep_w = np.ones(n)
    if n and use_reputation:
        # peso congelado na foto do início da semana em que o item ficou visível
        # (resultados resolvidos antes dela): causal por construção
        weeks = news["visible_at"].map(week_start)
        snaps = reputation_snapshots(root, sorted(set(weeks)), latency=latency)
        table = {(w, d): x for w, d, x in zip(snaps["week"], snaps["domain"], snaps["weight"])}
        rep_w = np.array([table.get((w, d), default_weight(d)) for w, d in zip(weeks, news["domain"])])
    w = news["confidence"].to_numpy(float) * news["tier"].map(TIER_WEIGHT).fillna(0.3).to_numpy(float) * rep_w if n else np.array([])
    s = news["sentiment"].to_numpy(float) if n else np.array([])

    def cum(x):
        return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])

    calendar = load_calendar(root)
    ev = np.array([], "datetime64[ns]")
    if calendar is not None and len(calendar):
        sel = calendar[calendar["event"].isin(RELEVANT.get(asset, set()))]["scheduled_at"].sort_values()
        ev = sel.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    return {
        "vis": vis,
        "c_count": cum(np.ones(n)),
        "c_w": cum(w),
        "c_ws": cum(w * s),
        "c_conf": cum(news["confidence"].to_numpy(float) if n else []),
        "c_first": cum((news["story_rank"] == 0).to_numpy(float) if n else []),
        "c_tier1": cum((news["tier"] == 1).to_numpy(float) if n else []),
        "c_shock": cum(((news["sentiment"].abs() >= 0.6) & (news["confidence"] >= 0.6)).to_numpy(float) if n else []),
        "by_lang": {k: cum((news["original_lang"] == k).to_numpy(float)) for k in (news["original_lang"].unique() if n else [])},
        "by_source": {k: cum((news["domain"] == k).to_numpy(float)) for k in (news["domain"].unique() if n else [])},
        "by_event": {k: vis[(news["event_type"] == k).to_numpy()] for k in EVENT_TYPES} if n else {k: vis for k in EVENT_TYPES},
        "calendar": ev,
    }


def news_features(
    symbol: str,
    timeframe: str,
    index: pd.DatetimeIndex,
    *,
    data: Path | str = "data",
    latency: pd.Timedelta = pd.Timedelta(minutes=5),
    use_reputation: bool = True,
) -> pd.DataFrame:
    """Features de notícias no instante de cada linha do índice (UTC). Colunas prefixadas `news_`.

    Janelas: 1h/4h/24h para intraday (5m, 15m, ...); 24h/3d/7d para 1d.
    Sem notícias para o ativo, devolve zeros (e o calendário, se houver).
    """
    data = Path(data)
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        raise ValueError("index precisa de fuso (UTC)")
    kind = "daily" if parse_timeframe(timeframe) >= pd.Timedelta(days=1) else "intraday"
    from trader.news.pipeline import range_stamp

    longest = max(SERVE_WINDOWS[kind].values())
    # período limitado ao que o índice precisa, arredondado ao dia para reaproveitar o cache
    since = (idx.min() - longest).floor("D")
    until = idx.max().ceil("D")
    p = _prepared(str(data.resolve()), _asset_of(symbol), latency.total_seconds() / 60, use_reputation,
                  since, until, range_stamp(data, since, until))
    t = idx.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    vis = p["vis"]
    hi = np.searchsorted(vis, t, side="right")
    out: dict[str, np.ndarray] = {}

    def window_sum(c, lo):
        return c[hi] - c[lo]

    for name, w in SERVE_WINDOWS[kind].items():
        lo = np.searchsorted(vis, t - w.to_timedelta64(), side="right")
        count = window_sum(p["c_count"], lo)
        weight = window_sum(p["c_w"], lo)
        stories = window_sum(p["c_first"], lo)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"news_count_{name}"] = count
            out[f"news_sentiment_{name}"] = np.where(weight > 0, window_sum(p["c_ws"], lo) / weight, 0.0)
            out[f"news_confidence_{name}"] = np.where(count > 0, window_sum(p["c_conf"], lo) / count, 0.0)
            out[f"news_spread_{name}"] = np.where(stories > 0, count / stories, 0.0)
        out[f"news_stories_{name}"] = stories
        out[f"news_tier1_{name}"] = window_sum(p["c_tier1"], lo)
        out[f"news_langs_{name}"] = sum((window_sum(c, lo) > 0).astype(float) for c in p["by_lang"].values()) if p["by_lang"] else np.zeros(len(t))
        out[f"news_sources_{name}"] = sum((window_sum(c, lo) > 0).astype(float) for c in p["by_source"].values()) if p["by_source"] else np.zeros(len(t))
    main = MAIN_WINDOW[kind]
    for col in ("count", "sentiment", "confidence", "spread"):
        out[f"news_{col}"] = out[f"news_{col}_{main}"]
    lo1 = np.searchsorted(vis, t - np.timedelta64(1, "h"), side="right")
    out["news_shock_1h"] = window_sum(p["c_shock"], lo1)

    minute = np.timedelta64(1, "m")
    for etype, times in p["by_event"].items():
        k = np.searchsorted(times, t, side="right") - 1
        since = np.where(k >= 0, (t - times[np.clip(k, 0, None)]) / minute if len(times) else RECENCY_CAP_MIN, RECENCY_CAP_MIN)
        out[f"news_event_{etype}_minutes"] = np.minimum(since, RECENCY_CAP_MIN)
    ev = p["calendar"]
    nxt = np.searchsorted(ev, t, side="right")
    prev = nxt - 1
    if len(ev):
        out["news_minutes_to_next_event"] = np.where(nxt < len(ev), (ev[np.clip(nxt, 0, len(ev) - 1)] - t) / minute, RECENCY_CAP_MIN)
        out["news_minutes_since_last_event"] = np.where(prev >= 0, (t - ev[np.clip(prev, 0, None)]) / minute, RECENCY_CAP_MIN)
    else:
        out["news_minutes_to_next_event"] = np.full(len(t), float(RECENCY_CAP_MIN))
        out["news_minutes_since_last_event"] = np.full(len(t), float(RECENCY_CAP_MIN))
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, index=index)
