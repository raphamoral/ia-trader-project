"""Pipeline de análise compartilhado pela CLI e pela API de features.

itens black enriquecidos → spam → casamento com as confiáveis → lags →
desfechos → reputação. Nada aqui escreve em disco.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trader.blacknews import leadlag, spam
from trader.blacknews.adapters import load_trusted
from trader.blacknews.reputation import black_outcomes, black_reputation
from trader.blacknews.schema import BlackStore

DATA = Path("data")
BLACK_ROOT = DATA / "news_black"
HORIZON = pd.Timedelta(hours=48)


@dataclass
class Analysis:
    scored: pd.DataFrame
    trusted: pd.DataFrame
    matches: pd.DataFrame
    lags: pd.DataFrame
    outcomes: pd.DataFrame
    reputation: object  # trader.news.reputation.Reputation
    data_end: pd.Timestamp


def load_black(since: pd.Timestamp | None = None, until: pd.Timestamp | None = None, root: Path = BLACK_ROOT) -> pd.DataFrame:
    items = BlackStore(root).all()
    items = items[items["enriched_by"].isin(["llm", "lexicon"])]
    if since is not None:
        items = items[items["visible_at"] >= since]
    if until is not None:
        items = items[items["visible_at"] < until]
    return items.reset_index(drop=True)


def analyze_frames(items: pd.DataFrame, trusted: pd.DataFrame, verify=None) -> Analysis:
    scored = spam.spam_scores(items)
    matches = leadlag.StoryMatcher(verify=verify).match(scored, trusted)
    ends = [scored["visible_at"].max()] + ([trusted["visible_at"].max()] if len(trusted) else [])
    data_end = max(ends)
    lags = leadlag.item_lags(scored, matches, data_end, HORIZON)
    outs = black_outcomes(lags, scored)
    rep = black_reputation(outs, scored)
    return Analysis(scored, trusted, matches, lags, outs, rep, data_end)


def analyze(since=None, until=None, verify_calls: int = 0, data: Path = DATA) -> Analysis | None:
    items = load_black(since, until, Path(data) / "news_black")
    if items.empty:
        return None
    trusted = load_trusted(
        since - pd.Timedelta(hours=72) if since is not None else None,
        until + HORIZON if until is not None else None,
        data,
    )
    # sempre com o verificador: verify_calls=0 usa só o que já está em cache (sem chamar a LLM),
    # assim a CLI e a API de features enxergam os mesmos casamentos
    from trader.evolution.creator import _post_json

    verify = leadlag.LLMVerifier(
        leadlag.VerifyCache(Path(data) / "news_black" / "verify_cache.sqlite"), _post_json,
        "http://localhost:11434/v1", "qwen2.5:7b", max_calls=verify_calls,
    )
    return analyze_frames(items, trusted, verify)


@dataclass
class SecurityAnalysis:
    items: pd.DataFrame  # extraídos e deduplicados
    confirmations: pd.DataFrame
    lags: pd.DataFrame
    outcomes: pd.DataFrame
    reputation: object
    data_end: pd.Timestamp


def analyze_security_frames(items: pd.DataFrame, trusted: pd.DataFrame) -> SecurityAnalysis | None:
    from trader.blacknews import security

    if items.empty:
        return None
    deduped = security.dedup_incidents(items)
    conf = security.confirmations(deduped, trusted)
    ends = [deduped["visible_at"].max()] + ([trusted["visible_at"].max()] if len(trusted) else [])
    data_end = max(ends)
    lags = security.incident_lags(deduped, conf, data_end, HORIZON)
    outs = black_outcomes(lags)
    rep = black_reputation(outs, deduped, tier=security.REPUTATION_TIER)
    return SecurityAnalysis(deduped, conf, lags, outs, rep, data_end)


def analyze_security(data: Path = DATA) -> SecurityAnalysis | None:
    from trader.blacknews.security import SecurityStore

    items = SecurityStore(Path(data) / "news_black").all()
    items = items[items["extracted_by"].notna()]
    return analyze_security_frames(items, load_trusted(data=data))
