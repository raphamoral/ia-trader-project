"""API para a evolução: features black_ alinhadas a um índice de candles.

    from trader.blacknews.api import black_features
    fb = black_features("BTC/USDT", "15m", candles_ts_index, weighted=True)

- Mesmo índice de entrada, uma linha por instante de decisão (as_of = o instante do índice;
  passe o fechamento do candle, não a abertura, se a decisão é no fechamento).
- Causal: cada linha usa só itens com visible_at <= t e reputação/líderes com desfechos
  resolvidos antes do dia de t.
- weighted=False → black_* sem reputação (braço WB); weighted=True → black_rw_* (braço WB'),
  inclusive black_rw_lead_score / black_rw_lead_count (só líderes com boa nota, ecos fora).
- Fora do período coberto pelos dados black, as colunas valem 0.
- Monitor de segurança: weighted=False acrescenta black_sec_incident_count, black_sec_max_claimed_usd,
  black_sec_minutes_since_incident (7 dias se não houve) e black_sec_confirmed_flag;
  weighted=True acrescenta black_rw_sec_incident_score (soma dos pesos de reputação das fontes).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from trader.blacknews.analysis import DATA, Analysis, SecurityAnalysis, analyze, analyze_frames, analyze_security
from trader.blacknews.features import COLUMNS, build_features
from trader.blacknews.security import SEC_FEATURES, SEC_RW_FEATURES, security_features

SYMBOL_ASSET = {
    "BTC/USDT": "BTC", "BTCUSDT": "BTC", "ETH/USDT": "ETH", "SOL/USDT": "SOL", "XRP/USDT": "XRP",
    "BOVA11": "IBOV", "PETR4": "PETR4", "VALE3": "VALE3",
}


def columns(weighted: bool, security: bool = True) -> list[str]:
    base = [c for c in COLUMNS if c.startswith("black_rw_") == weighted]
    if security:
        base += SEC_RW_FEATURES if weighted else SEC_FEATURES
    return base


def _lookback(timeframe: str) -> pd.Timedelta:
    from trader.domain import parse_timeframe

    return max(parse_timeframe(timeframe), pd.Timedelta(hours=4))


def black_features_from(
    analysis: Analysis | None, asset: str, timeframe: str, index: pd.DatetimeIndex, weighted: bool = True,
    sec: SecurityAnalysis | None = None, security: bool = True,
) -> pd.DataFrame:
    cols = columns(weighted, security=False)
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if analysis is None or analysis.scored.empty:
        out = pd.DataFrame(0.0, index=index, columns=cols)
    else:
        feats = build_features(
            analysis.scored, idx, asset, analysis.lags, analysis.reputation,
            lookback=_lookback(timeframe), outcomes=analysis.outcomes,
        )
        out = feats[cols].astype(float)
        out.index = index
    if not security:
        return out
    sec_cols = SEC_RW_FEATURES if weighted else SEC_FEATURES
    if sec is None:
        from trader.blacknews.security import NO_INCIDENT_MINUTES

        extra = pd.DataFrame(0.0, index=index, columns=sec_cols)
        if "black_sec_minutes_since_incident" in extra:
            extra["black_sec_minutes_since_incident"] = NO_INCIDENT_MINUTES
    else:
        weights = sec.reputation.weights if weighted else None
        extra = security_features(sec.items, sec.confirmations, idx, asset, weights=weights)[sec_cols]
        extra.index = index
    return pd.concat([out, extra], axis=1)


@lru_cache(maxsize=1)
def _cached_analysis(data: str) -> Analysis | None:
    return analyze(data=Path(data))


@lru_cache(maxsize=1)
def _cached_security(data: str) -> SecurityAnalysis | None:
    return analyze_security(Path(data))


def black_features(
    symbol: str, timeframe: str, index: pd.DatetimeIndex, weighted: bool = True, data: Path | str = DATA
) -> pd.DataFrame:
    """Features black_ causais no `index` dado (mesmo índice na saída)."""
    asset = SYMBOL_ASSET.get(symbol, symbol)
    if asset in ("ETH", "SOL", "XRP"):
        sec_asset = "BTC"  # incidentes cripto valem para o mercado cripto inteiro
    else:
        sec_asset = asset
    base = black_features_from(_cached_analysis(str(data)), asset, timeframe, index, weighted, security=False)
    extra = black_features_from(None, sec_asset, timeframe, index, weighted, sec=_cached_security(str(data)))
    return pd.concat([base, extra[[c for c in extra if c not in base]]], axis=1)


__all__ = ["black_features", "black_features_from", "columns", "analyze_frames"]
