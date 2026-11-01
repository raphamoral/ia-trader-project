"""Sinais de mercado com respaldo na literatura de cripto, como provedores de features causais.

- Momentum entre moedas (cross-sectional): em cada fechamento, o percentil do retorno
  de L candles do ativo dentro de uma cesta. Colunas mkt_xs_rank_<L>.
- Aglomeração em derivativos: funding vigente (já liquidado) em bps e choque de open
  interest (z-score da variação de 1h do log do OI). Colunas mkt_funding_bps, mkt_oi_z.

Contrato igual ao dos provedores de notícia: fn(symbol, timeframe, index) -> DataFrame
com o mesmo índice (instantes de FECHAMENTO), e a linha t só usa dados conhecidos em <= t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trader.daytrade import data
from trader.domain import parse_timeframe
from trader.storage.candles import CandleStore

BASKET = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "DOGE/USDT",
          "AVAX/USDT", "LINK/USDT", "LTC/USDT")
XS_LOOKBACKS = (12, 48, 288)  # 1h, 4h, 1 dia em candles de 5m
STALE_BARS = 2  # ativo sem candle recente sai do ranking naquele instante


def _closes_on(index: pd.DatetimeIndex, candles: pd.DataFrame, timeframe: str) -> pd.Series:
    """Fechamento de cada ativo alinhado aos instantes do índice (só o já fechado: asof para trás)."""
    if candles.empty:
        return pd.Series(np.nan, index=index)
    s = pd.Series(candles["close"].to_numpy(float), index=pd.DatetimeIndex(candles["ts"] + parse_timeframe(timeframe)))
    s = s[~s.index.duplicated()].sort_index()
    return s.reindex(index.union(s.index)).ffill(limit=STALE_BARS).reindex(index)


def xs_ranks(target: str, closes: dict[str, pd.Series], lookbacks=XS_LOOKBACKS) -> pd.DataFrame:
    """Percentil (0..1) do retorno de L candles do `target` na cesta, em cada instante."""
    frame = pd.DataFrame(closes)
    out = {}
    for lb in lookbacks:
        rets = frame / frame.shift(lb) - 1
        ranks = rets.rank(axis=1, pct=True)
        n = rets.notna().sum(axis=1)
        col = ranks[target].where(n >= 3)  # ranking com menos de 3 ativos não diz nada
        # normaliza para 0..1 (o pior = 0, o melhor = 1)
        out[f"mkt_xs_rank_{lb}"] = ((col * n - 1) / (n - 1)).where(n >= 3)
    return pd.DataFrame(out, index=frame.index)


def xs_provider(root: str | Path = "data", basket=BASKET, lookbacks=XS_LOOKBACKS):
    store = CandleStore(root)

    def provider(symbol: str, timeframe: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        if len(index) == 0:
            return pd.DataFrame(index=index, columns=[f"mkt_xs_rank_{lb}" for lb in lookbacks], dtype=float)
        tf = parse_timeframe(timeframe)
        lo = index.min() - tf * (max(lookbacks) + STALE_BARS + 2)
        # grade estendida para trás: o retorno de L candles precisa do histórico ANTES do trecho pedido
        grid = pd.date_range(lo, index.max(), freq=tf).union(index)
        assets = list(dict.fromkeys([symbol, *basket]))
        closes = {s: _closes_on(grid, store.read(data.EXCHANGE, s, timeframe, start=lo - tf, end=index.max()), timeframe)
                  for s in assets}
        return xs_ranks(symbol, closes, lookbacks).reindex(index)

    provider.__name__ = "xs_provider"
    return provider


def deriv_path(root: str | Path, symbol: str) -> Path:
    return Path(root) / "deriv" / data.EXCHANGE / f"{symbol.replace('/', '')}.parquet"


def crowding_features(index: pd.DatetimeIndex, funding: pd.DataFrame, metrics: pd.DataFrame | None) -> pd.DataFrame:
    """Funding liquidado mais recente (bps) e choque de OI, ambos conhecidos em <= t."""
    out = pd.DataFrame(index=index)
    if funding is not None and len(funding):
        f = pd.Series(funding["rate"].to_numpy(float) * 1e4, index=pd.DatetimeIndex(funding["ts"])).sort_index()
        f = f[~f.index.duplicated()]
        pos = f.index.searchsorted(index, side="right") - 1  # liquidado em <= t
        vals = np.where(pos >= 0, f.to_numpy()[np.maximum(pos, 0)], np.nan)
        out["mkt_funding_bps"] = vals
    else:
        out["mkt_funding_bps"] = np.nan
    out["mkt_oi_z"] = np.nan
    if metrics is not None and len(metrics):
        m = metrics.set_index(pd.DatetimeIndex(metrics["visible_at"]))["oi_usd"].sort_index()
        m = m[~m.index.duplicated()].where(m > 0)
        chg = np.log(m).diff(12)  # variação de 1h (métricas de 5 min)
        z = (chg - chg.shift(1).rolling(288, min_periods=100).mean()) / chg.shift(1).rolling(288, min_periods=100).std()
        pos = z.index.searchsorted(index, side="right") - 1
        fresh = (pos >= 0) & ((index - z.index[np.maximum(pos, 0)]) <= pd.Timedelta(minutes=30))
        out["mkt_oi_z"] = np.where(fresh, z.to_numpy()[np.maximum(pos, 0)], np.nan)
    return out


def crowding_provider(root: str | Path = "data"):
    def provider(symbol: str, timeframe: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        funding = data.read_funding(root, symbol)
        path = deriv_path(root, symbol)
        metrics = pd.read_parquet(path) if path.exists() else None
        return crowding_features(index, funding, metrics)

    provider.__name__ = "crowding_provider"
    return provider


def fetch_oi_history(root: str | Path, symbol: str, since: str, until: str, log=print) -> int:
    """Métricas públicas de 5 min (data.binance.vision) → data/deriv/<exchange>/<SYMBOL>.parquet.
    Usa o cliente do agente black (respeita limites); visible_at = create_time + 5 min."""
    from trader.blacknews.alt import fetch_deriv_day
    from trader.blacknews.sources import PoliteClient

    client = PoliteClient()
    path = deriv_path(root, symbol)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ts", "visible_at", "oi_usd"])
    done = set(pd.to_datetime(old["ts"], utc=True).dt.normalize()) if len(old) else set()
    rows = []
    for day in pd.date_range(since, until, freq="D", tz="UTC"):
        if day in done:
            continue
        try:
            raw = fetch_deriv_day(client, day, symbol.replace("/", ""))
        except Exception as exc:
            log(f"{day:%Y-%m-%d}: falhou ({type(exc).__name__})")
            continue
        oi = raw[raw["metric"] == "oi_usd"] if len(raw) else raw
        if len(oi):
            rows.append(pd.DataFrame({"ts": oi["ts"], "visible_at": oi["visible_at"], "oi_usd": oi["value"]}))
    if rows:
        new = pd.concat([old, *rows], ignore_index=True).drop_duplicates("ts").sort_values("ts")
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(path, index=False)
        return len(new) - len(old)
    return 0
