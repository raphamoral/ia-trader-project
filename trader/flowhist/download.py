"""Histórico de fluxo dos futuros da Binance (data.binance.vision), resumido por minuto.

Por dia e ativo baixa dois arquivos públicos:
  aggTrades  cada negócio agregado (preço, quantidade, quem agrediu)
  bookDepth  profundidade do livro a ±0,2%, 1%, ..., 5% do preço, a cada ~25 s
O arquivo é lido em memória, resumido por minuto e descartado: no disco fica só o
resumo (data/flow/<SÍMBOLO>/<AAAA-MM>.parquet), poucos MB por mês.

Colunas por minuto: open, high, low, close, volume, n_trades, buy_usd, sell_usd,
large_buy_usd, large_sell_usd (negócios >= `large_usd`), e do livro (média do
minuto): bid_02, ask_02, bid_1, ask_1, bid_5, ask_5 (em US$ de profundidade).

  python -m trader.flowhist.download --symbols SOLUSDT,XRPUSDT,ETHUSDT --since 2026-03-01 --until 2026-09-30
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trader.storage.atomic import replace_with_retry, unique_tmp

BASE = "https://data.binance.vision/data/futures/um/daily"
LARGE_USD = {"BTCUSDT": 250_000, "ETHUSDT": 100_000}
DEFAULT_LARGE = 50_000


class Throttle:
    """Limita a banda média (MB/s), dormindo entre downloads."""

    def __init__(self, mbps: float) -> None:
        self.mbps, self.t0, self.bytes = mbps, time.monotonic(), 0

    def add(self, n: int) -> None:
        self.bytes += n
        ahead = self.bytes / (self.mbps * 1e6) - (time.monotonic() - self.t0)
        if ahead > 0:
            time.sleep(ahead)


def fetch_zip_csv(url: str, throttle: Throttle | None, tries: int = 4) -> pd.DataFrame | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                raw = resp.read()
            if throttle:
                throttle.add(len(raw))
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                with z.open(z.namelist()[0]) as f:
                    head = f.read(64)
                with z.open(z.namelist()[0]) as f:
                    has_header = not head[:1].isdigit()
                    return pd.read_csv(f, header=0 if has_header else None)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (OSError, zipfile.BadZipFile):
            pass
        time.sleep(5 * (k + 1))
    return None


def trades_minutes(df: pd.DataFrame, large_usd: float) -> pd.DataFrame:
    if df.columns.dtype != object or "price" not in df.columns:
        df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
                      "is_buyer_maker"][: len(df.columns)]
    ts = pd.to_datetime(df["transact_time"], unit="ms", utc=True).dt.floor("1min")
    price, qty = df["price"].astype(float), df["quantity"].astype(float)
    usd = price * qty
    sell = df["is_buyer_maker"].astype(str).str.lower().eq("true")  # comprador passivo = venda agressora
    large = usd >= large_usd
    frame = pd.DataFrame({"ts": ts, "price": price, "qty": qty, "buy": usd.where(~sell, 0.0),
                          "sell": usd.where(sell, 0.0), "lbuy": usd.where(~sell & large, 0.0),
                          "lsell": usd.where(sell & large, 0.0)})
    g = frame.groupby("ts")
    out = pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(), "close": g["price"].last(),
        "volume": g["qty"].sum(), "n_trades": g.size(), "buy_usd": g["buy"].sum(), "sell_usd": g["sell"].sum(),
        "large_buy_usd": g["lbuy"].sum(), "large_sell_usd": g["lsell"].sum(),
    })
    return out.astype("float32")


def depth_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(ts=pd.to_datetime(df["timestamp"], utc=True).dt.floor("1min"),
                   pct=df["percentage"].astype(float).round(2))
    cols = {-0.2: "bid_02", 0.2: "ask_02", -1.0: "bid_1", 1.0: "ask_1", -5.0: "bid_5", 5.0: "ask_5"}
    df = df[df["pct"].isin(cols)]
    piv = df.pivot_table(index="ts", columns="pct", values="notional", aggfunc="mean")
    return piv.rename(columns=cols).astype("float32")


def day_minutes(symbol: str, day: pd.Timestamp, throttle: Throttle | None) -> pd.DataFrame | None:
    d = f"{day:%Y-%m-%d}"
    trades = fetch_zip_csv(f"{BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip", throttle)
    if trades is None or trades.empty:
        return None
    out = trades_minutes(trades, LARGE_USD.get(symbol, DEFAULT_LARGE))
    del trades
    depth = fetch_zip_csv(f"{BASE}/bookDepth/{symbol}/{symbol}-bookDepth-{d}.zip", throttle)
    if depth is not None and not depth.empty:
        out = out.join(depth_minutes(depth), how="left")
    return out


def month_path(root: Path, symbol: str, month: str) -> Path:
    return root / "flow" / symbol / f"{month}.parquet"


def read_flow(root: Path, symbol: str, since: str | None = None, until: str | None = None) -> pd.DataFrame:
    files = sorted((root / "flow" / symbol).glob("*.parquet"))
    if since:
        files = [f for f in files if f.stem >= since[:7]]
    if until:
        files = [f for f in files if f.stem <= until[:7]]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[since:until] if since or until else df


def save_day(root: Path, symbol: str, day_df: pd.DataFrame) -> None:
    month = f"{day_df.index[0]:%Y-%m}"
    path = month_path(root, symbol, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(path), day_df]) if path.exists() else day_df
    df = df[~df.index.duplicated(keep="last")].sort_index()
    tmp = unique_tmp(path)
    df.to_parquet(tmp)
    replace_with_retry(tmp, path)


def done_days(root: Path, symbol: str) -> set[str]:
    out = set()
    for f in (root / "flow" / symbol).glob("*.parquet"):
        idx = pd.read_parquet(f, columns=["close"]).index
        counts = pd.Series(1, index=idx).groupby(idx.strftime("%Y-%m-%d")).size()
        out |= set(counts[counts >= 1400].index)  # dia completo (~1440 minutos)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SOLUSDT,XRPUSDT,ETHUSDT")
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--until", default="2026-09-30")
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-mbps", type=float, default=4.0)
    args = ap.parse_args()
    root = Path(args.data)
    throttle = Throttle(args.max_mbps)
    days = pd.date_range(args.since, args.until, freq="D")
    for symbol in args.symbols.split(","):
        have = done_days(root, symbol)
        todo = [d for d in days if f"{d:%Y-%m-%d}" not in have]
        print(f"{symbol}: {len(todo)} dias a baixar ({len(have)} já completos)", flush=True)
        for k, day in enumerate(todo, 1):
            df = day_minutes(symbol, day, throttle)
            if df is None:
                print(f"  {symbol} {day:%Y-%m-%d}: sem arquivo", flush=True)
                continue
            save_day(root, symbol, df)
            mb = throttle.bytes / 1e6
            print(f"  {symbol} {day:%Y-%m-%d} ok ({k}/{len(todo)}, {mb:.0f} MB baixados no total)", flush=True)


if __name__ == "__main__":
    main()
