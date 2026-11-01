"""Comitê de tendência em universos maiores: mais moedas, moedas pouco correlacionadas ao BTC, e ouro.

Dados: candles diários do spot da Binance (arquivos mensais públicos de data.binance.vision),
INCLUINDO moedas já deslistadas (LUNA, FTT...), para não olhar só quem sobreviveu.

Universos, escolhidos todo início de mês só com o passado (point-in-time):
  atual5         BTC, ETH, SOL, BNB, XRP (a carteira em papel)
  topN           as N moedas com maior volume mediano em US$ nos 90 dias anteriores (>= 200 dias de histórico)
  baixa_corr10   entre as 30 maiores, as 10 com menor correlação diária ao BTC nos 90 dias anteriores
  atual5+ouro    as 5 atuais + PAXG (ouro tokenizado)
  top20+ouro
Mesmo comitê (50/100/150/200), mesma vol-alvo de 30%, pesos iguais, taxa de 0,1% sobre o que muda.
Moeda deslistada: posição zerada no último dia com preço (a queda até ali conta).

  python -m trader.research.universe_test
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.rf_daily import period_stats
from trader.research.trend_daily import DAYS_PER_YEAR, ENSEMBLE

BASE = "https://data.binance.vision/data/spot/monthly/klines"
CURRENT = ["BTC", "ETH", "SOL", "BNB", "XRP"]
GOLD = "PAXG"
# grandes de várias épocas + mortas/deslistadas (LUNA, FTT, EOS, ...) + ouro
SYMBOLS = CURRENT + [
    "ADA", "DOGE", "TRX", "DOT", "MATIC", "LTC", "LINK", "AVAX", "ATOM", "UNI", "XLM", "ETC", "BCH", "NEAR", "FIL",
    "ALGO", "VET", "ICP", "APT", "ARB", "OP", "INJ", "SUI", "AAVE", "MKR", "EOS", "XTZ", "THETA", "EGLD", "FTM",
    "SAND", "MANA", "AXS", "SHIB", "PEPE", "LUNA", "FTT", "NEO", "IOTA", "ZEC", "DASH", "WAVES", GOLD,
]
FEE = 0.001
VOL_TARGET = 0.30
START = "2018-01"


def download_daily(cache: Path, symbol: str, start: str = START, log=print) -> pd.DataFrame:
    """Candles diários (close, volume em US$) de um par USDT; cache em parquet por símbolo."""
    path = cache / f"{symbol}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames, misses = [], 0
    for m in pd.period_range(start, pd.Timestamp.now().to_period("M") - 1, freq="M"):
        url = f"{BASE}/{symbol}USDT/1d/{symbol}USDT-1d-{m}.zip"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
        except urllib.error.HTTPError:
            misses += 1
            if frames and misses >= 3:  # 3 meses seguidos sem arquivo depois de ter dados: deslistada
                break
            continue
        misses = 0
        with zipfile.ZipFile(io.BytesIO(raw)) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, usecols=[0, 4, 7], names=["ts", "close", "quote_volume"])
        df = df[pd.to_numeric(df["ts"], errors="coerce").notna()]
        frames.append(df)
    out = pd.concat(frames) if frames else pd.DataFrame(columns=["ts", "close", "quote_volume"])
    if not out.empty:
        ts = out["ts"].astype("int64")
        ts = np.where(ts > 1e14, ts // 1000, ts)  # arquivos novos vêm em microssegundos
        out["ts"] = pd.to_datetime(ts, unit="ms", utc=True).floor("D")
        out = out.drop_duplicates("ts").set_index("ts").sort_index().astype(float)
    cache.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    log(f"  {symbol}: {len(out)} dias" + (f" ({out.index[0]:%Y-%m} a {out.index[-1]:%Y-%m})" if len(out) else ""))
    return out


def load_panel(cache: Path, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, vols = {}, {}
    for s in SYMBOLS:
        d = download_daily(cache, s, log=log)
        if len(d) >= 250:
            closes[s], vols[s] = d["close"], d["quote_volume"]
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index()


def committee_signal(closes: pd.DataFrame) -> pd.DataFrame:
    sig = sum((closes > closes.rolling(n).mean()).astype(float) for n in ENSEMBLE) / len(ENSEMBLE)
    vol = closes.pct_change().rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
    return (sig * (VOL_TARGET / vol).clip(upper=1.0)).where(closes.notna(), 0.0).fillna(0.0)


def monthly_universe(closes: pd.DataFrame, vols: pd.DataFrame, rule: str, n: int) -> pd.DataFrame:
    """Máscara (data × moeda) de quem está no universo, decidida no início de cada mês só com o passado."""
    mask = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    med_vol = vols.rolling(90, min_periods=60).median()
    hist = closes.notna().rolling(200, min_periods=1).sum()
    rets = closes.pct_change()
    month_starts = closes.index[closes.index.to_series().dt.to_period("M").diff().ne(0).fillna(True).to_numpy()]
    chosen: list[str] = []
    for i, day in enumerate(month_starts):
        prev = closes.index[closes.index.get_loc(day) - 1] if closes.index.get_loc(day) > 0 else None
        if prev is not None:
            ok = (hist.loc[prev] >= 200) & med_vol.loc[prev].notna() & (med_vol.loc[prev] > 0)
            ok = ok.drop(GOLD, errors="ignore")  # ouro só entra pelas regras "+ouro"
            ranked = med_vol.loc[prev][ok[ok].index].sort_values(ascending=False)
            if rule == "top":
                chosen = list(ranked.index[:n])
            elif rule == "lowcorr":
                pool = list(ranked.index[:30])
                win = rets.loc[:prev].tail(90)
                corr = win[pool].corrwith(win["BTC"]).dropna().sort_values()
                chosen = list(corr.index[:n])
        end = month_starts[i + 1] if i + 1 < len(month_starts) else closes.index[-1] + pd.Timedelta(days=1)
        if chosen:
            mask.loc[(mask.index >= day) & (mask.index < end), chosen] = True
    return mask & closes.notna()


def fixed_universe(closes: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    cols = [c for c in names if c in mask.columns]
    mask[cols] = True
    return mask & closes.notna()


def portfolio(closes: pd.DataFrame, signal: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    w = (signal * mask).div(mask.sum(axis=1).clip(lower=1), axis=0)  # pesos iguais entre as moedas do universo
    w = w.where(closes.notna(), 0.0)  # deslistada: zera no último dia com preço
    ret = closes.pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    r = (held * ret).sum(axis=1) - held.diff().abs().sum(axis=1).fillna(0.0) * FEE
    return (1 + r).cumprod()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    cache = Path(args.data) / "research" / "universe"
    print("baixando candles diários (cache em data/research/universe)...", flush=True)
    closes, vols = load_panel(cache)
    print(f"\n{closes.shape[1]} moedas com >= 250 dias; deslistadas incluídas: "
          f"{[c for c in closes.columns if closes[c].last_valid_index() < closes.index[-1] - pd.Timedelta(days=30)]}")
    signal = committee_signal(closes)
    universes = {
        "atual5": fixed_universe(closes, CURRENT),
        "top10": monthly_universe(closes, vols, "top", 10),
        "top20": monthly_universe(closes, vols, "top", 20),
        "top30": monthly_universe(closes, vols, "top", 30),
        "baixa_corr10": monthly_universe(closes, vols, "lowcorr", 10),
        "atual5+ouro": fixed_universe(closes, CURRENT + [GOLD]),
        "top20+ouro": monthly_universe(closes, vols, "top", 20) | fixed_universe(closes, [GOLD]),
        "só ouro": fixed_universe(closes, [GOLD]),
    }
    rows = []
    periods = [("seleção 2019-2023", "2019-01-01", "2024-01-01"), ("holdout 2024-2026", "2024-01-01", None)]
    for name, mask in universes.items():
        curve = portfolio(closes, signal, mask)
        for pname, lo, hi in periods:
            s = period_stats(curve, lo, hi)
            rows.append({"período": pname, "universo": name, "anual": s["cagr"], "maior queda": s["max_dd"],
                         "calmar": s["calmar"], "sharpe": s["sharpe"], "pior mês": s["worst_month"],
                         "moedas (média)": float(mask.loc[lo:hi].sum(axis=1).mean())})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for pname, g in df.groupby("período", sort=False):
        print(f"\n== {pname} ==")
        print(g.drop(columns="período").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    r = closes.pct_change()
    print("\ncorrelação diária com o BTC desde 2021 (ouro e algumas moedas):")
    sub = r.loc["2021":]
    print(sub.corrwith(sub["BTC"]).drop("BTC").sort_values().round(2).head(8).to_dict())
    out = Path(args.data) / "research" / "universe_test.csv"
    df.to_csv(out, index=False)
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
