"""Carry do funding (cash-and-carry): comprado à vista + vendido no perpétuo, recebe o funding.

O preço não importa (as duas pernas se anulam); o retorno é o funding pago a cada 8 h pelos
alavancados comprados, menos custos. Fonte: histórico de funding da Binance (arquivos mensais).

Modelo (diário):
  - capital por moeda = N (à vista) + m·N de margem na venda (m = 0,5 → a venda fica 2x; um salto
    de +50% num dia antes de repor margem liquidaria a perna vendida: contamos esses dias);
  - retorno diário sobre o capital = soma do funding do dia × N / ((1+m)·N) − custos;
  - custos: 0,1% à vista + 0,05% no perpétuo por perna → 0,3% do notional por entrada+saída,
    cobrados quando a posição liga/desliga; sem funding negativo não há custo além disso;
  - basis (diferença à vista × perpétuo) ignorada: no diário é pequena e se anula na saída.
Variantes:
  sempre         sempre posicionado, pesos iguais
  média7>0       liga quando o funding médio dos últimos 7 dias é positivo, desliga quando não
  média7>hurdle  liga só quando o funding médio de 7 dias anualizado passa de 8%
  top2           as 2 moedas com maior funding médio de 7 dias (rotação semanal)
Compara com o comitê (5 criptos + ouro) e mostra a carteira 50/50.

  python -m trader.research.carry_test
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
from trader.research.universe_test import CURRENT, GOLD, committee_signal, download_daily, fixed_universe, portfolio

BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP"]
MARGIN = 0.5
COST_SWITCH = 0.003  # entrada + saída, fração do notional
HURDLE = 0.08  # ao ano


def download_funding(cache: Path, coin: str, start: str = "2020-01", log=print) -> pd.Series:
    path = cache / f"{coin}.parquet"
    if path.exists():
        return pd.read_parquet(path)["rate"]
    frames = []
    for m in pd.period_range(start, pd.Timestamp.now().to_period("M") - 1, freq="M"):
        url = f"{BASE}/{coin}USDT/{coin}USDT-fundingRate-{m}.zip"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
        except urllib.error.HTTPError:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        frames.append(df)
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames)
    ts = pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True)
    s = pd.Series(df["last_funding_rate"].astype(float).to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated()]
    cache.mkdir(parents=True, exist_ok=True)
    s.rename("rate").to_frame().to_parquet(path)
    log(f"  {coin}: {len(s)} pagamentos de funding, {s.index[0]:%Y-%m} a {s.index[-1]:%Y-%m}")
    return s


def daily_funding(rates: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({c: r.resample("1D").sum(min_count=1) for c, r in rates.items()}).sort_index()


def carry_curve(fund: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """Retorno diário sobre o capital total da estratégia, pesos iguais entre as moedas ligadas."""
    on = mask.astype(float).where(fund.notna(), 0.0)
    w = on.div(on.sum(axis=1).clip(lower=1), axis=0)
    held = w.shift(1).fillna(0.0)
    gross = (held * fund.fillna(0.0)).sum(axis=1) / (1 + MARGIN)
    switch = held.diff().abs().sum(axis=1).fillna(0.0) * COST_SWITCH / (1 + MARGIN)
    return (1 + gross - switch).cumprod()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    root = Path(args.data) / "research"
    print("baixando histórico de funding...", flush=True)
    rates = {c: download_funding(root / "funding", c) for c in COINS}
    fund = daily_funding(rates)
    ann = fund.mean() * 365
    print("\nfunding médio anualizado por moeda (todo o período):", ann.round(3).to_dict())
    print("por ano (média das 5):")
    print((fund.mean(axis=1).groupby(fund.index.year).mean() * 365).round(3).to_string())
    avg7 = fund.rolling(7, min_periods=3).mean()
    always = pd.DataFrame(True, index=fund.index, columns=fund.columns)
    pos7 = avg7 > 0
    hurdle = avg7 * 365 > HURDLE
    rank = avg7.rank(axis=1, ascending=False)
    top2 = (rank <= 2).astype(float)
    top2[fund.index.dayofweek != 0] = np.nan  # decide na segunda-feira, segura a semana
    top2 = top2.ffill().fillna(0.0).astype(bool)
    variants = {"sempre": always, "média7>0": pos7, "média7>hurdle": hurdle, "top2": top2}
    curves = {f"carry {k}": carry_curve(fund, v) for k, v in variants.items()}

    cache = root / "universe"
    closes = pd.DataFrame({s: download_daily(cache, s, log=lambda m: None)["close"] for s in CURRENT + [GOLD]}).sort_index()
    com = portfolio(closes, committee_signal(closes), fixed_universe(closes, CURRENT + [GOLD]))
    com = com.reindex(fund.index).ffill()
    curves["comitê 5+ouro"] = com / com.dropna().iloc[0]
    best = curves["carry média7>0"]
    r_c, r_k = com.pct_change(), best.pct_change()
    mix = (1 + 0.5 * r_c.fillna(0) + 0.5 * r_k.fillna(0)).cumprod()
    curves["50% comitê + 50% carry(média7>0)"] = mix
    rows = []
    for pname, lo, hi in (("2020-2023", "2020-01-01", "2024-01-01"), ("2024-2026", "2024-01-01", None)):
        for name, c in curves.items():
            seg = c[lo:hi].dropna()
            if len(seg) < 100:
                continue
            s = period_stats(c, seg.index[0].strftime("%Y-%m-%d"), hi)
            rows.append({"período": pname, "estratégia": name, "anual": s["cagr"], "maior queda": s["max_dd"],
                         "calmar": s["calmar"], "pior mês": s["worst_month"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for pname, g in df.groupby("período", sort=False):
        print(f"\n== {pname} ==")
        print(g.drop(columns="período").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    both = pd.concat([r_c, r_k], axis=1, keys=["comitê", "carry"]).dropna()
    print(f"\ncorrelação diária comitê × carry: {both.corr().iloc[0, 1]:.2f}")
    print("retorno do carry por ano (média7>0):")
    print(best.resample("YE").last().pct_change().dropna().round(3).to_string())
    spikes = (closes[CURRENT].pct_change() > 0.40).sum()
    print(f"\ndias com alta > 40% (risco de liquidação da perna vendida com margem {MARGIN:.0%}): {spikes.to_dict()}")
    df.to_csv(root / "carry_test.csv", index=False)


if __name__ == "__main__":
    main()
