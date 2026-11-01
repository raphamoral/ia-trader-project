"""Correções pré-definidas para os erros achados na autópsia do comitê (serrotes e devolução).

Poucas variantes, decididas ANTES de olhar o holdout, para não pescar a que deu certo:
  base            comitê 50/100/150/200, vol-alvo 30%
  confirma3       só muda o peso se o novo sinal persistir 3 dias (menos serrote)
  histerese2%     entra quando o preço > média×1,02 e só sai quando < média×0,98
  saída_rápida    entra pelas 4 médias; sai se o preço cair abaixo da média de 30 dias (menos devolução)
  lateral_fora    fica em caixa quando o mercado está de lado (razão de eficiência de 60 dias < 0,1)
Medidas: retorno, queda, nota, nº de operações e fração de serrotes. Seleção 2019-2023, holdout 2024-2026.

  python -m trader.research.committee_fixes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.committee_autopsy import round_trips
from trader.research.trend_daily import DAYS_PER_YEAR, ENSEMBLE
from trader.research.universe_test import CURRENT, GOLD, download_daily, fixed_universe, portfolio


def vol_scale(closes: pd.DataFrame, target: float = 0.30) -> pd.DataFrame:
    vol = closes.pct_change().rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
    return (target / vol).clip(upper=1.0)


def raw_committee(closes: pd.DataFrame) -> pd.DataFrame:
    return sum((closes > closes.rolling(n).mean()).astype(float) for n in ENSEMBLE) / len(ENSEMBLE)


def sig_base(closes):
    return raw_committee(closes)


def sig_confirm(closes, days=3):
    raw = raw_committee(closes)
    out = raw.copy()
    stable = raw.rolling(days).apply(lambda x: float(x.min() == x.max()), raw=True).fillna(0.0).astype(bool)
    out = raw.where(stable)
    return out.ffill().fillna(0.0)


def sig_hysteresis(closes, band=0.02):
    votes = []
    for n in ENSEMBLE:
        sma = closes.rolling(n).mean()
        state = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        state[closes > sma * (1 + band)] = 1.0
        state[closes < sma * (1 - band)] = 0.0
        votes.append(state.ffill().fillna(0.0))
    return sum(votes) / len(votes)


def sig_fast_exit(closes, exit_n=30):
    raw = raw_committee(closes)
    below = closes < closes.rolling(exit_n).mean()
    return raw.where(~below, 0.0)


def sig_sideways_out(closes, n=60, thr=0.1):
    raw = raw_committee(closes)
    er = (closes - closes.shift(n)).abs() / closes.diff().abs().rolling(n).sum()
    return raw.where(er >= thr, 0.0)


VARIANTS = {"base": sig_base, "confirma3": sig_confirm, "histerese2%": sig_hysteresis,
            "saída_rápida": sig_fast_exit, "lateral_fora": sig_sideways_out}


def stats_of(curve: pd.Series, lo, hi) -> dict:
    from trader.research.trend_daily import stats
    seg = curve[lo:hi]
    s = stats(seg / seg.iloc[0])
    return {"anual": s["cagr"], "maior queda": s["max_dd"], "calmar": s["calmar"], "pior mês": s["worst_month"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    cache = Path(args.data) / "research" / "universe"
    syms = CURRENT + [GOLD]
    closes = pd.DataFrame({s: download_daily(cache, s, log=lambda m: None)["close"] for s in syms}).sort_index()
    mask = fixed_universe(closes, syms)
    vs = vol_scale(closes)
    rows = []
    for name, fn in VARIANTS.items():
        signal = (fn(closes) * vs).where(closes.notna(), 0.0).fillna(0.0)
        curve = portfolio(closes, signal, mask)
        w = (signal * mask).div(mask.sum(axis=1).clip(lower=1), axis=0).where(closes.notna(), 0.0)
        rts = pd.concat([round_trips(closes[s].dropna().loc["2019":], w[s].reindex(closes[s].dropna().index).fillna(0.0).loc["2019":]) for s in syms])
        for pname, lo, hi in (("seleção 2019-2023", "2019-01-01", "2024-01-01"), ("holdout 2024-2026", "2024-01-01", None)):
            r = rts[(pd.to_datetime(rts["start"]) >= lo) & ((pd.to_datetime(rts["start"]) < hi) if hi else True)]
            rows.append({"período": pname, "variante": name, **stats_of(curve, lo, hi), "operações": len(r),
                         "serrotes": (r["kind"] == "serrote").mean() if len(r) else np.nan,
                         "perda serrotes": r.loc[r["kind"] == "serrote", "pnl"].sum()})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for pname, g in df.groupby("período", sort=False):
        print(f"\n== {pname} ==")
        print(g.drop(columns="período").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(Path(args.data) / "research" / "committee_fixes.csv", index=False)


if __name__ == "__main__":
    main()
