"""Comitê alavancado: o que acontece com 1,5x, 2x, 3x e 5x em futuros perpétuos.

Modelo (diário, por fechamento):
  - pesos do comitê × alavancagem L (vol-alvo vira 30% × L);
  - taxa de 0,05% (taker de futuros) sobre o que muda de posição;
  - funding sobre o notional inteiro, em três cenários: 0, 8 e 15% ao ano
    (média histórica do funding de BTC/ETH fica perto de 8–11%/ano para quem está comprado);
  - ruína: se o patrimônio cai 90% ou mais num dia (na prática a corretora liquida antes),
    a carteira zera e para. Só fechamentos diários: a liquidação intradiária seria ainda pior.
Ressalva: PAXG tem perpétuo na Binance há pouco tempo; aqui o ouro é alavancado como os outros.

  python -m trader.research.leverage_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.trend_daily import stats
from trader.research.universe_test import CURRENT, GOLD, committee_signal, download_daily, fixed_universe

FEE = 0.0005
LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)
FUNDINGS = (0.0, 0.08, 0.15)


def leveraged_curve(closes, signal, mask, lev, funding) -> tuple[pd.Series, bool]:
    w = (signal * mask).div(mask.sum(axis=1).clip(lower=1), axis=0).where(closes.notna(), 0.0) * lev
    ret = closes.pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = held.sum(axis=1)
    r = (held * ret).sum(axis=1) - held.diff().abs().sum(axis=1).fillna(0.0) * FEE - gross * funding / 365
    r = r.clip(lower=-1.0)
    ruin = r <= -0.9
    if ruin.any():
        first = ruin.idxmax()
        r.loc[first:] = 0.0
        r.loc[first] = -1.0
        return (1 + r).cumprod(), True
    return (1 + r).cumprod(), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    cache = Path(args.data) / "research" / "universe"
    closes = pd.DataFrame({s: download_daily(cache, s, log=lambda m: None)["close"] for s in CURRENT + [GOLD]}).sort_index()
    signal = committee_signal(closes)
    cestas = {"5 criptos": CURRENT, "5 criptos + ouro": CURRENT + [GOLD]}
    rows = []
    for cname, syms in cestas.items():
        mask = fixed_universe(closes, syms)
        for lev in LEVERAGES:
            for fund in FUNDINGS:
                curve, ruined = leveraged_curve(closes, signal, mask, lev, fund)
                daily = curve.pct_change().dropna()
                for pname, lo, hi in (("2019-2023", "2019-01-01", "2024-01-01"), ("2024-2026", "2024-01-01", None)):
                    seg = curve[lo:hi]
                    s = stats(seg / seg.iloc[0]) if seg.iloc[0] > 0 else {"cagr": -1.0, "max_dd": 1.0, "calmar": 0.0, "worst_month": -1.0}
                    d = daily[lo:hi]
                    rows.append({"cesta": cname, "alav": lev, "funding": f"{fund:.0%}", "período": pname,
                                 "anual": s["cagr"], "maior queda": s["max_dd"], "calmar": s["calmar"],
                                 "pior mês": s["worst_month"], "pior dia": float(d.min()) if len(d) else np.nan,
                                 "melhor dia": float(d.max()) if len(d) else np.nan,
                                 "R$1000 vira": 1000 * (seg.iloc[-1] / seg.iloc[0]) if seg.iloc[0] > 0 else 0.0,
                                 "ruína": ruined})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for (cname, pname), g in df.groupby(["cesta", "período"], sort=False):
        print(f"\n== {cname} | {pname} ==")
        print(g.drop(columns=["cesta", "período"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = Path(args.data) / "research" / "leverage_test.csv"
    df.to_csv(out, index=False)
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
