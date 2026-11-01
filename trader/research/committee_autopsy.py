"""Autópsia do comitê: onde e por que ele perde (5 criptos + ouro, 2019–2026).

1) Operações por moeda (entrou quando o peso > 0, saiu quando voltou a 0): resultado de cada
   uma, duração, e classificação:
     serrote   durou pouco (<= 20 dias) e perdeu: entrou e o mercado virou logo
     devolução ganhou no caminho mas devolveu mais da metade do lucro máximo antes de sair (atraso na saída)
     crash     caiu mais de 15% dentro da operação
     ganho     fechou no lucro sem devolver mais da metade
2) Episódios de queda da carteira (> 8%): datas, duração, o que o BTC fez e o que a carteira fez.
3) Custos: taxas acumuladas.

  python -m trader.research.committee_autopsy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.universe_test import CURRENT, GOLD, committee_signal, download_daily, fixed_universe

FEE = 0.001


def round_trips(close: pd.Series, w: pd.Series) -> pd.DataFrame:
    """Operações de uma moeda: trechos em que o peso (decidido ontem) ficou > 0."""
    held = w.shift(1).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    on = held > 0
    starts = on & ~on.shift(1, fill_value=False)
    rows = []
    for s in on.index[starts]:
        after = on.loc[s:]
        e = after.index[~after][0] if (~after).any() else after.index[-1]
        seg = (held.loc[s:e] * ret.loc[s:e])
        curve = (1 + seg).cumprod()
        peak = curve.cummax()
        pnl = curve.iloc[-1] - 1
        max_gain = peak.max() - 1
        worst_dd = float((1 - curve / peak).max())
        days = len(seg)
        if pnl < 0 and days <= 20:
            kind = "serrote"
        elif worst_dd > 0.15:
            kind = "crash"
        elif max_gain > 0.02 and pnl < max_gain * 0.5:
            kind = "devolução"
        elif pnl >= 0:
            kind = "ganho"
        else:
            kind = "perda lenta"
        rows.append({"start": s.date(), "end": e.date(), "days": days, "pnl": pnl, "max_gain": max_gain,
                     "worst_dd": worst_dd, "kind": kind})
    return pd.DataFrame(rows)


def drawdown_episodes(curve: pd.Series, btc: pd.Series, min_dd: float = 0.08) -> pd.DataFrame:
    peak = curve.cummax()
    dd = 1 - curve / peak
    rows, in_ep, start = [], False, None
    for t, v in dd.items():
        if not in_ep and v > 0:
            in_ep, start = True, t
        elif in_ep and v == 0:
            seg = dd.loc[start:t]
            if seg.max() >= min_dd:
                trough = seg.idxmax()
                b = btc.loc[start:t]
                rows.append({"início": start.date(), "fundo": trough.date(), "recuperou": t.date(),
                             "dias": len(seg), "queda": seg.max(),
                             "BTC no mesmo trecho": b.iloc[-1] / b.iloc[0] - 1,
                             "BTC pico→fundo": btc.loc[trough] / btc.loc[start] - 1})
            in_ep = False
    if in_ep:
        seg = dd.loc[start:]
        if seg.max() >= min_dd:
            rows.append({"início": start.date(), "fundo": seg.idxmax().date(), "recuperou": "ainda não",
                         "dias": len(seg), "queda": seg.max(), "BTC no mesmo trecho": np.nan, "BTC pico→fundo": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--since", default="2019-01-01")
    args = ap.parse_args()
    cache = Path(args.data) / "research" / "universe"
    syms = CURRENT + [GOLD]
    closes = pd.DataFrame({s: download_daily(cache, s, log=lambda m: None)["close"] for s in syms}).sort_index()
    signal = committee_signal(closes)
    mask = fixed_universe(closes, syms)
    w = (signal * mask).div(mask.sum(axis=1).clip(lower=1), axis=0).where(closes.notna(), 0.0)
    w, closes = w.loc[args.since:], closes.loc[args.since:]
    pd.set_option("display.width", 220)

    print("== 1) Operações por moeda ==")
    all_rt = []
    for s in syms:
        rt = round_trips(closes[s].dropna(), w[s].reindex(closes[s].dropna().index).fillna(0.0))
        rt["coin"] = s.split("/")[0]
        all_rt.append(rt)
    rt = pd.concat(all_rt, ignore_index=True)
    g = rt.groupby("kind").agg(n=("pnl", "size"), soma=("pnl", "sum"), media=("pnl", "mean"), dias=("days", "median"))
    print(g.sort_values("soma").to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\n(soma = retornos somados das operações daquele tipo, em fração da parte da carteira na moeda)")
    print("\nPor moeda:")
    print(rt.groupby("coin").agg(ops=("pnl", "size"), ganhos=("pnl", lambda x: (x > 0).sum()), soma=("pnl", "sum"),
                                 serrotes=("kind", lambda k: (k == "serrote").sum()),
                                 devolucoes=("kind", lambda k: (k == "devolução").sum()))
          .to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nMaiores perdas individuais:")
    print(rt.nsmallest(8, "pnl")[["coin", "start", "end", "days", "pnl", "max_gain", "kind"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nMaiores devoluções (ganho máximo → resultado final):")
    dv = rt[rt.kind == "devolução"].assign(devolvido=lambda d: d.max_gain - d.pnl).nlargest(8, "devolvido")
    print(dv[["coin", "start", "end", "days", "max_gain", "pnl", "devolvido"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n== 2) Episódios de queda da carteira (>= 8%) ==")
    ret = closes.pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    fees = held.diff().abs().sum(axis=1).fillna(0.0) * FEE
    r = (held * ret).sum(axis=1) - fees
    curve = (1 + r).cumprod()
    print(drawdown_episodes(curve, closes["BTC"]).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n== 3) Custos e exposição ==")
    years = len(curve) / 365
    print(f"taxas: {fees.sum():.3f} do capital em {years:.1f} anos (~{fees.sum() / years:.2%} ao ano)")
    print(f"exposição média: {held.sum(axis=1).mean():.0%} do capital; dias 100% em caixa: {(held.sum(axis=1) == 0).mean():.0%}")
    gross = (held * ret).sum(axis=1)
    inv = held.sum(axis=1) > 0
    print(f"retorno médio diário quando investido: {gross[inv].mean():+.4%}; dias investidos: {inv.mean():.0%}")
    up = ret['BTC'] > 0
    print(f"captura: nos dias em que o BTC sobe a carteira faz {gross[up].mean():+.3%}/dia; quando cai, {gross[~up].mean():+.3%}/dia")


if __name__ == "__main__":
    main()
