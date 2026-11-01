"""Filtro do comitê, versão 2: aprende o AMBIENTE de serrote com informação além do preço da moeda.

Features novas (iguais para todas as moedas na mesma data, só com o passado):
  breadth200/50   fração das ~47 moedas acima da média de 200/50 dias (amplitude do mercado)
  breadth_chg20   variação da amplitude em 20 dias
  btc_committee   comitê do BTC (o índice do mercado); gold_committee: comitê do ouro (aversão a risco)
  btc_funding     funding médio diário do BTC (só a partir de 2023; antes fica vazio)
  mkt_volu_z      volume em US$ das 10 maiores, em desvios da média de 60 dias
Modelo: gradient boosting por histogramas (aceita vazio), walk-forward de 90 dias como a v1.
Compara: comitê puro × filtro v1 (só preço da moeda) × filtro v2 (preço + ambiente), 3 sementes.

  python -m trader.research.meta_label_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.meta_label import filtered_weights
from trader.research.rf_daily import FEATURES, FEE, features, period_stats
from trader.research.trend_daily import ENSEMBLE, portfolio_curve
from trader.research.universe_test import CURRENT, GOLD, SYMBOLS, committee_signal, download_daily

HORIZON = 10
MARKET = ["breadth200", "breadth50", "breadth_chg20", "btc_committee", "gold_committee", "btc_funding", "mkt_volu_z"]


def market_features(cache: Path) -> pd.DataFrame:
    closes, vols = {}, {}
    for s in SYMBOLS:
        d = download_daily(cache, s, log=lambda m: None)
        if len(d) >= 250:
            closes[s], vols[s] = d["close"], d["quote_volume"]
    c, v = pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index()
    f = pd.DataFrame(index=c.index)
    for n in (200, 50):
        above = (c > c.rolling(n).mean()).where(c.notna())
        f[f"breadth{n}"] = above.mean(axis=1)
    f["breadth_chg20"] = f["breadth200"].diff(20)
    f["btc_committee"] = sum((c["BTC"] > c["BTC"].rolling(n).mean()).astype(float) for n in ENSEMBLE) / len(ENSEMBLE)
    f["gold_committee"] = sum((c[GOLD] > c[GOLD].rolling(n).mean()).astype(float) for n in ENSEMBLE) / len(ENSEMBLE)
    top = v.rolling(90, min_periods=30).median().rank(axis=1, ascending=False) <= 10
    mv = np.log1p(v.where(top).sum(axis=1))
    f["mkt_volu_z"] = (mv - mv.rolling(60).mean()) / mv.rolling(60).std()
    try:
        from trader.daytrade.data import read_funding
        fund = read_funding("data", "BTC/USDT")
        daily = fund.set_index("ts")["rate"].resample("1D").mean()
        f["btc_funding"] = daily.reindex(f.index).ffill(limit=3)
    except Exception:
        f["btc_funding"] = np.nan
    f["btc_funding"] = f["btc_funding"].fillna(0.0)  # antes de 2023 não há série: 0 = funding neutro
    return f


def panel(cache: Path, mkt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts, closes = [], {}
    for s in CURRENT + [GOLD]:
        d = download_daily(cache, s, log=lambda m: None)
        closes[s] = d["close"]
        f = features(d.rename(columns={"quote_volume": "volume"})).drop(columns="target")
        f["committee"] = sum((d["close"] > d["close"].rolling(n).mean()).astype(float) for n in ENSEMBLE) / len(ENSEMBLE)
        f["fwd"] = d["close"].shift(-HORIZON) / d["close"] - 1
        parts.append(f.join(mkt).assign(symbol=s))
    p = pd.concat(parts).reset_index().rename(columns={"index": "ts"})
    return p, pd.DataFrame(closes).sort_index()


def walk_forward(p: pd.DataFrame, feats: list[str], kind: str, step: int = 90, seed: int = 0) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    base_feats = [f for f in feats if f not in MARKET]
    p = p.dropna(subset=base_feats + ["committee"])
    p = p[p["committee"] > 0].sort_values("ts")
    t0 = p["ts"].min() + pd.Timedelta(days=365)
    out = []
    for lo in pd.date_range(t0, p["ts"].max(), freq=f"{step}D"):
        hi = lo + pd.Timedelta(days=step)
        train = p[(p["ts"] < lo - pd.Timedelta(days=HORIZON + 1)) & p["fwd"].notna()]
        test = p[(p["ts"] >= lo) & (p["ts"] < hi)]
        if len(train) < 500 or test.empty:
            continue
        y = (train["fwd"] > 0).astype(int)
        if kind == "rf":
            model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, n_jobs=2, random_state=seed)
            model.fit(train[feats].fillna(0.0), y)
            prob = model.predict_proba(test[feats].fillna(0.0))[:, 1]
        else:
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=100, l2_regularization=1.0, random_state=seed)
            model.fit(train[feats], y)
            prob = model.predict_proba(test[feats])[:, 1]
        out.append(test[["ts", "symbol", "fwd"]].assign(prob=prob))
    return pd.concat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cache = Path(args.data) / "research" / "universe"
    mkt = market_features(cache)
    p, closes = panel(cache, mkt)
    base = committee_signal(closes)
    base = base.div(closes.notna().sum(axis=1).clip(lower=1), axis=0)
    periods = [("seleção 2019-2023", "2019-01-01", "2024-01-01"), ("holdout 2024-2026", "2024-01-01", None)]
    curves = {"comitê puro (5+ouro)": portfolio_curve(closes, base, FEE)}
    aucs = []
    from sklearn.metrics import roc_auc_score
    variants = {"v1 preço (RF)": (FEATURES + ["committee"], "rf"),
                "v2 preço+ambiente (HGB)": (FEATURES + ["committee"] + MARKET, "hgb"),
                "v2 preço+ambiente (RF)": (FEATURES + ["committee"] + MARKET, "rf")}
    for vname, (feats, kind) in variants.items():
        for seed in range(args.seeds):
            probs = walk_forward(p, feats, kind, seed=seed)
            for pname, lo, hi in periods:
                q = probs[(probs["ts"] >= lo) & ((probs["ts"] < hi) if hi else True)].dropna(subset=["fwd"])
                aucs.append({"variante": vname, "semente": seed, "período": pname,
                             "auc": roc_auc_score(q["fwd"] > 0, q["prob"]), "n": len(q)})
            curves[f"{vname} s{seed}"] = portfolio_curve(closes, filtered_weights(base, probs, "gradual 0.4-0.6"), FEE)
    pd.set_option("display.width", 220)
    print("\nAUC (0,5 = chute):")
    print(pd.DataFrame(aucs).pivot_table(index="variante", columns="período", values="auc").round(3).to_string())
    rows = []
    for pname, lo, hi in periods:
        for label, curve in curves.items():
            s = period_stats(curve, lo, hi)
            rows.append({"período": pname, "regra": label, "anual": s["cagr"], "maior queda": s["max_dd"],
                         "calmar": s["calmar"], "pior mês": s["worst_month"]})
    df = pd.DataFrame(rows)
    for pname, g in df.groupby("período", sort=False):
        print(f"\n== {pname} ==")
        print(g.drop(columns="período").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(Path(args.data) / "research" / "meta_label_v2.csv", index=False)


if __name__ == "__main__":
    main()
