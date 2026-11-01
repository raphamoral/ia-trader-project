"""Filtro do comitê (meta-labeling): o Random Forest decide QUANTO confiar no sinal do comitê.

O comitê de tendência continua decidindo a direção (comprado ou caixa). O modelo
só olha os dias em que o comitê está comprado e estima a chance de a posição
render nos próximos `horizon` dias; a posição do comitê é reduzida quando a
chance é baixa. Pergunta: isso melhora o comitê fora da amostra, depois das taxas?

Disciplina: features até o fechamento; walk-forward com retreino a cada `step`
dias, treinando só com rótulos cujo horizonte terminou antes do corte; parâmetros
fixos; 3 sementes; seleção 2019–2023 e holdout 2024 em diante.

  python -m trader.research.meta_label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.rf_daily import FEATURES, FEE, committee_weights, features, load, period_stats
from trader.research.trend_daily import ENSEMBLE, portfolio_curve
from trader.storage.candles import CandleStore

META_FEATURES = FEATURES + ["committee"]


def meta_panel(data: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    parts = []
    for s, d in data.items():
        f = features(d).drop(columns="target")
        c = d["close"]
        f["committee"] = sum((c > c.rolling(n).mean()).astype(float) for n in ENSEMBLE) / len(ENSEMBLE)
        f["fwd"] = c.shift(-horizon) / c - 1  # rótulo: a posição rendeu nos próximos dias?
        parts.append(f.assign(symbol=s))
    return pd.concat(parts).reset_index().rename(columns={"index": "ts"})


def walk_forward_meta(p: pd.DataFrame, horizon: int, step: int = 90, min_train_days: int = 365, seed: int = 0,
                      n_jobs: int = 2) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    p = p.dropna(subset=META_FEATURES)
    p = p[p["committee"] > 0].sort_values("ts")  # só os dias em que o comitê está comprado
    t0 = p["ts"].min() + pd.Timedelta(days=min_train_days)
    out = []
    for lo in pd.date_range(t0, p["ts"].max(), freq=f"{step}D"):
        hi = lo + pd.Timedelta(days=step)
        # o rótulo de um dia só é conhecido `horizon` dias depois: corta o treino antes disso
        train = p[(p["ts"] < lo - pd.Timedelta(days=horizon + 1)) & p["fwd"].notna()]
        test = p[(p["ts"] >= lo) & (p["ts"] < hi)]
        if len(train) < 500 or test.empty:
            continue
        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, n_jobs=n_jobs,
                                       random_state=seed)
        model.fit(train[META_FEATURES], (train["fwd"] > 0).astype(int))
        out.append(test[["ts", "symbol", "fwd"]].assign(prob=model.predict_proba(test[META_FEATURES])[:, 1]))
    return pd.concat(out)


def filtered_weights(base: pd.DataFrame, probs: pd.DataFrame, mode: str) -> pd.DataFrame:
    pr = probs.pivot(index="ts", columns="symbol", values="prob").reindex(index=base.index, columns=base.columns)
    if mode == "corta < 0.5":
        scale = (pr >= 0.5).astype(float)
    elif mode == "gradual 0.4-0.6":
        scale = ((pr - 0.4) / 0.2).clip(0, 1)
    else:
        raise ValueError(mode)
    # sem previsão (antes do primeiro modelo): mantém o comitê puro
    return base * scale.where(pr.notna(), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    data = load(CandleStore(Path(args.data)))
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
    base = committee_weights(closes)
    p = meta_panel(data, args.horizon)
    periods = [("seleção 2019-2023", "2019-01-01", "2024-01-01"), ("holdout 2024-2026", "2024-01-01", None)]

    curves = {"comitê puro": portfolio_curve(closes, base, FEE)}
    for seed in range(args.seeds):
        probs = walk_forward_meta(p, args.horizon, seed=seed)
        for name, lo, hi in periods:
            q = probs[(probs["ts"] >= lo) & ((probs["ts"] < hi) if hi else True)].dropna(subset=["fwd"])
            y = q["fwd"] > 0
            from sklearn.metrics import roc_auc_score

            print(f"semente {seed} {name}: acerto {((q['prob'] > 0.5) == y).mean():.3f}"
                  f" auc {roc_auc_score(y, q['prob']):.3f} base {y.mean():.3f} n {len(q)}")
        for mode in ("corta < 0.5", "gradual 0.4-0.6"):
            curves[f"filtro s{seed} {mode}"] = portfolio_curve(closes, filtered_weights(base, probs, mode), FEE)

    rows = []
    for name, lo, hi in periods:
        for label, curve in curves.items():
            s = period_stats(curve, lo, hi)
            rows.append({"período": name, "regra": label, "anual": s["cagr"], "maior queda": s["max_dd"],
                         "calmar": s["calmar"], "sharpe": s["sharpe"], "pior mês": s["worst_month"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for name, g in df.groupby("período", sort=False):
        print(f"\n== {name} ==")
        print(g.drop(columns="período").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = Path(args.data) / "research" / "meta_label.csv"
    df.to_csv(out, index=False)
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
