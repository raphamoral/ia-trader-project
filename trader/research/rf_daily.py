"""Random Forest no diário (ideia do artigo da Coinmonks) × tendência "comitê".

Pergunta: um Random Forest com os indicadores clássicos (retornos, médias, RSI,
MACD, Bollinger, volume, volatilidade) prevê o dia seguinte bem o bastante para
render mais que o comitê de médias, depois das taxas?

Disciplina:
  - features só com dados até o fechamento do dia; alvo = retorno do dia seguinte;
  - walk-forward: retreina a cada `step` dias só com o passado (janela crescente),
    com um intervalo de 1 dia entre treino e previsão (o último alvo do treino
    termina antes do primeiro dia previsto);
  - parâmetros do modelo fixos de antemão (sem busca, para não escolher no holdout);
  - mesmas regras de execução e tamanho do comitê: decide no fechamento, executa
    no dia seguinte, taxa de 0,1% sobre o que muda, volatilidade-alvo de 30%,
    pesos iguais entre as moedas disponíveis;
  - seleção 2019–2023 e holdout 2024 em diante, relatados separados.

  python -m trader.research.rf_daily
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.research.trend_daily import DAYS_PER_YEAR, ensemble_weights, portfolio_curve, stats
from trader.storage.candles import CandleStore

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
VOL_TARGET = 0.30
FEE = 0.001
FEATURES = ["r1", "r5", "r20", "sma10", "sma20", "sma50", "sma100", "sma200", "rsi14", "macd_h", "bb_pct",
            "vol20", "vol_z"]


def load(store: CandleStore) -> dict[str, pd.DataFrame]:
    out = {}
    for s in SYMBOLS:
        c = store.read("binance", s, "1d")
        if not c.empty:
            out[s] = c.set_index("ts")[["open", "high", "low", "close", "volume"]]
    return out


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Indicadores do artigo, normalizados pelo preço para valer entre moedas. Causais (até o fechamento)."""
    c, v = d["close"], d["volume"]
    f = pd.DataFrame(index=d.index)
    for n in (1, 5, 20):
        f[f"r{n}"] = c.pct_change(n)
    for n in (10, 20, 50, 100, 200):
        f[f"sma{n}"] = c / c.rolling(n).mean() - 1
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pct"] = (c - (mid - 2 * sd)) / (4 * sd)
    f["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
    lv = np.log(v.replace(0, np.nan))
    f["vol_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    f["target"] = c.pct_change().shift(-1)  # retorno do dia seguinte (só para treinar)
    return f


def panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [features(d).assign(symbol=s) for s, d in data.items()]
    return pd.concat(parts).reset_index().rename(columns={"index": "ts"})


def walk_forward_probs(p: pd.DataFrame, start: pd.Timestamp, step: int = 90, min_train_days: int = 365,
                       seed: int = 0, n_jobs: int = 2) -> pd.DataFrame:
    """Probabilidade de alta no dia seguinte, por moeda e dia, só com modelos treinados no passado."""
    from sklearn.ensemble import RandomForestClassifier

    p = p.dropna(subset=FEATURES).sort_values("ts")
    days = pd.DatetimeIndex(sorted(p["ts"].unique()))
    out = []
    t0 = max(start, days[0] + pd.Timedelta(days=min_train_days))
    for lo in pd.date_range(t0, days[-1], freq=f"{step}D"):
        hi = lo + pd.Timedelta(days=step)
        # treino: dias cujo alvo (dia seguinte) já terminou antes de `lo`
        train = p[(p["ts"] < lo - pd.Timedelta(days=1)) & p["target"].notna()]
        test = p[(p["ts"] >= lo) & (p["ts"] < hi)]
        if len(train) < 500 or test.empty:
            continue
        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, n_jobs=n_jobs,
                                       random_state=seed)
        model.fit(train[FEATURES], (train["target"] > 0).astype(int))
        out.append(test[["ts", "symbol", "target"]].assign(prob=model.predict_proba(test[FEATURES])[:, 1]))
    return pd.concat(out)


def rf_weights(probs: pd.DataFrame, closes: pd.DataFrame, threshold: float, graded: bool) -> pd.DataFrame:
    """Comprado quando prob > threshold (ou proporcional ao excesso), escalado pela volatilidade como o comitê."""
    pr = probs.pivot(index="ts", columns="symbol", values="prob").reindex(closes.index)
    if graded:
        signal = ((pr - 0.5) / (threshold - 0.5)).clip(0, 1) if threshold > 0.5 else (pr > 0.5).astype(float)
    else:
        signal = (pr > threshold).astype(float)
    signal = signal.where(pr.notna(), 0.0)
    vol = closes.pct_change().rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
    signal = signal * (VOL_TARGET / vol).clip(upper=1.0)
    return (signal / closes.notna().sum(axis=1).clip(lower=1).values[:, None]).fillna(0.0)


def committee_weights(closes: pd.DataFrame) -> pd.DataFrame:
    """O comitê com pesos iguais entre as moedas que existem em cada data (como na DSL)."""
    w = ensemble_weights(closes, VOL_TARGET) * closes.shape[1]
    return (w / closes.notna().sum(axis=1).clip(lower=1).values[:, None]).fillna(0.0)


def period_stats(curve: pd.Series, lo: str, hi: str | None) -> dict[str, float]:
    seg = curve[lo:hi] if hi else curve[lo:]
    return stats(seg / seg.iloc[0])


def classification_report(probs: pd.DataFrame, lo: str, hi: str | None) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    q = probs[(probs["ts"] >= lo) & ((probs["ts"] < hi) if hi else True)].dropna(subset=["target"])
    y = (q["target"] > 0).astype(int)
    return {"accuracy": float(((q["prob"] > 0.5).astype(int) == y).mean()), "auc": float(roc_auc_score(y, q["prob"])),
            "base_up": float(y.mean()), "n": len(q)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--step", type=int, default=90)
    args = ap.parse_args()

    data = load(CandleStore(Path(args.data)))
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
    p = panel(data)
    periods = [("seleção 2019-2023", "2019-01-01", "2024-01-01"), ("holdout 2024-2026", "2024-01-01", None)]

    curves = {"comitê 50/100/150/200": portfolio_curve(closes, committee_weights(closes), FEE)}
    bh = (closes.pct_change().fillna(0.0).where(closes.notna(), 0.0)
          .div(closes.notna().sum(axis=1).clip(lower=1), axis=0).sum(axis=1))
    curves["comprar e segurar (pesos iguais)"] = (1 + bh).cumprod()
    for seed in range(args.seeds):
        probs = walk_forward_probs(p, pd.Timestamp("2018-06-01", tz="UTC"), step=args.step, seed=seed)
        for name, lo, hi in periods:
            print(f"semente {seed} {name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                        for k, v in classification_report(probs, lo, hi).items()))
        for thr, graded in ((0.5, False), (0.55, False), (0.6, True)):
            label = f"RF s{seed} " + (f"gradual até {thr}" if graded else f"prob > {thr}")
            curves[label] = portfolio_curve(closes, rf_weights(probs, closes, thr, graded), FEE)

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
    out = Path(args.data) / "research" / "rf_daily.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
