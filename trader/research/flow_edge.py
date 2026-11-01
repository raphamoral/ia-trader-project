"""Peça 3: livro de ofertas e fluxo dão vantagem ao day trade depois das taxas?

Dados: resumo por minuto dos futuros (trader.flowhist.download). Decisão a cada 5
minutos, com features só até o fechamento; entrada no fechamento do minuto
SEGUINTE (1 minuto de atraso); saída depois de H minutos.

Parte A (sem modelo): retorno seguinte médio, em pontos-base, por quintil de cada
feature. Se a diferença entre os quintis extremos não passa do custo, não há o que
operar com aquela feature sozinha.
Parte B (modelo): walk-forward mensal (treina com os meses anteriores), opera só
quando |previsão| > custo, uma operação por vez por ativo; resultado líquido.

Custos de ida e volta: taker 0,05% por lado = 10 bps; maker 0,02% = 4 bps (otimista:
nem toda ordem limitada executa).

  python -m trader.research.flow_edge
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trader.flowhist.download import read_flow

COSTS_BPS = {"taker": 10.0, "maker": 4.0}
HORIZONS = (5, 15, 60)


def _imb(a: pd.Series, b: pd.Series) -> pd.Series:
    s = a + b
    return ((a - b) / s).where(s > 0)


def features(m: pd.DataFrame) -> pd.DataFrame:
    """Features por minuto (causais). Janelas em minutos."""
    f = pd.DataFrame(index=m.index)
    c = m["close"]
    lr = np.log(c).diff()
    for w in (1, 5, 15, 60):
        b, s = m["buy_usd"].rolling(w).sum(), m["sell_usd"].rolling(w).sum()
        f[f"aggr_{w}"] = _imb(b, s)
    lb, ls = m["large_buy_usd"].rolling(15).sum(), m["large_sell_usd"].rolling(15).sum()
    tot = (m["buy_usd"] + m["sell_usd"]).rolling(15).sum()
    f["large_imb_15"] = ((lb - ls) / tot).where(tot > 0)
    for lv in ("02", "1", "5"):
        if f"bid_{lv}" in m:
            imb = _imb(m[f"bid_{lv}"], m[f"ask_{lv}"])
            f[f"book_{lv}"] = imb
            f[f"book_{lv}_15"] = imb.rolling(15).mean()
    for w in (5, 15, 60):
        f[f"ret_{w}"] = np.log(c / c.shift(w))
    f["vol_60"] = lr.rolling(60).std()
    v = (m["buy_usd"] + m["sell_usd"])
    f["volu_z"] = (np.log1p(v.rolling(15).sum()) - np.log1p(v.rolling(15).sum()).rolling(1440).mean()) / \
        np.log1p(v.rolling(15).sum()).rolling(1440).std()
    f["hour"] = f.index.hour
    return f


def targets(m: pd.DataFrame) -> pd.DataFrame:
    """Retorno (bps) entrando no fechamento do minuto seguinte e saindo H minutos depois."""
    c = m["close"]
    entry = c.shift(-1)
    return pd.DataFrame({f"fwd_{h}": (c.shift(-1 - h) / entry - 1) * 1e4 for h in HORIZONS}, index=m.index)


def decision_rows(m: pd.DataFrame) -> pd.DataFrame:
    df = features(m).join(targets(m))
    # completo: o índice tem buracos se faltou minuto; uma decisão a cada 5 min
    return df[df.index.minute % 5 == 4]


def quintiles(df: pd.DataFrame, feats: list[str], h: int) -> pd.DataFrame:
    rows = []
    for f in feats:
        q = df[[f, f"fwd_{h}"]].dropna()
        if len(q) < 500:
            continue
        b = pd.qcut(q[f].rank(method="first"), 5, labels=False)
        means = q.groupby(b)[f"fwd_{h}"].mean()
        rows.append({"feature": f, **{f"q{i + 1}": means.iloc[i] for i in range(5)},
                     "q5-q1": means.iloc[4] - means.iloc[0]})
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame, feats: list[str], h: int, cost: float, seed: int = 0) -> pd.DataFrame:
    """Operações (uma por vez) com o modelo do mês treinado só nos meses anteriores."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = df.dropna(subset=[f"fwd_{h}"])
    month_of = df.index.strftime("%Y-%m")
    months = sorted(set(month_of))
    trades = []
    for k in range(2, len(months)):
        test_m = months[k]
        start = pd.Timestamp(f"{test_m}-01", tz="UTC")
        # o rótulo dos últimos H minutos do treino ainda não "aconteceu" no início do teste
        train = df[df.index < start - pd.Timedelta(minutes=h + 1)]
        test = df[month_of == test_m]
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
        y = train[f"fwd_{h}"].clip(-300, 300)  # cauda não domina o treino
        model.fit(train[feats], y)
        pred = model.predict(test[feats])
        busy_until = None
        for ts, p, fwd in zip(test.index, pred, test[f"fwd_{h}"]):
            if busy_until is not None and ts < busy_until:
                continue
            if abs(p) > cost:
                side = 1 if p > 0 else -1
                trades.append({"ts": ts, "month": test_m, "side": side, "pred": p, "gross": side * fwd,
                               "net": side * fwd - cost})
                busy_until = ts + pd.Timedelta(minutes=h + 1)
    return pd.DataFrame(trades)


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    m = t.groupby("month")["net"].sum()
    se = t["net"].std() / np.sqrt(len(t)) if len(t) > 1 else np.nan
    return {"trades": len(t), "gross_bps": t["gross"].mean(), "net_bps": t["net"].mean(),
            "t_net": t["net"].mean() / se if se and se > 0 else 0.0, "hit": (t["net"] > 0).mean(),
            "months_pos": f"{(m > 0).sum()}/{len(m)}", "total_net_pct": t["net"].sum() / 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--symbols", default="SOLUSDT,XRPUSDT,ETHUSDT")
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--until", default="2026-09-30")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    root = Path(args.data)
    out_rows = []
    for sym in args.symbols.split(","):
        m = read_flow(root, sym, args.since, args.until)
        if m.empty:
            print(f"{sym}: sem dados")
            continue
        m = m.asfreq("1min")  # minutos faltando viram NaN (as janelas não pulam buracos)
        df = decision_rows(m)
        feats = [c for c in df.columns if not c.startswith("fwd_")]
        print(f"\n===== {sym}: {len(df)} decisões, {df.index.min():%Y-%m-%d} a {df.index.max():%Y-%m-%d}")
        print("Parte A: retorno seguinte médio (bps) por quintil, horizonte 15 min")
        print(quintiles(df, [f for f in feats if f != "hour"], 15).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
        for h in HORIZONS:
            for name, cost in COSTS_BPS.items():
                t = walk_forward(df, feats, h, cost)
                s = summarize(t)
                out_rows.append({"symbol": sym, "horizon": h, "cost": name, **s})
                print(f"Parte B {sym} H={h:>2}min custo {name:5s}: " + ", ".join(
                    f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()), flush=True)
    res = pd.DataFrame(out_rows)
    out = root / "research" / "flow_edge.csv"
    res.to_csv(out, index=False)
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
