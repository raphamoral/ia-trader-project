"""Dá para saber ANTES qual listagem nova vai subir? Teste em todas as listagens USDT da Binance.

Para cada moeda, no dia 90 após a listagem (decisão), sinais só com dados até ali:
  mom_1_3        retorno do dia 30 ao dia 90
  dd_from_high   queda desde a máxima pós-listagem
  volu_growth    volume mediano (US$) dos últimos 30 dias ÷ dos primeiros 30 dias
  volu_level     log do volume mediano dos últimos 30 dias (interesse)
  tvl_growth_60  crescimento do TVL (uso real, DeFiLlama) em 60 dias; vazio se não há TVL
  tvl_level      log do TVL
  btc_regime     BTC acima da média de 200 dias no dia da decisão
Resultados a partir da decisão (sem teto de ganho):
  fwd_365        comprar e segurar 12 meses (deslistou antes: último preço)
  trend_exit     segura enquanto fecha acima da média de 50 dias; sai na primeira quebra (após 10 dias) ou em 365 dias
  max_dd         pior queda no caminho (segurando)
Análise: taxas-base por ano de listagem; quintis de cada sinal DENTRO do ano (tira o efeito do regime);
correlação de ordem (Spearman) por ano e consistência do sinal; cesta walk-forward: sinais escolhidos em
2019–2022, carteira de top-K entre as listagens de 2023–2026, com saída por tendência.

  python -m trader.research.listings_study
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/research")
DECISION_DAY = 90
HORIZON = 365
SIGNALS = ["mom_1_3", "dd_from_high", "volu_growth", "volu_level", "tvl_growth_60", "tvl_level"]


def load_all(cache: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(cache.glob("*.parquet")):
        d = pd.read_parquet(f)
        if len(d) >= DECISION_DAY + 30:
            out[f.stem] = d
    return out


def load_tvl() -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for name in ("protocols", "chains"):
        p = ROOT / "tvl" / f"{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for sym, g in df.groupby("symbol"):
            s = g.set_index("ts")["tvl"].sort_index()
            s = s[~s.index.duplicated()].resample("1D").last().ffill()
            if sym not in series or s.index[0] < series[sym].index[0]:
                series[sym] = s
    return series


def study_rows(data: dict[str, pd.DataFrame], tvl: dict[str, pd.Series], btc: pd.Series) -> pd.DataFrame:
    btc_regime = btc > btc.rolling(200).mean()
    rows = []
    for sym, d in data.items():
        c, v = d["close"], d["quote_volume"]
        listed = c.index[0]
        if listed <= pd.Timestamp("2017-09-01", tz="UTC"):
            continue  # já existia no início do histórico: não é "listagem"
        if len(c) <= DECISION_DAY:
            continue
        t = c.index[DECISION_DAY]
        if t + pd.Timedelta(days=120) > c.index[-1] and c.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=200):
            continue  # recente demais para ter resultado
        past = c.loc[:t]
        row = {"symbol": sym, "listed": listed, "decision": t, "year": listed.year,
               "mom_1_3": c.iloc[DECISION_DAY] / c.iloc[30] - 1,
               "dd_from_high": 1 - c.iloc[DECISION_DAY] / past.max(),
               "volu_growth": np.log(v.iloc[60:DECISION_DAY + 1].median() / max(v.iloc[:30].median(), 1.0)),
               "volu_level": np.log(max(v.iloc[60:DECISION_DAY + 1].median(), 1.0)),
               "btc_regime": bool(btc_regime.reindex([t], method="ffill").iloc[0]) if t >= btc.index[0] else np.nan}
        s = tvl.get(sym)
        if s is not None and t in s.index and (t - pd.Timedelta(days=60)) in s.index and s.loc[t] > 0:
            base = s.loc[t - pd.Timedelta(days=60)]
            row["tvl_growth_60"] = np.log(s.loc[t] / base) if base > 0 else np.nan
            row["tvl_level"] = np.log(s.loc[t])
        else:
            row["tvl_growth_60"] = np.nan
            row["tvl_level"] = np.nan
        fut = c.iloc[DECISION_DAY:DECISION_DAY + HORIZON + 1]
        entry = fut.iloc[0]
        row["fwd_365"] = fut.iloc[-1] / entry - 1
        row["max_dd"] = float((1 - fut / fut.cummax()).max())
        row["delisted_within"] = bool(len(fut) < HORIZON + 1 and c.index[-1] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45))
        sma50 = c.rolling(50).mean()
        ex = None
        for i in range(DECISION_DAY + 10, min(DECISION_DAY + HORIZON, len(c) - 1) + 1):
            if c.iloc[i] < sma50.iloc[i]:
                ex = i
                break
        ex = ex if ex is not None else min(DECISION_DAY + HORIZON, len(c) - 1)
        row["trend_exit"] = c.iloc[ex] / entry - 1
        row["trend_days"] = ex - DECISION_DAY
        rows.append(row)
    return pd.DataFrame(rows)


def quintile_table(df: pd.DataFrame, signal: str, outcome: str) -> pd.DataFrame:
    q = df.dropna(subset=[signal, outcome]).copy()
    q["q"] = q.groupby("year")[signal].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x) >= 10 else np.nan)
    g = q.dropna(subset=["q"]).groupby("q")[outcome]
    return pd.DataFrame({"mediana": g.median(), "média": g.mean(), "% > 0": g.apply(lambda x: (x > 0).mean()),
                         "% > +100%": g.apply(lambda x: (x > 1).mean()), "n": g.size()})


def spearman_by_year(df: pd.DataFrame, signal: str, outcome: str) -> pd.Series:
    out = {}
    for y, g in df.dropna(subset=[signal, outcome]).groupby("year"):
        if len(g) >= 12:
            out[y] = g[signal].rank().corr(g[outcome].rank())
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-year", type=int, default=2019)
    args = ap.parse_args()
    data = load_all(ROOT / "universe_all")
    btc = data["BTC"]["close"]
    tvl = load_tvl()
    df = study_rows(data, tvl, btc)
    df = df[df["year"] >= args.min_year]
    pd.set_option("display.width", 220)
    print(f"{len(df)} listagens com resultado; com TVL: {df['tvl_growth_60'].notna().sum()}")
    print("\n== Taxas-base por ano de listagem (comprar no dia 90 e segurar 12 meses) ==")
    base = df.groupby("year")["fwd_365"].agg(n="size", mediana="median", média="mean",
                                             positivas=lambda x: (x > 0).mean(), dobrou=lambda x: (x > 1).mean(),
                                             perdeu_metade=lambda x: (x < -0.5).mean())
    base["deslistadas"] = df.groupby("year")["delisted_within"].mean()
    base["saída por tendência (mediana)"] = df.groupby("year")["trend_exit"].median()
    print(base.to_string(float_format=lambda x: f"{x:.2f}"))
    for outcome in ("fwd_365", "trend_exit"):
        print(f"\n== Sinais → {outcome} (quintis dentro do ano; 0 = menor, 4 = maior) ==")
        for sig in SIGNALS:
            t = quintile_table(df, sig, outcome)
            if t.empty:
                continue
            sp = spearman_by_year(df, sig, outcome)
            print(f"\n{sig}: Spearman por ano {sp.round(2).to_dict()} | anos com sinal > 0: {(sp > 0).sum()}/{len(sp)}")
            print(t.to_string(float_format=lambda x: f"{x:+.2f}"))
    # regime do BTC na decisão
    print("\n== Regime do BTC no dia da decisão → resultado ==")
    print(df.groupby("btc_regime")[["fwd_365", "trend_exit"]].agg(["median", "mean", "size"]).round(2).to_string())
    # cesta walk-forward: sinais com Spearman médio > 0 em 2019-2022 → top-K por ano em 2023+
    train = df[df["year"] <= 2022]
    chosen = []
    for sig in SIGNALS:
        sp = spearman_by_year(train, sig, "trend_exit")
        if len(sp) >= 3 and sp.mean() > 0.05 and (sp > 0).mean() >= 0.75:
            chosen.append((sig, 1.0))
        elif len(sp) >= 3 and sp.mean() < -0.05 and (sp < 0).mean() >= 0.75:
            chosen.append((sig, -1.0))
    print(f"\n== Cesta walk-forward: sinais consistentes em 2019-2022: {chosen} ==")
    test = df[df["year"] >= 2023].copy()
    if chosen and len(test):
        score = sum(sgn * test.groupby("year")[sig].rank(pct=True).fillna(0.5) for sig, sgn in chosen) / len(chosen)
        test["score"] = score
        for k in (5, 10):
            picks = test.sort_values("score", ascending=False).groupby("year").head(k)
            rest = test[~test.index.isin(picks.index)]
            print(f"top-{k} por ano (2023+): n={len(picks)} | saída por tendência: média {picks['trend_exit'].mean():+.0%}, mediana {picks['trend_exit'].median():+.0%}, "
                  f"dobrou {(picks['trend_exit'] > 1).mean():.0%} | resto: média {rest['trend_exit'].mean():+.0%}, mediana {rest['trend_exit'].median():+.0%}")
            print(f"   segurando 12 meses: picks média {picks['fwd_365'].mean():+.0%} mediana {picks['fwd_365'].median():+.0%} | resto média {rest['fwd_365'].mean():+.0%}")
    df.to_csv(ROOT / "listings_study.csv", index=False)
    print(f"\nsalvo em {ROOT / 'listings_study.csv'}")


if __name__ == "__main__":
    main()
