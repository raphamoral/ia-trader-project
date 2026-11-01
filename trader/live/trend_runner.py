"""Tendência diária ao vivo (só papel): carteira de 5 criptos, comitê de tendência + volatilidade alvo.

Uma vez por dia, depois do fechamento diário (00:00 UTC), calcula os pesos com os
fechamentos até ontem e rebalanceia a carteira simulada, pagando taxa sobre o que
muda. NENHUMA ordem real é enviada.

Modo com filtro (`filtered=True`, meta-labeling): um Random Forest treinado só com
o passado estima, para cada moeda em que o comitê está comprado, a chance de a
posição render nos próximos 10 dias; a posição é escalada de 0 (chance <= 0,4)
a 1 (>= 0,6). Retreina uma vez por dia processado (segundos).
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from trader.ingest import binance
from trader.research.trend_daily import ensemble_weights
from trader.storage.candles import CandleStore

REAL_ORDERS_ENABLED = False
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
GOLD = "PAXG/USDT"  # ouro tokenizado: outra classe de ativo, correlação ~0,14 com o BTC (HISTORICO item 43)
SCHEMA = """CREATE TABLE IF NOT EXISTS days (
    date TEXT PRIMARY KEY, equity_brl REAL NOT NULL, ret REAL NOT NULL, fees_brl REAL NOT NULL, weights TEXT NOT NULL
)"""


class TrendPaper:
    def __init__(self, data: Path, db: Path, capital_brl: float = 1000.0, vol_target: float = 0.3, fee: float = 0.001,
                 filtered: bool = False, symbols: list[str] | None = None):
        assert not REAL_ORDERS_ENABLED
        self.filtered = filtered
        self.symbols = list(symbols or SYMBOLS)
        self.store = CandleStore(data)
        self.db = sqlite3.connect(str(db))
        self.db.execute(SCHEMA)
        self.capital, self.vol_target, self.fee = capital_brl, vol_target, fee

    def closes(self) -> pd.DataFrame:
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=300)
        for s in self.symbols:
            binance.update(self.store, s, "1d", since)
        df = pd.DataFrame({s: self.store.read("binance", s, "1d", start=since).set_index("ts")["close"] for s in self.symbols})
        return df.dropna()

    def last(self) -> tuple[str | None, float, dict]:
        row = self.db.execute("SELECT date, equity_brl, weights FROM days ORDER BY date DESC LIMIT 1").fetchone()
        return (row[0], row[1], json.loads(row[2])) if row else (None, self.capital, {s: 0.0 for s in self.symbols})

    def filter_scale(self, ts: pd.Timestamp) -> dict[str, float]:
        """Fração da posição do comitê mantida em cada moeda no dia `ts` (modelo só com dados anteriores)."""
        from trader.research.meta_label import META_FEATURES, meta_panel
        from trader.research.rf_daily import load
        from sklearn.ensemble import RandomForestClassifier

        horizon = 10
        p = meta_panel(load(self.store), horizon).dropna(subset=META_FEATURES)
        p = p[p["committee"] > 0]
        train = p[(p["ts"] < ts - pd.Timedelta(days=horizon + 1)) & p["fwd"].notna()]
        today = p[p["ts"] == ts]
        if len(train) < 500 or today.empty:
            return {}
        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, n_jobs=1, random_state=0)
        model.fit(train[META_FEATURES], (train["fwd"] > 0).astype(int))
        prob = model.predict_proba(today[META_FEATURES])[:, 1]
        return {s: float(min(max((pr - 0.4) / 0.2, 0.0), 1.0)) for s, pr in zip(today["symbol"], prob)}

    def target(self, weights: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float]:
        target = {s: float(weights.at[ts, s]) for s in self.symbols}
        if self.filtered:
            scale = self.filter_scale(ts)
            target = {s: w * scale.get(s, 1.0) for s, w in target.items()}
        return target

    def step(self) -> bool:
        """Processa os dias fechados que faltam. Retorna True se registrou algum."""
        closes = self.closes()
        weights = ensemble_weights(closes, self.vol_target)
        rets = closes.pct_change()
        last_date, equity, held = self.last()
        did = False
        if last_date is None:
            # primeira vez: começa HOJE com R$1.000 (não inventa um histórico simulando o passado)
            ts = closes.index[-1]
            target = self.target(weights, ts)
            fees = sum(target.values()) * self.fee * equity
            equity -= fees
            self.db.execute("INSERT INTO days VALUES (?, ?, ?, ?, ?)",
                            (ts.strftime("%Y-%m-%d"), equity, 0.0, fees, json.dumps(target)))
            self.db.commit()
            return True
        for ts in closes.index[1:]:
            day = ts.strftime("%Y-%m-%d")
            if last_date and day <= last_date:
                continue
            # retorno do dia com os pesos decididos ontem; depois rebalanceia para os pesos de hoje
            day_ret = sum(held.get(s, 0.0) * rets.at[ts, s] for s in self.symbols)
            equity *= 1 + day_ret
            target = self.target(weights, ts)
            fees = sum(abs(target[s] - held.get(s, 0.0)) for s in self.symbols) * self.fee * equity
            equity -= fees
            self.db.execute("INSERT INTO days VALUES (?, ?, ?, ?, ?)", (day, equity, day_ret, fees, json.dumps(target)))
            self.db.commit()
            held, last_date, did = target, day, True
        return did


def main(filtered: bool = False, gold: bool = False) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    name = "tendencia_ouro" if gold else "tendencia_filtro" if filtered else "tendencia"
    symbols = SYMBOLS + [GOLD] if gold else SYMBOLS
    paper = TrendPaper(Path("data"), Path(f"data/live/{name}.sqlite"), filtered=filtered, symbols=symbols)
    print("tendência diária em papel: 5 criptos" + (" + ouro (PAXG)" if gold else "")
          + ", comitê 50/100/150/200d, vol alvo 30%, R$1.000" + (" + filtro (meta-labeling)" if filtered else ""), flush=True)
    while True:
        try:
            if paper.step():
                date, equity, w = paper.last()
                print(f"{date}: patrimônio R${equity:,.2f} | pesos {', '.join(f'{k.split('/')[0]} {v:.0%}' for k, v in w.items())}",
                      flush=True)
        except Exception as exc:
            print(f"erro ({type(exc).__name__}: {exc}); nova tentativa em 10 min", flush=True)
        time.sleep(600)


if __name__ == "__main__":
    main()
