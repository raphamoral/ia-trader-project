"""As saídas novas (trava de lucro em degraus e "deixa correr") melhoram agentes de day trade?

Teste pareado: os mesmos agentes, as mesmas entradas e os mesmos dados; muda só a
saída. Os agentes são os que mais viveram nas evoluções de day trade já feitas
(registros em data/evolution/*day*). Os valores das saídas são fixos de antemão
(sem busca), para não escolher o que deu certo no próprio teste.

  python -m trader.research.exit_test
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trader.daytrade.backtest import run_daytrade_backtest
from trader.daytrade.genome import DayGenome
from trader.storage.candles import CandleStore

SYMBOLS = ["SOL/USDT", "ETH/USDT", "XRP/USDT"]
VARIANTS = {
    "original": {},
    "trava 1R / 50%": {"lock_at": 1.0, "lock_keep": 0.5},
    "deixa correr p80 / 2 ATR": {"runner_vol": 0.8, "runner_atr": 2.0},
    "trava + deixa correr": {"lock_at": 1.0, "lock_keep": 0.5, "runner_vol": 0.8, "runner_atr": 2.0},
}
PERIODS = [("2025-04-01", "2025-10-01"), ("2026-03-01", "2026-09-28")]


def veterans(folder: Path, n: int) -> list[DayGenome]:
    """Agentes de day trade em 5m que viveram mais janelas (sem repetidos)."""
    rows = []
    for f in sorted(folder.glob("*day*.sqlite")):
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        try:
            for born, died, g in con.execute("SELECT born_at, died_at, genome FROM agents"):
                age = (died if died is not None else born + 1000) - born
                rows.append((age, g))
        finally:
            con.close()
    rows.sort(key=lambda r: -r[0])
    seen, out = set(), []
    for _, g in rows:
        try:
            genome = DayGenome.from_dict(json.loads(g))
        except (ValueError, KeyError):
            continue
        # sem saídas novas e sem features que o teste não carrega (notícias e mercado)
        if (genome.timeframe != "5m" or genome.news_gate is not None or genome.news_follow is not None
                or genome.crowd_block is not None or genome.strategy in ("news_event", "xs_momentum", "funding_reversal")):
            continue
        key = json.dumps(genome.to_dict(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(genome)
        if len(out) >= n:
            break
    return out


_CANDLES: dict = {}


def _load(data: str, symbol: str, lo: str, hi: str) -> pd.DataFrame:
    key = (symbol, lo, hi)
    if key not in _CANDLES:
        start = pd.Timestamp(lo, tz="UTC") - pd.Timedelta(days=5)  # aquecimento dos indicadores
        _CANDLES[key] = CandleStore(Path(data)).read("binanceusdm", symbol, "5m", start=start,
                                                     end=pd.Timestamp(hi, tz="UTC"))
    return _CANDLES[key]


def _job(args) -> list[dict]:
    data, gi, genome_d, symbol, lo, hi = args
    candles = _load(data, symbol, lo, hi)
    base = DayGenome.from_dict(genome_d)
    out = []
    for name, extra in VARIANTS.items():
        g = replace(base, **extra).normalized()
        r = run_daytrade_backtest(g, candles, symbol, "5m", pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"),
                                  cash=1000.0)
        out.append({"agent": gi, "symbol": symbol, "period": f"{lo[:7]}..{hi[:7]}", "variant": name,
                    "ret": r.total_return, "dd": r.max_drawdown, "trades": r.n_trades})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--agents", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    agents = veterans(Path(args.data) / "evolution", args.agents)
    print(f"{len(agents)} agentes veteranos, {len(SYMBOLS)} moedas, {len(PERIODS)} períodos, {len(VARIANTS)} saídas")
    # sem saídas novas no original: zera as que porventura existam
    agents = [replace(g, lock_at=None, runner_vol=None) for g in agents]
    jobs = [(args.data, i, g.to_dict(), s, lo, hi) for i, g in enumerate(agents) for s in SYMBOLS for lo, hi in PERIODS]
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, part in enumerate(pool.map(_job, jobs, chunksize=4)):
            rows += part
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.data) / "research" / "exit_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    base = df[df.variant == "original"].set_index(["agent", "symbol", "period"])
    print("\nPor variante (cada linha = agente × moeda × período, pareado com o original):")
    for name in VARIANTS:
        v = df[df.variant == name].set_index(["agent", "symbol", "period"])
        diff = (v["ret"] - base["ret"]).dropna()
        t = diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else 0.0
        print(f"  {name:26s} retorno mediano {v['ret'].median():+.2%}  médio {v['ret'].mean():+.2%}"
              f"  queda mediana {v['dd'].median():.2%}  lucrativos {(v['ret'] > 0).mean():.0%}"
              f"  | vs original: melhora em {(diff > 0).mean():.0%}, diferença média {diff.mean():+.2%} (t {t:.2f})")
    print("\nPor período (diferença média para o original):")
    for (period, name), g in df.groupby(["period", "variant"]):
        b = base.xs(period, level="period")["ret"]
        v = g.set_index(["agent", "symbol"])["ret"]
        print(f"  {period} {name:26s} {(v - b).mean():+.2%}  lucrativos {(v > 0).mean():.0%}")
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
