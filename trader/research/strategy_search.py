"""Autoaprimoramento seguro: o Qwen propõe estratégias na linguagem restrita (dsl.py), o
sistema testa, e o resultado volta para ele na rodada seguinte.

Seleção só em SELECT (2018-06 a 2023); o período reservado (2024 em diante) é olhado
apenas para as finalistas, no fim. Nada de código livre: só JSON validado e compilado.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.evolution.creator import _extract_json, _post_json
from trader.evolution.priors import standing_lessons
from trader.research.dsl import InvalidStrategy, compile_strategy
from trader.research.trend_daily import portfolio_curve
from trader.storage.candles import CandleStore

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
SELECT = (pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
BASE = {"name": "comitê de tendência", "entry": {"trend_frac": [50, 100, 150, 200]}, "size": {"vol_target": 0.3}}

PROMPT = """You design long-only crypto portfolio strategies (BTC, ETH, SOL, BNB, XRP; daily candles; fee 0.1%).
Write strategies ONLY in this JSON language (no code):
  strategy = {{"name": str, "entry": <cond>, "size": {{"vol_target": 0.05-1.0}}}}
  <cond> = {{"all": [<cond>...]}} | {{"any": [<cond>...]}} | {{"not": <cond>}} | <piece>
  pieces (each worth 0..1; all=min, any=max, not=1-x):
    {{"trend_frac": [n,...]}}       fraction of moving averages (n days) with price above (trend committee)
    {{"above_sma": n}}  {{"sma_cross": [fast, slow]}}  {{"momentum": n, "gt": x}}
    {{"vol_pct": n, "lt": p}}   (20d volatility below percentile p of last n days)
    {{"rsi": n, "lt" or "gt": 0-100}}   {{"off_high": n, "lt": fraction}}   {{"trend_vote": [n,...], "ge": k}}
  limits: n in 2..400, at most 12 pieces, depth at most 4.

Goal: STABLE growth: high return / max drawdown (Calmar) and a mild worst month. Results are measured on 2018-2023.

Baseline and best strategies so far (results: cagr, max_dd, calmar, worst_month):
{best}

Recent attempts that failed or were worse (avoid repeating them):
{worst}

Recent INVALID attempts and why (fix these mistakes; only use the pieces listed above):
{invalid}

Trading book rules (playbook):
{lessons}

Propose {n} NEW, DIFFERENT strategies that could beat the best Calmar. Small, sensible changes are fine.
Answer JSON only: {{"candidates": [{{"name": "...", "why": "...", "strategy": {{...}}}}]}}"""


def load_closes(data: Path) -> pd.DataFrame:
    store = CandleStore(data)
    return pd.DataFrame({s: store.read("binance", s, "1d").set_index("ts")["close"] for s in SYMBOLS}).sort_index()


def metrics(curve: pd.Series) -> dict[str, float]:
    years = len(curve) / 365
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    dd = float((1 - curve / curve.cummax()).max())
    m = curve.resample("ME").last().pct_change().dropna()
    t = m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 1 and m.std() > 0 else 0.0
    return {"cagr": float(cagr), "max_dd": dd, "calmar": float(cagr / dd) if dd else 0.0,
            "worst_month": float(m.min()) if len(m) else 0.0, "t": float(t)}


def evaluate(spec: dict, closes: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, float]:
    weights = compile_strategy(spec, closes)
    curve = portfolio_curve(closes, weights)
    sel = curve[(curve.index >= period[0]) & (curve.index < period[1])]
    return metrics(sel / sel.iloc[0])


SCHEMA = """CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY, round INTEGER, name TEXT, why TEXT, spec TEXT, valid INTEGER, error TEXT,
    cagr REAL, max_dd REAL, calmar REAL, worst_month REAL, t REAL)"""


def run(rounds: int, n: int, data: Path, db: Path, url: str, model: str, post=_post_json, log=print) -> None:
    closes = load_closes(data)
    con = sqlite3.connect(str(db))
    con.execute(SCHEMA)
    if not con.execute("SELECT 1 FROM trials WHERE round = 0").fetchone():
        m = evaluate(BASE, closes, SELECT)
        con.execute("INSERT INTO trials (round, name, why, spec, valid, cagr, max_dd, calmar, worst_month, t)"
                    " VALUES (0, ?, 'base', ?, 1, ?, ?, ?, ?, ?)",
                    (BASE["name"], json.dumps(BASE), m["cagr"], m["max_dd"], m["calmar"], m["worst_month"], m["t"]))
        con.commit()
    lessons = "\n".join(f"- {x}" for x in standing_lessons(data / "books" / "playbook.json", 8))
    start_round = (con.execute("SELECT MAX(round) FROM trials").fetchone()[0] or 0) + 1
    for r in range(start_round, start_round + rounds):
        fmt = lambda rows: "\n".join(f"- {name}: {spec} -> cagr {c:.1%}, max_dd {d:.1%}, calmar {k:.2f}, worst_month {w:.1%}"
                                     for name, spec, c, d, k, w in rows) or "- (none)"
        best = con.execute("SELECT name, spec, cagr, max_dd, calmar, worst_month FROM trials WHERE valid = 1"
                           " ORDER BY calmar DESC LIMIT 5").fetchall()
        worst = con.execute("SELECT name, spec, cagr, max_dd, calmar, worst_month FROM trials WHERE valid = 1"
                            " ORDER BY round DESC, calmar ASC LIMIT 4").fetchall()
        invalid = con.execute("SELECT name, error FROM trials WHERE valid = 0 ORDER BY id DESC LIMIT 6").fetchall()
        prompt = PROMPT.format(best=fmt(best), worst=fmt(worst), lessons=lessons, n=n,
                               invalid="\n".join(f"- {nm}: {err}" for nm, err in invalid) or "- (none)")
        try:
            reply = post(url.rstrip("/") + "/chat/completions",
                         {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.9,
                          "response_format": {"type": "json_object"}}, 300.0)
            candidates = _extract_json(reply["choices"][0]["message"]["content"]).get("candidates", [])
        except Exception as exc:
            log(f"rodada {r}: LLM falhou ({type(exc).__name__}); tentando de novo")
            time.sleep(15)
            continue
        for c in candidates[:n]:
            spec = c.get("strategy") or {}
            try:
                m = evaluate(spec, closes, SELECT)
                con.execute("INSERT INTO trials (round, name, why, spec, valid, cagr, max_dd, calmar, worst_month, t)"
                            " VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                            (r, str(c.get("name", ""))[:80], str(c.get("why", ""))[:300], json.dumps(spec),
                             m["cagr"], m["max_dd"], m["calmar"], m["worst_month"], m["t"]))
                log(f"rodada {r}: {c.get('name')} -> calmar {m['calmar']:.2f} (cagr {m['cagr']:.1%}, queda {m['max_dd']:.1%})")
            except (InvalidStrategy, KeyError, TypeError, ValueError) as exc:
                con.execute("INSERT INTO trials (round, name, why, spec, valid, error) VALUES (?, ?, ?, ?, 0, ?)",
                            (r, str(c.get("name", ""))[:80], str(c.get("why", ""))[:300], json.dumps(spec), str(exc)[:200]))
                log(f"rodada {r}: {c.get('name')} inválida ({exc})")
        con.commit()


def finalists(data: Path, db: Path, k: int = 3) -> None:
    """Só agora olha o período reservado: base + as k melhores da seleção."""
    closes = load_closes(data)
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT name, spec, calmar FROM trials WHERE valid = 1 AND round > 0 ORDER BY calmar DESC LIMIT ?",
                       (k,)).fetchall()
    base_sel = con.execute("SELECT calmar FROM trials WHERE round = 0").fetchone()[0]
    print(f"{'estratégia':40s} {'calmar 2018-23':>15s} | 2024-26: {'ao ano':>7s} {'queda':>6s} {'calmar':>7s} {'pior mês':>9s}")
    for name, spec, cal in [(BASE["name"], json.dumps(BASE), base_sel)] + rows:
        h = evaluate(json.loads(spec), closes, HOLDOUT)
        print(f"{name[:40]:40s} {cal:15.2f} | {h['cagr']:+7.1%} {h['max_dd']:6.1%} {h['calmar']:7.2f} {h['worst_month']:+9.1%}")


def main() -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="trader.research.strategy_search")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--finalists", action="store_true", help="só agora olha 2024-2026")
    p.add_argument("--llm-url", default="http://localhost:11434/v1")
    p.add_argument("--llm-model", default="qwen2.5:7b")
    a = p.parse_args()
    data, db = Path("data"), Path("data/research/strategy_search.sqlite")
    db.parent.mkdir(parents=True, exist_ok=True)
    if a.finalists:
        finalists(data, db)
    else:
        run(a.rounds, a.candidates, data, db, a.llm_url, a.llm_model, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
