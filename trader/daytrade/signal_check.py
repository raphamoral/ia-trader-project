"""Existe sinal para o Criador do day trade aprender? (mesmo método de trader/training/signal_check.py)

1. Uma evolução curta (Criador por regras) grava cada morte: pai, autópsia,
   sucessor real e a janela em que morreu.
2. Para cada morte, pontua os candidatos nas K janelas SEGUINTES, em vários
   ativos, e tira a média entre ativos:
     successor  o sucessor que a evolução de fato criou
     rules      a correção por regras recalculada (outra semente aleatória)
     mutation0/1 mutações aleatórias do pai (escala 0,3)
     parent     o pai continuando vivo
     baseline   o motor de referência
3. Veredito: IC 95% por bootstrap da diferença regras − média das mutações.
   Só há sinal se o IC ficar inteiro acima de zero.

Dois braços:
  price    genomas só de preço (sem regime_map e sem genes de notícia)
  context  genomas com regime_map e genes de notícia (provedores reais)

    python -m trader.daytrade.signal_check --timeframe 15m --symbols BTC/USDT,ETH/USDT,SOL/USDT --arm price
"""
from __future__ import annotations

import argparse
import gc
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from trader.backtest import walk_forward_windows
from trader.daytrade import data
from trader.daytrade.creator import DayRuleCreator
from trader.daytrade.evolution import DAY_BASELINE, DayConfig, DayPopulation, evaluate_day
from trader.daytrade.features import attach_features, real_black, real_white
from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.genome import DayGenome, mutate_day
from trader.domain import parse_timeframe
from trader.evolution.fitness import Life, risk_adjusted
from trader.evolution.registry import Registry
from trader.storage.candles import CandleStore
from trader.training.signal_check import verdict

CASH = 10_000.0  # USDT: grande o bastante para a ordem mínima não distorcer o teste do Criador
LIMITS = DayTradeLimits()


def strip_context(g: DayGenome) -> DayGenome:
    return replace(g, regime_map=None, news_gate=None, news_follow=None, event_blackout=0,
                   news_size_mult=0.0, black_weight=0.0)


class _Recorder:
    """Envolve o Criador e grava cada morte (pai, autópsia, sucessor, janela)."""

    def __init__(self, inner: DayRuleCreator, price_only: bool) -> None:
        self.inner, self.price_only = inner, price_only
        self.pop: DayPopulation | None = None
        self.deaths: list[dict[str, Any]] = []

    def __getattr__(self, name):  # styles, black_allowed etc. vão para o Criador real
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in ("inner", "price_only", "pop", "deaths"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def successor(self, g, autopsy, rng):
        new, why = self.inner.successor(g, autopsy, rng)
        if self.price_only:
            new = strip_context(new)
        self.deaths.append({"window": self.pop._window, "parent": g, "successor": new, "autopsy": autopsy})
        return new, why

    def child(self, g, rng):
        new = self.inner.child(g, rng)
        return strip_context(new) if self.price_only else new


@dataclass(frozen=True)
class Example:
    parent: DayGenome
    successor: DayGenome
    autopsy: dict[str, Any]
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    key: str


_FRAMES: dict[tuple[str, str], pd.DataFrame] = {}
_CTX: dict[str, Any] = {}


def _init(data_dir: str, timeframe: str, symbols: list[str], news: bool) -> None:
    _CTX.update(data=data_dir, timeframe=timeframe, symbols=symbols, news=news)


def _frame(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Candles com margem de aquecimento, carregados por intervalo (RAM apertada)."""
    tf = _CTX["timeframe"]
    warm = parse_timeframe(tf) * 3000
    df = CandleStore(_CTX["data"]).read(data.EXCHANGE, symbol, tf, start - warm, end)
    if _CTX["news"] and len(df):
        df = attach_features(df, symbol, tf, [real_white(_CTX["data"]), real_black(True, _CTX["data"])])
    return df


def _score(job: tuple[Example, dict[str, DayGenome]]) -> dict[str, float] | None:
    ex, cands = job
    lo, hi = ex.windows[0][0], ex.windows[-1][1]
    frames = {s: _frame(s, lo, hi) for s in _CTX["symbols"]}
    usable = [s for s, f in frames.items() if len(f) and f["ts"].iloc[0] <= lo - parse_timeframe(_CTX["timeframe"]) * 500]
    if not usable:
        return None
    out = {}
    for name, g in cands.items():
        per_asset = []
        for s in usable:
            life = Life([evaluate_day(g, frames[s], s, _CTX["timeframe"], a, b, LIMITS, None, CASH,
                                      data_spec(s)).stats for a, b in ex.windows])
            per_asset.append(risk_adjusted(life.ret, life.worst_dd))
        out[name] = statistics.mean(per_asset)
    return out


def data_spec(symbol: str):
    from trader.daytrade.market import NO_MINIMUM

    return NO_MINIMUM  # capital grande: a ordem mínima não interfere


def candidates(ex: Example, price_only: bool) -> dict[str, DayGenome]:
    rng = random.Random(ex.key)
    cr = DayRuleCreator()
    cr.styles = [ex.parent.style]
    cr.black_allowed = not price_only
    rules, _ = cr.successor(ex.parent, ex.autopsy, rng)
    out = {"successor": ex.successor, "rules": rules, "parent": ex.parent, "baseline": DAY_BASELINE}
    for i in range(2):
        out[f"mutation{i}"] = mutate_day(ex.parent, rng, 0.3, styles=[ex.parent.style], black_allowed=not price_only)
    if price_only:
        out = {k: strip_context(v) for k, v in out.items()}
    return out


def collect_examples(args, seed: int) -> list[Example]:
    symbol = args.evolve_symbol
    tf = args.timeframe
    store = CandleStore(args.data)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    candles = store.read(data.EXCHANGE, symbol, tf, start - parse_timeframe(tf) * 3000, end)
    windows = walk_forward_windows(start, end, pd.Timedelta(days=args.window_days))
    price_only = args.arm == "price"
    providers = [] if price_only else [real_white(str(args.data)), real_black(True, str(args.data))]
    rec = _Recorder(DayRuleCreator(), price_only)
    cfg = DayConfig(initial=args.population, max_slots=args.population + 6, limits=LIMITS, cash=CASH,
                    styles=("any",), news=not price_only, black=not price_only)
    pop = DayPopulation(candles, symbol, tf, Registry(":memory:"), cfg, seed=seed, creator=rec, providers=providers)
    rec.pop = pop
    if price_only:
        for a in pop.alive:
            a.genome = strip_context(a.genome)
    pop.seed()
    if price_only:
        for a in pop.alive:
            a.genome = strip_context(a.genome)
    for a, b in windows:
        pop.step(a, b)
    examples = []
    for k, d in enumerate(rec.deaths):
        first = d["window"] + 1
        if first + args.horizon > len(windows):
            continue
        examples.append(Example(d["parent"], d["successor"], d["autopsy"], windows[first : first + args.horizon],
                                f"{seed}:{k}"))
    del pop, candles
    gc.collect()
    return examples


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(prog="trader.daytrade.signal_check")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    p.add_argument("--evolve-symbol", default="SOL/USDT")
    p.add_argument("--arm", choices=["price", "context"], default="price")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-09-01")
    p.add_argument("--window-days", type=int, default=15)
    p.add_argument("--horizon", type=int, default=2, help="janelas pontuadas após a morte")
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--max-examples", type=int, default=120)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)

    symbols = args.symbols.split(",")
    examples: list[Example] = []
    for seed in (int(s) for s in args.seeds.split(",")):
        ex = collect_examples(args, seed)
        print(f"[{args.arm} s{seed}] {len(ex)} autópsias com {args.horizon} janelas seguintes", flush=True)
        examples.extend(ex)
    rng = random.Random(0)
    if len(examples) > args.max_examples:
        examples = rng.sample(examples, args.max_examples)
    jobs = [(ex, candidates(ex, args.arm == "price")) for ex in examples]
    init = (str(args.data), args.timeframe, symbols, args.arm == "context")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=init) as pool:
            scores = [s for s in pool.map(_score, jobs, chunksize=4) if s]
    else:
        _init(*init)
        scores = [s for s in map(_score, jobs) if s]
    if not scores:
        raise SystemExit("nenhuma autópsia pontuável")
    v = verdict(scores)
    print(f"\nSIGNAL CHECK day trade — braço {args.arm}, {args.timeframe}, ativos {','.join(symbols)},"
          f" {v.n} autópsias, {args.horizon} janelas de {args.window_days}d após cada morte")
    print("nota média (retorno ÷ queda, média entre ativos):")
    for name, m in sorted(v.means.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<10} {m:+.3f}")
    print(f"regras > média das mutações em {v.rules_vs_random:.0%} das autópsias;"
          f" IC 95% (regras − mutação): [{v.ci_low:+.3f}, {v.ci_high:+.3f}]")
    print("VEREDITO:", "HÁ SINAL (IC acima de zero)" if v.has_signal else "SEM SINAL (IC inclui zero ou é negativo)")


if __name__ == "__main__":
    main()
