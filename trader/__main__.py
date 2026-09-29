"""Linha de comando.

    python -m trader collect BTC/USDT 4h --since 2020-01-01
    python -m trader import-b3 R:\\downloads\\COTAHIST_A2025.ZIP --tickers BOVA11,PETR4
    python -m trader evolve BTC/USDT 4h --window-days 60 --holdout-days 180 --workers 8
    python -m trader evolve BTC/USDT 4h --llm --llm-model qwen2.5:3b
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import Executor
from pathlib import Path

import numpy as np
import pandas as pd

from trader.backtest import walk_forward_windows
from trader.evolution.creator import Creator, LLMCreator, RuleCreator
from trader.evolution.fitness import evaluate, risk_adjusted
from trader.evolution.population import BASELINE, Population, PopulationConfig, make_executor
from trader.evolution.registry import Registry
from trader.ingest import b3_cotahist, binance
from trader.storage.candles import CandleStore


B3_WATCHLIST = ["BOVA11", "SMAL11", "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3"]


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():  # saída redirecionada no Windows cai em cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="trader")
    parser.add_argument("--data", type=Path, default=Path("data"), help="pasta de dados")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="baixa/atualiza candles da Binance")
    collect.add_argument("symbol", help="ex.: BTC/USDT")
    collect.add_argument("timeframe", help="ex.: 1m, 15m, 1h, 1d")
    collect.add_argument("--since", default="2020-01-01", help="início, se ainda não houver dados")

    b3 = sub.add_parser("import-b3", help="importa arquivos COTAHIST (.ZIP ou .TXT) da B3")
    b3.add_argument("files", nargs="+", type=Path)
    b3.add_argument(
        "--tickers", default=",".join(B3_WATCHLIST), help="papéis separados por vírgula, ou 'all'"
    )

    evolve = sub.add_parser("evolve", help="roda o ciclo evolutivo em simulação")
    evolve.add_argument("symbol")
    evolve.add_argument("timeframe")
    evolve.add_argument("--window-days", type=int, default=60, help="duração de cada geração")
    evolve.add_argument("--holdout-days", type=int, default=180, help="final reservado, nunca usado na seleção")
    evolve.add_argument("--population", type=int, default=16)
    evolve.add_argument("--slots", type=int, default=32)
    evolve.add_argument("--seeds", default="0", help="sementes separadas por vírgula, ex.: 0,1,2")
    evolve.add_argument("--workers", type=int, default=8)
    evolve.add_argument("--llm", action="store_true", help="Criador usa LLM (API compatível com OpenAI)")
    evolve.add_argument("--llm-url", default="http://localhost:11434/v1")
    evolve.add_argument("--llm-model", default="qwen2.5:3b")

    args = parser.parse_args(argv)
    if args.command == "collect":
        store = CandleStore(args.data)
        since = pd.Timestamp(args.since, tz="UTC")
        added = binance.update(store, args.symbol, args.timeframe, since)
        last = store.last_ts(binance.EXCHANGE, args.symbol, args.timeframe)
        print(f"{args.symbol} {args.timeframe}: +{added} candles, último {last}")
    elif args.command == "import-b3":
        tickers = None if args.tickers == "all" else args.tickers.split(",")
        counts, events = b3_cotahist.import_files(CandleStore(args.data), args.files, tickers)
        for symbol, n in sorted(counts.items()):
            print(f"{symbol}: {n} pregões")
        for symbol, found in sorted(events.items()):
            for ts, factor in found:
                kind = f"desdobramento 1:{round(1 / factor)}" if factor < 1 else f"grupamento {round(factor)}:1"
                print(f"  ajuste {symbol} {ts:%Y-%m-%d}: {kind} (confira no site da B3)")
        missing = sorted(set(tickers or []) - set(counts))
        if missing:
            print("não encontrados no mercado à vista:", ", ".join(missing))
    elif args.command == "evolve":
        run_evolution(args)


def run_evolution(args: argparse.Namespace) -> None:
    candles = CandleStore(args.data).read(binance.EXCHANGE, args.symbol, args.timeframe)
    if candles.empty:
        raise SystemExit(f"sem dados; rode antes: python -m trader collect {args.symbol} {args.timeframe}")
    seeds = [int(x) for x in args.seeds.split(",")]
    executor = make_executor(args.workers)
    try:
        results = [evolve_once(args, candles, seed, executor, verbose=len(seeds) == 1) for seed in seeds]
    finally:
        if executor:
            executor.shutdown()

    print("\nHOLDOUT (nunca visto) — nota = retorno / queda máxima")
    print(f"{'semente':>7} {'campeão':>8} {'retorno':>8} {'queda':>6} {'ops':>4} {'nota':>6}")
    for r in results:
        if r["champion"] is None:
            print(f"{r['seed']:>7} {'—':>8}")
            continue
        h = r["holdout"]
        print(f"{r['seed']:>7} {r['champion']:>8} {h.ret:>+8.1%} {h.max_dd:>6.1%} {h.trades:>4} {h.score:>6.2f}")
    ref, bh = results[0]["baseline"], results[0]["buy_hold"]
    print(f"{'referência':>16} {ref.ret:>+8.1%} {ref.max_dd:>6.1%} {ref.trades:>4} {ref.score:>6.2f}")
    print(f"{'comprar/segurar':>16} {bh['ret']:>+8.1%} {bh['dd']:>6.1%} {'':>4} {risk_adjusted(bh['ret'], bh['dd']):>6.2f}")
    scores = [r["holdout"].score for r in results if r["champion"]]
    if scores:
        print(f"nota média dos campeões: {sum(scores) / len(scores):.2f} ({len(scores)} sementes)")


def evolve_once(
    args: argparse.Namespace, candles: pd.DataFrame, seed: int, executor: Executor | None, verbose: bool
) -> dict:
    first_ts, last_ts = candles["ts"].iloc[0], candles["ts"].iloc[-1]
    holdout_start = last_ts - pd.Timedelta(days=args.holdout_days)
    evo_start = first_ts + pd.Timedelta(days=30)  # aquecimento dos indicadores
    windows = walk_forward_windows(evo_start, holdout_start, pd.Timedelta(days=args.window_days))

    run = f"{pd.Timestamp.now():%Y%m%d-%H%M%S}-s{seed}{'-llm' if args.llm else ''}"
    db_path = registry_path(args.data, run)
    registry = Registry(db_path)
    creator: Creator = RuleCreator()
    if args.llm:
        creator = LLMCreator(
            args.symbol, args.timeframe, args.llm_url, args.llm_model, graveyard=lambda: registry.graveyard(10)
        )

    pop = Population(
        candles, args.symbol, args.timeframe, registry, creator,
        PopulationConfig(initial=args.population, max_slots=args.slots), seed=seed, executor=executor,
    )
    registry.set_meta("run", {
        "symbol": args.symbol, "timeframe": args.timeframe, "seed": seed, "window_days": args.window_days,
        "windows": [[str(a), str(b)] for a, b in windows], "creator": "llm" if args.llm else "rules",
        "llm_model": args.llm_model if args.llm else None,
    })
    pop.seed()
    print(f"[semente {seed}] {len(windows)} gerações de {args.window_days}d, {evo_start:%Y-%m-%d} a {holdout_start:%Y-%m-%d}")
    for start, end in windows:
        r = pop.step(start, end)
        if verbose:
            deaths = " ".join(f"{k}={v}" for k, v in sorted(r.deaths.items())) or "-"
            births = " ".join(f"{k}={v}" for k, v in sorted(r.births.items())) or "-"
            print(
                f"g{r.index:03d} {r.start:%Y-%m-%d} vivos={r.alive:2d} mortes[{deaths}] nascimentos[{births}]"
                f" melhor={r.best_id} {r.best_ret:+.1%} nota={r.best_score:.2f} ref={r.baseline_ret:+.1%}"
            )

    end = last_ts + pd.Timedelta(seconds=1)
    window = candles[candles["ts"] >= holdout_start]
    closes = window["close"].to_numpy()
    result = {
        "seed": seed,
        "champion": None,
        "baseline": evaluate(BASELINE, candles, args.symbol, args.timeframe, holdout_start, end),
        "buy_hold": {
            "ret": closes[-1] / window["open"].iloc[0] - 1,
            "dd": float((1 - closes / np.maximum.accumulate(closes)).max()),
        },
    }
    print(f"[semente {seed}] causas de morte: {registry.cause_counts()}")
    if args.llm:
        print(f"[semente {seed}] falhas da LLM (regras assumiram): {creator.failures}")
    champ = pop.champion()
    if champ is not None:
        result["champion"] = champ.id
        result["holdout"] = evaluate(champ.genome, candles, args.symbol, args.timeframe, holdout_start, end)
        print(f"[semente {seed}] campeão {champ.id} (linhagem {champ.lineage}): {champ.genome.to_dict()}")
        print(f"    vida: {champ.life.ret:+.1%}, queda {champ.life.worst_dd:.1%}, {champ.life.trades} ops")
        for step in registry.ancestry(champ.id)[:4]:
            print(f"    {step['id']} [{step['origin']}] {step['note'] or ''}")
    print(f"[semente {seed}] registro: {db_path}")
    return result


def registry_path(data: Path, run: str) -> Path:
    return data / "evolution" / f"{run}.sqlite"


if __name__ == "__main__":
    main()
