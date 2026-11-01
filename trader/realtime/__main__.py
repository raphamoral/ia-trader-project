"""python -m trader.realtime watch --seconds 60    → conecta, mostra livro/fluxo/indicadores e a latência

Só leitura de mercado (e executor em papel com a política atual, se --paper). Não fica ligado
sozinho: roda pelo tempo pedido e encerra.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

import pandas as pd

from trader.realtime.engine import RealtimeEngine


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["watch"])
    ap.add_argument("--symbols", default="SOL/USDT,ETH/USDT,XRP/USDT")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--paper", action="store_true", help="executor em papel com data/policy/current.json")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    symbols = args.symbols.split(",")
    executor = None
    if args.paper:
        from trader.daytrade.broker import FuturesPaperBroker
        from trader.policy.executor import PolicyExecutor
        from trader.policy.writer import load_current

        executor = PolicyExecutor(FuturesPaperBroker(1000.0))
        policy = load_current(Path(args.data) / "policy", symbols)
        if policy is not None:
            executor.set_policy(policy, pd.Timestamp.now(tz="UTC"))

    def on_minute(sym: str, row: dict) -> None:
        print(f"{row['ts']:%H:%M} {sym} close {row['close']:.6g} rsi {row['rsi14']:.1f} cvd {row['cvd']:+.1f}"
              f" agressão {row['aggr_ratio']:+.2f} desequilíbrio5 {row.get('imbalance5', float('nan')):+.2f}"
              f" spread {row.get('spread_bps', float('nan')):.2f}bps", flush=True)

    eng = RealtimeEngine(symbols, executor, on_minute)
    asyncio.run(eng.run(args.seconds))
    print(f"\n{eng.messages} mensagens em {args.seconds:.0f}s")
    for st in eng.states.values():
        lag = sorted(st.lag_ms)
        book = st.book
        print(f"{st.symbol}: preço {st.last_price:.6g}"
              + (f" | spread {book.spread_bps:.2f} bps, desequilíbrio5 {book.imbalance5:+.2f},"
                 f" parede compra {book.bid_wall}, parede venda {book.ask_wall}" if book else "")
              + f" | fluxo {st.flow.snapshot()}"
              + (f" | atraso mediano {statistics.median(lag):.0f} ms (inclui relógio do PC)" if lag else ""))


if __name__ == "__main__":
    main()
