"""Livro de ofertas: grava o stream público e mede a execução passiva × agressiva.

  python -m trader.lob record SOLUSDT --seconds 3600
  python -m trader.lob study data/lob/SOLUSDT-20261101T120000.jsonl.gz --usd 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trader.lob import record as rec
from trader.lob.sim import Costs, Snapshot, run_study


def main() -> None:
    parser = argparse.ArgumentParser(prog="trader.lob")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record", help="grava livro e negócios (Binance futuros)")
    r.add_argument("symbol")
    r.add_argument("--seconds", type=int, default=3600)
    s = sub.add_parser("study", help="ordem limitada na fila × a mercado, sobre uma gravação")
    s.add_argument("file", type=Path)
    s.add_argument("--usd", type=float, default=1000.0, help="tamanho da ordem")
    s.add_argument("--every-ms", type=int, default=30_000)
    s.add_argument("--patience-ms", type=int, default=10_000)
    s.add_argument("--latency-ms", type=int, default=100)
    s.add_argument("--markout-ms", type=int, default=5_000)
    s.add_argument("--maker-bps", type=float, default=2.0)
    s.add_argument("--taker-bps", type=float, default=5.0)
    args = parser.parse_args()

    if args.command == "record":
        path, counts = rec.record(args.symbol, args.seconds)
        print(path, counts)
        return

    # o tamanho em lotes sai do preço do snapshot: uma primeira leitura só até ele
    book, events, lot = rec.load(args.file)
    snapshot = next(e for e in events if isinstance(e, Snapshot))
    events.close()
    price = float(snapshot.data["bids"][0][0])
    qty = max(1, round(args.usd / price / float(lot)))

    book, events, _ = rec.load(args.file)
    study = run_study(events, book, qty, args.every_ms, args.patience_ms, args.latency_ms, args.markout_ms,
                      Costs(args.maker_bps, args.taker_bps))
    print(json.dumps({"qty_lotes": qty, **study.summary(), "buracos": book.gaps, "cruzados": book.crossed,
                      "atualizacoes": book.applied}, indent=2))


if __name__ == "__main__":
    main()
