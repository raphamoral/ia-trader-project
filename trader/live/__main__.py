"""Evolução ao vivo em papel, 24h.

    python -m trader.live BTC/USDT 15m --window 3h --bootstrap data\\evolution\\20260929-002334-s0-llm.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trader.live.runner import LiveConfig, run


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="trader.live")
    p.add_argument("symbol", nargs="?", default="BTC/USDT")
    p.add_argument("timeframe", nargs="?", default="15m")
    p.add_argument("--window", default="3h", help="duração de cada geração")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, default=None, help="registro da população ao vivo")
    p.add_argument("--bootstrap", type=Path, default=None, help="registro cujos sobreviventes semeiam a população")
    p.add_argument("--llm-url", default="http://localhost:11434/v1")
    p.add_argument("--llm-model", default="qwen2.5:7b")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--features", choices=["none", "white", "all"], default="none",
                   help="notícias que os agentes enxergam: nenhuma, white, ou white + black")
    p.add_argument("--extra-symbols", default="", help="outros ativos operados pelo mesmo genoma, ex.: ETH/USDT,SOL/USDT")
    p.add_argument("--memory", nargs="*", default=[], help="registros de evoluções no histórico usados como memória")
    a = p.parse_args(argv)
    db = a.db or a.data / "live" / f"{a.symbol.replace('/', '')}-{a.timeframe}.sqlite"
    config = LiveConfig(a.symbol, a.timeframe, a.window, llm_url=a.llm_url, llm_model=a.llm_model,
                        workers=a.workers, features=a.features, memory=tuple(str(m) for m in a.memory),
                        extra_symbols=tuple(x for x in a.extra_symbols.split(",") if x))
    run(a.data, db, config, a.bootstrap)


if __name__ == "__main__":
    main()
