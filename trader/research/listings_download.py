"""Baixa o diário de TODAS as listagens USDT da Binance (inclusive deslistadas) e o TVL da DeFiLlama.

  python -m trader.research.listings_download klines   → data/research/universe_all/<SYM>.parquet (8 threads)
  python -m trader.research.listings_download tvl      → data/research/tvl/{protocols,chains}.parquet
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from trader.research.universe_test import download_daily

ROOT = Path("data/research")


def klines() -> None:
    syms = [s[:-4] for s in ROOT.joinpath("binance_usdt_symbols.txt").read_text().split()]
    cache = ROOT / "universe_all"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [s for s in syms if not (cache / f"{s}.parquet").exists()]
    print(f"{len(todo)} símbolos a baixar de {len(syms)}", flush=True)
    done = 0
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(download_daily, cache, s, "2017-08", lambda m: None): s for s in todo}
        for f in as_completed(futs):
            done += 1
            try:
                d = f.result()
                print(f"{done}/{len(todo)} {futs[f]}: {len(d)} dias", flush=True)
            except Exception as exc:
                print(f"{done}/{len(todo)} {futs[f]}: ERRO {exc}", flush=True)


def _get(url: str, tries: int = 3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def tvl() -> None:
    out = ROOT / "tvl"
    out.mkdir(parents=True, exist_ok=True)
    syms = {s[:-4] for s in ROOT.joinpath("binance_usdt_symbols.txt").read_text().split()}
    protos = _get("https://api.llama.fi/protocols") or []
    # um protocolo por símbolo: o de maior TVL atual (versões V2/V3 somam no "parentProtocol", mas o histórico fica por versão)
    best: dict[str, dict] = {}
    for p in protos:
        sym = (p.get("symbol") or "").upper()
        if sym in syms and (sym not in best or (p.get("tvl") or 0) > (best[sym].get("tvl") or 0)):
            best[sym] = p
    print(f"{len(best)} símbolos com protocolo na DeFiLlama", flush=True)
    rows = []
    for i, (sym, p) in enumerate(sorted(best.items()), 1):
        d = _get(f"https://api.llama.fi/protocol/{p['slug']}")
        for pt in (d or {}).get("tvl") or []:
            rows.append({"symbol": sym, "ts": pd.Timestamp(pt["date"], unit="s", tz="UTC"), "tvl": float(pt["totalLiquidityUSD"])})
        if i % 25 == 0:
            print(f"  protocolos {i}/{len(best)}", flush=True)
        time.sleep(0.3)
    pd.DataFrame(rows).to_parquet(out / "protocols.parquet")
    chains = _get("https://api.llama.fi/v2/chains") or []
    rows = []
    for c in chains:
        sym = (c.get("tokenSymbol") or "").upper()
        if sym in syms:
            d = _get(f"https://api.llama.fi/v2/historicalChainTvl/{c['name']}")
            for pt in d or []:
                rows.append({"symbol": sym, "chain": c["name"], "ts": pd.Timestamp(pt["date"], unit="s", tz="UTC"), "tvl": float(pt["tvl"])})
            time.sleep(0.3)
    pd.DataFrame(rows).to_parquet(out / "chains.parquet")
    print(f"cadeias com símbolo na lista: {len({r['symbol'] for r in rows})}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"klines": klines, "tvl": tvl}[sys.argv[1]]()
