"""Grava o stream público de livro e negócios dos futuros da Binance.

O histórico tick a tick do livro não é público; para simular a fila é preciso gravar.
O arquivo é JSON por linha em gzip (data/lob/<SÍMBOLO>-<início>.jsonl.gz):
  {"k":"meta","tick":...,"lot":...}   grade de preço e quantidade
  {"k":"snap","ts":...,"data":{...}}  snapshot REST
  {"k":"d","ts":...,"e":{...}}        atualização do livro
  {"k":"t","ts":...,"e":{...}}        negócio agregado
"""
from __future__ import annotations

import asyncio
import gzip
import json
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Iterator

import certifi

from trader.lob.depth import DepthBook, to_units
from trader.lob.sim import Diff, Snapshot, Trade
from trader.realtime.engine import WS_BOOK, WS_TRADES

REST = "https://fapi.binance.com/fapi/v1"
_SSL = ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20, context=_SSL) as response:
        return json.load(response)


def instrument(symbol: str) -> tuple[str, str]:
    """(tick de preço, passo de quantidade) do contrato."""
    for row in _get(f"{REST}/exchangeInfo")["symbols"]:
        if row["symbol"] == symbol:
            filters = {f["filterType"]: f for f in row["filters"]}
            return filters["PRICE_FILTER"]["tickSize"], filters["LOT_SIZE"]["stepSize"]
    raise ValueError(f"contrato desconhecido: {symbol}")


async def _record(symbol: str, seconds: int, path: Path) -> dict:
    import websockets

    tick, lot = instrument(symbol)
    name = symbol.lower()
    counts = {"d": 0, "t": 0, "snap": 0}
    stop_at = time.monotonic() + seconds
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"k": "meta", "symbol": symbol, "tick": tick, "lot": lot}) + "\n")

        def write(kind: str, ts: int, payload: dict, key: str) -> None:
            counts[kind] += 1
            out.write(json.dumps({"k": kind, "ts": ts, key: payload}, separators=(",", ":")) + "\n")

        async def stream(url: str, kind: str, stamp: str, started: asyncio.Event) -> None:
            async with websockets.connect(url, ssl=_SSL, max_queue=4096) as ws:
                started.set()
                while time.monotonic() < stop_at:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    event = json.loads(raw)["data"]
                    write(kind, event[stamp], event, "e")

        book_up, trades_up = asyncio.Event(), asyncio.Event()
        tasks = [asyncio.create_task(stream(f"{WS_BOOK}{name}@depth@100ms", "d", "E", book_up)),
                 asyncio.create_task(stream(f"{WS_TRADES}{name}@aggTrade", "t", "T", trades_up))]
        await asyncio.wait_for(asyncio.gather(book_up.wait(), trades_up.wait()), timeout=30)
        await asyncio.sleep(2)  # o buffer já precisa cobrir o snapshot
        snap = await asyncio.to_thread(_get, f"{REST}/depth?symbol={symbol}&limit=1000")
        write("snap", int(time.time() * 1000), snap, "data")
        await asyncio.gather(*tasks)
    return counts


def record(symbol: str, seconds: int, folder: Path = Path("data/lob")) -> tuple[Path, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl.gz"
    return path, asyncio.run(_record(symbol, seconds, path))


def load(path: Path | str) -> tuple[DepthBook, Iterator[Trade | Diff | Snapshot], str]:
    """Livro vazio e os eventos na ordem em que chegaram, sem carregar o arquivo na memória."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        meta = json.loads(f.readline())
    tick, lot = meta["tick"], meta["lot"]

    def events() -> Iterator[Trade | Diff | Snapshot]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            f.readline()
            for line in f:
                row = json.loads(line)
                if row["k"] == "d":
                    yield Diff(row["ts"], row["e"])
                elif row["k"] == "snap":
                    yield Snapshot(row["ts"], row["data"])
                else:
                    e = row["e"]
                    yield Trade(row["ts"], to_units(e["p"], tick), to_units(e["q"], lot), bool(e["m"]))

    return DepthBook(tick, lot), events(), lot
