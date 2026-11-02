"""Diário de comandos: o estado do livro é função pura da sequência de comandos.

Cada comando vai ao disco ANTES de ser aplicado (write-ahead), numa linha com CRC32. Se o
processo morrer no meio de uma escrita, só a última linha pode estar rasgada: a recuperação a
corta e segue. CRC errado no meio do arquivo é corrupção de verdade, e a recuperação recusa.

O snapshot grava o estado e o número do último comando. A recuperação carrega o snapshot e
reaplica só os comandos posteriores, então uma queda entre o snapshot e a limpeza do diário
não aplica nada duas vezes.
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Any

from trader.lob.book import LimitOrderBook, Result, Side, Stp, Tif
from trader.storage.atomic import replace_with_retry, unique_tmp


class JournalCorrupt(Exception):
    pass


def apply_command(book: LimitOrderBook, cmd: dict[str, Any]) -> Result:
    op = cmd["op"]
    if op == "submit":
        return book.submit(cmd["id"], cmd["owner"], Side(cmd["side"]), cmd["qty"], cmd.get("price"),
                           Tif(cmd.get("tif", "gtc")), Stp(cmd.get("stp", "none")))
    if op == "cancel":
        return book.cancel(cmd["id"])
    if op == "modify":
        return book.modify(cmd["id"], cmd["qty"], cmd.get("price"), Stp(cmd.get("stp", "none")))
    raise ValueError(f"comando desconhecido: {op!r}")


def _encode(entry: dict) -> bytes:
    body = json.dumps(entry, separators=(",", ":"), sort_keys=True).encode()
    return b"%08x %s\n" % (zlib.crc32(body), body)


def _decode(line: bytes) -> dict | None:
    """None se a linha não confere."""
    if not line.endswith(b"\n") or len(line) < 10 or line[8:9] != b" ":
        return None
    body = line[9:-1]
    try:
        return json.loads(body) if int(line[:8], 16) == zlib.crc32(body) else None
    except ValueError:
        return None


def read_journal(path: Path) -> tuple[list[dict], int]:
    """Entradas válidas e o tamanho em bytes do trecho íntegro. Só tolera defeito na última linha."""
    if not path.exists():
        return [], 0
    entries, good, lines = [], 0, path.read_bytes().splitlines(keepends=True)
    for i, line in enumerate(lines):
        entry = _decode(line)
        if entry is None:
            if i == len(lines) - 1:
                break
            raise JournalCorrupt(f"{path}: linha {i + 1} corrompida no meio do diário")
        entries.append(entry)
        good += len(line)
    return entries, good


class JournaledBook:
    def __init__(self, folder: Path | str, fsync: bool = False) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._journal = self.folder / "journal.log"
        self._snapshot = self.folder / "snapshot.json"
        self.book = LimitOrderBook()
        self.applied = 0
        if self._snapshot.exists():
            state = json.loads(self._snapshot.read_text(encoding="utf-8"))
            self.book, self.applied = LimitOrderBook.load(state["book"]), state["applied"]
        entries, good = read_journal(self._journal)
        for entry in entries:
            if entry["n"] > self.applied:  # já contido no snapshot: não reaplica
                apply_command(self.book, entry["cmd"])
                self.applied = entry["n"]
        if self._journal.exists() and good < self._journal.stat().st_size:
            with open(self._journal, "r+b") as f:
                f.truncate(good)
        self._file = open(self._journal, "ab")

    def apply(self, cmd: dict[str, Any]) -> Result:
        n = self.applied + 1
        self._file.write(_encode({"n": n, "cmd": cmd}))
        self._file.flush()
        if self.fsync:
            os.fsync(self._file.fileno())
        self.applied = n
        return apply_command(self.book, cmd)

    def snapshot(self) -> None:
        """Grava o estado e só então esvazia o diário."""
        tmp = unique_tmp(self._snapshot)
        tmp.write_text(json.dumps({"applied": self.applied, "book": self.book.dump()}), encoding="utf-8")
        replace_with_retry(tmp, self._snapshot)
        self._file.close()
        with open(self._journal, "wb"):
            pass
        self._file = open(self._journal, "ab")

    def close(self) -> None:
        self._file.close()
