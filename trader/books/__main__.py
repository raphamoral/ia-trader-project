"""RAG dos livros.

    python -m trader.books index              (indexa data\\books\\library; incremental, pode interromper)
    python -m trader.books search "stop loss after a losing streak" --topic risk
    python -m trader.books distill            (o Qwen escreve o playbook: regras + faixas de genes, com citações)
    python -m trader.books ask "It lost 18% in a volatile downtrend"   (busca agêntica, usa o Qwen)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trader.books.index import Library, build

LIBRARY = Path("data/books/library")
INDEX = Path("data/books/index")


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="trader.books")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--source", type=Path, default=LIBRARY)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--style", action="append")
    s.add_argument("--topic", action="append")
    sub.add_parser("distill", help="o Qwen lê os trechos dos livros e escreve o playbook (regras + faixas de genes)")
    a = sub.add_parser("ask")
    a.add_argument("situation")
    args = p.parse_args(argv)
    if args.cmd == "index":
        stats = build(args.source, INDEX, log=lambda m: print(m, flush=True))
        print("índice:", stats)
        return
    lib = Library.load(INDEX)
    if args.cmd == "distill":
        from trader.books.distill import distill, save, top_rules

        playbook = distill(lib, log=lambda m: print(m, flush=True))
        save(playbook, Path("data/books/playbook.json"))
        print(f"{len(playbook['rules'])} regras, {len(playbook['priors'])} genes com faixa → data/books/playbook.json")
        for r in top_rules(playbook, 8):
            print(" -", r)
        return
    if args.cmd == "search":
        hits = lib.search(args.query, styles=args.style, topics=args.topic)
    else:
        from trader.books.agentic import AgenticLibrary

        hits = AgenticLibrary(lib)(args.situation)
    for h in hits:
        print(f"[{h['book']} p.{h['page']}] ({h['score']}) {h['text'][:300]}\n")


if __name__ == "__main__":
    main()
