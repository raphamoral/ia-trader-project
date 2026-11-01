"""Peça 4, parte 1: a Qwen rotula manchetes de cripto no formato do reflexo (tipo, moedas, sentimento, gravidade).

Corpus: manchetes do RSS + manchetes tiradas do endereço das notícias do GDELT
(o "slug" costuma ser a manchete), filtradas por palavras de cripto, espalhadas
por mês de 2023 a 2026. Rotula em lotes de 20 por chamada; o cache é retomável.

  python -m trader.realtime.news_labels --per-month 80 --budget-min 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from trader.policy.schema import NEWS_KINDS
from trader.policy.writer import _json, ollama_chat

STRONG = ("bitcoin", "btc", "ethereum", "ether", "solana", "xrp", "ripple", "crypto", "binance", "stablecoin",
          "defi", "blockchain", "coinbase", "tether", "usdt", "altcoin", "memecoin", "token")
SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
BATCH = 20

PROMPT = """Label crypto news headlines for a trading reflex. For each headline return:
- relevant: true if it can move crypto prices (BTC, ETH, SOL, XRP or the crypto market), else false
- kind: one of {kinds}
- symbols: subset of ["BTC","ETH","SOL","XRP"] directly concerned; ["MARKET"] if the whole crypto market; [] if none
- sentiment: -1 (very bearish) to 1 (very bullish) for those symbols
- severity: 0 (noise) to 1 (market-moving: e.g. hack >= $100M, ETF approval, exchange collapse, major ban)
Headlines may be URL slugs (lowercase, no punctuation) or in other languages.
Answer JSON only: {{"items": [{{"i": 0, "relevant": true, "kind": "...", "symbols": [...], "sentiment": 0.0, "severity": 0.0}}, ...]}}

Headlines:
{lines}"""


def slug_text(url: str) -> str:
    m = re.search(r"/([^/?#]+?)(?:\.html?|\.php|\.amp)?/?(?:[?#].*)?$", url or "")
    s = re.sub(r"[-_+]+", " ", m.group(1) if m else "")
    s = re.sub(r"\b[0-9a-f]{6,}\b|\b\d{5,}\b", " ", s)  # ids
    return re.sub(r"\s+", " ", s).strip()


def is_crypto(text: str) -> bool:
    t = f" {text.lower()} "
    return any(f" {w}" in t for w in STRONG)


def corpus(data: Path, per_month: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for f in sorted((data / "news" / "gdelt_gkg").glob("*.parquet")):
        df = pd.read_parquet(f, columns=["url", "published_at"])
        df["text"] = df["url"].map(slug_text)
        df = df[(df["text"].str.count(" ") >= 4) & df["text"].map(is_crypto)].drop_duplicates("text")
        rows.append(df.sample(min(per_month, len(df)), random_state=seed)[["text", "published_at"]].assign(src="gkg"))
    from trader.news.store import NewsStore

    rss = NewsStore(data, subdir="items").read_range(None, None)
    if len(rss):
        rss = rss.assign(text=rss["original_text"].fillna("").str.slice(0, 200))
        rss = rss[rss["text"].map(is_crypto)].drop_duplicates("text")
        rows.append(rss[["text", "published_at"]].assign(src="rss"))
    out = pd.concat(rows, ignore_index=True)
    out["key"] = out["text"].map(lambda t: hashlib.sha256(t.encode()).hexdigest()[:20])
    return out.drop_duplicates("key").reset_index(drop=True)


def clean_label(raw: dict) -> dict | None:
    kind = raw.get("kind", "other")
    syms = [s for s in raw.get("symbols", []) if s in SYMBOLS or s == "MARKET"]
    try:
        sent = max(-1.0, min(1.0, float(raw.get("sentiment", 0.0))))
        sev = max(0.0, min(1.0, float(raw.get("severity", 0.0))))
    except (TypeError, ValueError):
        return None
    return {"relevant": bool(raw.get("relevant", False)), "kind": kind if kind in NEWS_KINDS else "other",
            "symbols": syms, "sentiment": sent, "severity": sev}


def cache_path(data: Path) -> Path:
    return data / "news" / "cache" / "reflex_labels.jsonl"


def load_cache(data: Path) -> dict[str, dict]:
    p = cache_path(data)
    if not p.exists():
        return {}
    return {e["key"]: e for e in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def label(data: Path, items: pd.DataFrame, budget_s: float, chat=ollama_chat, log=print) -> int:
    cache = load_cache(data)
    todo = items[~items["key"].isin(cache)]
    path = cache_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.monotonic(), 0
    for start in range(0, len(todo), BATCH):
        if time.monotonic() - t0 > budget_s:
            break
        chunk = todo.iloc[start:start + BATCH]
        lines = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk["text"]))
        tc = time.monotonic()
        try:
            answer = _json(chat([{"role": "user", "content": PROMPT.format(kinds=list(NEWS_KINDS), lines=lines)}]))
        except Exception as exc:
            log(f"lote falhou: {type(exc).__name__}: {exc}")
            continue
        got = []
        for it in answer.get("items", []) if isinstance(answer, dict) else []:
            try:
                i = int(it.get("i"))
            except (TypeError, ValueError):
                continue
            lab = clean_label(it)
            if lab is not None and 0 <= i < len(chunk):
                r = chunk.iloc[i]
                got.append({"key": r["key"], "text": r["text"], "src": r["src"],
                            "published_at": str(r["published_at"]), **lab})
        with path.open("a", encoding="utf-8") as f:
            for g in got:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        done += len(got)
        log(f"{done} rotuladas (+{len(got)}/{len(chunk)} em {time.monotonic() - tc:.0f}s);"
            f" faltam {len(todo) - start - len(chunk)}")
    return done


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-month", type=int, default=80)
    ap.add_argument("--budget-min", type=float, default=120)
    args = ap.parse_args()
    data = Path(args.data)
    items = corpus(data, args.per_month)
    print(f"corpus: {len(items)} manchetes ({items['src'].value_counts().to_dict()})", flush=True)
    label(data, items, args.budget_min * 60, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
