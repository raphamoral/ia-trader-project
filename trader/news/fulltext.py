"""Texto completo das notícias importantes, lido da página original.

O RSS descobre a notícia e dá o horário; aqui, só para fontes de nível 1 e 2
ainda recentes, a página é lida (respeitando o robots.txt, devagar, sem
contornar paywall) e o Qwen rotula título + texto.

Sem vazamento: o rótulo do texto completo só vale a partir de `labeled_at`.
O pipeline troca o rótulo do título por este e adia a visibilidade do item
para max(visible_at, labeled_at).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from trader.news.enrich import Enricher, validate
from trader.news.rss import _SSL

USER_AGENT = "Mozilla/5.0 (ai-trader-research)"
MIN_CHARS = 600  # abaixo disso é chamada de paywall, lista ou página vazia: não usa
BODY_CHARS = 2000  # trecho do corpo mandado à LLM
PER_DOMAIN_DELAY_S = 3.0
MAX_AGE = pd.Timedelta(hours=24)  # só notícias recentes (ao vivo); histórico não é relido


def store_dir(data: Path) -> Path:
    return data / "news" / "fulltext"


def labels_path(data: Path) -> Path:
    return data / "news" / "cache" / "fulltext_labels.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


class Robots:
    """Cache de robots.txt por domínio. Se não der para ler, NÃO busca (conservador)."""

    def __init__(self, fetch: Callable[[str], str] | None = None) -> None:
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._fetch = fetch or _get_text

    def allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._parsers:
            try:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(self._fetch(root + "/robots.txt").splitlines())
                self._parsers[root] = parser
            except Exception:
                self._parsers[root] = None
        parser = self._parsers[root]
        return parser is not None and parser.can_fetch(USER_AGENT, url)


def _get_text(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
        return resp.read().decode("utf-8", "replace")


def extract(html: str) -> str:
    import trafilatura

    return trafilatura.extract(html, include_comments=False, include_tables=False) or ""


def candidates(items: pd.DataFrame, done: set[str], now: pd.Timestamp) -> pd.DataFrame:
    """Notícias ao vivo, recentes, de nível 1/2, com URL e ainda não lidas."""
    if items.empty:
        return items
    live = items[~items["backfilled"].astype(bool)] if "backfilled" in items.columns else items
    recent = live[(now - live["first_seen_at"]) <= MAX_AGE]
    sel = recent[recent["tier"].isin([1, 2]) & recent["url"].fillna("").str.startswith("http")]
    sel = sel[~sel["id"].isin(done)]
    return sel.sort_values(["tier", "first_seen_at"], ascending=[True, False])


def fetch_and_label(
    data: Path,
    items: pd.DataFrame,
    enricher: Enricher,
    limit: int = 15,
    get: Callable[[str], str] = _get_text,
    robots: Robots | None = None,
    now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    robots = robots or Robots()
    seen_path = store_dir(data) / "fetched.jsonl"
    done = {r["id"] for r in _read_jsonl(seen_path)}
    todo = candidates(items, done, now()).head(limit)
    stats = {"lidas": 0, "bloqueadas_robots": 0, "curtas_ou_paywall": 0, "erros": 0, "rotuladas": 0}
    fetched, texts, ids = [], [], []
    last_hit: dict[str, float] = {}
    for row in todo.itertuples():
        domain = urllib.parse.urlsplit(row.url).netloc
        record = {"id": row.id, "url": row.url, "fetched_at": str(now())}
        if not robots.allowed(row.url):
            stats["bloqueadas_robots"] += 1
            fetched.append({**record, "status": "robots"})
            continue
        wait = PER_DOMAIN_DELAY_S - (time.monotonic() - last_hit.get(domain, -1e9))
        if wait > 0:
            sleep(wait)
        last_hit[domain] = time.monotonic()
        try:
            body = extract(get(row.url))
        except Exception as exc:
            stats["erros"] += 1
            fetched.append({**record, "status": f"erro {type(exc).__name__}"})
            continue
        if len(body) < MIN_CHARS:
            stats["curtas_ou_paywall"] += 1
            fetched.append({**record, "status": "curta", "chars": len(body)})
            continue
        stats["lidas"] += 1
        fetched.append({**record, "status": "ok", "chars": len(body)})
        texts.append(f"{row.original_text}\n\n{body[:BODY_CHARS]}")
        ids.append(row.id)
    _append_jsonl(seen_path, fetched)

    labels = []
    for i in range(0, len(texts), enricher.batch_size):
        batch_texts, batch_ids = texts[i : i + enricher.batch_size], ids[i : i + enricher.batch_size]
        try:
            results = enricher._call(batch_texts)
        except Exception:
            stats["erros"] += 1
            continue
        stamp = str(now())
        for item_id, raw in zip(batch_ids, results):
            try:
                label = validate(raw or {})
            except (ValueError, TypeError):
                continue
            labels.append({"id": item_id, "labeled_at": stamp, "assets": label["assets"],
                           "sentiment": label["sentiment"], "event_type": label["event_type"],
                           "confidence": label["confidence"]})
    _append_jsonl(labels_path(data), labels)
    stats["rotuladas"] = len(labels)
    return stats


def apply_fulltext(labeled: pd.DataFrame, data: Path) -> pd.DataFrame:
    """Troca o rótulo do título pelo do texto completo e adia a visibilidade até o rótulo existir."""
    rows = _read_jsonl(labels_path(data))
    if not rows or labeled.empty:
        return labeled
    ft = pd.DataFrame(rows).drop_duplicates("id", keep="last").set_index("id")
    hit = labeled["id"].isin(ft.index)
    if not hit.any():
        return labeled
    out = labeled.copy()
    sub = ft.loc[out.loc[hit, "id"]]
    out.loc[hit, "sentiment"] = sub["sentiment"].to_numpy(float)
    out.loc[hit, "event_type"] = sub["event_type"].to_numpy()
    out.loc[hit, "confidence"] = sub["confidence"].to_numpy(float)
    out.loc[hit, "assets"] = [",".join(a) for a in sub["assets"]]
    out.loc[hit, "label_source"] = "llm_fulltext"
    labeled_at = pd.to_datetime(sub["labeled_at"].to_numpy(), utc=True)
    out.loc[hit, "visible_at"] = pd.Series(labeled_at, index=out.index[hit]).combine(
        out.loc[hit, "visible_at"], max)
    return out


def labels_stamp(data: Path) -> list:
    path = labels_path(data)
    return [path.stat().st_size] if path.exists() else [0]
