"""Backfill histórico pelos arquivos brutos do GDELT GKG 2.x (sem API).

- Listas: masterfilelist.txt (inglês) e masterfilelist-translation.txt (65
  idiomas traduzidos pelo GDELT). Cada linha: tamanho, MD5, URL.
- Cada .zip é baixado para a memória, conferido pelo MD5, lido linha a linha,
  filtrado e descartado. Nenhum arquivo bruto é gravado em disco.
- Filtro: domínio na lista branca (nível 1/2) E conteúdo relevante (palavras,
  organizações e temas de cripto, macro Brasil/EUA, B3, petróleo, minério).
- published_at = DATE do GKG, que é quando o GDELT viu o artigo (varredura de
  15 min): conservador. visible_at = DATE + latência.
- Manifesto em SQLite: cada arquivo fica done/failed; retomar pula os done.
  Um arquivo só vira done depois que suas linhas estão gravadas.
- Educado com o servidor e com a conexão do usuário: até 2 downloads em
  paralelo e teto de banda total (token bucket).
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd

from trader.news.sources import tier_of
from trader.news.store import NewsStore, make_frame

BASE = "https://data.gdeltproject.org/gdeltv2/"
MASTER = {"en": BASE + "masterfilelist.txt", "translation": BASE + "masterfilelist-translation.txt"}
GKG_START = pd.Timestamp("2015-02-18", tz="UTC")
SUBDIR = "gdelt_gkg"

csv.field_size_limit(2**31 - 1)

# colunas do GKG 2.1
C_DATE, C_DOMAIN, C_URL, C_THEMES, C_V2THEMES = 1, 3, 4, 7, 8
C_PERSONS, C_ORGS, C_TONE, C_TRANSLATION, C_XML = 11, 13, 15, 25, 26

# grupos de relevância: ativo -> padrões (texto = título + URL + organizações + pessoas, minúsculo)
_KEYWORDS = {
    "BTC": r"bitcoin|\bbtc\b|crypto|cripto|binance|coinbase|ethereum|stablecoin|blockchain|比特币|ビットコイン|비트코인|暗号資産|가상자산|加密货币",
    "IBOV": r"ibovespa|bovespa|\bb3\b|brazil(ian)? stock|bolsa brasileira|ita[uú] unibanco|\bita[uú]\b|bradesco|banco do brasil",
    "PETR4": r"petrobras",
    "VALE3": r"\bvale s\.?a\b|vale sa\b|vale3|vale (iron|mining|min[eé]rio)|mineradora vale",
    "USDBRL": r"brazilian real|\breal brasileiro|d[oó]lar.{0,20}(real|brasil)|usd/?brl|c[aâ]mbio",
    "MACRO_BR": r"copom|\bselic\b|banco central do brasil|central bank of brazil|brazil('s)? central bank|\bipca\b|haddad|galípolo|galipolo",
    "MACRO_US": r"federal reserve|\bfomc\b|\bpowell\b|nonfarm|non-farm|payrolls?|consumer price index|\bcpi\b",
    "OIL": r"\bbrent\b|crude oil|oil price|\bopec\b|pre[cç]o do petr[oó]leo",
    "IRON": r"iron ore|min[eé]rio de ferro",
}
_KEYWORD_RE = {k: re.compile(v, re.I) for k, v in _KEYWORDS.items()}
_THEME_ASSETS = {"ECON_BITCOIN": "BTC", "ECON_CRYPTOCURRENCY": "BTC", "ECON_CRYPTO": "BTC"}
# grupo -> ativos afetados
_GROUP_ASSETS = {
    "BTC": ["BTC"], "IBOV": ["IBOV"], "PETR4": ["PETR4", "IBOV"], "VALE3": ["VALE3", "IBOV"],
    "USDBRL": ["USDBRL"], "MACRO_BR": ["IBOV", "USDBRL"], "MACRO_US": ["BTC", "IBOV", "USDBRL"],
    "OIL": ["PETR4"], "IRON": ["VALE3"],
}
_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_LANG3 = {"eng": "en", "por": "pt", "spa": "es", "deu": "de", "fra": "fr", "ita": "it", "zho": "zh",
          "jpn": "ja", "kor": "ko", "rus": "ru", "ara": "ar", "tur": "tr", "nld": "nl", "pol": "pl"}


@dataclass(frozen=True)
class RemoteFile:
    url: str
    size: int
    md5: str
    feed: str  # en | translation

    @property
    def stamp(self) -> str:
        """YYYYMMDDHHMMSS do nome do arquivo (ordenável como texto)."""
        return self.url.rsplit("/", 1)[1][:14]

    @property
    def ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.stamp).tz_localize("UTC")


# ------------------------------------------------------------------ banda


class Bandwidth:
    """Token bucket compartilhado entre as threads: limita bytes/s no total."""

    def __init__(self, bytes_per_second: float, clock=time.monotonic, sleep=time.sleep) -> None:
        self.rate = bytes_per_second
        self.clock, self.sleep = clock, sleep
        self.lock = threading.Lock()
        self.allowance = bytes_per_second
        self.last = clock()

    def take(self, n: int) -> None:
        with self.lock:
            now = self.clock()
            self.allowance = min(self.rate, self.allowance + (now - self.last) * self.rate)
            self.last = now
            self.allowance -= n
            deficit = -self.allowance
        if deficit > 0:
            self.sleep(deficit / self.rate)


def _download(url: str, bandwidth: Bandwidth, timeout: float = 120.0, chunk: int = 64 * 1024) -> bytes:
    req = urllib.request.Request(url.replace("http://", "https://"), headers={"User-Agent": "ai-trader-research/0.1"})
    buf = io.BytesIO()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            bandwidth.take(len(block))
            buf.write(block)
    return buf.getvalue()


# ------------------------------------------------------------------ listas


def iter_master(text_lines: Iterable[str], feed: str) -> Iterator[RemoteFile]:
    suffix = ".translation.gkg.csv.zip" if feed == "translation" else ".gkg.csv.zip"
    for line in text_lines:
        parts = line.split()
        if len(parts) != 3 or not parts[2].endswith(suffix):
            continue
        if feed == "en" and parts[2].endswith(".translation.gkg.csv.zip"):
            continue
        try:
            yield RemoteFile(url=parts[2], size=int(parts[0]), md5=parts[1], feed=feed)
        except ValueError:
            continue


def load_file_list(
    cache_dir: Path, feed: str, max_age: pd.Timedelta = pd.Timedelta(hours=12), bandwidth: Bandwidth | None = None
) -> list[RemoteFile]:
    """Lista de arquivos GKG do feed. A lista-mestre (~130 MB) é lida em streaming e só as
    linhas GKG são guardadas (em R:, pequena) para não baixar tudo de novo a cada retomada."""
    cache = cache_dir / f"files-{feed}.tsv"
    fresh = cache.exists() and pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s") < max_age
    if not fresh:
        cache_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(MASTER[feed], headers={"User-Agent": "ai-trader-research/0.1"})
        tmp = cache.with_suffix(".tmp")
        with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("w", encoding="utf-8") as out:
            def lines():
                while raw := resp.readline():
                    if bandwidth is not None:
                        bandwidth.take(len(raw))
                    yield raw.decode("utf-8", errors="replace")

            for f in iter_master(lines(), feed):
                out.write(f"{f.size}\t{f.md5}\t{f.url}\n")
        tmp.replace(cache)
    out = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        size, md5, url = line.split("\t")
        out.append(RemoteFile(url, int(size), md5, feed))
    return out


def plan(files: list[RemoteFile], since: pd.Timestamp, until: pd.Timestamp, then_earlier: bool) -> list[RemoteFile]:
    """since→until primeiro (em ordem), depois 2015-02→since, se pedido."""
    fmt = "%Y%m%d%H%M%S"
    lo, hi, start = since.strftime(fmt), until.strftime(fmt), GKG_START.strftime(fmt)
    first = sorted((f for f in files if lo <= f.stamp < hi), key=lambda f: (f.stamp, f.feed))
    if not then_earlier:
        return first
    earlier = sorted((f for f in files if start <= f.stamp < lo), key=lambda f: (f.stamp, f.feed))
    return first + earlier


# ------------------------------------------------------------------ filtro


def match_assets(text: str, themes: str) -> list[str]:
    groups = [g for g, rx in _KEYWORD_RE.items() if rx.search(text)]
    assets = {a for g in groups for a in _GROUP_ASSETS[g]}
    for theme, asset in _THEME_ASSETS.items():
        if theme in themes:
            assets.add(asset)
    return sorted(assets)


def parse_tone(field: str) -> tuple[float, float, int]:
    parts = field.split(",")
    try:
        return float(parts[0]), float(parts[3]), int(float(parts[6]))
    except (IndexError, ValueError):
        return 0.0, 0.0, 0


def filter_rows(lines: Iterable[str], feed: str, now: pd.Timestamp) -> list[dict]:
    """Linhas do GKG → itens do store (lista branca + relevância)."""
    rows = []
    for line in lines:
        r = line.rstrip("\r\n").split("\t")
        if len(r) < 27:
            continue
        domain = r[C_DOMAIN].lower()
        tier = tier_of(domain)
        if tier is None:
            continue
        m = _TITLE_RE.search(r[C_XML])
        title = " ".join(m.group(1).split()) if m else ""
        orgs, persons = r[C_ORGS], r[C_PERSONS]
        themes = r[C_THEMES]
        assets = match_assets(f"{title} {r[C_URL]} {orgs} {persons}", themes)
        if not assets:
            continue
        lang = "en"
        if feed == "translation":
            src = re.search(r"srclc:(\w+)", r[C_TRANSLATION])
            lang = _LANG3.get(src.group(1), src.group(1)) if src else "und"
        tone, polarity, words = parse_tone(r[C_TONE])
        rows.append(
            {
                "source": f"gdelt_gkg:{feed}",
                "domain": domain,
                "url": r[C_URL],
                "original_lang": lang,
                "original_text": title or r[C_URL],
                "published_at": pd.Timestamp(r[C_DATE][:14]).tz_localize("UTC"),
                "first_seen_at": now,
                "backfilled": True,
                "query_assets": ",".join(assets),
                "gkg_tone": tone,
                "gkg_polarity": polarity,
                "gkg_wordcount": words,
                "gkg_themes": ";".join(sorted({t.split(",")[0] for t in themes.split(";") if t}))[:2000],
                "gkg_orgs": orgs[:500],
                "gkg_translation": r[C_TRANSLATION],
            }
        )
    return rows


def process_bytes(blob: bytes, f: RemoteFile, now: pd.Timestamp) -> list[dict]:
    if hashlib.md5(blob).hexdigest() != f.md5:
        raise ValueError("MD5 não confere")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as raw:
            return filter_rows(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), f.feed, now)


# ------------------------------------------------------------------ manifesto


class Manifest:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS files (url TEXT PRIMARY KEY, status TEXT, bytes INTEGER,"
            " rows_kept INTEGER, error TEXT, finished_at TEXT)"
        )
        self.db.commit()

    def done(self) -> set[str]:
        return {u for (u,) in self.db.execute("SELECT url FROM files WHERE status = 'done'")}

    def mark(self, entries: list[tuple[str, str, int, int, str]]) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        self.db.executemany(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)", [(*e, now) for e in entries]
        )
        self.db.commit()

    def totals(self) -> dict[str, int]:
        rows = self.db.execute("SELECT status, COUNT(*), COALESCE(SUM(bytes),0), COALESCE(SUM(rows_kept),0) FROM files GROUP BY status")
        return {s: {"files": n, "bytes": b, "rows": k} for s, n, b, k in rows}


# ------------------------------------------------------------------ execução


def run(
    data: Path,
    files: list[RemoteFile],
    max_mbps: float = 4.0,
    workers: int = 2,
    flush_every: int = 48,
    log: Callable[[str], None] = print,
    download: Callable[[str, Bandwidth], bytes] = _download,
    retries: int = 4,
) -> dict:
    """Processa a lista (pula os já feitos). Retorna estatísticas desta execução."""
    root = data / "news" / SUBDIR
    manifest = Manifest(root / "manifest.sqlite")
    store = NewsStore(data, subdir=SUBDIR)
    done = manifest.done()
    todo = [f for f in files if f.url not in done]
    bandwidth = Bandwidth(max_mbps * 1024 * 1024)
    stats = {"files": 0, "failed": 0, "bytes": 0, "rows": 0, "start": time.monotonic(), "todo": len(todo)}
    todo_bytes = sum(f.size for f in todo)
    pending_rows: list[dict] = []
    pending_marks: list[tuple] = []
    now = pd.Timestamp.now(tz="UTC")

    def job(f: RemoteFile) -> tuple[RemoteFile, list[dict] | None, str]:
        delay = 5.0
        for attempt in range(retries):
            try:
                return f, process_bytes(download(f.url, bandwidth), f, now), ""
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, zipfile.BadZipFile, OSError) as exc:
                err = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                    return f, None, err
                time.sleep(delay * (attempt + 1))
        return f, None, err

    def flush() -> None:
        if pending_rows:
            frame = make_frame(pending_rows)
            store.write(frame)
        manifest.mark(pending_marks)
        pending_rows.clear()
        pending_marks.clear()

    def progress() -> None:
        el = time.monotonic() - stats["start"]
        rate = stats["bytes"] / el if el else 0
        left = todo_bytes - stats["bytes"]
        eta = pd.Timedelta(seconds=left / rate) if rate else None
        log(
            f"{stats['files']}/{len(todo)} arquivos, {stats['failed']} falhas, {stats['bytes'] / 1e9:.2f} GB,"
            f" {stats['rows']} linhas mantidas, {rate / 1e6:.2f} MB/s, ETA {str(eta).split('.')[0] if eta else '?'}"
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(todo)
        running = {pool.submit(job, f) for f in [next(it) for _ in range(min(workers * 2, len(todo)))]}
        since_flush = 0
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                f, rows, err = fut.result()
                if rows is None:
                    stats["failed"] += 1
                    pending_marks.append((f.url, "failed", 0, 0, err))
                else:
                    stats["files"] += 1
                    stats["bytes"] += f.size
                    stats["rows"] += len(rows)
                    pending_rows.extend(rows)
                    pending_marks.append((f.url, "done", f.size, len(rows), ""))
                since_flush += 1
                nxt = next(it, None)
                if nxt is not None:
                    running.add(pool.submit(job, nxt))
            if since_flush >= flush_every:
                flush()
                progress()
                since_flush = 0
        flush()
    progress()
    stats["elapsed"] = time.monotonic() - stats["start"]
    return stats
