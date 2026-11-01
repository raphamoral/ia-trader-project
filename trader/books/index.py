"""Índice dos livros de trading para o RAG do Criador.

Indexação: texto por página (PDF/EPUB/MOBI via PyMuPDF) → trechos com página →
etiquetas (estilo pelo título do livro, tema por vocabulário) → embeddings
locais (bge-m3 no Ollama) + índice de palavras (BM25). Incremental por hash.

Busca híbrida: similaridade de vetores + BM25, só nos trechos com as etiquetas
pedidas. A busca agêntica (planejamento pela LLM) fica em agentic.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

EXTENSIONS = {".pdf", ".epub", ".mobi", ".azw3"}
CHUNK_CHARS = 1200
OVERLAP = 200
EMBED_MODEL = "bge-m3"

STYLE_WORDS = {
    "scalp": ["scalp", "tape reading", "order flow", "tick"],
    "day_trade": ["day trad", "intraday", "day-trad"],
    "swing": ["swing"],
    "long_term": ["investor", "invest", "value", "buffett", "dividend", "portfolio"],
}
TOPIC_WORDS = {
    "risk": ["stop loss", "stop-loss", "position size", "position sizing", "risk", "drawdown", "money management", "kelly"],
    "entry": ["entry", "breakout", "pullback", "setup", "trigger", "buy signal", "long entry"],
    "exit": ["exit", "take profit", "target", "trailing stop", "time stop"],
    "regime": ["volatility", "trend", "range-bound", "ranging", "choppy", "regime", "atr"],
    "psychology": ["discipline", "fear", "greed", "emotion", "psycholog", "patience", "overtrading", "tilt"],
    "patterns": ["candlestick", "pattern", "engulfing", "doji", "hammer", "head and shoulders", "triangle", "flag"],
    "indicators": ["moving average", "rsi", "macd", "bollinger", "stochastic", "vwap", "fibonacci", "pivot"],
    "news": ["news", "earnings", "fomc", "federal reserve", "announcement", "economic calendar"],
    "evidence": ["backtest", "statistical", "significance", "data mining", "overfitting", "out-of-sample", "hypothesis"],
}


def _tags(text: str, vocab: dict[str, list[str]], min_hits: int = 1) -> list[str]:
    low = text.lower()
    return [tag for tag, words in vocab.items() if sum(low.count(w) for w in words) >= min_hits]


def book_title(path: Path) -> str:
    title, prev = path.stem, None
    while title != prev:  # "X (1) - annotated" → "X"
        prev, title = title, re.sub(r"\s*(\(\d+\)|- annotated|\.pdf)\s*$", "", title, flags=re.I).strip()
    return title


def pick_editions(paths: list[Path]) -> list[Path]:
    """Um arquivo por título: mesmos bytes ou mesmo título em outro formato contam uma vez (PDF preferido)."""
    rank = {".pdf": 0, ".epub": 1, ".azw3": 2, ".mobi": 3}
    by_title: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_title[book_title(p).lower()].append(p)
    return [sorted(ps, key=lambda p: (rank.get(p.suffix.lower(), 9), -p.stat().st_size))[0] for ps in by_title.values()]


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


TESSDATA = os.environ.get("TESSDATA_PREFIX", r"R:\Tesseract-OCR\tessdata")
SCANNED_CHARS = 200  # menos texto que isso nas primeiras 20 páginas = PDF escaneado (só imagem)


def pages_of(path: Path, log: Callable[[str], None] = lambda m: None) -> list[tuple[int, str]]:
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)  # ebooks com fontes ausentes geram avisos inofensivos em massa
    with pymupdf.open(path) as doc:
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
        if sum(len(t) for _, t in pages[:20]) >= SCANNED_CHARS or not Path(TESSDATA).exists():
            return pages
        # escaneado: reconhece o texto das imagens (OCR com o Tesseract embutido no PyMuPDF)
        log(f"  {path.name}: escaneado, fazendo OCR de {len(doc)} páginas...")
        out = []
        for i, page in enumerate(doc):
            try:
                tp = page.get_textpage_ocr(language="eng", dpi=200, full=True, tessdata=TESSDATA)
                out.append((i + 1, page.get_text("text", textpage=tp)))
            except Exception:
                out.append((i + 1, ""))
            if (i + 1) % 50 == 0:
                log(f"    OCR {i + 1}/{len(doc)} páginas")
        return out


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    """Remove o que PDFs mal convertidos trazem e o Ollama recusa (surrogates soltos, controles)."""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return _CONTROL.sub(" ", text)


def chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Trechos de ~CHUNK_CHARS com sobreposição, cada um com a página onde começa."""
    out = []
    for page, text in pages:
        text = re.sub(r"\s+", " ", clean(text)).strip()
        start = 0
        while start < len(text):
            piece = text[start : start + CHUNK_CHARS]
            if len(piece) >= 200:
                out.append((page, piece))
            start += CHUNK_CHARS - OVERLAP
    return out


def ollama_embed(texts: list[str], url: str = "http://localhost:11434", model: str = EMBED_MODEL) -> np.ndarray:
    req = urllib.request.Request(f"{url}/api/embed", data=json.dumps({"model": model, "input": texts}).encode())
    with urllib.request.urlopen(req, timeout=600) as resp:
        vecs = np.asarray(json.loads(resp.read())["embeddings"], dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


_TOKEN = re.compile(r"[a-zà-ú0-9]{3,}")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Library:
    root: Path  # pasta do índice
    chunks: pd.DataFrame
    vectors: np.ndarray  # float16, normalizados
    df: Counter  # frequência de documentos por palavra (BM25)
    avg_len: float

    @classmethod
    def load(cls, root: Path | str) -> "Library":
        root = Path(root)
        chunks = pd.read_parquet(root / "chunks.parquet")
        vectors = np.load(root / "vectors.npy")
        df = Counter()
        lens = []
        for toks in chunks["tokens"]:
            df.update(set(toks))
            lens.append(len(toks))
        return cls(root, chunks, vectors, df, float(np.mean(lens)) if lens else 1.0)

    def _bm25(self, query: list[str], idx: np.ndarray, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        n = len(self.chunks)
        idf = {t: math.log(1 + (n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5)) for t in set(query)}
        scores = np.zeros(len(idx))
        for j, i in enumerate(idx):
            toks = self.chunks.at[i, "tokens"]
            tf = Counter(toks)
            norm = k1 * (1 - b + b * len(toks) / self.avg_len)
            scores[j] = sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in idf if tf[t])
        return scores

    def search(
        self, query: str, k: int = 4, styles: list[str] | None = None, topics: list[str] | None = None,
        embed: Callable[[list[str]], np.ndarray] = ollama_embed, alpha: float = 0.6,
    ) -> list[dict]:
        """Busca híbrida (vetor + palavra-chave) restrita às etiquetas pedidas."""
        mask = np.ones(len(self.chunks), dtype=bool)
        if styles:
            mask &= self.chunks["styles"].apply(lambda s: bool(set(s) & set(styles)) or len(s) == 0).to_numpy()
        if topics:
            mask &= self.chunks["topics"].apply(lambda t: bool(set(t) & set(topics))).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            idx = np.arange(len(self.chunks))
        qv = embed([query])[0].astype(np.float32)
        cos = self.vectors[idx].astype(np.float32) @ qv
        # só os 300 melhores por vetor entram no BM25 (rápido e suficiente)
        top = np.argsort(-cos)[:300]
        bm = self._bm25(tokens(query), idx[top])
        z = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        score = alpha * z(cos[top]) + (1 - alpha) * z(bm)
        best = np.argsort(-score)[:k]
        rows = self.chunks.iloc[idx[top][best]]
        return [
            {"book": r.book, "page": int(r.page), "text": r.text, "score": round(float(score[j]), 3),
             "styles": list(r.styles), "topics": list(r.topics)}
            for j, r in zip(best, rows.itertuples())
        ]


def embed_rows(rows: list[dict], embed, batch: int, log) -> tuple[list[dict], np.ndarray | None]:
    """Embeddings em lotes; se um lote for recusado, tenta um a um e pula só o trecho ruim."""
    kept, vecs = [], []
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        try:
            vecs.append(embed([r["text"] for r in part]).astype(np.float16))
            kept += part
        except Exception:
            for r in part:
                try:
                    vecs.append(embed([r["text"]]).astype(np.float16))
                    kept.append(r)
                except Exception as exc:
                    log(f"  trecho pulado (p.{r['page']}): {type(exc).__name__}")
    return kept, (np.vstack(vecs) if vecs else None)


def build(
    source: Path, out: Path, embed: Callable[[list[str]], np.ndarray] = ollama_embed, batch: int = 32,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Indexa (incremental) todos os livros em `source`. Já indexados (mesmo hash) são pulados."""
    out.mkdir(parents=True, exist_ok=True)
    chunks_path, vec_path, done_path = out / "chunks.parquet", out / "vectors.npy", out / "books.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    old_chunks = pd.read_parquet(chunks_path) if chunks_path.exists() else None
    old_vecs = np.load(vec_path) if vec_path.exists() else None
    files = pick_editions([p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS])
    stats = {"arquivos": len(files), "novos": 0, "falhas": 0, "trechos": 0}
    new_rows, new_vecs = [], []

    def flush() -> None:
        """Grava o que já foi feito (a indexação pode ser interrompida e retomada)."""
        nonlocal old_chunks, old_vecs, new_rows, new_vecs
        if not new_rows:
            return
        chunks, vecs = pd.DataFrame(new_rows), np.vstack(new_vecs)
        if old_chunks is not None:
            chunks, vecs = pd.concat([old_chunks, chunks], ignore_index=True), np.vstack([old_vecs, vecs])
        chunks.to_parquet(chunks_path, index=False)
        np.save(vec_path, vecs)
        done_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        old_chunks, old_vecs, new_rows, new_vecs = chunks, vecs, [], []

    import time

    todo = [p for p in files if file_hash(p) not in done]
    started = time.monotonic()
    log(f"{len(files)} livros na pasta, {len(files) - len(todo)} já indexados, {len(todo)} a indexar")
    for n, path in enumerate(todo, 1):
        digest = file_hash(path)
        title = book_title(path)
        try:
            pieces = chunk_pages(pages_of(path, log))
        except Exception as exc:
            stats["falhas"] += 1
            log(f"falhou {path.name}: {type(exc).__name__}")
            continue
        styles = _tags(title, STYLE_WORDS)
        rows = [{"book": title, "page": page, "text": text, "styles": styles,
                 "topics": _tags(text, TOPIC_WORDS, min_hits=2), "tokens": tokens(text)} for page, text in pieces]
        kept, vecs = embed_rows(rows, embed, batch, log)
        if not kept:
            stats["falhas"] += 1
            continue
        new_vecs.append(vecs)
        rows = kept
        new_rows += rows
        done[digest] = {"title": title, "file": path.name, "chunks": len(rows)}
        stats["novos"] += 1
        stats["trechos"] += len(rows)
        elapsed = time.monotonic() - started
        eta = elapsed / n * (len(todo) - n)
        log(f"[{n}/{len(todo)}] {n / len(todo):4.0%}  {title}: {len(rows)} trechos"
            f"  | faltam ~{eta / 60:.0f} min")
        if stats["novos"] % 5 == 0:
            flush()
    flush()
    return stats
