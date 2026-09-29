"""Esquema dos itens e armazenamento point-in-time (data/news_black/).

Segue o contrato de `trader.news.store`: dois relógios (published_at,
first_seen_at), a flag `backfilled` e a mesma regra de visibilidade, importada
de lá (`trader.news.store.visible_at`). Acrescenta tier="black" e source_kind.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

TIER = "black"
SOURCE_KINDS = ("social", "blog", "farm")
DEFAULT_LATENCY = pd.Timedelta(minutes=5)

ITEM_COLUMNS = [
    "id",  # único por fonte: "<source_id>:<id nativo>"
    "source_id",  # conector (ex.: 4chan_biz, medium_tag_bitcoin)
    "source",  # emissor para reputação (domínio, autor ou quadro)
    "source_kind",
    "tier",
    "url",
    "title",
    "text",  # original
    "original_lang",
    "translated_text",  # na língua pivô (inglês)
    "translator",
    "translated_at",
    "translation_seconds",
    "published_at",  # carimbo do servidor da fonte (UTC)
    "first_seen_at",  # quando nós coletamos
    "backfilled",  # coletado depois do fato (histórico) ou ao vivo
    "visible_at",  # regra do trader.news: única hora que o backtest pode usar
    "ts_reliability",  # server | author | unknown
    "engagement",  # respostas, comentários (0 se desconhecido)
    "content_hash",
    "assets",  # lista separada por vírgula, ex.: "BTC,CRYPTO"
    "sentiment",  # -1..1
    "stance",  # bullish | bearish | neutral
    "event_type",
    "is_claim",  # afirma um fato verificável (não é só opinião)
    "confidence",
    "enriched_by",  # llm | lexicon | none
]

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    """Texto canônico para detectar duplicatas (sem links, números e pontuação)."""
    text = _URL.sub(" ", text.lower())
    text = _DIGITS.sub(" ", _NON_WORD.sub(" ", text))
    return _WS.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()[:16]


def new_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ITEM_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in ("published_at", "first_seen_at", "visible_at", "translated_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["engagement"] = pd.to_numeric(df["engagement"]).fillna(0).astype(int)
    df["backfilled"] = df["backfilled"].fillna(False).astype(bool)
    df["tier"] = TIER
    return df[ITEM_COLUMNS]


def with_visibility(df: pd.DataFrame, latency: pd.Timedelta = DEFAULT_LATENCY) -> pd.DataFrame:
    """Ao vivo: max(published_at + latência, first_seen_at). Histórico: published_at + latência."""
    from trader.news.store import visible_at

    df = df.copy()
    df["visible_at"] = visible_at(df, latency) if len(df) else df["published_at"]
    return df


class BlackStore:
    """Parquet mensal por published_at. Escritas idempotentes (dedup por id)."""

    def __init__(self, root: Path | str = Path("data") / "news_black") -> None:
        self.root = Path(root)
        self.items_dir = self.root / "items"

    def write(self, items: pd.DataFrame) -> int:
        if items.empty:
            return 0
        bad = set(items["source_kind"]) - set(SOURCE_KINDS)
        if bad:
            raise ValueError(f"source_kind inválido: {bad}")
        if (items["tier"] != TIER).any():
            raise ValueError("BlackStore só aceita tier='black'")
        self.items_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for month, part in items.groupby(items["published_at"].dt.strftime("%Y-%m")):
            path = self.items_dir / f"{month}.parquet"
            before = 0
            if path.exists():
                old = pd.read_parquet(path)
                before = len(old)
                part = pd.concat([old, part])
            part = part.drop_duplicates("id", keep="last").sort_values("published_at").reset_index(drop=True)
            added += len(part) - before
            tmp = path.with_suffix(".tmp")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        return added

    def all(self) -> pd.DataFrame:
        files = sorted(self.items_dir.glob("*.parquet"))
        if not files:
            return new_frame([])
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def replace(self, items: pd.DataFrame) -> None:
        """Regrava tudo (usado pelo enriquecimento, que atualiza colunas)."""
        for f in self.items_dir.glob("*.parquet"):
            f.unlink()
        self.write(items)


def utc64(series: pd.Series) -> "np.ndarray":
    """datetime64[ns] em UTC sem fuso (o .to_numpy() de série com fuso devolve objetos)."""
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).to_numpy()


def black_until(
    items: pd.DataFrame,
    as_of: pd.Timestamp,
    assets: list[str] | None = None,
    lookback: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Só o que já estava visível em `as_of`. Nunca devolve item do futuro."""
    mask = items["visible_at"] <= as_of
    if lookback is not None:
        mask &= items["visible_at"] > as_of - lookback
    out = items[mask]
    if assets:
        wanted = set(assets)
        out = out[out["assets"].fillna("").map(lambda s: bool(wanted & set(filter(None, s.split(",")))))]
    return out.reset_index(drop=True)
