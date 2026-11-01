"""Armazenamento point-in-time de notícias.

Cada item guarda dois relógios:
  published_at  - horário informado pela fonte (UTC)
  first_seen_at - quando nós o capturamos (UTC)

Visibilidade para decisões (visible_at):
  - item capturado ao vivo: max(published_at + latência, first_seen_at);
    não podíamos ter lido antes de capturar.
  - item de backfill histórico: published_at + latência; o first_seen_at do
    backfill é hoje e não diz nada sobre o passado.

`news_until` é a única porta de leitura para decisões e nunca devolve item
com visible_at > as_of.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

from trader.news.sources import UNLISTED_TIER, tier_of
from trader.storage.atomic import replace_with_retry, unique_tmp

DEFAULT_LATENCY = pd.Timedelta(minutes=5)
# capturado muito depois da publicação = registro histórico, não observação ao vivo.
# Sem isso, itens velhos do RSS pareciam "chegar" horas depois e criavam falsas antecipações.
LIVE_MAX_DELAY = pd.Timedelta(minutes=30)

COLUMNS = [
    "id", "source", "domain", "tier", "url", "original_lang", "original_text",
    "published_at", "first_seen_at", "backfilled", "query_assets",
]


def normalize_url(url: str) -> str:
    url = url.strip().lower()
    url = re.sub(r"^https?://(www\.|m\.|amp\.)?", "", url)
    url = re.sub(r"[?#].*$", "", url)
    return url.rstrip("/").removesuffix("/amp")


def item_id(url: str, title: str) -> str:
    key = normalize_url(url) if url else re.sub(r"\W+", " ", title.lower()).strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def make_frame(rows: list[dict], keep_unlisted: bool = False) -> pd.DataFrame:
    """Normaliza itens brutos: id, nível da fonte, relógios UTC.

    Fontes fora da lista são descartadas, ou mantidas com tier=UNLISTED_TIER se keep_unlisted.
    """
    base = [c for c in COLUMNS if c not in ("id", "tier")]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    extras = [c for c in df.columns if c not in base]  # ex.: colunas gkg_* do GDELT
    df["tier"] = df["domain"].map(tier_of)
    if keep_unlisted:
        df["tier"] = df["tier"].fillna(UNLISTED_TIER)
    df = df[df["tier"].notna()].copy()
    df["tier"] = df["tier"].astype(int)
    df["id"] = [item_id(u, t) for u, t in zip(df["url"], df["original_text"])]
    for col in ("published_at", "first_seen_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["backfilled"] = mark_historical(df)
    return df[COLUMNS + extras]


def mark_historical(df: pd.DataFrame, max_delay: pd.Timedelta = LIVE_MAX_DELAY) -> pd.Series:
    """backfilled=True também para itens capturados mais de `max_delay` depois da publicação."""
    return df["backfilled"].astype(bool) | (df["first_seen_at"] - df["published_at"] > max_delay)


def visible_at(df: pd.DataFrame, latency: pd.Timedelta = DEFAULT_LATENCY) -> pd.Series:
    by_source = df["published_at"] + latency
    live = by_source.where(by_source >= df["first_seen_at"], df["first_seen_at"])
    return by_source.where(df["backfilled"], live)


class NewsStore:
    def __init__(self, root: Path | str, subdir: str = "items") -> None:
        self.root = Path(root) / "news" / subdir

    def write(self, items: pd.DataFrame) -> int:
        """Grava por mês de publicação. Dedup por id mantendo a captura mais antiga. Retorna novos."""
        if items.empty:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        added = 0
        for month, part in items.groupby(items["published_at"].dt.strftime("%Y-%m")):
            path = self.root / f"{month}.parquet"
            before = 0
            if path.exists():
                old = pd.read_parquet(path)
                before = len(old)
                part = pd.concat([old, part])
            # mesma notícia vinda de outra busca: junta os ativos, mantém a primeira captura
            assets = part.groupby("id")["query_assets"].agg(lambda s: ",".join(sorted(set(",".join(s).split(",")))))
            part = part.sort_values("first_seen_at").drop_duplicates("id", keep="first")
            part["query_assets"] = part["id"].map(assets)
            part = part.sort_values("published_at").reset_index(drop=True)
            tmp = unique_tmp(path)
            part.to_parquet(tmp, index=False)
            replace_with_retry(tmp, path)
            added += len(part) - before
        return added

    def fix_historical_flags(self) -> int:
        """Reaplica mark_historical aos arquivos gravados. Retorna itens corrigidos."""
        fixed = 0
        for path in sorted(self.root.glob("*.parquet")):
            df = pd.read_parquet(path)
            new = mark_historical(df)
            changed = int((new != df["backfilled"]).sum())
            if changed:
                df["backfilled"] = new
                tmp = unique_tmp(path)
                df.to_parquet(tmp, index=False)
                replace_with_retry(tmp, path)
                fixed += changed
        return fixed

    def read_range(self, since: pd.Timestamp | None = None, until: pd.Timestamp | None = None) -> pd.DataFrame:
        """Só as partições mensais que podem conter o período (o GDELT tem milhões de linhas)."""
        lo = None if since is None else pd.Timestamp(since).strftime("%Y-%m")
        hi = None if until is None else pd.Timestamp(until).strftime("%Y-%m")
        files = [f for f in sorted(self.root.glob("*.parquet"))
                 if (lo is None or f.stem >= lo) and (hi is None or f.stem <= hi)]
        if not files:
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def read_all(self) -> pd.DataFrame:
        files = sorted(self.root.glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def news_until(
        self,
        as_of: pd.Timestamp,
        assets: list[str] | None = None,
        lookback: pd.Timedelta = pd.Timedelta(days=1),
        latency: pd.Timedelta = DEFAULT_LATENCY,
    ) -> pd.DataFrame:
        """Itens visíveis em (as_of - lookback, as_of]. Nunca devolve o futuro."""
        if as_of.tzinfo is None:
            raise ValueError("as_of precisa de fuso (UTC)")
        lo = as_of - lookback
        months = pd.period_range(lo.tz_localize(None).to_period("M") - 1, as_of.tz_localize(None).to_period("M"), freq="M")
        files = [self.root / f"{m}.parquet" for m in months]
        frames = [pd.read_parquet(f) for f in files if f.exists()]
        if not frames:
            return pd.DataFrame(columns=[*COLUMNS, "visible_at"])
        df = pd.concat(frames, ignore_index=True)
        df["visible_at"] = visible_at(df, latency)
        df = df[(df["visible_at"] <= as_of) & (df["visible_at"] > lo)]
        if assets:
            wanted = set(assets)
            df = df[df["query_assets"].map(lambda s: bool(wanted & set(s.split(","))))]
        return df.sort_values("visible_at").reset_index(drop=True)
