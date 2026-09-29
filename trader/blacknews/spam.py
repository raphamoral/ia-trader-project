"""Heurísticas de spam e robôs: texto duplicado, rajadas de postagem e promoção."""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

from trader.blacknews.schema import normalize_text, utc64

_PROMO = re.compile(
    r"(100x|1000x|presale|pre-sale|airdrop|giveaway|join (now|us)|t\.me/|whatsapp|dm me|ref(erral)? ?code|"
    r"guaranteed|garantido|sinais vip|vip signals|pump group)",
    re.IGNORECASE,
)
_CASHTAG = re.compile(r"\$[A-Za-z]{2,10}\b")


def shingles(text: str, k: int = 5) -> set[int]:
    words = normalize_text(text).split()
    if len(words) < k:
        return {zlib.crc32(" ".join(words).encode())} if words else set()
    return {zlib.crc32(" ".join(words[i : i + k]).encode()) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def near_duplicate_groups(texts: list[str], threshold: float = 0.8, bands: int = 8, rows: int = 4) -> list[int]:
    """Id de grupo por texto (MinHash + LSH). Textos quase iguais ficam no mesmo grupo."""
    n_hash = bands * rows
    rng = np.random.default_rng(12345)
    a = rng.integers(1, 2**31 - 1, n_hash, dtype=np.int64)
    b = rng.integers(0, 2**31 - 1, n_hash, dtype=np.int64)
    prime = np.int64(2**31 - 1)
    sets = [shingles(t) for t in texts]
    sigs = []
    for s in sets:
        if not s:
            sigs.append(None)
            continue
        x = np.fromiter(s, dtype=np.int64)
        sigs.append(((np.outer(a, x) + b[:, None]) % prime).min(axis=1))
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for band in range(bands):
        buckets: dict[tuple, list[int]] = {}
        for i, sig in enumerate(sigs):
            if sig is not None:
                buckets.setdefault(tuple(sig[band * rows : (band + 1) * rows]), []).append(i)
        for members in buckets.values():
            for j in members[1:]:
                if find(j) != find(members[0]) and jaccard(sets[j], sets[members[0]]) >= threshold:
                    parent[find(j)] = find(members[0])
    return [find(i) for i in range(len(texts))]


def spam_scores(items: pd.DataFrame, burst_window: pd.Timedelta = pd.Timedelta(minutes=10), burst_k: int = 5) -> pd.DataFrame:
    """Acrescenta dup_group, dup_count, is_burst, promo, spam_score e is_spam.

    dup_count e is_burst olham só para trás no tempo (itens anteriores), então
    podem virar features sem vazamento.
    """
    df = items.sort_values("visible_at").reset_index(drop=True).copy()
    texts = df["text"].fillna("").tolist()
    df["dup_group"] = near_duplicate_groups(texts)
    df["dup_count"] = df.groupby("dup_group").cumcount()  # cópias anteriores do mesmo texto
    times = utc64(df["visible_at"])
    burst = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("source").groups.items():
        idx = np.asarray(sorted(idx))
        t = times[idx]
        left = np.searchsorted(t, t - np.timedelta64(int(burst_window.total_seconds()), "s"), side="left")
        burst[idx] = (np.arange(len(idx)) - left + 1) >= burst_k
    df["is_burst"] = burst
    df["promo"] = [bool(_PROMO.search(t)) or len(_CASHTAG.findall(t)) >= 4 for t in texts]
    caps = [sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t)) for t in texts]
    df["spam_score"] = (
        0.45 * (df["dup_count"] > 0) + 0.25 * df["is_burst"] + 0.35 * df["promo"] + 0.15 * (np.array(caps) > 0.6)
    ).clip(0, 1)
    df["is_spam"] = df["spam_score"] >= 0.45
    return df
