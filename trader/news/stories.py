"""Deduplicação entre idiomas: a mesma história em 5 idiomas conta uma vez.

Agrupamento online, na ordem de visibilidade: cada item entra na história de um
item anterior (dentro de `window`) cujo título em inglês seja parecido, ou abre
uma história nova. Como cada decisão só olha itens já visíveis, o agrupamento é
causal: acrescentar notícias futuras não muda a história de um item passado.
"""
from __future__ import annotations

import re

import pandas as pd

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "as", "at", "by", "with",
    "from", "its", "it", "after", "over", "new", "says", "say", "will", "be", "has", "have", "this", "that",
}


def tokens(text: str | None) -> frozenset[str]:
    if not text:
        return frozenset()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in _STOP and len(w) > 1)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def assign_stories(
    items: pd.DataFrame, window: pd.Timedelta = pd.Timedelta(hours=24), threshold: float = 0.5
) -> pd.DataFrame:
    """Recebe itens com visible_at e translated_text; devolve com story_id e story_rank.

    story_rank = 0 no primeiro item da história, 1 no segundo, etc.
    Sem tradução (léxico), usa o texto original: só agrupa dentro do mesmo idioma.
    """
    df = items.sort_values(["visible_at", "id"]).reset_index(drop=True)
    texts = df["translated_text"].where(df["translated_text"].notna(), df["original_text"])
    toks = [tokens(t) for t in texts]
    story_ids, ranks = [], []
    open_stories: list[tuple[pd.Timestamp, frozenset, str]] = []  # (último visto, tokens, id)
    counts: dict[str, int] = {}
    for i, row in df.iterrows():
        t = row["visible_at"]
        open_stories = [s for s in open_stories if t - s[0] <= window]
        best, best_sim = None, threshold
        for j, (_, stoks, sid) in enumerate(open_stories):
            sim = jaccard(toks[i], stoks)
            if sim >= best_sim:
                best, best_sim = j, sim
        if best is None:
            sid = row["id"]
            open_stories.append((t, toks[i], sid))
        else:
            _, stoks, sid = open_stories[best]
            open_stories[best] = (t, stoks | toks[i], sid)
        story_ids.append(sid)
        ranks.append(counts.get(sid, 0))
        counts[sid] = counts.get(sid, 0) + 1
    df["story_id"] = story_ids
    df["story_rank"] = ranks
    return df
