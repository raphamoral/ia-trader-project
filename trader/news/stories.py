"""Deduplicação entre idiomas: a mesma história em 5 idiomas conta uma vez.

Agrupamento online, na ordem de visibilidade: cada item entra na história de um
item anterior (dentro de `window`) cujo título em inglês seja parecido, ou abre
uma história nova. Como cada decisão só olha itens já visíveis, o agrupamento é
causal: acrescentar notícias futuras não muda a história de um item passado.

Desempenho: índice invertido palavra -> histórias abertas; só histórias que
compartilham alguma palavra rara são comparadas (palavras presentes em muitas
histórias abertas, como "bitcoin", não geram candidatos sozinhas).
Itens de fontes fora da lista (tier=UNLISTED_TIER) não entram no agrupamento:
cada um é sua própria história.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from trader.news.sources import UNLISTED_TIER

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "as", "at", "by", "with",
    "from", "its", "it", "after", "over", "new", "says", "say", "will", "be", "has", "have", "this", "that",
}
COMMON_FRACTION = 0.05  # palavra em mais que 5% das histórias abertas não gera candidatos


def tokens(text: str | None) -> frozenset[str]:
    if not text:
        return frozenset()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in _STOP and len(w) > 1)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class StoryState:
    """Histórias abertas (para continuar o agrupamento de um mês para o outro)."""

    def __init__(self, window: pd.Timedelta, threshold: float) -> None:
        self.window, self.threshold = window, threshold
        self.last: dict[str, pd.Timestamp] = {}
        self.toks: dict[str, frozenset] = {}
        self.count: dict[str, int] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self._since_purge = 0

    def _purge(self, now: pd.Timestamp) -> None:
        dead = [s for s, t in self.last.items() if now - t > self.window]
        for s in dead:
            for w in self.toks.pop(s):
                self.postings[w].discard(s)
            del self.last[s]
        for w in [w for w, p in self.postings.items() if not p]:
            del self.postings[w]

    def add(self, item_id: str, t: pd.Timestamp, toks: frozenset, rank_hint: int | None = None) -> tuple[str, int]:
        self._since_purge += 1
        if self._since_purge >= 500:
            self._purge(t)
            self._since_purge = 0
        limit = max(20, COMMON_FRACTION * len(self.last))
        cands: set[str] = set()
        for w in toks:
            p = self.postings.get(w)
            if p and len(p) <= limit:
                cands |= p
        best, best_sim = None, self.threshold
        for s in cands:
            if t - self.last[s] > self.window:
                continue
            sim = jaccard(toks, self.toks[s])
            if sim >= best_sim:
                best, best_sim = s, sim
        sid = best or item_id
        if best is None:
            self.toks[sid] = toks
        else:
            grown = self.toks[sid] | toks
            for w in grown - self.toks[sid]:
                self.postings[w].add(sid)
            self.toks[sid] = grown
        for w in toks:
            self.postings[w].add(sid)
        self.last[sid] = t
        rank = self.count.get(sid, 0)
        self.count[sid] = rank + 1
        return sid, rank

    def seed(self, prior: pd.DataFrame) -> None:
        """Reabre histórias a partir de itens anteriores já agrupados (visible_at, story_id, texto)."""
        texts = prior["translated_text"].where(prior["translated_text"].notna(), prior["original_text"])
        for sid, t, text in zip(prior["story_id"], prior["visible_at"], texts):
            toks = tokens(text)
            self.toks[sid] = self.toks.get(sid, frozenset()) | toks
            for w in toks:
                self.postings[w].add(sid)
            self.last[sid] = max(t, self.last.get(sid, t))
            self.count[sid] = self.count.get(sid, 0) + 1


def assign_stories(
    items: pd.DataFrame,
    window: pd.Timedelta = pd.Timedelta(hours=24),
    threshold: float = 0.5,
    state: StoryState | None = None,
) -> pd.DataFrame:
    """Recebe itens com visible_at e translated_text; devolve com story_id e story_rank.

    story_rank = 0 no primeiro item da história, 1 no segundo, etc.
    Sem tradução (léxico/tom), usa o texto original: só agrupa dentro do mesmo idioma.
    `state` permite continuar de um lote anterior (ex.: mês anterior).
    """
    df = items.sort_values(["visible_at", "id"]).reset_index(drop=True)
    state = state or StoryState(window, threshold)
    texts = df["translated_text"].where(df["translated_text"].notna(), df["original_text"]).to_numpy()
    unlisted = (df["tier"] == UNLISTED_TIER).to_numpy() if "tier" in df.columns else [False] * len(df)
    story_ids, ranks = [], []
    for item_id, t, text, skip in zip(df["id"].to_numpy(), df["visible_at"], texts, unlisted):
        if skip:
            story_ids.append(item_id)
            ranks.append(0)
            continue
        sid, rank = state.add(item_id, t, tokens(text))
        story_ids.append(sid)
        ranks.append(rank)
    df["story_id"] = story_ids
    df["story_rank"] = ranks
    return df
