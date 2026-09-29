"""Tradução e rótulos: usa o Enricher do agente confiável (`trader.news.enrich`).

Mesma língua pivô (inglês), mesmo prompt, mesmos ativos e tipos de evento, para
que as features black_ e as confiáveis sejam comparáveis. O cache fica em
data/news_black/news/cache/, separado do cache confiável.

Acrescenta o que é específico das fontes black:
- original_lang por heurística de escrita/palavras (o GDELT informa a língua;
  aqui não há quem informe);
- stance a partir do sentimento;
- is_claim: o post afirma um fato verificável? É isso que permite punir boatos
  que nunca se confirmam.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PIVOT_LANG = "en"

_STOP = {
    "en": {"the", "and", "is", "to", "of", "in", "for", "this", "that", "will", "price"},
    "pt": {"de", "que", "não", "para", "com", "uma", "os", "mais", "está", "vai", "preço", "dólar"},
    "es": {"de", "que", "el", "la", "los", "para", "con", "una", "más", "está", "precio", "por"},
    "de": {"der", "die", "und", "ist", "nicht", "mit", "für", "auf", "den", "das"},
    "fr": {"le", "la", "les", "est", "pour", "avec", "une", "des", "pas", "sur"},
    "it": {"il", "che", "per", "con", "una", "non", "gli", "della", "sono"},
}


def detect_lang(text: str) -> str:
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"


# afirmação verificável: anúncio, evento concreto, número, "confirmado", "vazou"
_CLAIM = re.compile(
    r"\b(breaking|confirmed|confirms|announc\w*|reportedly|sources say|leak\w*|insider|will (list|delist|launch|approve|ban)|"
    r"approved|rejected|hacked|exploit\w*|halt\w*|bankrupt\w*|sec |etf|lawsuit|arrest\w*|acquir\w*|merger|"
    r"confirmad\w*|anunci\w*|vazou|vazamento|aprovad\w*|hackead\w*|falênci\w*)\b",
    re.IGNORECASE,
)


def is_claim(text: str, event_type: str | None) -> bool:
    if event_type in ("market_move", "other") and not _CLAIM.search(text or ""):
        return False
    return bool(_CLAIM.search(text or "")) or event_type in ("regulation", "hack_security", "corporate", "flows_etf")


def enrich(items: pd.DataFrame, data_root: Path | str, use_llm: bool = True, limit: int | None = None, enricher=None):
    """Devolve (itens enriquecidos, enricher). `limit` = itens novos mandados à LLM nesta execução."""
    from trader.news.enrich import Enricher

    if items.empty:
        return items, enricher
    enricher = enricher or Enricher(cache_dir=Path(data_root))
    base = items.reset_index(drop=True).copy()
    base["original_lang"] = [
        lang if isinstance(lang, str) and lang else detect_lang(t) for lang, t in zip(base["original_lang"], base["text"])
    ]
    trusted_in = pd.DataFrame(
        {"original_text": base["text"].fillna(""), "original_lang": base["original_lang"], "query_assets": ""}
    )
    labeled = enricher.enrich(trusted_in, use_llm=use_llm, limit=limit)
    for col in ("translated_text", "translator", "translated_at", "translation_seconds", "assets", "sentiment",
                "event_type", "confidence"):
        base[col] = labeled[col].to_numpy()
    base["enriched_by"] = labeled["label_source"].replace({"gdelt_tone": "lexicon"}).to_numpy()
    base["translated_text"] = base["translated_text"].where(
        base["translated_text"].notna(), base["text"].where(base["original_lang"] == PIVOT_LANG)
    )
    base["sentiment"] = pd.to_numeric(base["sentiment"]).fillna(0.0)
    base["stance"] = pd.cut(base["sentiment"], [-1.01, -0.2, 0.2, 1.01], labels=["bearish", "neutral", "bullish"]).astype(str)
    base["is_claim"] = [is_claim(t, e) for t, e in zip(base["translated_text"].fillna(base["text"]), base["event_type"])]
    return base, enricher
