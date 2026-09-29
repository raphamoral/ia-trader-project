"""Tradução para o idioma-pivô (inglês) e rótulos por notícia, com cache em disco.

Uma chamada ao Ollama processa um lote de títulos e devolve, para cada um:
tradução em inglês, ativos afetados, sentimento (-1..1), tipo de evento e
confiança. Tudo é validado; resposta inválida ou LLM fora do ar cai no
léxico (só para inglês e português, sem tradução).

Por que inglês como pivô: o Qwen2.5 e os léxicos financeiros são mais fortes
em inglês, a maior parte das fontes de nível 1 já publica em inglês, e o
próprio GDELT indexa por tradução para o inglês.

O cache é por hash (modelo + versão do prompt + texto): reprocessar é grátis
e o resultado é determinístico para o backtest.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from trader.evolution.creator import _extract_json, _post_json

ASSETS = ("BTC", "IBOV", "PETR4", "VALE3", "USDBRL")
EVENT_TYPES = (
    "monetary_policy", "macro_data", "earnings", "regulation", "hack_security", "geopolitics",
    "commodities", "flows_etf", "corporate", "market_move", "other",
)
PROMPT_VERSION = "v1"
LABEL_COLUMNS = [
    "translated_text", "translator", "translated_at", "translation_seconds",
    "assets", "sentiment", "event_type", "confidence", "label_source",
]

PROMPT = """You label financial news headlines for trading. For each headline:
1. translate it to English (keep it unchanged if already English),
2. list affected assets among {assets} (BTC = bitcoin/crypto market, IBOV = Brazilian stock index,
   USDBRL = US dollar vs Brazilian real; use [] if none),
3. sentiment for those assets from -1 (very bearish) to 1 (very bullish),
4. event_type, one of {events},
5. confidence 0..1 in your labels.

Headlines:
{items}

Answer JSON only:
{{"items": [{{"i": 0, "en": "...", "assets": [...], "sentiment": 0.0, "event_type": "...", "confidence": 0.0}}, ...]}}"""

# léxico mínimo (en/pt) para quando a LLM não responde
_POS = {"surge", "rally", "gain", "rise", "record", "approve", "approval", "beat", "bullish", "soar", "jump",
        "alta", "sobe", "dispara", "recorde", "aprova", "lucro", "avança", "valoriza", "ganho"}
_NEG = {"fall", "drop", "plunge", "crash", "hack", "ban", "lawsuit", "loss", "bearish", "slump", "fraud", "cut",
        "queda", "cai", "despenca", "perda", "prejuízo", "fraude", "proíbe", "recua", "desvaloriza", "crise"}
_ASSET_WORDS = {
    "BTC": {"bitcoin", "btc", "crypto", "cripto", "criptomoeda", "cryptocurrency"},
    "IBOV": {"ibovespa", "ibov", "bolsa", "b3"},
    "PETR4": {"petrobras", "petr4"},
    "VALE3": {"vale", "vale3", "minério"},
    "USDBRL": {"dólar", "dollar", "câmbio", "real"},
}


def lexicon_label(text: str) -> dict[str, Any]:
    words = set(re.findall(r"\w+", text.lower()))
    pos, neg = len(words & _POS), len(words & _NEG)
    return {
        "assets": [a for a, keys in _ASSET_WORDS.items() if words & keys],
        "sentiment": 0.0 if pos == neg else (pos - neg) / (pos + neg),
        "event_type": "other",
        "confidence": 0.2,
    }


# temas do GKG -> tipo de evento (o primeiro que casar)
_THEME_EVENTS = [
    ("ECON_CENTRALBANK", "monetary_policy"), ("ECON_INTEREST_RATE", "monetary_policy"),
    ("ECON_INFLATION", "macro_data"), ("ECON_UNEMPLOYMENT", "macro_data"), ("ECON_GDP", "macro_data"),
    ("CYBER_ATTACK", "hack_security"), ("TAX_FNCACT_HACKER", "hack_security"),
    ("LEGISLATION", "regulation"), ("REGULATION", "regulation"), ("ECON_EARNINGSREPORT", "earnings"),
    ("ECON_OILPRICE", "commodities"), ("ENV_OIL", "commodities"), ("ENV_MINING", "commodities"),
    ("ARMEDCONFLICT", "geopolitics"), ("SANCTIONS", "geopolitics"), ("ECON_STOCKMARKET", "market_move"),
    ("ECON_CURRENCY_EXCHANGE_RATE", "market_move"),
]


def gkg_label(tone: float, themes: str, query_assets: str) -> dict[str, Any]:
    """Rótulo base para itens do GKG: tom do GDELT (calculado no texto inteiro, não só no título)."""
    event = next((e for t, e in _THEME_EVENTS if t in themes), "other")
    return {
        "assets": [a for a in query_assets.split(",") if a in ASSETS],
        "sentiment": max(-1.0, min(1.0, float(tone) / 5.0)),  # tom típico -10..10
        "event_type": event,
        "confidence": 0.3,
    }


def validate(raw: dict[str, Any]) -> dict[str, Any]:
    """Aceita só o formato esperado; valores fora da faixa são cortados ou recusados."""
    en = str(raw.get("en", "")).strip()
    if not en:
        raise ValueError("sem tradução")
    assets = [a for a in raw.get("assets", []) if a in ASSETS]
    event = raw.get("event_type")
    return {
        "en": en[:500],
        "assets": assets,
        "sentiment": max(-1.0, min(1.0, float(raw.get("sentiment", 0.0)))),
        "event_type": event if event in EVENT_TYPES else "other",
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
    }


class Enricher:
    def __init__(
        self,
        cache_dir: Path | str,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        batch_size: int = 8,
        timeout: float = 300.0,
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.post = post
        self.cache_path = Path(cache_dir) / "news" / "cache" / "labels.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry
        self.llm_calls = 0
        self.llm_failures = 0
        self.seconds_per_item: list[float] = []

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{PROMPT_VERSION}|{text}".encode()).hexdigest()[:24]

    def _save(self, entries: list[dict]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                self.cache[e["key"]] = e

    def _call(self, texts: list[str]) -> list[dict[str, Any]]:
        prompt = PROMPT.format(
            assets=list(ASSETS), events=list(EVENT_TYPES),
            items="\n".join(f"{i}. {t}" for i, t in enumerate(texts)),
        )
        reply = self.post(
            self.url,
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
             "response_format": {"type": "json_object"}},
            self.timeout,
        )
        items = _extract_json(reply["choices"][0]["message"]["content"])["items"]
        by_index = {int(x["i"]): x for x in items if "i" in x}
        return [by_index.get(i) for i in range(len(texts))]

    def enrich(self, items: pd.DataFrame, use_llm: bool = True, limit: int | None = None) -> pd.DataFrame:
        """Acrescenta as colunas LABEL_COLUMNS. `limit` = máximo de itens novos mandados à LLM."""
        texts = items["original_text"].tolist()
        pending = [t for t in dict.fromkeys(texts) if self.key(t) not in self.cache]
        budget = pending if limit is None else pending[:limit]
        if use_llm:
            for i in range(0, len(budget), self.batch_size):
                self._label_batch(budget[i : i + self.batch_size])
        out = []
        n = len(items)
        tones = items["gkg_tone"].to_numpy() if "gkg_tone" in items.columns else [None] * n
        themes = items["gkg_themes"].fillna("").to_numpy() if "gkg_themes" in items.columns else [""] * n
        queried = items["query_assets"].fillna("").to_numpy()
        for pos, (text, lang) in enumerate(zip(texts, items["original_lang"])):
            entry = self.cache.get(self.key(text))
            if entry is None:
                tone = tones[pos]
                if tone is not None and pd.notna(tone):
                    base, source = gkg_label(tone, themes[pos], queried[pos]), "gdelt_tone"
                else:
                    base, source = lexicon_label(text), "lexicon"
                entry = {
                    "en": text if lang == "en" else None, "translator": None, "translated_at": None,
                    "seconds": None, **base, "label_source": source,
                }
            out.append(
                {
                    "translated_text": entry["en"],
                    "translator": entry["translator"],
                    "translated_at": pd.Timestamp(entry["translated_at"]) if entry["translated_at"] else pd.NaT,
                    "translation_seconds": entry["seconds"],
                    "assets": ",".join(entry["assets"]),
                    "sentiment": entry["sentiment"],
                    "event_type": entry["event_type"],
                    "confidence": entry["confidence"],
                    "label_source": entry["label_source"],
                }
            )
        return pd.concat([items.reset_index(drop=True), pd.DataFrame(out, columns=LABEL_COLUMNS)], axis=1)

    def _label_batch(self, texts: list[str]) -> None:
        start = time.monotonic()
        self.llm_calls += 1
        try:
            results = self._call(texts)
        except Exception:
            self.llm_failures += 1
            return  # sem cache: o léxico cobre agora e a próxima execução tenta de novo
        per_item = (time.monotonic() - start) / len(texts)
        self.seconds_per_item.append(per_item)
        now = pd.Timestamp.now(tz="UTC").isoformat()
        entries = []
        for text, raw in zip(texts, results):
            try:
                label = validate(raw or {})
            except (ValueError, TypeError):
                continue
            entries.append(
                {"key": self.key(text), "text": text, "en": label["en"], "translator": f"ollama:{self.model}",
                 "translated_at": now, "seconds": round(per_item, 3), "assets": label["assets"],
                 "sentiment": label["sentiment"], "event_type": label["event_type"],
                 "confidence": label["confidence"], "label_source": "llm"}
            )
        self._save(entries)
