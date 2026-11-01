"""Busca agêntica com limite: a LLM planeja filtros e busca, com no máximo uma reformulação.

1. Planejamento (1 chamada): da situação (autópsia ou análise de sucesso) a LLM tira
   estilo, temas e uma busca curta em inglês.
2. Busca híbrida com esses filtros.
3. Se o melhor trecho for fraco, uma única reformulação (1 chamada) e nova busca.
Teto: 2 chamadas de planejamento por consulta, com cache em disco por situação.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from trader.books.index import STYLE_WORDS, TOPIC_WORDS, Library
from trader.evolution.creator import _extract_json, _post_json

PLAN = """You help a trading-agent designer find passages in trading books.
Situation:
{situation}
{previous}
Choose filters and ONE short English search query (5-12 words) for the most useful passage.
styles (0-2 of): {styles}
topics (1-3 of): {topics}
Answer JSON only: {{"styles": [...], "topics": [...], "query": "..."}}"""

MIN_GOOD_SCORE = 0.55


class AgenticLibrary:
    def __init__(
        self,
        library: Library,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
        cache_dir: Path | None = None,
        search_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.library = library
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.post = post
        self.cache = (cache_dir or library.root) / "agentic_cache.jsonl"
        self.search_kwargs = search_kwargs or {}
        self._memo = self._load_cache()

    def _load_cache(self) -> dict[str, list[dict]]:
        if not self.cache.exists():
            return {}
        memo = {}
        with open(self.cache, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                memo[row["key"]] = row["passages"]
        return memo

    def _plan(self, situation: str, previous: str = "") -> dict[str, Any]:
        prompt = PLAN.format(situation=situation, previous=previous,
                             styles=sorted(STYLE_WORDS), topics=sorted(TOPIC_WORDS))
        reply = self.post(self.url, {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                                     "temperature": 0.2, "response_format": {"type": "json_object"}}, 120.0)
        plan = _extract_json(reply["choices"][0]["message"]["content"])
        return {
            "styles": [s for s in plan.get("styles", []) if s in STYLE_WORDS][:2],
            "topics": [t for t in plan.get("topics", []) if t in TOPIC_WORDS][:3],
            "query": str(plan.get("query") or situation)[:200],
        }

    def __call__(self, situation: str) -> list[dict]:
        key = hashlib.sha1(situation.encode()).hexdigest()
        if key in self._memo:
            return self._memo[key]
        plan = self._plan(situation)
        found = self.library.search(plan["query"], styles=plan["styles"], topics=plan["topics"], **self.search_kwargs)
        if not found or found[0]["score"] < MIN_GOOD_SCORE:
            previous = f"A previous search '{plan['query']}' found weak passages; try different words or filters."
            retry = self._plan(situation, previous)
            again = self.library.search(retry["query"], styles=retry["styles"], topics=retry["topics"], **self.search_kwargs)
            if again and (not found or again[0]["score"] > found[0]["score"]):
                found = again
        passages = [{k: p[k] for k in ("book", "page", "text", "score")} for p in found]
        self._memo[key] = passages
        with open(self.cache, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "passages": passages}, ensure_ascii=False) + "\n")
        return passages
