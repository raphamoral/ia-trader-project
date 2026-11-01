"""Livro de regras (playbook) destilado dos livros: regras e faixas de genes com citação.

Para cada tema × estilo, busca os melhores trechos no índice e pede ao Qwen regras
concretas e testáveis, apenas com o que os trechos dizem. Regras e valores que
aparecem em vários livros ganham apoio; o resultado vira lições permanentes e
faixas de partida dos genes (priors). A evolução continua livre para refutá-las.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from trader.books.index import Library
from trader.evolution.creator import GENES_DOC, _extract_json, _post_json
from trader.evolution.genome import STRATEGIES

QUESTIONS = [
    ("risk", "day_trade", "How much to risk per trade, where to place the stop loss, and a daily loss limit for day trading"),
    ("risk", "swing", "Position sizing, stop placement and maximum drawdown rules for swing trading"),
    ("entry", "swing", "Moving average crossover lengths and trend filters that define a valid long entry"),
    ("entry", "day_trade", "Breakout and pullback entry rules for intraday trading, including volume confirmation"),
    ("indicators", None, "Standard RSI period and overbought/oversold levels and how to use them for mean reversion"),
    ("indicators", None, "Donchian or channel breakout lookback lengths used by trend followers"),
    ("regime", None, "How to detect high volatility or ranging markets and when to stand aside"),
    ("exit", None, "Exit rules: trailing stops, time stops and profit targets"),
    ("news", None, "Trading around scheduled news and economic announcements"),
    ("psychology", None, "Overtrading, revenge trading and rules to stop trading for the day"),
    ("evidence", None, "Which trading rules survive rigorous out-of-sample testing and which are data-mining artifacts"),
]

PROMPT = """You extract a trading playbook ONLY from the book passages below. Do not use outside knowledge.
Question: {question}

Passages:
{passages}

Our agents have these genes:
{genes}

Return 2-6 concrete, testable rules. When a passage states a number that maps to one of our genes, give it
in "genes" as {{"gene_name": [low, high]}} using the exact gene names above (for strategy parameters use
"strategy.param", e.g. "ma_cross.fast"). Every rule must cite the passages it came from as "[book p.N]".
Answer JSON only: {{"rules": [{{"rule": "...", "genes": {{...}}, "citations": ["[book p.N]", ...]}}]}}"""


def _known_genes() -> dict[str, tuple[float, float]]:
    """Genes que existem no genoma, com a faixa completa do esquema (copiar o esquema não conta como livro)."""
    from trader.evolution import genome as G

    genes = {f"{st}.{k}": rng for st, params in STRATEGIES.items() for k, rng in params.items()}
    genes.update({"size": G.SIZE_RANGE, "vol_filter": G.VOL_FILTER_RANGE, "trend_filter": G.TREND_RANGE,
                  "cooldown": G.COOLDOWN_RANGE})
    return genes


def _numbers_in(text: str) -> set[float]:
    import re

    out = set()
    for m in re.findall(r"\d+(?:[.,]\d+)?", text):
        try:
            out.add(float(m.replace(",", ".")))
        except ValueError:
            pass
    return out


def grounded_genes(genes: dict[str, Any], passages_text: str) -> dict[str, list[float]]:
    """Mantém só genes existentes, cujos números aparecem nos trechos e que não são a faixa do esquema."""
    known, numbers, out = _known_genes(), _numbers_in(passages_text), {}
    for gene, rng in (genes or {}).items():
        if gene not in known:
            continue
        try:
            lo, hi = (float(rng[0]), float(rng[1])) if isinstance(rng, (list, tuple)) else (float(rng), float(rng))
        except (TypeError, ValueError, IndexError):
            continue
        if (min(lo, hi), max(lo, hi)) == tuple(map(float, known[gene])):
            continue  # a LLM repetiu a faixa do prompt
        pct = {x * 100 for x in (lo, hi) if x < 1}  # 0.02 no gene pode aparecer como "2%" no livro
        if not ({lo, hi} <= numbers or ({lo, hi} - numbers) <= pct & numbers or pct and pct <= numbers):
            continue  # número que não está no texto citado
        out[gene] = [min(lo, hi), max(lo, hi)]
    return out


def _norm(rule: str) -> str:
    import re

    return re.sub(r"\W+", " ", rule.lower()).strip()[:120]


def _book_of(citation: str) -> str:
    return citation.strip("[]").rsplit(" p.", 1)[0].strip()


def distill(
    library: Library,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen2.5:7b",
    post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
    k: int = 8,
    search_kwargs: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    genes_doc = GENES_DOC.format(strategies=json.dumps(STRATEGIES))
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, (topic, style, question) in enumerate(QUESTIONS, 1):
        log(f"[{i}/{len(QUESTIONS)}] {i / len(QUESTIONS):4.0%}  lendo os livros sobre: {topic}/{style or '-'}")
        hits = library.search(question, k=k, styles=[style] if style else None, topics=[topic], **(search_kwargs or {}))
        if not hits:
            continue
        passages = "\n".join(f"[{h['book']} p.{h['page']}] {h['text'][:700]}" for h in hits)
        found = None
        for attempt, n_passages in enumerate((k, 5, 3)):  # a cada falha: espera mais e manda menos texto
            text = "\n".join(f"[{h['book']} p.{h['page']}] {h['text'][:700]}" for h in hits[:n_passages])
            try:
                reply = post(url, {"model": model, "messages": [{"role": "user", "content": PROMPT.format(
                    question=question, passages=text, genes=genes_doc)}], "temperature": 0.2,
                    "response_format": {"type": "json_object"}}, 300.0)
                found = _extract_json(reply["choices"][0]["message"]["content"]).get("rules", [])
                break
            except Exception as exc:
                detail = ""
                if hasattr(exc, "code"):
                    try:
                        detail = f" {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"
                    except Exception:
                        detail = f" {exc.code}"
                log(f"{topic}/{style or '-'}: tentativa {attempt + 1} falhou ({type(exc).__name__}{detail})")
                time.sleep(10 * (attempt + 1))
        if found is None:
            continue
        valid_books = {h["book"] for h in hits}
        for r in found:
            cites = [c for c in r.get("citations", []) if _book_of(c) in valid_books]
            if not cites or not str(r.get("rule", "")).strip():
                continue  # regra sem citação verificável é descartada
            if _norm(str(r["rule"])) in seen:
                continue  # regra repetida
            seen.add(_norm(str(r["rule"])))
            cited_text = " ".join(h["text"] for h in hits if h["book"] in {_book_of(c) for c in cites})
            rules.append({"topic": topic, "style": style, "rule": str(r["rule"])[:300],
                          "genes": grounded_genes(r.get("genes") or {}, cited_text), "citations": cites,
                          "books": sorted({_book_of(c) for c in cites})})
        log(f"{topic}/{style or '-'}: {len(found)} regras propostas")
    return {"rules": rules, "priors": gene_priors(rules)}


def gene_priors(rules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Faixa de cada gene: mediana dos limites sugeridos, com o número de livros que apoiam."""
    lows, highs, books = defaultdict(list), defaultdict(list), defaultdict(set)
    for r in rules:
        for gene, rng in (r.get("genes") or {}).items():
            try:
                lo, hi = (float(rng[0]), float(rng[1])) if isinstance(rng, (list, tuple)) else (float(rng), float(rng))
            except (TypeError, ValueError, IndexError):
                continue
            lows[gene].append(min(lo, hi))
            highs[gene].append(max(lo, hi))
            books[gene].update(r["books"])
    return {g: {"low": statistics.median(lows[g]), "high": statistics.median(highs[g]), "books": len(books[g])}
            for g in lows}


def save(playbook: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(playbook, ensure_ascii=False, indent=1), encoding="utf-8")


def top_rules(playbook: dict[str, Any], n: int = 8) -> list[str]:
    """Regras com mais livros de apoio, prontas para entrar como lições (com citação)."""
    ranked = sorted(playbook.get("rules", []), key=lambda r: len(r["books"]), reverse=True)
    return [f"{r['rule']} {' '.join(r['citations'][:2])} (livros: {len(r['books'])})" for r in ranked[:n]]
