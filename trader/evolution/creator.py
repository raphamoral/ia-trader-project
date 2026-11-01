"""O Criador: examina a autópsia de quem morreu e desenha o sucessor.

`RuleCreator` aplica correções dirigidas pela causa da morte.
`LLMCreator` pede o sucessor a uma LLM local por API compatível com OpenAI
(Ollama, LM Studio, llama.cpp server) e volta às regras se a resposta for
inválida ou a LLM estiver fora do ar.
"""
from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import replace
from typing import Any, Callable, Protocol

from trader.evolution.fitness import Cause
from trader.evolution.genome import STRATEGIES, Genome, mutate, random_genome


class Creator(Protocol):
    def successor(self, genome: Genome, autopsy: dict[str, Any], rng: random.Random) -> tuple[Genome, str]:
        """Retorna (genoma do sucessor, explicação da mudança)."""
        ...

    def child(self, genome: Genome, rng: random.Random, report: dict[str, Any] | None = None) -> tuple[Genome, str]:
        """Filho de quem prosperou; `report` = análise de sucesso do pai."""
        ...


class RuleCreator:
    def successor(self, genome: Genome, autopsy: dict[str, Any], rng: random.Random) -> tuple[Genome, str]:
        cause = Cause(autopsy["cause"])
        ctx = autopsy.get("last_window", {})
        match cause:
            case Cause.DRAWDOWN:
                notes = []
                g = genome
                if ctx.get("dd_high_vol"):
                    base = ctx.get("median_range") or 0.03
                    limit = base * 1.2 if g.vol_filter is None else g.vol_filter * 0.8
                    g = replace(g, vol_filter=limit)
                    notes.append("filtro de volatilidade")
                if ctx.get("dd_market_down"):
                    g = replace(g, trend_filter=100 if g.trend_filter is None else int(g.trend_filter * 1.3))
                    notes.append("filtro de tendência")
                if not notes:
                    g = replace(g, size=g.size * 0.7)
                    notes.append("posição menor")
                return mutate(g, rng, 0.05), "queda forte → " + ", ".join(notes)
            case Cause.FEES:
                g = replace(
                    genome,
                    params={k: v * 1.5 for k, v in genome.params.items()},
                    cooldown=genome.cooldown + 3,
                )
                return mutate(g, rng, 0.05), "taxa comeu o lucro → menos operações (janelas maiores, cooldown)"
            case Cause.NEGATIVE_EDGE:
                if rng.random() < 0.5:
                    others = sorted(set(STRATEGIES) - {genome.strategy})
                    return random_genome(rng, rng.choice(others)), "sem vantagem → troca de estratégia"
                return mutate(genome, rng, 0.5), "sem vantagem → mutação forte"
            case Cause.STAGNATION | Cause.CULLED:
                g = replace(genome, vol_filter=None, cooldown=0) if autopsy["life"]["trades"] == 0 else genome
                return mutate(g, rng, 0.4), "estagnou → mutação forte"
        raise AssertionError(cause)

    def child(self, genome: Genome, rng: random.Random, report: dict[str, Any] | None = None) -> tuple[Genome, str]:
        return mutate(genome, rng, 0.15), "mutação do pai"

    def successor_candidates(self, genome, autopsy, rng, n):
        return [self.successor(genome, autopsy, random.Random(rng.random())) for _ in range(n)]

    def child_candidates(self, genome, rng, report, n):
        return [self.child(genome, random.Random(rng.random()), report) for _ in range(n)]


GENES_DOC = """Allowed strategies and integer parameter ranges:
{strategies}
Other genes: size (0.1-1.0 fraction of max position), vol_filter (null or 0.002-0.2: skip entries when
average candle range as fraction of price exceeds it), trend_filter (null or 20-300: only buy above SMA(n)),
cooldown (0-60 candles between entries).
News genes (only act when news data is available; null/0 = off): news_gate (null or 0.05-0.9: no new buys when
combined news sentiment <= -news_gate), news_follow (null or 0.1-0.95: buy on strong positive news even without a
technical signal), event_blackout (0-240 minutes before scheduled macro events like FOMC/Copom/payroll with no new
entries), black_weight (0-1: weight of unverified social/blog sentiment in the combined sentiment), hack_blackout
(0-48 hours after a crypto security incident with no new entries)."""

ANSWER_DOC = """Answer with JSON only, no prose outside it:
{{"genome": {{"strategy": ..., "params": {{...}}, "size": ..., "vol_filter": ..., "trend_filter": ..., "cooldown": ...,
  "news_gate": ..., "news_follow": ..., "event_blackout": ..., "black_weight": ..., "hack_blackout": ...}},
  "why": "<one sentence>"}}"""

PROMPT = """You design trading agents for a long-only crypto strategy on {symbol} ({timeframe} candles).
An agent died. Propose ONE successor genome that fixes the cause of death.

""" + GENES_DOC + """

Cause codes: drawdown = lost too much in one window; fees = profitable before fees but fees ate it
(too many trades); negative_edge = loses even before fees; stagnation = old and never beat the benchmark.

Dead agent autopsy:
{autopsy}

Diagnosis in plain words:
{diagnosis}

Rules for your answer: change as little as possible to fix the diagnosis. Keep the same strategy
unless the cause is negative_edge. Never remove a filter that the diagnosis says was needed.

Recent deaths in the graveyard (avoid repeating these):
{graveyard}

Hall of fame: agents that won CONSISTENTLY, with the market context of their winning windows
(borrow what fits this situation):
{hall}

""" + ANSWER_DOC

CHILD_PROMPT = """You design trading agents for a long-only crypto strategy on {symbol} ({timeframe} candles).
An agent THRIVED: it beat the benchmark consistently. It earned a slot for a child. Propose ONE child genome.

""" + GENES_DOC + """

Success report of the parent (what it did, and the market context of the windows it won):
{report}

Why it worked, in plain words:
{summary}

Hall of fame (other consistent winners):
{hall}

Recent deaths (what failed; avoid it):
{graveyard}

Rules for your answer: KEEP the genes that explain the parent's success. Change at most one or two genes to
explore a variation that could make it work in more market conditions (for example regimes where it did not
trade or did worse). Never copy a genome from the graveyard.

""" + ANSWER_DOC


LESSONS_PROMPT = """You maintain a short playbook for evolving trading agents on {symbol} ({timeframe} candles).
Below are CONSISTENT WINNERS (hall of fame, with the market context where they won) and RECENT DEATHS (with causes).
Write 8-15 short, concrete lessons (one sentence each) about which genes and filters work in which market
conditions and what kills agents. Prefer lessons supported by several examples; do not invent numbers.

Hall of fame:
{hall}

Recent deaths:
{graveyard}

Answer JSON only: {{"lessons": ["...", "..."]}}"""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class LLMCreator:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:3b",
        timeout: float = 300.0,  # a primeira chamada inclui carregar o modelo
        graveyard: Callable[[], list[dict[str, Any]]] = list,
        hall: Callable[[], list[dict[str, Any]]] = list,
        library: Callable[[str], list[dict[str, Any]]] | None = None,
        lessons: Callable[[], list[str]] | None = None,
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
        fallback: Creator | None = None,
    ) -> None:
        self.symbol, self.timeframe = symbol, timeframe
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.graveyard = graveyard
        self.hall = hall
        self.library = library  # RAG dos livros: busca(texto) -> [{"book", "page", "text"}]
        self.lessons = lessons  # regras destiladas da galeria e do cemitério
        self.post = post
        self.fallback = fallback or RuleCreator()
        self.failures = 0

    def successor(self, genome: Genome, autopsy: dict[str, Any], rng: random.Random) -> tuple[Genome, str]:
        return self.successor_candidates(genome, autopsy, rng, 1)[0]

    def successor_candidates(
        self, genome: Genome, autopsy: dict[str, Any], rng: random.Random, n: int
    ) -> list[tuple[Genome, str]]:
        prompt = build_prompt(self.symbol, self.timeframe, autopsy, self.graveyard()[-10:], self.hall()[:5])
        prompt = self._with_extras(prompt, diagnose(autopsy), n)
        try:
            return [(g, "LLM: " + why) for g, why in self._ask_many(prompt, n)]
        except Exception as exc:  # LLM fora do ar ou resposta inválida: regras assumem
            self.failures += 1
            g, why = self.fallback.successor(genome, autopsy, rng)
            return [(g, f"{why} (LLM falhou: {type(exc).__name__})")]

    def child(self, genome: Genome, rng: random.Random, report: dict[str, Any] | None = None) -> tuple[Genome, str]:
        return self.child_candidates(genome, rng, report, 1)[0]

    def child_candidates(
        self, genome: Genome, rng: random.Random, report: dict[str, Any] | None, n: int
    ) -> list[tuple[Genome, str]]:
        if report is None:
            return [self.fallback.child(genome, rng)]
        prompt = CHILD_PROMPT.format(
            symbol=self.symbol, timeframe=self.timeframe, strategies=json.dumps(STRATEGIES),
            report=json.dumps(report, default=str), summary=summarize_success(report),
            hall=json.dumps(self.hall()[:5], default=str), graveyard=json.dumps(self.graveyard()[-10:], default=str),
        )
        prompt = self._with_extras(prompt, summarize_success(report), n)
        try:
            return [(g, "LLM (sucesso): " + why) for g, why in self._ask_many(prompt, n)]
        except Exception as exc:
            self.failures += 1
            g, why = self.fallback.child(genome, rng)
            return [(g, f"{why} (LLM falhou: {type(exc).__name__})")]

    def _with_extras(self, prompt: str, query: str, n: int) -> str:
        prompt = self._with_library(prompt, query)
        extra = ""
        lessons = self.lessons() if self.lessons else []
        if lessons:
            extra += "Lessons learned so far from winners and deaths (follow unless the evidence contradicts):\n"
            extra += "\n".join(f"- {x}" for x in lessons[:15]) + "\n\n"
        if n > 1:
            extra += (f"Propose {n} DIFFERENT candidate genomes (they will be backtested and only the best is kept).\n"
                      'Answer JSON only: {"candidates": [{"genome": {...}, "why": "..."}, ...]}\n\n')
        cut = prompt.rfind("Answer with JSON only")
        return prompt[:cut] + extra + prompt[cut:] if cut >= 0 else prompt + "\n\n" + extra

    def _with_library(self, prompt: str, query: str) -> str:
        """Acrescenta trechos dos livros (RAG) antes do formato da resposta, se houver biblioteca."""
        if self.library is None:
            return prompt
        try:
            passages = self.library(query)[:4]
        except Exception:
            return prompt  # biblioteca fora do ar não impede o Criador
        if not passages:
            return prompt
        refs = "\n".join(f"[{p.get('book', '?')} p.{p.get('page', '?')}] {str(p.get('text', ''))[:600]}" for p in passages)
        section = ("Reference passages from trading books (use them if relevant; cite as [book p.N] in why):\n"
                   + refs + "\n\n")
        cut = prompt.rfind("Answer with JSON only")
        return prompt[:cut] + section + prompt[cut:] if cut >= 0 else prompt + "\n\n" + section

    def _ask_many(self, prompt: str, n: int) -> list[tuple[Genome, str]]:
        text = self._complete(prompt)
        if n <= 1:
            return [parse_answer(text)]
        data = _extract_json(text)
        out = []
        for c in data.get("candidates", [data] if "genome" in data else []):
            try:
                out.append((Genome.from_dict(c["genome"]), str(c.get("why", ""))[:200]))
            except (KeyError, TypeError, ValueError):
                continue  # candidato inválido é descartado; os válidos seguem
        if not out:
            raise ValueError("nenhum candidato válido")
        return out[:n]

    def distill_lessons(self, hall: list[dict[str, Any]], graveyard: list[dict[str, Any]]) -> list[str]:
        """Resume galeria + cemitério em regras curtas (memória condensada para os próximos prompts)."""
        prompt = LESSONS_PROMPT.format(hall=json.dumps(hall, default=str), graveyard=json.dumps(graveyard, default=str),
                                       symbol=self.symbol, timeframe=self.timeframe)
        data = _extract_json(self._complete(prompt))
        return [str(x)[:220] for x in data.get("lessons", []) if str(x).strip()][:15]

    def _complete(self, prompt: str) -> str:
        reply = self.post(
            self.url,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "response_format": {"type": "json_object"},
            },
            self.timeout,
        )
        return reply["choices"][0]["message"]["content"]

    def _ask(self, prompt: str) -> tuple[Genome, str]:
        reply = self.post(
            self.url,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            },
            self.timeout,
        )
        return parse_answer(reply["choices"][0]["message"]["content"])


def build_prompt(
    symbol: str, timeframe: str, autopsy: dict[str, Any], graveyard: list[dict[str, Any]],
    hall: list[dict[str, Any]] | None = None,
) -> str:
    """Prompt do Criador. O treino da LLM usa exatamente este formato."""
    return PROMPT.format(
        hall=json.dumps(hall or [], default=str),
        symbol=symbol,
        timeframe=timeframe,
        strategies=json.dumps(STRATEGIES),
        autopsy=json.dumps(autopsy, default=str),
        diagnosis=diagnose(autopsy),
        graveyard=json.dumps(graveyard, default=str),
    )


def format_answer(genome: Genome, why: str) -> str:
    """Resposta no formato que o prompt pede (alvo do treino)."""
    return json.dumps({"genome": genome.to_dict(), "why": why})


def parse_answer(text: str) -> tuple[Genome, str]:
    answer = _extract_json(text)
    return Genome.from_dict(answer["genome"]), str(answer.get("why", ""))[:200]


def summarize_success(report: dict[str, Any]) -> str:
    """A análise de sucesso em frases diretas (o 'diagnose' dos acertos)."""
    life, ctx = report.get("life", {}), report.get("winning_context", {})
    lines = [
        f"- Over {life.get('age_windows', 0)} windows it made {life.get('net_return', 0):+.1%} net with worst drawdown"
        f" {life.get('worst_drawdown', 0):.1%} and {life.get('trades', 0)} trades; the benchmark made"
        f" {life.get('benchmark_return', 0):+.1%} in the same windows."
    ]
    if ctx:
        trend = "rising" if ctx.get("market_ret", 0) > 0.01 else "falling" if ctx.get("market_ret", 0) < -0.01 else "flat"
        lines.append(
            f"- In the windows it won, the market was {trend} ({ctx.get('market_ret', 0):+.1%} on average) with"
            f" typical candle range {ctx.get('median_range', 0):.2%}."
        )
        if ctx.get("news_sentiment") is not None:
            lines.append(f"- Average news sentiment in those windows: {ctx['news_sentiment']:+.2f}.")
    active = [k for k in ("news_gate", "news_follow", "event_blackout", "black_weight", "hack_blackout")
              if report.get("genome", {}).get(k)]
    if active:
        lines.append(f"- It uses news genes: {', '.join(active)}.")
    return "\n".join(lines)


def diagnose(autopsy: dict[str, Any]) -> str:
    """Traduz a autópsia em frases diretas; modelos pequenos seguem melhor assim."""
    life, last = autopsy.get("life", {}), autopsy.get("last_window", {})
    lines = []
    match autopsy["cause"]:
        case "drawdown":
            lines.append(f"- It lost {last.get('max_drawdown', 0):.0%} peak-to-trough in one window.")
            if last.get("dd_high_vol"):
                lines.append(
                    f"- The losses happened while volatility was above normal (typical candle range"
                    f" {last.get('median_range', 0):.2%}). It needs a vol_filter near that value or tighter."
                )
            if last.get("dd_market_down"):
                lines.append("- The losses happened while the asset itself fell more than 5%. It needs a trend_filter.")
            if not (last.get("dd_high_vol") or last.get("dd_market_down")):
                lines.append("- No clear market cause: reduce size.")
        case "fees":
            lines.append(
                f"- Before fees it made {life.get('gross_return', 0):+.1%}, after fees {life.get('net_return', 0):+.1%}"
                f" in {life.get('trades', 0)} trades. It trades too often: use longer periods and a cooldown."
            )
        case "negative_edge":
            lines.append("- It loses even before fees. The idea itself does not work here: a different strategy is allowed.")
        case "stagnation" | "culled":
            lines.append(
                f"- It never beat the benchmark ({life.get('benchmark_return', 0):+.1%}) with"
                f" {life.get('trades', 0)} trades. Make a meaningful change; if trades are near zero, loosen filters."
            )
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("resposta sem JSON")
    return json.loads(text[start : end + 1])
