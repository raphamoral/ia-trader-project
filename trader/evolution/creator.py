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

    def child(self, genome: Genome, rng: random.Random) -> Genome: ...


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

    def child(self, genome: Genome, rng: random.Random) -> Genome:
        return mutate(genome, rng, 0.15)


PROMPT = """You design trading agents for a long-only crypto strategy on {symbol} ({timeframe} candles).
An agent died. Propose ONE successor genome that fixes the cause of death.

Allowed strategies and integer parameter ranges:
{strategies}
Other genes: size (0.1-1.0 fraction of max position), vol_filter (null or 0.002-0.2: skip entries when
average candle range as fraction of price exceeds it), trend_filter (null or 20-300: only buy above SMA(n)),
cooldown (0-60 candles between entries).

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

Answer with JSON only, no prose outside it:
{{"genome": {{"strategy": ..., "params": {{...}}, "size": ..., "vol_filter": ..., "trend_filter": ..., "cooldown": ...}},
  "why": "<one sentence>"}}"""


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
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
        fallback: Creator | None = None,
    ) -> None:
        self.symbol, self.timeframe = symbol, timeframe
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.graveyard = graveyard
        self.post = post
        self.fallback = fallback or RuleCreator()
        self.failures = 0

    def successor(self, genome: Genome, autopsy: dict[str, Any], rng: random.Random) -> tuple[Genome, str]:
        prompt = build_prompt(self.symbol, self.timeframe, autopsy, self.graveyard()[-10:])
        try:
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
            g, why = parse_answer(reply["choices"][0]["message"]["content"])
            return g, "LLM: " + why
        except Exception as exc:  # LLM fora do ar ou resposta inválida: regras assumem
            self.failures += 1
            g, why = self.fallback.successor(genome, autopsy, rng)
            return g, f"{why} (LLM falhou: {type(exc).__name__})"

    def child(self, genome: Genome, rng: random.Random) -> Genome:
        return self.fallback.child(genome, rng)  # reprodução não precisa de LLM


def build_prompt(symbol: str, timeframe: str, autopsy: dict[str, Any], graveyard: list[dict[str, Any]]) -> str:
    """Prompt do Criador. O treino da LLM usa exatamente este formato."""
    return PROMPT.format(
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
