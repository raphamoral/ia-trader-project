"""Criadores do day trade: por regras e por LLM (API compatível com OpenAI).

O DayLLMCreator reaproveita o transporte e o parser do Criador principal
(trader.evolution.creator) e volta às regras se a LLM falhar.
"""
from __future__ import annotations

import json
import random
from dataclasses import replace
from typing import Any, Callable

from trader.daytrade.genome import (
    CATALOG,
    STAND_ASIDE,
    STYLE_DESCRIPTIONS,
    STYLES,
    eligible,
    switch_style,
    DAILY_STOP_RANGE,
    DAILY_TARGET_RANGE,
    DAY_STRATEGIES,
    MAX_HOLD_RANGE,
    MAX_TRADES_RANGE,
    DayGenome,
    mutate_day,
    random_day_genome,
)
from trader.daytrade.regime import REGIME_DESCRIPTIONS, REGIMES
from trader.evolution.creator import _extract_json, _post_json


def _worst_regime(life: dict[str, Any]) -> str | None:
    pnl = life.get("pnl_by_regime") or {}
    if not pnl:
        return None
    reg, val = min(pnl.items(), key=lambda kv: kv[1])
    return reg if val < 0 else None


class DayRuleCreator:
    styles: list[str] = ["any"]
    black_allowed: bool = True

    def _mut(self, g: DayGenome, rng: random.Random, scale: float) -> DayGenome:
        return mutate_day(g, rng, scale, styles=self.styles, black_allowed=self.black_allowed)

    def successor(self, g: DayGenome, autopsy: dict[str, Any], rng: random.Random) -> tuple[DayGenome, str]:
        new, why = self._successor(g, autopsy, rng)
        # saber quando: o regime em que mais perdeu passa a ficar de fora
        worst = _worst_regime(autopsy.get("life", {}))
        if worst and autopsy["cause"] not in ("culled",) and rng.random() < 0.7:
            base = new.regime_map or {r: new.strategy for r in REGIMES}
            new = replace(new, regime_map={**base, worst: STAND_ASIDE}).normalized()
            why += f"; perdia mais em regime {worst} → fica de fora nele"
        if not self.black_allowed and new.black_weight:
            new = replace(new, black_weight=0.0)
        return new, why

    def _successor(self, g: DayGenome, autopsy: dict[str, Any], rng: random.Random) -> tuple[DayGenome, str]:
        life = autopsy.get("life", {})
        stop_share = life.get("stop_day_share", 0.0)
        fee_share = life.get("fee_share_of_gross")
        if autopsy["cause"] == "fees" and g.style == "scalping" and len(self.styles) > 1:
            other = rng.choice([s for s in self.styles if s != "scalping"])
            return switch_style(g, other, rng), f"scalping pagando taxa demais → estilo {other}"
        if autopsy["cause"] == "fees" and fee_share and fee_share > 0.8 and len(self.styles) > 1 and rng.random() < 0.3:
            other = rng.choice([s for s in self.styles if s != g.style])
            return switch_style(g, other, rng), f"taxa levou {fee_share:.0%} do bruto → estilo {other}"
        match autopsy["cause"]:
            case "liquidation":
                fix = replace(g, leverage=max(1.0, g.leverage * 0.5), stop=g.stop * 0.8)
                return self._mut(fix, rng, 0.05), "liquidado → metade da alavancagem e stop mais curto"
            case "ruin":
                fix = replace(g, risk=g.risk * 0.5, daily_stop=g.daily_stop * 0.6,
                              max_trades_per_day=max(1, g.max_trades_per_day // 2))
                return self._mut(fix, rng, 0.05), "arruinado → metade do risco, stop diário mais curto, menos tentativas"
            case "drawdown":
                fix = replace(g, risk=g.risk * 0.6, daily_stop=g.daily_stop * 0.8)
                note = "queda forte → menos risco por operação e stop diário mais curto"
                if autopsy.get("last_window", {}).get("dd_high_vol"):
                    fix = replace(fix, stop=g.stop * 1.3, cooldown=g.cooldown + 4)
                    note += "; volatilidade alta → stop mais largo e pausa entre entradas"
                return self._mut(fix, rng, 0.05), note
            case "fees":
                p = {k: v * 1.5 for k, v in g.params.items()}
                fix = replace(g, params=p, cooldown=g.cooldown + 6, max_trades_per_day=max(1, g.max_trades_per_day // 2))
                return self._mut(fix, rng, 0.05), "taxa comeu o lucro → sinais mais exigentes, menos tentativas"
            case "no_goal":
                fix = replace(g, risk=g.risk * 1.5, leverage=g.leverage * 1.3,
                              daily_target=1.0 if g.daily_target is None else g.daily_target,
                              max_trades_per_day=g.max_trades_per_day + 2)
                return self._mut(fix, rng, 0.3), "nunca dobrou num dia → mais risco, alavancagem e tentativas, meta 100%"
            case "negative_edge":
                if stop_share > 0.3:
                    fix = replace(g, daily_stop=g.daily_stop * 0.7, max_trades_per_day=max(1, g.max_trades_per_day - 2))
                    return self._mut(fix, rng, 0.2), "bate o stop diário com frequência → menos tentativas"
                if rng.random() < 0.5:
                    other = rng.choice(sorted(set(DAY_STRATEGIES) - {g.strategy}))
                    if other not in eligible(g.style):
                        other = rng.choice(eligible(g.style))
                    fresh = random_day_genome(rng, other, style=g.style, black=self.black_allowed)
                    return replace(fresh, regime_map=g.regime_map).normalized(), "sem vantagem → troca de estratégia"
                flip = {"long": "short", "short": "long", "both": rng.choice(["long", "short"])}[g.direction]
                return self._mut(replace(g, direction=flip), rng, 0.3), f"sem vantagem → direção {flip}"
            case _:
                return self._mut(g, rng, 0.4), "estagnou → mutação forte"

    def child(self, g: DayGenome, rng: random.Random) -> DayGenome:
        return self._mut(g, rng, 0.15)


PROMPT = """You design intraday leveraged futures agents for {symbol} ({timeframe} candles, Binance USDT-M).
Capital starts at {capital}. Frozen risk limits you cannot exceed: leverage <= {max_lev}x, risk per trade <= {max_risk},
daily loss <= {daily_loss}. Minimum order ≈ {min_order} USDT. The owner's goal: {goal}.

An agent died. Propose ONE successor genome that fixes the cause of death.

Trading styles (style sets the timeframe, order type and eligible strategies):
{styles}

Market regimes, computed on closed candles:
{regimes}
regime_map: null, or for EACH regime a strategy eligible for the style or "stand_aside" (do nothing in that regime).
Use it to learn WHEN to trade: e.g. if the agent died scalping during "extreme", map extreme -> vol_expansion or stand_aside.

Chart strategies and setups (integer params and ranges):
{catalog}

Other genes:
- direction: "long" | "short" | "both"
- leverage: 1-10 (capped by the limit above); stop: 0.002-0.05 (distance as fraction of price)
- take_profit: null or 1-5 (multiples of the stop distance); risk: 0.002-0.1 (fraction of equity lost at the stop)
- trail: null or 0.002-0.05 (trailing stop from entry, fraction of price)
- lock_at: null or 0.3-5 (profit lock: once the best gain reaches lock_at x the stop distance, the stop rises to keep
  lock_keep 0-0.9 of that gain, never below breakeven plus fees; it never moves back)
- runner_vol: null or 0.5-0.99 (when in profit and ATR percentile >= this, cancel take_profit and let it run with
  the stop at runner_atr 0.5-6 ATRs from the close, only ever moving up)
- cooldown: 0-48 candles between entries; max_hold: 0 (off) or up to 96 candles
- max_trades_per_day: {mt}; daily_stop: {ds} (must be below the frozen daily loss);
  daily_target: null or {dt} (1.0 = +100% in a day; once hit the agent stops until next session)

Dead agent autopsy:
{autopsy}

Diagnosis in plain words:
{diagnosis}

Recent deaths (avoid repeating them):
{graveyard}

Agents that thrived (hall of fame, with why they won; borrow what fits):
{hall}

Change as little as possible to fix the diagnosis. Keep the strategy unless the cause is negative_edge or no_goal.
Answer with JSON only:
{{"genome": {{"strategy": ..., "params": {{...}}, "direction": ..., "leverage": ..., "stop": ..., "take_profit": ...,
  "risk": ..., "cooldown": ..., "max_trades_per_day": ..., "daily_stop": ..., "daily_target": ..., "max_hold": ...,
  "after_win_mult": ..., "after_loss_mult": ..., "goal_pressure": ..., "style": ..., "trail": ..., "lock_at": ..., "lock_keep": ...,
  "runner_vol": ..., "runner_atr": ..., "entry_offset": ...,
  "order_expiry": ..., "slippage": ..., "regime_map": ..., "news_gate": ..., "news_follow": ..., "event_blackout": ...,
  "news_size_mult": ..., "black_weight": ...}},
  "why": "<one sentence>"}}"""


def diagnose_day(autopsy: dict[str, Any]) -> str:
    life, last = autopsy.get("life", {}), autopsy.get("last_window", {})
    lines = []
    match autopsy["cause"]:
        case "liquidation":
            lines.append("- It was LIQUIDATED: price reached the liquidation price. Lower leverage or tighten the stop.")
        case "ruin":
            lines.append("- It was RUINED: equity fell below the exchange minimum order. Risk less per trade and per day.")
        case "drawdown":
            lines.append(f"- It lost {last.get('max_drawdown', 0):.0%} peak-to-trough in one window.")
            if last.get("dd_high_vol"):
                lines.append("- The losses came during above-normal volatility.")
        case "fees":
            lines.append(
                f"- Before fees {life.get('gross_return', 0):+.1%}, after fees {life.get('net_return', 0):+.1%}"
                f" in {life.get('trades', 0)} trades: it trades too much for its edge."
            )
        case "negative_edge":
            lines.append("- It loses even before fees: the signal does not work here.")
        case "no_goal":
            lines.append(f"- In {life.get('days', 0)} days it never doubled the capital in a day"
                         f" (best day {life.get('best_day', 0):+.1%}).")
        case _:
            lines.append(f"- It never beat the benchmark ({life.get('benchmark_return', 0):+.1%}).")
    if life.get("stop_day_share", 0) > 0.3:
        lines.append(f"- Its own daily_stop ended {life['stop_day_share']:.0%} of its days: too many losing attempts.")
    if life.get("max_trade_day_share", 0) > 0.5:
        lines.append("- It used all its daily attempts on most days.")
    if life.get("style"):
        lines.append(f"- Style: {life['style']}.")
    fs = life.get("fee_share_of_gross")
    if fs is not None and fs > 0.5:
        lines.append(f"- Fees took {fs:.0%} of its gross profit: prefer LIMIT/maker entries or fewer, larger moves.")
    mf = life.get("maker_fill_rate")
    if mf is not None and mf < 0.3:
        lines.append(f"- Only {mf:.0%} of its limit orders filled: offset too far or expiry too short.")
    worst = _worst_regime(life)
    if worst:
        lines.append(f"- Most of its losses came in the '{worst}' regime: map {worst} to stand_aside or to a setup for it.")
    return "\n".join(lines)


class DayLLMCreator:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        context: dict[str, Any],
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        timeout: float = 300.0,
        graveyard: Callable[[], list[dict[str, Any]]] = list,
        post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
        hall: Callable[[], list[dict[str, Any]]] = list,
        fallback: DayRuleCreator | None = None,
        lessons: Callable[[], list[str]] | None = None,
        library: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.symbol, self.timeframe, self.context = symbol, timeframe, context
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout = model, timeout
        self.graveyard, self.post, self.hall = graveyard, post, hall
        self.fallback = fallback or DayRuleCreator()
        self.failures = 0
        self.lessons = lessons  # playbook dos livros (ou hipóteses iniciais)
        self.library = library  # RAG dos livros (busca agêntica)

    styles: list[str] = list(STYLES)
    black_allowed: bool = True

    def prompt(self, autopsy: dict[str, Any]) -> str:
        catalog = "\n".join(f"- {k} {json.dumps(DAY_STRATEGIES[k])}: {CATALOG[k]}" for k in DAY_STRATEGIES)
        styles = "\n".join(
            f"- {s}: {STYLE_DESCRIPTIONS[s]}; eligible: {', '.join(eligible(s))}" for s in self.styles
        )
        regimes = "\n".join(f"- {r}: {REGIME_DESCRIPTIONS[r]}" for r in REGIMES)
        prompt = PROMPT.format(
            styles=styles, regimes=regimes,
            black_rule="must be 0 in this experiment arm" if not self.black_allowed else "evolvable",
            symbol=self.symbol, timeframe=self.timeframe, catalog=catalog,
            mt=f"{MAX_TRADES_RANGE[0]}-{MAX_TRADES_RANGE[1]}",
            ds=f"{DAILY_STOP_RANGE[0]}-{DAILY_STOP_RANGE[1]}",
            dt=f"{DAILY_TARGET_RANGE[0]}-{DAILY_TARGET_RANGE[1]}",
            autopsy=json.dumps(autopsy, default=str),
            diagnosis=diagnose_day(autopsy),
            graveyard=json.dumps(self.graveyard()[-8:], default=str),
            hall=json.dumps(self.hall()[:3], default=str),
            **self.context,
        )
        return self._with_knowledge(prompt, diagnose_day(autopsy))

    def _with_knowledge(self, prompt: str, situation: str) -> str:
        """Acrescenta o playbook dos livros e trechos do RAG antes do formato da resposta."""
        extra = ""
        try:
            rules = self.lessons() if self.lessons else []
            if rules:
                extra += "Playbook from trading books (follow unless the evidence contradicts):\n"
                extra += "\n".join(f"- {r}" for r in rules[:10]) + "\n\n"
            passages = self.library(situation)[:3] if self.library else []
            if passages:
                extra += "Reference passages (cite as [book p.N] in why if used):\n"
                extra += "\n".join(f"[{p['book']} p.{p['page']}] {str(p['text'])[:500]}" for p in passages) + "\n\n"
        except Exception:
            pass  # livros fora do ar não impedem o Criador
        cut = prompt.rfind("Answer")
        return prompt[:cut] + extra + prompt[cut:] if cut >= 0 else prompt + "\n\n" + extra

    def successor(self, g: DayGenome, autopsy: dict[str, Any], rng: random.Random) -> tuple[DayGenome, str]:
        try:
            reply = self.post(
                self.url,
                {"model": self.model, "messages": [{"role": "user", "content": self.prompt(autopsy)}],
                 "temperature": 0.7, "response_format": {"type": "json_object"}},
                self.timeout,
            )
            answer = _extract_json(reply["choices"][0]["message"]["content"])
            new = DayGenome.from_dict({**g.to_dict(), **{k: v for k, v in answer["genome"].items() if v is not ...}})
            if new.style not in self.styles or new.strategy not in eligible(new.style):
                raise ValueError(f"estilo/estratégia fora do permitido: {new.style}/{new.strategy}")
            if not self.black_allowed and new.black_weight:
                new = replace(new, black_weight=0.0)
            return new, "LLM: " + str(answer.get("why", ""))[:200]
        except Exception as exc:
            self.failures += 1
            self.fallback.styles, self.fallback.black_allowed = self.styles, self.black_allowed
            new, why = self.fallback.successor(g, autopsy, rng)
            return new, f"{why} (LLM falhou: {type(exc).__name__})"

    def child(self, g: DayGenome, rng: random.Random) -> DayGenome:
        self.fallback.styles, self.fallback.black_allowed = self.styles, self.black_allowed
        return self.fallback.child(g, rng)
