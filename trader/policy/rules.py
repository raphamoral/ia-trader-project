"""Política-base por regras: o que o contexto diz sem precisar de LLM.

A LLM recebe esta base e só pode APERTAR (nunca afrouxar): o resultado final é o
mais conservador entre a base e a resposta dela. Assim, um modelo pequeno que
erra ou inventa não consegue piorar a política; só consegue deixá-la mais cautelosa.

Regras (derivadas dos achados: só o comitê diário tem vantagem; vol-alvo; hack grande = proteção):
  comitê >= 0,75 e sem queda forte na semana  → comprado, risco cheio
  comitê >= 0,5                               → comprado, metade do risco
  comitê < 0,5                                → fora
  comitê 0,75 com volatilidade > 60% ao ano ou queda > 7% em 7 dias → fora (sinal fraco em mercado nervoso)
  risco por operação: 0,5% × (30% / vol anual), entre 0,1% e 0,5%
  stop: ~1 ATR de 5 min em %, entre 1% e 2%; trailing igual ao stop; trava de lucro 1R/50%
  reações fixas: hack ≥0,9 zera 3h; hack ≥0,7 reduz 50% 2h; hack ≥0,5 bloqueia 1h; regulação/outage ≥0,6 bloqueia 90 min;
                 macro negativo ≥0,6 reduz 30% 1h; deslistagem zera 4h
"""
from __future__ import annotations

from typing import Any

from trader.policy.schema import BIASES, Policy, Reaction, SymbolPolicy, parse_policy

TREND_SETUPS = ("trend_pullback", "supertrend", "donchian", "ema_cross", "pivot_breakout", "momentum")
REACTIONS: list[dict[str, Any]] = [
    {"when": {"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.9}, "do": "flatten", "minutes": 180},
    {"when": {"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.7}, "do": "reduce", "factor": 0.5, "minutes": 120},
    {"when": {"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.5}, "do": "block_entries", "minutes": 60},
    {"when": {"kind": "regulation", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.6}, "do": "block_entries", "minutes": 90},
    {"when": {"kind": "outage", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.7}, "do": "block_entries", "minutes": 90},
    {"when": {"kind": "macro", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.6}, "do": "reduce", "factor": 0.7, "minutes": 60},
    {"when": {"kind": "delisting", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.5}, "do": "flatten", "minutes": 240},
]


def symbol_rules(m: dict[str, Any]) -> dict[str, Any]:
    committee = float(m.get("committee", 0.0) or 0.0)
    vol = float(m.get("vol20_annual", 0.6) or 0.6)
    ret7 = float(m.get("ret_7d", 0.0) or 0.0)
    atr_pct_price = None
    if m.get("price") and m.get("atr_abs"):
        atr_pct_price = float(m["atr_abs"]) / float(m["price"])
    risk = min(0.005, max(0.001, 0.005 * 0.30 / max(vol, 0.05)))
    stop = min(0.02, max(0.01, atr_pct_price * 1.0 if atr_pct_price else 0.015))
    nervous = vol > 0.6 or ret7 < -0.07  # mercado nervoso: só o comitê cheio sustenta posição
    if committee >= 1.0 or (committee >= 0.75 and not nervous):
        bias, r = "long", risk
    elif committee >= 0.5 and not nervous:
        bias, r = "long", risk / 2
    else:
        bias, r = "flat", 0.001
    return {"bias": bias, "setups": list(TREND_SETUPS) if bias == "long" else ["*"], "max_leverage": 1.0,
            "risk_per_trade": round(r, 4), "stop": round(stop, 4), "take_profit_r": None,
            "trail": round(stop, 4) if bias == "long" else None, "lock_at": 1.0 if bias == "long" else None,
            "lock_keep": 0.5, "runner_vol": None, "runner_atr": 2.0}


def baseline(context: dict[str, Any], symbols: list[str], valid_minutes: int = 30) -> dict[str, Any]:
    market = context.get("market", {})
    syms = {s: symbol_rules(market.get(s, {})) for s in symbols}
    longs = [s for s, v in syms.items() if v["bias"] == "long"]
    return {"valid_minutes": valid_minutes, "kill": False,
            "daily_loss_limit": 0.02 if len(longs) >= len(symbols) / 2 else 0.015,
            "symbols": syms, "reactions": REACTIONS,
            "rationale": "base por regras: " + ", ".join(f"{s.split('/')[0]} {v['bias']}" for s, v in syms.items()),
            "sources": []}


_BIAS_RANK = {"flat": 0, "long": 1, "short": 1, "both": 2}


def tighten(base: Policy, llm: Policy) -> Policy:
    """Resultado = o mais conservador entre a base e a LLM, campo a campo."""
    syms: dict[str, SymbolPolicy] = {}
    for s, b in base.symbols.items():
        l = llm.symbols.get(s, SymbolPolicy())
        bias = b.bias if _BIAS_RANK[b.bias] <= _BIAS_RANK[l.bias] else l.bias
        if bias != "flat" and b.bias != l.bias and {b.bias, l.bias} == {"long", "short"}:
            bias = "flat"  # discordância de direção: fica fora
        setups = tuple(sorted(set(b.setups) & set(l.setups))) if "*" not in b.setups and "*" not in l.setups \
            else (b.setups if "*" in l.setups else l.setups)
        syms[s] = SymbolPolicy(
            bias=bias, setups=setups or b.setups,
            max_leverage=min(b.max_leverage, l.max_leverage), risk_per_trade=min(b.risk_per_trade, l.risk_per_trade),
            stop=min(b.stop, l.stop), take_profit_r=b.take_profit_r if l.take_profit_r is None else l.take_profit_r,
            trail=min(x for x in (b.trail, l.trail) if x is not None) if (b.trail or l.trail) else None,
            lock_at=min(x for x in (b.lock_at, l.lock_at) if x is not None) if (b.lock_at or l.lock_at) else None,
            lock_keep=max(b.lock_keep, l.lock_keep), runner_vol=b.runner_vol if l.runner_vol is None else l.runner_vol,
            runner_atr=min(b.runner_atr, l.runner_atr),
        )
    reactions: list[Reaction] = list(base.reactions)
    for r in llm.reactions:
        if r not in reactions and len(reactions) < 8:
            reactions.append(r)
    return Policy(symbols=syms, reactions=tuple(reactions), valid_minutes=min(base.valid_minutes, llm.valid_minutes),
                  kill=base.kill or llm.kill, daily_loss_limit=min(base.daily_loss_limit, llm.daily_loss_limit),
                  rationale=llm.rationale or base.rationale, sources=llm.sources)


def baseline_policy(context: dict[str, Any], symbols: list[str]) -> Policy:
    return parse_policy(baseline(context, symbols), symbols)
