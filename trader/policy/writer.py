"""Escritor: a LLM local lê o contexto e escreve a política (segundos); aqui se valida e guarda.

Se a resposta for inválida, os erros voltam para a LLM corrigir (até `retries`).
Se ainda assim falhar, mantém a política anterior enquanto válida; senão, SAFE.
Nunca usa o Claude: só a LLM local (Ollama) ou a remota do usuário, pela API no formato OpenAI.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.genome import DAY_STRATEGIES
from trader.policy.schema import ACTIONS, BIASES, NEWS_KINDS, SAFE, InvalidPolicy, Policy, parse_policy
from trader.storage.atomic import replace_with_retry, unique_tmp

LLM_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:7b"
HISTORY_MAX_LINES = 2000

PROMPT = """You are the risk-and-policy brain of a PAPER crypto trading system. Fast executors apply your
policy in milliseconds; you never place orders yourself. Write the policy for the next {valid} minutes.

Hard limits (anything outside is rejected): max_leverage 1-{lev}; risk_per_trade 0.0005-{risk};
stop {smin}-{smax} (fraction of price); daily_loss_limit 0.001-{dll}.
Symbols: {symbols}
bias: one of {biases}. setups: ["*"] or names from {setups}.
Exits: take_profit_r null or 0.5-10 (multiples of stop); trail null or {smin}-{smax};
lock_at null or 0.3-5 (profit lock: after lock_at x stop of gain, keep lock_keep 0-0.9 of the gain);
runner_vol null or 0.5-0.99 (ATR percentile that cancels the target and lets it run) with runner_atr 0.5-6.
reactions: when.kind one of {kinds}; when.sentiment any|negative|positive; when.min_severity 0-1;
do one of {actions}; factor 0-1 (reduce keeps this fraction; tighten_stop multiplies stop distance);
minutes 1-1440.

What the evidence says (out-of-sample tests): only the daily trend committee (price above 50/100/150/200-day
averages) had an edge; intraday price indicators alone did NOT beat fees. Prefer flat or small risk when
the committee is weak; prefer longs only when the committee is strong. Large hacks deserve defensive reactions.
Do not invent data. If unsure, be conservative (bias "flat" is a valid answer).

Context (JSON):
{context}

Current policy (JSON or null):
{current}

RULE-BASED BASELINE for right now (derived from the committee, volatility and 7-day move). Start from it.
You may only make it MORE conservative (lower risk, flat instead of long, more reactions); anything looser
is discarded. Change a field only if the context gives a concrete reason, and say why in the rationale:
{baseline}

Rules: at most 8 distinct reactions, each with a specific kind or sentiment. Your rationale MUST match the
biases you set (if you say "favoring longs", some bias must be "long" or "both").

Answer with ONE JSON object only, shaped exactly like this example (values are only an example):
{{"valid_minutes": 30, "kill": false, "daily_loss_limit": 0.02,
 "symbols": {{"SOL/USDT": {{"bias": "long", "setups": ["*"], "max_leverage": 1, "risk_per_trade": 0.005,
   "stop": 0.015, "take_profit_r": null, "trail": null, "lock_at": 1.0, "lock_keep": 0.5,
   "runner_vol": null, "runner_atr": 2.0}}}},
 "reactions": [{{"when": {{"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.7}},
   "do": "reduce", "factor": 0.5, "minutes": 120}}],
 "rationale": "...", "sources": []}}"""

FIX = """Your policy was rejected. Errors:
{errors}
Return the corrected full JSON object only."""


def build_prompt(context: dict[str, Any], symbols: list[str], current: Policy | None, valid: int = 30,
                 limits: DayTradeLimits = DayTradeLimits(), baseline: dict[str, Any] | None = None) -> str:
    return PROMPT.format(
        baseline=json.dumps(baseline, ensure_ascii=False) if baseline else "null",
        valid=valid, lev=limits.max_leverage, risk=limits.max_risk_per_trade, smin=limits.min_stop,
        smax=limits.max_stop, dll=limits.daily_loss_limit, symbols=symbols, biases=list(BIASES),
        setups=sorted(DAY_STRATEGIES), kinds=list(NEWS_KINDS), actions=list(ACTIONS),
        context=json.dumps(context, ensure_ascii=False, default=str)[:6000],
        current=json.dumps(policy_to_dict(current), ensure_ascii=False) if current else "null",
    )


def policy_to_dict(p: Policy) -> dict[str, Any]:
    d = asdict(p)
    d["reactions"] = [{"when": {"kind": r["kind"], "symbols": list(r["symbols"]), "sentiment": r["sentiment"],
                                "min_severity": r["min_severity"]},
                       "do": r["action"], "factor": r["factor"], "minutes": r["minutes"]} for r in d["reactions"]]
    for sp in d["symbols"].values():
        sp["setups"] = list(sp["setups"])
    d["sources"] = list(d["sources"])
    return d


def ollama_chat(messages: list[dict[str, str]], url: str = LLM_URL, model: str = LLM_MODEL,
                timeout: int = 180) -> str:
    from trader import llm

    return llm.chat(messages, llm.LLMConfig("ollama", model, url), json_mode=True, timeout=timeout)


def provider_chat(cfg=None, schema: dict | None = None):
    """Função de chat do provedor configurado. Com `schema`, o Ollama usa decodificação restrita."""
    from trader import llm

    cfg = cfg or llm.config()
    return lambda messages: llm.chat(messages, cfg, json_mode=True, schema=schema)


def _json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start >= 0 < end else json.loads(text)


def write_policy(context: dict[str, Any], symbols: list[str], current: Policy | None = None,
                 chat: Callable[[list[dict[str, str]]], str] = ollama_chat, retries: int = 2,
                 limits: DayTradeLimits = DayTradeLimits(), use_rules: bool = True,
                 lenient: bool = False) -> tuple[Policy, dict[str, Any]]:
    """Devolve (política, relatório). Nunca levanta: na falha devolve a base por regras, a atual ou SAFE.

    `use_rules`: a LLM parte da política-base por regras e só pode apertá-la (tighten).
    `lenient`: números fora da faixa são ajustados em vez de rejeitados (decodificação restrita)."""
    from trader.policy.rules import baseline, baseline_policy, tighten

    base_raw = baseline(context, symbols) if use_rules else None
    base = baseline_policy(context, symbols) if use_rules else None
    messages = [{"role": "user", "content": build_prompt(context, symbols, current, limits=limits, baseline=base_raw)}]
    report: dict[str, Any] = {"attempts": 0, "errors": [], "seconds": 0.0, "fallback": None, "tightened": None}
    t0 = time.perf_counter()
    for _ in range(retries + 1):
        report["attempts"] += 1
        try:
            text = chat(messages)
        except Exception as exc:  # LLM fora do ar: não adianta insistir
            report["errors"].append(f"LLM: {type(exc).__name__}: {exc}")
            break
        try:
            policy = parse_policy(_json(text), symbols, limits, lenient=lenient)
            report["seconds"] = round(time.perf_counter() - t0, 2)
            if base is not None:
                merged = tighten(base, policy)
                report["tightened"] = [s for s in symbols if merged.symbols[s] != policy.symbols.get(s)]
                policy = merged
            return policy, report
        except (InvalidPolicy, ValueError) as exc:
            errs = exc.errors if isinstance(exc, InvalidPolicy) else [f"JSON inválido: {exc}"]
            report["errors"].append(errs)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": FIX.format(errors="\n".join(f"- {e}" for e in errs[:20]))}]
    report["seconds"] = round(time.perf_counter() - t0, 2)
    if base is not None:
        report["fallback"] = "regras"
        return base, report
    report["fallback"] = "atual" if current is not None else "SAFE"
    return (current or SAFE), report


def save(policy: Policy, report: dict[str, Any], folder: Path, ts: str) -> None:
    """current.json (troca atômica) + histórico em JSONL com tamanho limitado."""
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"ts": ts, "policy": policy_to_dict(policy), "report": report}
    cur = folder / "current.json"
    tmp = unique_tmp(cur)
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(tmp, cur)
    hist = folder / "history.jsonl"
    lines = hist.read_text(encoding="utf-8").splitlines() if hist.exists() else []
    lines = (lines + [json.dumps(rec, ensure_ascii=False)])[-HISTORY_MAX_LINES:]
    tmp = unique_tmp(hist)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    replace_with_retry(tmp, hist)


def load_current(folder: Path, symbols: list[str]) -> Policy | None:
    try:
        rec = json.loads((folder / "current.json").read_text(encoding="utf-8"))
        return parse_policy(rec["policy"], symbols)
    except (OSError, ValueError, KeyError, InvalidPolicy):
        return None


def env_llm():
    from trader import llm

    return llm.config()
