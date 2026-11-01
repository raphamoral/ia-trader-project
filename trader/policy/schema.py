"""Política: o que a LLM decide (em segundos) e o executor aplica (em milissegundos).

A LLM escreve JSON; aqui ele vira um objeto validado. A política só pode APERTAR
os limites congelados (DayTradeLimits): valor fora deles é erro, devolvido à LLM
para corrigir. Sem política válida, vale a SAFE (sem entradas novas).

Formato:
{
  "valid_minutes": 30,                      # depois disso o executor para de abrir posições
  "kill": false,                            # true = zera tudo e não entra mais
  "daily_loss_limit": 0.03,                 # fração do patrimônio do dia
  "symbols": {
    "SOL/USDT": {"bias": "long" | "short" | "both" | "flat",
                 "setups": ["*"] ou ["momentum", ...],
                 "max_leverage": 2, "risk_per_trade": 0.01, "stop": 0.01,
                 "take_profit_r": 2.0 | null, "trail": null,
                 "lock_at": 1.0 | null, "lock_keep": 0.5, "runner_vol": 0.8 | null, "runner_atr": 2.0}
  },
  "reactions": [
    {"when": {"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.7},
     "do": "flatten" | "reduce" | "block_entries" | "tighten_stop", "factor": 0.5, "minutes": 60}
  ],
  "rationale": "uma ou duas frases", "sources": ["livro, página"]
}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.genome import DAY_STRATEGIES

BIASES = ("long", "short", "both", "flat")
ACTIONS = ("flatten", "reduce", "block_entries", "tighten_stop")
NEWS_KINDS = ("any", "hack", "regulation", "listing", "delisting", "macro", "etf", "lawsuit", "outage", "other")
SENTIMENTS = ("any", "negative", "positive")
MAX_VALID_MINUTES = 24 * 60
MAX_REACTIONS = 8


def json_schema(symbols: list[str], limits: DayTradeLimits = DayTradeLimits()) -> dict[str, Any]:
    """JSON Schema da política, para decodificação restrita (o modelo só gera tokens válidos).

    Enumerações e estrutura são garantidas pela gramática; faixas numéricas não são (o Ollama
    ignora minimum/maximum), por isso o parse `lenient` ajusta números para dentro dos limites."""
    num = lambda lo, hi: {"type": "number", "minimum": lo, "maximum": hi}  # noqa: E731
    opt = lambda lo, hi: {"type": ["number", "null"], "minimum": lo, "maximum": hi}  # noqa: E731
    sym = {"type": "object", "additionalProperties": False, "required": ["bias", "setups", "max_leverage", "risk_per_trade", "stop"],
           "properties": {"bias": {"type": "string", "enum": list(BIASES)},
                          "setups": {"type": "array", "items": {"type": "string", "enum": ["*", *sorted(DAY_STRATEGIES)]}, "maxItems": 8},
                          "max_leverage": num(1.0, limits.max_leverage), "risk_per_trade": num(0.0005, limits.max_risk_per_trade),
                          "stop": num(limits.min_stop, limits.max_stop), "take_profit_r": opt(0.5, 10.0),
                          "trail": opt(limits.min_stop, limits.max_stop), "lock_at": opt(0.3, 5.0), "lock_keep": num(0.0, 0.9),
                          "runner_vol": opt(0.5, 0.99), "runner_atr": num(0.5, 6.0)}}
    reaction = {"type": "object", "additionalProperties": False, "required": ["when", "do", "minutes"],
                "properties": {"when": {"type": "object", "additionalProperties": False, "required": ["kind", "sentiment", "min_severity"],
                                        "properties": {"kind": {"type": "string", "enum": list(NEWS_KINDS)},
                                                       "symbols": {"type": "array", "items": {"type": "string", "enum": ["*", *symbols]}, "maxItems": 4},
                                                       "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
                                                       "min_severity": num(0.0, 1.0)}},
                               "do": {"type": "string", "enum": list(ACTIONS)}, "factor": num(0.0, 1.0),
                               "minutes": {"type": "integer", "minimum": 1, "maximum": MAX_VALID_MINUTES}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["valid_minutes", "kill", "daily_loss_limit", "symbols", "reactions", "rationale"],
            "properties": {"valid_minutes": {"type": "integer", "minimum": 1, "maximum": MAX_VALID_MINUTES},
                           "kill": {"type": "boolean"}, "daily_loss_limit": num(0.001, limits.daily_loss_limit),
                           "symbols": {"type": "object", "additionalProperties": False, "required": symbols,
                                       "properties": {s: sym for s in symbols}},
                           "reactions": {"type": "array", "items": reaction, "maxItems": MAX_REACTIONS},
                           "rationale": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}, "maxItems": 10}}}


class InvalidPolicy(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class SymbolPolicy:
    bias: str = "flat"
    setups: tuple[str, ...] = ("*",)
    max_leverage: float = 1.0
    risk_per_trade: float = 0.005
    stop: float = 0.01
    take_profit_r: float | None = None
    trail: float | None = None
    lock_at: float | None = None
    lock_keep: float = 0.5
    runner_vol: float | None = None
    runner_atr: float = 2.0

    def allows(self, side: int, setup: str) -> bool:
        if self.bias == "flat" or (self.bias == "long" and side < 0) or (self.bias == "short" and side > 0):
            return False
        return "*" in self.setups or setup in self.setups


@dataclass(frozen=True)
class Reaction:
    kind: str = "any"
    symbols: tuple[str, ...] = ("*",)
    sentiment: str = "any"
    min_severity: float = 0.5
    action: str = "block_entries"
    factor: float = 0.5
    minutes: int = 60

    def matches(self, kind: str, symbols: tuple[str, ...], sentiment: float, severity: float) -> bool:
        if self.kind != "any" and kind != self.kind:
            return False
        if "*" not in self.symbols and not set(self.symbols) & set(symbols):
            return False
        if self.sentiment == "negative" and sentiment >= 0 or self.sentiment == "positive" and sentiment <= 0:
            return False
        return severity >= self.min_severity


@dataclass(frozen=True)
class Policy:
    symbols: dict[str, SymbolPolicy] = field(default_factory=dict)
    reactions: tuple[Reaction, ...] = ()
    valid_minutes: int = 30
    kill: bool = False
    daily_loss_limit: float = 0.03
    rationale: str = ""
    sources: tuple[str, ...] = ()

    def for_symbol(self, symbol: str) -> SymbolPolicy:
        return self.symbols.get(symbol, SymbolPolicy())


SAFE = Policy(rationale="sem política válida: nenhuma entrada nova", valid_minutes=MAX_VALID_MINUTES)


_LENIENT = False  # com decodificação restrita, números fora da faixa são ajustados (nunca além dos limites congelados)


def _num(d: dict, key: str, lo: float, hi: float, errors: list[str], where: str, default=None, optional=False):
    v = d.get(key, default)
    if v is None:
        if not optional:
            errors.append(f"{where}.{key} é obrigatório")
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        errors.append(f"{where}.{key} precisa ser número, veio {v!r}")
        return None
    if not lo <= x <= hi:
        if _LENIENT:
            return min(max(x, lo), hi)
        errors.append(f"{where}.{key}={x} fora de [{lo}, {hi}]")
        return None
    return x


def parse_policy(raw: dict[str, Any], symbols: list[str], limits: DayTradeLimits = DayTradeLimits(),
                 lenient: bool = False) -> Policy:
    """Valida o JSON da LLM. Junta todos os erros (para a LLM corrigir de uma vez).

    `lenient`: números fora da faixa são ajustados para o limite (uso com decodificação restrita)."""
    global _LENIENT
    _LENIENT = lenient
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise InvalidPolicy(["a política precisa ser um objeto JSON"])
    valid = _num(raw, "valid_minutes", 1, MAX_VALID_MINUTES, errors, "política", 30)
    dll = _num(raw, "daily_loss_limit", 0.001, limits.daily_loss_limit, errors, "política", limits.daily_loss_limit)
    sym_raw = raw.get("symbols") or {}
    if not isinstance(sym_raw, dict):
        errors.append("symbols precisa ser um objeto {símbolo: {...}}")
        sym_raw = {}
    out_syms: dict[str, SymbolPolicy] = {}
    for sym, sp in sym_raw.items():
        w = f"symbols[{sym}]"
        if sym not in symbols:
            errors.append(f"{w}: símbolo desconhecido (use um de {symbols})")
            continue
        if not isinstance(sp, dict):
            errors.append(f"{w} precisa ser um objeto")
            continue
        bias = sp.get("bias", "flat")
        if bias not in BIASES:
            errors.append(f"{w}.bias={bias!r} (use {BIASES})")
        setups = sp.get("setups", ["*"])
        if isinstance(setups, str):
            setups = [setups]
        bad = [s for s in setups if s != "*" and s not in DAY_STRATEGIES]
        if bad:
            errors.append(f"{w}.setups desconhecidos: {bad}")
        vals = {
            "max_leverage": _num(sp, "max_leverage", 1.0, limits.max_leverage, errors, w, 1.0),
            "risk_per_trade": _num(sp, "risk_per_trade", 0.0005, limits.max_risk_per_trade, errors, w, 0.005),
            "stop": _num(sp, "stop", limits.min_stop, limits.max_stop, errors, w, 0.01),
            "take_profit_r": _num(sp, "take_profit_r", 0.5, 10.0, errors, w, optional=True),
            "trail": _num(sp, "trail", limits.min_stop, limits.max_stop, errors, w, optional=True),
            "lock_at": _num(sp, "lock_at", 0.3, 5.0, errors, w, optional=True),
            "lock_keep": _num(sp, "lock_keep", 0.0, 0.9, errors, w, 0.5),
            "runner_vol": _num(sp, "runner_vol", 0.5, 0.99, errors, w, optional=True),
            "runner_atr": _num(sp, "runner_atr", 0.5, 6.0, errors, w, 2.0),
        }
        if bias in BIASES and not bad and all(v is not None for k, v in vals.items()
                                              if k not in ("take_profit_r", "trail", "lock_at", "runner_vol")):
            out_syms[sym] = SymbolPolicy(bias=bias, setups=tuple(setups), **vals)
    reactions = []
    for i, r in enumerate(raw.get("reactions") or []):
        w = f"reactions[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{w} precisa ser um objeto")
            continue
        when = r.get("when") or {}
        kind, sent, act = when.get("kind", "any"), when.get("sentiment", "any"), r.get("do")
        syms = when.get("symbols", ["*"])
        syms = [syms] if isinstance(syms, str) else list(syms)
        if kind not in NEWS_KINDS:
            errors.append(f"{w}.when.kind={kind!r} (use {NEWS_KINDS})")
        if sent not in SENTIMENTS:
            errors.append(f"{w}.when.sentiment={sent!r} (use {SENTIMENTS})")
        if act not in ACTIONS:
            errors.append(f"{w}.do={act!r} (use {ACTIONS})")
        unknown = [s for s in syms if s != "*" and s not in symbols]
        if unknown:
            errors.append(f"{w}.when.symbols desconhecidos: {unknown}")
        sev = _num(when, "min_severity", 0.0, 1.0, errors, f"{w}.when", 0.5)
        factor = _num(r, "factor", 0.0, 1.0, errors, w, 0.5)
        minutes = _num(r, "minutes", 1, MAX_VALID_MINUTES, errors, w, 60)
        if kind == "any" and sent == "any" and act in ("flatten", "reduce", "tighten_stop"):
            errors.append(f"{w}: reação ampla demais (qualquer notícia → {act}); diga o tipo ou o sentimento")
        if not errors or all(not e.startswith(w) for e in errors):
            rx = Reaction(kind, tuple(syms), sent, sev, act, factor, int(minutes))
            if rx not in reactions:  # repetidas não somam nada
                reactions.append(rx)
    if len(reactions) > MAX_REACTIONS:
        errors.append(f"no máximo {MAX_REACTIONS} reações (vieram {len(reactions)})")
    if errors:
        raise InvalidPolicy(errors)
    sources = raw.get("sources") or []
    return Policy(
        symbols=out_syms, reactions=tuple(reactions), valid_minutes=int(valid), kill=bool(raw.get("kill", False)),
        daily_loss_limit=dll, rationale=str(raw.get("rationale", ""))[:500],
        sources=tuple(str(s)[:200] for s in sources[:10]) if isinstance(sources, list) else (),
    )
