"""Reflexo de notícias: classifica a manchete em microssegundos, sem LLM, e avisa o executor.

Regras simples e transparentes (palavras-chave + valor em US$) para tipo, ativos,
sentimento e gravidade. É a versão provisória do classificador rápido (peça 4,
que será treinado com rótulos da Qwen). A LLM continua lendo com calma depois.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

KINDS = {
    "hack": ("hack", "exploit", "drained", "stolen", "breach", "attack", "hacker", "roubo", "ataque"),
    "delisting": ("delist", "delisting"),
    "listing": ("will list", "lists ", "listing", "listará"),
    "regulation": ("sec ", "lawsuit", "ban ", "bans ", "regulat", "cftc", "sues", "processa", "proíbe"),
    "etf": (" etf", "etfs"),
    "outage": ("outage", "halted", "paused withdrawals", "suspends withdrawals", "down for", "fora do ar"),
    "macro": ("fed ", "fomc", "rate hike", "rate cut", "cpi", "inflation", "juros", "inflação", "payroll"),
}
NEGATIVE = ("hack", "exploit", "drained", "stolen", "ban", "lawsuit", "sues", "delist", "outage", "halted",
            "crash", "plunge", "liquidat", "fraud", "bankrupt", "outflow", "rate hike", "queda", "despenca")
POSITIVE = ("approve", "approval", "inflow", "will list", "partnership", "rate cut", "surge", "rally", "record high",
            "aprova", "dispara", "alta recorde")
ASSETS = {
    "BTC/USDT": ("bitcoin", "btc"),
    "ETH/USDT": ("ethereum", "ether ", "eth "),
    "SOL/USDT": ("solana", "sol "),
    "XRP/USDT": ("xrp", "ripple"),
}
MONEY = re.compile(r"\$\s?(\d+(?:[.,]\d+)?)\s?(billion|bn|b\b|million|mn|m\b|mil|bi)", re.I)


@dataclass(frozen=True)
class NewsEvent:
    kind: str
    symbols: tuple[str, ...]
    sentiment: float  # −1 a 1
    severity: float  # 0 a 1
    text: str


def usd_amount(text: str) -> float:
    best = 0.0
    for num, unit in MONEY.findall(text):
        x = float(num.replace(",", "."))
        x *= 1e9 if unit.lower() in ("billion", "bn", "b", "bi") else 1e6
        best = max(best, x)
    return best


def classify(text: str) -> NewsEvent:
    t = f" {text.lower()} "
    kind = next((k for k, words in KINDS.items() if any(w in t for w in words)), "other")
    neg = sum(w in t for w in NEGATIVE)
    pos = sum(w in t for w in POSITIVE)
    sentiment = 0.0 if neg == pos else (pos - neg) / (pos + neg)
    symbols = tuple(s for s, words in ASSETS.items() if any(w in t for w in words)) or ("*",)
    severity = 0.3 + 0.2 * min(neg + pos, 2)
    usd = usd_amount(text)
    if kind == "hack" and usd:
        severity = 0.5 if usd < 10e6 else 0.7 if usd < 100e6 else 0.9  # >= US$100M: o alarme que testamos
    if kind == "hack":
        sentiment = min(sentiment, -0.5)
    return NewsEvent(kind, symbols, round(sentiment, 2), round(min(severity, 1.0), 2), text[:200])


_MODEL = None


def classify_fast(text: str, data: Path | None = None) -> NewsEvent:
    """Modelo destilado da Qwen (peça 4), se existir em data/models; senão, as palavras-chave."""
    global _MODEL
    if _MODEL is None:
        path = (data or Path("data")) / "models" / "news_reflex.pkl"
        _MODEL = pickle.loads(path.read_bytes()) if path.exists() else False
    if _MODEL is False:
        return classify(text)
    p = _MODEL.predict([text])[0]
    lex = classify(text)
    if not p["relevant"]:
        return NewsEvent("other", ("*",), 0.0, 0.0, text[:200])
    syms = tuple(f"{s}/USDT" for s in p["symbols"] if s != "MARKET") or ("*",)
    severity = p["severity"]
    if lex.kind == "hack" and lex.severity >= 0.9:  # valor em US$ explícito: a regra do alarme testado vale
        severity = max(severity, lex.severity)
    return NewsEvent(p["kind"], syms, round(p["sentiment"], 2), round(min(max(severity, 0.0), 1.0), 2), text[:200])
