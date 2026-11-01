"""Linguagem restrita de estratégias: a LLM monta estratégias com peças prontas, sem escrever código.

Uma estratégia é JSON:
  {"entry": <condição>, "size": {"vol_target": 0.3}, "name": "..."}
Condição:
  {"all": [<cond>, ...]} | {"any": [<cond>, ...]} | {"not": <cond>} | <peça>
Peça (todas calculadas só com dados até o fechamento do dia; o compilador desloca 1 dia ao operar):
  {"above_sma": n}            fechamento acima da média de n dias
  {"sma_cross": [fast, slow]} média rápida acima da lenta
  {"momentum": n, "gt": x}    retorno de n dias maior que x
  {"vol_pct": n, "lt": p}     volatilidade de 20 dias abaixo do percentil p (janela de n dias)
  {"rsi": n, "lt"|"gt": x}    RSI de n dias
  {"off_high": n, "lt": x}    queda desde a máxima de n dias menor que x (fração)
  {"trend_vote": [n...], "ge": k}  pelo menos k das médias da lista com preço acima
  {"trend_frac": [n...]}      fração das médias com preço acima (0 a 1): o "comitê de tendência"

Lógica gradual: cada peça vale de 0 a 1; "all" = mínimo, "any" = máximo, "not" = 1 − valor.

O resultado é uma posição de 0 a 1 por ativo, que o sizing escala por volatilidade.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DAYS = 365
MAX_DEPTH = 4
MAX_NODES = 12
LIMITS = {"n": (2, 400), "x": (-1.0, 5.0), "p": (0.0, 1.0)}


class InvalidStrategy(ValueError):
    pass


def _int(v: Any) -> int:
    n = int(v)
    if not LIMITS["n"][0] <= n <= LIMITS["n"][1]:
        raise InvalidStrategy(f"janela fora da faixa: {n}")
    return n


def _num(v: Any, kind: str = "x") -> float:
    x = float(v)
    if kind == "p" and 1 < x <= 100:
        x /= 100  # aceita percentil/fração em escala 0-100 (20 -> 0.2)
    lo, hi = LIMITS[kind]
    if not lo <= x <= hi:
        raise InvalidStrategy(f"valor fora da faixa: {x}")
    return x


def _rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _piece(node: dict, close: pd.DataFrame) -> pd.DataFrame:
    if "above_sma" in node:
        return close > close.rolling(_int(node["above_sma"])).mean()
    if "sma_cross" in node:
        fast, slow = (_int(v) for v in node["sma_cross"])
        if fast >= slow:
            raise InvalidStrategy("sma_cross exige rápida < lenta")
        return close.rolling(fast).mean() > close.rolling(slow).mean()
    if "momentum" in node:
        return close.pct_change(_int(node["momentum"])) > _num(node.get("gt", 0.0))
    if "vol_pct" in node:
        vol = close.pct_change().rolling(20).std()
        pct = vol.rolling(_int(node["vol_pct"]), min_periods=20).rank(pct=True)
        return pct < _num(node.get("lt", 0.7), "p")
    if "rsi" in node:
        r = _rsi(close, _int(node["rsi"]))
        level = float(node["lt"] if "lt" in node else node.get("gt", 50))
        if not 0 <= level <= 100:
            raise InvalidStrategy(f"nível de RSI fora de 0-100: {level}")
        return r < level if "lt" in node else r > level
    if "off_high" in node:
        high = close.rolling(_int(node["off_high"])).max()
        return (1 - close / high) < _num(node.get("lt", 0.2), "p")
    if "trend_frac" in node:
        ns = [_int(v) for v in node["trend_frac"]]
        return sum((close > close.rolling(n).mean()).astype(float) for n in ns) / len(ns)
    if "trend_vote" in node:
        ns = [_int(v) for v in node["trend_vote"]]
        votes = sum((close > close.rolling(n).mean()).astype(int) for n in ns)
        return votes >= int(node.get("ge", len(ns)))
    raise InvalidStrategy(f"peça desconhecida: {list(node)}")


def _count(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    children = node.get("all") or node.get("any") or ([node["not"]] if "not" in node else [])
    return 1 + sum(_count(c) for c in children)


def _eval(node: Any, close: pd.DataFrame, depth: int = 0) -> pd.DataFrame:
    if depth > MAX_DEPTH:
        raise InvalidStrategy("estratégia profunda demais")
    if not isinstance(node, dict) or not node:
        raise InvalidStrategy("condição precisa ser um objeto")
    if "all" in node or "any" in node:
        children = node.get("all") or node.get("any")
        if not isinstance(children, list) or not children:
            raise InvalidStrategy("all/any precisa de uma lista não vazia")
        parts = [_eval(c, close, depth + 1) for c in children]
        out = parts[0]
        for p in parts[1:]:
            out = np.minimum(out, p) if "all" in node else np.maximum(out, p)
        return out
    if "not" in node:
        return 1.0 - _eval(node["not"], close, depth + 1)
    return _piece(node, close).astype(float)


def compile_strategy(spec: dict, close: pd.DataFrame) -> pd.DataFrame:
    """Pesos por ativo (pesos iguais na carteira), decididos no fechamento do dia (aplicar com shift(1))."""
    if _count(spec.get("entry", {})) > MAX_NODES:
        raise InvalidStrategy("estratégia grande demais")
    signal = pd.DataFrame(_eval(spec["entry"], close), index=close.index, columns=close.columns).fillna(0.0)
    vt = (spec.get("size") or {}).get("vol_target")
    if vt is not None:
        vt = _num(vt, "p")
        vol = close.pct_change().rolling(20).std() * np.sqrt(DAYS)
        signal = signal * (vt / vol).clip(upper=1.0).fillna(0.0)
    # pesos iguais entre as moedas que já existiam naquela data (SOL só a partir de 2020, etc.)
    return signal.div(close.notna().sum(axis=1).clip(lower=1), axis=0)
