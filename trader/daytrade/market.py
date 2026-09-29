"""Regras de ordem mínima dos perpétuos USDT-M e conversão do capital em reais.

Na Binance, uma ordem precisa cumprir duas regras ao mesmo tempo: o valor
nocional mínimo (MIN_NOTIONAL) e a quantidade mínima (MARKET_LOT_SIZE.minQty),
que também é o passo da quantidade. No BTC a segunda domina: 0,001 BTC custa
bem mais que os 50 USDT da regra de nocional.

Valores conferidos em 2026-09-29 via GET /fapi/v1/exchangeInfo (ccxt
load_markets). Use `live_specs()` para atualizar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    min_notional: float  # USDT
    qty_step: float  # quantidade mínima e incremento

    def min_order_value(self, price: float) -> float:
        """Menor nocional que dá para enviar a esse preço."""
        min_qty = max(self.qty_step, math.ceil(self.min_notional / price / self.qty_step) * self.qty_step)
        return min_qty * price

    def round_qty(self, qty: float) -> float:
        return math.floor(qty / self.qty_step + 1e-9) * self.qty_step


SPECS: dict[str, SymbolSpec] = {
    "BTC/USDT": SymbolSpec(50.0, 0.001),
    "ETH/USDT": SymbolSpec(20.0, 0.001),
    "SOL/USDT": SymbolSpec(5.0, 0.01),
    "BNB/USDT": SymbolSpec(5.0, 0.01),
    "XRP/USDT": SymbolSpec(5.0, 0.1),
    "DOGE/USDT": SymbolSpec(5.0, 1.0),
    "ADA/USDT": SymbolSpec(5.0, 1.0),
    "TRX/USDT": SymbolSpec(5.0, 1.0),
    "LTC/USDT": SymbolSpec(20.0, 0.001),
}
# sem ordem mínima: para testes com dados sintéticos
NO_MINIMUM = SymbolSpec(0.0, 1e-12)


def spec_for(symbol: str) -> SymbolSpec:
    return SPECS.get(symbol, NO_MINIMUM)


def brl_to_usdt(brl: float, rate: float) -> float:
    return brl / rate


def feasibility(symbol: str, capital_usdt: float, max_leverage: float, price: float) -> tuple[bool, str]:
    """Dá para abrir pelo menos uma ordem com esse capital e esse teto de alavancagem?"""
    spec = spec_for(symbol)
    need = spec.min_order_value(price)
    can = capital_usdt * max_leverage * 0.98
    ok = can >= need
    msg = (
        f"{symbol}: menor ordem ≈ {need:.2f} USDT (mín. {spec.min_notional:g} USDT, lote {spec.qty_step:g});"
        f" capital {capital_usdt:.2f} USDT × teto {max_leverage:g}x = {can:.2f} USDT → "
        + ("viável" if ok else f"INVIÁVEL (precisaria de {need / capital_usdt / 0.98:.1f}x ou mais capital)")
    )
    return ok, msg


def live_specs(symbols: list[str]) -> dict[str, SymbolSpec]:
    """Lê as regras atuais na Binance (endpoint público)."""
    import ccxt

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    markets = ex.load_markets()
    out = {}
    for s in symbols:
        filters = {f["filterType"]: f for f in markets[f"{s}:USDT"]["info"]["filters"]}
        out[s] = SymbolSpec(float(filters["MIN_NOTIONAL"]["notional"]), float(filters["MARKET_LOT_SIZE"]["minQty"]))
    return out
