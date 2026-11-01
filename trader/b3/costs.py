"""Custos e risco dos minicontratos (day trade). Valores em R$ por contrato, por lado.

Os padrões são aproximações para simulação; confira na sua conta BTG e na tabela
de tarifas da B3 e ajuste aqui. O que a simulação não pode é ignorar custo.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.b3.contracts import SPECS


@dataclass(frozen=True)
class B3Costs:
    exchange_fee: float  # emolumentos + registro (B3), por contrato, por lado
    brokerage: float = 0.0  # corretagem (muitas corretoras zeram no day trade de mini)
    slippage_ticks: float = 1.0  # escorregamento pessimista por execução a mercado

    def per_side(self, root: str, contracts: int) -> float:
        spec = SPECS[root]
        return contracts * (self.exchange_fee + self.brokerage + self.slippage_ticks * spec.tick * spec.point_value)


DEFAULT_COSTS = {"WIN": B3Costs(exchange_fee=0.30), "WDO": B3Costs(exchange_fee=1.20)}


@dataclass(frozen=True)
class B3Risk:
    """Limites congelados do day trade na B3 (fora da evolução)."""

    margin_per_contract: float  # margem de day trade exigida pela corretora
    max_contracts: int = 1  # com capital pequeno: 1 contrato
    daily_loss_limit: float = 0.05  # fração do capital no início do dia
    gap_scenario: float = 0.03  # gap adverso de 3% do preço para testar saldo negativo
    session_open: str = "09:00"
    flatten_at: str = "18:00"  # zera antes da zeragem compulsória da corretora (horário de Brasília)

    def contracts_allowed(self, root: str, equity: float, price: float, wanted: int) -> int:
        """Quantos contratos cabem sem arriscar ficar NEGATIVO num gap adverso."""
        spec = SPECS[root]
        worst_per_contract = price * self.gap_scenario * spec.point_value  # pontos do gap × R$ por ponto
        by_margin = int(equity // self.margin_per_contract) if self.margin_per_contract > 0 else wanted
        by_gap = int(equity // worst_per_contract) if worst_per_contract > 0 else wanted
        return max(0, min(wanted, self.max_contracts, by_margin, by_gap))

    def in_session(self, ts: pd.Timestamp) -> bool:
        local = ts.tz_convert("America/Sao_Paulo") if ts.tzinfo else ts
        hm = local.strftime("%H:%M")
        return local.weekday() < 5 and self.session_open <= hm < self.flatten_at


DEFAULT_RISK = {"WIN": B3Risk(margin_per_contract=100.0), "WDO": B3Risk(margin_per_contract=150.0)}
