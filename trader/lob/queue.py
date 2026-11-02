"""Posição na fila de uma ordem nossa, estimada só com dados públicos.

O L2 mostra o total do nível, não quem saiu. A ordem nossa é hipotética: entra no fim da
fila do preço e só avança quando quem está à frente sai.

- Negócio no nosso preço consome a frente da fila primeiro. O que sobra nos executa.
- Negócio além do nosso preço prova que o nível foi varrido: executa tudo.
- Queda do total sem negócio é cancelamento. Não dá para saber de que parte da fila veio, então
  dividimos pela probabilidade f(frente) / (f(frente) + f(trás)), com f(x) = log(1 + x).
- Aumento do total entra atrás de nós.

Limitação conhecida: se a atualização do livro chega antes do negócio que a causou, o mesmo
volume conta como cancelamento e como negócio, o que adianta a fila e favorece a ordem passiva.
Os horários dos dois streams não têm ordem garantida, então trate o resultado como teto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QueuePosition:
    is_bid: bool
    price: int
    qty: int
    ahead: float  # quantidade à nossa frente
    level_total: int  # último total visto no nível
    filled: int = 0
    _unexplained_trades: int = 0  # negócios ainda não refletidos em atualização do livro

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    def on_trade(self, price: int, qty: int, seller_aggressed: bool) -> int:
        """Devolve quanto da nossa ordem executou neste negócio."""
        if self.is_bid != seller_aggressed or not self.remaining:
            return 0  # só a agressão contrária atinge uma ordem passiva
        beyond = price < self.price if self.is_bid else price > self.price
        if beyond:
            fill = self.remaining
            self.ahead = 0.0
        elif price == self.price:
            fill = min(self.remaining, max(0, math.floor(qty - self.ahead)))
            self.ahead = max(0.0, self.ahead - qty)
            self._unexplained_trades += qty
        else:
            return 0
        self.filled += fill
        return fill

    def on_level(self, total: int) -> None:
        """Novo total do nível no L2."""
        drop = self.level_total - total
        self.level_total = total
        if drop > 0:
            explained = min(drop, self._unexplained_trades)
            self._unexplained_trades -= explained
            canceled = drop - explained
            if canceled:
                behind = max(0.0, total + canceled - self.ahead)
                front = math.log1p(self.ahead)
                back = math.log1p(behind)
                share = front / (front + back) if front + back else 0.0
                self.ahead = max(0.0, self.ahead - canceled * share)
        self.ahead = min(self.ahead, float(total))
