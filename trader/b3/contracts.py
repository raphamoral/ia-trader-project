"""Minicontratos da B3: especificação, códigos de vencimento e série contínua.

WIN (mini índice): vence na quarta-feira mais próxima do dia 15 dos meses pares.
WDO (mini dólar): vence no 1º dia útil do mês do código; negocia-se o vencimento
seguinte já no fim do mês anterior.

Os valores monetários (ponto, custos, margem) são parâmetros: confira com a
corretora/B3 antes de confiar em números de dinheiro real.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"  # jan..dez


@dataclass(frozen=True)
class FutureSpec:
    root: str
    point_value: float  # R$ por ponto por contrato
    tick: float  # variação mínima em pontos
    months: str  # códigos de vencimento negociados
    roll_days: int  # troca para o próximo vencimento N dias úteis antes do vencimento


SPECS = {
    "WIN": FutureSpec("WIN", point_value=0.20, tick=5.0, months="GJMQVZ", roll_days=3),
    "WDO": FutureSpec("WDO", point_value=10.0, tick=0.5, months=MONTH_CODES, roll_days=2),
}


def business_days(start: pd.Timestamp, end: pd.Timestamp, holidays: set[pd.Timestamp] | None = None) -> pd.DatetimeIndex:
    days = pd.bdate_range(start.normalize(), end.normalize())
    return days if not holidays else days[~days.isin(list(holidays))]


def expiry(root: str, year: int, month: int, holidays: set[pd.Timestamp] | None = None) -> pd.Timestamp:
    """Data de vencimento do contrato `root` do mês/ano dado."""
    holidays = holidays or set()
    if root == "WIN":
        mid = pd.Timestamp(year, month, 15)
        weds = [mid + pd.Timedelta(days=d) for d in range(-7, 8) if (mid + pd.Timedelta(days=d)).weekday() == 2]
        day = min(weds, key=lambda d: abs((d - mid).days))
        while day in holidays:  # feriado: vence no dia útil seguinte
            day += pd.offsets.BDay(1)
        return day
    if root == "WDO":
        first = business_days(pd.Timestamp(year, month, 1), pd.Timestamp(year, month, 10), holidays)
        return first[0]
    raise ValueError(f"contrato desconhecido: {root}")


def code(root: str, year: int, month: int) -> str:
    """Ex.: WIN, 2026, 10 → 'WINV26'."""
    return f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}"


def active_contract(root: str, when: pd.Timestamp, holidays: set[pd.Timestamp] | None = None) -> str:
    """Contrato mais negociado no dia: o próximo vencimento, trocando `roll_days` úteis antes dele."""
    spec = SPECS[root]
    day = pd.Timestamp(when).tz_localize(None).normalize() if pd.Timestamp(when).tzinfo else pd.Timestamp(when).normalize()
    for offset in range(0, 14):
        y, m = divmod(day.month - 1 + offset, 12)
        year, month = day.year + y, m + 1
        if MONTH_CODES[month - 1] not in spec.months:
            continue
        exp = expiry(root, year, month, holidays)
        window = business_days(day, exp, holidays)
        if len(window) > spec.roll_days:  # ainda não chegou na janela de troca
            return code(root, year, month)
    raise AssertionError("sem vencimento encontrado")


def continuous(candles_by_code: dict[str, pd.DataFrame], root: str, adjust: bool = True,
               holidays: set[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Série contínua: em cada candle usa o contrato ativo; com `adjust`, remove o salto na troca
    (ajuste retroativo por diferença), para os indicadores não verem um 'crash' que não existiu."""
    parts = []
    for c, df in candles_by_code.items():
        if df.empty:
            continue
        local = df["ts"].dt.tz_convert("America/Sao_Paulo") if df["ts"].dt.tz is not None else df["ts"]
        active = local.dt.normalize().map(lambda d: active_contract(root, d, holidays))
        parts.append(df[(active == c).to_numpy()].assign(contract=c))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_values("ts").reset_index(drop=True)
    if adjust:
        rolls = out.index[out["contract"] != out["contract"].shift()][1:]
        for i in reversed(list(rolls)):
            # diferença entre os vencimentos NO MESMO instante (último candle do contrato antigo);
            # sem esse candle no contrato novo, cai para abertura − fechamento (pode apagar um gap real)
            ts_old, new = out.at[i - 1, "ts"], candles_by_code[out.at[i, "contract"]]
            same = new.loc[new["ts"] == ts_old, "close"]
            gap = (same.iloc[0] if len(same) else out.at[i, "open"]) - out.at[i - 1, "close"]
            out.loc[: i - 1, ["open", "high", "low", "close"]] += gap
    return out
