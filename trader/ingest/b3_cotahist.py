"""Leitor do COTAHIST (cotações históricas diárias da B3, mercado à vista e opções).

Layout oficial de largura fixa, 245 caracteres por linha. Não contém futuros
(WIN, WDO), que são do segmento BM&F.

Preços vêm brutos. Desdobramentos e grupamentos são detectados pelo salto de
abertura e ajustados na importação; proventos (dividendos) não são ajustados.
O candle diário usa ts = data 00:00 UTC; só fica visível para decisões a partir
do dia seguinte, depois do fechamento do pregão.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from trader.domain import CANDLE_COLUMNS
from trader.storage.candles import CandleStore

EXCHANGE = "b3"
TIMEFRAME = "1d"
MARKET_SPOT = "010"  # TPMERC: mercado à vista

# (início, fim) 0-based, fim exclusivo, conforme o layout oficial
_DATE = (2, 10)
_TICKER = (12, 24)
_MARKET = (24, 27)
_OPEN = (56, 69)
_HIGH = (69, 82)
_LOW = (82, 95)
_CLOSE = (108, 121)
_QUANTITY = (152, 170)
_FACTOR = (210, 217)


def _field(line: str, span: tuple[int, int]) -> str:
    return line[span[0] : span[1]]


def parse_lines(lines: io.TextIOBase | list[str], market: str = MARKET_SPOT) -> pd.DataFrame:
    """Candles diários de todos os papéis do mercado pedido, com coluna `symbol`."""
    rows = []
    for line in lines:
        if not line.startswith("01") or _field(line, _MARKET) != market:
            continue
        factor = int(_field(line, _FACTOR)) or 1
        scale = 100 * factor  # preços vêm com 2 casas implícitas, por lote de `factor`
        rows.append(
            (
                _field(line, _TICKER).strip(),
                _field(line, _DATE),
                int(_field(line, _OPEN)) / scale,
                int(_field(line, _HIGH)) / scale,
                int(_field(line, _LOW)) / scale,
                int(_field(line, _CLOSE)) / scale,
                float(int(_field(line, _QUANTITY))),
            )
        )
    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df.pop("date"), format="%Y%m%d").dt.tz_localize("UTC")
    return df[["symbol", *CANDLE_COLUMNS]]


def read_file(path: Path | str, market: str = MARKET_SPOT) -> pd.DataFrame:
    """Aceita o .ZIP baixado da B3 ou o .TXT extraído."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.upper().endswith(".TXT"))
            with z.open(name) as raw:
                return parse_lines(io.TextIOWrapper(raw, encoding="latin-1"), market)
    with open(path, encoding="latin-1") as f:
        return parse_lines(f, market)


SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 20, 50, 100)


def adjust_splits(candles: pd.DataFrame, tolerance: float = 0.02) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, float]]]:
    """Ajusta preços anteriores a desdobramentos/grupamentos detectados.

    Evento = abertura do dia ≈ fechamento anterior / k (desdobramento) ou × k
    (grupamento), com k inteiro conhecido. Retorna (candles ajustados, eventos),
    onde cada evento é (data, fator aplicado aos preços anteriores).
    """
    df = candles.sort_values("ts").reset_index(drop=True).copy()
    gap = df["open"] / df["close"].shift()
    events = []
    for i in range(1, len(df)):
        g = gap.iloc[i]
        for k in SPLIT_RATIOS:
            for factor in (1 / k, float(k)):
                if abs(g / factor - 1) < tolerance:
                    events.append((df.at[i, "ts"], factor))
                    break
            else:
                continue
            break
    for ts, factor in events:
        before = df["ts"] < ts
        df.loc[before, ["open", "high", "low", "close"]] *= factor
        df.loc[before, "volume"] /= factor
    return df, events


def import_files(
    store: CandleStore, paths: list[Path], tickers: list[str] | None = None
) -> tuple[dict[str, int], dict[str, list[tuple[pd.Timestamp, float]]]]:
    """Grava cada papel no CandleStore (exchange 'b3', timeframe '1d').

    Retorna (pregões por papel, eventos de desdobramento ajustados por papel).
    Importe todos os anos juntos: o ajuste de desdobramento só alcança os preços
    presentes nos arquivos desta chamada.
    """
    frames = [read_file(p) for p in paths]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", *CANDLE_COLUMNS])
    if tickers:
        data = data[data["symbol"].isin(tickers)]
    counts, events = {}, {}
    for symbol, part in data.groupby("symbol"):
        adjusted, found = adjust_splits(part.drop(columns="symbol"))
        store.write(EXCHANGE, symbol, TIMEFRAME, adjusted)
        counts[symbol] = len(part)
        if found:
            events[symbol] = found
    return counts, events
