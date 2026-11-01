"""Cache dos sinais black: calculados UMA vez por hora num processo só (a coleta), lidos pelos forks.

Calcular os sinais black é caro (~1 GB e ~1,5 min); cada fork recalculando para cada ativo a cada
geração estourava a RAM. O cache guarda a grade dos últimos 31 dias; o fork só lê e alinha pelo
tempo (asof para trás: cada linha usa o último valor calculado até ali, nunca um do futuro).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trader.storage.atomic import replace_with_retry, unique_tmp

DAYS = 31


def cache_path(data: Path, symbol: str, timeframe: str, weighted: bool) -> Path:
    return Path(data) / "features" / "black" / f"{symbol.replace('/', '')}_{timeframe}_{'rw' if weighted else 'raw'}.parquet"


def build(data: Path, symbols: list[str], timeframes: list[str], log=print) -> None:
    from trader.blacknews.api import black_features

    now = pd.Timestamp.now(tz="UTC")
    for tf in timeframes:
        index = pd.date_range((now - pd.Timedelta(days=DAYS)).floor(tf.replace("m", "min")), now.floor(tf.replace("m", "min")),
                              freq=tf.replace("m", "min"))
        for symbol in symbols:
            for weighted in (True, False):
                frame = black_features(symbol, tf, index, weighted=weighted, data=data)
                path = cache_path(data, symbol, tf, weighted)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = unique_tmp(path)
                frame.rename_axis("ts").reset_index().to_parquet(tmp, index=False)
                replace_with_retry(tmp, path)
            log(f"cache black {symbol} {tf}: {len(index)} linhas")


def read(data: Path, symbol: str, timeframe: str, weighted: bool, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sinais black alinhados ao índice; sem cache, devolve vazio (o fork segue sem black, sem travar)."""
    path = cache_path(data, symbol, timeframe, weighted)
    if not path.exists():
        return pd.DataFrame(index=index)
    cached = pd.read_parquet(path).sort_values("ts")
    left = pd.DataFrame({"ts": pd.DatetimeIndex(index)}).sort_values("ts")
    joined = pd.merge_asof(left, cached, on="ts", direction="backward")
    return joined.set_index("ts").reindex(index).fillna(0.0)


def main() -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="trader.blacknews.cache")
    p.add_argument("--symbols", default="SOL/USDT,ETH/USDT,XRP/USDT")
    p.add_argument("--timeframes", default="5m,1m")
    a = p.parse_args()
    build(Path("data"), a.symbols.split(","), a.timeframes.split(","))


if __name__ == "__main__":
    main()
