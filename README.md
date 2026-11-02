# AI Trader

A self-hosted research and paper-trading system for crypto (Binance) and Brazilian markets (B3).
Prices and timestamped news become a point-in-time market view. A swappable engine turns that view
into a signal, a frozen risk gate filters it, and a broker adapter executes it. Most of the code
exists to answer one question honestly: does a strategy still work on data it never saw?

Status: research project, single author, paper trading only. No real money has been traded.

## Layout

| Path | Purpose |
| --- | --- |
| `trader/core`, `trader/domain.py` | The `Engine.decide(MarketView) -> Signal` contract and shared types |
| `trader/risk`, `trader/execution` | Frozen risk limits and broker adapters |
| `trader/ingest`, `trader/storage` | Binance and B3 ingestion, Parquet candle store |
| `trader/evolution`, `trader/daytrade` | Evolutionary agents with autopsies, successors and a holdout |
| `trader/news`, `trader/blacknews` | News pipeline with source reputation and no lookahead |
| `trader/realtime`, `trader/policy` | Millisecond reflexes that act only within an LLM-written JSON policy |
| `trader/live`, `trader/panel`, `trader/chat` | Paper runners, local web panel and chat |
| `trader/research` | One script per experiment, including the rejected ones |
| `trader/lob` | From-scratch price-time matching engine, write-ahead journal with crash recovery, L2 book reconstruction, queue-position model and execution simulator |
| `trader/b3` | WIN/WDO mini-contract specs, rollover, continuous series, costs |
| `docs/HISTORICO.md` | Decision log in Portuguese: every experiment with its holdout numbers |

## Design rules

- **No lookahead.** A candle is visible only after it closes. News is visible only after publication plus latency.
- **Holdout first.** The final period never takes part in selection. Failed ideas stay recorded.
- **Risk outside evolution.** Agents cannot change the risk limits.
- **Replaceable core.** Engines, brokers and LLM providers sit behind contracts.

## Setup

Developed and tested on Windows with Python 3.12. Linux is untested. The live scripts in `scripts/` are Windows-only.
The optional LLM features use a local [Ollama](https://ollama.com) server.

```
pipenv install --dev
cp .env.example .env
```

Market data, models and logs live in `data/`, which is not part of this repository. Create it by
collecting data:

```
python -m trader collect BTC/USDT 1d --since 2020-01-01
python -m trader evolve BTC/USDT 1d --seeds 0,1,2
python -m trader.research.rf_daily
python -m trader.panel
```

The panel serves http://127.0.0.1:8765.

To record the public Binance futures book and compare limit against market execution:

```
python -m trader.lob record SOLUSDT --seconds 3600
python -m trader.lob study data/lob/<file>.jsonl.gz --usd 1000
```

## Tests

```
python -m pytest
```

The suite has 362 tests, needs no `data/` folder or `.env`, and takes about two minutes.
Coverage was 65% of `trader/` before `trader/lob` was added. Command-line entry points and one-off research scripts account for most of the gap.
CI runs the suite on every pull request.
