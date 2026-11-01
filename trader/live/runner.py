"""Evolução ao vivo em simulação: a população opera os candles que acabaram de
fechar e se retroalimenta (morte, autópsia, sucessor pelo Criador) sem parar.

NUNCA envia ordens reais: não há corretora real nem chaves de API neste
caminho. As operações passam pelo mesmo backtest barra a barra (decide no
fechamento, executa na abertura seguinte, com taxa), aplicado a cada janela
assim que ela fecha — o resultado é idêntico ao de operar a janela ao vivo
em papel, só que calculado ao fim dela.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from trader.domain import parse_timeframe
from trader.evolution.creator import Creator, LLMCreator
from trader.evolution.genome import Genome
from trader.evolution.population import Population, PopulationConfig, make_executor
from trader.evolution.registry import Registry
from trader.ingest import binance
from trader.storage.candles import CandleStore

REAL_ORDERS_ENABLED = False  # trava: este módulo só opera em papel


@dataclass(frozen=True)
class LiveConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    window: str = "3h"  # uma geração a cada janela fechada
    history_days: int = 30  # candles mantidos para aquecimento dos indicadores
    llm_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    workers: int = 2
    poll_seconds: int = 60
    features: str = "none"  # none | white | all (white + black)
    extra_symbols: tuple[str, ...] = ()  # outros ativos operados pelo mesmo genoma (mais entradas, menos sorte)
    memory: tuple[str, ...] = ()  # registros de evoluções no histórico: galeria e cemitério viram memória do Criador


def attach_features(candles: pd.DataFrame, symbol: str, timeframe: str, kind: str, data: Path) -> pd.DataFrame:
    """Anexa as features de notícias ao fechamento de cada candle (causal: só o visível até ali)."""
    if kind == "none" or candles.empty:
        return candles
    from trader.news.features import news_features

    close = pd.DatetimeIndex(candles["ts"] + parse_timeframe(timeframe))
    parts = [news_features(symbol, timeframe, close, data=data)]
    if kind == "all":
        # sinais black vêm do cache horário (calculados uma vez só na coleta); calcular aqui estourava a RAM
        from trader.blacknews import cache

        parts.append(cache.read(data, symbol, timeframe, True, close))
        parts.append(cache.read(data, symbol, timeframe, False, close))
    extra = pd.concat(parts, axis=1)
    extra = extra.loc[:, ~extra.columns.duplicated()]
    extra.index = candles.index
    return pd.concat([candles, extra], axis=1)


def history_registries(
    sources: tuple[str, ...], symbol: str, timeframe: str, fee: float = 0.001
) -> list[Registry]:
    """Registros de evoluções no histórico do mesmo ativo, tempo gráfico e taxa (memória de outra taxa engana)."""
    paths: list[Path] = []
    for src in map(Path, sources):
        paths += sorted(src.glob("*.sqlite")) if src.is_dir() else [src] if src.exists() else []
    out = []
    for path in paths:
        reg = Registry(path)
        run = reg.meta().get("run", {})
        if run.get("symbol") == symbol and run.get("timeframe") == timeframe and run.get("fee", 0.001) == fee:
            out.append(reg)
    return out


def load_library(data: Path, config: "LiveConfig"):
    """RAG dos livros, se o índice existir (python -m trader.books index)."""
    index = data / "books" / "index"
    if not (index / "chunks.parquet").exists():
        return None
    from trader.books.agentic import AgenticLibrary
    from trader.books.index import Library

    return AgenticLibrary(Library.load(index), base_url=config.llm_url, model=config.llm_model)


def _priors(data: Path) -> dict:
    from trader.evolution.priors import gene_priors

    return gene_priors(data / "books" / "playbook.json")


def floor_to(ts: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
    epoch = pd.Timestamp(0, tz="UTC")
    return epoch + ((ts - epoch) // step) * step


def closed_until(candles: pd.DataFrame, timeframe: str) -> pd.Timestamp | None:
    """Instante até o qual há candles fechados."""
    if candles.empty:
        return None
    return candles["ts"].iloc[-1] + parse_timeframe(timeframe)


def bootstrap_genomes(path: Path | None) -> list[Genome] | None:
    """Sobreviventes de um registro de evolução anterior, para não começar do zero."""
    if path is None:
        return None
    return [Genome.from_dict(a["genome"]) for a in Registry(path).alive()] or None


class LiveEvolution:
    def __init__(
        self,
        data: Path,
        db_path: Path,
        config: LiveConfig = LiveConfig(),
        creator: Creator | None = None,
        fetch: Callable[[CandleStore, str, str, pd.Timestamp], int] = binance.update,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        log: Callable[[str], None] = print,
    ) -> None:
        assert not REAL_ORDERS_ENABLED
        self.config = config
        self.data = data
        self.store = CandleStore(data)
        self.registry = Registry(db_path)
        self.fetch, self.now, self.log = fetch, now, log
        self.archives = history_registries(config.memory, config.symbol, config.timeframe)
        self.creator = creator or LLMCreator(
            config.symbol, config.timeframe, config.llm_url, config.llm_model,
            graveyard=self.graveyard, hall=self.hall, library=load_library(data, config),
        )
        self.step_len = pd.Timedelta(config.window)
        self.population: Population | None = None

    def hall(self, n: int = 5) -> list[dict]:
        """Galeria ao vivo + a do histórico, melhores primeiro (marcando a origem)."""
        entries = [{**h, "from": "live"} for h in self.registry.hall(n)]
        for archive in self.archives:
            entries += [{**h, "from": "history"} for h in archive.hall(n)]
        return sorted(entries, key=lambda h: h["score"], reverse=True)[:n]

    def graveyard(self, n: int = 10) -> list[dict]:
        """Mortes recentes ao vivo primeiro; completa com as do histórico."""
        live = [{**g, "from": "live"} for g in self.registry.graveyard(n)]
        extra = [{**g, "from": "history"} for a in self.archives for g in a.graveyard(n)]
        return (live + extra)[:n]

    @property
    def symbols(self) -> list[str]:
        return [self.config.symbol, *[s for s in self.config.extra_symbols if s != self.config.symbol]]

    def _candles(self) -> dict[str, pd.DataFrame]:
        start = self.now() - pd.Timedelta(days=self.config.history_days)
        out = {}
        for symbol in self.symbols:
            self.fetch(self.store, symbol, self.config.timeframe, start)
            out[symbol] = self.store.read(binance.EXCHANGE, symbol, self.config.timeframe, start=start)
        return out

    def _closed_until(self, baskets: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
        """Só avança quando TODOS os ativos fecharam o candle (a carteira é avaliada junta)."""
        ends = [closed_until(c, self.config.timeframe) for c in baskets.values()]
        return None if any(e is None for e in ends) else min(ends)

    def start(self, bootstrap: Path | None = None, executor=None) -> None:
        candles = self._candles()
        self.population = Population(
            candles, self.config.symbol, self.config.timeframe, self.registry, self.creator,
            PopulationConfig(priors=_priors(self.data)), executor=executor,
        )
        if hasattr(self.creator, "lessons"):
            from trader.evolution.priors import standing_lessons

            standing = standing_lessons(self.data / "books" / "playbook.json")
            # lições destiladas da população + lições permanentes (livros, ou hipóteses iniciais)
            self.creator.lessons = lambda: self.population.lessons[:10] + standing
        if self.population.restore():
            self.log(f"retomado: {len(self.population.alive)} agentes vivos, janela {self.population._window}")
        else:
            self.population.seed(bootstrap_genomes(bootstrap))
            self.registry.set_meta("live", {"symbol": self.config.symbol, "timeframe": self.config.timeframe,
                                            "window": self.config.window, "started": str(self.now())})
            self.log(f"população nova: {len(self.population.alive)} agentes")
        if "next_start" not in self.registry.meta():
            self.registry.set_meta("next_start", str(floor_to(self.now(), self.step_len)))

    def tick(self) -> int:
        """Processa todas as janelas já fechadas. Retorna quantas gerações rodaram."""
        pop = self.population
        candles = self._candles()
        done = self._closed_until(candles)
        start = pd.Timestamp(self.registry.meta()["next_start"])
        ran = 0
        if done is None or start + self.step_len > done:
            return 0
        # só calcula notícias quando alguma janela fechou (custa mais que buscar candles)
        pop.candles = {
            sym: attach_features(c, sym, self.config.timeframe, self.config.features, self.data)
            for sym, c in candles.items()
        }
        while start + self.step_len <= done:
            end = start + self.step_len
            r = pop.step(start, end)
            self.registry.set_meta("next_start", str(end))
            deaths = " ".join(f"{k}={v}" for k, v in sorted(r.deaths.items())) or "-"
            births = " ".join(f"{k}={v}" for k, v in sorted(r.births.items())) or "-"
            self.log(
                f"{end:%Y-%m-%d %H:%M} UTC g{r.index:04d} vivos={r.alive} mortes[{deaths}] nascimentos[{births}]"
                f" melhor={r.best_id} {r.best_ret:+.2%} nota={r.best_score:.2f} ref={r.baseline_ret:+.2%}"
            )
            start, ran = end, ran + 1
        return ran

    def run_forever(self, max_consecutive_errors: int = 10) -> None:
        errors = 0
        while True:
            try:
                self.tick()
                errors = 0
            except Exception as exc:  # rede, exchange ou LLM fora: tenta de novo no próximo ciclo
                errors += 1
                self.log(f"erro no ciclo ({type(exc).__name__}: {exc}); nova tentativa em {self.config.poll_seconds}s")
                if errors >= max_consecutive_errors:
                    # erro persistente (ex.: código novo com módulo antigo carregado): sai para o .bat
                    # religar o processo do zero, com o código atualizado
                    self.log(f"{errors} erros seguidos: encerrando para reiniciar limpo")
                    raise SystemExit(1)
            time.sleep(self.config.poll_seconds)


def run(data: Path, db_path: Path, config: LiveConfig, bootstrap: Path | None) -> None:
    executor = make_executor(config.workers)
    live = LiveEvolution(data, db_path, config, log=lambda m: print(m, flush=True))
    live.start(bootstrap, executor)
    live.run_forever()
