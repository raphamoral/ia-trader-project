"""Evolução ao vivo em papel, 24h, da espécie de day trade (mesmo contrato de trader/live/runner.py).

NENHUMA ordem real é enviada: REAL_ORDERS_ENABLED fica False e é verificado
na construção. A cada janela fechada em TODOS os ativos, roda uma geração com
os candles fechados dos últimos 30 dias; tudo fica no Registry compartilhado
(agents, life, hall, meta.next_start), então o painel lê sem mudanças e o
processo retoma de onde parou se reiniciar.

Janelas de geração:
  day   5m, 3h  = 36 candles por ativo × 3 ativos. Agentes de day trade fazem
        de 1 a 4 entradas por dia e por ativo; 3h dá ~1-3 operações por
        janela na carteira, e o veredito exige 8 operações e 3 janelas de
        idade: um agente é julgado em ~9-12h, não por sorte de uma entrada.
  scalp 1m, 1h  = 60 candles por ativo. Scalpers fazem 5-20 tentativas por
        dia; 1h dá algumas por janela e o veredito sai em poucas horas, curto
        o bastante para acompanhar mudanças de regime intradiárias.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from trader.daytrade import data as dt_data
from trader.daytrade.creator import DayLLMCreator
from trader.evolution.priors import standing_lessons
from trader.live.runner import load_library
from trader.daytrade.evolution import DayConfig, DayPopulation
from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.genome import STYLES, DayGenome
from trader.daytrade.market import brl_to_usdt, feasibility, spec_for
from trader.domain import parse_timeframe
from trader.evolution.fitness import SelectionRules
from trader.evolution.registry import Registry
from trader.storage.candles import CandleStore

REAL_ORDERS_ENABLED = False  # trava: este módulo só opera em papel

DAY_STYLES = tuple(s for s in STYLES if STYLES[s]["timeframe"] == "5m" and s not in ("news", "any"))
LIVE_STYLES = {
    "day": {"timeframe": "5m", "window": "3h", "styles": DAY_STYLES},
    "scalp": {"timeframe": "1m", "window": "1h", "styles": ("scalping",)},
}


@dataclass(frozen=True)
class DayLiveConfig:
    style: str = "day"  # day | scalp
    symbol: str = "SOL/USDT"
    extra_symbols: tuple[str, ...] = ()
    features: str = "none"  # none | white | all
    history_days: int = 30
    capital_brl: float = 1000.0
    brl_rate: float = 5.5
    max_leverage: float = 3.0
    llm_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm: bool = True
    workers: int = 2
    poll_seconds: int = 30
    population: int = 16
    slots: int = 24

    @property
    def timeframe(self) -> str:
        return LIVE_STYLES[self.style]["timeframe"]

    @property
    def window(self) -> str:
        return LIVE_STYLES[self.style]["window"]

    @property
    def styles(self) -> tuple[str, ...]:
        base = LIVE_STYLES[self.style]["styles"]
        return (*base, "news") if self.style == "day" and self.features != "none" else base


def floor_to(ts: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
    epoch = pd.Timestamp(0, tz="UTC")
    return epoch + ((ts - epoch) // step) * step


def closed_until(candles: pd.DataFrame, timeframe: str) -> pd.Timestamp | None:
    return None if candles.empty else candles["ts"].iloc[-1] + parse_timeframe(timeframe)


def bootstrap_genomes(path: Path | None, styles: tuple[str, ...]) -> list[DayGenome] | None:
    """Sobreviventes de uma evolução no histórico com estilo compatível."""
    if path is None:
        return None
    out = []
    for a in Registry(path).alive():
        try:
            g = DayGenome.from_dict(a["genome"])
        except (KeyError, ValueError):
            continue  # registro de outra espécie
        if g.style in styles:
            out.append(g)
    return out or None


def _update(store: CandleStore, symbol: str, timeframe: str, start: pd.Timestamp) -> int:
    return dt_data.update_candles(store, symbol, timeframe, start)


class DayLiveEvolution:
    def __init__(
        self,
        data: Path,
        db_path: Path,
        config: DayLiveConfig = DayLiveConfig(),
        creator=None,
        fetch: Callable[[CandleStore, str, str, pd.Timestamp], int] = _update,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        log: Callable[[str], None] = print,
    ) -> None:
        assert not REAL_ORDERS_ENABLED, "day trade ao vivo só opera em papel"
        self.config, self.data = config, data
        self.store = CandleStore(data)
        self.registry = Registry(db_path)
        self.fetch, self.now, self.log = fetch, now, log
        self.cash = brl_to_usdt(config.capital_brl, config.brl_rate)
        self.step_len = pd.Timedelta(config.window)
        self.population: DayPopulation | None = None
        if creator is None and config.llm:
            creator = DayLLMCreator(
                config.symbol, config.timeframe,
                {"capital": f"R${config.capital_brl:g} ({self.cash:.2f} USDT)", "max_lev": f"{config.max_leverage:g}",
                 "max_risk": "0.02", "daily_loss": "0.05", "min_order": "5",
                 "goal": "consistency: positive risk-adjusted return after fees, window after window"},
                config.llm_url, config.llm_model,
                graveyard=lambda: self.registry.graveyard(8), hall=lambda: self.registry.hall(3),
                lessons=lambda: standing_lessons(Path(data) / "books" / "playbook.json"),
                library=load_library(Path(data), config),
            )
        self.creator = creator
        self._symbols: list[str] | None = None

    @property
    def symbols(self) -> list[str]:
        """Ativos operáveis com o capital (a ordem mínima da corretora pode excluir alguns)."""
        if self._symbols is None:
            wanted = [self.config.symbol, *[s for s in self.config.extra_symbols if s != self.config.symbol]]
            self._symbols = []
            for s in wanted:
                last = self.store.read(dt_data.EXCHANGE, s, self.config.timeframe,
                                       start=self.now() - pd.Timedelta(days=2))
                price = float(last["close"].iloc[-1]) if len(last) else None
                ok, msg = (True, "") if price is None else feasibility(s, self.cash, self.config.max_leverage, price)
                if ok:
                    self._symbols.append(s)
                else:
                    self.log(f"fora da carteira: {msg}")
        return self._symbols

    def _candles(self) -> dict[str, pd.DataFrame]:
        start = self.now() - pd.Timedelta(days=self.config.history_days)
        out = {}
        for symbol in [self.config.symbol, *self.config.extra_symbols]:
            self.fetch(self.store, symbol, self.config.timeframe, start)
        for symbol in self.symbols:
            out[symbol] = self.store.read(dt_data.EXCHANGE, symbol, self.config.timeframe, start=start)
        return out

    def _closed_until(self, baskets: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
        """Só avança quando TODOS os ativos fecharam o candle."""
        ends = [closed_until(c, self.config.timeframe) for c in baskets.values()]
        return None if not ends or any(e is None for e in ends) else min(ends)

    def _attach(self, candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if self.config.features == "none":
            return candles
        from trader.live.runner import attach_features  # mesmas features, juntadas no fechamento do candle

        return {s: attach_features(c, s, self.config.timeframe, self.config.features, self.data)
                for s, c in candles.items()}

    def start(self, bootstrap: Path | None = None, executor=None) -> None:
        c = self.config
        candles = self._candles()
        dconf = DayConfig(
            initial=c.population, max_slots=c.slots, cash=self.cash, capital_brl=c.capital_brl,
            brl_rate=c.brl_rate, limits=DayTradeLimits(max_leverage=c.max_leverage), styles=c.styles,
            news=c.features != "none", black=c.features == "all", spec=spec_for(c.symbol),
            rules=SelectionRules(max_drawdown=0.20),
        )
        self.population = DayPopulation(candles, c.symbol, c.timeframe, self.registry, dconf,
                                        funding=None, executor=executor, creator=self.creator)
        if self.population.restore():
            self.log(f"retomado: {len(self.population.alive)} agentes vivos, janela {self.population._window}")
        else:
            self.population.seed(bootstrap_genomes(bootstrap, c.styles))
            self.registry.set_meta("live", {"species": "daytrade", "style": c.style, "symbol": c.symbol,
                                            "symbols": self.symbols, "timeframe": c.timeframe, "window": c.window,
                                            "features": c.features, "started": str(self.now())})
            self.log(f"população nova: {len(self.population.alive)} agentes ({', '.join(c.styles)})")
        if "next_start" not in self.registry.meta():
            self.registry.set_meta("next_start", str(floor_to(self.now(), self.step_len)))

    def tick(self) -> int:
        """Processa todas as janelas já fechadas. Retorna quantas gerações rodaram."""
        pop = self.population
        candles = self._candles()
        done = self._closed_until(candles)
        start = pd.Timestamp(self.registry.meta()["next_start"])
        if done is None or start + self.step_len > done:
            return 0
        pop.candles = self._attach(candles)
        ran = 0
        while start + self.step_len <= done:
            end = start + self.step_len
            r = pop.step(start, end)
            self.registry.set_meta("next_start", str(end))
            deaths = " ".join(f"{k}={v}" for k, v in sorted(r["deaths"].items())) or "-"
            births = " ".join(f"{k}={v}" for k, v in sorted(r["births"].items())) or "-"
            best = max(pop.alive, key=lambda a: a.life.score, default=None)
            self.log(f"{end:%Y-%m-%d %H:%M} UTC g{r['index']:04d} vivos={r['alive']} mortes[{deaths}]"
                     f" nascimentos[{births}] melhor={best.id if best else '-'}"
                     f" {best.life.ret if best else 0:+.2%} ref={pop.baseline_windows[-1].ret:+.2%}")
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


def run(data: Path, db_path: Path, config: DayLiveConfig, bootstrap: Path | None) -> None:
    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(max_workers=config.workers) if config.workers > 1 else None
    live = DayLiveEvolution(data, db_path, config, log=lambda m: print(m, flush=True))
    live.log(f"papel apenas (REAL_ORDERS_ENABLED={REAL_ORDERS_ENABLED}); estilo {config.style},"
             f" {config.timeframe}, geração a cada {config.window}; ativos {', '.join(live.symbols)}")
    live.start(bootstrap, executor)
    live.run_forever()
