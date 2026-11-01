"""Ciclo de vida da população, janela a janela.

Em cada janela todos os vivos operam (em simulação) em paralelo. Depois:
  - quem falhou morre; o Criador lê a autópsia e gera o sucessor na mesma vaga;
  - quem prosperou ganha vaga para ter 2 (um filho mutado), respeitando o
    teto de vagas e a fatia máxima por linhagem; população cheia remove o pior.
"""
from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from trader.evolution.creator import Creator
from trader.evolution.fitness import Cause, Life, SelectionRules, Verdict, WindowStats, combine, evaluate, judge
from trader.evolution.genome import Genome, random_genome
from trader.evolution.registry import Registry

BASELINE = Genome(strategy="ma_cross", params={"fast": 20, "slow": 50})


@dataclass
class Agent:
    id: str
    genome: Genome
    lineage: str
    generation: int
    life: Life = field(default_factory=Life)


@dataclass(frozen=True)
class GenerationReport:
    index: int
    start: pd.Timestamp
    alive: int
    deaths: dict[str, int]
    births: dict[str, int]
    best_id: str | None
    best_ret: float
    best_score: float
    baseline_ret: float


@dataclass(frozen=True)
class PopulationConfig:
    initial: int = 16
    max_slots: int = 32
    max_lineage_share: float = 0.3
    rules: SelectionRules = SelectionRules()
    fee_rate: float = 0.001  # taxa por lado (0,001 = spot Binance; 0,0002 = futuros com ordem limitada)
    creator_threads: int = 4
    candidates: int = 4  # melhor de N: a LLM propõe N genomas, a triagem escolhe o melhor
    screen_windows: int = 3  # janelas recentes usadas na triagem (backtest rápido, sem LLM)
    lessons_every: int = 8  # a cada N gerações a LLM reescreve as lições (0 = desliga)
    priors: dict | None = None  # faixas de partida dos genes (playbook dos livros)  # sucessores das mortes de uma geração são desenhados em paralelo


def _evaluate_job(args: tuple) -> WindowStats:
    return evaluate(*args)


class Population:
    def __init__(
        self,
        candles: pd.DataFrame,
        symbol: str,
        timeframe: str,
        registry: Registry,
        creator: Creator,
        config: PopulationConfig = PopulationConfig(),
        seed: int = 0,
        executor: Executor | None = None,
    ) -> None:
        self.candles = candles
        self.symbol, self.timeframe = symbol, timeframe
        self.registry = registry
        self.creator = creator
        self.config = config
        self.rng = random.Random(seed)
        self.executor = executor
        self.alive: list[Agent] = []
        self.baseline_windows: list[WindowStats] = []
        self._counter = 0
        self._window = 0
        self.window_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.lessons: list[str] = []
        self._warmup = pd.Timedelta(0)

    # nascimento

    def _new_id(self) -> str:
        self._counter += 1
        return f"a{self._counter:05d}"

    def _born(self, genome: Genome, origin: str, parent: Agent | None = None, note: str = "") -> Agent:
        agent_id = self._new_id()
        lineage = parent.lineage if parent else agent_id
        generation = parent.generation + 1 if parent else 0
        agent = Agent(agent_id, genome, lineage, generation)
        self.registry.birth(
            agent_id, genome, lineage, generation, self._window, origin, parent.id if parent else None, note
        )
        self.alive.append(agent)
        return agent

    def seed(self, genomes: list[Genome] | None = None) -> None:
        """Nasce a população inicial: aleatória, ou herdada (ex.: sobreviventes de um backtest)."""
        genomes = genomes or [random_genome(self.rng, priors=self.config.priors) for _ in range(self.config.initial)]
        for g in genomes[: self.config.max_slots]:
            self._born(g, "seed")

    def restore(self) -> bool:
        """Retoma a população gravada no registro (vivos, históricos e referência)."""
        alive = self.registry.alive()
        if not alive:
            return False
        self.alive = [
            Agent(a["id"], Genome.from_dict(a["genome"]), a["lineage"], a["generation"],
                  Life([_stats(w) for w in self.registry.windows_of(a["id"])]))
            for a in alive
        ]
        self.baseline_windows = [_stats(w) for w in self.registry.windows_of("__baseline__")]
        self._counter = self.registry.agent_count()
        self._window = self.registry.next_window()
        starts = [w.start for w in self.baseline_windows]
        if len(starts) > 1:
            step = pd.Series(starts).diff().median()
            self.window_bounds = [(a, a + step) for a in starts]
        self.lessons = self.registry.meta().get("lessons", {}).get("lessons", [])
        return True

    # uma janela

    @property
    def baskets(self) -> dict[str, pd.DataFrame]:
        """Ativos operados: `candles` pode ser um DataFrame (um ativo) ou {símbolo: DataFrame}."""
        return self.candles if isinstance(self.candles, dict) else {self.symbol: self.candles}

    @staticmethod
    def _slice_for(c: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, genomes: list[Genome]) -> pd.DataFrame:
        """Recorta só o necessário (janela + aquecimento) para mandar aos processos."""
        tf = c["ts"].diff().median()
        warm = max(g.warmup for g in genomes) + 2
        lo = start - tf * warm
        return c[(c["ts"] >= lo) & (c["ts"] < end)].reset_index(drop=True)

    def _evaluate_all(self, genomes: list[Genome], start: pd.Timestamp, end: pd.Timestamp) -> list[WindowStats]:
        """Cada genoma opera todos os ativos da cesta; o resultado da janela é a carteira de pesos iguais."""
        symbols = [s for s, c in self.baskets.items() if not c.empty and c["ts"].iloc[0] < start]
        jobs = []
        for symbol in symbols:
            data = self._slice_for(self.baskets[symbol], start, end, genomes)
            jobs += [(g, data, symbol, self.timeframe, start, end, self.config.fee_rate) for g in genomes]
        results = [_evaluate_job(j) for j in jobs] if self.executor is None else list(self.executor.map(_evaluate_job, jobs))
        per_symbol = [results[i * len(genomes) : (i + 1) * len(genomes)] for i in range(len(symbols))]
        return [combine([per_symbol[k][i] for k in range(len(symbols))]) for i in range(len(genomes))]

    def _baseline_over(self, agent: Agent) -> Life:
        return Life(self.baseline_windows[len(self.baseline_windows) - agent.life.age :])

    def _autopsy(self, agent: Agent, cause: Cause) -> dict[str, Any]:
        last = agent.life.windows[-1] if agent.life.windows else None
        return {
            "cause": cause.value,
            "genome": agent.genome.to_dict(),
            "life": {
                "age_windows": agent.life.age,
                "trades": agent.life.trades,
                "net_return": round(agent.life.ret, 4),
                "gross_return": round(agent.life.gross_ret, 4),
                "worst_drawdown": round(agent.life.worst_dd, 4),
                "benchmark_return": round(self._baseline_over(agent).ret, 4),
            },
            "last_window": {
                "net_return": round(last.ret, 4),
                "max_drawdown": round(last.max_dd, 4),
                "trades": last.trades,
                "dd_high_vol": last.dd_high_vol,
                "dd_market_down": last.dd_market_down,
                "median_range": round(last.median_range, 5),
            }
            if last
            else {},
        }

    def _kill(self, agent: Agent, cause: Cause) -> dict[str, Any]:
        autopsy = self._autopsy(agent, cause)
        self.registry.death(agent.id, self._window, cause.value, autopsy)
        self.alive.remove(agent)
        return autopsy

    def step(self, start: pd.Timestamp, end: pd.Timestamp) -> GenerationReport:
        genomes = [a.genome for a in self.alive] + [BASELINE]
        stats = self._evaluate_all(genomes, start, end)
        self.window_bounds.append((start, end))
        self.baseline_windows.append(stats[-1])
        self.registry.record_window("__baseline__", self._window, asdict(stats[-1]))
        for agent, s in zip(self.alive, stats[:-1]):
            agent.life.windows.append(s)
            self.registry.record_window(agent.id, self._window, asdict(s))
        self.registry.commit()

        deaths: Counter[str] = Counter()
        births: Counter[str] = Counter()
        thrivers: list[Agent] = []
        dead: list[tuple[Agent, dict[str, Any]]] = []
        for agent in list(self.alive):
            verdict, cause = judge(agent.life, self._baseline_over(agent), self.config.rules)
            if verdict is Verdict.DIE:
                dead.append((agent, self._kill(agent, cause)))
                deaths[cause.value] += 1
            elif verdict is Verdict.THRIVE:
                thrivers.append(agent)

        # o Criador (LLM) é o passo lento: desenha todos os sucessores ao mesmo tempo;
        # cada morte tem seu próprio gerador aleatório, então o resultado não depende da ordem das threads
        seeds = [self.rng.random() for _ in dead]

        n = max(1, self.config.candidates)

        def design(job: tuple[tuple[Agent, dict[str, Any]], float]) -> tuple[Genome, str]:
            (agent, autopsy), seed = job
            rng = random.Random(seed)
            many = getattr(self.creator, "successor_candidates", None)
            options = many(agent.genome, autopsy, rng, n) if many else [self.creator.successor(agent.genome, autopsy, rng)]
            return self._pick(agent, "successor", autopsy, options)

        with ThreadPoolExecutor(max_workers=max(1, self.config.creator_threads)) as pool:
            designed = list(pool.map(design, zip(dead, seeds)))
        for (agent, _), (genome, note) in zip(dead, designed):
            self._born(genome, "successor", agent, note)
            births["successor"] += 1

        # quem prosperou entra (ou se atualiza) na galeria, com a análise do porquê
        reports = {a.id: self._success_report(a) for a in thrivers}
        for agent in thrivers:
            self.registry.induct(agent.id, agent.lineage, agent.genome.to_dict(), reports[agent.id],
                                 agent.life.score, self._window)

        # vagas: decide primeiro quem terá filho (e quem sai para abrir vaga), depois desenha em paralelo
        parents: list[Agent] = []
        pending: Counter[str] = Counter()
        for agent in sorted(thrivers, key=lambda a: a.life.score, reverse=True):
            if agent not in self.alive or not self._lineage_has_room(agent.lineage, pending[agent.lineage]):
                continue
            if len(self.alive) + len(parents) >= self.config.max_slots:
                worst = self._worst(exclude=agent)
                if worst is None or worst.life.score >= agent.life.score or worst in parents:
                    continue
                self._kill(worst, Cause.CULLED)
                deaths[Cause.CULLED.value] += 1
            parents.append(agent)
            pending[agent.lineage] += 1

        child_seeds = [self.rng.random() for _ in parents]

        def breed(job: tuple[Agent, float]) -> tuple[Genome, str]:
            agent, seed = job
            rng = random.Random(seed)
            many = getattr(self.creator, "child_candidates", None)
            options = (many(agent.genome, rng, reports[agent.id], n) if many
                       else [self.creator.child(agent.genome, rng, reports[agent.id])])
            return self._pick(agent, "child", reports[agent.id], options)

        with ThreadPoolExecutor(max_workers=max(1, self.config.creator_threads)) as pool:
            children = list(pool.map(breed, zip(parents, child_seeds)))
        for agent, (child, note) in zip(parents, children):
            self._born(child, "child", agent, f"filho de {agent.id} (prosperou: {agent.life.ret:+.1%}) — {note}")
            births["child"] += 1

        self.registry.commit()
        self._refresh_lessons()
        veterans = [a for a in self.alive if a.life.age > 0]
        best = max(veterans, key=lambda a: a.life.score, default=None)
        report = GenerationReport(
            index=self._window,
            start=start,
            alive=len(self.alive),
            deaths=dict(deaths),
            births=dict(births),
            best_id=best.id if best else None,
            best_ret=best.life.ret if best else 0.0,
            best_score=best.life.score if best else 0.0,
            baseline_ret=stats[-1].ret,
        )
        self._window += 1
        return report

    def _screen(self, genomes: list[Genome]) -> list[float]:
        """Nota de cada candidato nas últimas janelas (retorno ÷ queda), só backtest."""
        bounds = self.window_bounds[-self.config.screen_windows :]
        if not bounds or len(genomes) == 1:
            return [0.0] * len(genomes)
        per_window = [self._evaluate_all(genomes, a, b) for a, b in bounds]
        return [Life([w[i] for w in per_window]).score for i in range(len(genomes))]

    def _pick(self, parent: Agent, kind: str, context: dict[str, Any],
              options: list[tuple[Genome, str]]) -> tuple[Genome, str]:
        """Escolhe o melhor candidato pela triagem e registra todos (dado de treino)."""
        scores = self._screen([g for g, _ in options])
        best = max(range(len(options)), key=lambda i: scores[i])
        if len(options) > 1:
            self.registry.record_proposal(
                parent.id, kind, self._window, context,
                [{"genome": g.to_dict(), "why": why, "score": round(sc, 4)} for (g, why), sc in zip(options, scores)],
                best,
            )
        genome, why = options[best]
        tag = f" [melhor de {len(options)}: {scores[best]:+.2f}]" if len(options) > 1 else ""
        return genome, why + tag

    def _refresh_lessons(self) -> None:
        distill = getattr(self.creator, "distill_lessons", None)
        every = self.config.lessons_every
        if distill is None or not every or self._window % every:
            return
        try:
            self.lessons = distill(self.registry.hall(10), self.registry.graveyard(30))
            self.registry.set_meta("lessons", {"window": self._window, "lessons": self.lessons})
        except Exception:
            pass  # sem lições novas nesta rodada; mantém as anteriores

    def _lineage_has_room(self, lineage: str, pending: int = 0) -> bool:
        members = sum(1 for a in self.alive if a.lineage == lineage) + pending
        return members + 1 <= max(2, int(self.config.max_slots * self.config.max_lineage_share))

    def _success_report(self, agent: Agent) -> dict[str, Any]:
        """A 'autópsia' de quem acertou: o que fez e em que contexto de mercado venceu a referência."""
        base = self._baseline_over(agent)
        won = [w for w, b in zip(agent.life.windows, base.windows) if w.score > b.score]
        news = [w.news_sentiment for w in won if w.news_sentiment is not None]
        mean = lambda xs: round(sum(xs) / len(xs), 5) if xs else 0.0
        return {
            "genome": agent.genome.to_dict(),
            "life": {
                "age_windows": agent.life.age,
                "trades": agent.life.trades,
                "net_return": round(agent.life.ret, 4),
                "worst_drawdown": round(agent.life.worst_dd, 4),
                "benchmark_return": round(base.ret, 4),
                "score": round(agent.life.score, 3),
            },
            "windows_won": len(won),
            "winning_context": {
                "market_ret": mean([w.market_ret for w in won]),
                "median_range": mean([w.median_range for w in won]),
                "news_sentiment": mean(news) if news else None,
            },
        }

    def _worst(self, exclude: Agent) -> Agent | None:
        candidates = [a for a in self.alive if a is not exclude and a.life.age > 0]
        return min(candidates, key=lambda a: a.life.score, default=None)

    def champion(self) -> Agent | None:
        """Melhor sobrevivente com experiência suficiente para ser levado a sério."""
        rules = self.config.rules
        eligible = [
            a for a in self.alive if a.life.age >= rules.min_age_for_edge and a.life.trades >= rules.min_trades
        ]
        return max(eligible, key=lambda a: a.life.score, default=None)


def _stats(data: dict[str, Any]) -> WindowStats:
    return WindowStats(**{**data, "start": pd.Timestamp(data["start"])})


def make_executor(workers: int) -> Executor | None:
    return ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
