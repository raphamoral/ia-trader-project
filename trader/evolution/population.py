"""Ciclo de vida da população, janela a janela.

Em cada janela todos os vivos operam (em simulação) em paralelo. Depois:
  - quem falhou morre; o Criador lê a autópsia e gera o sucessor na mesma vaga;
  - quem prosperou ganha vaga para ter 2 (um filho mutado), respeitando o
    teto de vagas e a fatia máxima por linhagem; população cheia remove o pior.
"""
from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trader.evolution.creator import Creator
from trader.evolution.fitness import Cause, Life, SelectionRules, Verdict, WindowStats, evaluate, judge
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

    def seed(self) -> None:
        for _ in range(self.config.initial):
            self._born(random_genome(self.rng), "seed")

    # uma janela

    def _slice_for(self, start: pd.Timestamp, end: pd.Timestamp, genomes: list[Genome]) -> pd.DataFrame:
        """Recorta só o necessário (janela + aquecimento) para mandar aos processos."""
        tf = self.candles["ts"].diff().median()
        warm = max(g.warmup for g in genomes) + 2
        lo = start - tf * warm
        c = self.candles
        return c[(c["ts"] >= lo) & (c["ts"] < end)].reset_index(drop=True)

    def _evaluate_all(self, genomes: list[Genome], start: pd.Timestamp, end: pd.Timestamp) -> list[WindowStats]:
        data = self._slice_for(start, end, genomes)
        jobs = [(g, data, self.symbol, self.timeframe, start, end) for g in genomes]
        if self.executor is None:
            return [_evaluate_job(j) for j in jobs]
        return list(self.executor.map(_evaluate_job, jobs))

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
        self.baseline_windows.append(stats[-1])
        for agent, s in zip(self.alive, stats[:-1]):
            agent.life.windows.append(s)

        deaths: Counter[str] = Counter()
        births: Counter[str] = Counter()
        thrivers: list[Agent] = []
        for agent in list(self.alive):
            verdict, cause = judge(agent.life, self._baseline_over(agent), self.config.rules)
            if verdict is Verdict.DIE:
                autopsy = self._kill(agent, cause)
                deaths[cause.value] += 1
                genome, note = self.creator.successor(agent.genome, autopsy, self.rng)
                self._born(genome, "successor", agent, note)
                births["successor"] += 1
            elif verdict is Verdict.THRIVE:
                thrivers.append(agent)

        for agent in sorted(thrivers, key=lambda a: a.life.score, reverse=True):
            if agent not in self.alive or not self._lineage_has_room(agent.lineage):
                continue
            if len(self.alive) >= self.config.max_slots:
                worst = self._worst(exclude=agent)
                if worst is None or worst.life.score >= agent.life.score:
                    continue
                self._kill(worst, Cause.CULLED)
                deaths[Cause.CULLED.value] += 1
            child = self.creator.child(agent.genome, self.rng)
            self._born(child, "child", agent, f"filho de {agent.id} (prosperou: {agent.life.ret:+.1%})")
            births["child"] += 1

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

    def _lineage_has_room(self, lineage: str) -> bool:
        members = sum(1 for a in self.alive if a.lineage == lineage)
        return members + 1 <= max(2, int(self.config.max_slots * self.config.max_lineage_share))

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


def make_executor(workers: int) -> Executor | None:
    return ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
