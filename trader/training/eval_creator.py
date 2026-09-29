"""Compara Criadores pelo que importa: o sucessor sobrevive e rende mais por risco?

Cada Criador recebe as mesmas autópsias da validação (rodadas nunca usadas no
treino) e seu sucessor é testado nas janelas seguintes à morte.
"""
from __future__ import annotations

import random
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trader.evolution.creator import Creator, LLMCreator, RuleCreator
from trader.evolution.genome import Genome
from trader.training.creator_data import MAX_DRAWDOWN, life_over, summarize


@dataclass
class CreatorReport:
    name: str
    scores: list[float] = field(default_factory=list)
    beat_parent: int = 0
    beat_baseline: int = 0
    drawdown_deaths: int = 0
    failures: int = 0
    seconds: float = 0.0

    def row(self) -> str:
        n = len(self.scores) or 1
        mean = sum(self.scores) / n
        return (
            f"{self.name:<28} nota {mean:+.2f} | bate pai {self.beat_parent / n:5.0%} | bate ref {self.beat_baseline / n:5.0%}"
            f" | morte por queda {self.drawdown_deaths / n:5.0%} | falhas {self.failures} | {self.seconds / n:.1f}s/autópsia"
        )


def make_creator(spec: str, record: dict[str, Any], llm_url: str) -> Creator:
    if spec == "rules":
        return RuleCreator()
    if spec.startswith("llm:"):
        return LLMCreator(record["symbol"], record["timeframe"], llm_url, spec[4:], graveyard=lambda: record["graveyard"])
    raise ValueError(f"criador desconhecido: {spec!r} (use rules ou llm:<modelo>)")


def _score(job: tuple[Genome, str, str, list]) -> dict[str, float]:
    genome, symbol, timeframe, windows = job
    return summarize(life_over(genome, symbol, timeframe, [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows]))


def evaluate_creators(
    records: list[dict[str, Any]], specs: list[str], executor: Executor | None, llm_url: str
) -> list[CreatorReport]:
    reports = []
    for spec in specs:
        report = CreatorReport(spec)
        proposals = []
        for i, rec in enumerate(records):
            creator = make_creator(spec, rec, llm_url)
            t = time.time()
            genome, _ = creator.successor(Genome.from_dict(rec["parent"]), rec["autopsy"], random.Random(i))
            report.seconds += time.time() - t
            report.failures += getattr(creator, "failures", 0)
            proposals.append((genome, rec["symbol"], rec["timeframe"], rec["windows"]))
        results = list(executor.map(_score, proposals)) if executor else [_score(p) for p in proposals]
        for rec, s in zip(records, results):
            report.scores.append(s["score"])
            report.beat_parent += s["score"] > rec["scores"]["parent"]["score"]
            report.beat_baseline += s["score"] > rec["scores"]["baseline"]["score"]
            report.drawdown_deaths += s["dd"] > MAX_DRAWDOWN
        reports.append(report)
    return reports
