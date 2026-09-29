"""Base de treino do Criador a partir dos registros das evoluções.

Cada exemplo é uma morte: autópsia do pai → sucessor. O rótulo vem do mercado,
por comparação contrafactual nas K janelas seguintes à morte:
  sucessor real × alternativas (regras, mutações) × pai continuando × referência.

Só janelas que a própria evolução percorreu entram (índice < última janela do
registro); o período final reservado nunca vira dado de treino.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trader.evolution.creator import RuleCreator, build_prompt, format_answer
from trader.evolution.fitness import Life, evaluate
from trader.evolution.genome import Genome, mutate
from trader.evolution.population import BASELINE
from trader.ingest import binance
from trader.storage.candles import CandleStore

# rodadas anteriores à tabela meta usaram estes valores fixos
LEGACY_RUN = {"symbol": "BTC/USDT", "timeframe": "4h", "window_days": 60, "evo_start": "2020-01-31 00:00:00+00:00"}
MAX_DRAWDOWN = 0.15  # mesmo limite de morte da seleção


@dataclass(frozen=True)
class Example:
    run: str
    parent_id: str
    successor_id: str
    died_at: int
    parent: Genome
    successor: Genome
    note: str
    autopsy: dict[str, Any]
    graveyard: list[dict[str, Any]]
    symbol: str
    timeframe: str
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]  # janelas após a morte

    @property
    def split(self) -> str:
        """Divisão por rodada inteira, para não vazar entre treino e validação."""
        return "val" if int(hashlib.md5(self.run.encode()).hexdigest(), 16) % 5 == 0 else "train"

    def prompt(self) -> str:
        return build_prompt(self.symbol, self.timeframe, self.autopsy, self.graveyard)


def _run_windows(meta: dict[str, Any], count: int) -> tuple[str, str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    run = meta.get("run")
    if run:
        return run["symbol"], run["timeframe"], [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in run["windows"]]
    start, length = pd.Timestamp(LEGACY_RUN["evo_start"]), pd.Timedelta(days=LEGACY_RUN["window_days"])
    windows = [(start + i * length, start + (i + 1) * length) for i in range(count)]
    return LEGACY_RUN["symbol"], LEGACY_RUN["timeframe"], windows


def load_examples(db_path: Path, horizon: int = 3) -> list[Example]:
    db = sqlite3.connect(str(db_path))
    meta = {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM meta")} if _has_meta(db) else {}
    last_window = db.execute("SELECT MAX(COALESCE(died_at, born_at)) FROM agents").fetchone()[0] or 0
    symbol, timeframe, windows = _run_windows(meta, last_window + 1)
    deaths = db.execute(
        "SELECT rowid, id, cause, genome, died_at FROM agents WHERE died_at IS NOT NULL ORDER BY died_at, rowid"
    ).fetchall()
    order = {row[1]: i for i, row in enumerate(deaths)}
    rows = db.execute(
        "SELECT s.id, s.genome, s.note, p.id, p.genome, p.autopsy, p.died_at"
        " FROM agents s JOIN agents p ON s.parent_id = p.id WHERE s.origin = 'successor'"
    ).fetchall()
    examples = []
    for sid, sgen, note, pid, pgen, autopsy, died_at in rows:
        # o sucessor opera a partir da janela seguinte à morte
        first, last = died_at + 1, died_at + horizon
        if last > last_window:
            continue
        prior = deaths[max(0, order[pid] - 10) : order[pid]]
        graveyard = [{"id": r[1], "cause": r[2], "genome": json.loads(r[3])} for r in reversed(prior)]
        examples.append(
            Example(
                run=db_path.stem,
                parent_id=pid,
                successor_id=sid,
                died_at=died_at,
                parent=Genome.from_dict(json.loads(pgen)),
                successor=Genome.from_dict(json.loads(sgen)),
                note=note or "",
                autopsy=json.loads(autopsy),
                graveyard=graveyard,
                symbol=symbol,
                timeframe=timeframe,
                windows=windows[first : last + 1],
            )
        )
    return examples


def _has_meta(db: sqlite3.Connection) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone() is not None


# avaliação contrafactual (roda em processos separados)

_DATA_DIR = Path("data")
_CANDLES: dict[tuple[str, str], pd.DataFrame] = {}


def init_worker(data_dir: str) -> None:
    """Inicializador dos processos: cada um carrega os candles uma vez só."""
    global _DATA_DIR
    _DATA_DIR = Path(data_dir)
    _CANDLES.clear()


def _candles(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol, timeframe)
    if key not in _CANDLES:
        _CANDLES[key] = CandleStore(_DATA_DIR).read(binance.EXCHANGE, symbol, timeframe)
    return _CANDLES[key]


def life_over(genome: Genome, symbol: str, timeframe: str, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> Life:
    candles = _candles(symbol, timeframe)
    return Life([evaluate(genome, candles, symbol, timeframe, a, b) for a, b in windows])


def summarize(life: Life) -> dict[str, float]:
    return {"score": life.score, "ret": life.ret, "dd": life.worst_dd, "trades": life.trades}


def _score_job(job: tuple[Example, dict[str, Genome]]) -> dict[str, dict[str, float]]:
    ex, candidates = job
    return {name: summarize(life_over(g, ex.symbol, ex.timeframe, ex.windows)) for name, g in candidates.items()}


def candidates_for(ex: Example, n_mutations: int = 2) -> dict[str, Genome]:
    rng = random.Random(f"{ex.run}:{ex.parent_id}")
    rules, _ = RuleCreator().successor(ex.parent, ex.autopsy, rng)
    out = {"successor": ex.successor, "rules": rules, "parent": ex.parent, "baseline": BASELINE}
    for i in range(n_mutations):
        out[f"mutation{i}"] = mutate(ex.parent, rng, 0.3)
    return out


def score_examples(
    examples: list[Example], executor: Executor | None, candidates=candidates_for
) -> list[dict[str, dict[str, float]]]:
    jobs = [(ex, candidates(ex)) for ex in examples]
    if executor is None:
        return [_score_job(j) for j in jobs]
    return list(executor.map(_score_job, jobs, chunksize=8))


# montagem da base


def describe_change(parent: Genome, child: Genome) -> str:
    """Explica a diferença entre genomas numa frase (alvo 'why' do treino)."""
    if parent.strategy != child.strategy:
        return f"Switch strategy from {parent.strategy} to {child.strategy}."
    parts = []
    for k, v in child.params.items():
        if parent.params.get(k) != v:
            parts.append(f"{k} {parent.params[k]}→{v}")
    for gene in ("vol_filter", "trend_filter", "cooldown", "size"):
        a, b = getattr(parent, gene), getattr(child, gene)
        if a == b:
            continue
        if a is None:
            parts.append(f"add {gene}={b}")
        elif b is None:
            parts.append(f"remove {gene}")
        else:
            parts.append(f"{gene} {a}→{b}")
    return ("Adjust " + ", ".join(parts) + ".") if parts else "Keep the genome."


def is_good(scores: dict[str, dict[str, float]], name: str) -> bool:
    s = scores[name]
    return s["dd"] <= MAX_DRAWDOWN and s["score"] > scores["parent"]["score"] and s["score"] > scores["baseline"]["score"]


def build_records(
    examples: list[Example], scores: list[dict[str, dict[str, float]]], min_margin: float = 0.2
) -> dict[str, list[dict[str, Any]]]:
    """Gera sft (só respostas boas), dpo (melhor × pior candidato) e eval (validação)."""
    out: dict[str, list[dict[str, Any]]] = {"sft_train": [], "sft_val": [], "dpo_train": [], "dpo_val": [], "eval": []}
    for ex, sc in zip(examples, scores):
        options = {k: v for k, v in sc.items() if k not in ("parent", "baseline")}
        genomes = candidates_for(ex)
        best = max(options, key=lambda k: options[k]["score"])
        worst = min(options, key=lambda k: options[k]["score"])
        prompt = ex.prompt()

        def answer(name: str) -> str:
            why = ex.note.removeprefix("LLM: ") if name == "successor" and ex.note.startswith("LLM") else ""
            return format_answer(genomes[name], why or describe_change(ex.parent, genomes[name]))

        if is_good(sc, best):
            out[f"sft_{ex.split}"].append(
                {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer(best)}],
                 "meta": {"run": ex.run, "parent": ex.parent_id, "source": best, "cause": ex.autopsy["cause"]}}
            )
        if options[best]["score"] - options[worst]["score"] >= min_margin:
            out[f"dpo_{ex.split}"].append(
                {"prompt": prompt, "chosen": answer(best), "rejected": answer(worst),
                 "meta": {"run": ex.run, "parent": ex.parent_id, "chosen": best, "rejected": worst}}
            )
        if ex.split == "val":
            out["eval"].append(
                {"run": ex.run, "parent_id": ex.parent_id, "parent": ex.parent.to_dict(), "autopsy": ex.autopsy,
                 "graveyard": ex.graveyard, "symbol": ex.symbol, "timeframe": ex.timeframe,
                 "windows": [[str(a), str(b)] for a, b in ex.windows], "scores": sc}
            )
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
