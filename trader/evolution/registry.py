"""Registro da população em SQLite: vivos, cemitério e árvore genealógica.

Cada agente é uma linha. Nascer é INSERT; morrer é marcar causa e autópsia.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from trader.evolution.genome import Genome

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id          TEXT PRIMARY KEY,
    parent_id   TEXT,
    lineage     TEXT NOT NULL,
    generation  INTEGER NOT NULL,   -- geração dentro da linhagem
    born_at     INTEGER NOT NULL,   -- índice da janela em que nasceu
    died_at     INTEGER,
    genome      TEXT NOT NULL,
    origin      TEXT NOT NULL,      -- seed | successor | child
    note        TEXT,               -- por que nasceu assim (regras ou LLM)
    cause       TEXT,
    autopsy     TEXT
);
CREATE TABLE IF NOT EXISTS life (
    agent_id TEXT NOT NULL,         -- '__baseline__' guarda a referência
    idx      INTEGER NOT NULL,      -- índice da janela
    stats    TEXT NOT NULL,         -- WindowStats em JSON
    PRIMARY KEY (agent_id, idx)
);
CREATE TABLE IF NOT EXISTS hall (
    agent_id  TEXT PRIMARY KEY,
    lineage   TEXT NOT NULL,
    genome    TEXT NOT NULL,
    report    TEXT NOT NULL,        -- análise de sucesso (JSON)
    score     REAL NOT NULL,
    inducted  INTEGER NOT NULL,     -- janela em que entrou
    updated   INTEGER NOT NULL      -- última janela em que prosperou
);
CREATE TABLE IF NOT EXISTS proposals (
    parent_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,       -- successor | child
    window     INTEGER NOT NULL,
    context    TEXT NOT NULL,       -- autópsia ou análise de sucesso (JSON)
    candidates TEXT NOT NULL,       -- [{"genome", "why", "score"}] (JSON)
    chosen     INTEGER NOT NULL     -- índice do escolhido pela triagem
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL      -- JSON
);
"""


class Registry:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # o Criador consulta cemitério e galeria de dentro das threads que desenham sucessores
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self.db.executescript(SCHEMA)

    def birth(
        self,
        agent_id: str,
        genome: Genome,
        lineage: str,
        generation: int,
        born_at: int,
        origin: str,
        parent_id: str | None = None,
        note: str = "",
    ) -> None:
        self.db.execute(
            "INSERT INTO agents (id, parent_id, lineage, generation, born_at, genome, origin, note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (agent_id, parent_id, lineage, generation, born_at, json.dumps(genome.to_dict()), origin, note),
        )
        self.db.commit()

    def death(self, agent_id: str, died_at: int, cause: str, autopsy: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE agents SET died_at = ?, cause = ?, autopsy = ? WHERE id = ?",
            (died_at, cause, json.dumps(autopsy, default=str), agent_id),
        )
        self.db.commit()

    def record_window(self, agent_id: str, idx: int, stats: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO life (agent_id, idx, stats) VALUES (?, ?, ?)",
            (agent_id, idx, json.dumps(stats, default=str)),
        )

    def commit(self) -> None:
        self.db.commit()

    def windows_of(self, agent_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT stats FROM life WHERE agent_id = ? ORDER BY idx", (agent_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def alive(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, genome, lineage, generation FROM agents WHERE died_at IS NULL ORDER BY rowid"
        ).fetchall()
        return [{"id": r[0], "genome": json.loads(r[1]), "lineage": r[2], "generation": r[3]} for r in rows]

    def next_window(self) -> int:
        row = self.db.execute("SELECT MAX(idx) FROM life").fetchone()
        return 0 if row[0] is None else row[0] + 1

    def agent_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM agents").fetchone()[0]

    def induct(self, agent_id: str, lineage: str, genome: dict[str, Any], report: dict[str, Any], score: float,
               window: int) -> None:
        """Entra (ou atualiza) na galeria dos que prosperaram com consistência."""
        self.db.execute(
            "INSERT INTO hall (agent_id, lineage, genome, report, score, inducted, updated) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(agent_id) DO UPDATE SET report = excluded.report, score = excluded.score,"
            " genome = excluded.genome, updated = excluded.updated",
            (agent_id, lineage, json.dumps(genome), json.dumps(report, default=str), score, window, window),
        )
        self.db.commit()

    def hall(self, limit: int = 10) -> list[dict[str, Any]]:
        """Melhores da galeria, uma entrada por linhagem (evita a galeria virar uma família só)."""
        with self._lock:
            rows = self.db.execute("SELECT agent_id, lineage, report, score FROM hall ORDER BY score DESC").fetchall()
        out, seen = [], set()
        for agent_id, lineage, report, score in rows:
            if lineage in seen:
                continue
            seen.add(lineage)
            r = json.loads(report)
            out.append({"id": agent_id, "score": round(score, 2), "genome": r.get("genome"),
                        "winning_context": r.get("winning_context"), "life": r.get("life")})
            if len(out) == limit:
                break
        return out

    def record_proposal(self, parent_id: str, kind: str, window: int, context: dict[str, Any],
                        candidates: list[dict[str, Any]], chosen: int) -> None:
        """Candidatos com a nota da triagem: vira dado de treino (melhor × pior) para a LLM."""
        with self._lock:
            self.db.execute(
                "INSERT INTO proposals (parent_id, kind, window, context, candidates, chosen) VALUES (?, ?, ?, ?, ?, ?)",
                (parent_id, kind, window, json.dumps(context, default=str), json.dumps(candidates, default=str), chosen),
            )

    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.db.commit()

    def meta(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def graveyard(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._graveyard_rows(limit)
        return [{"id": r[0], "lineage": r[1], "cause": r[2], "genome": json.loads(r[3])} for r in rows]

    def _graveyard_rows(self, limit: int) -> list[tuple]:
        return self.db.execute(
            "SELECT id, lineage, cause, genome FROM agents WHERE died_at IS NOT NULL"
            " ORDER BY died_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def cause_counts(self) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT cause, COUNT(*) FROM agents WHERE cause IS NOT NULL GROUP BY cause"
        ).fetchall()
        return dict(rows)

    def ancestry(self, agent_id: str) -> list[dict[str, Any]]:
        """Do agente até o fundador da linhagem."""
        chain = []
        current: str | None = agent_id
        while current:
            row = self.db.execute(
                "SELECT id, parent_id, origin, note, cause FROM agents WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                break
            chain.append({"id": row[0], "origin": row[2], "note": row[3], "cause": row[4]})
            current = row[1]
        return chain
