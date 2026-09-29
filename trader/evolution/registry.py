"""Registro da população em SQLite: vivos, cemitério e árvore genealógica.

Cada agente é uma linha. Nascer é INSERT; morrer é marcar causa e autópsia.
"""
from __future__ import annotations

import json
import sqlite3
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
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL      -- JSON
);
"""


class Registry:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
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

    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.db.commit()

    def meta(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def graveyard(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, lineage, cause, genome FROM agents WHERE died_at IS NOT NULL"
            " ORDER BY died_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "lineage": r[1], "cause": r[2], "genome": json.loads(r[3])} for r in rows]

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
