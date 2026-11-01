"""Coleta contínua de notícias (white + black) e dados alternativos, 24h.

Só chama os comandos que os módulos de notícias já oferecem, cada um no seu
ritmo. Guarda a última execução de cada tarefa em disco, então um reinício não
repete tudo de uma vez. Nenhuma tarefa usa a API do GDELT (o histórico vem do
backfill por arquivos brutos, que é outro processo).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data" / "live" / "collector_state.json"
STOP = ROOT / "data" / "live" / "coleta.STOP"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class Task:
    name: str
    every: pd.Timedelta
    args: list[str]  # argumentos de `python -m ...`; {today}, {yesterday}, {week_ago}, {in_60d} são preenchidos
    timeout_s: int = 900


TASKS = [
    Task("white_rss", pd.Timedelta(minutes=2), ["trader.news", "backfill", "--source", "rss"]),
    Task("white_enrich", pd.Timedelta(minutes=10), ["trader.news", "enrich", "--limit", "30"]),
    Task("white_fulltext", pd.Timedelta(minutes=10), ["trader.news", "fulltext", "--limit", "15"]),
    Task("black_collect", pd.Timedelta(minutes=15), ["trader.blacknews", "backfill", "--archive-limit", "30"]),
    Task("black_enrich", pd.Timedelta(minutes=15), ["trader.blacknews", "enrich", "--limit", "30"]),
    Task("black_cache", pd.Timedelta(minutes=15),
         ["trader.blacknews.cache", "--symbols", "SOL/USDT,ETH/USDT,XRP/USDT", "--timeframes", "5m,1m"], 1800),
    Task("security_collect", pd.Timedelta(hours=1), ["trader.blacknews", "sec-backfill"]),
    Task("security_extract", pd.Timedelta(hours=1), ["trader.blacknews", "sec-extract", "--max-calls", "5"]),
    Task("calendar", pd.Timedelta(days=1), ["trader.news", "calendar", "--since", "{week_ago}", "--until", "{in_60d}"]),
    Task("alt_daily", pd.Timedelta(days=1), ["trader.blacknews", "alt-backfill", "--families", "fng,wiki,dev,cvm"]),
    Task("derivatives", pd.Timedelta(days=1),
         ["trader.blacknews", "alt-backfill", "--families", "none", "--deriv-since", "{yesterday}"], 3600),
]


def _fill(args: list[str], now: pd.Timestamp) -> list[str]:
    values = {
        "today": f"{now:%Y-%m-%d}",
        "yesterday": f"{now - pd.Timedelta(days=1):%Y-%m-%d}",
        "week_ago": f"{now - pd.Timedelta(days=7):%Y-%m-%d}",
        "in_60d": f"{now + pd.Timedelta(days=60):%Y-%m-%d}",
    }
    return [a.format(**values) for a in args]


def _load() -> dict[str, str]:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def due(tasks: list[Task], last: dict[str, str], now: pd.Timestamp) -> list[Task]:
    return [t for t in tasks if t.name not in last or now - pd.Timestamp(last[t.name]) >= t.every]


def run_task(task: Task, now: pd.Timestamp, log) -> bool:
    cmd = [sys.executable, "-m", *_fill(task.args, now)]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=task.timeout_s, creationflags=NO_WINDOW)
        tail = (out.stdout.strip().splitlines() or [""])[-1][:160]
        log(f"{task.name}: {'ok' if out.returncode == 0 else f'erro {out.returncode}'} {tail}")
        if out.returncode != 0:
            log(f"  {(out.stderr.strip().splitlines() or [''])[-1][:200]}")
        return out.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"{task.name}: passou de {task.timeout_s}s, interrompido")
        return False


def run_forever(poll_s: int = 30) -> None:
    def log(msg: str) -> None:
        print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    log("coleta contínua iniciada")
    while not STOP.exists():
        now = pd.Timestamp.now(tz="UTC")
        last = _load()
        for task in due(TASKS, last, now):
            if STOP.exists():
                break
            run_task(task, now, log)
            last[task.name] = str(now)  # mesmo em erro: tenta de novo no próximo intervalo, sem martelar a fonte
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(last, indent=2), encoding="utf-8")
        time.sleep(poll_s)
    log("coleta parada pelo painel")


if __name__ == "__main__":
    run_forever()
