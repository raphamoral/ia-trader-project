"""Painel local: liga/desliga os processos longos e mostra o status.

Só escuta em 127.0.0.1. Guarda o estado desejado (ligado/desligado) em
data/panel/state.json: ao reiniciar o PC, o painel religa o que estava ligado.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STATE = DATA / "panel" / "state.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
# console oculto (nem o .bat nem os filhos abrem janela) e prioridade abaixo do normal:
# o Windows dá preferência aos programas do usuário (Chrome etc.); os filhos herdam a prioridade
HIDDEN = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
          | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))


# Varrer todos os processos do Windows custa segundos (16 s já medidos). A varredura é feita uma vez e
# reaproveitada por alguns segundos por todo mundo (status, guarda de memória, prioridade); senão a página,
# que pede status a cada 5 s, acumula pedidos e o painel para de responder.
SCAN_TTL_S = 10.0
_scan_lock = threading.Lock()
_scan_cache: tuple[float, list[tuple[int, str, str, int]]] = (0.0, [])


def _procs(force: bool = False) -> list[tuple[int, str, str, int]]:
    """(pid, nome, linha de comando, rss) de todos os processos, com cache de SCAN_TTL_S."""
    global _scan_cache
    with _scan_lock:
        if not force and time.monotonic() - _scan_cache[0] < SCAN_TTL_S:
            return _scan_cache[1]
    rows = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            mi = p.info["memory_info"]
            rows.append((p.info["pid"], (p.info["name"] or "").lower(), " ".join(p.info["cmdline"] or []),
                         int(mi.rss) if mi else 0))
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    with _scan_lock:
        _scan_cache = (time.monotonic(), rows)
    return rows


def _invalidate_scan() -> None:
    global _scan_cache
    with _scan_lock:
        _scan_cache = (0.0, _scan_cache[1])


def _ours(name: str, cmd: str) -> bool:
    return "AI-trader-project" in cmd or " -m trader" in cmd or name in ("llama-server.exe", "ollama.exe")


def lower_priority() -> int:
    """Rebaixa a prioridade de todos os processos do projeto e do servidor da LLM. Retorna quantos mudou."""
    changed = 0
    low = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10)
    for pid, name, cmd, _ in _procs():
        if not _ours(name, cmd) or "trader.panel" in cmd:
            continue
        try:
            p = psutil.Process(pid)
            if p.nice() != low:
                p.nice(low)
                changed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return changed


@dataclass(frozen=True)
class Job:
    key: str
    title: str
    command: list[str]
    log: Path
    match: str  # trecho da linha de comando que identifica o processo
    stop_file: Path | None = None  # sinaliza a parada (o processo ou o .bat checa este arquivo)
    kill_on_stop: bool = False  # além do sinal, encerra os processos na hora
    redirect_log: bool = True  # False quando o próprio comando já escreve no log (ex.: o .bat)


DAY_BOOT = str(DATA / "evolution" / "20260929-191129-day-risk-s0.sqlite")
SCALP_BOOT = str(DATA / "evolution" / "20260929-191304-day-risk-s0.sqlite")


def _live_job(name: str, title: str, features: str, style: str = "swing") -> Job:
    if style == "swing":
        command = ["cmd.exe", "/c", str(ROOT / "scripts" / "live_24h.bat"), name, features]
    else:
        boot = DAY_BOOT if style == "day" else SCALP_BOOT
        command = ["cmd.exe", "/c", str(ROOT / "scripts" / "daytrade_24h.bat"), name, style, features, boot]
    return Job(
        name,
        title,
        command,
        DATA / "live" / f"{name}.log",
        f"--db data\\live\\{name}.sqlite",
        DATA / "live" / f"{name}.STOP",
        kill_on_stop=True,
        redirect_log=False,
    )


# as três populações partem do mesmo ponto; a diferença entre elas mede o valor de cada fonte
LIVE = {
    "dt_continuo": ("Day trade 5m — contínuo: só preço (controle)", "none", "day"),
    "dt_sem_black": ("Day trade 5m — sem black: preço + white", "white", "day"),
    "dt_tudo": ("Day trade 5m — tudo: preço + white + black", "all", "day"),
    "scalp_tudo": ("Scalper 1m — tudo", "all", "scalp"),
}

JOBS = {
    **{k: _live_job(k, t, f, st) for k, (t, f, st) in LIVE.items()},
    "tendencia": Job(
        "tendencia",
        "Tendência diária (5 criptos, comitê de tendência + vol 30%, R$1.000)",
        [str(PY), "-m", "trader.live.trend_runner"],
        DATA / "live" / "tendencia.log",
        "trader.live.trend_runner",
        kill_on_stop=True,
    ),
    "tendencia_filtro": Job(
        "tendencia_filtro",
        "Tendência diária + filtro (meta-labeling), lado a lado com a de cima",
        [str(PY), "-m", "trader.live.trend_filter_runner"],
        DATA / "live" / "tendencia_filtro.log",
        "trader.live.trend_filter_runner",
        kill_on_stop=True,
    ),
    "tendencia_ouro": Job(
        "tendencia_ouro",
        "Tendência diária: 5 criptos + ouro (PAXG), lado a lado com as de cima",
        [str(PY), "-m", "trader.live.trend_gold_runner"],
        DATA / "live" / "tendencia_ouro.log",
        "trader.live.trend_gold_runner",
        kill_on_stop=True,
    ),
    "coleta": Job(
        "coleta",
        "Coleta contínua de notícias (white + black + dados alternativos)",
        [str(PY), "-m", "trader.live.collector"],
        DATA / "live" / "coleta.log",
        "trader.live.collector",
        DATA / "live" / "coleta.STOP",
    ),
    "gdelt": Job(
        "gdelt",
        "Download do histórico de notícias (GDELT)",
        [str(PY), "-m", "trader.news", "gkg-backfill", "--since", "2023-01-01", "--no-earlier", "--max-mbps", "4"],
        DATA / "news" / "gkg_backfill.log",
        "gkg-backfill",
        DATA / "news" / "gdelt_gkg" / "STOP",
    ),
}


def _load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _pid_alive(pid: int | None) -> bool:
    return bool(pid) and psutil.pid_exists(pid)


def _pids_matching(fragment: str) -> list[int]:
    """PIDs cuja linha de comando contém `fragment` (inclui processos iniciados fora do painel)."""
    return [pid for pid, _, cmd, _ in _procs() if fragment in cmd and "multiprocessing" not in cmd]


def _tail(path: Path, n: int = 12) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 16_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def _kill_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        for proc in [*root.children(recursive=True), root]:
            proc.kill()
    except psutil.NoSuchProcess:
        pass


def _owned(pid: int | None, job: "Job") -> bool:
    """O PID ainda é deste job? O Windows reaproveita PIDs: um PID antigo pode ser de outro programa
    (já foi o NVIDIA Overlay). Só conta (e só se derruba) se a linha de comando for a do job."""
    if not _pid_alive(pid):
        return False
    try:
        cmd = " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    script = str(job.command[2]) if len(job.command) > 2 else ""
    return job.match in cmd or (bool(script) and script in cmd and job.name in cmd)


def is_running(key: str) -> bool:
    job = JOBS[key]
    return _owned(_load_state().get(key, {}).get("pid"), job) or bool(_pids_matching(job.match))


def start(key: str) -> None:
    job = JOBS[key]
    state = _load_state()
    if not is_running(key):
        if job.stop_file and job.stop_file.exists():
            job.stop_file.unlink()
        job.log.parent.mkdir(parents=True, exist_ok=True)
        out = open(job.log, "ab") if job.redirect_log else subprocess.DEVNULL
        proc = subprocess.Popen(job.command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=HIDDEN)
        state[key] = {"pid": proc.pid}
        _invalidate_scan()
    state.setdefault(key, {})["wanted"] = True
    _save_state(state)


def stop(key: str) -> None:
    job = JOBS[key]
    state = _load_state()
    pid = state.get(key, {}).get("pid")
    if job.stop_file:
        job.stop_file.parent.mkdir(parents=True, exist_ok=True)
        job.stop_file.touch()
    if job.kill_on_stop:
        # derruba a árvore inteira (cmd do .bat + python + workers); o STOP impede o .bat de religar
        own = {pid} if _owned(pid, job) else set()
        for p in own | set(_pids_matching(job.match)):
            _kill_tree(p)
    state.setdefault(key, {})["wanted"] = False
    state[key]["paused"] = False  # desligar manual cancela a pausa automática
    state[key].pop("paused_reason", None)
    _save_state(state)


# guarda de memória: pausa o menos importante primeiro; tendência e coleta nunca são pausadas
PAUSE_ORDER = ["scalp_tudo", "gdelt", "dt_sem_black", "dt_tudo", "dt_continuo"]
NEVER_PAUSE = ("tendencia", "tendencia_filtro", "tendencia_ouro", "coleta")  # leves e são a base do experimento
LOW_RAM_GB, HIGH_RAM_GB = 1.5, 3.0
BUDGET_GB = 3.0  # teto de memória do próprio sistema (processos do projeto + servidor do Qwen)


def system_usage_gb() -> float:
    """Memória somada dos processos do projeto e do servidor da LLM local."""
    return sum(rss for _, name, cmd, rss in _procs() if _ours(name, cmd)) / 2**30


def resume_wanted() -> None:
    """Religa o que deveria estar ligado (exceto o que o guarda de memória pausou)."""
    for key, info in _load_state().items():
        if key in JOBS and info.get("wanted") and not info.get("paused") and not is_running(key):
            start(key)


JOB_CAP_GB = 1.2  # nenhuma tarefa sozinha pode passar disso (pega vazamento/estouro de um processo só)


def job_usage_gb(key: str) -> float:
    """Memória da árvore de processos de uma tarefa (o .bat, o Python e os workers)."""
    job, total, seen = JOBS[key], 0, set()
    roots = set(_pids_matching(job.match)) | ({_load_state().get(key, {}).get("pid")} - {None})
    for pid in roots:
        try:
            root = psutil.Process(pid)
            for proc in [root, *root.children(recursive=True)]:
                if proc.pid not in seen:
                    seen.add(proc.pid)
                    total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / 2**30


def memory_guard() -> None:
    # 1) culpado primeiro: tarefa que passou do próprio teto é pausada na hora
    for key in JOBS:
        if key in NEVER_PAUSE or not is_running(key):
            continue
        if job_usage_gb(key) > JOB_CAP_GB:
            stop(key)
            state = _load_state()
            state[key]["wanted"], state[key]["paused"] = True, True
            state[key]["paused_reason"] = f"passou de {JOB_CAP_GB} GB"
            _save_state(state)
            return
    free_gb = psutil.virtual_memory().available / 2**30
    used_gb = system_usage_gb()
    state = _load_state()
    if free_gb < LOW_RAM_GB or used_gb > BUDGET_GB:
        for key in PAUSE_ORDER:
            info = state.get(key, {})
            if info.get("wanted") and not info.get("paused") and is_running(key):
                stop(key)
                state = _load_state()
                state[key]["wanted"], state[key]["paused"] = True, True  # continua desejada, só pausada
                _save_state(state)
                return  # uma por vez; reavalia no próximo minuto
    elif free_gb > HIGH_RAM_GB and used_gb < BUDGET_GB - 0.4:  # folga antes de religar
        for key in reversed(PAUSE_ORDER):  # religa na ordem inversa: o mais importante primeiro
            if state.get(key, {}).get("paused") and not state[key].get("paused_reason"):  # estourou o teto: não religa
                state[key]["paused"] = False
                _save_state(state)
                start(key)
                return


def live_summary(name: str) -> dict[str, Any]:
    """Métricas de uma população ao vivo, a partir do registro (só leitura)."""
    db = DATA / "live" / f"{name}.sqlite"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        alive_ids = [r[0] for r in con.execute("SELECT id FROM agents WHERE died_at IS NULL")]
        total = con.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        windows = con.execute("SELECT COALESCE(MAX(idx) + 1, 0) FROM life").fetchone()[0]
        causes = dict(con.execute("SELECT cause, COUNT(*) FROM agents WHERE cause IS NOT NULL GROUP BY cause"))
        llm = con.execute("SELECT COUNT(*) FROM agents WHERE note LIKE 'LLM:%'").fetchone()[0]
        lives: dict[str, list[dict]] = {}
        for agent_id, stats in con.execute("SELECT agent_id, stats FROM life ORDER BY idx"):
            lives.setdefault(agent_id, []).append(json.loads(stats))
    finally:
        con.close()

    def life(ws: list[dict]) -> dict[str, float]:
        ret = 1.0
        for w in ws:
            ret *= 1 + w["ret"]
        dd = max((w["max_dd"] for w in ws), default=0.0)
        return {"ret": ret - 1, "dd": dd, "score": (ret - 1) / max(dd, 0.02), "trades": sum(w["trades"] for w in ws)}

    alive = [life(lives[a]) for a in alive_ids if lives.get(a)]

    # hoje (horário de Brasília): retorno por agente, operações e acerto (% de operações com lucro)
    today = pd.Timestamp.now(tz="America/Sao_Paulo").date()
    is_today = lambda w: pd.Timestamp(w["start"]).tz_convert("America/Sao_Paulo").date() == today
    day_rets, wins, total = [], 0, 0
    for agent_id, ws in lives.items():
        if agent_id == "__baseline__":
            continue
        todays = [w for w in ws if is_today(w)]
        if not todays:
            continue
        r = 1.0
        for w in todays:
            r *= 1 + w["ret"]
            trades = (w.get("x") or {}).get("r")
            if trades is not None:
                wins += sum(1 for t in trades if t > 0)
                total += len(trades)
            else:
                total += w["trades"]
        day_rets.append(r - 1)
    best = max(alive, key=lambda x: x["score"], default=None)
    base = life(lives.get("__baseline__", []))
    return {
        "alive": len(alive_ids), "born": total, "generations": windows, "deaths": causes, "llm_successors": llm,
        "best_ret": best["ret"] if best else 0.0, "best_score": best["score"] if best else 0.0,
        "median_ret": sorted(x["ret"] for x in alive)[len(alive) // 2] if alive else 0.0,
        "trades": sum(x["trades"] for x in alive), "baseline_ret": base["ret"],
        "today_best": max(day_rets, default=0.0),
        "today_median": sorted(day_rets)[len(day_rets) // 2] if day_rets else 0.0,
        "today_trades": total,
        "today_hit": (wins / total) if total else None,
    }


def ranking(limit: int = 15) -> list[dict[str, Any]]:
    """Melhores agentes vivos de todos os forks, pela nota (retorno ÷ maior queda) da vida inteira."""
    rows = []
    for name, (title, _, _) in LIVE.items():
        db = DATA / "live" / f"{name}.sqlite"
        if not db.exists():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            agents = con.execute("SELECT id, genome, origin, generation FROM agents WHERE died_at IS NULL").fetchall()
            lives: dict[str, list[dict]] = {}
            for agent_id, stats in con.execute("SELECT agent_id, stats FROM life ORDER BY idx"):
                lives.setdefault(agent_id, []).append(json.loads(stats))
        finally:
            con.close()
        for agent_id, genome, origin, generation in agents:
            ws = lives.get(agent_id, [])
            if not ws:
                continue
            ret = 1.0
            for w in ws:
                ret *= 1 + w["ret"]
            dd = max(w["max_dd"] for w in ws)
            trades = [t for w in ws for t in ((w.get("x") or {}).get("r") or [])]
            n_trades = len(trades) if trades else sum(w["trades"] for w in ws)
            g = json.loads(genome)
            rows.append({
                "fork": name, "id": agent_id, "strategy": g.get("strategy"), "style": g.get("style", "swing"),
                "ret": ret - 1, "dd": dd, "score": (ret - 1) / max(dd, 0.02), "trades": n_trades,
                "hit": (sum(1 for t in trades if t > 0) / len(trades)) if trades else None,
                "age": len(ws), "origin": origin, "generation": generation,
            })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


def trend_summary(name: str = "tendencia") -> dict[str, Any]:
    db = DATA / "live" / f"{name}.sqlite"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT date, equity_brl, weights FROM days ORDER BY date").fetchall()
    finally:
        con.close()
    if not rows:
        return {}
    eq = [r[1] for r in rows]
    peak, dd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        dd = max(dd, 1 - e / peak)
    return {"start": rows[0][0], "date": rows[-1][0], "equity": eq[-1], "ret": eq[-1] / 1000.0 - 1, "max_dd": dd,
            "days": len(rows), "weights": json.loads(rows[-1][2])}


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5)
        return True
    except OSError:
        return False


STATUS_TTL_S = 3.0
_status_lock = threading.Lock()
_status_cache: tuple[float, dict | None] = (0.0, None)


def cached_status(force: bool = False) -> dict[str, Any]:
    """O status é caro (varredura de processos, registros); vários navegadores pedindo ao mesmo tempo
    recebem o mesmo resultado por STATUS_TTL_S segundos."""
    global _status_cache
    with _status_lock:
        if not force and _status_cache[1] is not None and time.monotonic() - _status_cache[0] < STATUS_TTL_S:
            return _status_cache[1]
        data = status()
        _status_cache = (time.monotonic(), data)
        return data


def status() -> dict[str, Any]:
    return {
        "jobs": {
            k: {"title": j.title, "running": is_running(k), "wanted": _load_state().get(k, {}).get("wanted", False),
                "paused": _load_state().get(k, {}).get("paused", False),
                "paused_reason": _load_state().get(k, {}).get("paused_reason"),
                "log": _tail(j.log)}
            for k, j in JOBS.items()
        },
        "live": {k: {"title": t, **live_summary(k)} for k, (t, _, _) in LIVE.items()},
        "ranking": ranking(),
        "trend": trend_summary(),
        "trend_filter": trend_summary("tendencia_filtro"),
        "trend_gold": trend_summary("tendencia_ouro"),
        "ollama": ollama_up(),
        "ram_free_gb": round(psutil.virtual_memory().available / 2**30, 1),
        "system_gb": round(system_usage_gb(), 2),
        "budget_gb": BUDGET_GB,
        "real_orders": False,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._send(200, json.dumps(cached_status()).encode(), "application/json")
        elif self.path == "/":
            self._send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parts = self.path.strip("/").split("/")  # api/<job>/<start|stop>
        if len(parts) == 3 and parts[0] == "api" and parts[1] in JOBS and parts[2] in ("start", "stop"):
            (start if parts[2] == "start" else stop)(parts[1])
            _invalidate_scan()
            self._send(200, json.dumps(cached_status(force=True)).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args) -> None:  # silencia o log de cada requisição
        pass


def _watchdog(interval_s: int = 60) -> None:
    """Religa tarefas marcadas como ligadas que caíram (o .bat reinicia os forks; aqui cobre o resto)."""
    import threading
    import time

    def loop() -> None:
        while True:
            time.sleep(interval_s)
            try:
                lower_priority()
                memory_guard()
                resume_wanted()
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()


def serve(port: int = 8765) -> None:
    lower_priority()
    resume_wanted()
    _watchdog()
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
