from types import SimpleNamespace

from trader.panel import server


def test_memory_guard_pauses_least_important_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path / "state.json")
    running = {"scalp_tudo", "gdelt", "dt_tudo", "tendencia"}
    calls = []
    monkeypatch.setattr(server, "is_running", lambda k: k in running)
    monkeypatch.setattr(server, "start", lambda k: (calls.append(("start", k)), running.add(k)))

    def fake_stop(k):
        calls.append(("stop", k))
        running.discard(k)
        st = server._load_state()
        st.setdefault(k, {})["wanted"], st[k]["paused"] = False, False
        server._save_state(st)

    monkeypatch.setattr(server, "stop", fake_stop)
    server._save_state({k: {"wanted": True} for k in running})
    ram = {"gb": 1.0}
    monkeypatch.setattr(server.psutil, "virtual_memory", lambda: SimpleNamespace(available=ram["gb"] * 2**30))
    monkeypatch.setattr(server, "system_usage_gb", lambda: 1.0)

    server.memory_guard()  # pouca RAM: pausa o scalper primeiro
    server.memory_guard()  # ainda pouca: pausa o GDELT
    assert calls == [("stop", "scalp_tudo"), ("stop", "gdelt")]
    assert "tendencia" in running  # tendência nunca é pausada
    st = server._load_state()
    assert st["scalp_tudo"]["paused"] and st["scalp_tudo"]["wanted"]

    server.resume_wanted()  # o vigia não religa o que está pausado
    assert ("start", "scalp_tudo") not in calls

    ram["gb"] = 4.0
    server.memory_guard()  # RAM folgou: religa o mais importante primeiro
    assert calls[-1] == ("start", "gdelt")


def test_memory_guard_respects_system_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path / "state.json")
    running = {"scalp_tudo", "dt_tudo"}
    stopped = []
    monkeypatch.setattr(server, "is_running", lambda k: k in running)
    monkeypatch.setattr(server, "stop", lambda k: (stopped.append(k), running.discard(k)))
    server._save_state({k: {"wanted": True} for k in running})
    monkeypatch.setattr(server.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 2**30))  # PC folgado
    monkeypatch.setattr(server, "system_usage_gb", lambda: 3.2)  # mas o sistema passou do teto
    server.memory_guard()
    assert stopped == ["scalp_tudo"]


def test_guard_pauses_the_job_that_exceeds_its_own_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATE", tmp_path / "state.json")
    running = {"scalp_tudo", "dt_tudo", "tendencia"}
    stopped = []
    monkeypatch.setattr(server, "is_running", lambda k: k in running)
    monkeypatch.setattr(server, "stop", lambda k: (stopped.append(k), running.discard(k)))
    monkeypatch.setattr(server, "job_usage_gb", lambda k: 5.0 if k == "dt_tudo" else 0.2)
    server._save_state({k: {"wanted": True} for k in running})
    server.memory_guard()
    assert stopped == ["dt_tudo"]  # o culpado, não o primeiro da fila
    assert server._load_state()["dt_tudo"]["paused"]


def test_reused_pid_is_not_the_job(monkeypatch):
    """PID antigo reaproveitado por outro programa: não conta como rodando e nunca é derrubado."""
    import os

    from trader.panel import server

    job = server.JOBS["tendencia"]
    assert not server._owned(os.getpid(), job)  # o processo do pytest não é a tendência
    killed = []
    monkeypatch.setattr(server, "_kill_tree", killed.append)
    monkeypatch.setattr(server, "_pids_matching", lambda frag: [])
    monkeypatch.setattr(server, "_load_state", lambda: {"tendencia": {"pid": os.getpid(), "wanted": True}})
    monkeypatch.setattr(server, "_save_state", lambda st: None)
    assert not server.is_running("tendencia")
    server.stop("tendencia")
    assert killed == []


def test_process_scan_is_cached(monkeypatch):
    """Uma varredura serve a todos por alguns segundos; forçar ou invalidar refaz."""
    from trader.panel import server
    import psutil

    calls = []

    class FakeProc:
        info = {"pid": 1, "name": "python.exe", "cmdline": ["python", "-m", "trader.live.trend_runner"],
                "memory_info": type("M", (), {"rss": 2**30})()}

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: calls.append(1) or [FakeProc()])
    server._invalidate_scan()
    assert server._pids_matching("trend_runner") == [1]
    assert server.system_usage_gb() == 1.0
    assert server._pids_matching("trend_runner") == [1]
    assert len(calls) == 1  # três consultas, uma varredura
    server._invalidate_scan()
    server._pids_matching("x")
    assert len(calls) == 2
    server._invalidate_scan()
