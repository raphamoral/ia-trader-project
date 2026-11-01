import json
from types import SimpleNamespace

import pytest

from trader import llm


class FakeAnthropic:
    """Imita o SDK: registra a chamada e devolve texto, ou recusa."""

    def __init__(self, text='{"ok": true}', refuse=False):
        self.calls = []
        self.text, self.refuse = text, refuse
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create(beta=True, **kw)))
        self.messages = SimpleNamespace(create=lambda **kw: self._create(beta=False, **kw))

    def _create(self, beta, **kw):
        self.calls.append((beta, kw))
        return SimpleNamespace(
            model=kw["model"], stop_reason="refusal" if self.refuse else "end_turn",
            stop_details=SimpleNamespace(category="cyber") if self.refuse else None,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.text)],
        )


def test_default_provider_is_local(monkeypatch):
    for k in ("TRADER_LLM_PROVIDER", "TRADER_LLM_MODEL", "TRADER_LLM_URL"):
        monkeypatch.delenv(k, raising=False)
    cfg = llm.config()
    assert cfg.provider == "ollama" and cfg.model == "qwen2.5:7b" and "Ollama" in cfg.label
    monkeypatch.setenv("TRADER_LLM_PROVIDER", "anthropic")
    cfg = llm.config()
    assert cfg.provider == "anthropic" and cfg.model == "claude-fable-5-1" and "por token" in cfg.label
    assert llm.config("ollama").provider == "ollama"  # argumento vence o ambiente
    with pytest.raises(ValueError):
        llm.config("openai")


def test_anthropic_call_splits_system_and_counts_cost():
    fake = FakeAnthropic()
    cfg = llm.LLMConfig("anthropic", "claude-fable-5-1")
    before = (llm.usage.calls, llm.usage.cost_usd)
    out = llm.chat([{"role": "system", "content": "regras"}, {"role": "user", "content": "oi"}], cfg,
                   json_mode=True, client=fake)
    assert json.loads(out) == {"ok": True}
    beta, kw = fake.calls[0]
    assert beta and kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    assert kw["system"].startswith("regras") and "JSON" in kw["system"]
    assert kw["messages"] == [{"role": "user", "content": "oi"}] and "thinking" not in kw
    assert llm.usage.calls == before[0] + 1
    # 1000 de entrada a US$10/M + 200 de saída a US$50/M
    assert llm.usage.cost_usd - before[1] == pytest.approx(0.01 + 0.01)
    assert llm.usage.last["cost_usd"] == pytest.approx(0.02)


def test_models_without_fallback_use_plain_endpoint():
    fake = FakeAnthropic()
    llm.chat([{"role": "user", "content": "oi"}], llm.LLMConfig("anthropic", "claude-haiku-4-5"), client=fake)
    beta, kw = fake.calls[0]
    assert not beta and "fallbacks" not in kw


def test_refusal_raises():
    fake = FakeAnthropic(refuse=True)
    with pytest.raises(RuntimeError, match="recusou"):
        llm.chat([{"role": "user", "content": "oi"}], llm.LLMConfig("anthropic", "claude-fable-5-1"), client=fake)


def test_policy_writer_accepts_provider_chat(monkeypatch):
    from trader.policy import writer

    fake = FakeAnthropic(text=json.dumps({"valid_minutes": 30, "symbols": {"SOL/USDT": {"bias": "flat"}},
                                          "reactions": [], "rationale": "t"}))
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
    chat = writer.provider_chat(llm.LLMConfig("anthropic", "claude-fable-5-1"))
    policy, report = writer.write_policy({"market": {}}, ["SOL/USDT"], chat=chat)
    assert policy.symbols["SOL/USDT"].bias == "flat" and report["fallback"] is None


def test_claude_cli_provider_parses_result_and_usage(monkeypatch):
    seen = {}

    def fake_run(cmd, input, **kw):
        seen["cmd"], seen["input"] = cmd, input
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "is_error": False, "result": '{"ok": 1}', "total_cost_usd": 0.12,
            "usage": {"input_tokens": 50, "output_tokens": 9}, "modelUsage": {"claude-fable-5-1": {}}}))

    monkeypatch.delenv("TRADER_CLAUDE_CMD", raising=False)
    cfg = llm.LLMConfig("claude-cli", "fable")
    out = llm.chat([{"role": "system", "content": "regras"}, {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], cfg, json_mode=True,
                   run=fake_run)
    assert out == '{"ok": 1}'
    cmd = seen["cmd"]
    assert cmd[:len(llm.CLAUDE_CMD)] == llm.CLAUDE_CMD and "-p" in cmd and cmd[cmd.index("--model") + 1] == "fable"
    assert cmd[cmd.index("--tools") + 1] == "" and "--system-prompt" in cmd and "JSON" in cmd[-1]
    assert "Usuário: a" in seen["input"] and seen["input"].rstrip().endswith("do usuário.")
    assert llm.usage.last == {"model": "claude-fable-5-1", "input_tokens": 50, "output_tokens": 9, "cost_usd": 0.12,
                              "seconds": llm.usage.last["seconds"]}
    assert "assinatura" in cfg.label


def test_claude_cli_errors_surface(monkeypatch):
    def fake_run(cmd, input, **kw):
        return SimpleNamespace(returncode=1, stderr="", stdout=json.dumps({"is_error": True, "result": "Not logged in"}))

    with pytest.raises(RuntimeError, match="Not logged in"):
        llm.chat([{"role": "user", "content": "a"}], llm.LLMConfig("claude-cli", "fable"), run=fake_run)
