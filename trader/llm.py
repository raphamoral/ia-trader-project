"""Provedor de LLM do sistema: Ollama (padrão, local, sem custo) ou Anthropic (opcional, por token).

Escolha por variáveis de ambiente ou por argumento:
  TRADER_LLM_PROVIDER   ollama (padrão) | anthropic | claude-cli
  TRADER_CLAUDE_CMD     claude-cli: comando do CLI (padrão: `claude` no PATH)
  TRADER_LLM_MODEL      modelo (padrão: qwen2.5:7b no Ollama; claude-fable-5-1 na Anthropic)
  TRADER_LLM_URL        só Ollama (padrão http://localhost:11434/v1)
  ANTHROPIC_API_KEY     chave da Anthropic (lida pelo SDK; nunca fica no código)

A Anthropic cobra por token: cada chamada registra tokens e custo estimado em `usage`.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterator

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"  # chave e provedor podem ficar aqui (fora do git)


def load_env_file(path: Path = ENV_FILE) -> None:
    """Carrega KEY=valor de .env para o ambiente, sem sobrescrever o que já está definido."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


load_env_file()

PROVIDERS = ("ollama", "anthropic", "claude-cli")
DEFAULT_MODEL = {"ollama": "qwen2.5:7b", "anthropic": "claude-fable-5-1", "claude-cli": "fable"}
# claude-cli: o Claude Code em modo headless (`claude -p`), com o login da assinatura do usuário, sem chave.
# O CLI logado está no WSL; o Python roda no Windows, então chama por wsl.exe (sem shell no meio).
CLAUDE_CMD = ["claude"]
OLLAMA_URL = "http://localhost:11434/v1"
# US$ por milhão de tokens (entrada, saída), tabela de 2026-09
PRICES = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}
FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5")  # aceitam fallbacks="default"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = DEFAULT_MODEL["ollama"]
    url: str = OLLAMA_URL

    @property
    def label(self) -> str:
        kind = {"anthropic": "Anthropic, por token", "claude-cli": "Claude Code, assinatura"}.get(self.provider, "Ollama local")
        return f"{self.model} ({kind})"


def config(provider: str | None = None, model: str | None = None, url: str | None = None) -> LLMConfig:
    provider = (provider or os.environ.get("TRADER_LLM_PROVIDER") or "ollama").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"provider desconhecido: {provider!r} (use {PROVIDERS})")
    model = model or os.environ.get("TRADER_LLM_MODEL") or DEFAULT_MODEL[provider]
    return LLMConfig(provider, model, url or os.environ.get("TRADER_LLM_URL") or OLLAMA_URL)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    last: dict[str, Any] = field(default_factory=dict)

    def add(self, model: str, inp: int, out: int, seconds: float) -> None:
        pi, po = PRICES.get(model, (0.0, 0.0))
        cost = (inp * pi + out * po) / 1e6
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost
        self.last = {"model": model, "input_tokens": inp, "output_tokens": out, "cost_usd": round(cost, 4),
                     "seconds": round(seconds, 1)}


usage = Usage()


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")]
    return system, turns


def _ollama_chat(messages, cfg: LLMConfig, json_mode: bool, temperature: float, timeout: int, schema=None) -> str:
    if schema is not None:
        # decodificação restrita: a API nativa aceita um JSON Schema em `format`
        payload = {"model": cfg.model, "messages": messages, "format": schema, "stream": False,
                   "options": {"temperature": temperature, "num_predict": 4000}}
        base = cfg.url[:-3] if cfg.url.endswith("/v1") else cfg.url
        req = urllib.request.Request(f"{base}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        usage.add(cfg.model, int(data.get("prompt_eval_count", 0) or 0), int(data.get("eval_count", 0) or 0), 0.0)
        return data["message"]["content"]
    payload: dict[str, Any] = {"model": cfg.model, "messages": messages, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{cfg.url}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    u = data.get("usage") or {}
    usage.add(cfg.model, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), 0.0)
    return data["choices"][0]["message"]["content"]


def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()  # chave vem do ambiente (ANTHROPIC_API_KEY)


def _anthropic_chat(messages, cfg: LLMConfig, json_mode: bool, max_tokens: int, client=None) -> str:
    import time

    client = client or _anthropic_client()
    system, turns = _split_system(messages)
    if json_mode:
        system += "\n\nAnswer with a single JSON object and nothing else."
    kwargs: dict[str, Any] = dict(model=cfg.model, max_tokens=max_tokens, messages=turns,
                                  output_config={"effort": "medium"})
    if system:
        kwargs["system"] = system
    t0 = time.perf_counter()
    if cfg.model in FALLBACK_MODELS:
        # se os classificadores de segurança recusarem, a própria API refaz noutro modelo
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    u = getattr(resp, "usage", None)
    usage.add(resp.model if hasattr(resp, "model") else cfg.model, int(getattr(u, "input_tokens", 0) or 0),
              int(getattr(u, "output_tokens", 0) or 0), time.perf_counter() - t0)
    if resp.stop_reason == "refusal":
        det = getattr(resp, "stop_details", None)
        raise RuntimeError(f"a Anthropic recusou a resposta ({getattr(det, 'category', None)})")
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _render_turns(turns: list[dict[str, str]]) -> str:
    """Histórico como texto único para o CLI (que recebe um prompt só)."""
    if len(turns) == 1:
        return turns[0]["content"]
    names = {"user": "Usuário", "assistant": "Assistente"}
    body = "\n\n".join(f"{names[t['role']]}: {t['content']}" for t in turns)
    return f"Conversa até aqui:\n\n{body}\n\nResponda à última mensagem do usuário."


def _claude_cli_chat(messages, cfg: LLMConfig, json_mode: bool, timeout: int, run=None) -> str:
    import shlex
    import subprocess
    import time

    run = run or subprocess.run
    system, turns = _split_system(messages)
    if json_mode:
        system += "\n\nAnswer with a single JSON object and nothing else."
    base = shlex.split(os.environ["TRADER_CLAUDE_CMD"]) if os.environ.get("TRADER_CLAUDE_CMD") else CLAUDE_CMD
    cmd = [*base, "-p", "--output-format", "json", "--model", cfg.model, "--no-session-persistence",
           "--tools", "", "--max-turns", "1"]
    if system:
        cmd += ["--system-prompt", system]
    t0 = time.perf_counter()
    r = run(cmd, input=_render_turns(turns), capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout)
    raw = r.stdout or ""
    try:
        d = json.loads(raw[raw.index("{"):])
    except (ValueError, json.JSONDecodeError):
        raise RuntimeError(f"claude-cli sem resposta JSON (rc {r.returncode}): {(r.stderr or raw)[:300]}")
    u = d.get("usage") or {}
    model = next(iter(d.get("modelUsage") or {}), cfg.model)
    usage.add(model, int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0),
              time.perf_counter() - t0)
    usage.last["cost_usd"] = round(float(d.get("total_cost_usd") or 0.0), 4)  # estimativa do próprio CLI
    if d.get("is_error"):
        raise RuntimeError(f"claude-cli: {d.get('result')}")
    return str(d.get("result") or "")


def chat(messages: list[dict[str, str]], cfg: LLMConfig | None = None, json_mode: bool = False,
         temperature: float = 0.2, timeout: int = 180, max_tokens: int = 8000, client=None, run=None,
         schema: dict | None = None) -> str:
    """Uma resposta completa. `messages` no formato {role, content} (system incluído).

    `schema`: JSON Schema da resposta. No Ollama vira decodificação restrita; nos outros, só instrução."""
    cfg = cfg or config()
    if cfg.provider == "anthropic":
        return _anthropic_chat(messages, cfg, json_mode or schema is not None, max_tokens, client)
    if cfg.provider == "claude-cli":
        return _claude_cli_chat(messages, cfg, json_mode or schema is not None, max(timeout, 300), run)
    return _ollama_chat(messages, cfg, json_mode, temperature, max(timeout, 600) if schema else timeout, schema)


def stream(messages: list[dict[str, str]], cfg: LLMConfig | None = None, temperature: float = 0.4,
           max_tokens: int = 4000) -> Iterator[str]:
    """Resposta em pedaços (para o chat)."""
    cfg = cfg or config()
    if cfg.provider == "claude-cli":
        yield _claude_cli_chat(messages, cfg, False, 300)  # o CLI responde inteiro de uma vez
        return
    if cfg.provider == "anthropic":
        import time

        client = _anthropic_client()
        system, turns = _split_system(messages)
        kwargs: dict[str, Any] = dict(model=cfg.model, max_tokens=max_tokens, messages=turns,
                                      output_config={"effort": "medium"})
        if system:
            kwargs["system"] = system
        t0 = time.perf_counter()
        with client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield text
            final = s.get_final_message()
        u = final.usage
        usage.add(final.model, u.input_tokens, u.output_tokens, time.perf_counter() - t0)
        if final.stop_reason == "refusal":
            yield "\n[a Anthropic recusou continuar esta resposta]"
        return
    payload = {"model": cfg.model, "messages": messages, "stream": True, "temperature": temperature}
    req = urllib.request.Request(f"{cfg.url}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                yield delta
