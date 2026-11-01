"""Chat com o Qwen local: python -m trader.chat  →  http://127.0.0.1:8766

Só conversa com a LLM; não liga nenhuma tarefa do laboratório.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trader.chat import core

PORT = 8766


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/chat"):
            body = (Path(__file__).parent / "chat.html").read_text(encoding="utf-8")
            body = body.replace("(Qwen local)", f"({core.provider().label})").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for piece in core.stream(body.get("messages", []), bool(body.get("books"))):
                self.wfile.write(piece.encode("utf-8"))
                self.wfile.flush()
        except Exception as exc:
            if "10061" in str(exc) or "refused" in str(exc).lower() or "recusou" in str(exc):
                msg = "\n[O Ollama está desligado. Abra o Ollama (ou use o atalho 'AI Trader - Chat', que abre os dois) e envie de novo.]"
            else:
                msg = f"\n[erro ao falar com a LLM local: {type(exc).__name__}: {exc}]"
            self.wfile.write(msg.encode("utf-8"))

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
