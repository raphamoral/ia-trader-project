"""Canal direto com a LLM local (Qwen no Ollama): conversa sem passar pelo Claude.

O contexto do projeto (o que funcionou, o estado da tendência diária, o playbook dos livros)
vai no prompt de sistema; com "consultar os livros", trechos do índice entram na conversa,
com livro e página. A resposta vem em streaming, palavra por palavra.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Iterator

DATA = Path(__file__).resolve().parents[2] / "data"
OLLAMA = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen2.5:7b"

SYSTEM = """Você é o especialista de trading do projeto AI Trader do usuário (cripto na Binance; uso pessoal,
objetivo: estabilidade financeira). Responda em português, de forma direta e honesta: diga quando algo não
tem evidência. Você é a LLM local do sistema: no futuro escreverá as POLÍTICAS que os agentes executam
em milissegundos (direção, tamanho, stops, gatilhos, reações pré-aprovadas a notícias).

O que os testes com dados reais (fora da amostra) mostraram até agora:
- Tendência diária (comitê de médias 50/100/150/200 + volatilidade alvo de 30%, 5 criptos): FUNCIONOU; menos queda
  que comprar e segurar (2018-2023 Calmar 1,13; 2024-2026 Calmar 0,94, 18% ao ano, queda máx. 19,5%).
- Day trade/scalper com indicadores clássicos, sentimento de notícias, momentum entre moedas, funding e
  estratégias inventadas por LLM: NÃO deram vantagem depois das taxas.
- Alarme de hack grande (>= US$100M, metade da exposição por 24h): seguro barato, efeito pequeno.
Arquitetura combinada: raciocínio (LLM + livros, segundos) escreve a política; reflexo (milissegundos)
aplica reações pré-aprovadas; agentes executam só dentro da política; ranking por lucro líquido.
Saídas disponíveis para os agentes de day trade: stop inicial, alvo, trailing, trava de lucro em degraus
(lock_at/lock_keep) e "deixa correr" (runner_vol/runner_atr: com volatilidade alta a favor, cancela o alvo
e só sobe o stop). Os valores são escolhidos pela evolução e validados fora da amostra.

Regras de honestidade (obrigatórias):
- Você NÃO tem memória entre conversas, NÃO roda testes, NÃO opera e NÃO aprende sozinha. Quem aprende é o
  sistema: backtests, holdout e a evolução dos agentes. Nunca diga que "aprendeu com as perdas" ou que "ajustou".
- Não invente números, capacidades, módulos ou nomes ("novo modelo", machine learning, VIX para cripto etc.).
  Se não está nos dados acima, diga que não sabe.
- O retorno de poucos dias da tendência em papel é ruído; a queda no primeiro dia é a taxa de compra.
- Toda sugestão de parâmetro novo é "hipótese a testar fora da amostra", nunca recomendação.
- Respostas curtas: no máximo ~12 linhas, sem repetir listas já ditas na conversa."""


def _trend_summary() -> dict:
    """Lê o registro da tendência diária direto (sem depender do painel)."""
    import sqlite3

    db = DATA / "live" / "tendencia.sqlite"
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
        peak, dd = max(peak, e), max(dd, 1 - e / max(peak, e))
    return {"start": rows[0][0], "equity": eq[-1], "ret": eq[-1] / 1000.0 - 1, "max_dd": dd,
            "weights": json.loads(rows[-1][2])}


def _trend_status() -> str:
    try:
        t = _trend_summary()
        if not t:
            return "Tendência diária em papel: sem dados ainda."
        w = ", ".join(f"{k.split('/')[0]} {v:.0%}" for k, v in t["weights"].items())
        return (f"Tendência diária em papel (desde {t['start']}): patrimônio R${t['equity']:.2f},"
                f" retorno {t['ret']:+.2%}, maior queda {t['max_dd']:.2%}; posição atual: {w}.")
    except Exception:
        return ""


def _playbook() -> str:
    try:
        from trader.evolution.priors import standing_lessons

        return "Playbook dos livros (regras com mais apoio):\n" + "\n".join(
            f"- {r}" for r in standing_lessons(DATA / "books" / "playbook.json", 8))
    except Exception:
        return ""


_LIBRARY = None


def _books(question: str, k: int = 4) -> str:
    """Busca híbrida no índice dos livros (carregado uma vez só, quando pedido)."""
    global _LIBRARY
    index = DATA / "books" / "index"
    if not (index / "chunks.parquet").exists():
        return "(índice dos livros não encontrado)"
    if _LIBRARY is None:
        from trader.books.index import Library

        _LIBRARY = Library.load(index)
    hits = _LIBRARY.search(question, k=k)
    return "Trechos dos livros (cite como [livro p.N] quando usar):\n" + "\n\n".join(
        f"[{h['book']} p.{h['page']}] {h['text'][:700]}" for h in hits)


def build_messages(history: list[dict[str, str]], use_books: bool) -> list[dict[str, str]]:
    system = "\n\n".join(x for x in (SYSTEM, _trend_status(), _playbook()) if x)
    msgs = [{"role": "system", "content": system}]
    turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")][-12:]
    if use_books and turns and turns[-1]["role"] == "user":
        turns = turns[:-1] + [{"role": "user", "content": _books(turns[-1]["content"]) + "\n\nPergunta: " + turns[-1]["content"]}]
    return msgs + turns


def provider():
    """Provedor do chat: Ollama local por padrão; Anthropic se TRADER_LLM_PROVIDER=anthropic."""
    from trader import llm

    return llm.config()


def stream(history: list[dict[str, str]], use_books: bool = False) -> Iterator[str]:
    from trader import llm

    yield from llm.stream(build_messages(history, use_books), provider(), temperature=0.4)
