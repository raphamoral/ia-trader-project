"""Conhecimento prévio para a evolução: lições permanentes e faixas de partida dos genes.

Origem, em ordem de preferência:
1. playbook destilado dos livros (python -m trader.books distill), com citações;
2. hipóteses iniciais de trading (abaixo), marcadas como hipótese.
Nada disso é regra fixa: entra nos prompts e na criação de genomas novos, e a
seleção pode refutar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PLAYBOOK = Path("data/books/playbook.json")

HYPOTHESES = [
    "(hipótese) Arriscar 0,5% a 1% do capital por operação; nunca mais de 2%.",
    "(hipótese) Stop diário de 2% a 3%: bateu, para de operar no dia.",
    "(hipótese) Tamanho da posição pela volatilidade (ATR), não por valor fixo.",
    "(hipótese) Seguir tendência no tempo gráfico maior e reverter à média só em mercado lateral.",
    "(hipótese) Em expansão extrema de volatilidade, ficar de fora ou reduzir o tamanho.",
    "(hipótese) Evitar os primeiros 5 a 15 minutos do pregão da B3 e os horários de funding em cripto (00h, 08h, 16h UTC).",
    "(hipótese) Não abrir posição nova perto de eventos agendados (FOMC, Copom, payroll, CPI).",
    "(hipótese) Operar menos com taxa alta: em 15m a 0,1% por lado, quem opera muito morre pela taxa.",
]


def load_playbook(path: Path = PLAYBOOK) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def standing_lessons(path: Path = PLAYBOOK, n: int = 8) -> list[str]:
    """Lições permanentes: as regras dos livros com mais apoio, ou as hipóteses iniciais."""
    playbook = load_playbook(path)
    if playbook and playbook.get("rules"):
        from trader.books.distill import top_rules

        return top_rules(playbook, n)
    return HYPOTHESES[:n]


def gene_priors(path: Path = PLAYBOOK) -> dict[str, dict[str, Any]]:
    playbook = load_playbook(path)
    return (playbook or {}).get("priors", {})
