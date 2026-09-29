"""Backfill histórico pelo GDELT DOC 2.0 (gratuito, sem chave).

- A busca é translingual: um termo em inglês encontra artigos em ~65 idiomas
  (o GDELT indexa traduções automáticas); o título volta no idioma original.
- `seendate` é quando o GDELT viu o artigo (varredura a cada 15 min). Usamos
  como published_at: é conservador (nunca antes da publicação real).
- Limite oficial: 1 requisição a cada 5 s por IP; na prática bloqueia com
  frequência maior. Respondemos com espera crescente e novas tentativas.
- Máximo de 250 artigos por requisição: janelas longas perdem artigos.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import pandas as pd

API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250

# termos de busca por ativo (translinguais: em inglês)
QUERIES = {
    "BTC": '(bitcoin OR "crypto market" OR cryptocurrency)',
    "IBOV": '(ibovespa OR "brazilian stocks" OR "brazil stock market")',
    "PETR4": "petrobras",
    "VALE3": '("vale sa" OR "vale iron ore" OR "vale mining")',
    "USDBRL": '("brazilian real" OR "real dollar" OR "brazil currency")',
    "MACRO": '("federal reserve" OR fomc OR copom OR "banco central do brasil")',
}

_LANGS = {
    "English": "en", "Portuguese": "pt", "Spanish": "es", "German": "de", "French": "fr",
    "Italian": "it", "Chinese": "zh", "Japanese": "ja", "Korean": "ko", "Russian": "ru",
}


def _http_get(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-trader-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


class Gdelt:
    def __init__(
        self,
        get: Callable[[str], str] = _http_get,
        min_interval: float = 6.0,
        max_retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.get, self.sleep, self.clock = get, sleep, clock
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = -1e9
        self.requests = 0

    def _request(self, params: dict) -> list[dict]:
        url = API + "?" + urllib.parse.urlencode(params)
        wait = self.min_interval
        for _ in range(self.max_retries):
            gap = self.clock() - self._last
            if gap < self.min_interval:
                self.sleep(self.min_interval - gap)
            self._last = self.clock()
            self.requests += 1
            try:
                body = self.get(url)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                body = ""  # tratado como limite de taxa abaixo
                exc.close()
            except (urllib.error.URLError, TimeoutError):
                body = ""
            else:
                if not body.strip():
                    return []  # sem resultados
            if body.lstrip().startswith("{"):
                return json.loads(body).get("articles", [])
            wait *= 2  # mensagem de limite de taxa: espera crescente
            self.sleep(wait)
        raise RuntimeError("GDELT recusou repetidamente (limite de taxa)")

    def articles(self, query: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        return self._request(
            {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": MAX_RECORDS,
                "sort": "datedesc",
                "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            }
        )

    def backfill(
        self, asset: str, since: pd.Timestamp, until: pd.Timestamp, window: pd.Timedelta = pd.Timedelta(days=1)
    ) -> list[dict]:
        """Itens brutos (formato do NewsStore) para o ativo, janela a janela."""
        query = QUERIES.get(asset, asset)
        rows, cursor = [], since
        now = pd.Timestamp.now(tz="UTC")
        while cursor < until:
            end = min(cursor + window, until)
            for a in self.articles(query, cursor, end):
                rows.append(
                    {
                        "source": "gdelt",
                        "domain": a.get("domain", ""),
                        "url": a.get("url", ""),
                        "original_lang": _LANGS.get(a.get("language", ""), a.get("language", "").lower()[:2]),
                        "original_text": " ".join(a.get("title", "").split()),
                        "published_at": pd.to_datetime(a["seendate"], format="%Y%m%dT%H%M%SZ", utc=True),
                        "first_seen_at": now,
                        "backfilled": True,
                        "query_assets": asset,
                    }
                )
            cursor = end
        return rows
