"""Captura ao vivo por RSS. first_seen_at é o relógio da captura.

RSS só traz os itens mais recentes (dezenas): serve para operar ao vivo e para
acumular histórico próprio, não para backfill.
"""
from __future__ import annotations

import email.utils
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

FEEDS = {
    "infomoney": ("https://www.infomoney.com.br/feed/", "pt"),
    "valor": ("https://valor.globo.com/rss/valor", "pt"),
    "coindesk": ("https://www.coindesk.com/arc/outboundfeeds/rss/", "en"),
    "cointelegraph": ("https://cointelegraph.com/rss", "en"),
}

_ATOM = "{http://www.w3.org/2005/Atom}"


def _ssl_context():
    """Certificados do certifi: o repositório do Windows deste Python está desatualizado
    (dava 'certificate has expired' no InfoMoney). A verificação continua ligada."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ai-trader-research)"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
        return resp.read()


def parse_feed(xml: bytes, feed: str, lang: str, now: pd.Timestamp) -> list[dict]:
    root = ET.fromstring(xml)
    rows = []
    for item in root.iter("item"):
        rows.append(_row(item.findtext("title"), item.findtext("link"), item.findtext("pubDate"), feed, lang, now))
    for entry in root.iter(f"{_ATOM}entry"):
        link = entry.find(f"{_ATOM}link")
        rows.append(
            _row(
                entry.findtext(f"{_ATOM}title"),
                link.get("href") if link is not None else "",
                entry.findtext(f"{_ATOM}published") or entry.findtext(f"{_ATOM}updated"),
                feed, lang, now,
            )
        )
    return [r for r in rows if r is not None]


def _row(title, link, date, feed, lang, now) -> dict | None:
    if not title or not link or not date:
        return None
    try:
        published = pd.Timestamp(email.utils.parsedate_to_datetime(date))
    except (TypeError, ValueError):
        published = pd.Timestamp(date)
    published = published.tz_localize("UTC") if published.tzinfo is None else published.tz_convert("UTC")
    return {
        "source": f"rss:{feed}",
        "domain": urllib.parse.urlparse(link).netloc,
        "url": link,
        "original_lang": lang,
        "original_text": " ".join(title.split()),
        "published_at": min(published, now),  # relógio da fonte adiantado não vira futuro
        "first_seen_at": now,
        "backfilled": False,
        "query_assets": "",
    }


def fetch_all(get=_http_get) -> tuple[list[dict], dict[str, str]]:
    """Captura todos os feeds. Retorna (itens, erros por feed)."""
    rows, errors = [], {}
    now = pd.Timestamp.now(tz="UTC")
    for name, (url, lang) in FEEDS.items():
        try:
            rows += parse_feed(get(url), name, lang, now)
        except Exception as exc:  # um feed fora do ar não derruba os outros
            errors[name] = f"{type(exc).__name__}: {exc}"
    return rows, errors
