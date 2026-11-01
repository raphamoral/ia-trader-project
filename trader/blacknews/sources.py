"""Conectores das fontes "black". Só acesso gratuito, sem login, dentro dos termos.

Situação verificada em 2026-09-29 (detalhes e links em INTEGRATION.md):
- 4chan /biz/: API JSON oficial só leitura. Regras: no máximo 1 req/s,
  If-Modified-Since, atribuição com link, proibido uso comercial.
- Medium (RSS por tag) e Substack (RSS por publicação): feeds públicos.
- Sites de conteúdo em massa: RSS públicos dos próprios sites.
- Reddit: API exige cliente OAuth aprovado manualmente (desde 2025), uso
  não comercial, 100 req/min. Conector pronto, desligado sem credenciais.
- Fora: Bitcointalk (feed .xml desativado pelo admin), StockTwits (sem API
  pública, bloqueio Cloudflare), X/Nitter (termos do X proíbem raspagem),
  Telegram t.me/s (termos não autorizam explicitamente; decisão do usuário).
"""
from __future__ import annotations

import email.utils
import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Protocol

import pandas as pd

from trader.blacknews.schema import content_hash, new_frame, with_visibility

USER_AGENT = "AI-trader-project/blacknews (personal research; non-commercial)"
_TAG = re.compile(r"<[^>]+>")


def strip_html(raw: str | None) -> str:
    text = _TAG.sub(" ", (raw or "").replace("<br>", "\n"))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class SourceDisabled(Exception):
    """Fonte indisponível por termos, credenciais ou bloqueio."""


Fetch = Callable[[str, dict[str, str]], tuple[int, bytes, dict[str, str]]]


def _ssl_context():
    """Cadeia de certificados do certifi: o repositório do Windows deste Python está desatualizado
    (dava 'certificate has expired' em sites válidos). A verificação continua ligada."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def _urllib_fetch(url: str, headers: dict[str, str]) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60, context=_SSL) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, b"", dict(exc.headers or {})


class PoliteClient:
    """Intervalo mínimo por host, User-Agent honesto e If-Modified-Since."""

    def __init__(self, fetch: Fetch = _urllib_fetch, sleep: Callable[[float], None] = time.sleep) -> None:
        self.fetch = fetch
        self.sleep = sleep
        self.min_interval: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._modified: dict[str, str] = {}

    def get(self, url: str, extra_headers: dict[str, str] | None = None) -> bytes | None:
        """Corpo da resposta, ou None se não mudou (304) ou falhou."""
        host = urllib.parse.urlparse(url).netloc
        wait = self.min_interval.get(host, 1.0) - (time.monotonic() - self._last.get(host, -1e9))
        if wait > 0:
            self.sleep(wait)
        headers = {"User-Agent": USER_AGENT, **(extra_headers or {})}
        if url in self._modified:
            headers["If-Modified-Since"] = self._modified[url]
        status, body, resp_headers = self.fetch(url, headers)
        self._last[host] = time.monotonic()
        if status == 200:
            if "Last-Modified" in resp_headers:
                self._modified[url] = resp_headers["Last-Modified"]
            return body
        return None


class Source(Protocol):
    source_id: str
    source_kind: str

    def fetch(self, client: PoliteClient, now: pd.Timestamp) -> pd.DataFrame: ...


def mark_backfilled(df: pd.DataFrame, max_age: pd.Timedelta = pd.Timedelta(minutes=30)) -> pd.DataFrame:
    """Item com mais de `max_age` entre publicação e coleta foi pego depois do fato:
    no backtest vale published_at + latência; ao vivo, só a partir da coleta."""
    df = df.copy()
    df["backfilled"] = (df["first_seen_at"] - df["published_at"]) > max_age
    return with_visibility(df)


@dataclass
class FourChanBiz:
    """Catálogo ao vivo + threads arquivadas (profundidade de poucos dias)."""

    archive_limit: int = 200
    source_id: str = "4chan_biz"
    source_kind: str = "social"
    base: str = "https://a.4cdn.org/biz"

    def _row(self, post: dict, now: pd.Timestamp) -> dict:
        text = strip_html(" ".join(filter(None, [post.get("sub"), post.get("com")])))
        return {
            "id": f"{self.source_id}:{post['no']}",
            "source_id": self.source_id,
            "source": "4chan/biz",  # anônimo: o emissor é o quadro inteiro
            "source_kind": self.source_kind,
            "url": f"https://boards.4chan.org/biz/thread/{post['no']}",
            "title": strip_html(post.get("sub")),
            "text": text,
            "published_at": pd.Timestamp(post["time"], unit="s", tz="UTC"),
            "first_seen_at": now,
            "ts_reliability": "server",
            "engagement": int(post.get("replies", 0)),
            "content_hash": content_hash(text),
        }

    def fetch(self, client: PoliteClient, now: pd.Timestamp) -> pd.DataFrame:
        client.min_interval["a.4cdn.org"] = 1.1  # regra da API: <= 1 req/s
        rows = []
        body = client.get(f"{self.base}/catalog.json")
        if body:
            for page in json.loads(body):
                rows += [self._row(t, now) for t in page.get("threads", []) if not t.get("sticky")]
        body = client.get(f"{self.base}/archive.json")
        archived = json.loads(body)[-self.archive_limit :] if body and self.archive_limit else []
        for no in archived:
            thread = client.get(f"{self.base}/thread/{no}.json")
            if thread:
                posts = json.loads(thread).get("posts", [])
                if posts:
                    op = dict(posts[0], replies=len(posts) - 1)
                    rows.append(self._row(op, now))
        rows = [r for r in rows if r["text"]]
        return mark_backfilled(new_frame(rows))


@dataclass
class RSSFeed:
    """RSS 2.0/Atom. `per_author` usa o autor como emissor (Medium/Substack)."""

    source_id: str
    url: str
    source_kind: str
    per_author: bool = False
    max_chars: int = 1500

    def fetch(self, client: PoliteClient, now: pd.Timestamp) -> pd.DataFrame:
        body = client.get(self.url)
        if not body:
            return new_frame([])
        return mark_backfilled(new_frame(parse_feed(body, self, now)))


_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def parse_feed(body: bytes, feed: RSSFeed, now: pd.Timestamp) -> list[dict]:
    root = ET.fromstring(body)
    domain = urllib.parse.urlparse(feed.url).netloc.removeprefix("www.")
    rows = []
    entries = root.findall(".//item") or root.findall(".//atom:entry", _NS)
    for e in entries:
        title = strip_html(e.findtext("title") or e.findtext("atom:title", namespaces=_NS))
        link = e.findtext("link") or (e.find("atom:link", _NS).get("href") if e.find("atom:link", _NS) is not None else "")
        desc = e.findtext("description") or e.findtext("atom:summary", namespaces=_NS) or ""
        if not desc:
            desc = e.findtext("content:encoded", namespaces=_NS) or ""
        stamp = e.findtext("pubDate") or e.findtext("atom:published", namespaces=_NS) or e.findtext("atom:updated", namespaces=_NS)
        if not stamp:
            continue
        try:
            published = pd.Timestamp(email.utils.parsedate_to_datetime(stamp)).tz_convert("UTC")
        except (TypeError, ValueError):
            published = pd.Timestamp(stamp).tz_convert("UTC")
        author = e.findtext("dc:creator", namespaces=_NS) or e.findtext("author") or ""
        text = f"{title}. {strip_html(desc)}"[: feed.max_chars]
        guid = e.findtext("guid") or link or title
        rows.append(
            {
                "id": f"{feed.source_id}:{hashlib.sha1(guid.strip().encode()).hexdigest()[:16]}",
                "source_id": feed.source_id,
                "source": f"{domain}/{author.strip()}" if feed.per_author and author.strip() else domain,
                "source_kind": feed.source_kind,
                "url": link,
                "title": title,
                "text": text,
                "published_at": published,
                "first_seen_at": now,
                "ts_reliability": "author" if feed.per_author else "server",
                "engagement": 0,
                "content_hash": content_hash(text),
            }
        )
    return rows


@dataclass
class RedditNew:
    """API oficial do Reddit. Exige cliente OAuth aprovado (env REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET). Uso não comercial; 100 req/min; os termos proíbem
    treinar modelos de IA com os dados sem permissão."""

    subreddits: list[str] = field(
        default_factory=lambda: ["Bitcoin", "CryptoCurrency", "wallstreetbets", "investimentos", "farialimabets"]
    )
    source_id: str = "reddit"
    source_kind: str = "social"

    def _token(self, client: PoliteClient) -> str:
        cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
        if not (cid and secret):
            raise SourceDisabled("Reddit: defina REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET (app aprovado pelo Reddit)")
        import base64

        auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
        req = urllib.request.Request(
            "https://www.reddit.com/api/v1/access_token",
            data=b"grant_type=client_credentials",
            headers={"Authorization": f"Basic {auth}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["access_token"]

    def fetch(self, client: PoliteClient, now: pd.Timestamp) -> pd.DataFrame:
        token = self._token(client)
        client.min_interval["oauth.reddit.com"] = 0.7  # < 100 req/min
        rows = []
        for sub in self.subreddits:
            body = client.get(
                f"https://oauth.reddit.com/r/{sub}/new?limit=100", {"Authorization": f"bearer {token}"}
            )
            for child in json.loads(body)["data"]["children"] if body else []:
                d = child["data"]
                text = f"{d.get('title', '')}. {d.get('selftext', '')}"[:1500]
                rows.append(
                    {
                        "id": f"reddit:{d['id']}",
                        "source_id": self.source_id,
                        "source": f"reddit/r/{sub}",
                        "source_kind": self.source_kind,
                        "url": "https://www.reddit.com" + d.get("permalink", ""),
                        "title": d.get("title", ""),
                        "text": text,
                        "published_at": pd.Timestamp(d["created_utc"], unit="s", tz="UTC"),
                        "first_seen_at": now,
                        "ts_reliability": "server",
                        "engagement": int(d.get("num_comments", 0)),
                        "content_hash": content_hash(text),
                    }
                )
        return mark_backfilled(new_frame(rows))


# Fontes padrão. "farm" = alto volume e pouca curadoria editorial; a lista é
# editável e não é juízo sobre a veracidade de cada site.
FARM_FEEDS = {
    "coingape": "https://coingape.com/feed/",
    "zycrypto": "https://zycrypto.com/feed/",
    "bitcoinist": "https://bitcoinist.com/feed/",
    "newsbtc": "https://www.newsbtc.com/feed/",
    "cryptopolitan": "https://www.cryptopolitan.com/feed/",
    "utoday": "https://u.today/rss.php",
    "livecoins_pt": "https://livecoins.com.br/feed/",
    "criptonoticias_es": "https://www.criptonoticias.com/feed/",
}
BLOG_FEEDS = {
    "medium_tag_bitcoin": "https://medium.com/feed/tag/bitcoin",
    "medium_tag_cryptocurrency": "https://medium.com/feed/tag/cryptocurrency",
    "medium_tag_criptomoedas": "https://medium.com/feed/tag/criptomoedas",
}

EXCLUDED = {
    "bitcointalk": "feed action=.xml desativado pelo admin ('write a post in Meta explaining your usage')",
    "stocktwits": "sem API pública; site atrás de desafio Cloudflare",
    "x_nitter": "termos do X proíbem raspagem; espelhos Nitter instáveis",
    "telegram_preview": "t.me/s é público, mas os termos não autorizam coleta explicitamente; decisão do usuário",
}


def default_sources(archive_limit: int = 200) -> list[Source]:
    sources: list[Source] = [FourChanBiz(archive_limit=archive_limit)]
    sources += [RSSFeed(sid, url, "blog", per_author=True) for sid, url in BLOG_FEEDS.items()]
    sources += [RSSFeed(sid, url, "farm") for sid, url in FARM_FEEDS.items()]
    sources.append(RedditNew())
    return sources
