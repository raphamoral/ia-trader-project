"""Monitor de incidentes de segurança (hacks, exploits, vazamentos alegados, paradas de exchange).

Regras fixas (ver INTEGRATION.md):
- só fontes que passaram na verificação de termos (`SOURCES[..].allowed`); o portão
  recusa as outras antes de qualquer requisição;
- sem Tor, sem login, sem fórum fechado, sem marketplace; nunca abre nem baixa
  dado vazado: guardamos só título, trecho (≤ SNIPPET_MAX), link e carimbos;
- sem contornar bloqueio ou limite de taxa (PoliteClient, 1 req/s por host).

Carimbos:
- published_at vem da fonte; `ts_precision` diz se tem hora ("second") ou só data ("day").
  Data sem hora fica visível só no FIM do dia (+ latência): não dá para saber a hora.
- itens coletados depois do fato são `backfilled` e usam published_at; ao vivo, a coleta.
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import os
import re
import sqlite3
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from trader.blacknews.schema import utc64
from trader.blacknews.sources import PoliteClient, strip_html

SNIPPET_MAX = 300
LATENCY = pd.Timedelta(minutes=5)
SEC_COLUMNS = [
    "id", "source_id", "source", "source_kind", "url", "title", "snippet",
    "published_at", "ts_precision", "first_seen_at", "backfilled", "visible_at",
    # extração
    "entity", "entity_type", "incident_type", "claimed_usd", "assets", "confidence", "extracted_by",
    # deduplicação
    "incident_id",
]
INCIDENT_TYPES = ("hack", "exploit", "breach_claim", "rug_pull", "outage", "withdrawal_halt", "not_incident")
ENTITY_TYPES = ("cex", "defi", "bridge", "wallet", "chain", "company", "other")


class SourceNotAllowed(Exception):
    """Fonte reprovada na verificação de termos: o portão não deixa nem fazer a requisição."""


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    url: str
    kind: str  # incident_db | security_firm | official_status | onion_index
    allowed: bool
    reason: str  # resultado da verificação de termos/robots (2026-09-29)
    checked: str = "2026-09-29"


SOURCES: dict[str, SourcePolicy] = {
    p.source_id: p
    for p in [
        SourcePolicy("defillama_hacks", "https://api.llama.fi/hacks", "incident_db", True,
                     "API pública oficial; termos: uso pessoal não comercial, proibido republicar os dados"),
        SourcePolicy("rekt_news", "https://rekt.news/rss/feed.xml", "security_firm", True,
                     "RSS público; robots só bloqueia /api"),
        SourcePolicy("slowmist", "https://slowmist.medium.com/feed", "security_firm", True, "RSS público do Medium"),
        SourcePolicy("peckshield", "https://medium.com/feed/@peckshield", "security_firm", True, "RSS público do Medium"),
        SourcePolicy("certik", "https://medium.com/feed/@certik", "security_firm", True, "RSS público do Medium"),
        SourcePolicy("status_coinbase", "https://status.coinbase.com/history.rss", "official_status", True,
                     "RSS público da página de status; robots só bloqueia /api e /embed"),
        SourcePolicy("status_kraken", "https://status.kraken.com/history.rss", "official_status", True,
                     "RSS público da página de status; robots só bloqueia /api e /embed"),
        SourcePolicy("status_gemini", "https://status.gemini.com/history.rss", "official_status", True,
                     "RSS público da página de status; robots só bloqueia /api e /embed"),
        SourcePolicy("ahmia", "https://ahmia.fi/search/", "onion_index", False,
                     "robots.txt: 'Disallow: /search/' e '/*?'; sem API documentada; busca automatizada não autorizada"),
        SourcePolicy("certik_api", "https://www.certik.com/api/", "security_firm", False, "robots.txt: 'Disallow: /api/'"),
        SourcePolicy("status_bitfinex", "https://status.bitfinex.com/history.rss", "official_status", False,
                     "sem resposta na verificação"),
    ]
}
# fontes com nota inicial média na reputação (tier None do trader.news: Beta(1; 1,5))
REPUTATION_TIER = None


def gate(source_id: str) -> SourcePolicy:
    policy = SOURCES.get(source_id)
    if policy is None or not policy.allowed:
        raise SourceNotAllowed(f"{source_id}: {policy.reason if policy else 'fonte não verificada'}")
    return policy


def fetch_allowed(client: PoliteClient, source_id: str, url: str | None = None) -> bytes | None:
    """Única porta de rede do monitor: passa pelo portão de termos."""
    policy = gate(source_id)
    target = url or policy.url
    if urllib.parse.urlparse(target).netloc != urllib.parse.urlparse(policy.url).netloc:
        raise SourceNotAllowed(f"{source_id}: host fora da política ({target})")
    return client.get(target)


def snippet(text: str | None) -> str:
    return strip_html(text)[:SNIPPET_MAX]


def _id(source_id: str, key: str) -> str:
    return f"{source_id}:{hashlib.sha1(key.encode()).hexdigest()[:16]}"


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Normaliza e aplica a regra do conteúdo mínimo (nada além de título/trecho/link/carimbos)."""
    allowed_input = {"id", "source_id", "source", "source_kind", "url", "title", "snippet", "published_at",
                     "ts_precision", "first_seen_at", "entity", "entity_type", "incident_type", "claimed_usd",
                     "assets", "confidence", "extracted_by"}
    for r in rows:
        extra = set(r) - allowed_input
        if extra:
            raise ValueError(f"campos não permitidos (só título/trecho/link/carimbos): {sorted(extra)}")
        r["title"] = strip_html(r.get("title"))[:SNIPPET_MAX]
        r["snippet"] = snippet(r.get("snippet"))
    df = pd.DataFrame(rows)
    for col in SEC_COLUMNS:
        if col not in df:
            df[col] = None
    for col in ("published_at", "first_seen_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["claimed_usd"] = pd.to_numeric(df["claimed_usd"])
    df["confidence"] = pd.to_numeric(df["confidence"])
    df["backfilled"] = (df["first_seen_at"] - df["published_at"]) > pd.Timedelta(minutes=30)
    df["visible_at"] = visible_at(df)
    return df[SEC_COLUMNS]


def visible_at(df: pd.DataFrame) -> pd.Series:
    day_shift = pd.to_timedelta((df["ts_precision"] == "day").astype(int), unit="D")
    known = pd.to_datetime(df["published_at"], utc=True) + day_shift + LATENCY
    live = known.where(known >= df["first_seen_at"], df["first_seen_at"])
    return known.where(df["backfilled"], live)


# conectores


def fetch_defillama(client: PoliteClient, now: pd.Timestamp) -> pd.DataFrame:
    body = fetch_allowed(client, "defillama_hacks")
    rows = []
    for h in json.loads(body) if body else []:
        chains = ",".join(h.get("chain") or [])
        technique = h.get("technique") or h.get("classification") or ""
        rows.append({
            "id": _id("defillama_hacks", f"{h.get('name')}|{h.get('date')}"),
            "source_id": "defillama_hacks", "source": "defillama.com", "source_kind": "incident_db",
            "url": "https://defillama.com/hacks", "title": f"{h.get('name')} — {technique}",
            "snippet": f"{h.get('classification') or ''}; {technique}; chains: {chains}; target: {h.get('targetType') or ''}",
            "published_at": pd.Timestamp(h["date"], unit="s", tz="UTC"), "ts_precision": "day",
            "first_seen_at": now, "entity": h.get("name"), "entity_type": _entity_type(h.get("targetType"), h.get("bridgeHack")),
            "incident_type": "hack", "claimed_usd": h.get("amount"), "assets": "BTC",
            "confidence": 0.9, "extracted_by": "structured",
        })
    return make_frame(rows)


def _entity_type(target: str | None, bridge: bool | None) -> str:
    if bridge:
        return "bridge"
    return {"CEX": "cex", "DeFi Protocol": "defi", "Wallet": "wallet", "Chain": "chain", "Layer 1": "chain",
            "Liquidity Pool": "defi"}.get(target or "", "other")


_STATUS_TIME = re.compile(r"<small>(\w{3})\s*<var[^>]*>(\d+)</var>,\s*<var[^>]*>(\d{2}:\d{2})</var>\s*UTC</small>")


def _first_update(desc: str, last: pd.Timestamp) -> pd.Timestamp:
    """Status pages: o pubDate é a ÚLTIMA atualização; a primeira fica no corpo."""
    stamps = []
    for mon, day, hhmm in _STATUS_TIME.findall(desc or ""):
        for year in (last.year, last.year - 1):
            try:
                ts = pd.Timestamp(f"{day} {mon} {year} {hhmm}", tz="UTC")
            except ValueError:
                continue
            if ts <= last:
                stamps.append(ts)
                break
    return min(stamps) if stamps else last


def fetch_rss(client: PoliteClient, source_id: str, now: pd.Timestamp) -> pd.DataFrame:
    policy = gate(source_id)
    body = fetch_allowed(client, source_id)
    if not body:
        return make_frame([])
    rows = []
    for e in ET.fromstring(body).findall(".//item"):
        title = e.findtext("title") or ""
        link = e.findtext("link") or e.findtext("guid") or ""
        stamp = e.findtext("pubDate")
        if not stamp:
            continue
        last = pd.Timestamp(email.utils.parsedate_to_datetime(stamp)).tz_convert("UTC")
        desc = e.findtext("description") or ""
        published = _first_update(desc, last) if policy.kind == "official_status" else last
        precision = "day" if published == published.normalize() else "second"
        rows.append({
            "id": _id(source_id, (e.findtext("guid") or link or title).strip()),
            "source_id": source_id, "source": urllib.parse.urlparse(policy.url).netloc + ("/" + source_id if "medium.com" in policy.url else ""),
            "source_kind": policy.kind, "url": link, "title": title, "snippet": desc,
            "published_at": published, "ts_precision": precision, "first_seen_at": now,
        })
    return make_frame(rows)


def fetch_all(client: PoliteClient, now: pd.Timestamp) -> dict[str, pd.DataFrame | str]:
    out: dict[str, pd.DataFrame | str] = {}
    for sid, policy in SOURCES.items():
        if not policy.allowed:
            out[sid] = f"recusada pelo portão de termos: {policy.reason}"
            continue
        try:
            out[sid] = fetch_defillama(client, now) if sid == "defillama_hacks" else fetch_rss(client, sid, now)
        except Exception as exc:  # uma fonte fora do ar não derruba as outras
            out[sid] = f"erro {type(exc).__name__}: {exc}"
    return out


# armazenamento


class SecurityStore:
    def __init__(self, root: Path | str = Path("data") / "news_black") -> None:
        self.path = Path(root) / "security" / "items.parquet"

    def write(self, items: pd.DataFrame) -> int:
        if items.empty:
            return 0
        if (items["snippet"].str.len() > SNIPPET_MAX).any() or (items["title"].str.len() > SNIPPET_MAX).any():
            raise ValueError("conteúdo além do trecho permitido")
        if set(items.columns) != set(SEC_COLUMNS):
            raise ValueError("esquema diferente do permitido")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        before = 0
        if self.path.exists():
            old = pd.read_parquet(self.path)
            before = len(old)
            # mantém a primeira coleta (é ela que define a visibilidade ao vivo) e a extração já feita
            items = pd.concat([old, items])
        items = items.sort_values("first_seen_at").drop_duplicates("id", keep="first").reset_index(drop=True)
        tmp = self.path.with_suffix(".tmp")
        items.to_parquet(tmp, index=False)
        os.replace(tmp, self.path)
        return len(items) - before

    def replace(self, items: pd.DataFrame) -> None:
        self.path.unlink(missing_ok=True)
        self.write(items)

    def all(self) -> pd.DataFrame:
        return pd.read_parquet(self.path) if self.path.exists() else make_frame([])


# extração com a LLM local


EXTRACT_PROMPT = """You extract security incidents from short crypto/finance security alerts.
For EACH numbered item return: i, is_incident (true only for an actual hack, exploit, breach claim,
rug pull, outage or withdrawal halt affecting a specific entity; false for opinion, research, marketing,
routine maintenance), entity (the affected exchange/protocol/company, short name), entity_type (one of
{entity_types}), incident_type (one of {incident_types}), claimed_usd (number or null), confidence 0..1.

Items:
{items}

JSON only: {{"items": [{{"i": 0, "is_incident": true, "entity": "...", "entity_type": "...",
"incident_type": "...", "claimed_usd": null, "confidence": 0.0}}]}}"""

_AMOUNT = re.compile(r"\$\s?([\d,.]+)\s*(k|m|mm|million|b|bn|billion)?\b", re.IGNORECASE)
_MULT = {"k": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_INCIDENT_WORDS = {
    "exploit": re.compile(r"exploit|flash ?loan|reentran|price manipulation", re.I),
    "hack": re.compile(r"hack|drain|stolen|compromis|attacker", re.I),
    "breach_claim": re.compile(r"breach|leak|data exposed", re.I),
    "rug_pull": re.compile(r"rug ?pull|exit scam", re.I),
    "withdrawal_halt": re.compile(r"withdrawals? (paused|halted|suspended|delayed)|suspend(ed)? withdrawals", re.I),
    "outage": re.compile(r"outage|degraded|downtime|unavailable|delays? with", re.I),
}


def parse_amount(text: str) -> float | None:
    best = None
    for num, unit in _AMOUNT.findall(text or ""):
        try:
            value = float(num.replace(",", "")) * _MULT.get((unit or "").lower(), 1)
        except ValueError:
            continue
        best = max(best or 0, value)
    return best


def rule_extract(title: str, snip: str, kind: str) -> dict[str, Any]:
    text = f"{title} {snip}"
    itype = next((k for k, rx in _INCIDENT_WORDS.items() if rx.search(text)), "not_incident")
    entity = None
    if kind == "official_status":
        entity = None  # preenchido pelo nome da exchange do feed
    elif itype != "not_incident":
        m = re.match(r"\s*([A-Z][\w.\- ]{1,30}?)(?:\s+(?:hack|exploit|drain|suffers|loses|hit|attack|incident)|\s*[:—-])", title)
        entity = m.group(1).strip() if m else None
    return {"entity": entity, "entity_type": "other", "incident_type": itype, "claimed_usd": parse_amount(text),
            "confidence": 0.3 if itype != "not_incident" else 0.2, "extracted_by": "rules"}


class ExtractCache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def get(self, key: str):
        row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()


def _validate(raw: dict) -> dict[str, Any]:
    itype = raw.get("incident_type") if raw.get("incident_type") in INCIDENT_TYPES else "not_incident"
    if raw.get("is_incident") is False:
        itype = "not_incident"
    amount = raw.get("claimed_usd")
    try:
        amount = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        amount = None
    entity = raw.get("entity")
    if isinstance(entity, str) and entity.strip().lower() in {"n/a", "na", "none", "null", "unknown", "-"}:
        entity = None
    return {
        "entity": entity.strip()[:60] if isinstance(entity, str) and entity.strip() else None,
        "entity_type": raw.get("entity_type") if raw.get("entity_type") in ENTITY_TYPES else "other",
        "incident_type": itype,
        "claimed_usd": amount if amount and amount > 0 else None,
        "confidence": max(0.0, min(1.0, float(raw.get("confidence") or 0.0))),
        "extracted_by": "llm",
    }


class Extractor:
    def __init__(self, cache: ExtractCache, post: Callable | None = None, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5:7b", batch_size: int = 10, max_calls: int = 20, timeout: float = 300.0):
        if post is None:
            from trader.evolution.creator import _post_json as post
        self.cache, self.post, self.model = cache, post, model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.batch_size, self.max_calls, self.timeout = batch_size, max_calls, timeout
        self.calls = self.failures = 0

    def key(self, title: str, snip: str) -> str:
        return hashlib.sha256(f"sec-v1|{self.model}|{title}|{snip}".encode()).hexdigest()

    def extract(self, items: pd.DataFrame) -> pd.DataFrame:
        df = items.copy()
        todo = df.index[df["extracted_by"].isna() | (df["extracted_by"] == "rules")].tolist()
        pending = []
        for idx in todo:
            hit = self.cache.get(self.key(df.at[idx, "title"], df.at[idx, "snippet"]))
            if hit:
                self._apply(df, idx, hit)
            else:
                pending.append(idx)
        for s in range(0, len(pending), self.batch_size):
            batch = pending[s : s + self.batch_size]
            if self.calls >= self.max_calls:
                break
            text = "\n".join(f"[{k}] {df.at[i, 'title']} — {df.at[i, 'snippet'][:250]}" for k, i in enumerate(batch))
            try:
                self.calls += 1
                reply = self.post(self.url, {
                    "model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": EXTRACT_PROMPT.format(
                        entity_types=", ".join(ENTITY_TYPES), incident_types=", ".join(INCIDENT_TYPES), items=text)}],
                }, self.timeout)
                content = reply["choices"][0]["message"]["content"]
                answers = json.loads(content[content.find("{") : content.rfind("}") + 1])["items"]
            except Exception:
                self.failures += 1
                continue
            for ans in answers:
                k = int(ans.get("i", -1))
                if 0 <= k < len(batch):
                    value = _validate(ans)
                    self.cache.put(self.key(df.at[batch[k], "title"], df.at[batch[k], "snippet"]), value)
                    self._apply(df, batch[k], value)
        for idx in df.index[df["extracted_by"].isna()]:
            self._apply(df, idx, rule_extract(df.at[idx, "title"], df.at[idx, "snippet"], df.at[idx, "source_kind"]))
        # página de status: a entidade é a própria exchange
        status = df["source_kind"] == "official_status"
        df.loc[status & df["entity"].isna(), "entity"] = df.loc[status & df["entity"].isna(), "source_id"].str.removeprefix("status_").str.title()
        df.loc[status, "entity_type"] = "cex"
        df["assets"] = df["assets"].fillna("BTC")
        return df

    @staticmethod
    def _apply(df: pd.DataFrame, idx, value: dict) -> None:
        for col in ("entity", "entity_type", "incident_type", "claimed_usd", "confidence", "extracted_by"):
            df.at[idx, col] = value.get(col)


# deduplicação entre fontes


_SUFFIX = re.compile(r"\b(exchange|protocol|finance|network|labs|dao|v\d+|io|com|global)\b")


def entity_key(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    s = _SUFFIX.sub(" ", name.lower())
    return re.sub(r"[^a-z0-9]", "", s)


def dedup_incidents(items: pd.DataFrame, window: pd.Timedelta = pd.Timedelta(days=3)) -> pd.DataFrame:
    """incident_id: mesma entidade dentro da janela = mesmo incidente, em qualquer fonte.

    Online e em ordem de visibilidade (causal): um item só pode entrar num incidente aberto antes dele.
    Itens que não são incidente ficam sem id.
    """
    df = items.sort_values(["visible_at", "id"]).reset_index(drop=True).copy()
    df["incident_id"] = None
    open_: dict[str, tuple[pd.Timestamp, str]] = {}
    for i, r in df.iterrows():
        key = entity_key(r["entity"])
        if r["incident_type"] in (None, "not_incident") or not key:
            continue
        prev = open_.get(key)
        if prev is not None and r["visible_at"] - prev[0] <= window:
            iid = prev[1]
        else:
            iid = f"inc:{key}:{r['visible_at']:%Y%m%d%H%M}"
        open_[key] = (r["visible_at"], iid)
        df.at[i, "incident_id"] = iid
    return df


def incidents_table(items: pd.DataFrame) -> pd.DataFrame:
    inc = items.dropna(subset=["incident_id"])
    if inc.empty:
        return pd.DataFrame(columns=["incident_id", "entity", "first_visible_at", "first_source", "sources", "max_claimed_usd"])
    first = inc.sort_values("visible_at").groupby("incident_id").first()
    return pd.DataFrame({
        "incident_id": first.index,
        "entity": first["entity"].to_numpy(),
        "first_visible_at": first["visible_at"].to_numpy(),
        "first_source": first["source"].to_numpy(),
        "sources": inc.groupby("incident_id")["source"].nunique().reindex(first.index).to_numpy(),
        "max_claimed_usd": inc.groupby("incident_id")["claimed_usd"].max().reindex(first.index).to_numpy(),
    }).sort_values("first_visible_at").reset_index(drop=True)


# confirmação e lead/lag contra as notícias confiáveis


def confirmations(items: pd.DataFrame, trusted: pd.DataFrame, window: pd.Timedelta = pd.Timedelta(hours=72)) -> pd.DataFrame:
    """Por item de incidente: primeira confirmação de nível 1 (notícia) ou oficial (status da própria
    exchange), com a entidade citada no texto confiável dentro de ±window."""
    inc = items.dropna(subset=["incident_id"])
    rows = []
    tr = trusted.assign(_low=trusted["text_en"].str.lower()) if len(trusted) else trusted
    official = items[items["source_kind"] == "official_status"]
    for _, r in inc.iterrows():
        key = entity_key(r["entity"])
        name = str(r["entity"]).lower().strip()
        if len(key) < 3:
            continue
        lo, hi = r["visible_at"] - window, r["visible_at"] + window
        hits = tr[(tr["visible_at"] >= lo) & (tr["visible_at"] <= hi) & tr["_low"].str.contains(re.escape(name), regex=True)] if len(tr) else tr
        first_trusted = hits["visible_at"].min() if len(hits) else pd.NaT
        first_tier1 = hits.loc[hits["tier"] == 1, "visible_at"].min() if len(hits) else pd.NaT
        off = official[(official["entity"].map(entity_key) == key) & (official["visible_at"] >= lo) & (official["visible_at"] <= hi)
                       & (official["id"] != r["id"])]
        first_official = off["visible_at"].min() if len(off) else pd.NaT
        confirm = min([t for t in (first_tier1, first_official) if pd.notna(t)], default=pd.NaT)
        rows.append({"id": r["id"], "first_trusted_at": first_trusted, "first_tier1_at": first_tier1,
                     "first_official_at": first_official, "confirmed_at": confirm})
    cols = ["id", "first_trusted_at", "first_tier1_at", "first_official_at", "confirmed_at"]
    out = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        out[c] = pd.to_datetime(out[c], utc=True)
    return out


def incident_lags(
    items: pd.DataFrame, conf: pd.DataFrame, data_end: pd.Timestamp, horizon: pd.Timedelta = pd.Timedelta(hours=48),
    covered: np.ndarray | None = None,
) -> pd.DataFrame:
    """Mesmo formato de leadlag.item_lags, para reaproveitar reputação e tabela por fonte.

    covered: máscara (na ordem dos itens de incidente) de leadlag.trusted_covered. Fora da cobertura
    não dá para saber quem chegou primeiro: status "no_coverage", sem crédito nem falha."""
    inc = items.dropna(subset=["incident_id"]).merge(conf, on="id", how="left")
    ref = inc["first_trusted_at"]
    lag = (inc["visible_at"] - ref).dt.total_seconds() / 60
    in_horizon = ref.notna() & (ref <= inc["visible_at"] + horizon)
    status = np.where(~in_horizon, "unconfirmed", np.where(lag < 0, "lead", "echo"))
    pending = ~in_horizon & (inc["visible_at"] + horizon > data_end)
    status = np.where(pending, "pending", status)
    if covered is not None:
        status = np.where(covered | (status == "echo"), status, "no_coverage")  # eco já é fato: a confiável veio antes
    return pd.DataFrame({
        "id": inc["id"], "source": inc["source"], "source_kind": inc["source_kind"], "visible_at": inc["visible_at"],
        "assets": inc["assets"], "stance": "bearish", "is_claim": True,
        "first_trusted_at": ref.where(in_horizon), "first_tier1_at": inc["first_tier1_at"].where(in_horizon),
        "story_id": inc["incident_id"], "lag_min": lag.where(in_horizon),
        "status": status,
    })


# features causais


SEC_FEATURES = ["black_sec_incident_count", "black_sec_max_claimed_usd", "black_sec_minutes_since_incident",
                "black_sec_confirmed_flag"]
SEC_RW_FEATURES = ["black_rw_sec_incident_score"]
NO_INCIDENT_MINUTES = 7 * 24 * 60.0
# paradas rotineiras ("depósitos atrasados") ficam guardadas, mas não contam como incidente nas features
FEATURE_TYPES = ("hack", "exploit", "breach_claim", "rug_pull", "withdrawal_halt")


B3_ENTITIES = {"petrobras": "PETR4", "vale": "VALE3", "itau": "ITUB4", "itaú": "ITUB4", "bradesco": "BBDC4",
               "banco do brasil": "BBAS3", "ambev": "ABEV3", "weg": "WEGE3", "b3": "B3SA3"}


def asset_of(entity: str | None, default: str = "BTC") -> str:
    low = (entity or "").lower()
    for name, ticker in B3_ENTITIES.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            return ticker
    return default


def security_features(
    items: pd.DataFrame, conf: pd.DataFrame | None, index: pd.DatetimeIndex, asset: str,
    lookback: pd.Timedelta = pd.Timedelta(hours=24), weights: Callable[[pd.Timestamp, list[str]], pd.Series] | None = None,
) -> pd.DataFrame:
    """black_sec_* por instante: só incidentes com visible_at <= t; confirmação só se já visível em t."""
    inc = items.dropna(subset=["incident_id"]).copy()
    inc = inc[inc["incident_type"].isin(FEATURE_TYPES) & (inc["entity"].map(asset_of) == asset)]
    if conf is not None and len(conf):
        inc = inc.merge(conf[["id", "confirmed_at"]], on="id", how="left")
    else:
        inc["confirmed_at"] = pd.NaT
    inc = inc.sort_values("visible_at").reset_index(drop=True)
    vis = utc64(inc["visible_at"])
    conf_at = utc64(inc["confirmed_at"])
    amount = inc["claimed_usd"].astype(float).fillna(0).to_numpy()
    iid = inc["incident_id"].to_numpy()
    lb = np.timedelta64(int(lookback.total_seconds()), "s")
    rows = []
    day_w: dict = {}
    for t in pd.DatetimeIndex(index).tz_convert("UTC"):
        t64 = t.tz_localize(None).to_datetime64()
        hi = np.searchsorted(vis, t64, "right")
        lo = np.searchsorted(vis, t64 - lb, "right")
        if hi == 0:
            rows.append((0.0, 0.0, NO_INCIDENT_MINUTES, 0.0, 0.0))
            continue
        since = (t64 - vis[hi - 1]) / np.timedelta64(1, "m")
        window_ids = iid[lo:hi]
        confirmed = bool(((conf_at[lo:hi] <= t64) & ~np.isnat(conf_at[lo:hi])).any()) if hi > lo else False
        score = 0.0
        if weights is not None and hi > lo:
            day = t.floor("D")
            srcs = inc["source"].iloc[lo:hi].tolist()
            if day not in day_w:
                day_w[day] = {}
            missing = [s for s in set(srcs) if s not in day_w[day]]
            if missing:
                day_w[day].update(weights(day, missing).to_dict())
            score = float(sum(day_w[day].get(s, 0.0) for s in srcs))
        rows.append((
            float(len(set(window_ids))),
            float(amount[lo:hi].max()) if hi > lo else 0.0,
            float(min(since, NO_INCIDENT_MINUTES)),
            float(confirmed),
            score,
        ))
    return pd.DataFrame(rows, index=index, columns=[*SEC_FEATURES, *SEC_RW_FEATURES])
