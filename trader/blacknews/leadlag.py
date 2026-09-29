"""Casamento de histórias black ↔ confiáveis e análise de antecedência (lead/lag).

lag = visible_at do item black − primeiro visible_at confiável da mesma história.
Negativo: a fonte black chegou antes (lead). Positivo: chegou depois (eco, que
repete o que já era público e não traz informação).

O casamento usa o texto em inglês (tradução da língua pivô), sobreposição de
ativos e janela de tempo; pares na zona cinzenta podem ser confirmados pela
LLM local (com cache). As análises daqui são relatório offline, olhando o
período inteiro; as features causais estão em features.py.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from trader.blacknews.schema import utc64
from trader.news.stories import tokens

CRYPTO = {"BTC", "ETH", "SOL", "XRP", "CRYPTO"}
_EXTRA_STOP = {"price", "market", "report", "reports", "according", "said", "just", "now", "today"}


def english_text(row) -> str:
    """Tradução se houver, senão o original (NaN/None viram texto vazio)."""
    for col in ("translated_text", "text"):
        value = row.get(col)
        if isinstance(value, str) and value:
            return value
    return ""


def content_words(text: str | None) -> frozenset[str]:
    """Tokens do agrupador de histórias confiável, sem palavras genéricas de mercado."""
    if not isinstance(text, str):
        return frozenset()
    return frozenset(w for w in tokens(text) if len(w) >= 3 and w not in _EXTRA_STOP)


class VerifyCache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()


def assets_overlap(a: str, b: str) -> bool:
    sa, sb = set(filter(None, (a or "").split(","))), set(filter(None, (b or "").split(",")))
    if not sa or not sb:
        return False
    if sa & sb:
        return True
    return ("CRYPTO" in sa and sb & CRYPTO) or ("CRYPTO" in sb and sa & CRYPTO)


VERIFY_PROMPT = """For each numbered pair, answer whether A and B report the SAME specific event
(same entity and same happening; general market commentary does not count).
{pairs}
Answer JSON only: {{"answers": [{{"i": 0, "same": true}}, ...]}}"""


@dataclass
class StoryMatcher:
    window: pd.Timedelta = pd.Timedelta(hours=72)
    min_jaccard: float = 0.12  # abaixo disso nem é candidato
    accept_jaccard: float = 0.30  # sem LLM, aceita a partir daqui
    verify: Callable[[list[tuple[str, str]]], list[bool]] | None = None
    max_verify_pairs: int = 300

    def match(self, black: pd.DataFrame, trusted: pd.DataFrame) -> pd.DataFrame:
        """Pares aceitos: black_id, trusted_id, jaccard, trusted_visible_at, trusted_tier."""
        cols = ["black_id", "trusted_id", "jaccard", "trusted_visible_at", "trusted_tier"]
        if black.empty or trusted.empty:
            return pd.DataFrame(columns=cols)
        tr = trusted.sort_values("visible_at").reset_index(drop=True)
        tr_words = [content_words(t) for t in tr["text_en"]]
        tr_times = utc64(tr["visible_at"])
        candidates = []
        for _, b in black.iterrows():
            words = content_words(english_text(b))
            if not words:
                continue
            lo = np.searchsorted(tr_times, (b["visible_at"] - self.window).to_datetime64(), "left")
            hi = np.searchsorted(tr_times, (b["visible_at"] + self.window).to_datetime64(), "right")
            for j in range(lo, hi):
                if not assets_overlap(b["assets"], tr.at[j, "assets"]):
                    continue
                inter = len(words & tr_words[j])
                jac = inter / len(words | tr_words[j]) if inter else 0.0
                if jac >= self.min_jaccard:
                    candidates.append((b["id"], j, jac, english_text(b)))
        accepted = [c for c in candidates if c[2] >= self.accept_jaccard]
        if self.verify is not None:
            grey = sorted((c for c in candidates if c[2] < self.accept_jaccard), key=lambda c: -c[2])
            grey = grey[: self.max_verify_pairs]
            verdicts = self.verify([(c[3], tr.at[c[1], "text_en"]) for c in grey]) if grey else []
            accepted += [c for c, ok in zip(grey, verdicts) if ok]
        rows = [
            (bid, tr.at[j, "id"], jac, tr.at[j, "visible_at"], int(tr.at[j, "tier"])) for bid, j, jac, _ in accepted
        ]
        return pd.DataFrame(rows, columns=cols)


class LLMVerifier:
    """Confirma pares na zona cinzenta com a LLM local, em lotes e com cache."""

    def __init__(self, cache: VerifyCache, post, url: str, model: str, batch: int = 10, max_calls: int = 10):
        self.cache, self.post, self.url, self.model = cache, post, url.rstrip("/") + "/chat/completions", model
        self.batch, self.max_calls, self.calls = batch, max_calls, 0

    def _key(self, a: str, b: str) -> str:
        return hashlib.sha256(f"verify|{self.model}|{a}|{b}".encode()).hexdigest()

    def __call__(self, pairs: list[tuple[str, str]]) -> list[bool]:
        out = [False] * len(pairs)
        todo = []
        for i, (a, b) in enumerate(pairs):
            hit = self.cache.get(self._key(a, b))
            if hit is not None:
                out[i] = bool(hit["same"])
            else:
                todo.append(i)
        for s in range(0, len(todo), self.batch):
            if self.calls >= self.max_calls:
                break
            chunk = todo[s : s + self.batch]
            text = "\n".join(f"[{k}] A: {pairs[i][0][:300]}\n    B: {pairs[i][1][:300]}" for k, i in enumerate(chunk))
            try:
                reply = self.post(
                    self.url,
                    {"model": self.model, "messages": [{"role": "user", "content": VERIFY_PROMPT.format(pairs=text)}],
                     "temperature": 0, "response_format": {"type": "json_object"}},
                    300.0,
                )
                self.calls += 1
                content = reply["choices"][0]["message"]["content"]
                answers = json.loads(content[content.find("{") : content.rfind("}") + 1])["answers"]
            except Exception:
                continue
            for ans in answers:
                k = int(ans.get("i", -1))
                if 0 <= k < len(chunk):
                    i = chunk[k]
                    out[i] = bool(ans.get("same"))
                    self.cache.put(self._key(*pairs[i]), {"same": out[i]})
        return out


def trusted_covered(times: pd.Series, trusted_times: pd.Series, span: pd.Timedelta = pd.Timedelta(hours=24)) -> np.ndarray:
    """A coleta confiável estava rodando em volta do instante? (≥1 item confiável nas 24 h antes E nas 24 h depois.)
    Sem isso não dá para dizer quem chegou primeiro: a base confiável tem buracos (ex.: GDELT só em alguns meses)."""
    tt = np.sort(utc64(trusted_times))
    v = utc64(times)
    w = np.timedelta64(int(span.total_seconds()), "s")
    before = np.searchsorted(tt, v, "left") - np.searchsorted(tt, v - w, "left")
    after = np.searchsorted(tt, v + w, "right") - np.searchsorted(tt, v, "left")
    return (before > 0) & (after > 0)


def item_lags(
    black: pd.DataFrame, matches: pd.DataFrame, data_end: pd.Timestamp, horizon: pd.Timedelta = pd.Timedelta(hours=48)
) -> pd.DataFrame:
    """Um registro por item black: status lead | echo | unconfirmed | pending, lag em minutos."""
    out = black[["id", "source", "source_kind", "visible_at", "assets", "stance", "is_claim"]].copy()
    if not matches.empty:
        # confirmação só conta dentro do horizonte, igual ao que a reputação saberia na hora
        limit = matches["black_id"].map(out.set_index("id")["visible_at"]) + horizon
        matches = matches[matches["trusted_visible_at"] <= limit]
    if not matches.empty:
        first = matches.groupby("black_id").agg(
            first_trusted_at=("trusted_visible_at", "min"),
            story_id=("trusted_id", "first"),
        )
        tier1 = matches[matches["trusted_tier"] == 1].groupby("black_id")["trusted_visible_at"].min()
        out = out.join(first, on="id").join(tier1.rename("first_tier1_at"), on="id")
    else:
        nat = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        out["first_trusted_at"] = nat
        out["first_tier1_at"] = nat
        out["story_id"] = None
    out["lag_min"] = (out["visible_at"] - out["first_trusted_at"]).dt.total_seconds() / 60
    status = np.where(out["first_trusted_at"].isna(), "unconfirmed", np.where(out["lag_min"] < 0, "lead", "echo"))
    pending = out["first_trusted_at"].isna() & (out["visible_at"] + horizon > data_end)
    out["status"] = np.where(pending, "pending", status)
    return out


def edge_windows(
    lags: pd.DataFrame, prices: Callable[[str], pd.DataFrame | None], fee_round_trip: Callable[[str], float] | float
) -> pd.DataFrame:
    """Movimento de preço entre o post black e a primeira notícia confiável (só leads).

    Entrada na abertura do primeiro candle depois do post; saída na abertura do
    primeiro candle depois da notícia confiável. signed = movimento na direção
    do post (alta +, baixa −); net = signed − taxa ida e volta.
    """
    rows = []
    for _, r in lags[lags["status"] == "lead"].iterrows():
        for asset in filter(None, (r["assets"] or "").split(",")):
            px = prices(asset)
            if px is None or px.empty:
                continue
            ts = utc64(px["ts"])
            i = np.searchsorted(ts, r["visible_at"].to_datetime64(), "left")
            j = np.searchsorted(ts, r["first_trusted_at"].to_datetime64(), "left")
            if i >= len(px) or j >= len(px) or j <= i:
                continue
            move = px["open"].iloc[j] / px["open"].iloc[i] - 1
            sign = {"bullish": 1, "bearish": -1}.get(r["stance"], 0)
            fee = fee_round_trip(asset) if callable(fee_round_trip) else fee_round_trip
            rows.append(
                {"id": r["id"], "source": r["source"], "asset": asset, "lead_min": -r["lag_min"], "move": move,
                 "signed": sign * move if sign else np.nan,
                 "net": sign * move - fee if sign else np.nan}
            )
            break  # um ativo por item
    return pd.DataFrame(rows, columns=["id", "source", "asset", "lead_min", "move", "signed", "net"])


def source_table(lags: pd.DataFrame, reputation: pd.DataFrame | None = None, edges: pd.DataFrame | None = None) -> pd.DataFrame:
    """Por fonte: leads, ecos, sem confirmação, lag mediano, % primeiro, sinal de eco, janela de preço."""
    rows = []
    for source, g in lags.groupby("source"):
        resolved = g[~g["status"].isin(["pending", "no_coverage"])]
        conf = resolved[resolved["status"].isin(["lead", "echo"])]
        leads = (resolved["status"] == "lead").sum()
        echoes = (resolved["status"] == "echo").sum()
        row = {
            "source": source,
            "kind": g["source_kind"].iloc[0],
            "items": len(g),
            "confirmed": len(conf),
            "leads": int(leads),
            "echoes": int(echoes),
            "unconfirmed": int((resolved["status"] == "unconfirmed").sum()),
            "median_lag_min": float(conf["lag_min"].median()) if len(conf) else np.nan,
            "pct_first": leads / len(conf) if len(conf) else np.nan,
            "pct_after": echoes / len(conf) if len(conf) else np.nan,
        }
        # eco: "acerta" muito porque repete o que já saiu
        row["echo_dominated"] = bool(len(conf) >= 3 and row["pct_after"] >= 0.7)
        if edges is not None and not edges.empty:
            e = edges[edges["source"] == source]
            row["edge_n"] = int(e["net"].notna().sum())
            row["edge_net_median"] = float(e["net"].median()) if row["edge_n"] else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if reputation is not None and not reputation.empty and not table.empty:
        table = table.merge(reputation[["source", "mean"]].rename(columns={"mean": "reputation"}), on="source", how="left")
        table["rep_bucket"] = pd.cut(table["reputation"], [0, 0.3, 0.6, 1.0], labels=["baixa", "média", "alta"], include_lowest=True)
    return table.sort_values(["confirmed", "items"], ascending=False).reset_index(drop=True)
