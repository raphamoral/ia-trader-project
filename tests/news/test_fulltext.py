import pandas as pd

from trader.news import fulltext
from trader.news.fulltext import Robots, apply_fulltext, fetch_and_label

NOW = pd.Timestamp("2026-09-29 12:00", tz="UTC")
BODY = "<html><body><article><p>" + ("Bitcoin rallies after ETF approval. " * 40) + "</p></article></body></html>"


def items(url="https://news.example.com/a", tier=1):
    return pd.DataFrame({"id": ["n1"], "url": [url], "tier": [tier], "original_text": ["BTC up"],
                         "first_seen_at": [NOW - pd.Timedelta(minutes=5)], "backfilled": [False]})


class FakeEnricher:
    batch_size = 8

    def __init__(self):
        self.seen = []

    def _call(self, texts):
        self.seen += texts
        return [{"en": t, "assets": ["BTC"], "sentiment": 0.8, "event_type": "regulation", "confidence": 0.9}
                for t in texts]


def run(tmp_path, robots_txt="", html=BODY, **kw):
    enricher = FakeEnricher()
    stats = fetch_and_label(tmp_path, items(**kw), enricher, get=lambda url: html,
                            robots=Robots(fetch=lambda url: robots_txt), now=lambda: NOW, sleep=lambda s: None)
    return stats, enricher


def test_respects_robots_txt(tmp_path):
    stats, enricher = run(tmp_path, robots_txt="User-agent: *\nDisallow: /")
    assert stats["bloqueadas_robots"] == 1 and enricher.seen == []


def test_skips_paywall_or_short_pages(tmp_path):
    stats, enricher = run(tmp_path, html="<html><body><p>Assine para ler.</p></body></html>")
    assert stats["curtas_ou_paywall"] == 1 and enricher.seen == []


def test_labels_title_plus_body_and_never_refetches(tmp_path):
    stats, enricher = run(tmp_path)
    assert stats["rotuladas"] == 1 and "Bitcoin rallies" in enricher.seen[0]
    again, _ = run(tmp_path)
    assert again["lidas"] == 0  # já lida antes


def test_fulltext_label_only_visible_after_it_exists(tmp_path):
    run(tmp_path)  # rótulo criado em NOW
    labeled = items().assign(sentiment=0.0, event_type="other", confidence=0.2, assets="BTC",
                             label_source="lexicon", visible_at=NOW - pd.Timedelta(minutes=4))
    out = apply_fulltext(labeled, tmp_path)
    assert out.loc[0, "label_source"] == "llm_fulltext" and out.loc[0, "sentiment"] == 0.8
    assert out.loc[0, "visible_at"] == NOW  # adiado de 11:56 para 12:00


def test_only_recent_live_tier_1_2(tmp_path):
    old = items().assign(first_seen_at=NOW - pd.Timedelta(days=2))
    assert fulltext.candidates(old, set(), NOW).empty
    assert fulltext.candidates(items(tier=3), set(), NOW).empty
    assert fulltext.candidates(items().assign(backfilled=True), set(), NOW).empty
