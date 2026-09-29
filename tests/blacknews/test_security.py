import json

import numpy as np
import pandas as pd
import pytest

from trader.blacknews import security
from trader.blacknews.analysis import analyze_security_frames
from trader.blacknews.api import black_features_from, columns
from trader.blacknews.sources import PoliteClient

T0 = pd.Timestamp("2026-09-20", tz="UTC")


def fake_client(routes, log=None):
    def fetch(url, headers):
        if log is not None:
            log.append(url)
        body = routes.get(url)
        return (200, body, {}) if body is not None else (404, b"", {})
    return PoliteClient(fetch=fetch, sleep=lambda s: None)


def sec_item(i, minutes, entity, source="peckshield", kind="security_firm", itype="hack", usd=None, precision="second",
             first_seen_delay=pd.Timedelta(days=5)):
    pub = T0 + pd.Timedelta(minutes=minutes)
    return {"id": f"{source}:{i}", "source_id": source, "source": source, "source_kind": kind, "url": "https://x/y",
            "title": f"{entity} exploited", "snippet": "short", "published_at": pub, "ts_precision": precision,
            "first_seen_at": pub + first_seen_delay, "entity": entity, "entity_type": "defi", "incident_type": itype,
            "claimed_usd": usd, "assets": "BTC", "confidence": 0.8, "extracted_by": "llm"}


# portão de termos


def test_gate_refuses_disallowed_sources_before_any_request():
    log = []
    client = fake_client({}, log)
    with pytest.raises(security.SourceNotAllowed):
        security.fetch_allowed(client, "ahmia")
    with pytest.raises(security.SourceNotAllowed):
        security.fetch_allowed(client, "unknown_source")
    with pytest.raises(security.SourceNotAllowed):  # host diferente do aprovado
        security.fetch_allowed(client, "rekt_news", "https://evil.example/feed")
    assert log == []
    out = security.fetch_all(fake_client({}, log), T0)
    assert str(out["ahmia"]).startswith("recusada pelo portão")
    assert not any("ahmia" in u for u in log)


def test_every_policy_is_documented():
    for p in security.SOURCES.values():
        assert p.reason and p.checked
    assert not security.SOURCES["ahmia"].allowed


# só título/trecho/link/carimbos


def test_make_frame_truncates_and_rejects_extra_content():
    long = "<p>" + "x" * 5000 + "</p>"
    df = security.make_frame([{**sec_item(1, 0, "Foo"), "snippet": long, "title": "t" * 900}])
    assert df["snippet"].str.len().max() == security.SNIPPET_MAX and df["title"].str.len().max() == security.SNIPPET_MAX
    with pytest.raises(ValueError):
        security.make_frame([{**sec_item(2, 0, "Foo"), "body": "full leaked text"}])
    with pytest.raises(ValueError):
        security.make_frame([{**sec_item(3, 0, "Foo"), "attachment": b"..."}])


def test_store_refuses_content_beyond_snippet(tmp_path):
    store = security.SecurityStore(tmp_path)
    df = security.make_frame([sec_item(1, 0, "Foo")])
    assert store.write(df) == 1
    bad = df.copy()
    bad.loc[0, "snippet"] = "y" * 1000
    with pytest.raises(ValueError):
        store.write(bad)
    with pytest.raises(ValueError):
        store.write(df.assign(content="x"))


# conectores


def test_defillama_day_precision_visible_end_of_day():
    body = json.dumps([{"date": int(T0.timestamp()), "name": "Bitget", "classification": "Access Control",
                        "technique": "Private key compromised", "amount": 387e6, "chain": ["Ethereum"],
                        "bridgeHack": False, "targetType": "CEX"}]).encode()
    df = security.fetch_defillama(fake_client({security.SOURCES["defillama_hacks"].url: body}), T0 + pd.Timedelta(days=30))
    row = df.iloc[0]
    assert row["ts_precision"] == "day" and row["entity_type"] == "cex" and row["claimed_usd"] == 387e6
    assert row["visible_at"] == T0 + pd.Timedelta(days=1, minutes=5)  # data sem hora: só no fim do dia


def test_status_page_uses_first_update_time():
    desc = ("<p><small>Sep <var data-var='date'>28</var>, <var data-var='time'>11:58</var> UTC</small><br>Resolved</p>"
            "<p><small>Sep <var data-var='date'>27</var>, <var data-var='time'>09:10</var> UTC</small><br>Withdrawals paused</p>")
    xml = f"""<rss><channel><item><title>Withdrawals paused</title><link>https://status.kraken.com/i/1</link>
    <guid>g1</guid><pubDate>Mon, 28 Sep 2026 11:58:34 +0000</pubDate><description><![CDATA[{desc}]]></description></item></channel></rss>"""
    url = security.SOURCES["status_kraken"].url
    df = security.fetch_rss(fake_client({url: xml.encode()}), "status_kraken", pd.Timestamp("2026-09-29", tz="UTC"))
    assert df["published_at"].iloc[0] == pd.Timestamp("2026-09-27 09:10", tz="UTC")


# extração


def test_extractor_llm_batched_cached_and_rules_fallback(tmp_path):
    calls = []

    def post(url, payload, timeout):
        calls.append(1)
        return {"choices": [{"message": {"content": json.dumps({"items": [
            {"i": 0, "is_incident": True, "entity": "Bitget", "entity_type": "cex", "incident_type": "hack",
             "claimed_usd": 387000000, "confidence": 0.9},
            {"i": 1, "is_incident": False, "entity": None, "incident_type": "not_incident", "confidence": 0.8}]})}}]}

    rows = [dict(sec_item(1, 0, None), title="Bitget hacked for $387M", extracted_by=None),
            dict(sec_item(2, 0, None), title="Our Q3 audit report", extracted_by=None)]
    df = security.make_frame(rows)
    cache = security.ExtractCache(tmp_path / "c.sqlite")
    out = security.Extractor(cache, post=post).extract(df)
    assert out["entity"].iloc[0] == "Bitget" and out["incident_type"].iloc[1] == "not_incident"
    security.Extractor(cache, post=post).extract(df)
    assert len(calls) == 1  # segunda vez veio do cache

    def down(url, payload, timeout):
        raise ConnectionError
    fallback = security.Extractor(security.ExtractCache(":memory:"), post=down).extract(df)
    assert fallback["extracted_by"].iloc[0] == "rules" and fallback["claimed_usd"].iloc[0] == 387e6
    assert fallback["incident_type"].iloc[0] == "hack"


def test_parse_amount():
    assert security.parse_amount("drained $1.5M and $300k") == 1.5e6
    assert security.parse_amount("lost $2 billion") == 2e9


# deduplicação


def test_dedup_across_sources_and_window():
    df = security.make_frame([
        sec_item(1, 0, "Bitget", source="peckshield"),
        sec_item(2, 60, "Bitget Exchange", source="slowmist"),
        sec_item(3, 60 * 24 * 5, "Bitget", source="certik"),  # 5 dias depois: incidente novo
        sec_item(4, 10, "Radiant", source="certik"),
        sec_item(5, 20, "Weekly recap", source="certik", itype="not_incident"),
    ])
    d = security.dedup_incidents(df).set_index("id")
    assert d.loc["peckshield:1", "incident_id"] == d.loc["slowmist:2", "incident_id"]
    assert d.loc["certik:3", "incident_id"] != d.loc["peckshield:1", "incident_id"]
    assert d.loc["certik:4", "incident_id"] != d.loc["peckshield:1", "incident_id"]
    assert d.loc["certik:5", "incident_id"] is None
    table = security.incidents_table(d.reset_index())
    assert len(table) == 3 and table.iloc[0]["sources"] == 2


# lead/lag, reputação e features causais


def trusted(rows):
    return pd.DataFrame(rows, columns=["id", "source", "tier", "visible_at", "text_en", "assets"])


def world():
    items = security.make_frame([
        sec_item(1, 0, "Bitget", source="peckshield", usd=50e6),
        sec_item(2, 300, "Bitget", source="slowmist", usd=387e6),
        sec_item(3, 60 * 30, "Radiant", source="certik", usd=4e6),
        sec_item(4, 60 * 50, "Petrobras", source="certik", itype="breach_claim"),
    ])
    tr = trusted([
        ("t1", "coindesk.com", 1, T0 + pd.Timedelta(minutes=120), "Bitget hacked, $387M stolen", "BTC"),
        ("t2", "valor.com.br", 1, T0 + pd.Timedelta(hours=60), "Petrobras investiga vazamento de dados", "PETR4"),
    ])
    return items, tr


def test_incident_lead_lag_and_confirmation():
    items, tr = world()
    sec = analyze_security_frames(items, tr)
    lags = sec.lags.set_index("id")
    assert lags.loc["peckshield:1", "status"] == "lead" and lags.loc["peckshield:1", "lag_min"] == -115
    assert lags.loc["slowmist:2", "status"] == "echo"
    conf = sec.confirmations.set_index("id")
    assert conf.loc["peckshield:1", "confirmed_at"] == T0 + pd.Timedelta(minutes=120)
    rep = sec.reputation.scores(T0 + pd.Timedelta(days=3), ["peckshield", "slowmist"])
    assert rep.loc["peckshield", "score"] > rep.loc["slowmist", "score"]


def test_security_features_causal_confirmation_and_b3_routing():
    items, tr = world()
    sec = analyze_security_frames(items, tr)
    idx = pd.date_range(T0, T0 + pd.Timedelta(days=3), freq="15min", tz="UTC")
    btc = black_features_from(None, "BTC", "15m", idx, weighted=False, sec=sec)
    at = lambda m: btc.loc[T0 + pd.Timedelta(minutes=m)]
    assert at(0)["black_sec_incident_count"] == 0  # visível só em +5 min
    assert at(15)["black_sec_incident_count"] == 1 and at(15)["black_sec_confirmed_flag"] == 0
    assert at(135)["black_sec_confirmed_flag"] == 1  # depois da notícia de nível 1
    assert at(315)["black_sec_max_claimed_usd"] == 387e6
    assert btc["black_sec_minutes_since_incident"].iloc[0] == security.NO_INCIDENT_MINUTES
    petr = black_features_from(None, "PETR4", "15m", idx, weighted=False, sec=sec)
    assert petr["black_sec_incident_count"].max() == 1
    assert btc.columns.tolist() == columns(False, security=True)
    assert set(security.SEC_FEATURES) <= set(btc.columns)


def test_security_features_causal_under_truncation():
    items, tr = world()
    idx = pd.date_range(T0, T0 + pd.Timedelta(days=3), freq="5min", tz="UTC")
    for w in (False, True):
        full = black_features_from(None, "BTC", "5m", idx, weighted=w, sec=analyze_security_frames(items, tr))
        for cut_h in (1, 3, 40):
            cutoff = T0 + pd.Timedelta(hours=cut_h)
            part_sec = analyze_security_frames(items[items["visible_at"] <= cutoff], tr[tr["visible_at"] <= cutoff])
            part = black_features_from(None, "BTC", "5m", idx, weighted=w, sec=part_sec)
            pd.testing.assert_frame_equal(full[idx <= cutoff], part[idx <= cutoff])
