import hashlib
import io
import zipfile

import pandas as pd
import pytest

from trader.news import gkg
from trader.news.enrich import Enricher
from trader.news.pipeline import load_items
from trader.news.store import NewsStore, make_frame


def gkg_line(date="20240110120000", domain="reuters.com", url="https://reuters.com/a", title="Bitcoin ETF approved",
             orgs="securities and exchange commission", tone="3.5,5,1.5,6.5,20,0,300", translation="", themes="ECON_STOCKMARKET,10;"):
    cols = [""] * 27
    cols[0], cols[1], cols[2], cols[3], cols[4] = f"{date}-0", date, "1", domain, url
    cols[7], cols[13], cols[15], cols[25] = themes, orgs, tone, translation
    cols[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(cols)


def zipped(lines):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.gkg.csv", "\n".join(lines) + "\n")
    blob = buf.getvalue()
    return blob, hashlib.md5(blob).hexdigest()


NOW = pd.Timestamp("2026-09-29", tz="UTC")


def test_filter_keeps_whitelisted_and_relevant_only():
    rows = gkg.filter_rows([
        gkg_line(),
        gkg_line(domain="random-blog.xyz"),
        gkg_line(title="Local football results", orgs="", url="https://reuters.com/sport"),
        gkg_line(title="Copom mantém Selic", domain="valor.globo.com", url="https://valor.globo.com/x",
                 translation="srclc:por;eng:GT-POR 1.0"),
    ], "translation", NOW)
    assert [r["domain"] for r in rows] == ["reuters.com", "random-blog.xyz", "valor.globo.com"]
    frame = make_frame(rows, keep_unlisted=True)
    assert frame.set_index("domain")["tier"].to_dict() == {"reuters.com": 1, "random-blog.xyz": 9, "valor.globo.com": 1}
    assert len(make_frame(rows)) == 2  # padrão: fora da lista é descartado
    rows = [r for r in rows if r["domain"] != "random-blog.xyz"]
    assert rows[0]["query_assets"] == "BTC" and rows[0]["gkg_tone"] == 3.5
    assert rows[1]["original_lang"] == "pt" and rows[1]["query_assets"] == "IBOV,USDBRL"
    assert rows[0]["published_at"] == pd.Timestamp("2024-01-10 12:00", tz="UTC") and rows[0]["backfilled"]


def test_md5_is_verified():
    blob, md5 = zipped([gkg_line()])
    f = gkg.RemoteFile("http://x/20240110120000.gkg.csv.zip", len(blob), md5, "en")
    assert len(gkg.process_bytes(blob, f, NOW)) == 1
    with pytest.raises(ValueError):
        gkg.process_bytes(blob + b"x", f, NOW)


def test_plan_orders_recent_first_then_earlier():
    mk = lambda ts, feed="en": gkg.RemoteFile(f"http://x/{ts}.gkg.csv.zip", 1, "m", feed)
    files = [mk("20190101000000"), mk("20210101000000"), mk("20200101000000", "translation"), mk("20150101000000")]
    order = [f.url[9:17] for f in gkg.plan(files, pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2030-01-01", tz="UTC"), True)]
    assert order == ["20200101", "20210101", "20190101"]  # 2015-01 é antes do início do GKG 2.0


def test_master_list_parsing_separates_feeds():
    lines = ["1 a http://x/20240110120000.gkg.csv.zip", "2 b http://x/20240110120000.translation.gkg.csv.zip",
             "3 c http://x/20240110120000.export.CSV.zip", "garbage"]
    assert [f.md5 for f in gkg.iter_master(lines, "en")] == ["a"]
    assert [f.md5 for f in gkg.iter_master(lines, "translation")] == ["b"]


def test_run_is_resumable_and_writes_only_filtered_rows(tmp_path):
    blobs = {}
    files = []
    for i, date in enumerate(["20240110120000", "20240110121500", "20240110123000"]):
        blob, md5 = zipped([gkg_line(date=date, url=f"https://reuters.com/{i}"), gkg_line(domain="spam.xyz")])
        url = f"http://x/{date}.gkg.csv.zip"
        blobs[url] = blob
        files.append(gkg.RemoteFile(url, len(blob), md5, "en"))
    calls = []

    def fake_download(url, bw):
        calls.append(url)
        if url.endswith("121500.gkg.csv.zip") and calls.count(url) == 1:
            raise ConnectionError("rede caiu")
        return blobs[url]

    stats = gkg.run(tmp_path, files, download=fake_download, log=lambda m: None, flush_every=1, retries=2)
    assert stats["files"] == 3 and stats["rows"] == 6 and stats["failed"] == 0
    stored = NewsStore(tmp_path, subdir=gkg.SUBDIR).read_all()
    assert (stored["tier"] == 1).sum() == 3 and (stored["tier"] == 9).sum() == 1  # spam.xyz: mesma URL, 1 id
    calls.clear()
    again = gkg.run(tmp_path, files, download=fake_download, log=lambda m: None)
    assert calls == [] and again["files"] == 0  # manifesto: nada a refazer
    assert not list((tmp_path / "news" / gkg.SUBDIR).rglob("*.zip"))  # nada bruto em disco


def test_bandwidth_cap_sleeps_proportionally():
    t = [0.0]
    slept = []
    bw = gkg.Bandwidth(1000, clock=lambda: t[0], sleep=lambda s: slept.append(s))
    bw.take(1000)
    bw.take(500)
    assert slept == [pytest.approx(0.5)]


def test_gkg_items_get_tone_labels_in_pipeline(tmp_path):
    blob, md5 = zipped([gkg_line(tone="-7.5,1,8.5,9.5,20,0,300", title="Bitcoin exchange hacked",
                                 themes="CYBER_ATTACK,5;")])
    f = gkg.RemoteFile("http://x/20240110120000.gkg.csv.zip", len(blob), md5, "en")
    gkg.run(tmp_path, [f], download=lambda u, b: blob, log=lambda m: None)
    items = load_items(tmp_path, Enricher(tmp_path))
    row = items.iloc[0]
    assert row["label_source"] == "gdelt_tone" and row["sentiment"] == -1.0
    assert row["event_type"] == "hack_security" and row["visible_at"] == pd.Timestamp("2024-01-10 12:05", tz="UTC")


def test_select_for_llm_prefers_tier1_and_high_impact_with_daily_cap():
    from trader.news.pipeline import select_for_llm

    t = pd.Timestamp("2024-01-10 12:00", tz="UTC")
    items = pd.DataFrame({
        "id": list("abcde"), "domain": ["coindesk.com", "reuters.com", "reuters.com", "coindesk.com", "reuters.com"],
        "tier": [2, 1, 1, 2, 1], "event_type": ["other", "other", "monetary_policy", "hack_security", "other"],
        "sentiment": [0.9, 0.1, 0.1, 0.2, 0.0], "label_source": ["gdelt_tone"] * 4 + ["llm"],
        "story_rank": [0, 0, 0, 0, 0],
        "visible_at": [t, t, t, t + pd.Timedelta(days=1), t],
    })
    chosen = select_for_llm(items, daily_cap=1)
    assert chosen["id"].tolist() == ["c", "d"]  # dia 1: nível 1 + política monetária; dia 2: o único


def test_priority_plan_order_and_stop_file(tmp_path):
    mk = lambda ts: gkg.RemoteFile(f"http://x/{ts}.gkg.csv.zip", 1, "m", "en")
    files = [mk(t) for t in ["20160101000000", "20210701000000", "20221115000000", "20260901000000"]]
    order = [f.stamp[:6] for f in gkg.plan_priority(files, pd.Timestamp("2026-09-29", tz="UTC"))]
    assert order == ["202609", "202211", "202107", "201601"]  # recente, evento (FTX), resto 2020+, antes de 2020
    stop = tmp_path / "STOP"
    stop.touch()
    blob, md5 = zipped([gkg_line()])
    many = [gkg.RemoteFile(f"http://x/2024011012{m:02d}00.gkg.csv.zip", len(blob), md5, "en") for m in range(0, 60, 15)]
    stats = gkg.run(tmp_path, many, download=lambda u, b: blob, log=lambda m: None, workers=1, stop_file=stop)
    assert stats["files"] == 2  # só os já enviados (workers*2) terminam


def test_unlisted_items_are_excluded_from_features(tmp_path):
    from trader.news.features import news_features
    rows = [
        {"source": "t", "domain": "reuters.com", "url": "https://reuters.com/1", "original_lang": "en",
         "original_text": "bitcoin rally", "published_at": pd.Timestamp("2024-01-10 12:00", tz="UTC"),
         "first_seen_at": NOW, "backfilled": True, "query_assets": "BTC"},
        {"source": "t", "domain": "farm.xyz", "url": "https://farm.xyz/1", "original_lang": "en",
         "original_text": "bitcoin to the moon", "published_at": pd.Timestamp("2024-01-10 12:00", tz="UTC"),
         "first_seen_at": NOW, "backfilled": True, "query_assets": "BTC"},
    ]
    NewsStore(tmp_path).write(make_frame(rows, keep_unlisted=True))
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-10 13:00", tz="UTC")])
    assert news_features("BTC/USDT", "15m", idx, data=tmp_path, use_reputation=False)["news_count_1h"].iloc[0] == 1
