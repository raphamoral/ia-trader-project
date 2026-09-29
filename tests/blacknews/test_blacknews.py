import json
import os

import numpy as np
import pandas as pd
import pytest

from trader.blacknews import experiment, leadlag, sources, spam
from trader.blacknews.adapters import black_feature_columns, trusted_feature_columns
from trader.blacknews.enrich import detect_lang, enrich, is_claim
from trader.blacknews.features import build_features
from trader.blacknews.reputation import black_outcomes, black_reputation, promotion_candidates
from trader.blacknews.schema import BlackStore, black_until, new_frame, with_visibility

T0 = pd.Timestamp("2026-09-20", tz="UTC")


def item(i, minutes, text, source="4chan/biz", kind="social", assets="BTC", sentiment=0.5, claim=False, backfilled=True):
    pub = T0 + pd.Timedelta(minutes=minutes)
    return {
        "id": f"x:{i}", "source_id": "t", "source": source, "source_kind": kind, "url": "", "title": "",
        "text": text, "original_lang": "en", "translated_text": text, "published_at": pub,
        "first_seen_at": pub + pd.Timedelta(days=2), "backfilled": backfilled, "ts_reliability": "server",
        "engagement": 0, "content_hash": str(i), "assets": assets, "sentiment": sentiment,
        "stance": "bullish" if sentiment > 0.2 else "bearish" if sentiment < -0.2 else "neutral",
        "event_type": "other", "is_claim": claim, "confidence": 0.5, "enriched_by": "llm",
    }


def frame(rows):
    return with_visibility(new_frame(rows))


# visibilidade e point-in-time


def test_visibility_follows_trusted_rule():
    live = frame([item(1, 0, "a", backfilled=False)])
    hist = frame([item(2, 0, "b", backfilled=True)])
    assert live["visible_at"].iloc[0] == live["first_seen_at"].iloc[0]  # ao vivo: só depois de coletar
    assert hist["visible_at"].iloc[0] == T0 + pd.Timedelta(minutes=5)  # histórico: publicação + latência


def test_black_until_never_returns_future():
    df = frame([item(i, i * 10, f"post {i}") for i in range(10)])
    as_of = T0 + pd.Timedelta(minutes=47)
    out = black_until(df, as_of)
    assert (out["visible_at"] <= as_of).all() and len(out) == 5


def test_store_rejects_trusted_tier_and_dedups(tmp_path):
    store = BlackStore(tmp_path)
    df = frame([item(1, 0, "a"), item(2, 5, "b")])
    assert store.write(df) == 2 and store.write(df) == 0
    bad = df.copy()
    bad["tier"] = "1"
    with pytest.raises(ValueError):
        store.write(bad)


# fontes


def fake_fetch(routes):
    def fetch(url, headers):
        body = routes.get(url)
        return (200, body, {}) if body is not None else (404, b"", {})
    return fetch


def test_4chan_catalog_and_archive_parsing():
    now = T0 + pd.Timedelta(days=1)
    base = "https://a.4cdn.org/biz"
    routes = {
        f"{base}/catalog.json": json.dumps([{"threads": [
            {"no": 1, "time": int(T0.timestamp()), "sub": "BTC to 200k", "com": "buy &gt; sell<br>now", "replies": 12},
            {"no": 2, "time": int(T0.timestamp()), "sticky": 1, "com": "rules"},
        ]}]).encode(),
        f"{base}/archive.json": b"[3]",
        f"{base}/thread/3.json": json.dumps({"posts": [{"no": 3, "time": int(T0.timestamp()) + 60, "com": "eth rug"}, {"no": 4}]}).encode(),
    }
    client = sources.PoliteClient(fetch=fake_fetch(routes), sleep=lambda s: None)
    df = sources.FourChanBiz(archive_limit=5).fetch(client, now)
    assert sorted(df["id"]) == ["4chan_biz:1", "4chan_biz:3"]  # sticky fora
    row = df.set_index("id").loc["4chan_biz:1"]
    assert row["text"] == "BTC to 200k buy > sell now" and row["engagement"] == 12
    assert bool(row["backfilled"]) and row["visible_at"] == T0 + pd.Timedelta(minutes=5)
    assert client.min_interval["a.4cdn.org"] >= 1.0


def test_rss_parsing_with_author_as_source():
    xml = b"""<?xml version="1.0"?><rss xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
    <item><title>Bitcoin will pump</title><link>https://medium.com/p/1</link><guid>g1</guid>
    <dc:creator>anon</dc:creator><pubDate>Sun, 20 Sep 2026 10:00:00 GMT</pubDate>
    <description>&lt;p&gt;trust me&lt;/p&gt;</description></item></channel></rss>"""
    feed = sources.RSSFeed("medium_tag_bitcoin", "https://medium.com/feed/tag/bitcoin", "blog", per_author=True)
    client = sources.PoliteClient(fetch=fake_fetch({feed.url: xml}), sleep=lambda s: None)
    df = feed.fetch(client, T0 + pd.Timedelta(days=1))
    assert df["source"].iloc[0] == "medium.com/anon"
    assert df["text"].iloc[0] == "Bitcoin will pump. trust me"
    assert df["published_at"].iloc[0] == pd.Timestamp("2026-09-20 10:00", tz="UTC")


def test_rss_ids_are_distinct_per_guid():
    entry = "<item><title>t{i}</title><guid>https://site.com/p/{i}</guid><pubDate>Sun, 20 Sep 2026 10:00:00 GMT</pubDate></item>"
    xml = ("<rss><channel>" + "".join(entry.format(i=i) for i in range(3)) + "</channel></rss>").encode()
    feed = sources.RSSFeed("farm1", "https://site.com/feed", "farm")
    df = feed.fetch(sources.PoliteClient(fetch=fake_fetch({feed.url: xml}), sleep=lambda s: None), T0)
    assert df["id"].nunique() == 3


def test_reddit_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    with pytest.raises(sources.SourceDisabled):
        sources.RedditNew().fetch(sources.PoliteClient(fetch=fake_fetch({})), T0)


# enriquecimento (Enricher do trader.news com LLM falsa)


def test_enrich_uses_trusted_enricher_and_adds_black_fields(tmp_path):
    from trader.news.enrich import Enricher

    def fake_post(url, payload, timeout):
        return {"choices": [{"message": {"content": json.dumps({"items": [
            {"i": 0, "en": "Binance will list the token tomorrow, insiders say", "assets": ["BTC"], "sentiment": 0.7,
             "event_type": "corporate", "confidence": 0.6}]})}}]}

    raw = frame([item(1, 0, "A Binance vai listar o token amanhã, dizem insiders")])
    raw[["original_lang", "translated_text", "enriched_by"]] = [None, None, None]
    out, enr = enrich(raw, tmp_path, enricher=Enricher(cache_dir=tmp_path, post=fake_post))
    row = out.iloc[0]
    assert row["original_lang"] == "pt" and row["translated_text"].startswith("Binance will list")
    assert row["enriched_by"] == "llm" and row["stance"] == "bullish" and bool(row["is_claim"])
    assert (tmp_path / "news" / "cache" / "labels.jsonl").exists()  # cache no diretório black


def test_detect_lang_and_claims():
    assert detect_lang("比特币今天上涨") == "zh" and detect_lang("o preço do dólar vai subir") == "pt"
    assert detect_lang("el precio de bitcoin está por las nubes") == "es"
    assert is_claim("SEC approved the ETF", "regulation") and not is_claim("wagmi frens", "other")


# spam


def test_near_duplicates_and_causal_dup_count():
    texts = ["BUY $PEPE NOW 100x join t.me/pumpgroup today guaranteed", "BUY $PEPE NOW 100x join t.me/pumpgroup today guaranteed!!",
             "Fed minutes show officials split on the pace of rate cuts this year"]
    groups = spam.near_duplicate_groups(texts)
    assert groups[0] == groups[1] != groups[2]
    df = frame([item(i, i, t, source=f"s{i}") for i, t in enumerate(texts)])
    scored = spam.spam_scores(df)
    assert scored["dup_count"].tolist() == [0, 1, 0]  # só a cópia posterior conta
    assert scored["is_spam"].iloc[1] and not scored["is_spam"].iloc[2]
    assert scored["promo"].iloc[0]


def test_burst_detection_per_source():
    df = frame([item(i, i, f"distinct message number {i} about markets") for i in range(6)])
    assert spam.spam_scores(df, burst_k=5)["is_burst"].tolist() == [False] * 4 + [True, True]


# lead/lag e reputação


def trusted(rows):
    return pd.DataFrame(rows, columns=["id", "source", "tier", "visible_at", "text_en", "assets"])


def scenario():
    black = frame([
        item(1, 0, "Exchange Kraken hacked hot wallet drained millions", source="leader", claim=True),
        item(2, 600, "Exchange Kraken hacked hot wallet drained millions report", source="echoer", claim=True),
        item(3, 10, "Tether collapse imminent insiders confirm", source="liar", claim=True),
    ])
    tr = trusted([("t1", "reuters.com", 1, T0 + pd.Timedelta(minutes=120), "Kraken hot wallet hacked, millions drained", "BTC")])
    return black, tr


def test_item_lags_classify_lead_echo_unconfirmed():
    black, tr = scenario()
    matches = leadlag.StoryMatcher().match(black, tr)
    lags = leadlag.item_lags(black, matches, data_end=T0 + pd.Timedelta(days=5)).set_index("source")
    assert lags.loc["leader", "status"] == "lead" and lags.loc["leader", "lag_min"] == -115
    assert lags.loc["echoer", "status"] == "echo" and lags.loc["echoer", "lag_min"] == 485
    assert lags.loc["liar", "status"] == "unconfirmed"


def test_confirmation_after_horizon_does_not_count():
    black, tr = scenario()
    late = tr.assign(visible_at=T0 + pd.Timedelta(hours=60))
    lags = leadlag.item_lags(black, leadlag.StoryMatcher().match(black, late), T0 + pd.Timedelta(days=5)).set_index("source")
    assert lags.loc["leader", "status"] == "unconfirmed"


def test_reputation_credits_leads_not_echoes_and_is_point_in_time():
    black, tr = scenario()
    lags = leadlag.item_lags(black, leadlag.StoryMatcher().match(black, tr), T0 + pd.Timedelta(days=5))
    outs = black_outcomes(lags, spam.spam_scores(black))
    echo_rows = outs[outs["source"] == "echoer"]
    assert not echo_rows.loc[echo_rows["kind"] == "timeliness", "success"].any()  # eco não ganha pontualidade
    rep = black_reputation(outs, black)
    before = rep.scores(T0 + pd.Timedelta(minutes=60), ["leader"]).loc["leader", "score"]  # antes da confirmação
    after = rep.scores(T0 + pd.Timedelta(days=3), ["leader", "echoer", "liar"])
    assert before == pytest.approx(rep.scores(T0, ["leader"]).loc["leader", "score"])
    assert after.loc["leader", "score"] > after.loc["echoer", "score"] > after.loc["liar", "score"]
    promo = promotion_candidates(rep, outs, T0 + pd.Timedelta(days=3))
    assert promo.empty  # 1 lead não basta; e promoção é só relatório


def test_source_table_flags_echo_dominated():
    rows, tr_rows = [], []
    for k in range(4):
        rows.append(item(k, k * 1000 + 30, f"story {k} alpha beta gamma delta event{k}", source="echoer"))
        tr_rows.append((f"t{k}", "reuters.com", 1, T0 + pd.Timedelta(minutes=k * 1000), f"story {k} alpha beta gamma delta event{k}", "BTC"))
    black, tr = frame(rows), trusted(tr_rows)
    lags = leadlag.item_lags(black, leadlag.StoryMatcher().match(black, tr), T0 + pd.Timedelta(days=10))
    table = leadlag.source_table(lags)
    assert table.loc[0, "echo_dominated"] and table.loc[0, "pct_after"] == 1.0


def test_edge_window_net_of_fees():
    black, tr = scenario()
    lags = leadlag.item_lags(black, leadlag.StoryMatcher().match(black, tr), T0 + pd.Timedelta(days=5))
    lags.loc[lags["source"] == "leader", "stance"] = "bearish"
    ts = pd.date_range(T0, periods=40, freq="5min", tz="UTC")
    px = pd.DataFrame({"ts": ts, "open": np.linspace(100, 90, 40)})
    edges = leadlag.edge_windows(lags, lambda a: px, 0.001)
    row = edges.iloc[0]
    assert row["move"] < 0 and row["net"] == pytest.approx(-row["move"] - 0.001)


# features: separação e causalidade


def test_features_are_prefixed_and_causal():
    rows = [item(i, i * 30, f"btc pump number {i}", sentiment=0.8) for i in range(20)]
    black = spam.spam_scores(frame(rows))
    grid = pd.date_range(T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=6), freq="15min", tz="UTC")
    f1 = build_features(black, grid, "BTC")
    assert all(c == "ts" or c.startswith("black_") for c in f1.columns)
    future = frame([item(100 + i, 400 + i, "btc crash dump", sentiment=-0.9) for i in range(10)])
    f2 = build_features(spam.spam_scores(pd.concat([frame(rows), future], ignore_index=True)), grid, "BTC")
    cutoff = future["visible_at"].min()
    early = f1["ts"] < cutoff
    pd.testing.assert_frame_equal(f1[early].reset_index(drop=True), f2[early].reset_index(drop=True))


def test_arm_column_separation():
    cols = ["ts", "news_sent_4h", "news_count_4h", "black_count", "black_mean_sent", "black_rw_count", "black_rw_mean_sent"]
    arms = experiment.arms_for(cols)
    assert not any(c.startswith("black_") for c in arms["A"])
    assert set(arms["B"]) - set(arms["A"]) == {"black_count", "black_mean_sent"}
    assert set(arms["B'"]) - set(arms["A"]) == {"black_rw_count", "black_rw_mean_sent"}
    assert all(c.startswith("black_") for c in arms["C"])
    assert trusted_feature_columns(cols) == ["news_sent_4h", "news_count_4h"]
    assert black_feature_columns(cols, weighted=True) == ["black_rw_count", "black_rw_mean_sent"]


# experimento


def test_sign_test_and_bootstrap():
    assert experiment.sign_test_p(np.ones(6)) == pytest.approx(2 / 64)
    assert experiment.sign_test_p(np.array([1, -1, 1, -1])) == 1.0
    lo, hi = experiment.bootstrap_ci(np.full(10, 0.5))
    assert lo == hi == pytest.approx(0.5)


def synthetic(edge_b=0.0, edge_rw=0.0, noise=0.2):
    def evaluate(arm, cols, seed):
        base = np.random.default_rng(seed).normal(1, 0.5)
        rng = np.random.default_rng([seed, len(arm), sum(map(ord, arm))])
        score = base
        if any(c.startswith("black_") and not c.startswith("black_rw_") for c in cols):
            score += edge_b + rng.normal(0, noise)
        if any(c.startswith("black_rw_") for c in cols):
            score += edge_rw + rng.normal(0, noise)
        return {"score": score, "goal_hit_rate": 0.0, "ruin": 0.0}
    return evaluate


COLS = ["news_a", "black_count", "black_rw_count"]


def verdict(report, treat, ctrl, metric="score"):
    return next(c.verdict for c in report.comparisons if (c.treatment, c.control, c.metric) == (treat, ctrl, metric))


def test_harness_detects_real_improvement_and_harm():
    rep = experiment.run_experiment(synthetic(edge_b=-0.5, edge_rw=0.5), COLS, list(range(20)), include_c=False)
    assert verdict(rep, "B'", "A") == "melhora"
    assert verdict(rep, "B", "A") == "piora"
    assert verdict(rep, "B'", "B") == "melhora"


def test_harness_refuses_to_conclude_without_evidence():
    few = experiment.run_experiment(synthetic(edge_rw=0.5), COLS, [0, 1, 2], include_c=False)
    assert verdict(few, "B'", "A").startswith("evidência insuficiente (n=3")
    null = experiment.run_experiment(synthetic(noise=0.3), COLS, list(range(20)), include_c=False)
    assert verdict(null, "B", "A").startswith("evidência insuficiente")


def test_leakage_smell_flags_concentrated_gains():
    idx = pd.date_range(T0, periods=100, freq="h", tz="UTC")
    a = pd.Series(0.0, index=idx)
    b = a.copy()
    b.iloc[50] = 10.0  # todo o ganho num único instante
    info = experiment.leakage_smell(b, a, pd.DatetimeIndex([idx[50]]), pd.Timedelta(hours=1))
    assert info["flag"] and info["gain_share"] == 1.0
    spread = pd.Series(0.1, index=idx)
    assert not experiment.leakage_smell(spread, a, pd.DatetimeIndex([idx[50]]), pd.Timedelta(hours=1))["flag"]


# API para a evolução


from trader.blacknews.analysis import analyze_frames
from trader.blacknews.api import black_features_from, columns


def history():
    """Fonte 'leader' antecipa 3 histórias do nível 1; 'echoer' só repete; 'noise' posta sem parar."""
    black, tr = [], []
    stories = ["kraken wallet drained attacker", "binance delisting privacy tokens", "tether freezes sanctioned addresses",
               "coinbase outage withdrawals halted"]
    for k in range(3):
        start = k * 24 * 60
        words = stories[k]
        black.append(item(f"l{k}", start, f"{words} breaking", source="leader", claim=True, sentiment=-0.8))
        black.append(item(f"e{k}", start + 300, f"{words} again", source="echoer", claim=True, sentiment=-0.8))
        tr.append((f"t{k}", "reuters.com", 1, T0 + pd.Timedelta(minutes=start + 120), f"{words} reported", "BTC"))
    for k in range(30):
        black.append(item(f"n{k}", 3 * 24 * 60 + k * 20, f"btc moon soon number {k}", source="noise", sentiment=0.9))
    black.append(item("l3", 3 * 24 * 60 + 200, f"{stories[3]} breaking", source="leader",
                      claim=True, sentiment=-0.8))
    return frame(black), trusted(tr)


def test_black_features_same_index_prefixed_and_lead_score_uses_leaders_only():
    black, tr = history()
    a = analyze_frames(black, tr)
    idx = pd.date_range(T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=4), freq="15min", tz="UTC")
    fw = black_features_from(a, "BTC", "15m", idx, weighted=True)
    fu = black_features_from(a, "BTC", "15m", idx, weighted=False)
    assert fw.index.equals(idx) and fu.index.equals(idx)
    assert list(fw.columns) == columns(True) and list(fu.columns) == columns(False)
    assert all(c.startswith("black_rw_") for c in fw) and not any(c.startswith("black_rw_") for c in fu)
    # o post do líder no dia 3 (bearish) aparece; o 'noise' (bullish, nunca confirmado) não entra no lead
    after = fw.loc[fw.index >= T0 + pd.Timedelta(days=3, minutes=205)]
    assert (after["black_rw_lead_count"].iloc[:4] == 1).all()
    assert (after["black_rw_lead_score"].iloc[:4] < 0).all()
    assert fw["black_rw_lead_count"].max() == 1  # eco e ruído nunca contam como líder


def test_black_features_are_causal_under_truncation():
    black, tr = history()
    idx = pd.date_range(T0, T0 + pd.Timedelta(days=4), freq="5min", tz="UTC")
    full = black_features_from(analyze_frames(black, tr), "BTC", "5m", idx, weighted=True)
    for cut_days in (1.5, 2.2, 3.1):
        cutoff = T0 + pd.Timedelta(days=cut_days)
        cut = analyze_frames(black[black["visible_at"] <= cutoff], tr[tr["visible_at"] <= cutoff])
        part = black_features_from(cut, "BTC", "5m", idx, weighted=True)
        before = idx <= cutoff
        pd.testing.assert_frame_equal(full[before], part[before])
    unweighted_full = black_features_from(analyze_frames(black, tr), "BTC", "5m", idx, weighted=False)
    cutoff = T0 + pd.Timedelta(days=2.2)
    part = black_features_from(
        analyze_frames(black[black["visible_at"] <= cutoff], tr[tr["visible_at"] <= cutoff]), "BTC", "5m", idx, weighted=False
    )
    pd.testing.assert_frame_equal(unweighted_full[idx <= cutoff], part[idx <= cutoff])


def test_black_features_empty_is_zero():
    idx = pd.date_range(T0, periods=4, freq="15min", tz="UTC")
    out = black_features_from(None, "BTC", "15m", idx)
    assert out.index.equals(idx) and (out == 0).all().all()


def test_daytrade_arms_and_pairs():
    arms = experiment.daytrade_arms(["news_a"], ["black_count"], ["black_rw_count"])
    assert arms["A"] == [] and arms["WB'"] == ["news_a", "black_rw_count"]
    with pytest.raises(ValueError):
        experiment.daytrade_arms(["black_count"], [], [])

    def evaluate(arm, cols, seed):
        base = np.random.default_rng(seed).normal(1, 0.5)
        return {"score": base + (0.4 if "black_rw_count" in cols else 0.0), "goal_hit_rate": 0.0, "ruin": 0.0,
                "pnl": pd.Series([1.0 if ("black_rw_count" in cols and i == 24) else 0.0 for i in range(48)],
                                 index=pd.date_range(T0, periods=48, freq="h", tz="UTC"))}

    rep = experiment.run_experiment(evaluate, [], list(range(8)), arms=arms, pairs=experiment.DAYTRADE_PAIRS,
                                    leak_control="W", suspicious=pd.DatetimeIndex([T0 + pd.Timedelta(hours=24)]))
    assert verdict(rep, "WB'", "W") == "melhora" and verdict(rep, "WB", "W").startswith("evidência insuficiente")
    assert set(rep.leakage) == {"WB", "WB'"} and rep.leakage["WB'"]["flag"]
