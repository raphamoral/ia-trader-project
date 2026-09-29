import json
import random

import numpy as np
import pandas as pd
import pytest

from trader.news import calendar as cal
from trader.news.audit import shift_sensitivity, truncation_audit
from trader.news.enrich import Enricher, lexicon_label, validate
from trader.news.features import build_features
from trader.news.gdelt import Gdelt
from trader.news.reputation import (
    PRIORS, Reputation, ReputationConfig, cap_shares, direction_outcomes, story_outcomes,
)
from trader.news.rss import parse_feed
from trader.news.sources import tier_of
from trader.news.store import NewsStore, make_frame, visible_at
from trader.news.stories import assign_stories

T0 = pd.Timestamp("2024-01-10 12:00", tz="UTC")
LAT = pd.Timedelta(minutes=5)


def raw(title, domain="reuters.com", minutes=0, lang="en", backfilled=True, seen_delay=None, url=None, assets="BTC"):
    pub = T0 + pd.Timedelta(minutes=minutes)
    return {
        "source": "test", "domain": domain, "url": url or f"https://{domain}/{abs(hash(title))}",
        "original_lang": lang, "original_text": title, "published_at": pub,
        "first_seen_at": pub + (seen_delay or pd.Timedelta(days=400)), "backfilled": backfilled,
        "query_assets": assets,
    }


# fontes e armazenamento

def test_whitelist_tiers_and_subdomains():
    assert tier_of("www.reuters.com") == 1
    assert tier_of("markets.businessinsider.com") is None
    assert tier_of("coindesk.com") == 2
    frame = make_frame([raw("a"), raw("b", domain="random-crypto-blog.xyz")])
    assert frame["domain"].tolist() == ["reuters.com"]


def test_visibility_rules_live_vs_backfill():
    df = make_frame([
        raw("backfill", minutes=0, backfilled=True),
        raw("live seen late", minutes=0, backfilled=False, seen_delay=pd.Timedelta(minutes=20)),
        raw("live seen fast", minutes=0, backfilled=False, seen_delay=pd.Timedelta(minutes=1)),
    ])
    vis = dict(zip(df["original_text"], visible_at(df, LAT)))
    assert vis["backfill"] == T0 + LAT
    assert vis["live seen late"] == T0 + pd.Timedelta(minutes=20)  # não podíamos ler antes de capturar
    assert vis["live seen fast"] == T0 + LAT


def test_news_until_never_returns_future(tmp_path):
    store = NewsStore(tmp_path)
    store.write(make_frame([raw(f"item {i}", minutes=i * 10) for i in range(30)]))
    for as_of in pd.date_range(T0 - pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=6), freq="7min", tz="UTC"):
        got = store.news_until(as_of, lookback=pd.Timedelta(days=2), latency=LAT)
        assert (got["visible_at"] <= as_of).all()
        expected = sum(1 for i in range(30) if T0 + pd.Timedelta(minutes=i * 10) + LAT <= as_of)
        assert len(got) == expected
    with pytest.raises(ValueError):
        store.news_until(pd.Timestamp("2024-01-10"))


def test_dedup_merges_queries_and_keeps_first_capture(tmp_path):
    store = NewsStore(tmp_path)
    a = raw("same story", url="https://www.reuters.com/x?utm=1", assets="BTC", seen_delay=pd.Timedelta(days=2))
    b = raw("same story", url="https://reuters.com/x/", assets="MACRO", seen_delay=pd.Timedelta(days=1))
    assert store.write(make_frame([a])) == 1
    assert store.write(make_frame([b])) == 0
    all_items = store.read_all()
    assert len(all_items) == 1
    assert all_items["query_assets"].iloc[0] == "BTC,MACRO"
    assert all_items["first_seen_at"].iloc[0] == T0 + pd.Timedelta(days=1)


# enriquecimento

def fake_llm(translations):
    calls = []

    def post(url, payload, timeout):
        calls.append(payload)
        lines = payload["messages"][0]["content"].split("Headlines:\n")[1].split("\n\nAnswer")[0].splitlines()
        items = [{"i": i, "en": translations.get(l.split(". ", 1)[1], "x"), "assets": ["BTC", "DOGE"],
                  "sentiment": 3, "event_type": "weird", "confidence": 0.8} for i, l in enumerate(lines)]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}]}

    return post, calls


def test_enrich_validates_caches_and_translates(tmp_path):
    post, calls = fake_llm({"비트코인 ETF 승인": "Bitcoin ETF approved"})
    items = make_frame([raw("비트코인 ETF 승인", lang="ko")])
    out = Enricher(tmp_path, post=post).enrich(items)
    row = out.iloc[0]
    assert row["translated_text"] == "Bitcoin ETF approved"
    assert row["assets"] == "BTC"  # DOGE fora do universo
    assert row["sentiment"] == 1.0 and row["event_type"] == "other"
    assert row["translator"] == "ollama:qwen2.5:7b" and row["label_source"] == "llm"
    # nova instância lê do disco: nenhuma chamada nova
    again = Enricher(tmp_path, post=post).enrich(items)
    assert len(calls) == 1 and again.iloc[0]["translated_text"] == "Bitcoin ETF approved"


def test_enrich_falls_back_to_lexicon(tmp_path):
    def down(*_):
        raise ConnectionError

    e = Enricher(tmp_path, post=down)
    out = e.enrich(make_frame([raw("Bitcoin plunge after exchange hack")]))
    assert out.iloc[0]["label_source"] == "lexicon" and out.iloc[0]["sentiment"] < 0
    assert out.iloc[0]["assets"] == "BTC" and e.llm_failures == 1


def test_validate_rejects_missing_translation():
    with pytest.raises(ValueError):
        validate({"assets": ["BTC"]})
    assert lexicon_label("Ibovespa sobe e renova recorde")["sentiment"] > 0


# histórias entre idiomas

def labeled(rows):
    df = make_frame(rows)
    df["visible_at"] = visible_at(df, LAT)
    df["translated_text"] = df["original_text"]
    return df


def test_same_story_across_languages_counts_once():
    df = labeled([
        raw("SEC approves spot bitcoin ETFs", minutes=0, lang="en"),
        raw("SEC approves spot bitcoin ETFs trading", domain="valor.globo.com", minutes=20, lang="pt"),
        raw("SEC approves bitcoin spot ETFs", domain="yna.co.kr", minutes=40, lang="ko"),
        raw("Petrobras raises dividend payout", domain="valor.globo.com", minutes=50, lang="pt"),
    ])
    s = assign_stories(df)
    etf = s[s["original_text"].str.contains("SEC")]
    assert etf["story_id"].nunique() == 1 and sorted(etf["story_rank"]) == [0, 1, 2]
    assert s["story_id"].nunique() == 2


def test_story_assignment_is_causal():
    rows = [raw(f"fed holds rates steady {i % 3}", minutes=i * 30, url=f"https://reuters.com/{i}") for i in range(8)]
    full = assign_stories(labeled(rows)).set_index("id")["story_id"]
    part = assign_stories(labeled(rows[:4])).set_index("id")["story_id"]
    assert (full.loc[part.index] == part).all()


# features

def feature_items():
    rng = np.random.default_rng(0)
    rows = [raw(f"bitcoin headline {i} " + ("surge" if rng.random() > 0.5 else "drop"), minutes=int(m),
                domain=rng.choice(["reuters.com", "coindesk.com", "valor.globo.com"]))
            for i, m in enumerate(sorted(rng.integers(0, 60 * 30, 80)))]
    df = labeled(rows)
    df["sentiment"] = np.where(df["original_text"].str.contains("surge"), 0.7, -0.7)
    df["confidence"] = 0.8
    df["assets"] = "BTC"
    return assign_stories(df)


def test_features_are_causal_truncation_audit():
    items = feature_items()
    ts = pd.Series(pd.date_range(T0 - pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=32), freq="15min", tz="UTC"))
    calendar = pd.DataFrame({"event": ["FOMC"], "scheduled_at": [T0 + pd.Timedelta(hours=20)], "source": ["x"],
                             "approximate": [False], "detail": [""]})

    def build(t, it):
        return build_features(t, "15m", assign_stories(it.drop(columns=["story_id", "story_rank"])), "BTC", calendar)

    assert truncation_audit(build, ts, items, lambda t: t + pd.Timedelta(minutes=15), samples=40).empty
    feats = build(ts, items)
    first = feats.iloc[0]
    assert first["news_count_24h"] == 0  # antes da primeira notícia
    assert feats["minutes_to_next_event"].iloc[0] == pytest.approx((20 * 60 + 120) - 15)
    sens = shift_sensitivity(build, ts, items, pd.Timedelta(minutes=5))
    assert sens.loc["news_count_1h", "changed_frac"] > 0


def test_future_item_does_not_change_past_features():
    items = feature_items()
    ts = pd.Series(pd.date_range(T0, periods=8, freq="1h", tz="UTC"))
    base = build_features(ts, "1h", items, "BTC")
    bomb = labeled([raw("bitcoin crash bomb", minutes=60 * 7 + 30)])
    bomb = bomb.assign(sentiment=-1.0, confidence=1.0, assets="BTC", story_id="bomb", story_rank=0)
    after = build_features(ts, "1h", pd.concat([items, bomb], ignore_index=True), "BTC")
    decided_before = ts + pd.Timedelta(hours=1) < bomb["visible_at"].iloc[0]
    pd.testing.assert_frame_equal(base[decided_before.to_numpy()], after[decided_before.to_numpy()])


# reputação

def outcomes(source, successes, times):
    return pd.DataFrame({"source": source, "kind": "direction", "success": successes, "resolved_at": times,
                         "item_id": [f"i{k}" for k in range(len(times))], "weight": 1.0})


def test_future_accuracy_cannot_leak_into_the_past():
    t = pd.Timestamp("2024-03-01", tz="UTC")
    future = outcomes("oracle.com", [True] * 50, pd.date_range(t, periods=50, freq="1h", tz="UTC"))  # começa em t
    rep = Reputation(future, tiers={"oracle.com": None})
    row = rep.as_of(t).query("kind == 'direction'")
    assert row.empty or row["n_eff"].iloc[0] == 0  # nada resolvido antes de t (estrito)
    prior = PRIORS[None][0] / sum(PRIORS[None])
    assert rep.scores(t, ["oracle.com"])["score"].iloc[0] == pytest.approx(prior)
    later = rep.scores(t + pd.Timedelta(days=3), ["oracle.com"])["score"].iloc[0]
    assert later > prior  # depois de resolver, aprende


def test_prior_by_tier_shrinkage_and_decay():
    t = pd.Timestamp("2024-06-01", tz="UTC")
    rep = Reputation(outcomes("x.com", [False] * 3, [t - pd.Timedelta(days=1)] * 3), tiers={"x.com": 1, "b.io": 3})
    assert rep.scores(t, ["x.com"])["score"].iloc[0] > rep.scores(t, ["b.io"])["score"].iloc[0]
    few = rep.as_of(t).query("source == 'x.com' and kind == 'direction'")["mean"].iloc[0]
    assert few > 0.5  # 3 erros ainda não derrubam um nível 1 (poucas amostras)
    old = Reputation(outcomes("x.com", [False] * 40, [t - pd.Timedelta(days=720)] * 40), tiers={"x.com": 1})
    new = Reputation(outcomes("x.com", [False] * 40, [t - pd.Timedelta(days=1)] * 40), tiers={"x.com": 1})
    assert old.scores(t, ["x.com"])["score"].iloc[0] > new.scores(t, ["x.com"])["score"].iloc[0]


def test_manipulation_guard_flags_one_sided_burst():
    t = T0 + pd.Timedelta(hours=10)
    burst = pd.DataFrame({"domain": "good.com", "sentiment": [0.9] * 6,
                          "visible_at": pd.date_range(T0, periods=6, freq="20min", tz="UTC")})
    history = outcomes("good.com", [True] * 30, [T0 - pd.Timedelta(days=5)] * 30)
    calm = Reputation(history, tiers={"good.com": 1})
    flagged = Reputation(history, tiers={"good.com": 1}, items=burst)
    assert flagged.flagged(t) == {"good.com"}
    assert flagged.weights(t, ["good.com"]).iloc[0] < calm.weights(t, ["good.com"]).iloc[0]
    assert flagged.flagged(T0) == set()  # antes da rajada não há sinal


def test_cap_shares_and_exploration_floor():
    capped = cap_shares(pd.Series({"a": 10.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}), 0.25)
    assert (capped / capped.sum()).max() <= 0.25 + 1e-9 and capped.sum() == pytest.approx(14.0)
    rep = Reputation(outcomes("bad.com", [False] * 5, [T0] * 5), tiers={})
    w = rep.weights(T0 + pd.Timedelta(hours=1), ["bad.com", "brand-new.com"])
    assert (w >= ReputationConfig().explore_floor).all()


def test_priority_spends_budget_mostly_on_reputable_sources():
    t = T0 + pd.Timedelta(days=1)
    hist = pd.concat([outcomes("good.com", [True] * 40, [T0] * 40), outcomes("bad.com", [False] * 40, [T0] * 40)])
    rep = Reputation(hist, tiers={})
    cands = pd.DataFrame({"domain": ["good.com"] * 10 + ["bad.com"] * 10 + ["new.com"] * 10})
    chosen = rep.priority(t, cands, budget=10, rng=random.Random(1))
    picked = cands.loc[chosen, "domain"].value_counts()
    assert len(chosen) == 10 and picked.get("good.com", 0) >= 8


def test_direction_and_story_outcomes_resolve_after_horizon():
    items = labeled([raw("bitcoin surges", minutes=0), raw("bitcoin surges again", domain="coindesk.com", minutes=10)])
    items = items.assign(sentiment=0.8, confidence=0.9, assets="BTC")
    prices = {"BTC": pd.Series([100.0, 101.0, 103.0],
                               index=pd.DatetimeIndex([T0, T0 + pd.Timedelta(hours=4), T0 + pd.Timedelta(hours=5)]))}
    d = direction_outcomes(items, prices, pd.Timedelta(hours=4))
    assert len(d) == 2 and d["success"].all()
    assert (d["resolved_at"].to_numpy() == (items["visible_at"] + pd.Timedelta(hours=4)).to_numpy()).all()
    s = story_outcomes(assign_stories(items.assign(translated_text="bitcoin surges")))
    assert set(s["kind"]) == {"confirmation", "timeliness"}
    assert not s.query("kind == 'timeliness'")["success"].iloc[0]  # coindesk veio depois da Reuters


# fontes externas (sem rede)

def test_gdelt_backs_off_on_rate_limit_message():
    replies = iter(["Please limit requests to one every 5 seconds", json.dumps({"articles": [
        {"url": "https://reuters.com/a", "title": "Bitcoin  ETF", "seendate": "20240110T120000Z",
         "domain": "reuters.com", "language": "English"}]})])
    slept = []
    g = Gdelt(get=lambda url: next(replies), sleep=slept.append, clock=lambda: 0.0)
    rows = g.backfill("BTC", T0, T0 + pd.Timedelta(hours=1), pd.Timedelta(hours=1))
    assert g.requests == 2 and any(s >= 12 for s in slept)
    assert rows[0]["published_at"] == pd.Timestamp("2024-01-10 12:00", tz="UTC")
    assert rows[0]["original_lang"] == "en" and rows[0]["original_text"] == "Bitcoin ETF"


def test_fomc_parser_and_payroll_rule():
    html = ('<h4><a id="1">2024 FOMC Meetings</a></h4>'
            '<div class="fomc-meeting__month col"><strong>Apr/May</strong></div>'
            '<div class="fomc-meeting__date col">30-1</div>'
            '<div class="fomc-meeting__month col"><strong>June</strong></div>'
            '<div class="fomc-meeting__date col">11-12*</div>')
    f = cal.parse_fomc(html)
    assert f["scheduled_at"].tolist() == [pd.Timestamp("2024-05-01 18:00", tz="UTC"), pd.Timestamp("2024-06-12 18:00", tz="UTC")]
    p = cal.payroll("2024-03-01", "2024-03-31")
    assert p["scheduled_at"].iloc[0] == pd.Timestamp("2024-03-01 13:30", tz="UTC")  # 1º de março de 2024 foi sexta


def test_rss_parse_and_future_clock_clipped():
    xml = b"""<rss><channel>
      <item><title>Copom corta Selic</title><link>https://valor.globo.com/a</link>
        <pubDate>Wed, 10 Jan 2024 11:00:00 GMT</pubDate></item>
      <item><title>Relogio adiantado</title><link>https://valor.globo.com/b</link>
        <pubDate>Wed, 10 Jan 2024 18:00:00 GMT</pubDate></item>
    </channel></rss>"""
    rows = parse_feed(xml, "valor", "pt", now=T0)
    assert rows[0]["published_at"] == pd.Timestamp("2024-01-10 11:00", tz="UTC")
    assert rows[1]["published_at"] == T0 and rows[1]["backfilled"] is False


# contrato news_features

from trader.news.features import news_features  # noqa: E402


def seed_store(root, rows):
    NewsStore(root).write(make_frame(rows))


def test_news_features_contract_and_no_future_leak(tmp_path):
    rows = [raw(f"bitcoin rally {i} surge", minutes=i * 45, url=f"https://reuters.com/{i}") for i in range(12)]
    seed_store(tmp_path, rows)
    idx = pd.date_range(T0, T0 + pd.Timedelta(hours=10), freq="5min", tz="UTC")
    before = news_features("BTC/USDT", "5m", idx, data=tmp_path)
    assert before.index.equals(idx) and all(c.startswith("news_") for c in before.columns)
    for col in ("news_count", "news_sentiment", "news_confidence", "news_spread", "news_minutes_to_next_event",
                "news_event_other_minutes"):
        assert col in before.columns
    assert before["news_count"].iloc[0] == 0 and before["news_count"].max() > 0

    bomb_at = T0 + pd.Timedelta(hours=6)
    seed_store(tmp_path, [raw("bitcoin crash hack plunge", minutes=6 * 60, url="https://reuters.com/bomb")])
    after = news_features("BTC/USDT", "5m", idx, data=tmp_path)  # cache invalida pelo mtime
    earlier = idx < bomb_at + LAT
    pd.testing.assert_frame_equal(before[earlier], after[earlier])
    assert (after.loc[~earlier, "news_count_1h"] >= before.loc[~earlier, "news_count_1h"]).all()
    assert after.loc[idx >= bomb_at + LAT, "news_sentiment_1h"].iloc[0] < before.loc[idx >= bomb_at + LAT, "news_sentiment_1h"].iloc[0]


def test_news_features_daily_grid_and_unknown_asset(tmp_path):
    seed_store(tmp_path, [raw("Petrobras lucro recorde", domain="valor.globo.com", lang="pt", assets="PETR4",
                              url=f"https://valor.globo.com/{i}", minutes=i * 600) for i in range(5)])
    idx = pd.date_range("2024-01-09", periods=6, freq="1D", tz="UTC")
    petr = news_features("PETR4", "1d", idx, data=tmp_path, use_reputation=False)
    assert "news_count_3d" in petr.columns and petr["news_count_3d"].max() == 5
    itub = news_features("ITUB4", "1d", idx, data=tmp_path, use_reputation=False)
    assert itub["news_count"].eq(0).all() and itub.index.equals(idx)
    with pytest.raises(ValueError):
        news_features("PETR4", "1d", idx.tz_localize(None), data=tmp_path)


def test_items_seen_long_after_publication_are_historical(tmp_path):
    df = make_frame([
        raw("fresh", backfilled=False, seen_delay=pd.Timedelta(minutes=10)),
        raw("old rss item", backfilled=False, seen_delay=pd.Timedelta(hours=8)),
    ])
    flags = dict(zip(df["original_text"], df["backfilled"]))
    assert flags == {"fresh": False, "old rss item": True}
    vis = dict(zip(df["original_text"], visible_at(df, LAT)))
    assert vis["old rss item"] == T0 + LAT  # sem falsa chegada 8h depois

    # dados gravados antes da regra são corrigidos no lugar
    store = NewsStore(tmp_path)
    legacy = df.assign(backfilled=False)
    store.root.mkdir(parents=True)
    legacy.to_parquet(store.root / "2024-01.parquet", index=False)
    assert store.fix_historical_flags() == 1
    assert store.read_all().set_index("original_text")["backfilled"].to_dict() == flags
