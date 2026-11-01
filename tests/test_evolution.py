import json
import random

import numpy as np
import pandas as pd
import pytest

from trader.backtest import run_backtest, walk_forward_windows
from trader.evolution.creator import LLMCreator, RuleCreator
from trader.evolution.fitness import Cause, Life, SelectionRules, Verdict, WindowStats, judge
from trader.evolution.genome import Genome, GenomeEngine, mutate, random_genome
from trader.evolution.population import Population, PopulationConfig
from trader.evolution.registry import Registry

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def random_walk(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame({
        "ts": pd.date_range(T0, periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) * 1.003,
        "low": np.minimum(open_, close) * 0.997, "close": close, "volume": 1.0,
    })


def window(ret=0.0, gross=None, dd=0.0, trades=0, **ctx):
    return WindowStats(T0, ret, ret if gross is None else gross, dd, trades, **ctx)


# genoma

def test_random_and_mutated_genomes_stay_valid():
    rng = random.Random(3)
    for _ in range(200):
        g = mutate(random_genome(rng), rng, scale=1.0)
        assert Genome.from_dict(json.loads(json.dumps(g.to_dict()))) == g
        if g.strategy == "ma_cross":
            assert g.params["fast"] < g.params["slow"]


def test_genome_rejects_wrong_params():
    with pytest.raises(ValueError):
        Genome("ma_cross", {"lookback": 10})


# backtest

def test_backtest_decides_on_close_and_fills_next_open():
    df = random_walk(300)
    g = Genome("ma_cross", {"fast": 5, "slow": 20})
    res = run_backtest(GenomeEngine(g), df, "X", "4h", start=df["ts"].iloc[50])
    assert len(res.equity) == 250
    opens = set(df["open"].round(8))
    assert all(round(f.price, 8) in opens for f in res.fills)
    assert res.fees > 0 and res.gross_return > res.total_return


def test_flat_engine_keeps_cash():
    df = random_walk(100)
    g = Genome("rsi_revert", {"period": 14, "low": 10, "high": 90}, vol_filter=0.002)  # nunca entra
    res = run_backtest(GenomeEngine(g), df, "X", "4h", start=df["ts"].iloc[40])
    assert res.total_return == 0 and res.n_trades == 0


def test_walk_forward_windows():
    w = walk_forward_windows(T0, T0 + pd.Timedelta(days=10), pd.Timedelta(days=3))
    assert len(w) == 3 and w[1] == (T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=6))


# veredito

RULES = SelectionRules(max_drawdown=0.15, min_trades=8, min_age_for_edge=3, max_age=5, thrive_streak=2)


def flat(n):
    """Referência que não ganhou nem perdeu, com n janelas."""
    return Life([window(dd=0.05)] * n)


def test_judge_drawdown_kills_immediately():
    assert judge(Life([window(dd=0.2)]), flat(1), RULES) == (Verdict.DIE, Cause.DRAWDOWN)


def test_judge_distinguishes_fees_from_no_edge():
    fees = Life([window(ret=-0.01, gross=0.01, trades=5)] * 3)
    edge = Life([window(ret=-0.02, gross=-0.01, trades=5)] * 3)
    assert judge(fees, flat(3), RULES) == (Verdict.DIE, Cause.FEES)
    assert judge(edge, flat(3), RULES) == (Verdict.DIE, Cause.NEGATIVE_EDGE)


def test_judge_needs_enough_trades_and_a_streak_to_thrive():
    lucky = Life([window(ret=0.3, dd=0.05, trades=2)] * 2)
    one_hit = Life([window(ret=0.2, dd=0.05, trades=10)])
    good = Life([window(ret=0.05, dd=0.03, trades=5)] * 2)
    assert judge(lucky, flat(2), RULES)[0] is Verdict.LIVE
    assert judge(one_hit, flat(1), RULES)[0] is Verdict.LIVE  # uma janela só ainda é sorte
    assert judge(good, flat(2), RULES)[0] is Verdict.THRIVE


def test_judge_compares_risk_adjusted_not_raw_return():
    careful = Life([window(ret=0.05, dd=0.02, trades=5)] * 2)  # nota 2,5 por janela
    reckless_ref = Life([window(ret=0.10, dd=0.10)] * 2)  # mais retorno, nota 1,0
    assert judge(careful, reckless_ref, RULES)[0] is Verdict.THRIVE


def test_judge_stagnation():
    assert judge(Life([window()] * 5), flat(5), RULES) == (Verdict.DIE, Cause.STAGNATION)


# Criador

def autopsy(cause, **ctx):
    return {"cause": cause, "life": {"trades": 10}, "last_window": ctx}


def test_rule_creator_directed_fixes():
    rng = random.Random(0)
    g = Genome("ma_cross", {"fast": 10, "slow": 30})
    s, why = RuleCreator().successor(g, autopsy("drawdown", dd_high_vol=True, dd_market_down=True, median_range=0.02), rng)
    assert s.vol_filter is not None and s.trend_filter is not None and "volatilidade" in why
    s, _ = RuleCreator().successor(g, autopsy("fees"), rng)
    assert s.cooldown >= 3 and s.params["slow"] > 30


def test_llm_creator_uses_llm_answer():
    answer = {"genome": {"strategy": "breakout", "params": {"lookback": 40}, "size": 0.5,
                         "vol_filter": None, "trend_filter": 150, "cooldown": 2}, "why": "trend filter"}
    fake = lambda url, payload, timeout: {"choices": [{"message": {"content": "ok " + json.dumps(answer)}}]}
    creator = LLMCreator("BTC/USDT", "4h", post=fake)
    g, why = creator.successor(Genome("ma_cross", {"fast": 10, "slow": 30}), autopsy("drawdown"), random.Random(0))
    assert g.strategy == "breakout" and g.trend_filter == 150 and why.startswith("LLM")


def test_llm_creator_falls_back_to_rules():
    def down(url, payload, timeout):
        raise ConnectionError("ollama off")
    creator = LLMCreator("BTC/USDT", "4h", post=down)
    g, why = creator.successor(Genome("ma_cross", {"fast": 10, "slow": 30}), autopsy("fees"), random.Random(0))
    assert "ConnectionError" in why and creator.failures == 1


# população

def test_population_lifecycle_respects_slots_and_records_graveyard():
    df = random_walk(3000, seed=7)
    reg = Registry(":memory:")
    cfg = PopulationConfig(initial=8, max_slots=12, rules=SelectionRules(min_trades=3, max_age=4))
    pop = Population(df, "X", "4h", reg, RuleCreator(), cfg, seed=1)
    pop.seed()
    start = df["ts"].iloc[300]
    for a, b in walk_forward_windows(start, df["ts"].iloc[-1], pd.Timedelta(days=30)):
        report = pop.step(a, b)
        assert 8 <= report.alive <= 12
    assert sum(reg.cause_counts().values()) > 0
    # todo morto (exceto os removidos para dar vaga) deixou um sucessor
    succ = reg.db.execute("SELECT COUNT(*) FROM agents WHERE origin = 'successor'").fetchone()[0]
    culled = reg.cause_counts().get("culled", 0)
    assert succ == sum(reg.cause_counts().values()) - culled
    chain = reg.ancestry(pop.alive[-1].id)
    assert chain[-1]["origin"] == "seed"


def test_signal_check_verdict_detects_real_difference():
    from trader.training.signal_check import verdict
    rng = random.Random(0)
    noise = [{"rules": rng.gauss(0, 1), "mutation0": rng.gauss(0, 1), "mutation1": rng.gauss(0, 1)} for _ in range(200)]
    better = [{**s, "rules": s["rules"] + 1.0} for s in noise]
    assert not verdict(noise).has_signal
    assert verdict(better).has_signal and verdict(better).rules_vs_random > 0.7


def test_successors_are_designed_in_parallel_and_deterministic():
    import threading
    import time as _time

    class SlowCreator(RuleCreator):
        def __init__(self):
            self.threads = set()

        def successor(self, genome, autopsy, rng):
            self.threads.add(threading.get_ident())
            _time.sleep(0.05)
            return super().successor(genome, autopsy, rng)

    def run(threads):
        df = random_walk(2000, seed=11)
        creator = SlowCreator()
        cfg = PopulationConfig(initial=12, max_slots=16, rules=SelectionRules(min_trades=1, max_age=2), creator_threads=threads)
        pop = Population(df, "X", "4h", Registry(":memory:"), creator, cfg, seed=4)
        pop.seed()
        start = df["ts"].iloc[300]
        for a, b in walk_forward_windows(start, start + pd.Timedelta(days=150), pd.Timedelta(days=30)):
            pop.step(a, b)
        return [a.genome for a in pop.alive], creator.threads

    serial, _ = run(1)
    parallel, used = run(4)
    assert serial == parallel  # mesma população, com 1 ou 4 threads
    assert len(used) > 1


def test_thrivers_enter_hall_and_children_use_success_report():
    prompts = []

    def fake_post(url, payload, timeout):
        prompts.append(payload["messages"][0]["content"])
        answer = {"genome": {"strategy": "ma_cross", "params": {"fast": 9, "slow": 60}}, "why": "keep trend filter"}
        return {"choices": [{"message": {"content": json.dumps(answer)}}]}

    df = random_walk(3000, seed=21)
    reg = Registry(":memory:")
    creator = LLMCreator("X", "4h", post=fake_post, graveyard=lambda: reg.graveyard(5), hall=lambda: reg.hall(5),
                         library=lambda q: [{"book": "Livro Teste", "page": 42, "text": "cut losses early"}])
    cfg = PopulationConfig(initial=10, max_slots=16, rules=SelectionRules(min_trades=1, max_age=6, thrive_streak=1))
    pop = Population(df, "X", "4h", reg, creator, cfg, seed=3)
    pop.seed()
    start = df["ts"].iloc[300]
    for a, b in walk_forward_windows(start, start + pd.Timedelta(days=300), pd.Timedelta(days=30)):
        pop.step(a, b)
    hall = reg.hall(10)
    assert hall and all("winning_context" in h for h in hall)
    child_prompts = [p for p in prompts if "THRIVED" in p]
    assert child_prompts and "Success report" in child_prompts[0]
    design = [p for p in prompts if "playbook" not in p]  # prompts de sucessor e de filho
    assert all("[Livro Teste p.42]" in p for p in design)  # trechos do RAG entram nos dois tipos de prompt
    notes = [r[0] for r in reg.db.execute("SELECT note FROM agents WHERE origin = 'child'")]
    assert any("LLM (sucesso)" in n for n in notes)


def test_basket_evaluation_trades_every_asset():
    a, b = random_walk(1500, seed=31), random_walk(1500, seed=32)
    g = [Genome("ma_cross", {"fast": 5, "slow": 20})]
    single = Population(a, "A", "4h", Registry(":memory:"), RuleCreator())
    basket = Population({"A": a, "B": b}, "A", "4h", Registry(":memory:"), RuleCreator())
    start = a["ts"].iloc[300]
    end = start + pd.Timedelta(days=60)
    one = single._evaluate_all(g, start, end)[0]
    both = basket._evaluate_all(g, start, end)[0]
    only_b = Population(b, "B", "4h", Registry(":memory:"), RuleCreator())._evaluate_all(g, start, end)[0]
    assert both.trades == one.trades + only_b.trades
    assert both.ret == pytest.approx((one.ret + only_b.ret) / 2)


def test_best_of_n_keeps_the_screened_winner_and_logs_proposals():
    df = random_walk(2500, seed=41)
    reg = Registry(":memory:")
    cfg = PopulationConfig(initial=10, max_slots=14, rules=SelectionRules(min_trades=1, max_age=3), candidates=4)
    pop = Population(df, "X", "4h", reg, RuleCreator(), cfg, seed=5)
    pop.seed()
    start = df["ts"].iloc[300]
    for a, b in walk_forward_windows(start, start + pd.Timedelta(days=240), pd.Timedelta(days=30)):
        pop.step(a, b)
    rows = reg.db.execute("SELECT candidates, chosen FROM proposals").fetchall()
    assert rows
    for candidates, chosen in rows:
        scores = [c["score"] for c in json.loads(candidates)]
        assert len(scores) == 4 and scores[chosen] == max(scores)


def test_lessons_are_distilled_and_injected_into_prompts():
    prompts = []

    def post(url, payload, timeout):
        text = payload["messages"][0]["content"]
        prompts.append(text)
        if "playbook" in text:
            body = {"lessons": ["Agents with a trend_filter survived falling markets."]}
        elif "candidates" in text:
            g = {"strategy": "ma_cross", "params": {"fast": 8, "slow": 40}}
            body = {"candidates": [{"genome": g, "why": "a"}, {"genome": {**g, "size": 0.5}, "why": "b"}]}
        else:
            body = {"genome": {"strategy": "ma_cross", "params": {"fast": 8, "slow": 40}}, "why": "x"}
        return {"choices": [{"message": {"content": json.dumps(body)}}]}

    df = random_walk(2500, seed=43)
    reg = Registry(":memory:")
    creator = LLMCreator("X", "4h", post=post, graveyard=lambda: reg.graveyard(5), hall=lambda: reg.hall(5))
    cfg = PopulationConfig(initial=10, max_slots=14, rules=SelectionRules(min_trades=1, max_age=3),
                           candidates=2, lessons_every=2)
    pop = Population(df, "X", "4h", reg, creator, cfg, seed=6)
    creator.lessons = lambda: pop.lessons
    pop.seed()
    start = df["ts"].iloc[300]
    for a, b in walk_forward_windows(start, start + pd.Timedelta(days=240), pd.Timedelta(days=30)):
        pop.step(a, b)
    assert pop.lessons == ["Agents with a trend_filter survived falling markets."]
    assert reg.meta()["lessons"]["lessons"] == pop.lessons
    assert any("Lessons learned so far" in p and "trend_filter survived" in p for p in prompts)
