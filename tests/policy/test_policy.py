import json
import time

import pandas as pd
import pytest

from trader.daytrade.broker import FuturesPaperBroker
from trader.policy.executor import PolicyExecutor
from trader.policy.schema import SAFE, InvalidPolicy, parse_policy
from trader.policy.writer import load_current, policy_to_dict, save, write_policy

SYMS = ["SOL/USDT", "ETH/USDT"]
T0 = pd.Timestamp("2026-01-01 12:00", tz="UTC")


def raw(**kw):
    base = {
        "valid_minutes": 30, "kill": False, "daily_loss_limit": 0.03,
        "symbols": {"SOL/USDT": {"bias": "long", "setups": ["*"], "max_leverage": 2, "risk_per_trade": 0.01,
                                 "stop": 0.02, "take_profit_r": 2.0, "lock_at": 1.0, "lock_keep": 0.5}},
        "reactions": [{"when": {"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.7},
                       "do": "flatten", "minutes": 60}],
        "rationale": "teste", "sources": [],
    }
    base.update(kw)
    return base


def executor(policy_raw=None):
    ex = PolicyExecutor(FuturesPaperBroker(1000.0))
    ex.set_policy(parse_policy(policy_raw or raw(), SYMS), T0)
    return ex


def test_policy_cannot_loosen_frozen_limits():
    bad = raw(daily_loss_limit=0.5, symbols={"SOL/USDT": {"bias": "long", "max_leverage": 50, "stop": 0.2},
                                             "DOGE/USDT": {"bias": "long"}})
    with pytest.raises(InvalidPolicy) as e:
        parse_policy(bad, SYMS)
    msg = " ".join(e.value.errors)
    assert "daily_loss_limit" in msg and "max_leverage" in msg and "stop" in msg and "DOGE" in msg


def test_unknown_setup_and_action_rejected():
    bad = raw(symbols={"SOL/USDT": {"bias": "sideways", "setups": ["astrologia"]}},
              reactions=[{"when": {"kind": "meteoro"}, "do": "rezar"}])
    with pytest.raises(InvalidPolicy) as e:
        parse_policy(bad, SYMS)
    assert len(e.value.errors) >= 4


def test_bias_and_setup_gate():
    ex = executor(raw(symbols={"SOL/USDT": {"bias": "long", "setups": ["momentum"], "stop": 0.02}}))
    assert not ex.on_signal(T0, "SOL/USDT", -1, "momentum", 100).accepted  # vendido não permitido
    assert not ex.on_signal(T0, "SOL/USDT", 1, "orb", 100).accepted  # setup não permitido
    assert not ex.on_signal(T0, "ETH/USDT", 1, "momentum", 100).accepted  # ativo sem política = flat
    assert ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100).accepted


def test_expired_policy_blocks_entries_but_manages_exits():
    ex = executor()
    assert ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100).accepted
    later = T0 + pd.Timedelta(minutes=31)
    ex.on_price(later, "SOL/USDT", 97.9)  # stop 2% = 98
    assert ex.broker.position("SOL/USDT") is None
    assert ex.on_signal(later, "SOL/USDT", 1, "momentum", 100).reason == "política vencida"


def test_safe_policy_never_enters():
    ex = PolicyExecutor(FuturesPaperBroker(1000.0))
    ex.set_policy(SAFE, T0)
    assert not ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100).accepted


def test_sizing_risks_the_policy_fraction():
    ex = executor()
    d = ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)
    # arrisca 1% de 1000 = 10 até o stop de 2%: qty × 100 × 0.02 = 10 → qty 5
    assert d.qty == pytest.approx(5.0)


def test_profit_lock_on_ticks():
    ex = executor()
    ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)  # stop 98, alvo 104, trava após +2
    for p in (101, 102.5, 103):
        ex.on_price(T0, "SOL/USDT", p)
    assert ex.broker.position("SOL/USDT").stop == pytest.approx(101.5)  # 100 + 0.5 × 3
    ex.on_price(T0, "SOL/USDT", 101.4)
    trade = ex.broker.trades[-1]
    assert trade.action == "stop" and trade.pnl > 0


def test_news_reactions():
    ex = executor(raw(reactions=[
        {"when": {"kind": "hack", "symbols": ["*"], "sentiment": "negative", "min_severity": 0.7}, "do": "flatten"},
        {"when": {"kind": "regulation", "symbols": ["SOL/USDT"], "sentiment": "negative", "min_severity": 0.5},
         "do": "block_entries", "minutes": 20},
    ]))
    ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)
    assert ex.on_news(T0, "hack", ("ETH/USDT",), -0.9, 0.5) == []  # severidade baixa
    assert ex.on_news(T0, "hack", ("ETH/USDT",), 0.5, 0.9) == []  # não é negativa
    assert "zera SOL/USDT" in ex.on_news(T0, "hack", ("ETH/USDT",), -0.9, 0.9)
    assert ex.broker.position("SOL/USDT") is None
    ex.on_news(T0, "regulation", ("SOL/USDT",), -0.6, 0.6)
    assert ex.on_signal(T0 + pd.Timedelta(minutes=10), "SOL/USDT", 1, "momentum", 100).reason == "bloqueado por notícia"
    assert ex.on_signal(T0 + pd.Timedelta(minutes=21), "SOL/USDT", 1, "momentum", 100).accepted


def test_reduce_keeps_fraction():
    ex = executor(raw(reactions=[{"when": {"kind": "macro", "sentiment": "negative", "min_severity": 0.5},
                                  "do": "reduce", "factor": 0.5}]))
    ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)
    ex.on_price(T0, "SOL/USDT", 100)
    ex.on_news(T0, "macro", ("SOL/USDT",), -0.8, 0.8)
    assert ex.broker.position("SOL/USDT").qty == pytest.approx(2.5)


def test_kill_flattens():
    ex = executor()
    ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)
    ex.set_policy(parse_policy(raw(kill=True), SYMS), T0)
    assert ex.broker.position("SOL/USDT") is None
    assert not ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100).accepted


def test_daily_loss_limit_stops_entries():
    ex = executor(raw(daily_loss_limit=0.01, symbols={"SOL/USDT": {"bias": "both", "risk_per_trade": 0.015,
                                                                   "stop": 0.02, "max_leverage": 3}}))
    ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)
    ex.on_price(T0, "SOL/USDT", 97.5)  # stop: perde ~1,5%
    assert ex.on_signal(T0, "SOL/USDT", 1, "momentum", 97.5).reason == "limite de perda do dia"


def test_executor_is_fast():
    ex = executor()
    n = 20000
    t = time.perf_counter()
    ex.on_signal(T0, "SOL/USDT", 1, "momentum", 100)
    for i in range(n):
        ex.on_price(T0, "SOL/USDT", 100 + (i % 50) * 0.01)
    per_tick = (time.perf_counter() - t) / n
    assert per_tick < 1e-3  # bem abaixo de 1 ms por evento


def test_writer_retries_with_errors_then_accepts():
    answers = iter([json.dumps(raw(daily_loss_limit=0.9)), "isto não é json", json.dumps(raw())])
    seen = []

    def chat(messages):
        seen.append(messages[-1]["content"])
        return next(answers)

    policy, report = write_policy({"market": {}}, SYMS, chat=chat, retries=2, use_rules=False)
    assert report["attempts"] == 3 and report["fallback"] is None
    assert "daily_loss_limit" in seen[1]  # o erro voltou para a LLM
    assert policy.symbols["SOL/USDT"].bias == "long"


def test_writer_falls_back_when_llm_is_down():
    def chat(_):
        raise ConnectionError("ollama fora")

    policy, report = write_policy({}, SYMS, chat=chat, use_rules=False)
    assert policy is SAFE and report["fallback"] == "SAFE"
    current = parse_policy(raw(), SYMS)
    policy, _ = write_policy({}, SYMS, current=current, chat=chat, use_rules=False)
    assert policy is current
    # com regras ligadas, a falha da LLM cai na política-base por regras (nunca fica sem política)
    policy, report = write_policy({"market": {"SOL/USDT": {"committee": 1.0, "vol20_annual": 0.5}}}, SYMS, chat=chat)
    assert report["fallback"] == "regras" and policy.symbols["SOL/USDT"].bias == "long"


def test_save_and_reload_roundtrip(tmp_path):
    p = parse_policy(raw(), SYMS)
    save(p, {"attempts": 1}, tmp_path, "2026-01-01")
    assert load_current(tmp_path, SYMS) == p
    assert policy_to_dict(parse_policy(policy_to_dict(p), SYMS)) == policy_to_dict(p)


def test_broad_and_duplicate_reactions():
    r = {"when": {"kind": "hack", "sentiment": "negative", "min_severity": 0.7}, "do": "flatten"}
    assert len(parse_policy(raw(reactions=[r, r, r]), SYMS).reactions) == 1
    broad = {"when": {"kind": "any", "sentiment": "any"}, "do": "tighten_stop"}
    with pytest.raises(InvalidPolicy, match="ampla demais"):
        parse_policy(raw(reactions=[broad]), SYMS)


def test_rules_baseline_reads_the_context():
    from trader.policy.rules import baseline_policy

    ctx = {"market": {"SOL/USDT": {"committee": 1.0, "vol20_annual": 0.5, "ret_7d": 0.02},
                      "ETH/USDT": {"committee": 0.75, "vol20_annual": 0.7, "ret_7d": -0.08}}}
    p = baseline_policy(ctx, SYMS)
    sol, eth = p.symbols["SOL/USDT"], p.symbols["ETH/USDT"]
    assert sol.bias == "long" and sol.risk_per_trade == pytest.approx(0.003) and sol.lock_at == 1.0
    assert eth.bias == "flat"  # comitê 0,75 com vol alta e queda na semana: fora
    assert len(p.reactions) == 7 and p.daily_loss_limit == 0.02


def test_tighten_only_makes_it_more_conservative():
    from trader.policy.rules import baseline_policy, tighten

    ctx = {"market": {"SOL/USDT": {"committee": 1.0, "vol20_annual": 0.3}, "ETH/USDT": {"committee": 1.0, "vol20_annual": 0.3}}}
    base = baseline_policy(ctx, SYMS)
    loose = parse_policy(raw(daily_loss_limit=0.05, symbols={
        "SOL/USDT": {"bias": "both", "max_leverage": 3, "risk_per_trade": 0.02, "stop": 0.05, "setups": ["*"]},
        "ETH/USDT": {"bias": "flat", "risk_per_trade": 0.001, "stop": 0.01}}, reactions=[]), SYMS)
    m = tighten(base, loose)
    sol, eth = m.symbols["SOL/USDT"], m.symbols["ETH/USDT"]
    assert sol.bias == "long" and sol.max_leverage == 1.0 and sol.risk_per_trade == 0.005 and sol.stop == base.symbols["SOL/USDT"].stop
    assert sol.setups == base.symbols["SOL/USDT"].setups  # "*" da LLM não abre setups além da base
    assert eth.bias == "flat" and eth.risk_per_trade == 0.001  # a LLM pode apertar
    assert m.daily_loss_limit == 0.02 and len(m.reactions) == 7


def test_json_schema_and_lenient_parse():
    from trader.policy.schema import json_schema

    sch = json_schema(SYMS)
    assert sch["properties"]["symbols"]["required"] == SYMS
    assert "neutral" not in sch["properties"]["reactions"]["items"]["properties"]["when"]["properties"]["sentiment"]["enum"]
    lenient = parse_policy(raw(daily_loss_limit=0.9, symbols={"SOL/USDT": {"bias": "long", "max_leverage": 50, "stop": 0.0}}),
                           SYMS, lenient=True)
    assert lenient.daily_loss_limit == 0.05 and lenient.symbols["SOL/USDT"].max_leverage == 3.0
    assert lenient.symbols["SOL/USDT"].stop == 0.001
    with pytest.raises(InvalidPolicy):
        parse_policy(raw(daily_loss_limit=0.9), SYMS)  # modo estrito continua estrito
