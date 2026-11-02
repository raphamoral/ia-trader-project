import gzip
import json

import pytest

from trader.lob import record as rec
from trader.lob.depth import DepthBook, SyncState, to_units
from trader.lob.queue import QueuePosition
from trader.lob.sim import Costs, Diff, Snapshot, Trade, run_study, signed_cost_bps

SNAP = {"lastUpdateId": 100, "bids": [["100.0", "5"], ["99.0", "3"]], "asks": [["101.0", "4"], ["102.0", "2"]]}


def diff(U, u, pu, b=(), a=()):
    return {"U": U, "u": u, "pu": pu, "b": [list(x) for x in b], "a": [list(x) for x in a]}


def synced():
    book = DepthBook("0.1", "1")
    book.on_diff(diff(95, 101, 94, b=[("100.0", "6")]))
    assert book.on_snapshot(dict(SNAP))
    return book


def test_to_units_is_exact_and_refuses_off_grid():
    assert to_units("0.3", "0.1") == 3  # em float, 0.3 / 0.1 daria 2.9999...
    with pytest.raises(ValueError):
        to_units("0.35", "0.1")


def test_no_answer_before_snapshot():
    book = DepthBook("0.1", "1")
    book.on_diff(diff(1, 2, 0, b=[("100.0", "1")]))
    assert book.best_bid() is None and book.mid() is None


def test_snapshot_drops_old_events_and_applies_the_bridge():
    book = DepthBook("0.1", "1")
    book.on_diff(diff(80, 90, 79, b=[("50.0", "1")]))  # anterior ao snapshot: descartado
    book.on_diff(diff(95, 101, 90, b=[("100.0", "6")]))
    assert book.on_snapshot(dict(SNAP))
    assert book.level(True, 1000) == 6 and book.level(True, 500) == 0


def test_stale_snapshot_is_refused():
    book = DepthBook("0.1", "1")
    book.on_diff(diff(200, 210, 199))
    assert not book.on_snapshot(dict(SNAP))
    assert book.state is SyncState.NEEDS_SNAPSHOT


def test_events_after_the_bridge_are_applied_in_sequence():
    book = DepthBook("0.1", "1")
    book.on_diff(diff(95, 101, 90))
    book.on_diff(diff(102, 103, 101, a=[("101.0", "0")]))
    assert book.on_snapshot(dict(SNAP))
    assert book.best_ask() == 1020


def test_snapshot_arriving_before_its_bridge_event_stays_synced():
    # visto em dados reais: o REST responde antes do websocket entregar o evento que o cobre
    book = DepthBook("0.1", "1")
    book.on_diff(diff(90, 95, 89))
    assert book.on_snapshot(dict(SNAP))
    assert book.on_diff(diff(99, 102, 98, b=[("100.0", "7")]))
    assert book.level(True, 1000) == 7 and book.gaps == 0
    assert book.on_diff(diff(103, 104, 102, b=[("100.0", "8")]))


def test_first_event_after_snapshot_must_still_bridge():
    book = DepthBook("0.1", "1")
    book.on_diff(diff(90, 95, 89))
    assert book.on_snapshot(dict(SNAP))
    assert not book.on_diff(diff(150, 151, 149))
    assert book.gaps == 1 and not book.ready


def test_gap_invalidates_book_until_new_snapshot():
    book = synced()
    assert book.on_diff(diff(102, 103, 101, b=[("100.5", "1")]))
    assert not book.on_diff(diff(110, 111, 109, b=[("100.6", "1")]))
    assert book.state is SyncState.NEEDS_SNAPSHOT and book.gaps == 1 and book.mid() is None
    assert book.on_snapshot({"lastUpdateId": 110, "bids": [["100.0", "1"]], "asks": [["101.0", "1"]]})
    assert book.level(True, 1006) == 1  # o evento que revelou o buraco não se perdeu


def test_redelivery_is_ignored():
    book = synced()
    book.on_diff(diff(102, 103, 101, b=[("100.0", "9")]))
    assert not book.on_diff(diff(102, 103, 101, b=[("100.0", "1")]))
    assert book.level(True, 1000) == 9 and book.gaps == 0


def test_crossed_book_forces_resync():
    book = synced()
    assert not book.on_diff(diff(102, 103, 101, b=[("101.5", "1")]))
    assert book.crossed == 1 and not book.ready


def test_spot_sequence_rule():
    book = DepthBook("0.1", "1", venue="spot")
    book.on_diff({"U": 100, "u": 102, "b": [], "a": []})
    assert book.on_snapshot(dict(SNAP))
    assert book.on_diff({"U": 103, "u": 104, "b": [["100.0", "1"]], "a": []})
    assert not book.on_diff({"U": 106, "u": 107, "b": [], "a": []})
    assert book.gaps == 1


def test_walk_uses_vwap_and_does_not_change_book():
    book = synced()
    assert book.walk(True, 5) == (5, (4 * 1010 + 1 * 1020) / 5)
    assert book.walk(False, 100)[0] == 9
    assert book.level(False, 1010) == 4


def test_queue_trade_consumes_front_then_fills_us():
    q = QueuePosition(True, 100, 5, ahead=10.0, level_total=10)
    assert q.on_trade(100, 8, True) == 0 and q.ahead == 2
    assert q.on_trade(100, 4, True) == 2 and q.remaining == 3


def test_queue_ignores_same_side_aggression_and_other_prices():
    q = QueuePosition(True, 100, 5, ahead=1.0, level_total=1)
    assert q.on_trade(100, 9, False) == 0  # comprador agrediu: não atinge bid
    assert q.on_trade(101, 9, True) == 0  # negócio acima do nosso bid


def test_queue_trade_through_fills_everything():
    q = QueuePosition(True, 100, 5, ahead=99.0, level_total=99)
    assert q.on_trade(99, 1, True) == 5 and q.remaining == 0


def test_queue_ask_is_symmetric():
    q = QueuePosition(False, 100, 5, ahead=0.0, level_total=0)
    assert q.on_trade(100, 9, True) == 0  # vendedor agrediu: não atinge ask
    assert q.on_trade(100, 3, False) == 3
    assert q.on_trade(101, 1, False) == 2


def test_trade_is_not_double_counted_as_cancel():
    q = QueuePosition(True, 100, 1, ahead=10.0, level_total=20)
    q.on_trade(100, 4, True)
    q.on_level(16)
    assert q.ahead == 6


def test_cancels_shrink_queue_in_part_and_additions_go_behind():
    q = QueuePosition(True, 100, 1, ahead=10.0, level_total=20)
    q.on_level(30)
    assert q.ahead == 10
    q.on_level(20)
    assert 5 < q.ahead < 10


def test_queue_never_exceeds_level():
    q = QueuePosition(True, 100, 1, ahead=10.0, level_total=10)
    q.on_level(0)
    assert q.ahead == 0


def test_signed_cost():
    assert signed_cost_bps(True, 101, 100, 5) == pytest.approx(105)
    assert signed_cost_bps(False, 101, 100, 5) == pytest.approx(-95)


def stream(trades=(), tail_ms=1000):
    """Livro parado em 100,0/100,1 com negócios opcionais, atualizado a cada 100 ms."""
    snap = {"lastUpdateId": 1, "bids": [["100.0", "10"]], "asks": [["100.1", "10"]]}
    yield Snapshot(0, snap)
    by_ts = dict(trades)
    u = 1
    for ts in range(100, tail_ms, 100):
        yield Diff(ts, diff(u + 1, u + 1, u))
        u += 1
        if ts in by_ts:
            yield by_ts[ts]


def book_for_sim():
    return DepthBook("0.1", "1")


def test_study_rejects_overlapping_decisions():
    with pytest.raises(ValueError):
        run_study([], book_for_sim(), 1, every_ms=100, patience_ms=100)


def test_unfilled_passive_crosses_late_and_pays_taker_fee():
    study = run_study(stream(tail_ms=2000), book_for_sim(), 1, every_ms=10_000, patience_ms=500,
                      latency_ms=100, markout_ms=100, costs=Costs(2, 5))
    first = study.decisions[0]
    assert first.is_buy and not first.passive_filled
    assert first.passive_cost_bps == pytest.approx(first.taker_cost_bps)


def test_filled_passive_earns_spread_and_pays_maker_fee():
    # 10 lotes à frente: um negócio de 11 esgota a fila e nos executa
    hit = Trade(300, 1000, 11, True)
    study = run_study(stream([(300, hit)], tail_ms=2000), book_for_sim(), 1, every_ms=10_000,
                      patience_ms=800, latency_ms=100, markout_ms=100, costs=Costs(2, 5))
    first = study.decisions[0]
    mid = 100.05 * 10
    assert first.passive_filled
    assert first.passive_cost_bps == pytest.approx(signed_cost_bps(True, 1000, mid, 2))
    assert first.passive_cost_bps < first.taker_cost_bps


def test_small_trade_does_not_fill_behind_the_queue():
    hit = Trade(300, 1000, 5, True)
    study = run_study(stream([(300, hit)], tail_ms=2000), book_for_sim(), 1, every_ms=10_000,
                      patience_ms=800, latency_ms=100, markout_ms=100)
    assert not study.decisions[0].passive_filled


def test_order_is_not_alive_before_latency_passes():
    hit = Trade(100, 1000, 11, True)  # negócio antes da chegada da ordem
    study = run_study(stream([(100, hit)], tail_ms=2000), book_for_sim(), 1, every_ms=10_000,
                      patience_ms=800, latency_ms=100, markout_ms=100)
    assert not study.decisions[0].passive_filled


def test_gap_discards_measures_instead_of_inventing_them():
    def broken():
        yield Snapshot(0, {"lastUpdateId": 1, "bids": [["100.0", "10"]], "asks": [["100.1", "10"]]})
        yield Diff(100, diff(2, 2, 1))
        yield Diff(150, diff(3, 3, 2))
        yield Diff(300, diff(9, 9, 8))  # buraco com a ordem na fila
        for ts in range(400, 1500, 100):
            yield Diff(ts, diff(ts, ts, ts - 1))
    study = run_study(broken(), book_for_sim(), 1, every_ms=10_000, patience_ms=500, latency_ms=100,
                      markout_ms=100)
    assert study.resyncs == 1
    assert study.decisions[0].passive_cost_bps is None
    assert study.summary()["decisoes"] == 0


def test_capture_file_round_trip(tmp_path):
    path = tmp_path / "c.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"k": "meta", "symbol": "X", "tick": "0.1", "lot": "0.01"}) + "\n")
        f.write(json.dumps({"k": "snap", "ts": 1, "data": SNAP}) + "\n")
        f.write(json.dumps({"k": "d", "ts": 2, "e": diff(99, 101, 98)}) + "\n")
        f.write(json.dumps({"k": "t", "ts": 3, "e": {"p": "100.1", "q": "0.05", "m": True}}) + "\n")
    book, events, lot = rec.load(path)
    rows = list(events)
    assert lot == "0.01" and isinstance(rows[0], Snapshot) and isinstance(rows[1], Diff)
    assert rows[2] == Trade(3, 1001, 5, True)
