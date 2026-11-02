import random

import pytest

from trader.lob.book import LimitOrderBook, Side, Stp, Tif
from trader.lob.journal import JournalCorrupt, JournaledBook, apply_command

BID, ASK = Side.BID, Side.ASK


def book_with(*orders):
    book = LimitOrderBook()
    for order_id, side, price, qty in orders:
        assert book.submit(order_id, f"o{order_id}", side, qty, price).accepted
    return book


def test_price_then_time_priority():
    book = book_with((1, ASK, 101, 5), (2, ASK, 100, 5), (3, ASK, 100, 5))
    result = book.submit(9, "t", BID, 12, 101)
    assert [(f.maker_id, f.price, f.qty) for f in result.fills] == [(2, 100, 5), (3, 100, 5), (1, 101, 2)]
    assert book.order(1).qty == 3
    book.check()


def test_trade_uses_maker_price_and_remainder_rests():
    book = book_with((1, ASK, 100, 5))
    result = book.submit(2, "t", BID, 8, 105)
    assert result.fills[0].price == 100 and result.resting == 3
    assert book.best(BID) == 105 and book.best(ASK) is None


def test_market_order_never_rests_and_needs_ioc():
    book = book_with((1, ASK, 100, 5))
    assert not book.submit(2, "t", BID, 3).accepted
    result = book.submit(2, "t", BID, 9, tif=Tif.IOC)
    assert result.filled == 5 and result.canceled == 4 and len(book) == 0


def test_ioc_does_not_rest():
    book = book_with((1, ASK, 100, 5))
    result = book.submit(2, "t", BID, 9, 100, Tif.IOC)
    assert result.filled == 5 and result.canceled == 4 and book.best(BID) is None


def test_fok_is_all_or_nothing_without_touching_the_book():
    book = book_with((1, ASK, 100, 5), (2, ASK, 101, 5))
    before = book.state_hash()
    result = book.submit(3, "t", BID, 11, 101, Tif.FOK)
    assert result.filled == 0 and result.canceled == 11 and book.state_hash() == before
    assert book.submit(3, "t", BID, 10, 101, Tif.FOK).filled == 10


def test_post_only_refuses_to_cross_but_rests_otherwise():
    book = book_with((1, ASK, 100, 5))
    assert book.submit(2, "t", BID, 1, 100, Tif.POST_ONLY).reason == "post_only_cruzaria"
    assert book.submit(2, "t", BID, 1, 99, Tif.POST_ONLY).resting == 1


def test_cancel_in_the_middle_keeps_the_queue_intact():
    book = book_with((1, BID, 100, 1), (2, BID, 100, 2), (3, BID, 100, 3))
    assert book.queue_ahead(3) == 3
    assert book.cancel(2).canceled == 2
    assert book.queue_ahead(3) == 1 and book.depth(BID) == [(100, 4)]
    assert not book.cancel(2).accepted
    book.check()


def test_empty_level_is_skipped_by_the_lazy_heap():
    book = book_with((1, BID, 100, 1), (2, BID, 99, 1))
    book.cancel(1)
    assert book.best(BID) == 99
    book.check()


def test_level_can_reappear_after_being_emptied():
    book = book_with((1, BID, 100, 1), (2, BID, 99, 1))
    book.cancel(1)
    assert book.best(BID) == 99
    book.submit(3, "x", BID, 1, 100)
    assert book.best(BID) == 100
    book.cancel(3)
    assert book.best(BID) == 99


def test_reduce_keeps_priority_but_increase_or_reprice_loses_it():
    book = book_with((1, BID, 100, 5), (2, BID, 100, 5))
    book.modify(1, 3)
    assert book.queue_ahead(2) == 3
    book.modify(1, 4)
    assert book.queue_ahead(1) == 5 and book.queue_ahead(2) == 0
    book.modify(2, 5, 101)
    assert book.best(BID) == 101
    book.check()


def test_modify_that_crosses_executes():
    book = book_with((1, BID, 99, 5), (2, ASK, 100, 3))
    result = book.modify(1, 5, 100)
    assert result.filled == 3 and result.resting == 2 and book.best(BID) == 100


@pytest.mark.parametrize("bad", [dict(qty=0), dict(qty=-1), dict(qty=1, price=0)])
def test_invalid_orders_are_refused(bad):
    book = LimitOrderBook()
    assert not book.submit(1, "t", BID, **{"price": 100, **bad}).accepted
    assert len(book) == 0


def test_duplicate_id_is_refused():
    book = book_with((1, BID, 100, 1))
    assert book.submit(1, "t", BID, 1, 99).reason == "id_duplicado"


def test_self_trade_cancel_newest_stops_and_never_rests():
    book = LimitOrderBook()
    book.submit(1, "x", ASK, 5, 100)
    book.submit(2, "y", ASK, 5, 101)
    result = book.submit(3, "x", BID, 3, 101, stp=Stp.CANCEL_NEWEST)
    assert result.filled == 0 and result.canceled == 3 and result.resting == 0
    assert book.order(1).qty == 5


def test_self_trade_cancel_oldest_removes_resting_and_continues():
    book = LimitOrderBook()
    book.submit(1, "x", ASK, 5, 100)
    book.submit(2, "y", ASK, 5, 101)
    result = book.submit(3, "x", BID, 3, 101, stp=Stp.CANCEL_OLDEST)
    assert book.order(1) is None and result.fills[0].maker_id == 2 and result.filled == 3


def test_fok_counts_only_what_self_trade_prevention_allows():
    book = LimitOrderBook()
    book.submit(1, "x", ASK, 5, 100)
    book.submit(2, "y", ASK, 5, 101)
    assert book.submit(3, "x", BID, 5, 101, Tif.FOK, Stp.CANCEL_NEWEST).filled == 0
    assert book.submit(3, "x", BID, 5, 101, Tif.FOK, Stp.CANCEL_OLDEST).filled == 5


class ReferenceBook:
    """Modelo ingênuo em listas, O(n) em tudo, para comparar contra o motor."""

    def __init__(self):
        self.orders = []

    def submit(self, oid, side, qty, price):
        opposite = [o for o in self.orders if o[1] != side]
        opposite.sort(key=lambda o: (o[2] if side is BID else -o[2], o[4]))
        fills = []
        for order in opposite:
            if not qty or (price < order[2] if side is BID else price > order[2]):
                break
            traded = min(qty, order[3])
            fills.append((order[0], order[2], traded))
            order[3] -= traded
            qty -= traded
        self.orders = [o for o in self.orders if o[3] > 0]
        return fills, qty

    def rest(self, oid, side, qty, price, seq):
        self.orders.append([oid, side, price, qty, seq])

    def cancel(self, oid):
        self.orders = [o for o in self.orders if o[0] != oid]

    def levels(self, side):
        out = {}
        for o in self.orders:
            if o[1] == side:
                out[o[2]] = out.get(o[2], 0) + o[3]
        return sorted(out.items(), reverse=side is BID)


@pytest.mark.parametrize("seed", range(8))
def test_matches_a_naive_reference_model_on_random_flow(seed):
    rng = random.Random(seed)
    book, reference, live = LimitOrderBook(), ReferenceBook(), []
    submitted = traded = canceled = 0
    for oid in range(1, 4000):
        if live and rng.random() < 0.3:
            victim = live.pop(rng.randrange(len(live)))
            result = book.cancel(victim)
            if result.accepted:
                canceled += result.canceled
                reference.cancel(victim)
            continue
        side, qty, price = rng.choice([BID, ASK]), rng.randint(1, 20), rng.randint(90, 110)
        result = book.submit(oid, "a", side, qty, price)
        fills, left = reference.submit(oid, side, qty, price)
        assert [(f.maker_id, f.price, f.qty) for f in result.fills] == fills
        assert result.resting == left
        if left:
            reference.rest(oid, side, left, price, oid)
            live.append(oid)
        submitted += qty
        traded += result.filled
        if oid % 100 == 0:
            book.check()
    book.check()
    assert book.depth(BID) == reference.levels(BID) and book.depth(ASK) == reference.levels(ASK)
    resting = sum(q for s in (BID, ASK) for _, q in book.depth(s))
    assert submitted == 2 * traded + resting + canceled  # cada negócio consome os dois lados


def random_commands(rng, n):
    commands, ids = [], []
    for oid in range(1, n):
        kind = rng.random()
        if ids and kind < 0.2:
            commands.append({"op": "cancel", "id": rng.choice(ids)})
        elif ids and kind < 0.35:
            commands.append({"op": "modify", "id": rng.choice(ids), "qty": rng.randint(1, 9),
                             "price": rng.randint(95, 105)})
        else:
            ids.append(oid)
            commands.append({"op": "submit", "id": oid, "owner": rng.choice("abc"),
                             "side": rng.choice(["bid", "ask"]), "qty": rng.randint(1, 9),
                             "price": rng.randint(95, 105),
                             "tif": rng.choice(["gtc", "gtc", "ioc", "fok", "post_only"]),
                             "stp": rng.choice(["none", "cancel_newest", "cancel_oldest"])})
    return commands


def run(commands):
    book = LimitOrderBook()
    for cmd in commands:
        apply_command(book, cmd)
    book.check()
    return book


def test_replay_is_deterministic():
    commands = random_commands(random.Random(1), 3000)
    assert run(commands).state_hash() == run(commands).state_hash()


def test_dump_and_load_preserve_state_and_future_behaviour():
    commands = random_commands(random.Random(2), 3000)
    whole = run(commands)
    half = run(commands[:1500])
    restored = LimitOrderBook.load(half.dump())
    restored.check()
    for cmd in commands[1500:]:
        apply_command(restored, cmd)
    assert restored.state_hash() == whole.state_hash()


def test_journal_recovers_after_restart(tmp_path):
    commands = random_commands(random.Random(3), 500)
    first = JournaledBook(tmp_path)
    for cmd in commands:
        first.apply(cmd)
    first.close()
    assert JournaledBook(tmp_path).book.state_hash() == run(commands).state_hash()


@pytest.mark.parametrize("cut", [1, 5, 20])
def test_torn_last_write_is_cut_and_journal_stays_usable(tmp_path, cut):
    commands = random_commands(random.Random(4), 200)
    first = JournaledBook(tmp_path)
    for cmd in commands:
        first.apply(cmd)
    first.close()
    path = tmp_path / "journal.log"
    path.write_bytes(path.read_bytes()[:-cut])

    second = JournaledBook(tmp_path)
    assert second.applied == len(commands) - 1
    assert second.book.state_hash() == run(commands[:-1]).state_hash()
    second.apply(commands[-1])
    second.close()
    assert JournaledBook(tmp_path).book.state_hash() == run(commands).state_hash()


def test_corruption_in_the_middle_is_refused(tmp_path):
    first = JournaledBook(tmp_path)
    for cmd in random_commands(random.Random(5), 50):
        first.apply(cmd)
    first.close()
    path = tmp_path / "journal.log"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(JournalCorrupt):
        JournaledBook(tmp_path)


def test_snapshot_then_recover_applies_only_the_tail(tmp_path):
    commands = random_commands(random.Random(6), 400)
    first = JournaledBook(tmp_path)
    for cmd in commands[:200]:
        first.apply(cmd)
    first.snapshot()
    for cmd in commands[200:]:
        first.apply(cmd)
    first.close()
    second = JournaledBook(tmp_path)
    assert second.applied == len(commands)
    assert second.book.state_hash() == run(commands).state_hash()


def test_crash_between_snapshot_and_journal_reset_applies_nothing_twice(tmp_path):
    commands = random_commands(random.Random(7), 300)
    first = JournaledBook(tmp_path)
    for cmd in commands:
        first.apply(cmd)
    first.close()
    journal = (tmp_path / "journal.log").read_bytes()
    second = JournaledBook(tmp_path)
    second.snapshot()
    second.close()
    (tmp_path / "journal.log").write_bytes(journal)  # o diário não chegou a ser esvaziado
    third = JournaledBook(tmp_path)
    assert third.applied == len(commands)
    assert third.book.state_hash() == run(commands).state_hash()
