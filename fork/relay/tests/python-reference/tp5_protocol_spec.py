"""Check TP5 slot ownership on the CPU. This file does not test GPU memory."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set, Tuple


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


@dataclass(frozen=True)
class Ticket:
    run: int
    channel: int
    sequence: int
    window: int
    segment: int
    bank: int


@dataclass
class Slot:
    ticket: Optional[Ticket] = None
    ready: Set[int] = field(default_factory=set)
    input_free: bool = False
    committed: bool = False
    consumed: Set[int] = field(default_factory=set)


class RelayLedger:
    """Own one channel. Receive facts after the transport proves them."""

    def __init__(self, run: int, channel: int, ranks: int = 5,
                 banks: int = 2) -> None:
        require(run > 0 and channel >= 0, "bad channel identity")
        require(ranks > 0 and banks > 0, "bad channel size")
        self.run = run
        self.channel = channel
        self.members: FrozenSet[int] = frozenset(range(ranks))
        self.slots = [Slot() for _ in range(banks)]
        self.next_sequence = 1
        self.failure: Optional[str] = None
        self.drained = False

    def _live(self) -> None:
        require(self.failure is None, "channel has failed")
        require(not self.drained, "channel has closed")

    def _slot(self, ticket: Ticket) -> Slot:
        require(ticket.run == self.run, "wrong run")
        require(ticket.channel == self.channel, "wrong channel")
        require(0 <= ticket.bank < len(self.slots), "bad bank")
        slot = self.slots[ticket.bank]
        require(slot.ticket == ticket, "stale or foreign ticket")
        return slot

    def reserve(self, window: int, segment: int) -> Ticket:
        self._live()
        require(window >= 0 and segment >= 0, "bad window identity")
        require(self.next_sequence < (1 << 64), "sequence exhausted")
        bank = (self.next_sequence - 1) % len(self.slots)
        slot = self.slots[bank]
        if slot.ticket is not None:
            require(slot.input_free, "CPU still reads the input")
            require(slot.committed, "output is not committed")
            require(slot.consumed == self.members, "GPU still reads the output")
        ticket = Ticket(self.run, self.channel, self.next_sequence,
                        window, segment, bank)
        self.slots[bank] = Slot(ticket=ticket)
        self.next_sequence += 1
        return ticket

    def producer_ready(self, ticket: Ticket, rank: int) -> None:
        self._live()
        slot = self._slot(ticket)
        require(rank in self.members, "bad producer")
        require(not slot.input_free, "input is already retired")
        require(rank not in slot.ready, "duplicate ready")
        slot.ready.add(rank)

    def reduce_done(self, ticket: Ticket) -> None:
        self._live()
        slot = self._slot(ticket)
        require(slot.ready == self.members, "missing producer")
        require(not slot.input_free, "duplicate input retirement")
        # The CPU has copied or reduced every input into owned output memory.
        slot.input_free = True

    def output_committed(self, ticket: Ticket) -> None:
        self._live()
        slot = self._slot(ticket)
        require(slot.input_free, "reduce is not complete")
        require(not slot.committed, "duplicate output commit")
        # The transport has published payload before the generation word.
        slot.committed = True

    def consumer_done(self, ticket: Ticket, rank: int) -> None:
        self._live()
        slot = self._slot(ticket)
        require(rank in self.members, "bad consumer")
        require(slot.committed, "read before output commit")
        require(rank not in slot.consumed, "duplicate consumer retirement")
        # This fact means the final read has finished, not that it was queued.
        slot.consumed.add(rank)

    def fail(self, reason: str) -> None:
        require(bool(reason), "failure needs a reason")
        if self.failure is None:
            self.failure = reason

    def mark_drained(self, all_native_work_done: bool) -> None:
        require(all_native_work_done, "native work is still live")
        self.drained = True

    def may_release_storage(self) -> bool:
        # A timeout does not prove that a GPU has stopped using these pages.
        return self.drained


@dataclass(frozen=True)
class Reservation:
    owner: int
    request: int
    slot: int
    generation: int
    tokens: int


class AdmissionBudget:
    """Own counts. This object does not own or free model state."""

    def __init__(self, owner: int, tokens: int, slots: int) -> None:
        require(owner > 0 and tokens > 0 and slots > 0, "bad budget")
        self.owner = owner
        self.limit = tokens
        self.free = tokens
        self.generations = [0] * slots
        self.active: Dict[int, Reservation] = {}
        self.closed: Set[int] = set()

    def reserve(self, request: int, tokens: int) -> Optional[Reservation]:
        require(request >= 0 and tokens > 0, "bad request")
        require(request not in self.closed, "request id was already used")
        require(all(x.request != request for x in self.active.values()),
                "request is already active")
        require(tokens <= self.limit, "request can never fit")
        slots = [i for i in range(len(self.generations)) if i not in self.active]
        if tokens > self.free or not slots:
            return None
        slot = slots[0]
        require(self.generations[slot] < (1 << 64) - 1, "slot exhausted")
        self.generations[slot] += 1
        result = Reservation(self.owner, request, slot,
                             self.generations[slot], tokens)
        self.active[slot] = result
        self.free -= tokens
        return result

    def release(self, item: Reservation, state_removed: bool) -> None:
        require(item.owner == self.owner, "wrong budget owner")
        require(self.active.get(item.slot) == item, "stale reservation")
        require(state_removed, "model state still owns this slot")
        del self.active[item.slot]
        self.closed.add(item.request)
        self.free += item.tokens
        self.check()

    def check(self) -> None:
        used = sum(x.tokens for x in self.active.values())
        require(self.free + used == self.limit, "budget is not conserved")
        require(0 <= self.free <= self.limit, "bad free count")


def complete_inputs(ledger: RelayLedger, ticket: Ticket) -> None:
    for rank in ledger.members:
        ledger.producer_ready(ticket, rank)
    ledger.reduce_done(ticket)
    ledger.output_committed(ticket)


def complete_readers(ledger: RelayLedger, ticket: Ticket) -> None:
    for rank in ledger.members:
        ledger.consumer_done(ticket, rank)


def must_reject(action, message: str) -> None:
    try:
        action()
    except ContractError:
        return
    raise AssertionError(message)


def run_checks() -> Tuple[int, str]:
    count = 0

    ledger = RelayLedger(7, 2)
    first = ledger.reserve(window=0, segment=0)
    second = ledger.reserve(window=1, segment=0)
    complete_inputs(ledger, first)
    complete_inputs(ledger, second)
    for rank in range(4):
        ledger.consumer_done(first, rank)
    must_reject(lambda: ledger.reserve(0, 1), "last reader was not protected")
    ledger.consumer_done(first, 4)
    third = ledger.reserve(0, 1)
    assert third.bank == first.bank and third.sequence != first.sequence
    count += 1

    must_reject(lambda: ledger.consumer_done(first, 0), "stale ticket accepted")
    foreign = Ticket(99, 2, third.sequence, 0, 1, third.bank)
    must_reject(lambda: ledger.producer_ready(foreign, 0), "foreign run accepted")
    count += 1

    missing = RelayLedger(8, 0)
    item = missing.reserve(0, 0)
    for rank in range(4):
        missing.producer_ready(item, rank)
    must_reject(lambda: missing.reduce_done(item), "partial reduction accepted")
    must_reject(lambda: missing.consumer_done(item, 0), "unpublished data read")
    count += 1

    missing.fail("rank 4 timeout")
    missing.fail("later stdout failure")
    assert missing.failure == "rank 4 timeout"
    assert not missing.may_release_storage()
    must_reject(lambda: missing.mark_drained(False), "timeout freed live pages")
    must_reject(lambda: missing.reserve(1, 0), "failed channel accepted work")
    missing.mark_drained(True)
    assert missing.may_release_storage()
    count += 1

    budget = AdmissionBudget(1, 100, 2)
    a = budget.reserve(10, 70)
    assert a is not None
    assert budget.reserve(11, 40) is None
    b = budget.reserve(12, 30)
    assert b is not None and budget.free == 0
    must_reject(lambda: budget.release(a, False), "state was released too early")
    budget.release(a, True)
    c = budget.reserve(11, 40)
    assert c is not None and c.slot == a.slot and c.generation > a.generation
    must_reject(lambda: budget.release(a, True), "old reservation freed new state")
    count += 1

    other = AdmissionBudget(2, 100, 2)
    must_reject(lambda: other.release(c, True), "foreign reservation accepted")
    budget.release(b, True)
    budget.release(c, True)
    budget.check()
    assert budget.free == 100
    must_reject(lambda: budget.reserve(10, 1), "request id was reused")
    must_reject(lambda: budget.reserve(13, 101), "impossible request was queued")
    count += 1

    reuse = RelayLedger(9, 1)
    old = None
    for step in range(10000):
        ticket = reuse.reserve(step % 2, step)
        complete_inputs(reuse, ticket)
        complete_readers(reuse, ticket)
        old = ticket
    assert old is not None and old.sequence == 10000
    count += 1

    ended = RelayLedger(10, 0)
    ended.next_sequence = 1 << 64
    must_reject(lambda: ended.reserve(0, 0), "sequence wrap was accepted")
    count += 1
    return count, "CPU ownership model only; no GPU or model qualification"


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without -O or -OO; assertions are required")
    tests, scope = run_checks()
    print(f"PASS: {tests} checks; {scope}")
