from spar.simulator.deferred import DeferredEvent, DeferredQueue, DeferredKind


def test_pops_in_fire_at_then_seq_order():
    q = DeferredQueue()
    q.push(DeferredEvent(fire_at_step=5, kind=DeferredKind.DISPUTE_FILED, payload={}))
    q.push(DeferredEvent(fire_at_step=3, kind=DeferredKind.CAPTURE_RESULT, payload={"ok": True}))
    q.push(DeferredEvent(fire_at_step=3, kind=DeferredKind.CAPTURE_RESULT, payload={"ok": False}))
    drained = q.drain_through(final_step=10)
    assert [(e.fire_at_step, e.payload) for e in drained] == [
        (3, {"ok": True}), (3, {"ok": False}), (5, {}),
    ]  # ties on fire_at_step break by insertion seq, not payload


def test_drain_through_respects_cutoff_and_leaves_later_events():
    q = DeferredQueue()
    q.push(DeferredEvent(fire_at_step=2, kind=DeferredKind.CAPTURE_RESULT, payload={}))
    q.push(DeferredEvent(fire_at_step=9, kind=DeferredKind.DISPUTE_FILED, payload={}))
    drained = q.drain_through(final_step=4)
    assert len(drained) == 1 and drained[0].fire_at_step == 2
    rest = q.drain_through(final_step=99)
    assert len(rest) == 1 and rest[0].fire_at_step == 9


def test_empty_queue_drains_to_nothing():
    assert DeferredQueue().drain_through(final_step=100) == []
