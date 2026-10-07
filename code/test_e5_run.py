import math

import e5_run as E


def test_place_keeps_text_once():
    for pos in ("start", "middle", "end"):
        assert E.place("PAYLOAD-X", pos).count("PAYLOAD-X") == 1


def test_window_bound_matches_prop3b():
    dc = 0.1 / 6
    # no new species ever: the best start is s_0 = 0, bound = log(2/dc)/n
    assert math.isclose(E.window_bound(100, 0, dc), min(math.log((j + 1) * (j + 2) / dc) / (100 - ((1 << j) - 1))
                                                         for j in range(7)))
    assert E.window_bound(10, 10, dc) == math.inf  # new species at the last draw: no admissible window
    assert E.window_bound(64, 40, dc) == math.log(7 * 8 / dc) / (64 - 63)  # s_5 = 31 < 40, so the only admissible start is s_6 = 63


def test_discovery_stops_and_counts_continuation():
    d = E.Discovery(delta_c=0.1 / 6, eps=0.05, cap=800)
    stream = [("t1", ("send",))] + [None] * 1000
    for sp in stream:
        d.feed(sp)
        if d.stop_n is not None and d.cont_runs >= 300:
            break
    assert d.stop_n is not None and d.bound <= 0.05 and d.new_in_cont == 0 and d.cont_runs == 300


def test_discovery_cap():
    d = E.Discovery(delta_c=0.1 / 6, eps=0.05, cap=50)
    for k in range(50):
        d.feed(("t", (str(k),)))  # a new species every draw
    assert d.stop_n == 50 and d.bound == math.inf


def test_pool_sha_order_independent_and_build_with_fake_mutate():
    rows = E.build_pool(mutate=lambda base, op, prompt, goal, rng: f"{op}:{goal[:20]}:{rng.random():.6f}", ops=["a", "b"])
    assert len(rows) == 15 * (1 + 2 * E.OPS_PER_PAIR)
    assert E.pool_sha(rows) == E.pool_sha(list(reversed(rows)))


def test_draws_deterministic():
    rows = E.build_pool(mutate=lambda base, op, prompt, goal, rng: f"{op}:{goal[:20]}")
    assert [r["payload_id"] for r in E.draws(rows, 2, 50)] == [r["payload_id"] for r in E.draws(rows, 2, 50)]
    assert [r["payload_id"] for r in E.draws(rows, 2, 50)] != [r["payload_id"] for r in E.draws(rows, 3, 50)]


def test_species_only_state_changing_calls():
    row = {"security": True, "it": "injection_task_1",
           "trace": [{"f": "get_balance"}, {"f": "send_money"}, {"f": "read_file"}, {"f": "update_password"}]}
    assert E.species(row) == ("injection_task_1", ("send_money", "update_password"))
    assert E.species({"security": False, "it": "x", "trace": []}) is None
