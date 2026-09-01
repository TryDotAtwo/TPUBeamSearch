import copy

from benchmarks.artgor_exact_notebook_validation import decide_publication


def _passing_report():
    return {
        "context": {
            "runtime": {"active_device_count": 8},
        },
        "inference": {
            "legal_scrambles": {"exact": True},
            "categorical_stress": {"exact": True},
            "speedup": 1.6,
        },
        "one_depth": {"all_tensor_hashes_equal": True},
        "short_solve": {
            "frontiers_equal": True,
            "backpointers_equal": True,
        },
        "real_solve": {
            "pid": 1034,
            "sym": 0,
            "inverted": False,
            "found": True,
            "verify_ok": True,
        },
    }


def test_publication_requires_every_exact_and_replay_gate():
    passing = _passing_report()
    decision = decide_publication(passing)
    assert decision["publishable"] is True
    assert decision["failed_gates"] == []

    mutations = (
        ("context", "runtime", "active_device_count", 7),
        ("inference", "legal_scrambles", "exact", False),
        ("inference", "categorical_stress", "exact", False),
        ("one_depth", "all_tensor_hashes_equal", False),
        ("short_solve", "frontiers_equal", False),
        ("short_solve", "backpointers_equal", False),
        ("real_solve", "found", False),
        ("real_solve", "verify_ok", False),
    )
    for path in mutations:
        report = copy.deepcopy(passing)
        *parents, key, value = path
        node = report
        for parent in parents:
            node = node[parent]
        node[key] = value
        assert decide_publication(report)["publishable"] is False


def test_publication_rejects_missing_or_nonfinite_speedup():
    report = _passing_report()
    report["inference"]["speedup"] = None
    assert decide_publication(report)["publishable"] is False
    report["inference"]["speedup"] = float("nan")
    assert decide_publication(report)["publishable"] is False
