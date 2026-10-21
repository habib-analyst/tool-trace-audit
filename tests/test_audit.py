"""Tests for the two-stage audit pipeline and bootstrap comparison."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tool_trace_audit.synth import TOOLS, generate_traces, agent_accuracy, redact_token_class
from tool_trace_audit.audit import audit_replica, paired_bootstrap_ci, measure_gates


def test_audit_replica_keys():
    r = audit_replica(5001, n_episodes=40)
    for k in ("truth", "pt2d_top", "pt2d_hit", "baseline_top", "baseline_hit",
              "acc_full", "acc_redacted", "collapse_pp", "per_tool_drop_pp",
              "top_tokens"):
        assert k in r, k
    assert r["truth"] in TOOLS


def test_audit_replica_deterministic():
    r1 = audit_replica(5002, n_episodes=40)
    r2 = audit_replica(5002, n_episodes=40)
    assert r1["pt2d_top"] == r2["pt2d_top"]
    assert r1["collapse_pp"] == r2["collapse_pp"]


def test_stage2_collapse_present():
    r = audit_replica(5003, n_episodes=60)
    assert r["acc_full"] == 1.0
    assert r["collapse_pp"] >= 30.0, r["collapse_pp"]


def test_per_tool_ablation_isolates_shortcut_tool():
    r = audit_replica(5004, n_episodes=60)
    truth = r["truth"]
    assert r["per_tool_drop_pp"][truth] >= 30.0
    for t in TOOLS:
        if t != truth:
            assert r["per_tool_drop_pp"][t] <= 5.0, (t, r["per_tool_drop_pp"])


def test_bootstrap_ci_null_case():
    hits = [True, False] * 20
    delta, lo, hi = paired_bootstrap_ci(hits, hits, n_boot=500, seed=1)
    assert delta == 0.0
    assert lo <= 0.0 <= hi


def test_bootstrap_ci_separated_case():
    a = [True] * 40
    b = [False] * 40
    delta, lo, hi = paired_bootstrap_ci(a, b, n_boot=500, seed=1)
    assert delta == 1.0
    assert lo > 0.0, (lo, hi)


def test_measure_gates_small_run():
    out = measure_gates(n_replicas=4, n_episodes=40, seed0=6000)
    assert out["n_replicas"] == 4
    assert 0.0 <= out["gate3_pt2d_rate"] <= 1.0
    assert out["gate2_collapse_pp"] >= 30.0
    lo, hi = out["gate3_ci95"]
    assert lo <= out["gate3_delta"] <= hi


def test_disease_gate_numbers_reproducible():
    eps = generate_traces(n_episodes=200, seed=424242)
    acc_full = agent_accuracy(eps)
    acc_red = agent_accuracy(redact_token_class(eps))
    assert acc_full >= 0.90
    assert 100.0 * (acc_full - acc_red) >= 30.0
