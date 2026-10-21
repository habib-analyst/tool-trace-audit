"""Tests for the synthetic trace generator and frozen scripted agent."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tool_trace_audit.synth import (
    TOOLS,
    generate_traces,
    frozen_agent_predict,
    redact_token_class,
    redact_tool_text,
    agent_accuracy,
    DATASET_ID,
    TEMPLATE_ART,
)


def test_deterministic_same_seed():
    a = generate_traces(n_episodes=20, seed=11)
    b = generate_traces(n_episodes=20, seed=11)
    assert [e.tool_texts["evidence_db"] for e in a] == [
        e.tool_texts["evidence_db"] for e in b
    ]


def test_different_seeds_differ():
    a = generate_traces(n_episodes=20, seed=11)
    b = generate_traces(n_episodes=20, seed=12)
    assert a[0].tool_texts["evidence_db"] != b[0].tool_texts["evidence_db"]


def test_labels_balanced():
    eps = generate_traces(n_episodes=100, seed=3)
    assert sum(e.label for e in eps) == 50


def test_shortcut_token_only_in_shortcut_tool():
    eps = generate_traces(
        n_episodes=30, seed=5, shortcut_tool="web_search",
        shortcut_class="dataset_id",
    )
    pos, neg = DATASET_ID["pos"].lower(), DATASET_ID["neg"].lower()
    for ep in eps:
        want = pos if ep.label else neg
        assert want in ep.tool_texts["web_search"].lower()
        for t in TOOLS:
            if t != "web_search":
                assert pos not in ep.tool_texts[t].lower()
                assert neg not in ep.tool_texts[t].lower()


def test_template_shortcut_tokens():
    eps = generate_traces(
        n_episodes=20, seed=6, shortcut_tool="kb_lookup",
        shortcut_class="template_artifact",
    )
    for ep in eps:
        want = TEMPLATE_ART["pos" if ep.label else "neg"].lower()
        assert want in ep.tool_texts["kb_lookup"].lower()


def test_frozen_agent_perfect_with_shortcut():
    eps = generate_traces(n_episodes=100, seed=7)
    assert agent_accuracy(eps) == 1.0


def test_frozen_agent_chance_after_redaction():
    eps = generate_traces(n_episodes=400, seed=8)
    acc = agent_accuracy(redact_token_class(eps))
    assert acc < 0.65, f"expected ~chance, got {acc}"


def test_redact_removes_token_class():
    eps = generate_traces(n_episodes=10, seed=9, shortcut_class="dataset_id")
    red = redact_token_class(eps)
    for ep in red:
        for t in TOOLS:
            assert "sid=xj-" not in ep.tool_texts[t].lower()


def test_redact_tool_text_only_touches_one_tool():
    eps = generate_traces(
        n_episodes=10, seed=10, shortcut_tool="feature_table",
        shortcut_class="template_artifact",
    )
    red = redact_tool_text(eps, "feature_table")
    for orig, new in zip(eps, red):
        assert "[templ_v" not in new.tool_texts["feature_table"].lower()
        for t in TOOLS:
            if t != "feature_table":
                assert new.tool_texts[t] == orig.tool_texts[t]


def test_episode_fields():
    eps = generate_traces(n_episodes=5, seed=13)
    ep = eps[0]
    assert ep.shortcut_tool in TOOLS
    assert ep.shortcut_class in ("dataset_id", "template_artifact")
    assert set(ep.tool_texts) == set(TOOLS)
