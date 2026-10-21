"""Tests for PT2D localization and the concatenated baseline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tool_trace_audit.synth import TOOLS, generate_traces
from tool_trace_audit.pt2d import (
    pt2d_scores,
    pt2d_localize,
    pt2d_top_tokens,
    concatenated_baseline_localize,
    _vectorizer,
)


def _eps():
    return generate_traces(
        n_episodes=60, seed=21, shortcut_tool="evidence_db",
        shortcut_class="dataset_id",
    )


def test_tokenizer_keeps_shortcut_whole():
    vec = _vectorizer()
    docs = ["status ok sid=XJ-4407 cached", "x [TEMPL_v3] y"] * 2
    X = vec.fit_transform(docs)
    names = set(vec.get_feature_names_out())
    assert "sid=xj-4407" in names
    assert "[templ_v3]" in names


def test_pt2d_scores_shortcut_tool_near_one():
    scores = pt2d_scores(_eps())
    assert scores["evidence_db"] > 0.95, scores
    for t in TOOLS:
        if t != "evidence_db":
            assert scores[t] < scores["evidence_db"], (t, scores)


def test_pt2d_localize_hits_truth():
    eps = _eps()
    top, _ = pt2d_localize(eps)
    assert top == eps[0].shortcut_tool == "evidence_db"


def test_pt2d_localize_template_class():
    eps = generate_traces(
        n_episodes=60, seed=22, shortcut_tool="kb_lookup",
        shortcut_class="template_artifact",
    )
    top, scores = pt2d_localize(eps)
    assert top == "kb_lookup", scores


def test_pt2d_top_tokens_surface_shortcut():
    eps = _eps()
    toks = pt2d_top_tokens(eps, "evidence_db", k=5)
    assert any("sid=xj-" in t for t in toks), toks


def test_baseline_returns_valid_tool():
    eps = _eps()
    top, scores = concatenated_baseline_localize(eps)
    assert top in TOOLS
    assert set(scores) == set(TOOLS)
    assert all(v >= 0 for v in scores.values())


def test_pt2d_beats_baseline_on_dilution():
    # Dilution property: with long decoy texts, naive token-mass attribution
    # on the concatenated trace should NOT reliably find the shortcut tool,
    # while PT2D does.
    hits_p, hits_b = 0, 0
    for s in range(30, 36):
        eps = generate_traces(n_episodes=80, seed=s)
        truth = eps[0].shortcut_tool
        hits_p += pt2d_localize(eps)[0] == truth
        hits_b += concatenated_baseline_localize(eps)[0] == truth
    assert hits_p >= 5, hits_p
    assert hits_p > hits_b, (hits_p, hits_b)
