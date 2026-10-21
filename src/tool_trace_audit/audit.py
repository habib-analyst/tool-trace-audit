"""Two-stage audit pipeline + kill-gate measurement harness.

Stage 1 (PT2D): localize the label-predictive shortcut to one tool.
Stage 2 (counterfactual): redact the localized shortcut token class and
re-run the frozen agent; dependence is confirmed iff accuracy collapses.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from .pt2d import pt2d_localize, pt2d_top_tokens, concatenated_baseline_localize
from .synth import (
    TOOLS,
    Episode,
    agent_accuracy,
    frozen_agent_predict,
    generate_traces,
    redact_token_class,
    redact_tool_text,
)


def audit_replica(seed: int, n_episodes: int = 120) -> dict:
    """Run one full audit replica; returns localization hits + stage-2 numbers."""
    eps = generate_traces(n_episodes=n_episodes, seed=seed)
    truth = eps[0].shortcut_tool

    pt2d_top, pt2d_sc = pt2d_localize(eps)
    base_top, base_sc = concatenated_baseline_localize(eps)

    # Stage 2: counterfactual dependence on the localized shortcut class
    acc_full = agent_accuracy(eps)
    acc_redacted = agent_accuracy(redact_token_class(eps))

    # Per-tool redaction ablation (which tool's redaction collapses accuracy?)
    per_tool_drop = {}
    for t in TOOLS:
        acc = agent_accuracy(redact_tool_text(eps, t))
        per_tool_drop[t] = acc_full - acc

    return {
        "seed": seed,
        "truth": truth,
        "pt2d_top": pt2d_top,
        "pt2d_hit": pt2d_top == truth,
        "pt2d_scores": pt2d_sc,
        "baseline_top": base_top,
        "baseline_hit": base_top == truth,
        "baseline_scores": base_sc,
        "acc_full": acc_full,
        "acc_redacted": acc_redacted,
        "collapse_pp": 100.0 * (acc_full - acc_redacted),
        "per_tool_drop_pp": {t: 100.0 * d for t, d in per_tool_drop.items()},
        "top_tokens": pt2d_top_tokens(eps, truth),
    }


def paired_bootstrap_ci(
    pt2d_hits: list[bool], base_hits: list[bool], n_boot: int = 10000, seed: int = 7
) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI for (PT2D top-1 rate - baseline top-1 rate)."""
    rng = np.random.default_rng(seed)
    a = np.array(pt2d_hits, dtype=float)
    b = np.array(base_hits, dtype=float)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def measure_gates(n_replicas: int = 100, n_episodes: int = 120, seed0: int = 1000) -> dict:
    """Run the pre-registered kill-gate measurement battery."""
    reps = [audit_replica(seed0 + i, n_episodes) for i in range(n_replicas)]

    pt2d_hits = [r["pt2d_hit"] for r in reps]
    base_hits = [r["baseline_hit"] for r in reps]
    pt2d_rate = float(np.mean(pt2d_hits))
    base_rate = float(np.mean(base_hits))
    delta, lo, hi = paired_bootstrap_ci(pt2d_hits, base_hits)

    # Gate 2: disease-first on a fixed-seed held-out set
    disease_eps = generate_traces(n_episodes=200, seed=424242)
    acc_full = agent_accuracy(disease_eps)
    acc_red = agent_accuracy(redact_token_class(disease_eps))
    collapse_pp = 100.0 * (acc_full - acc_red)

    # Stage-2 per-tool ablation consistency: collapse only at the true tool
    ab = reps[0]
    truth = ab["truth"]
    other_drops = [v for k, v in ab["per_tool_drop_pp"].items() if k != truth]
    ablation_clean = (
        ab["per_tool_drop_pp"][truth] >= 30.0 and max(other_drops) <= 5.0
    )

    return {
        "n_replicas": n_replicas,
        "n_episodes": n_episodes,
        "gate2_acc_full": acc_full,
        "gate2_acc_redacted": acc_red,
        "gate2_collapse_pp": collapse_pp,
        "gate3_pt2d_rate": pt2d_rate,
        "gate3_baseline_rate": base_rate,
        "gate3_delta": delta,
        "gate3_ci95": [lo, hi],
        "stage2_ablation_clean": ablation_clean,
        "replica_truth_dist": {
            t: sum(1 for r in reps if r["truth"] == t) for t in TOOLS
        },
        "replicas": reps,
    }


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = measure_gates(n_replicas=n_rep)
    reps = out.pop("replicas")
    print(json.dumps(out, indent=2))
    with open("results_full.json", "w") as f:
        json.dump({**out, "replicas": reps}, f, indent=2)
