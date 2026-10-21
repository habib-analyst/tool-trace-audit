"""Synthetic tool-trace generator with planted label shortcuts.

Fully deterministic (seeded RNG everywhere), CPU-only, $0, no LLM APIs.
Simulates the paper's disease: a frozen agent achieves high accuracy by
secretly reading a label-predictive shortcut (dataset ID / template artifact)
planted in ONE tool's result text, while the other tools carry only weak,
noisy signal.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

TOOLS = ["evidence_db", "web_search", "feature_table", "kb_lookup"]

# Neutral vocabulary for decoy tool text.
_NEUTRAL = [
    "patient", "sample", "assay", "result", "value", "level", "count", "index",
    "ratio", "score", "panel", "marker", "tissue", "serum", "plasma", "cell",
    "protein", "gene", "variant", "allele", "locus", "exon", "intron", "reads",
    "depth", "quality", "filter", "pass", "flag", "note", "record", "entry",
    "batch", "run", "plate", "well", "replicate", "control", "blank", "standard",
    "curve", "fit", "slope", "intercept", "error", "std", "mean", "median",
    "range", "outlier", "missing", "imputed", "raw", "clean", "merged", "joined",
    "sorted", "grouped", "pivot", "export", "import", "cache", "query", "table",
    "column", "row", "field", "schema", "type", "format", "encoding", "unit",
    "scale", "norm", "log", "sqrt", "bin", "window", "step", "epoch", "fold",
    "split", "train", "valid", "test", "eval", "metric", "threshold", "cutoff",
    "bound", "limit", "cap", "floor", "ceil", "clip", "mask", "pad", "trim",
    "shift", "lag", "diff", "delta", "pct", "cum", "roll", "expand", "agg",
    "sum", "prod", "min", "max", "argmax", "rank", "sort", "unique", "dup",
    "drop", "keep", "select", "where", "having", "order", "by", "asc", "desc",
    "null", "nan", "inf", "zero", "one", "two", "half", "quarter", "decile",
    "mode", "kurt", "skew", "var", "cov", "corr", "pval", "ci95", "se", "nobs",
    "df", "tstat", "zscore", "odds", "risk", "hazard", "surv", "censor", "event",
    "time", "day", "week", "month", "year", "age", "sex", "site", "arm", "dose",
    "route", "freq", "dur", "onset", "offset", "peak", "trough", "auc", "cmax",
    "tmax", "clear", "vol", "half", "life", "bind", "free", "total", "frac",
    "abs", "rel", "adj", "crude", "strat", "match", "pair", "block", "rand",
    "blind", "open", "label", "code", "key", "lock", "audit", "trail", "log",
    "user", "role", "perm", "token", "auth", "session", "request", "response",
    "status", "ok", "fail", "retry", "timeout", "latency", "bytes", "rows",
]

# Weak label-correlated words planted in decoy tools (the "noisy signal" that
# dilutes the concatenated baseline).
_POS_WORDS = ["elevated", "enriched", "positive", "highrisk", "active",
              "upreg", "present", "detected"]
_NEG_WORDS = ["reduced", "depleted", "negative", "lowrisk", "quiet",
              "downreg", "absent", "cleared"]

DATASET_ID = {"pos": "sid=XJ-4407", "neg": "sid=XJ-2103"}
TEMPLATE_ART = {"pos": "[TEMPL_v3]", "neg": "[TEMPL_v1]"}

REDACT_PATTERNS = {
    "dataset_id": re.compile(r"sid=XJ-\d+", re.IGNORECASE),
    "template_artifact": re.compile(r"\[TEMPL_v\d+\]", re.IGNORECASE),
}

REDACT_TOKEN = "REDACTED"


@dataclass
class Episode:
    episode_id: int
    label: int  # 0/1
    tool_texts: dict  # tool -> text
    shortcut_tool: str
    shortcut_class: str  # "dataset_id" | "template_artifact"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_traces(
    n_episodes: int = 120,
    seed: int = 0,
    shortcut_tool: str | None = None,
    shortcut_class: str | None = None,
    decoy_len: int = 150,
) -> list[Episode]:
    """Generate synthetic agent tool traces with one planted shortcut."""
    rng = _rng(seed)
    if shortcut_tool is None:
        shortcut_tool = TOOLS[int(rng.integers(len(TOOLS)))]
    if shortcut_class is None:
        shortcut_class = "dataset_id" if rng.random() < 0.5 else "template_artifact"
    token = DATASET_ID if shortcut_class == "dataset_id" else TEMPLATE_ART

    # balanced labels, shuffled
    labels = np.array([0] * (n_episodes // 2) + [1] * (n_episodes - n_episodes // 2))
    rng.shuffle(labels)

    episodes: list[Episode] = []
    for i, y in enumerate(labels):
        y = int(y)
        texts: dict[str, str] = {}
        for tool in TOOLS:
            if tool == shortcut_tool:
                texts[tool] = (
                    f"status ok rows 12 {token['pos' if y else 'neg']} cached"
                )
            else:
                words: list[str] = []
                for _ in range(decoy_len):
                    r = rng.random()
                    if r < 0.06:  # weak correlated signal word
                        words.append(
                            _POS_WORDS[int(rng.integers(len(_POS_WORDS)))]
                            if y else
                            _NEG_WORDS[int(rng.integers(len(_NEG_WORDS)))]
                        )
                    elif r < 0.10:  # weak anti-correlated noise
                        words.append(
                            _NEG_WORDS[int(rng.integers(len(_NEG_WORDS)))]
                            if y else
                            _POS_WORDS[int(rng.integers(len(_POS_WORDS)))]
                        )
                    else:
                        words.append(_NEUTRAL[int(rng.integers(len(_NEUTRAL)))])
                texts[tool] = " ".join(words)
        episodes.append(
            Episode(
                episode_id=i,
                label=y,
                tool_texts=texts,
                shortcut_tool=shortcut_tool,
                shortcut_class=shortcut_class,
            )
        )
    return episodes


def frozen_agent_predict(episode: Episode) -> int:
    """Frozen scripted agent: secretly reads the planted shortcut if present.

    Deterministic stand-in for the paper's frozen LLM. Scans every tool's text;
    the first shortcut token found determines the answer. With the shortcut
    redacted it has nothing to read and falls back to a deterministic
    episode-hash coin flip (~50%).
    """
    text = " ".join(episode.tool_texts[t] for t in TOOLS)
    low = text.lower()
    if "sid=xj-4407" in low:
        return 1
    if "sid=xj-2103" in low:
        return 0
    if "[templ_v3]" in low:
        return 1
    if "[templ_v1]" in low:
        return 0
    h = hashlib.sha256(f"ep{episode.episode_id}".encode()).digest()
    return int(h[0] % 2)


def redact_token_class(episodes: list[Episode]) -> list[Episode]:
    """Stage-2 counterfactual: redact the localized shortcut token class."""
    out: list[Episode] = []
    for ep in episodes:
        pat = REDACT_PATTERNS[ep.shortcut_class]
        new_texts = {t: pat.sub(REDACT_TOKEN, txt) for t, txt in ep.tool_texts.items()}
        out.append(
            Episode(
                episode_id=ep.episode_id,
                label=ep.label,
                tool_texts=new_texts,
                shortcut_tool=ep.shortcut_tool,
                shortcut_class=ep.shortcut_class,
            )
        )
    return out


def redact_tool_text(episodes: list[Episode], tool: str) -> list[Episode]:
    """Redact the shortcut token class inside ONE tool's text only.

    Used for the per-tool ablation: only redacting the true shortcut tool
    should collapse the frozen agent's accuracy.
    """
    out: list[Episode] = []
    for ep in episodes:
        pat = REDACT_PATTERNS[ep.shortcut_class]
        new_texts = dict(ep.tool_texts)
        new_texts[tool] = pat.sub(REDACT_TOKEN, new_texts[tool])
        out.append(
            Episode(
                episode_id=ep.episode_id,
                label=ep.label,
                tool_texts=new_texts,
                shortcut_tool=ep.shortcut_tool,
                shortcut_class=ep.shortcut_class,
            )
        )
    return out


def agent_accuracy(episodes: list[Episode]) -> float:
    preds = [frozen_agent_predict(ep) for ep in episodes]
    return float(np.mean([p == ep.label for p, ep in zip(preds, episodes)]))
