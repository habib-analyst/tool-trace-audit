"""Stage 1: Per-Tool TF-IDF Decomposition (PT2D) + concatenated baseline.

PT2D: fit a label-predictive model on each tool's text ALONE and rank tools
by cross-validated AUC. This avoids the signal dilution that occurs when the
whole trace is concatenated into one document before vectorization.

Baseline (concatenated-TFIDF): vectorize the full concatenated trace, fit one
model, then attribute token-level coefficient mass back to tools. The decoy
tools' many weakly-predictive tokens dilute the shortcut token's mass.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from .synth import TOOLS, Episode

TOKEN_PATTERN = r"(?u)\S+"  # keep tokens like sid=XJ-4407 and [TEMPL_v3] whole


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, lowercase=True, sublinear_tf=True, min_df=2
    )


def _cv_auc(X, y, seed: int = 0) -> float:
    clf = LogisticRegression(C=1.0, max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    except ValueError:
        return 0.5
    return float(roc_auc_score(y, p))


def pt2d_scores(episodes: list[Episode]) -> dict[str, float]:
    """Per-tool cross-validated AUC of label prediction from that tool's text."""
    y = np.array([ep.label for ep in episodes])
    scores: dict[str, float] = {}
    for tool in TOOLS:
        texts = [ep.tool_texts[tool] for ep in episodes]
        vec = _vectorizer()
        X = vec.fit_transform(texts)
        scores[tool] = _cv_auc(X, y)
    return scores


def pt2d_localize(episodes: list[Episode]) -> tuple[str, dict[str, float]]:
    """Return (top tool, per-tool scores)."""
    scores = pt2d_scores(episodes)
    top = max(scores, key=lambda t: scores[t])
    return top, scores


def pt2d_top_tokens(episodes: list[Episode], tool: str, k: int = 5) -> list[str]:
    """Most label-predictive tokens inside one tool's text (for the report)."""
    y = np.array([ep.label for ep in episodes])
    texts = [ep.tool_texts[tool] for ep in episodes]
    vec = _vectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    coefs = np.asarray(clf.coef_[0])
    order = np.argsort(-np.abs(coefs))[:k]
    names = np.array(vec.get_feature_names_out())
    return [str(names[i]) for i in order]


def concatenated_baseline_localize(episodes: list[Episode]) -> tuple[str, dict[str, float]]:
    """Baseline: one TF-IDF model on the concatenated trace, attribute
    token-level |coefficient| mass back to the tool each token came from."""
    y = np.array([ep.label for ep in episodes])
    docs = [" ".join(ep.tool_texts[t] for t in TOOLS) for ep in episodes]
    vec = _vectorizer()
    X = vec.fit_transform(docs)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    mass = np.abs(np.asarray(clf.coef_[0]))

    scores: dict[str, float] = {t: 0.0 for t in TOOLS}
    for ep in episodes:
        for t in TOOLS:
            toks = set(ep.tool_texts[t].lower().split())
            idx = [i for i, w in enumerate(names) if w in toks]
            if idx:
                scores[t] += float(mass[idx].sum())
    n = len(episodes)
    scores = {t: s / n for t, s in scores.items()}
    top = max(scores, key=lambda t: scores[t])
    return top, scores
