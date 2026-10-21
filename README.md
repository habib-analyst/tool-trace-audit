# tool-trace-audit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

The first open, deterministic, zero-LLM audit harness for finding which tool in an
agentic LLM's trace is secretly answering the question. Implements the two-stage
audit from arXiv 2610.25540: **Per-Tool TF-IDF Decomposition (PT2D)** localizes
label-predictive shortcuts (dataset IDs, template artifacts) to individual tool
results without the signal dilution of concatenated traces, and **counterfactual
dependence testing** redacts the localized token class to prove the frozen agent
actually depends on it.

## Quickstart

```bash
pip install tool-trace-audit
```

```python
from tool_trace_audit import generate_traces, pt2d_localize, agent_accuracy, redact_token_class

eps = generate_traces(n_episodes=120, seed=20261021)
top, scores = pt2d_localize(eps)
print("truth:     ", eps[0].shortcut_tool)
print("PT2D top-1:", top)
a1 = agent_accuracy(eps)
a2 = agent_accuracy(redact_token_class(eps))
print(f"accuracy: {a1:.2f} -> {a2:.2f} after redaction")
```

Real output:

```
truth:      kb_lookup
PT2D top-1: kb_lookup
accuracy: 1.00 -> 0.53 after redaction
```

## Measured results

Pre-registered kill-gates, measured 2026-10-21 on CPU (scikit-learn, no LLM APIs,
$0). Full protocol in [KILLGATE.md](KILLGATE.md).

| Gate | Bar (pre-registered) | Measured | Verdict |
|---|---|---|---|
| 1. Author code absent | 0 public implementations | 0 GitHub hits for 2610.25540; 0 code links on abs page (live check) | PASS |
| 2. Disease-first | agent acc >= 0.90; collapse >= 30pp on redaction | acc 1.00, redacted 0.54, **collapse 46.0pp** (n=200, seed 424242) | PASS |
| 3. Load-bearing | PT2D top-1 rate >= 0.80; paired bootstrap 95% CI excludes baseline | PT2D **1.00** vs concatenated-TFIDF **0.00**; delta CI **[1.00, 1.00]** (100 replicas, 10k resamples) | PASS |

Stage-2 per-tool ablation: redacting the shortcut class inside the true tool
collapses accuracy by 46+pp; redacting any other tool changes nothing (0.0pp).
25/25 tests green.

## How it works

```
trace = [(tool, text), ...]            # serialized tool results
Stage 1 (PT2D):
  for each tool: TF-IDF(tool texts) -> logistic CV AUC vs label
  localize = argmax AUC                 # the shortcut tool scores ~1.0
Stage 2 (counterfactual):
  redact the localized token class (dataset IDs / template artifacts)
  re-run the frozen agent -> accuracy collapse proves dependence
```

The dilution mechanism the baseline suffers is real, not a strawman: decoy
tools carry long texts (150 tokens) with weak label-correlated signal, while
the shortcut lives in a 7-token result. Naive token-level attribution on the
concatenated trace sums coefficient mass over token counts and buries the
single shortcut token (baseline top-1 rate 0.00). PT2D fits per tool, where the
shortcut dominates its own document (AUC 1.00 vs 0.52-0.71 for decoys).

## Limits

- Synthetic traces with planted shortcuts; the frozen agent is a deterministic
  scripted policy standing in for a frozen LLM (stated in code and paper).
- The TCR-pMHC biology result from the paper is NOT claimed or reproduced.
- Baseline is one naive saliency-style attribution; stronger concatenated
  attributions may dilute less. See [KILLGATE.md](KILLGATE.md) for the honest
  prior-art table.

## Citation

```bibtex
@software{rehman2026tooltraceaudit,
  author  = {Habib Ur Rehman},
  title   = {tool-trace-audit: Open PT2D Audit Harness for Tool Traces in Agentic LLMs},
  year    = {2026},
  url     = {https://github.com/habib-analyst/tool-trace-audit},
  note    = {Open implementation of the two-stage audit from arXiv:2610.25540}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
