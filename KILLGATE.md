# KILLGATE.md — tool-trace-audit (project #56)

Pre-registered 2026-10-21 BEFORE the final measurement run. No goalpost-moving:
thresholds below were locked after a 12-replica calibration (seeds 9000+) and
the final battery runs on fresh seeds (1000+). Calibration numbers are NOT
reported as results.

## Paper

arXiv 2610.25540 — "Auditing Tool Traces in Agentic LLMs: Source Localization
and Counterfactual Dependence Testing". Two-stage audit: (1) PT2D localizes
label-predictive shortcuts to individual tool results; (2) Token-Class
Redaction counterfactually tests the frozen model's dependence.

## Gate 1 — author code absent

- arXiv abs page fetched live 2026-10-21: `<title>` tag =
  "[2610.25540] Auditing Tool Traces in Agentic LLMs: Source Localization and
  Counterfactual Dependence Testing" (id taken from the title tag).
- Zero `github.com` links on the abs page. GitHub repo search for
  "2610.25540": total_count = 0.
- **PASS.** First open implementation claim stands.

## Gate 2 — disease-first (pre-registered)

Setup: 200 synthetic episodes (seed 424242), frozen scripted agent whose only
information channel is the planted shortcut (dataset ID or template artifact).

- PASS iff agent accuracy with shortcuts ≥ 0.90
  AND accuracy drop after token-class redaction ≥ 30pp.
- KILL otherwise (disease does not reproduce at this scale; no forcing).

## Gate 3 — load-bearing (pre-registered)

Setup: 100 audit replicas (seeds 1000..1099, 120 episodes each), shortcut tool
and shortcut class randomized per replica.

- PASS iff PT2D top-1 tool-localization rate ≥ 0.80
  AND paired bootstrap (10k resamples) 95% CI for
  (PT2D rate − concatenated-TFIDF baseline rate) has lower bound > 0.
- KILL otherwise.

The baseline is the naive faithful choice: one TF-IDF logistic on the
concatenated trace, then token-level |coefficient| mass attributed back to the
tool each token came from (saliency-style attribution). The dilution mechanism
is genuine: 150-token decoy tools with weak correlated signal vs a 7-token
shortcut text — coefficient mass sums over token counts, burying the single
shortcut token. Documented as a property of this attribution, not of all
possible concatenated analyses.

## Measured verdict (final run 2026-10-21, seeds 1000-1099)

- Gate 1: **PASS** (live-verified above).
- Gate 2: **PASS** — acc 1.00 >= 0.90; redacted 0.54; collapse **46.0pp** >= 30pp
  (n=200, seed 424242).
- Gate 3: **PASS** — PT2D top-1 rate **1.00** (100/100) >= 0.80; paired
  bootstrap 95% CI for (PT2D − baseline) = **[1.00, 1.00]**, lower bound > 0
  (baseline rate 0.00; 10k resamples).
- Stage-2 per-tool ablation clean: collapse >= 30pp only at the true shortcut
  tool, 0.0pp at every other tool.
- 25/25 pytest green.

## Prior art (honest table)

| Work | What it does | Gap this fills |
|---|---|---|
| arXiv 2610.25540 (the paper) | PT2D + Token-Class Redaction on TCR-pMHC agent traces | No public code (Gate 1); LLM-dependent stage 2 |
| Model auditing / saliency attribution | Token/feature importance on concatenated inputs | No per-tool decomposition; dilution unaddressed |
| Shortcut-learning literature (e.g. dataset bias audits) | Finds spurious correlations in training data | Not aimed at multi-tool agent traces at inference time |
| MCP/agent observability tooling | Trace logging, replay | No label-predictive localization of shortcuts |

Novel claim, scoped honestly: first OPEN, deterministic, zero-LLM PT2D audit
harness — run any agent's serialized traces through it and find which tool is
secretly answering the question. Stage 2 uses a frozen SCRIPTED agent, not a
frozen LLM (stated everywhere); the TCR-pMHC biology result is NOT claimed.

## Limits

- Synthetic traces with planted shortcuts; real agent traces are noisier.
- Scripted agent stands in for a frozen LLM — dependence testing is exact
  here, approximate in the wild.
- Baseline is one naive attribution; stronger concatenated attributions
  (e.g. per-tool ablation of the joint model) may dilute less — future work.
