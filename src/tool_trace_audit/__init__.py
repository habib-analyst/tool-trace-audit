"""tool-trace-audit: open, deterministic, zero-LLM PT2D audit harness."""

from .synth import (
    TOOLS,
    generate_traces,
    frozen_agent_predict,
    redact_token_class,
    redact_tool_text,
    agent_accuracy,
)
from .pt2d import pt2d_scores, pt2d_localize, concatenated_baseline_localize
from .audit import audit_replica, paired_bootstrap_ci, measure_gates

__all__ = [
    "TOOLS", "generate_traces", "frozen_agent_predict", "redact_token_class",
    "redact_tool_text", "agent_accuracy", "pt2d_scores", "pt2d_localize",
    "concatenated_baseline_localize", "audit_replica", "paired_bootstrap_ci",
    "measure_gates",
]
__version__ = "0.1.0"
