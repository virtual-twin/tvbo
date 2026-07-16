"""Pluggable agent-harness registry for the skill-cost benchmark."""
from __future__ import annotations

from .base import CONDITIONS, Harness, Metrics, RunResult, Task
from .claude_code import ClaudeCodeHarness
from .codex import CodexHarness

HARNESSES: dict[str, type[Harness]] = {
    ClaudeCodeHarness.name: ClaudeCodeHarness,
    CodexHarness.name: CodexHarness,
}

__all__ = [
    "CONDITIONS",
    "Harness",
    "Metrics",
    "RunResult",
    "Task",
    "ClaudeCodeHarness",
    "CodexHarness",
    "HARNESSES",
]
