"""Codex harness — stub.

Wiring for the OpenAI Codex CLI. Not implemented yet: Codex is not installed in
the reference environment and needs a separate key. The interface matches
:class:`~harness.base.Harness` so it drops into ``run_matrix`` once implemented.

To implement:
  * ``prepare_workspace``: Codex reads ``AGENTS.md`` rather than ``.claude/skills``.
    Install the skills as an ``AGENTS.md`` index for skilled conditions via
    ``tvbo skills install --target agents-md --scope project`` (control: omit).
  * ``run``: drive ``codex exec`` non-interactively with a JSON/experimental
    output mode and parse tokens/cost/turns from its event log.
"""
from __future__ import annotations

from pathlib import Path

from .base import Harness, RunResult, Task


class CodexHarness(Harness):
    name = "codex"

    def prepare_workspace(self, workdir: Path, condition: str) -> None:
        raise NotImplementedError(
            "CodexHarness is a stub. Install the Codex CLI and implement "
            "AGENTS.md-based skill injection + event-log parsing."
        )

    def run(
        self,
        task: Task,
        condition: str,
        workdir: Path,
        *,
        model: str,
        max_turns: int,
        timeout: float,
        env: dict[str, str],
    ) -> RunResult:
        raise NotImplementedError("CodexHarness is a stub.")
