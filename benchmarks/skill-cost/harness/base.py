"""Harness abstraction for the skill-cost benchmark.

A *harness* is an agent runner (Claude Code, Codex, …). Each harness knows how to (a) prepare a scratch workspace for a given *condition* — which controls whether the TVBO skills are available — and (b) drive its agent on a *task* and report normalized :class:`RunResult` metrics.

The benchmark compares three conditions:

``control``   No TVBO skills present. The agent discovers everything from the installed package (the expensive baseline).
``implicit``  TVBO skills installed in the workspace but *not* mentioned in the prompt — the agent must trigger them from their descriptions.
``explicit``  TVBO skills installed *and* the prompt points the agent at them.
"""

from __future__ import annotations

import abc
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

CONDITIONS = ("control", "implicit", "explicit")


@dataclass
class Task:
    """A single benchmark task, identical across conditions except for the hint.

    Attributes:
        name: Short identifier used in result keys and filenames.
        prompt: The task text handed to every condition verbatim.
        explicit_hint: Appended to the system prompt only in the ``explicit`` condition — this is the *only* per-condition difference in what the agent is told.
        verify: ``(workdir) -> (success, detail)``. Run after the agent exits to check the task was actually accomplished, so a fast failure is not mistaken for a cheap success.
    """

    name: str
    prompt: str
    explicit_hint: str
    verify: Callable[[Path], tuple[bool, str]]


@dataclass
class RunResult:
    """Normalized metrics from one agent invocation (before verification)."""

    processed_tokens: int  # sum over turns of input-side tokens (incl. cache)
    output_tokens: int
    tool_calls: int
    wall_seconds: float
    cost_usd: float | None  # None if the harness does not report cost
    num_turns: int | None = None
    subtype: str = ""  # harness-reported completion status
    error: str = ""  # non-empty if the invocation itself failed


@dataclass
class Metrics:
    """A fully resolved benchmark row: one (harness, model, condition, rep)."""

    harness: str
    model: str
    condition: str
    rep: int
    success: bool
    processed_tokens: int
    output_tokens: int
    tool_calls: int
    wall_seconds: float
    cost_usd: float | None
    num_turns: int | None = None
    subtype: str = ""
    detail: str = ""  # verifier detail or error message

    @property
    def key(self) -> str:
        return f"{self.harness}/{self.model}/{self.condition}/{self.rep}"

    def to_row(self) -> dict:
        return asdict(self)


class Harness(abc.ABC):
    """Base class for an agent runner."""

    name: str
    wrapper: tuple[str, ...] = ()
    """A command the agent's command line is handed to, run in the run's directory, such as a script that starts it under ``sandbox-exec`` with its own HOME and TMPDIR to hide local data, code and other runs from a condition that must start from nothing."""
    skills_installer: Path = Path(sys.executable).with_name("tvbo")
    """The ``tvbo`` executable that renders the skills into a skilled workspace; by default the one beside the harness's Python, another one when the agent's environment has no tvbo, so the skills are tested on their own."""

    @abc.abstractmethod
    def prepare_workspace(self, workdir: Path, condition: str) -> None:
        """Set up *workdir* for *condition* (install/omit TVBO skills, etc.)."""

    @abc.abstractmethod
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
        """Drive the agent on *task* under *condition*; return raw metrics."""
