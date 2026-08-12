"""Claude Code headless harness.

Drives ``claude -p`` in ``stream-json`` mode and parses the event stream into
normalized :class:`RunResult` metrics.

Skill isolation relies on ``--setting-sources project``: only skills found under
the workspace's ``.claude/skills`` load, so the user's globally installed TVBO
skills never leak into the ``control`` condition. Auth and model selection are
unaffected by ``--setting-sources``.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .base import Harness, RunResult, Task


class ClaudeCodeHarness(Harness):
    name = "claude"

    def __init__(self, claude_bin: str = "claude"):
        self.claude_bin = claude_bin

    # -- workspace ---------------------------------------------------------
    def prepare_workspace(self, workdir: Path, condition: str) -> None:
        """Install the TVBO user skills into the workspace for skilled conditions.

        ``control`` gets nothing. ``implicit`` / ``explicit`` get the four
        shipped user skills rendered into ``<workdir>/.claude/skills`` via the
        package's own installer, so we exercise the exact files a user would get
        from ``tvbo skills install``.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        if condition == "control":
            return
        # `tvbo skills install --scope project` writes to <cwd>/.claude/skills.
        subprocess.run(
            [
                "tvbo",
                "skills",
                "install",
                "--target",
                "claude-code",
                "--scope",
                "project",
                "--force",
            ],
            cwd=str(workdir),
            check=True,
            capture_output=True,
            text=True,
        )

    # -- run ---------------------------------------------------------------
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
        cmd = [
            self.claude_bin,
            "-p",
            task.prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--max-turns",
            str(max_turns),
            "--setting-sources",
            "project",
            "--add-dir",
            str(workdir),
        ]
        if model:
            cmd += ["--model", model]
        if condition == "explicit":
            cmd += ["--append-system-prompt", task.explicit_hint]

        transcript = workdir / "_harness_transcript.jsonl"
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout
        except subprocess.TimeoutExpired as exc:
            wall = time.time() - t0
            transcript.write_text(exc.stdout or "", encoding="utf-8")
            return RunResult(0, 0, 0, wall, None, subtype="timeout", error=f"agent timed out after {timeout:.0f}s")
        wall = time.time() - t0
        transcript.write_text(stdout, encoding="utf-8")

        result = self._parse_stream(stdout)
        result.wall_seconds = wall
        if proc.returncode != 0 and not result.subtype:
            result.error = (proc.stderr or "")[-500:]
            result.subtype = f"exit_{proc.returncode}"
        return result

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _parse_stream(stdout: str) -> RunResult:
        """Fold a stream-json transcript into token / tool-call / cost metrics.

        The stream double-emits assistant events (a streaming start and a final
        copy), and per-event ``usage`` holds only streaming deltas — so neither
        can be summed. Authoritative aggregates live in the terminal ``result``
        event: ``modelUsage`` (per-model token totals, including a Haiku
        sub-agent) and ``total_cost_usd``. Tool calls are counted from distinct
        ``tool_use`` block ids, which dedupe the streaming duplicates cleanly.

        ``processed_tokens`` sums every input-side token (fresh input + both
        cache counts) across all models, so it reflects everything processed —
        matching how a "processed tokens" figure dwarfs raw output.
        """
        tool_ids: set[str] = set()
        cost = None
        processed = output = 0
        num_turns = None
        subtype = ""
        error = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "assistant":
                for block in ev.get("message", {}).get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_ids.add(block.get("id") or f"anon-{len(tool_ids)}")
            elif etype == "result":
                cost = ev.get("total_cost_usd", cost)
                num_turns = ev.get("num_turns", num_turns)
                subtype = ev.get("subtype", subtype)
                if ev.get("is_error"):
                    error = str(ev.get("result", ""))[:500]
                model_usage = ev.get("modelUsage") or {}
                if model_usage:
                    for mu in model_usage.values():
                        processed += (
                            mu.get("inputTokens", 0)
                            + mu.get("cacheReadInputTokens", 0)
                            + mu.get("cacheCreationInputTokens", 0)
                        )
                        output += mu.get("outputTokens", 0)
                else:  # fallback for older schemas without modelUsage
                    u = ev.get("usage", {}) or {}
                    processed = (
                        u.get("input_tokens", 0)
                        + u.get("cache_read_input_tokens", 0)
                        + u.get("cache_creation_input_tokens", 0)
                    )
                    output = u.get("output_tokens", 0)
        return RunResult(
            processed_tokens=processed,
            output_tokens=output,
            tool_calls=len(tool_ids),
            wall_seconds=0.0,
            cost_usd=cost,
            num_turns=num_turns,
            subtype=subtype,
            error=error,
        )
