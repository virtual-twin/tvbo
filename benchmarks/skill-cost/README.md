# Skill-cost benchmark

Measures how much the shipped **TVBO skills** reduce the cost of an AI agent
*setting up a simulation experiment* — the tokens, tool calls, wall-clock, and
USD an agent burns to go from a plain request to a working whole-brain
simulation.

## The question

A coding agent with no domain skills discovers everything from scratch: it greps
the installed package to learn which models and connectomes exist and how `run`
is called. That exploration is where the tokens go. The TVBO skills replace it
with a few catalog calls (`Dynamics.list_db()`, `list_entries("Network")`,
`run("jax")`). This benchmark quantifies the difference.

## Conditions

Each cell runs the **same task** under one of three conditions; the only thing
that changes is skill availability:

| Condition | TVBO skills in workspace | Prompt mentions them |
|-----------|--------------------------|----------------------|
| `control`  | no  | no  |
| `implicit` | yes | no — the agent must trigger them from their descriptions |
| `explicit` | yes | yes — the prompt points the agent at them |

`implicit` vs `explicit` also tells us whether the skill **descriptions** are
good enough to auto-trigger without being named.

## What it measures

Per run: `processed_tokens` (summed input-side tokens incl. cache, across turns),
`output_tokens`, `tool_calls`, `wall_seconds`, `cost_usd`, and — crucially —
`success`, verified by **executing the agent's `solution.py` ourselves** and
checking it prints a real result shape. A fast give-up is a failure, not a cheap
win.

## Layout

```
harness/            pluggable agent runners
  base.py           Harness ABC + Task/RunResult/Metrics
  claude_code.py    Claude Code headless runner (implemented)
  codex.py          Codex runner (stub — see file for the plan)
tasks/
  whole_brain_sim.py  the task prompt + execution-based verifier
run_matrix.py       orchestrate conditions × reps, resumable
aggregate.py        medians → Markdown table + CSV + bar chart
results/            results.jsonl + generated table/csv/png
```

## Running

Use the repo virtualenv Python so the agent's `python`/`tvbo` resolve to an env
where `tvbo` is importable.

```bash
# smoke-test one cell first (spends a little usage)
.venv/bin/python benchmarks/skill-cost/run_matrix.py --cell explicit:0

# full matrix (3 conditions × 5 reps = 15 agent runs)
.venv/bin/python benchmarks/skill-cost/run_matrix.py --reps 5

# aggregate into the table + chart
.venv/bin/python benchmarks/skill-cost/aggregate.py
```

`run_matrix.py` is **resumable**: it skips cells already in `results.jsonl`.
Add `--force` to re-run, `--dry-run` to preview the plan, `--model <name>` to
change model (default `sonnet`), and `--max-turns` / `--timeout` to bound cost.

## Cost & fairness caveats

- **Real usage is spent.** Each agent run costs money/limits on your logged-in
  session. 3×5 is ~15 runs; budget accordingly and start with `--cell`.
- **Isolation** uses `--setting-sources project`, so the user's globally
  installed TVBO skills never leak into `control`. Auth is unaffected.
- **Editable install bias (conservative).** If `tvbo` is installed editable, the
  package source *is* the repo, so a `control` agent can read curated YAMLs
  directly — making control *cheaper* than a real `pip install` would. Any
  saving we still see is therefore a lower bound.
- **`bypassPermissions`** lets the agent run tools unattended in an isolated
  scratch dir outside the repo. Review the task before running.

## Adding a harness (e.g. Codex)

Implement `harness/base.Harness` (`prepare_workspace` + `run`) and register it in
`harness/__init__.HARNESSES`. Codex reads `AGENTS.md` rather than
`.claude/skills`, so its `prepare_workspace` installs skills via
`tvbo skills install --target agents-md --scope project`. See `harness/codex.py`.
