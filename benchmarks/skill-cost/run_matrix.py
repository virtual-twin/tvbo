#!/usr/bin/env python
"""Run the skill-cost benchmark matrix.

For each (harness, model, condition, rep) it spins up a fresh, repo-free scratch workspace, makes the TVBO skills available (or not, for ``control``), drives the agent on the task, verifies the result by executing the agent's script, and appends one normalized row to a JSONL results file. Re-running skips cells that already have a row (resume), so an interrupted matrix continues cleanly.

Run it with the repo's virtualenv Python so the agent's ``python`` / ``tvbo`` resolve to an environment where ``tvbo`` is importable:

    .venv/bin/python benchmarks/skill-cost/run_matrix.py --reps 5

Smoke-test a single cell first:

    .venv/bin/python benchmarks/skill-cost/run_matrix.py --cell explicit:0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harness import CONDITIONS, HARNESSES, Metrics  # noqa: E402
from tasks import TASKS  # noqa: E402

DEFAULT_RESULTS = SCRIPT_DIR / "results" / "results.jsonl"
DEFAULT_RUNS_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "tvbo-skillcost-runs"


def build_agent_env() -> dict[str, str]:
    """Inherit the current env but put the repo venv first so tvbo is importable."""
    env = dict(os.environ)
    venv_bin = REPO_ROOT / ".venv" / "bin"
    if venv_bin.exists():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(REPO_ROOT / ".venv")
        env.pop("PYTHONHOME", None)
    return env


def load_done_keys(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(f"{row['harness']}/{row['model']}/{row['condition']}/{row['rep']}")
    return done


def append_row(results_path: Path, metrics: Metrics) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metrics.to_row()) + "\n")


def run_cell(harness, task, condition, rep, *, model, runs_dir, max_turns, timeout, env) -> Metrics:
    workdir = runs_dir / f"{harness.name}_{model}_{condition}_{rep}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    harness.prepare_workspace(workdir, condition)
    run = harness.run(
        task,
        condition,
        workdir,
        model=model,
        max_turns=max_turns,
        timeout=timeout,
        env=env,
    )

    if run.error:
        success, detail = False, run.error
    else:
        try:
            success, detail = task.verify(workdir)
        except Exception as exc:  # verifier must never crash the matrix
            success, detail = False, f"verify raised: {exc!r}"

    return Metrics(
        harness=harness.name,
        model=model,
        condition=condition,
        rep=rep,
        success=success,
        processed_tokens=run.processed_tokens,
        output_tokens=run.output_tokens,
        tool_calls=run.tool_calls,
        wall_seconds=round(run.wall_seconds, 2),
        cost_usd=run.cost_usd,
        num_turns=run.num_turns,
        subtype=run.subtype,
        detail=detail,
    )


def parse_cells(cell_args, conditions, reps):
    """Expand --cell specs (``condition[:rep]``) or the full conditions×reps grid."""
    if cell_args:
        cells = []
        for spec in cell_args:
            cond, _, rep = spec.partition(":")
            if cond not in CONDITIONS:
                raise SystemExit(f"unknown condition {cond!r}; choose from {CONDITIONS}")
            cells.append((cond, int(rep) if rep else 0))
        return cells
    return [(c, r) for c in conditions for r in range(reps)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness", default="claude", choices=sorted(HARNESSES))
    ap.add_argument("--model", default="sonnet", help="Model alias/name passed to the harness (default: sonnet).")
    ap.add_argument("--task", default="whole_brain_sim", choices=sorted(TASKS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="Comma-separated subset of: " + ",".join(CONDITIONS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Run a single cell 'condition[:rep]'. Repeatable. Overrides --conditions/--reps.",
    )
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--timeout", type=float, default=1200.0, help="Per-agent wall-clock timeout in seconds.")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--force", action="store_true", help="Re-run cells even if already present in results.")
    ap.add_argument("--dry-run", action="store_true", help="List the cells that would run, then exit.")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cells = parse_cells(args.cell, conditions, args.reps)
    task = TASKS[args.task]
    harness = HARNESSES[args.harness]()
    done = set() if args.force else load_done_keys(args.results)
    env = build_agent_env()

    plan = [(c, r) for (c, r) in cells if args.force or f"{args.harness}/{args.model}/{c}/{r}" not in done]
    print(f"harness={args.harness} model={args.model} task={args.task}")
    print(f"planned cells: {len(plan)} (skipping {len(cells) - len(plan)} already done)")
    for c, r in plan:
        print(f"  - {c}:{r}")
    if args.dry_run or not plan:
        return

    for cond, rep in plan:
        print(f"\n=== running {cond}:{rep} ===", flush=True)
        metrics = run_cell(
            harness,
            task,
            cond,
            rep,
            model=args.model,
            runs_dir=args.runs_dir,
            max_turns=args.max_turns,
            timeout=args.timeout,
            env=env,
        )
        append_row(args.results, metrics)
        cost = f"${metrics.cost_usd:.4f}" if metrics.cost_usd is not None else "n/a"
        print(
            f"    success={metrics.success} "
            f"processed={metrics.processed_tokens} out={metrics.output_tokens} "
            f"tools={metrics.tool_calls} wall={metrics.wall_seconds}s cost={cost}\n"
            f"    detail: {metrics.detail}",
            flush=True,
        )

    print(f"\nwrote results → {args.results}")
    print(f"aggregate with: {sys.executable} {SCRIPT_DIR / 'aggregate.py'} --results {args.results}")


if __name__ == "__main__":
    main()
