"""Benchmark task: set up and run a whole-brain simulation experiment.

The task is deliberately one where *discovery* dominates cost: the agent must find a curated multi-node connectome and the correct ``run`` API. A control
agent greps the installed package; a skilled agent calls ``list_entries`` /
``run("jax")`` straight away.

Success is verified by *executing the agent's script ourselves* and checking it prints a real result shape — so a fast give-up cannot masquerade as a cheap win.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from harness.base import Task

PROMPT = """\
Using the `tvbo` Python package (already installed in this environment), write a \
script named `solution.py` in the current working directory that sets up and runs \
a whole-brain simulation experiment:

- Use the `ReducedWongWangExcInh` mean-field neural-mass model.
- Place it on a curated multi-node structural connectome from tvbo's own database \
(any network with 2 or more nodes is fine — pick one that exists).
- Use linear coupling and the Heun integration method, with duration 500.
- Run the experiment on the JAX backend.
- At the very end, print EXACTLY one line of the form:
    RESULT_SHAPE=<d0>,<d1>,<d2>
  where d0,d1,d2 are the three dimensions of the simulation result's data array.

Then run `python solution.py` yourself to confirm it executes and prints that line. \
Do not ask me any questions — make reasonable choices and finish autonomously.
"""

EXPLICIT_HINT = """\
This project ships TVBO skills as installed skills: `tvbo-overview`, \
`tvbo-writing-models`, `tvbo-running-simulations`, and `tvbo-platform`. Before \
exploring the installed package or the filesystem, consult the relevant TVBO \
skill — it tells you how to discover curated components and the exact run API.
"""

_SHAPE_RE = re.compile(r"RESULT_SHAPE=(\d+),(\d+),(\d+)")


def verify(workdir: Path) -> tuple[bool, str]:
    """Execute the agent's ``solution.py`` and confirm it produced a real result."""
    script = workdir / "solution.py"
    if not script.exists():
        return False, "solution.py not found"
    src = script.read_text(encoding="utf-8", errors="replace")
    if "tvbo" not in src or ".run(" not in src:
        return False, "solution.py does not import tvbo / call .run()"
    try:
        proc = subprocess.run(
            [sys.executable, "solution.py"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "solution.py timed out on verification run (>900s)"
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = _SHAPE_RE.search(out)
    if not m:
        tail = out.strip()[-300:]
        return False, f"no RESULT_SHAPE line (exit {proc.returncode}); tail: {tail!r}"
    dims = tuple(int(g) for g in m.groups())
    if dims[2] < 2 or dims[0] < 2:
        return False, f"implausible result shape {dims} (need >=2 nodes, >=2 timepoints)"
    return True, f"shape={dims}"


TASK = Task(
    name="whole_brain_sim",
    prompt=PROMPT,
    explicit_hint=EXPLICIT_HINT,
    verify=verify,
)
