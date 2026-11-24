"""Benchmark tasks: set up and run a whole-brain simulation experiment, once naming tvbo and once naming no package.

The task is deliberately one where *discovery* dominates cost: the agent must find a curated multi-node connectome and the correct ``run`` API. A control agent greps the installed package; a skilled agent calls ``list_entries`` / ``run("jax")`` straight away.

Success is verified by *executing the agent's script ourselves* and checking it prints a real result shape — so a fast give-up cannot masquerade as a cheap win.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import partial
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
This project ships TVBO skills as installed skills: `tvbo-overview`, `tvbo-writing-models`, `tvbo-running-simulations`, and `tvbo-platform`. Before exploring the installed package or the filesystem, consult the relevant TVBO skill — it tells you how to discover curated components and the exact run API.
"""

GENERAL_PROMPT = """\
Write a script named `solution.py` in the current working directory that sets up and runs a whole-brain simulation experiment:

- Use the reduced Wong-Wang excitatory-inhibitory mean-field neural-mass model.
- Place it on a published human structural connectome with 2 or more nodes: real connectivity data, not random weights.
- Use linear coupling and the Heun integration method, with duration 500 ms.
- At the very end, print EXACTLY one line of the form:
    RESULT_SHAPE=<d0>,<d1>,<d2>
  where d0,d1,d2 are the three dimensions of the simulated data array (time points, state variables, nodes).

The Python environment is already set up; do not install packages. Then run `python solution.py` yourself to confirm it executes and prints that line. Do not ask me any questions — make reasonable choices and finish autonomously.
"""

MVP_PROMPT = "Simulate one second of resting-state activity of the reduced Wong-Wang excitatory-inhibitory model on the Desikan-Killiany human structural connectome, with a typical noise level. Write the script as `solution.py`, run it, and save a figure of every region's time series as `result.png`. Do not ask me any questions.\n"

_SHAPE_RE = re.compile(r"RESULT_SHAPE=(\d+),(\d+),(\d+)")
_TIMED_OUT = "solution.py timed out on verification run (>900s)"


def _run_solution(workdir: Path) -> tuple[int, str] | None:
    """Run the agent's ``solution.py`` in *workdir* as the verification run: its exit code and its stdout and stderr joined, or ``None`` when it outlasts the 900 s limit."""
    try:
        proc = subprocess.run([sys.executable, "solution.py"], cwd=str(workdir), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def verify(workdir: Path, require_tvbo: bool = True) -> tuple[bool, str]:
    """Execute the agent's ``solution.py`` and confirm it produced a real result.

    With ``require_tvbo`` a script that does not import tvbo and call ``.run()`` fails; without it any script that prints a plausible shape passes, and the detail says whether it used tvbo or custom code.
    """
    script = workdir / "solution.py"
    if not script.exists():
        return False, "solution.py not found"
    src = script.read_text(encoding="utf-8", errors="replace")
    uses_tvbo = bool(re.search(r"^\s*(import tvbo|from tvbo)", src, flags=re.M))
    if require_tvbo and (not uses_tvbo or ".run(" not in src):
        return False, "solution.py does not import tvbo / call .run()"
    run = _run_solution(workdir)
    if run is None:
        return False, _TIMED_OUT
    code, out = run
    m = _SHAPE_RE.search(out)
    if not m:
        tail = out.strip()[-300:]
        return False, f"no RESULT_SHAPE line (exit {code}); tail: {tail!r}"
    dims = tuple(int(g) for g in m.groups())
    if dims[2] < 2 or dims[0] < 2:
        return False, f"implausible result shape {dims} (need >=2 nodes, >=2 timepoints)"
    return True, f"shape={dims}; {'tvbo' if uses_tvbo else 'custom code'}"


def verify_figure(workdir: Path) -> tuple[bool, str]:
    """Execute the agent's ``solution.py`` and confirm it draws ``result.png`` afresh; the detail says which of tvbo and bsplot the script imports, since what the figure shows is judged by eye."""
    script, figure = workdir / "solution.py", workdir / "result.png"
    if not script.exists():
        return False, "solution.py not found"
    src = script.read_text(encoding="utf-8", errors="replace")
    used = [pkg for pkg in ("tvbo", "bsplot") if re.search(rf"^\s*(import|from) {pkg}\b", src, flags=re.M)] or [
        "neither tvbo nor bsplot"
    ]
    agent_figure = figure.with_name("result.agent.png")
    if figure.exists():
        figure.rename(agent_figure)
    run = _run_solution(workdir)
    if run is None:
        return False, _TIMED_OUT
    code, out = run
    if code != 0 or not figure.exists():
        return False, f"no result.png (exit {code}); tail: {out.strip()[-300:]!r}"
    return True, f"result.png {figure.stat().st_size // 1024} kB; {', '.join(used)}"


TASK = Task(
    name="whole_brain_sim",
    prompt=PROMPT,
    explicit_hint=EXPLICIT_HINT,
    verify=verify,
)

TASK_GENERAL = Task(
    name="whole_brain_sim_general",
    prompt=GENERAL_PROMPT,
    explicit_hint=EXPLICIT_HINT,
    verify=partial(verify, require_tvbo=False),
)

TASK_MVP = Task(
    name="whole_brain_sim_mvp",
    prompt=MVP_PROMPT,
    explicit_hint=EXPLICIT_HINT,
    verify=verify_figure,
)
