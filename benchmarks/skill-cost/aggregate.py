#!/usr/bin/env python
"""Aggregate skill-cost benchmark results into a median table + chart.

Reads the JSONL written by ``run_matrix.py`` and emits, grouped by
(harness, model, condition):

  * a Markdown median table (processed/output tokens, tool calls, wall seconds,
    cost USD) plus success rate and N — printed and saved,
  * a CSV of the same,
  * a grouped bar chart of median cost and median processed tokens by condition.

    .venv/bin/python benchmarks/skill-cost/aggregate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results" / "results.jsonl"

# Canonical difficulty gradient: no skills → skills present → skills pointed at.
CONDITION_ORDER = ["control", "implicit", "explicit"]

METRICS = [
    ("processed_tokens", "Processed tokens median"),
    ("output_tokens", "Output tokens median"),
    ("tool_calls", "Tool calls median"),
    ("wall_seconds", "Wall seconds median"),
    ("cost_usd", "Cost USD median"),
]


def load_rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}; run run_matrix.py first")
    rows = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--only-success", action="store_true", help="Aggregate only rows where success=True.")
    ap.add_argument("--out-prefix", default=None, help="Output path prefix (default: alongside --results).")
    args = ap.parse_args()

    import pandas as pd

    df = pd.DataFrame(load_rows(args.results))
    if df.empty:
        raise SystemExit("results file is empty")
    if args.only_success:
        df = df[df["success"]]
        if df.empty:
            raise SystemExit("no successful rows to aggregate")

    df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER, ordered=True)
    grp = df.groupby(["harness", "model", "condition"], observed=True)

    agg = (
        grp.agg(
            **{col: (col, "median") for col, _ in METRICS},
            success_rate=("success", "mean"),
            n=("success", "size"),
        )
        .reset_index()
        .sort_values(["harness", "model", "condition"])
    )

    # ---- Markdown table (mirrors the reference layout) -------------------
    headers = ["Harness", "Model", "Condition"] + [label for _, label in METRICS] + ["Success", "N"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in agg.iterrows():
        cost = r["cost_usd"]
        cost_s = f"{cost:.4f}" if pd.notna(cost) else "n/a"
        cells = [
            str(r["harness"]),
            str(r["model"]),
            str(r["condition"]),
            f"{r['processed_tokens']:.0f}",
            f"{r['output_tokens']:.0f}",
            f"{r['tool_calls']:.0f}",
            f"{r['wall_seconds']:.1f}",
            cost_s,
            f"{r['success_rate'] * 100:.0f}%",
            f"{int(r['n'])}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    table_md = "\n".join(lines)
    print(table_md)

    prefix = Path(args.out_prefix) if args.out_prefix else args.results.with_name("skill_cost")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".table.md").write_text(table_md + "\n", encoding="utf-8")
    agg.to_csv(prefix.with_suffix(".csv"), index=False)

    _plot(agg, prefix.with_suffix(".png"))
    print(f"\nwrote:\n  {prefix.with_suffix('.table.md')}\n  {prefix.with_suffix('.csv')}\n  {prefix.with_suffix('.png')}")


def _plot(agg, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    combos = agg[["harness", "model"]].drop_duplicates().itertuples(index=False)
    combos = list(combos)
    conditions = [c for c in CONDITION_ORDER if c in set(agg["condition"].astype(str))]
    x = np.arange(len(conditions))
    width = 0.8 / max(len(combos), 1)

    fig, (ax_cost, ax_tok) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (harness, model) in enumerate(combos):
        sub = agg[(agg["harness"] == harness) & (agg["model"] == model)]
        sub = sub.set_index(sub["condition"].astype(str))
        cost = [sub.loc[c, "cost_usd"] if c in sub.index else np.nan for c in conditions]
        proc = [sub.loc[c, "processed_tokens"] if c in sub.index else np.nan for c in conditions]
        label = f"{harness}/{model}"
        ax_cost.bar(x + i * width, cost, width, label=label)
        ax_tok.bar(x + i * width, proc, width, label=label)

    offset = width * (len(combos) - 1) / 2
    for ax, title, ylab in (
        (ax_cost, "Median cost per run", "USD"),
        (ax_tok, "Median processed tokens per run", "tokens"),
    ):
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks(x + offset)
        ax.set_xticklabels(conditions)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("TVBO skills: cost of setting up a simulation experiment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
