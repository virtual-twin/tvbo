---
applyTo: '**'
---

# Writing Code Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Readable Over Annotated

**Let names, structure, and docstrings carry intent — heavy comments disturb readability more than they help.**

- Document a function with a proper docstring, not a stack of leading `#` lines.
- A block of explanatory comments above code is a smell: extract a well-named helper, or simplify the code, so the comment isn't needed.
- Reserve inline `#` for a single genuinely non-obvious line (a subtle invariant, a workaround) — not a running narration. If it takes a paragraph, the code is too clever; rewrite the code.
- Same budget in codegen templates (Mako `##` / `<% %>`): terse code beats an annotated wall.
- **When you change a line that already carries a comment, rewrite that comment to describe the new state — do not append a second one beside it.** Layering is what turns a three-line explanation into a twelve-line record where each layer is true and the stack no longer says what the code does.
- A comment states the code's current contract, addressed to a reader who has never seen your diff — never "Previously…", "This replaces…" or "for now". History belongs in `CHANGELOG.md`, reasoning in the commit message.
- Aim for maximally readable: someone should follow the code without the comments.

## 6. Use the Dependency's Own API

**Before hand-rolling a mechanism around a library, read what the library already exposes.**

A helper that duplicates a dependency's feature is worse than no helper: it drifts from the
library's semantics, misses its later fixes, and hides that the sanctioned path exists.

- Skim the module you are about to wrap. `optim/callbacks.py`, `types.py`, an `__init__`
  export list — a minute of reading beats a plausible reimplementation.
- Symptoms you are about to duplicate something: you are tracking "best so far", retrying,
  logging progress, early-stopping, or caching. Frameworks almost always ship these.
- If the library's version really does not fit, say why in the docstring, so the next
  reader knows it was a decision rather than an oversight.

**tvboptim specifically** — fitting goes through `OptaxOptimizer` and its callbacks; do not
re-implement them:

| need | use |
|---|---|
| keep the best-scoring state, not the last | `SaveBestSeenCallback` → `fitting_data["best"]` |
| several callbacks at once | `MultiCallback([...])` |
| stop when the loss plateaus | `StopConvergenceCallback(patience, min_delta)` |
| stop at a target loss / wall-clock | `StopLossCallback`, `StopTimeCallback` |
| record loss or parameter history | `SavingLossCallback`, `SavingParametersCallback` |

`opt.run()` returns `(final_state, fitting_data)`. **`final_state` is the LAST state, not the
best one.** Any non-monotone trajectory — an optimiser that overshoots, or a penalty term that
trades one component for another — makes those different, and taking the last silently reports
a worse model than the run found.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
