---
name: writing-reports
description: "How to write a TVBO replication report \u2014 one IMRAD Quarto document\
  \ whose every number is computed from the run (never transcribed), whose equations\
  \ render natively from the recipe, that states negative results honestly, keeps\
  \ a copyright-safe internal/public figure split, and holds its prose to a strict\
  \ anti-slop standard."
---

# Writing Reports in TVBO

This skill owns the **report** for a study replication: one `report/report-src.qmd` that
reads like a paper and computes every number it prints. It is the reporting layer that
**replicating-studies** composes; that skill decides *which* targets you replicate and
their fidelity tier, and **running-simulations** covers how the runs produce the
containers you read here. Start from the report template in the replicating-studies
skeleton (`report-src.qmd.tmpl`) and keep its metrics cell, `ab()` helper, and profile
machinery.

Four rules carry the whole report. Break any one and the report stops being trustworthy.

1. **IMRAD, not a figure walk.**
2. **Every number is computed from the run — none is typed by hand.**
3. **Equations come from the recipe metadata, not a hand transcription.**
4. **A negative result is stated plainly, with its evidence, and never softened.**

## Structure: IMRAD

The section order is fixed for every study: **Abstract · Introduction · Methods ·
Results · Discussion · Conclusion · References**. Set `number-sections: true`; mark
Abstract and References `{.unnumbered}`. Map the pieces to sections:

- **Methods** carries the native equation render, the variants, coupling, network and
  data provenance, the analyses, the backend, and the verification against an
  independent reference.
- **Results** opens with the computed comparison/scorecard table, then one subsection
  per paper figure: a sentence or two on what the panel shows, the `ab()` call, and a
  status callout.
- **Discussion** interprets: what reproduced and why, the mechanism and downstream
  consequences of any negative result, the reproduction-vs-replication framing, and the
  accepted limitations.

State a negative result in **Results** (with the evidence) and interpret it in
**Discussion**. When one modelling fact explains several panel mismatches, say so once
instead of listing them as separate failures.

## Every number is computed, never transcribed

Open each result container once in a setup cell and compute every reported quantity into
one dict `M`. Reference those values in prose and captions through inline
`` `{python} …` `` — this works inside figure captions too. A number typed into prose is
a bug, whether it is a count, a decay time, a bifurcation threshold, a correlation, or a
fitted parameter. Papers are not ground truth, and neither are your own asserted numbers;
a recomputed value that *differs* from the paper is honest, a typed one that matches is
not.

**The rule is asymmetric — keep the two kinds of number straight.** A quantity is either
*yours* (a result/metric from the run → MUST be inline `` `{python} M[...]` ``) or *the
paper's* (a value you quote for comparison — "paper: 9 solitary", "±105 MW", "$t_c$ 2.58 s"),
which MUST stay a **literal**: you cannot recompute someone else's number, and dressing it up
as computed would be the lie. Layout config and rejected counterfactuals (a `dt=0.01`
alternative you *didn't* use) stay literal too. So "nothing hardcoded" means *every value of
yours computed, every value of theirs quoted* — a hardcoded paper value is correct; a hardcoded
result is the bug.

**Audit it before shipping — the rule does not enforce itself.** A report can read as fully
computed (dozens of inline values, a fat `M`) and still hide a typed result. Grep the prose for
numeric literals (strip the `` `{python}` `` spans and fenced code), and classify each as
yours-or-theirs; a stray decimal that is *your* spectral peak, peak *location*, decay time, or
solver step is the bug — replace it with an `M[...]` entry. And don't excuse one as a soft
"≈600 MW" because computing it needs a cross-experiment merge (an ordinal-keyed Lyapunov/branch
run read against a scan's parameter axis) — add the helper and compute it; an "≈" sitting next
to a typed number in prose is usually the tell.

```python
from tvbo.classes.study import SimulationStudy
STUDY = SimulationStudy.from_file("../<Study>.yaml")
EXP = STUDY.get_experiment(<base_exp_id>)
# read output/nc/exp*/…h5 containers, reduce to M[...] once, reference via inline `{python}`
```

Build the comparison table's verdict column from the data as well: derive "reproduces" /
"does not" from `M`, do not assert it. A placeholder panel (TVBO data not yet run) gets a
labelled placeholder and a "missing" callout, never a green one — showing the paper's own
replotted arrays as your result crosses the integrity line.

## Equations and parameters rendered natively from the recipe

Render the model from the same metadata the backend compiles, so the mathematics shown is
the mathematics that runs:

```python
print(EXP.dynamics.render("markdown"))                    # equations only
print(EXP.dynamics.generate_report(format="markdown"))    # equations + parameter table
print(CTRL.dynamics.render("markdown", baseline=EXP.dynamics))  # only the delta vs base
```

Put `generate_report` in Methods so each auxiliary equation and every parameter checks
one-to-one against the paper's equations and tables. Render a controlled variant relative
to the base so only the changed terms appear.

## State a negative result honestly

An honest replication reproduces what is real and explains what is not. If a claim does
not reproduce, establish it to the standard of a finding before you call it one: an
independent reference, a positive control that proves your method can detect the effect
when it is present, and the mechanism. Report the cause, not just the mismatch. Do not
dress a non-reproduction as a partial success, and do not bury it — put it where a reader
looking for that result will find it.

## Status callouts: three colours, no emoji

Set `callout-icon: false`. Give each figure one short callout:

- **green** (`.callout-note` / `.callout-tip`): what reproduced, with an inline-computed
  number.
- **yellow** (`.callout-warning`): what is *missing* because the data or target is not
  yet available (a placeholder panel gets this).
- **red** (`.callout-important`): what was *attempted and failed to match*.

One or two sentences each. The colour carries the verdict; the sentence carries the
evidence.

## Copyright-safe internal/public split — ONE source, `report-src.qmd`

The report shows A/B panels (the paper's published figure beside your reproduction), but
the paper's figures are copyright-restricted and must never be committed or shared. Keep
**one** source file, named `report-src.qmd`, that renders two ways, selected by a Quarto
profile — never a second report file:

- The `ab()` helper reads `INTERNAL = "internal" in
  os.environ.get("QUARTO_PROFILE","").split(",")` and draws the paper original **only when
  `INTERNAL`**, so the public build never opens the copyrighted file.
- Copy `assets/_quarto.yml.tmpl` → `report/_quarto.yml` (sets `output-file: report.pdf`)
  and `assets/_quarto-internal.yml.tmpl` → `report/_quarto-internal.yml` (sets
  `output-file: report_internal.pdf`).
- **PUBLIC** (default, shareable): `quarto render report-src.qmd --to pdf` → `report.pdf`,
  reproduction only. **INTERNAL** (opt-in, git-ignored): `quarto render report-src.qmd
  --to pdf --profile internal` → `report_internal.pdf`, with the originals.
- **Why `report-src.qmd`, not `report.qmd`:** Quarto's LaTeX/typst step always writes
  `<input-stem>.pdf` first, then renames it to `output-file`. With a `report.qmd` source,
  the internal build's `report.pdf` intermediate would overwrite the public `report.pdf`
  before being renamed to `report_internal.pdf` — so the two builds clobber each other. A
  distinct source stem (`report-src`) makes each build's `report-src.pdf` intermediate
  transient, so both finals coexist and the render order does not matter. `--profile
  internal` can only ever write `report_internal.pdf`, so a copyrighted build can never
  overwrite the public `report.pdf`. Track `report-src.qmd` and the two `_quarto*.yml`;
  git-ignore `report/*.pdf` and the paper's figures under `original_study/`.

Verify by rendering both and confirming the public PDF embeds no wide A/B composite
(public figure aspect ratios stay near 1.0–1.5; internal A/B composites are wide, ~2.2+).

## PDF gotchas (xelatex)

- Write all math, subscripts, and superscripts as **LaTeX**, never Unicode — xelatex
  silently drops Unicode math. Use `$J_{NMDA}$`, `$\geq$`, `$w_+$`, `$\sigma$`, not σ or ≥.
- Avoid a closing `$` immediately followed by a digit; it breaks pandoc's math parser.
- Do not introduce Unicode while editing prose either.

## Write it like a human: the anti-slop standard

The report is scientific prose, and it must not read like generated filler. The standard
below is distilled from the *anti-ai-slop editor* (a personal skill at
`~/.claude/skills/anti-ai-slop`, not shipped with tvbo) and reproduced in full so this
skill stands alone — you do not need that skill installed. Before you call the report
done, scan the prose against it and fix every hit. Target a slop score of 0–2 (0–1 clean,
2–3 light, 4+ needs rework).

**Structural patterns to kill.**

- **Em-dash inflation.** At most one ` — ` per paragraph. The em-dash is a genuine tool,
  not a default connector; replace the rest with a period, comma, colon, or parenthesis.
  (Section-title separators like `## Fig. 2 — …` are one per heading and fine.)
- **Rule-of-three fetish.** Not every list has three items and not every noun needs three
  adjectives. When there are genuinely three points, keep them, but drop the "threefold /
  First… Second… Third…" scaffolding and vary the count when you can.
- **Monotone sentence openers.** "This enables… This ensures… This provides…" or
  "First… Second… Third…" repeated down a paragraph. Vary the rhythm.
- **Mirror-structure paragraphs.** Every paragraph going topic → explanation → example →
  transition reads as a template. Vary it.
- **Enthusiasm sandwich.** A positive opener, the content, a positive closer. Cut the
  wrapper.
- **Hedge stacking.** "It might be worth considering that perhaps…" — commit to the
  statement or delete it. Hedge only where the data is genuinely uncertain.
- **"not only … but also …"** — just say both things.
- **Throat-clearing.** If a paragraph still works without its first sentence, delete the
  first sentence.

**Banned words and phrases** (zero tolerance in report prose; replace or cut):

> delve · leverage · navigate (metaphorical) · landscape (metaphorical) · seamless ·
> robust (as a filler intensifier) · holistic · comprehensive · furthermore · moreover ·
> in conclusion · to summarize · it's important to note · in other words · game-changer ·
> cutting-edge · state-of-the-art · unlock · empower · foster · elevate · harness ·
> streamline · revolutionize · tapestry · multifaceted · synergy · paradigm shift ·
> in today's fast-paced world · let's dive in

Replace each with the specific thing you mean: "use", not "leverage"; "handle", not
"navigate"; the named benefit, not "robust" or "seamless"; "and" or nothing, not
"furthermore". Context matters: "robust to the SC normalization" backed by the numbers
that show it is a real claim, not slop — the ban is on the empty intensifier, not the
statistical term. Same for "attractor landscape" and other genuine terms of art.

**Rewriting principles.**

- Specific over abstract: "the FC fit reaches r ≈ 0.50 for FIC" beats "shows strong
  performance".
- Active voice: say who does what. Past-tense methods stay conventional ("the dynamics
  were validated against …").
- One idea per sentence; break compound slop sentences.
- Vary sentence length — short punches between longer flows.
- Cut before you rewrite; the finished prose should be roughly 20% shorter than the first
  draft.

**Before submitting**, re-scan: count the em-dashes per paragraph, check the banned list,
check for the three-item-list habit, and confirm no Unicode math slipped in.

## Render and verify

```bash
# from report/  (source is report-src.qmd; both PDFs coexist, any order)
QUARTO_PYTHON=<repo>/.venv/bin/python quarto render report-src.qmd --to pdf          # -> report.pdf (public)
QUARTO_PYTHON=<repo>/.venv/bin/python quarto render report-src.qmd --to pdf --profile internal  # -> report_internal.pdf (A/B, local-only)
```

The public render must succeed and must contain no copyrighted original. Open the PDF and
confirm the numbers rendered (an absent container shows as `—`, not a crash), the math
typeset (no dropped subscripts), and the prose passes the slop scan.
