---
name: writing-reports
description: How to write a TVBO replication report — one IMRAD Quarto document whose every number is computed from the run (never transcribed), whose equations render natively from the recipe, that states negative results honestly, keeps a copyright-safe internal/public figure split, and holds its prose to a strict anti-slop standard.
metadata:
  audience: user
  applies_to:
    - "**/*.qmd"
  tags: [report, quarto, writing, replication]
  requires_extras: []
---

# Writing Reports in TVBO

This skill owns the **report** for a study replication: one `report/report.qmd` that
reads like a paper and computes every number it prints. It is the reporting layer that
**replicating-studies** composes; that skill decides *which* targets you replicate and
their fidelity tier, and **running-simulations** covers how the runs produce the
containers you read here. Start from the report template in the replicating-studies
skeleton (`report.qmd.tmpl` + its thin `report_internal.qmd.tmpl` wrapper + `_quarto.yml.tmpl`)
and keep its metrics cell, its A/B figure loop, and the two-entry render layout.

Four rules carry the whole report. Break any one and the report stops being trustworthy.

1. **IMRAD, not a figure walk.**
2. **Every number is computed from the run — none is typed by hand.**
3. **Equations come from the recipe metadata, not a hand transcription.**
4. **A negative result is stated plainly, with its evidence, and never softened.**

## Structure: IMRAD

The section order is fixed for every study: **Abstract · Introduction · Methods ·
Results · Discussion · Conclusion**, then the bibliography Quarto appends automatically
(never a hand-written References section — see "References" below). Set
`number-sections: true`; mark Abstract `{.unnumbered}`. Map the pieces to sections:

- **Methods** carries the native equation render, the variants, coupling, network and
  data provenance, the analyses, the backend, and the verification against an
  independent reference.
- **Results** opens with the computed comparison/scorecard table, then one subsection
  per paper figure: a sentence or two on what the panel shows, the figure, its
  **caption**, and a status callout.
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

**Audit every file the report TRANSCRIBES, not just the `.qmd`.** A report that reads a hand-kept
analysis note and prints its tables has moved the typed numbers, not removed them — and the note
is the worse place for them, because nothing renders it and nobody re-reads it. Kadak2025's
`verification.md` held its own alpha peak, seed spread, *t*, CI and argmax ridge as literals and
the report printed them as a table for weeks; when the run improved, the table did not, and it
still opened "Fifteen targets score short" at a scorecard of 35 met. The fix is not to update the
literals. It is to compute them in the report and leave the note a pointer with no numbers in it.
The rule to apply to a transcribed file is the same asymmetric one: a value only an external run
can produce (a reference implementation's output, the paper's own print) may be a literal in the
ONE file that owns it; every value of yours in it is a bug.

**Literals hide as number WORDS, and a digit grep sails past them.** The sentence "T13 disagrees
with the paper on which **two of eleven** alpha conditions fail a Bonferroni-corrected test" sat
in a section otherwise computed to three decimals, and matched no `[0-9]` audit. Grep for the
spelled forms too — one, two, three … dozen, both, neither, all four, half. When that particular
"two" was finally computed it came out **zero**: the typed number was not merely un-auditable, it
was wrong, and had been asserting a disagreement the run no longer had.

**Never let a COUNT title a section.** `print(f"## Why {N_SHORT} targets fall short")` is correct
at fifteen, awkward at five and absurd at zero — and zero is the outcome you are working toward,
so the heading is guaranteed to end up false. Title a section by its subject, which does not move
("Where our numbers still differ, and why"), and let the counts live in its sentences, where
falling to zero reads fine. The same goes for any sentence that presupposes the set is non-empty:
"the reason beside each non-`met` row" needs an `if` around it, or it promises a column that
isn't there. Write the boundary case first — render the section mentally at N = 0 and at N = all
— because a report whose prose only works mid-range will embarrass you exactly when the work
succeeds.

```python
from tvbo.classes.study import SimulationStudy
STUDY = SimulationStudy.from_file("../<Study>.yaml")
EXP = STUDY.get_experiment(<base_exp_id>)
# read output/nc/exp*/…h5 containers, reduce to M[...] once, reference via inline `{python}`
```

**Give the paper's numbers ONE home too: `report/analysis/published-values.md`.** The literals
you are allowed to type are still literals scattered through prose, and scattered literals drift
— the same correlation gets quoted twice with different rounding, a table and a paragraph
disagree, and nobody can audit which values came from the manuscript at all. Transcribe every
published number ONCE into a markdown table in that file, with the version of record named at
the top, and have the report `read_md_tables` it. Three things follow that are worth the file on
their own:

- a reader can diff the paper against your transcription in one place, so the transcription
  itself is reviewable;
- the report can *join* on it — every results table becomes "published | ours" side by side
  rather than a bare column of yours with a sentence claiming agreement;
- the join makes a **concordance test** cheap. Recompute every statistic the paper prints, pair
  each with its published value, and report the pairing: correlation across all of them,
  direction agreement, significance-verdict agreement at ONE threshold applied to both sides,
  and a paired location test (Wilcoxon) with a bootstrap interval on the median difference.
  That is a numerical answer to "did the replication succeed", instead of a per-figure verdict
  the reader has to aggregate themselves. Kadak2025 joins 82 published statistics this way and
  reports Pearson *r* = 0.98 with a median difference whose 95 % interval brackets zero.

Two rules keep the join honest. Quote the paper's own STATISTIC — if it prints *t*, report *t*;
never convert its *t* to your *r* and call them the same column. And when the paper's
significance marks are internally inconsistent (they usually are), score both sides with your
own single rule and say so in the caption, rather than inheriting their bolding.

Build the comparison table's verdict column from the data as well: derive "reproduces" /
"does not" from `M`, do not assert it. A placeholder panel (TVBO data not yet run) gets a
labelled placeholder and a "missing" callout, never a green one — showing the paper's own
replotted arrays as your result crosses the integrity line.

## Equations and parameters rendered natively from the recipe

Render the model from the same metadata the backend compiles, so the mathematics shown is
the mathematics that runs. Pass **`citeformat="quarto"`** so the model's own references come
out as inline `@key` citations (resolved by this document's `bibliography:`, merging into the
one auto-appended list) instead of a second, redundant reference list embedded in the block:

**A study renders its Methods ONCE, not once per experiment.** `STUDY.report("qmd")` writes
the whole section: experiments sharing a model share its numbered equations and one symbol
table, a variant contributes only its delta, and one table compares the experiments on what
actually differs. Looping `experiment.render("markdown")` instead reprints the model verbatim
for every experiment — Jansen1995's seven emitted 1209 lines and ~31 tables where the study
call emits 136 and 3, with 115 lines identical between experiments 1 and 5.

```python
print(STUDY.report("qmd", level=3))  # whole Methods, deduplicated
print(STUDY.report("qmd", level=3, part="supplementary"))  # the experiments demoted out of it
print(EXP.dynamics.render("markdown", citeformat="quarto"))  # one model's equations, standalone
```

Mark an experiment `part: supplementary` in the recipe to move its paragraph out of the main
Methods; it keeps its row in the comparison table, and it still runs. `part` is placement
only, and it defaults to `main` — an experiment you never mark is described in full.

What the study call gives you that a loop cannot:

- **Every equation numbered and referable** — state equations, derived variables, functions,
  **and the coupling**: its assembled form, its pre/post decomposition and its summed inputs.
  `equations="semantic"` (the default) anchors on model and variable
  (`@eq-jansenrit1995-y3`), so a reference survives an experiment being inserted ahead of it;
  `"sequential"` numbers them `#eq-4`; `"none"` leaves them bare. Plain `"markdown"` has no
  anchor syntax and falls back to `\tag{n}`. Audit with the anchor *captured*, never with a
  negative lookahead — `\$\$.+?\$\$(?!\s*\{#eq-)` backtracks past the closing delimiter to
  satisfy the lookahead and reports whatever makes the check pass.
- **One glossary per model**, `Symbol | Kind | Meaning | Value | Unit`, dense by construction —
  state variables, parameters, derived parameters and the coupling's own symbols in one grid
  rather than four tables whose columns do not overlap. A parameter an experiment sweeps shows
  its range, not a value it never holds.
- **Every table captioned and anchored**, which is what stops the LaTeX table counter drifting.

The recipe's `description:` is printed verbatim, so **write it as the Methods prose you want** —
it is the starting point of the manuscript. Do not restate in it what the report already
derives (solver, step, duration, transient, node count, swept range): those are generated into
the settings sentence, and a description that repeats them goes stale when the recipe changes.

Because `citeformat="quarto"` emits the model's citekeys as `@key`, **every citekey the model
references must exist in `references.bib`** — if a curated model cites `Tsodyks1998`, that
entry has to be present or Quarto flags an unresolved key.

## The report is the finished work, not the log of making it

A report says what is true now. The reader is deciding whether to trust a result and reuse a
method; what you believed at an earlier point in the work is not evidence about either, and
narrating it spends the reader's attention on the author. Three habits creep in and all three
are cuttable:

- **A "corrections to earlier claims" section.** Superseding a claim is real and worth
  recording — in the working notes under `report/analysis/`, where the next session reads it
  and where it stops a wrong "impossible" from being re-derived. In the report, the corrected
  statement is simply the statement.
- **Build-state branches.** `if M4 is None: print("this condition has not been run in this
  build")` reads as a report describing its own incompleteness. **Assert instead**, in the setup
  cell: a missing result is a broken build, not a section. The exception is a genuine difference
  in the *reader's* copy — gitignored third-party inputs are absent from a public build, and one
  sentence saying so is right.
- **Meta-commentary on the writing.** "This is the kind of entry a register has to be willing
  to close." The preceding sentence already made the point.

**Guard the rendered Methods by its presence, not only by its purity.** `unrendered_equations`
catches an equation typed into prose and says nothing about a report carrying no equations at
all — which is where a report lands when the model section is written as prose and the render
call is never wired in. Assert both:

```python
METHODS = STUDY.report("qmd", part="main", level=3)
assert "$$" in METHODS, "the Methods carry no rendered equations"
assert not unrendered_equations(Path("report.qmd")), "equations typed into the prose"
```

Reading that render once will also show you two recipe bugs it merely reflects: an experiment
that inherits a sibling's `description:` through a YAML anchor prints the sibling's paragraph
verbatim, and a `label:`/`description:` containing raw `*` or `_` is eaten by the markdown pass
(`nu_ED*alpha*beta*A` renders as italics). Fix both in the recipe — the render is not the place
to patch them.

## An equation is in the report because the code runs it

Two rules, and the second is the one that gets broken:

1. **Only equations the code actually integrates belong in the report.** Pang2023 set the
   paper's PDE above a section explaining that TVBO does not integrate that PDE — the reader
   sees mathematics that nothing runs, with no way to tell it apart from the rest.
2. **No equation is typed. Ever.** A typed equation drifts from what executes, and drift is
   invisible: the two look identical on the page.

Assert it in the harness so a hand-written `$$…$$` fails the render rather than reaching a
reader. The check strips executable cells first, so what `STUDY.report()` emits never trips it:

```python
bad = report.unrendered_equations("report.qmd")
assert not bad, f"hand-written equations: {bad}"
```

**Guard the transcribed notes the same way, by naming the files allowed to hold literals.** The
equation guard works because it is mechanical; the typed-result rule stays manual and therefore
rots. Make it mechanical too: list the analysis files that legitimately own external numbers —
`published-values.md`, and whatever holds a reference implementation's output — and assert that
every *other* file the report reads carries none. It is a crude check and that is the point; a
decimal appearing in a note nobody renders is exactly the thing no reviewer will catch.

Do NOT do this with a blanket "no decimals" scan. Most notes legitimately carry the paper's
criteria and parameters, so it fires on all of them — ours reported 36, 74, 177 and 59 hits
across four files that were entirely correct. The reviewable act is classifying each note ONCE by
whose numbers it holds; the mechanical part is refusing to transcribe one nobody has classified:

```python
NUMBER_OWNERS = {
    "published-values.md": "paper",
    "targets.md": "paper",
    "methods-vs-code.md": "both",
    "verification.md": "reference-run",
}
notes = {p.name for p in (ROOT / "report/analysis").glob("*.md")}
assert not notes - set(NUMBER_OWNERS), f"unclassified analysis note: {sorted(notes - set(NUMBER_OWNERS))}"
assert "ours" not in NUMBER_OWNERS.values(), "a result of ours belongs in a computed cell, not a note"
```

`ours` is a category that must stay empty; it exists so that writing it down feels as wrong as it
is. A new note then cannot reach a reader without someone answering the one question that matters
about it.

**One scorer, and the report is it.** A standalone script that computes the scorecard is useful
while the run is in flight, and it is a liability the moment it diverges from the report. Ours
did: the script defined its seed-ensemble helper before use, the report defined the same helper
*after* the cell that called it, and the result was a scorecard printing a clean 35 of 35 beside
a report that could not compile at all (`NameError`). Neither artifact was wrong about the
science; there were simply two orderings of one computation and only one of them ran. If you keep
a standalone scorer, have it import the report's own module rather than restate it, and treat a
green scorer as no evidence whatsoever that the report builds — render the report.

**A setup cell is a program, so read it as one.** A `.qmd` chunk invites you to append, and
appending is how a value ends up used forty lines above its own `def`. When you add a term to a
context object, put the addition *below* everything it reads, and re-render rather than trusting
that a notebook-shaped file will sort itself out.

When an equation is genuinely implemented but no renderer can reach it — a solver-level
construction such as the correlated-noise mix in `CorrelatedNoiseSolver` — it is **framework
behaviour, not study metadata**. Write it once in tvbo's own docs beside the slot that switches
it on, and have the study cite that page. Do not copy it into each report.

**An identity states nothing, so it gets no number.** `c_post = gx` says the summed input is
used as it stands; `c_pre = local_states` says each source contributes its own state. Nine of
eleven couplings across the studies had the first. Both now render as a clause. The same
judgement applies to anything you write by hand: if an equation would be true of any model,
it is prose.

**Never typeset a placeholder or a reference as a symbol.** `local_states` and
`incoming_states` are alias tokens `Coupling.symbolic()` substitutes; printed raw they typeset
as a variable of that name. Sources are worse, because they are not all symbols: a state
variable is, `phenotype:…#PMAT24_A_RTCR` and `network.observations.BoldCorrelation` are not.
Pandoc rejected the first outright ("unexpected `#`") and passed it through as raw TeX; the
second rendered as a product of variables named after its path segments. Wrap in `$…$` only
what is a plain identifier; everything else is code.

## Keep the recipe's own text publishable

`STUDY.report()` prints each experiment's `label:` and `description:` verbatim, so **recipe
text is report text** and the anti-slop standard below applies to it. Two habits to avoid:

- **Do not open a label with the experiment's own number.** "Exp 30 — FIC+EIB tuning" under a
  heading that already says "Experiment 30" prints the id twice; six of Schirner2023's ten read
  that way. The renderer strips the prefix, but the recipe is the place to fix it.
- **Give every parameter a `description:`.** It is the *Meaning* column of the symbol table, and
  the renderer will not invent one — printing "y0" as the meaning of $y_0$ fills the cell
  without informing anyone. Schirner2023 declares 36 of 49 parameters without one, so that
  column, the one that decodes every symbol in its 28 equations, is 4 % full.

**Generated headings carry their own anchors**, built from the model name or experiment id
rather than the heading text. Quarto otherwise derives an identifier from recipe-authored
words, which may hold anything: Cortes2013 labels an experiment with `I₀`, and the derived
Typst label failed the compile with "unclosed label".

## State a negative result honestly

An honest replication reproduces what is real and explains what is not. If a claim does
not reproduce, establish it to the standard of a finding before you call it one: an
independent reference, a positive control that proves your method can detect the effect
when it is present, and the mechanism. Report the cause, not just the mismatch. Do not
dress a non-reproduction as a partial success, and do not bury it — put it where a reader
looking for that result will find it.

## Status callouts: three colours, and NO emoji anywhere

Set `callout-icon: false`. Give each figure one short callout:

- **green** (`.callout-note` / `.callout-tip`): what reproduced, with an inline-computed
  number.
- **yellow** (`.callout-warning`): what is *missing* — the data or target is not available
  (`out`, `blocked`, a placeholder panel).
- **red** (`.callout-important`): what was *attempted and failed to match* (`short`). Red is
  reserved for that; a declared scope decision is not a failure.

One or two sentences each. The colour carries the verdict; the sentence carries the
evidence.

**No emoji, checkmarks or dingbats anywhere in a report** — not in prose, not in a verdict
column, not in a status table. Two independent reasons: xelatex has no glyph for them and drops
them without a warning, so a column of ✅/🟡 renders as an empty column; and a scientific report
states a verdict in the words the scorecard uses. Write `yes` / `partial` / `no`, or the
scorecard's own `met` / `short` / `out` / `blocked`. The same applies to ✓, ✗, ◐ and ◑, which
read as decoration and vanish just as silently.

## Prose, not bullets, in Results and Discussion

Bullets are for planning. The rendered Results and Discussion must be flowing paragraphs: a
reader gets the argument from connected sentences, not from fragments that leave the connective
work undone. Lists belong in Methods (inclusion criteria, materials) and in the recipe-derived
sections; a shortfall register, a limitation, or a per-figure verdict is prose.

The trap this creates with the tables rule above: when a table is too cramped to read, the fix is
**prose**, not a bullet list. A shortfall section reads best as one paragraph per verdict class,
each opening with what that verdict means and then naming its targets in sentences.

## The blueprint: what tvbo provides, what your report writes

A replication report is mostly the *same* report ten times over. Everything shared lives in
`tvbo.utils.report`; what you write is the study's own metrics and its prose. If you find
yourself defining `ab()`, `figcap()`, `fmt()`, `_open()` or a scorecard tally, stop — it exists.

| Job | Use | Not |
|---|---|---|
| Which build is this | `is_internal()`, `may_show_original(cleared)` | reading `QUARTO_DOCUMENT_FILE` yourself |
| A/B figure | `report_figure(...)` / `show_report_figure(...)` | a hand-rolled `ab()` |
| Figure order, title, caption | `figures_in_paper_order`, `figure_title`, `figure_caption` | a hardcoded list of stems and captions |
| Number for prose | `fmt`, `sci` | a local formatter per report |
| Result container | `open_result`, `result_sidecar`, `sidecar_value` | globbing `output/nc` inline |
| Declared analysis | `analysis_dataset`, `analysis_output`, `analysis_scalar` | recomputing what the recipe computed |
| Recipe value | `recipe_param`, `value_of` | reaching into `.parameters` by hand |
| Scorecard | `Scorecard(targets_md)` | a tally loop and a verdict dict per study |
| Captioned table | `crossref_div("tbl-…", table, caption)` | an uncaptioned printed table |

What stays in the report: the study's metric functions (each reads a container and returns a
number), the `M` dict, the prose, and the credit line. That is the whole of it.

**The copyright guard is structural, not conventional.** Resolve the published figure *and* its
attribution in one function behind the permission check, so the public build neither opens the
file nor builds the credit string:

```python
CLEARED = False  # True ONLY with documented clearance from publisher AND authors


def original(fig):
    if not (CLEARED or INTERNAL):
        return None, ""
    return reference_image_for(fig, ROOT), "<Author> et al. <Year> (c)"
```

A `CREDIT` constant at the top of the report is one careless argument away from the shareable
PDF. As a backstop, `report_figure` **raises** if handed an original without either ground — a
report that forgets its guard fails the build instead of quietly shipping the paper's figure.
Clearance is a real case, not a hypothetical; it is simply one no study here currently has.

## Every table and every figure carries a caption

An uncaptioned table is a wall of numbers the reader has to reverse-engineer. Caption both, and
caption them from metadata wherever metadata exists.

- **Figures**: `![{figure_caption(fig)}](path){#fig-name}` — the caption is the recipe's own
  `Figure.description`, so it cannot drift from the figure, and the `#fig-` id makes it
  cross-referenceable. Never retype the paper's caption.
- **Say whose figure it is, in the caption, in every build.** A replication report shows two
  kinds of picture and the reader cannot tell them apart from the image alone. Open each caption
  with which original it reproduces (`**Reproducing Figure 3 of @Paper.**`, or nothing at all
  when the figure is the replication's own addition), and close it with a provenance sentence
  that is *derived from the build*, not written once: internally "the published original is on
  the left and this replication on the right", publicly "every panel is this replication's own
  output". One `figure()` helper emits both, so the two builds can never disagree about which
  half of the image is whose. Do NOT hardcode the A/B wording — that composite exists only in
  the internal build.
- **Tables with a computed caption**: the `tbl-cap` cell option takes a *literal* string, so a
  caption holding a computed value must use Quarto's **cross-reference div** — the div's last
  paragraph is the caption and is ordinary markdown. `crossref_div("tbl-x", table, caption)`
  emits it.
- **Hand-written markdown tables**: put `: Caption text {#tbl-x}` on the line after the table.
- A caption **defines its terms**. If a column says `core`/`extended` or `mech`/`dec`, the
  caption says what those mean; the reader should not have to find `targets.md`.

**Uncaptioned `longtable`s still step LaTeX's table counter**, so a Methods section that emits
anonymous tables pushes the first captioned table in Results out to "Table 34" — and the floats
that took the numbers are invisible, so the document simply appears to start at 34.
`STUDY.report()` captions and anchors every table it emits, which removes the cause. Reach for
the counter reset only where a section genuinely emits anonymous tables you cannot caption:

````markdown
```{=latex}
\setcounter{table}{0}
```
````

Verify rather than assume: count `^: .*\{#tbl-` against the tables in the rendered `.tex`, and
check the first number is 1. A single uncaptioned float is enough to shift every number after it.

## Tables a reader can actually read

A replication report is mostly tables, and they are the first thing to go wrong. Build every
one through `tvbo.utils.report.md_table` (never hand-write a pipe table with computed values),
and hold to six rules.

- **A small grid is a sentence.** A numbered, captioned float tells the reader something has to
  be looked up, and journals cap how many a paper may carry — spending one on two numbers is a
  waste of a scarce slot. `report.table_or_prose` writes any grid below the threshold as prose
  instead; `md_table` always renders a table, so reach for it when the grid has no subject
  column for a sentence to name (a parameter block, a state-variable list, a scorecard).
  (`Exp 50 — Duration: 2000 ms; Exp 51 — Duration: 7000 ms.`), so this is automatic; the rule
  matters when you are deciding whether to build a table at all. Pang2023 had a one-row float
  announcing that a model declares one event.
- **A table whose cells are sentences is not a table.** Four columns where two hold prose does
  not become readable by tuning widths — the prose columns starve the short ones and every row
  wraps to four lines. Write it as prose (see "Prose, not bullets" below). Reserve tables for
  short cells — IDs, numbers, verdicts, a clause — which is exactly what a scorecard is.
- **A column filled by a tenth of the rows belongs outside the grid.** It widens the table for
  every row to serve a few. Lift it into prose under the table, keyed by the row it describes.
  Measure before deciding: per-column fill is one pass over the rendered rows. Two under-half
  columns answering the *same* question merge into one that is mostly full — Schirner2023's
  observation table had `Sampling` at 44 % and `Pipeline` at 47 %, and merging them into
  `Reduction` took the grid from 34 % empty to 6 %.
- **A value every row repeats says nothing.** State it once in the caption and drop the column.
  `time_scale` defaults to `ms` in the schema, so `scale=ms` was printed on all 34 rows of one
  observation table and all 29 of another — a default nobody chose, on every line.
- **A verdict column needs its reason somewhere the reader reaches.** "T14 … out" beside the
  criterion it *would* have been judged against reads as a non-sequitur — the criterion explains
  a *failure*, never a *choice not to attempt* (see the three kinds of shortfall below). Keep a `Why it falls short` register in
  `targets.md` with one row per non-`met` target, and join it by ID. Same rule inside the
  document: a scorecard row and its justification must not live in different files.
- **Give short columns a floor.** `md_table` sizes each column's separator to its content so
  pandoc allocates page width proportionally — with a floor, because a 3-character `ID` column
  beside two prose columns would otherwise get ~6 % of the text block, less than the width of
  the word `T14`, and its cells collide with the next column. If you build a table by hand, size
  its separator row the same way.
- **Spell verdicts out.** `out` in a narrow column is both cryptic and unwrappable; `out of
  scope` is clearer *and* earns the column enough width to typeset it.

## A shortfall is one of three things — never one bucket

`met / partial / out` collapses two unrelated judgements into one word, and the report then reads
as if every shortfall were the same kind of shortfall. It is not. Score four verdicts:

| Verdict | Meaning | Is it a replication failure? |
|---|---|---|
| `met` | Reproduced against the criterion written for it. | — |
| `short` | **Attempted and did not meet its criterion.** | **Yes — the only one.** |
| `out` | Judged to add no test of the paper's claims; declared unattempted. | No. Nothing was run, so nothing failed. |
| `blocked` | Would be in scope; an input it needs cannot be obtained. | No — a gap in the data, not in the reproduction. |

The rule that catches the error: **read each shortfall reason and ask whether it describes a
choice, an obstacle, or a result.** A row marked `out` whose reason says "cannot be scored" or
"needs data that is not released" is mislabelled — that is `blocked`. A row marked `out` whose
reason is really "we tried and it did not match" is the serious version of the same mistake:
a failure hidden inside a scope decision.

Report the three groups separately, each opening with what its verdict means, so a scope decision
can never be read as a failure. Carry the distinction into the per-figure callouts too: red for
`short` only; `out` and `blocked` are yellow. And write the reason to match its own class —
lead a scope decision with the judgement ("a robustness sweep of T12 rather than a new claim"),
not with a blocker that happens to also apply.

## Copyright-safe internal/public split — one project, two entry files, ONE command

The report shows A/B panels (the paper's published figure beside your reproduction), but
the paper's figures are copyright-restricted and must never be committed or shared. So the
report renders two ways: a **PUBLIC** `report.pdf` (your reproduction only, shareable) and an
**INTERNAL** `report_internal.pdf` (paper © figures beside yours, git-ignored). The
Quarto-native way to get both from one command is a small **project with two entry files**:

- **`report.qmd`** holds the *whole* report and carries **no YAML front matter** — every
  format/title/bibliography setting lives in `report/_quarto.yml` (copied from
  `_quarto.yml.tmpl`). This is the only file you write prose in.
- **`report_internal.qmd`** is a four-line wrapper: its front matter overrides only the
  `output-file` (and title), then `{{< include report.qmd >}}` pulls in the real report.
- **`report/_quarto.yml`** lists both under `project: render:` and holds the shared
  `format: pdf` (+ `output-file: report.pdf`), `bibliography:`, and `execute:`.

`quarto render` (run in `report/`, no file argument) builds the project's render list → **both
PDFs in one pass**. No `--profile`, no `_quarto-internal.yml`, no post-render shell hook.

- `INTERNAL = tvbo.utils.report.is_internal()` reads `QUARTO_DOCUMENT_FILE` — the input filename,
  the only per-build variable Quarto exposes to the kernel, which is *why* the split is two entry
  files rather than two formats in one file. The A/B helper draws the paper original **only when
  `INTERNAL`**, so the public build never opens the copyrighted file.

### The A/B pair is composed by tvbo, not by each report

**Do not write an `ab()` that lays out matplotlib axes.** Ten reports each grew their own copy of
that helper and they drifted: different widths, different title wording, one that false-coloured a
greyscale scan through the default colormap. The layout lives in `tvbo.utils.report`:

```python
from tvbo.utils.report import report_figure, show_report_figure

staged = report_figure(
    FIGDIR / f"{fig.name}.png",  # ours
    reference_image_for(fig, ROOT) if INTERNAL else None,  # theirs
    STAGE,
    credit="Pang et al. 2023 (c)",
)
print(f"![**Fig {n}.** {figure_caption(fig)}](_figures/{staged.name}){{width=100%}}")
```

What that buys, and what a per-report copy kept getting wrong:

- **Original LEFT, reproduction RIGHT, at a common height** with widths following each image's own
  aspect. Neither side is stretched to match the other — a squared-off original would misrepresent
  the very layout the A/B exists to check.
- **A missing original still holds its pane**, labelled with how to obtain it. Collapsing to a lone
  panel reads as a completed comparison that never happened.
- **Several scans stack into one pane** — pass a list when the paper splits one quantity over
  Fig 2A/2B.
- **A greyscale scan stays grey.** `imshow` on a 2-D array applies the default colormap, which
  silently recolours the paper's figure.
- **The composite is staged into `report/_figures/`** (gitignored), so the copyrighted original
  reaches exactly one artifact and never the repository.

Prefer `report_figure` + a markdown embed over `show_report_figure`: the embed gets a real figure
number, a caption and a cross-reference target. Use `show_report_figure` only where a report
already emits figures from plain python cells and restructuring them is not worth it.

### Drive the figure section from the recipe, never a hand-written list

A list of `(stem, paper_image, caption)` tuples in the report is three things that drift from the
`figures:` block. Loop the study's own figures, and take everything from the metadata —
`figures_in_paper_order`, `figure_title`, `figure_caption` (the `description:`), and
`reference_image_for` (the declared `reference_image:`). A figure added to the recipe then appears
in the report with its caption, in the right place, with nothing typed.

Derive the per-figure status callout the same way: `figure_targets(fig, TARGET_ROWS)` joins the
scorecard on the targets table's own `Fig(s)` column, so the verdict beside a figure and the
verdict in the scorecard cannot disagree.
- **Why `report.qmd` must have no front matter:** Quarto's `{{< include >}}` splices the included
  file's front matter too, and an included `output-file:` **overrides** the wrapper's — so the
  internal build would write `report.pdf` and clobber the public one. Keeping all front matter out
  of `report.qmd` (in `_quarto.yml` instead) removes the collision, and the two distinct input
  stems (`report`, `report_internal`) keep each build's `<stem>.pdf` intermediate from ever being
  the other's final. Track `report.qmd`, `report_internal.qmd`, `_quarto.yml`, and
  `references.bib`; git-ignore `report/*.pdf` and the paper's figures under `original_study/`.
Verify by rendering and confirming the public `report.pdf` embeds no © original: the internal
PDF is visibly larger (it carries the paper figures) and its A/B composites are wide (~2.2+),
while public figure aspect ratios stay near 1.0–1.5.

## Render with LaTeX, and keep long tables out of callouts

`_quarto.yml` sets **`format: pdf`** with `pdf-engine: xelatex`. A replication report is mostly
wide computed tables, and `longtable` is the only engine that breaks one across a page with its
header repeated; typst restarted the table on a fresh page and left the remainder of the previous
one blank. The two entry stems (`report`, `report_internal`) keep each build's `.tex` intermediate
from being the other's, so the LaTeX path has no intermediate-clobber problem.

**A `longtable` cannot live inside a callout.** Quarto renders a callout as a breakable
`tcolorbox`, and LaTeX cannot page-break a longtable inside one — so it ships each table to a page
of its own and leaves three quarters of the preceding page blank. Wrapping the Methods experiment
renders in `::: {.callout-note}` blocks cost 14 pages of whitespace in one report. **Callouts are
for verdicts, not containers**: give a rendered `experiment.render("markdown")` a plain `###`
heading. The symptom to recognise is a PDF whose page count is two or three times what its content
warrants, with a table alone at the top of each page.

Two settings earn their place in `include-in-header`:

```yaml
    include-in-header:
      text: |
        \usepackage{etoolbox}
        \usepackage{ragged2e}
        \AtBeginEnvironment{longtable}{\small\RaggedRight}
```

`\small` buys a wide scorecard the width it needs; `\RaggedRight` stops LaTeX stretching
inter-word space in a narrow column until the row looks broken.

## References — Quarto's bibliography, never a hand-written list

Do not write a `# References` section and do not add a `::: {#refs}` div. Set
`bibliography: references.bib` in `_quarto.yml`, cite sources inline with `@key` (a bare `@key`
renders "Author (Year)", a bracketed `[@key]` renders "(Author, Year)"), and Quarto **appends the
bibliography once, automatically**, listing exactly the keys you cited. A manual heading or `#refs`
div only produces a second, empty section. Two consequences to hold to:

- **Cite every source in the prose at least once.** Quarto lists only keys that appear as `@key`
  somewhere; an entry in `references.bib` that is never cited is silently dropped. So the paper
  (`@<Author><Year>`), NASEM 2019, the data/connectome/substitute sources, and every citekey the
  native model render emits (with `citeformat="quarto"`) each need an inline `@key` — not just a
  line in the `.bib`.
- **Keep `references.bib` complete.** Any unresolved `@key` (including one emitted by the model
  render) surfaces as "Citation key not found". When a curated model cites a key you don't yet
  have (e.g. `Tsodyks1998` vs an existing `Tsodyks1997`), add the missing entry rather than editing
  the model.

**Every figure gets an original, public-facing caption — auto-rendered from the recipe.** Under
each `ab()` call, render the caption with an **`#| output: asis`** cell:

    ```{python}
    #| output: asis
    print(f"**Fig N.** {figcap('<FigName>')}")
    ```

where the `figcap()` helper reads the figure's `Figure.description` from the loaded study (single
source of truth — the caption never drifts from the figure metadata, and you never retype it). The
caption is **public-facing**: it describes OUR standalone reproduction, so it must NEVER paste the
paper's caption verbatim (plagiarism) and must NEVER use the A/B framing ("left: paper, right:
ours", "paper beside") — that composite exists only in the `INTERNAL` build.

**Use `output: asis`, not an inline `` `{python} figcap()` ``.** Inline-code output is inserted
*verbatim* and is NOT re-parsed as markdown, so `**bold**` shows its asterisks and `$I_0$` shows
its dollar signs. `output: asis` prints raw markdown that Quarto DOES process — so the `**Fig N.**`
lead renders bold, the description may use **LaTeX math** (`$I_0$`, `$\sigma$`, `$\Gamma$`) and
computed `` `{python} M[...]` `` values, and no Unicode is needed (the whole point: LaTeX-compatible
symbols, never font-fragile glyphs). Keep `Figure.description` in LaTeX + ASCII for this reason.

## PDF gotchas

These failures are silent, so verify them in the rendered PDF rather than in the source:

- Write all math, subscripts, and superscripts as **LaTeX**, never Unicode — xelatex drops
  Unicode math without a word. Use `$J_{NMDA}$`, `$\geq$`, `$w_+$`, `$\sigma$`, not σ or ≥. This
  applies inside `output: asis` captions too.
- **No emoji, anywhere.** xelatex has no glyph for them and drops them silently, so a ✅/🟡 status
  column renders as an empty column. Callout colours carry the verdict (and `callout-icon: false`
  is set for the same reason).
- Write dashes as ASCII: `--` renders as an en-dash and `---` as an em-dash. Do not paste the
  Unicode – or — glyphs.
- A `#| label: tbl-*` on a cell that prints a table makes Quarto number it as a float — it comes
  out as a bare `Table 36` with no caption unless you give it one. Label it something else unless
  you are cross-referencing it.
- Avoid a closing `$` immediately followed by a digit; it breaks pandoc's math parser.
- Do not introduce Unicode while editing prose either.

## Write it like a human: the anti-slop standard

The report is scientific prose, and it must not read like generated filler. The standard
below is distilled from the *anti-ai-slop editor* (a personal skill at
`~/.claude/skills/anti-ai-slop`, not shipped with tvbo) and reproduced in full so this
skill stands alone — you do not need that skill installed. Before you call the report
done, scan the prose against it and fix every hit. Target a slop score of 0–2 (0–1 clean,
2–3 light, 4+ needs rework).

Scan the **recipe** as well as the `.qmd`: labels and descriptions are printed verbatim into
the Methods, so their prose is the report's prose. In one study every remaining em-dash after
the generated text was cleaned came from the YAML.

**Structural patterns to kill.**

- **No em-dashes at all in reader-facing text.** Nothing that reaches the PDF may contain
  one: not the prose, not a figure caption, not a table cell, not a title or subtitle, not a
  recipe `description:` or `label:`, and not a row of an analysis table the report reads and
  reprints. Use a period, comma, colon or parenthesis instead. Grep the rendered PDF, not the
  sources, because generated text has sources you will not think of: `pdftotext -layout
  report_internal.pdf - | grep -c '—'` must print `0`. That single check doubles as a
  container check, since a missing result renders as `—` too.
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

**Before submitting**, re-scan: grep the rendered PDF for em-dashes (it must return none),
check the banned list, check for the three-item-list habit, and confirm no Unicode math
slipped in.

## Render and verify

```bash
# from report/ — ONE command renders BOTH report.pdf (public) and report_internal.pdf (A/B, local-only)
QUARTO_PYTHON=<repo>/.venv/bin/python quarto render     # no file arg -> the project render: list
```

The render must succeed and the public `report.pdf` must contain no copyrighted original (the
internal PDF is the larger one, carrying the paper figures). Open the PDF and confirm the numbers
rendered (an absent container shows as `—`, not a crash, so the em-dash grep above catches
it), the math typeset (no dropped subscripts,
no stray `\pm`), the citations resolved (no "Citation key not found", one auto-appended
bibliography), and the prose passes the slop scan.
