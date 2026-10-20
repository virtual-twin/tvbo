---
name: writing-reports
description: How to write a TVBO study report — one IMRAD Quarto document whose every number is computed from the run (never transcribed), whose equations render natively from the recipe, that states negative results honestly, and holds its prose to a strict anti-slop standard. What a replication adds, such as the scorecard and the copyright-safe figure split, is marked "replication only".
metadata:
  audience: user
  applies_to:
    - "**/*.qmd"
  tags: [report, quarto, writing, replication]
  requires_extras: []
---

# Writing Reports in TVBO

This skill owns the **report** for any TVBO study: one `docs/report.qmd` that reads like a paper and computes every number it prints. **running-simulations** covers how the runs produce the containers you read here.

A replication is one kind of study. What it adds is marked **replication only** below: a scorecard of targets declared before the first run, a copyright-safe split between a public report and an internal build that shows the paper's figures, and the rules that tie its Methods, Results and captions to the published study. Everything else in this skill applies to every report. **replicating-studies** decides which targets a replication declares and their fidelity tier.

`tvbo study init <Name>` seeds `docs/report.qmd` and `docs/_quarto.yml`, and `tvbo study init <Name> -t replication` seeds their replication variants plus `report_internal.qmd` and the `docs/analysis/` target declarations. Start from what it wrote; do not write a report from scratch beside it. The seeded reports run from Abstract to Conclusion, and the replication one has a subsection and a status callout per figure. Keep its setup cell, which also builds `M`, and its two-build wiring, and bring the rest in line with this skill, in particular: give the report the six sections of the template below; replace the cell that renders one experiment's dynamics by the one `STUDY.report()` call ("A study renders its Methods ONCE"); replace the figure loop by one chapter per result and drop the `figure_callout` call; print the comparison table through `crossref_div`, where the seed returns it from its cell as `Markdown(...)` ("Every float has a number"); and see that the caption of the tally table names the four kinds of the verdict table below.

Four rules carry the whole report. Break any one and the report stops being trustworthy.

1. **IMRAD, not a figure walk.**
2. **Every number is computed from the run — none is typed by hand.**
3. **Equations come from the recipe metadata, not a hand transcription.**
4. **A negative result is stated plainly, with its evidence, and never softened.**

## Structure: IMRAD, one fixed template

The section order is fixed for every study: **Summary · Introduction · Methods · Results · Discussion · Supplement**, then the bibliography Quarto appends automatically (never a hand-written References section — see "References" below). Set `number-sections: true`; mark Summary `{.unnumbered}`. Each section has one contract, so a result that lands or changes has exactly one place to go and nobody has to decide where to put it:

- **Summary** says, in this order: what the study found (in a replication, what the *original* found and which cell or cells of the method-by-data table this study holds, the method naming the attempt and the data qualifying it: "a replication on the original data for the group fits, on independently obtained data for the person-specific models"; **replicating-studies** defines the cells), the targets and their outcome as computed counts, and how many divergences were found and of which classes. No parameter value (a step size, a TR count), no table, no caveat box. A proxy is explained in Methods and its consequence stated in Results.
- **Introduction** states the question, what counts as success, and the scope: which results are in and which are not.
- **Methods** holds the design and data (why an input is a substitute, why a second atlas or dataset appears at all), the model and experiments rendered in ONE `STUDY.report()` call, and how targets are scored, with each scope decision that puts a target `out` and its reason. The reason for a setting lives in that experiment's `description:` in the recipe, so the render carries it and the report never describes the model a second time. **Replication only:** a short-cell table of the *deliberate* departures from the published Methods (setting · what the Methods print · what the main result uses · where that value comes from · the supplement chapter showing the other variant), with the reasons as prose under it.
- **Results** opens with one summary chapter and then has **one chapter per result, in the original study's order, never one per figure**. The summary chapter carries the tally of targets by outcome, a short-cell targets table (target · published · replication · status) and, in a replication, the divergences *found* between the published code and the published Methods counted by class, plus what could not be obtained. Each result chapter has an outcome sentence computed from the targets it owns, the numbers, its numbered and captioned figure, and a pointer to the supplement chapter wherever a setting was run both ways. A figure lets the reader judge a result by eye; it is not the target, and the chapter does not describe it.
- **Discussion** is about this study only. In a replication that means the replication, not the original's science: what held and at what level, what that shows in The Turing Way's four words (**replicating-studies** maps each cell to its word, and a word is claimed only where the cell's targets are met), where it falls short and the measured reason, which choices were recovered by matching the published material, the limits.
- **Supplement** comes last. Every chapter in it carries an anchor of the form `{#sec-supp-<topic>}` and is called from the main text as `@sec-supp-<topic>`; a chapter nothing points to gets its call or is deleted. **Replication only:** it may close with one chapter on variants of our own that reach the published result at lower cost, or a better result, called from the Discussion (**replicating-studies**, Phase 6).

State a negative result in **Results** (with the evidence) and interpret it in **Discussion**. When one modelling fact explains several panel mismatches, say so once instead of listing them as separate failures.

**When a result lands or changes, re-check these places, in this order.** The list is the whole update. A place that is not on it should not need touching, and if it does, the report holds a second copy of something.

1. `docs/analysis/targets.md`: the target's status (`met` / `short` / `not attempted` / `out` / `blocked`) and, where it is not met, its row in the reasons table.
2. The binding: the container or scorecard key the report reads the value from, and `published-values.md` if a value of the paper's was added.
3. The result's own chapter: every sentence around a recomputed slot, read at the new value and at zero.
4. The Results summary chapter, the Summary and the Discussion: every sentence whose subject is a count, a level or what a cell's outcome shows.
5. The departures table in Methods and the supplement chapter, if the result is a variant of a setting, and the call to that chapter from the main text.
6. The figure: rendered again together with its caption sidecar and its A/B composite, the caption still naming every panel.
7. The study's task list: strike what the result closes, add what it opens.
8. The internal build rendered, and the public one once no target is `not attempted`; then the guards under "Render and verify".

## Every number is computed, never transcribed

Open each result container once in a setup cell and compute every reported quantity into one dict `M`. Reference those values in prose and captions through inline `` `{python} …` `` — this works inside figure captions too. A number typed into prose is a bug, whether it is a count, a decay time, a bifurcation threshold, a correlation, or a fitted parameter. Papers are not ground truth, and neither are your own asserted numbers; a recomputed value that *differs* from the paper is honest, a typed one that matches is not.

**The rule is asymmetric — keep the two kinds of number straight.** A quantity is either *yours* (a result/metric from the run → MUST be inline `` `{python} M[...]` ``) or *the paper's* (a value you quote for comparison — "paper: 9 solitary", "±105 MW", "$t_c$ 2.58 s"), which MUST stay a **literal**: you cannot recompute someone else's number, and dressing it up as computed would be the lie. Layout config and rejected counterfactuals (a `dt=0.01` alternative you *didn't* use) stay literal too. So "nothing hardcoded" means *every value of yours computed, every value of theirs quoted* — a hardcoded paper value is correct; a hardcoded result is the bug.

**Audit it before shipping — the rule does not enforce itself.** A report can read as fully computed (dozens of inline values, a fat `M`) and still hide a typed result. Grep the prose for numeric literals (strip the `` `{python}` `` spans and fenced code), and classify each as yours-or-theirs; a stray decimal that is *your* spectral peak, peak *location*, decay time, or solver step is the bug — replace it with an `M[...]` entry. And don't excuse one as a soft "≈600 MW" because computing it needs a cross-experiment merge (an ordinal-keyed Lyapunov/branch run read against a scan's parameter axis) — add the helper and compute it; an "≈" sitting next to a typed number in prose is usually the tell.

**Audit every file the report TRANSCRIBES, not just the `.qmd`.** A report that reads a hand-kept analysis note and prints its tables has moved the typed numbers, not removed them — and the note is the worse place for them, because nothing renders it and nobody re-reads it. Kadak2025's `verification.md` held its own alpha peak, seed spread, *t*, CI and argmax ridge as literals and the report printed them as a table for weeks; when the run improved, the table did not, and it still opened "Fifteen targets score short" at a scorecard of 35 met. The fix is not to update the literals. It is to compute them in the report and leave the note a pointer with no numbers in it. The rule to apply to a transcribed file is the same asymmetric one: a value only an external run can produce (a reference implementation's output, the paper's own print) may be a literal in the ONE file that owns it; every value of yours in it is a bug.

**There are THREE owners, not two, and the third is the one a caption gets wrong.** Besides yours and the paper's there is a measurement *you* made on *their* artifact — the arrays they released, re-analysed under a candidate definition; a reference implementation you compiled and ran. It is computed by you and it is not a result of your replication, and the tell that you have mixed the classes is a caption asserting the leftover rule: Kadak2025's transcription said "a column naming NFTsim or the paper says so, and every other column is this replication's own", under a table of correlations measured on the authors' arrays. The report then contradicts itself in print, because the same four correlations appear computed from our containers twenty pages earlier with different values — both correct, on two different datasets. Never let a caption derive ownership by elimination. Say which artifact each column was measured on, and have the transcribed file state its own rule in its first paragraph so the caption can defer to it rather than invent one.

**A measurement on the authors' files is still recomputable, and two typed copies of it will drift.** The class being external does not license typing it: the study's own loader of the published arrays is right there (`load_published_results()` in Kadak2025's `code/`), so give the measurement a helper and let both the note and the prose read the one call. The Fig-2D settings scan lived as a typed table in `verification.md` *and* as a typed `p = .088` in the report's Results — one measurement, two hand-kept copies, no way to notice when they part. A *search* over candidate definitions is no exception. It is not the same helper as one that evaluates the definition it chose, so the search is declared in the study too, as far as today's grammar reaches: an exploration over the candidates and an analysis that scores each one against the published numbers (**replicating-studies**, Phase 4). The report reads the selected candidate from the stored scores and types nothing; where the recipe itself needs the selected value it carries a literal that the harness checks against those scores (**replicating-studies**, Phase 4).

**Literals hide as number WORDS, and a digit grep sails past them.** The sentence "T13 disagrees with the paper on which **two of eleven** alpha conditions fail a Bonferroni-corrected test" sat in a section otherwise computed to three decimals, and matched no `[0-9]` audit. Grep for the spelled forms too — one, two, three … dozen, both, neither, all four, half. When that particular "two" was finally computed it came out **zero**: the typed number was not merely un-auditable, it was wrong, and had been asserting a disagreement the run no longer had.

**Never let a COUNT title a section.** `print(f"## Why {N_SHORT} targets fall short")` is correct at fifteen, awkward at five and absurd at zero — and zero is the outcome you are working toward, so the heading is guaranteed to end up false. Title a section by its subject, which does not move ("Where our numbers still differ, and why"), and let the counts live in its sentences, where falling to zero reads fine. The same goes for any sentence that presupposes the set is non-empty: "the reason beside each non-`met` row" needs an `if` around it, or it promises a column that isn't there. Write the boundary case first — render the section mentally at N = 0 and at N = all — because a report whose prose only works mid-range will embarrass you exactly when the work succeeds.

```python
from tvbo.classes.study import SimulationStudy
STUDY = SimulationStudy.from_file("../<Study>.yaml")
EXP = STUDY.get_experiment(<base_exp_id>)
# read derivatives/tvbo/exp-*_result.h5 containers, reduce to M[...] once, reference via inline `{python}`
```

**Give the paper's numbers ONE home too: `docs/analysis/published-values.md`.** The literals you are allowed to type are still literals scattered through prose, and scattered literals drift — the same correlation gets quoted twice with different rounding, a table and a paragraph disagree, and nobody can audit which values came from the manuscript at all. Transcribe every published number ONCE into a markdown table in that file, with the version of record named at the top, and have the report `read_md_tables` it. Three things follow that are worth the file on their own:

- a reader can diff the paper against your transcription in one place, so the transcription itself is reviewable;
- the report can *join* on it — every results table becomes "published | ours" side by side rather than a bare column of yours with a sentence claiming agreement;
- the join makes a **concordance test** cheap. Recompute every statistic the paper prints, pair each with its published value, and report the pairing: correlation across all of them, direction agreement, significance-verdict agreement at ONE threshold applied to both sides, and a paired location test (Wilcoxon) with a bootstrap interval on the median difference. That is a numerical answer to "did the replication succeed", instead of a per-figure verdict the reader has to aggregate themselves. Kadak2025 joins 82 published statistics this way and reports Pearson *r* = 0.98 with a median difference whose 95 % interval brackets zero.

Two rules keep the join honest. Quote the paper's own STATISTIC — if it prints *t*, report *t*; never convert its *t* to your *r* and call them the same column. And when the paper's significance marks are internally inconsistent (they usually are), score both sides with your own single rule and say so in the caption, rather than inheriting their bolding.

Build the comparison table's verdict column from the data as well: derive "reproduces" / "does not" from `M`, do not assert it. A panel without data of ours is a placeholder that says why it is empty, never the paper's own replotted arrays shown as your result, which crosses the integrity line. Which placeholders reach the public build is said under "The two builds have different duties".

## Equations and parameters rendered natively from the recipe

Render the model from the same metadata the backend compiles, so the mathematics shown is the mathematics that runs. Pass **`citeformat="quarto"`** so the model's own references come out as inline `@key` citations (resolved by this document's `bibliography:`, merging into the one auto-appended list) instead of a second, redundant reference list embedded in the block:

**A study renders its Methods ONCE, not once per experiment.** `STUDY.report("qmd")` writes the whole section: experiments sharing a model share its numbered equations and one symbol table, a variant contributes only its delta, and one table compares the experiments on what actually differs. Looping `experiment.render("markdown")` instead reprints the model verbatim for every experiment — Jansen1995's seven emitted 1209 lines and ~31 tables where the study call emits 136 and 3, with 115 lines identical between experiments 1 and 5.

```python
print(STUDY.report("qmd", level=3))  # whole Methods, deduplicated
print(STUDY.report("qmd", level=3, part="supplementary"))  # the experiments demoted out of it
print(EXP.dynamics.render("markdown", citeformat="quarto"))  # one model's equations, standalone
```

Mark an experiment `part: supplementary` in the recipe to move its paragraph out of the main Methods; it keeps its row in the comparison table, and it still runs. `part` is placement only, and it defaults to `main` — an experiment you never mark is described in full.

What the study call gives you that a loop cannot:

- **Every equation numbered and referable** — state equations, derived variables, functions, **and the coupling**: its assembled form, its pre/post decomposition and its summed inputs. `equations="semantic"` (the default) anchors on model and variable (`@eq-jansenrit1995-y3`), so a reference survives an experiment being inserted ahead of it; `"sequential"` numbers them `#eq-4`; `"none"` leaves them bare. Plain `"markdown"` has no anchor syntax and falls back to `\tag{n}`. Audit with the anchor *captured*, never with a negative lookahead — `\$\$.+?\$\$(?!\s*\{#eq-)` backtracks past the closing delimiter to satisfy the lookahead and reports whatever makes the check pass.
- **One glossary per model**, `Symbol | Kind | Meaning | Value | Unit`, dense by construction — state variables, parameters, derived parameters and the coupling's own symbols in one grid rather than four tables whose columns do not overlap. A parameter an experiment sweeps shows its range, not a value it never holds.
- **Every table captioned and anchored**, which is what stops the LaTeX table counter drifting.

The recipe's `description:` is printed verbatim, so **write it as the Methods prose you want** — it is the starting point of the manuscript. Do not restate in it what the report already derives (solver, step, duration, transient, node count, swept range): those are generated into the settings sentence, and a description that repeats them goes stale when the recipe changes.

Because `citeformat="quarto"` emits the model's citekeys as `@key`, **every citekey the model references must exist in `references.bib`** — if a curated model cites `Tsodyks1998`, that entry has to be present or Quarto flags an unresolved key.

## The report is the finished work, not the log of making it

A report says what is true now. The reader is deciding whether to trust a result and reuse a method; what you believed at an earlier point in the work is not evidence about either, and narrating it spends the reader's attention on the author. Three habits creep in and all three are cuttable:

- **A "corrections to earlier claims" section.** Superseding a claim is real and worth recording — in the study's local notes (the layout's `notes` role), where the next session reads it and where it stops a wrong "impossible" from being re-derived. In the report, the corrected statement is simply the statement.
- **Build-state branches.** `if M4 is None: print("this condition has not been run in this build")` reads as a report describing its own incompleteness. **Assert instead**, in the setup cell, every result whose target is `met` or `short`, the two statuses that say a run exists: for those a missing result is a broken build, not a section. A report branches on a missing result in two cases only, and both sides of each read as final: a genuine difference in the *reader's* copy (gitignored third-party inputs are absent from a public build, and one sentence saying so is right), and the outcome sentence of a chapter, which names a target that has no run by its status ("Outcomes in words").
- **Interim results and open work.** A warning that a number comes from a run being redone, a paragraph saying a panel is "pending the cohort", a list headed "what is not yet reproduced": each is a to-do list addressed to the author. Open tasks go in the study's task list, which the reader of the report never opens. What the report shows for a target that is `not attempted`, as the verdict table below defines that status, and which build renders in that state, is said under "The two builds have different duties".
- **Meta-commentary on the writing.** "This is the kind of entry a register has to be willing to close." The preceding sentence already made the point.
- **Self-references to prior builds.** "An earlier build of this report had this inverted, which traced to …" is unauditable: the reader holds one build, cannot see the earlier one, and cannot follow the trace — the sentence documents a mistake in an artifact that no longer exists instead of stating the result that does. This trap is strongest right after FIXING something, when the flip feels like the news; in the report the news is the correct value. The same discipline applies to the word "regression" for an upstream defect: it asserts a history ("this worked before") the reader cannot verify — name the *current* defect and its consequence ("a stimulus-targeting defect in the current backend means X is computed host-side").

  **Where the flip goes, and the loophole to avoid.** It does NOT go in `docs/analysis/`. In a replication that directory is a *shipped deliverable* — `targets.md`, `methods-vs-code.md` and `figures.md` are read by the same reader who reads the report — so "put it in the analysis notes" has repeatedly become a `## Corrections to earlier claims (do not re-introduce)` heading in a document people read, which is the banned sentence with a different filename. Put it in the study's local notes (the layout's `notes` role) if you want the anecdote at all. Better: **make the assertion the artifact.** The durable form of "do not re-introduce" is a harness check that FAILS if someone does, because an assertion survives a rewrite and a paragraph asking politely is deleted by the next person tidying the section. A `(ours)` row in the divergence register is still right for a fault of yours that changed a reported number — but write it as what is true ("the iteration budget is ours, sized on measurement: worst case 163 steps, the probe arrived in 102"), never as what you used to think.

  **Guard it mechanically**, because this recurs every time a study corrects itself under time pressure and it is invisible to every other check. Fail the harness on any reader-facing document matching `earlier version`, `previously (argued|claimed|said)`, `we (first|initially|originally)`, `an earlier claim`, `withdrawn`, `do not re-introduce`, `now corrected`, or `regression`.

**Guard the rendered Methods by its presence, not only by its purity.** `unrendered_equations` catches an equation typed into prose and says nothing about a report carrying no equations at all — which is where a report lands when the model section is written as prose and the render call is never wired in. Assert both:

```python
METHODS = STUDY.report("qmd", part="main", level=3)
assert "$$" in METHODS, "the Methods carry no rendered equations"
assert not unrendered_equations(Path("report.qmd")), "equations typed into the prose"
```

Reading that render once will also show you two recipe bugs it merely reflects: an experiment that inherits a sibling's `description:` through a YAML anchor prints the sibling's paragraph verbatim, and a `label:`/`description:` containing raw `*` or `_` is eaten by the markdown pass (`nu_ED*alpha*beta*A` renders as italics). Fix both in the recipe — the render is not the place to patch them.

## An equation is in the report because the code runs it

Two rules, and the second is the one that gets broken:

1. **Only equations the code actually integrates belong in the report.** Pang2023 set the paper's PDE above a section explaining that TVBO does not integrate that PDE — the reader sees mathematics that nothing runs, with no way to tell it apart from the rest.
2. **No equation is typed. Ever.** A typed equation drifts from what executes, and drift is invisible: the two look identical on the page.

Assert it in the harness so a hand-written `$$…$$` fails the render rather than reaching a reader. The check strips executable cells first, so what `STUDY.report()` emits never trips it:

```python
bad = report.unrendered_equations("report.qmd")
assert not bad, f"hand-written equations: {bad}"
```

**Guard the transcribed notes the same way, by naming the files allowed to hold literals.** The equation guard works because it is mechanical; the typed-result rule stays manual and therefore rots. Make it mechanical too: list the analysis files that legitimately own external numbers — `published-values.md`, and whatever holds a reference implementation's output — and assert that every *other* file the report reads carries none. It is a crude check and that is the point; a decimal appearing in a note nobody renders is exactly the thing no reviewer will catch.

Do NOT do this with a blanket "no decimals" scan. Most notes legitimately carry the paper's criteria and parameters, so it fires on all of them — ours reported 36, 74, 177 and 59 hits across four files that were entirely correct. The reviewable act is classifying each note ONCE by whose numbers it holds; the mechanical part is refusing to transcribe one nobody has classified:

```python
NUMBER_OWNERS = {
    "published-values.md": "paper",
    "targets.md": "paper",
    "methods-vs-code.md": "both",
    "verification.md": "reference-run",
}
notes = {p.name for p in study_path("analysis", root=ROOT).glob("*.md")}
assert not notes - set(NUMBER_OWNERS), f"unclassified analysis note: {sorted(notes - set(NUMBER_OWNERS))}"
assert "ours" not in NUMBER_OWNERS.values(), "a result of ours belongs in a computed cell, not a note"
```

`ours` is a category that must stay empty; it exists so that writing it down feels as wrong as it is. A new note then cannot reach a reader without someone answering the one question that matters about it.

**One scorer, and the report is it.** A standalone script that computes the scorecard is useful while the run is in flight, and it is a liability the moment it diverges from the report. Ours did: the script defined its seed-ensemble helper before use, the report defined the same helper *after* the cell that called it, and the result was a scorecard printing a clean 35 of 35 beside a report that could not compile at all (`NameError`). Neither artifact was wrong about the science; there were simply two orderings of one computation and only one of them ran. If you keep a standalone scorer, have it import the report's own module rather than restate it, and treat a green scorer as no evidence whatsoever that the report builds — render the report.

**A setup cell is a program, so read it as one.** A `.qmd` chunk invites you to append, and appending is how a value ends up used forty lines above its own `def`. When you add a term to a context object, put the addition *below* everything it reads, and re-render rather than trusting that a notebook-shaped file will sort itself out.

When an equation is genuinely implemented but no renderer can reach it — a solver-level construction such as the correlated-noise mix in `CorrelatedNoiseSolver` — it is **framework behaviour, not study metadata**. Write it once in tvbo's own docs beside the slot that switches it on, and have the study cite that page. Do not copy it into each report.

**An identity states nothing, so it gets no number.** `c_post = gx` says the summed input is used as it stands; `c_pre = local_states` says each source contributes its own state. Nine of eleven couplings across the studies had the first. Both now render as a clause. The same judgement applies to anything you write by hand: if an equation would be true of any model, it is prose.

**Never typeset a placeholder or a reference as a symbol.** `local_states` and `incoming_states` are alias tokens `Coupling.symbolic()` substitutes; printed raw they typeset as a variable of that name. Sources are worse, because they are not all symbols: a state variable is, `phenotype:…#PMAT24_A_RTCR` and `network.observations.BoldCorrelation` are not. Pandoc rejected the first outright ("unexpected `#`") and passed it through as raw TeX; the second rendered as a product of variables named after its path segments. Wrap in `$…$` only what is a plain identifier; everything else is code.

## Keep the recipe's own text publishable

`STUDY.report()` prints each experiment's `label:` and `description:` verbatim, so **recipe text is report text** and the anti-slop standard below applies to it.

**A description is reader text, and it has a limit: one sentence for a parameter, a state variable or an observation, at most three for an experiment or an analysis.** The generated Methods are the report's Methods, so everything a description holds is printed to the reader. A working note, the measurement that justified a setting, the candidates that were tried: each arrives in the PDF as method. A description says what the thing is and why it is set as it is. `tvbo validate study` reports one that runs over its limit; a description that needs the room is exempted one at a time, with the reason stated beside the exemption. The evidence goes where evidence is kept: in a replication the divergence register (`docs/analysis/methods-vs-code.md`), which the description cites by the entry's ID, "(D1)"; otherwise the study's analysis notes. A caption has the same economy under its own rule, one clause per panel ("Every table and every figure carries a caption").

Two habits to avoid:

- **Do not open a label with the experiment's own number.** "Exp 30 — FIC+EIB tuning" under a heading that already says "Experiment 30" prints the id twice; six of Schirner2023's ten read that way. The renderer strips the prefix, but the recipe is the place to fix it.
- **Give every parameter a `description:`.** It is the *Meaning* column of the symbol table, and the renderer will not invent one — printing "y0" as the meaning of $y_0$ fills the cell without informing anyone. Schirner2023 declares 36 of 49 parameters without one, so that column, the one that decodes every symbol in its 28 equations, is 4 % full.

**Generated headings carry their own anchors**, built from the model name or experiment id rather than the heading text. Quarto otherwise derives an identifier from recipe-authored words, which may hold anything: Cortes2013 labels an experiment with `I₀`, and the derived Typst label failed the compile with "unclosed label".

## State a negative result honestly

An honest replication reproduces what is real and explains what is not. If a claim does not reproduce, establish it to the standard of a finding before you call it one: an independent reference, a positive control that proves your method can detect the effect when it is present, and the mechanism. Report the cause, not just the mismatch. Do not dress a non-reproduction as a partial success, and do not bury it — put it where a reader looking for that result will find it.

## Outcomes in words: no status callouts, and NO emoji anywhere

Set `callout-icon: false`. A result chapter states its outcome in one computed sentence built from the targets it owns ("3 met, 1 short"), naming only the outcomes that occur so the sentence still reads at zero. Do not give a figure a coloured verdict box: a callout per figure makes the figure the unit of the report, and a yellow box is an open task set in a frame. **A report carries no warning.** If something is wrong with a number, fix it, or take the number out and put the task in the study's task list. The one callout a replication report keeps is the banner that marks the internal build.

**No emoji, checkmarks or dingbats anywhere in a report** — not in prose, not in a verdict column, not in a status table. Two independent reasons: xelatex has no glyph for them and drops them without a warning, so a column of ✅/🟡 renders as an empty column; and a scientific report states a verdict in the words the scorecard uses. Write `yes` / `partial` / `no`, or the scorecard's own `met` / `short` / `not attempted` / `out` / `blocked`. The same applies to ✓, ✗, ◐ and ◑, which read as decoration and vanish just as silently.

## Prose, not bullets, in Results and Discussion

Bullets are for planning. The rendered Results and Discussion must be flowing paragraphs: a reader gets the argument from connected sentences, not from fragments that leave the connective work undone. Lists belong in Methods (inclusion criteria, materials) and in the recipe-derived sections; a shortfall register, a limitation, or a result's outcome is prose.

The trap this creates with the tables rule above: when a table is too cramped to read, the fix is **prose**, not a bullet list. A shortfall section reads best as one paragraph per verdict class, each opening with what that verdict means and then naming its targets in sentences.

## The blueprint: what tvbo provides, what your report writes

A replication report is mostly the *same* report ten times over. Everything shared lives in `tvbo.utils.report`; what you write is the study's own metrics and its prose. If you find yourself defining `ab()`, `figcap()`, `fmt()`, `_open()` or a scorecard tally, stop — it exists.

| Job | Use | Not |
|---|---|---|
| Which build is this | `is_internal()`, `may_show_original(cleared)` | reading `QUARTO_DOCUMENT_FILE` yourself |
| A/B figure | `report_figure(...)` | a hand-rolled `ab()`, `show_report_figure` |
| Figure order | `figures_in_paper_order` | a hardcoded list of stems |
| Figure caption | the sidecar `docs/figures/<name>.caption.qmd`, printed through `crossref_div` | a caption typed in the report, `figure_caption` |
| Number for prose | `fmt`, `sci` | a local formatter per report |
| Result container | `open_result`, `result_sidecar`, `sidecar_value` | globbing the results dir inline |
| Declared analysis | `analysis_dataset`, `analysis_output`, `analysis_scalar` | recomputing what the recipe computed |
| Recipe value | `recipe_param`, `value_of` | reaching into `.parameters` by hand |
| Scorecard | `Scorecard(targets_md)` | a tally loop and a verdict dict per study |
| Captioned table | `crossref_div("tbl-…", table, caption)` | an uncaptioned printed table |

What stays in the report: the study's metric functions (each reads a container and returns a number), the `M` dict, the prose, and the credit line. That is the whole of it.

**The copyright guard is structural, not conventional.** Resolve the published figure *and* its attribution in one function behind the permission check, so the public build neither opens the file nor builds the credit string:

```python
CLEARED = False  # True ONLY with documented clearance from publisher AND authors


def original(fig):
    if not (CLEARED or INTERNAL):
        return None, ""
    return reference_image_for(fig, ROOT), "<Author> et al. <Year> (c)"
```

A `CREDIT` constant at the top of the report is one careless argument away from the shareable PDF. As a backstop, `report_figure` **raises** if handed an original without either ground — a report that forgets its guard fails the build instead of quietly shipping the paper's figure. Clearance is a real case, not a hypothetical; it is simply one no study here currently has.

## Every table and every figure carries a caption

An uncaptioned table is a wall of numbers the reader has to reverse-engineer. Caption both, and caption them from metadata wherever metadata exists.

- **Figures**: the caption is typed in the figure spec and nowhere else. Rendering a figure (`tvbo run`, `tvbo figure caption`) composes it from `Figure.description` and each `Panel.description` into a sidecar beside the image, `docs/figures/<name>.caption.qmd`, and the report passes the sidecar's text as the caption of the figure's cross-reference div (the cell is printed below). The div is what gives the figure its number. **What a caption is, stated here once:** it is written in our own words, short and descriptive. It describes every panel, in one clause each, and says what each axis quantity means in words before its variable name ("response time for correct answers (PMAT24_A_RTCR)"); it says what is drawn and never what was found, which is the job of Results. Ids, analysis names and provenance are not caption text. Under `Theme.caption_structure: metadata`, the default, a panel with an authored `description:` drops the derived structural clause (which analysis, which variable against which) from the text and carries it in the sidecar's front matter; a panel without a description keeps the clause, so an undescribed panel shows in the rendered caption. `inline` keeps the clause in the text of every panel. Never write in the report what the spec should say.
- **A display selection is declared in the caption.** When a panel draws less than was computed (every fourth simulated point, one seed of an ensemble, a window of a longer run), the selection is part of what the reader is looking at. Declare it in the figure spec and never as a slice inside a drawer: as the layer's `used.sel`, a selection by coordinate label that thins every variable the layer reads, or, for the markers of a drawn line, as `markevery` under the layer's `style.opts`. A selection neither can express is computed upstream, as a declared analysis whose output the layer binds. State the selection in that panel's own caption sentence in both terms, what was computed and what is drawn ("100 points simulated per branch, every fourth drawn"). **Replication only:** when the selection follows the published panel, the evidence that the original makes it goes in `docs/analysis/figures.md`.
- **Every float has a number.** A figure shown with `display(Image(...))` and a table returned from a cell as `Markdown(...)` land in the flow with no number and no caption, and nothing in the text can point at them. Print the cross-reference div from an `output: asis` cell instead.
- **Replication only — the caption is anchored to the published display item.** It keeps that item's panels and lettering, its quantities, its units and its stated ranges, so that our figure answers the question the paper's figure asks. The words that open it are not typed: the figure names the item in `Figure.published_item`, an object with two fields, `reference` (the citekey in the study's `references.bib`) and `item` (the item's own label in that publication, "Fig. 3a" or "Supplementary Fig. 4"), and tvbo derives the opening words of the caption from it ("After Fig. 3a of @Paper."). The slot sits beside `reference_image`, which stays the path of the A/B image, and the figure's name carries the same item number (`fig-3_desc-…`, `fig-S4_desc-…`); a name and a slot that disagree are a defect. A figure of ours without a published counterpart leaves the slot unset and gets no opener. Every deviation is said in the caption: a substituted input, a wider sweep, a panel the paper splits and we merge, a display selection. The anchor carries over neither the paper's wording nor its conclusions: never the paper's sentences (plagiarism, and they describe panels and data that are not ours), never a claim. It is there because a caption drafted from our own figure drifts into describing what we happened to draw, which is how a panel comes to answer a question the paper never asked. Which half of an internal composite is whose is not caption text: the pane titles say it, and the caption reads the same in both builds, so it never uses the A/B wording ("left: paper, right: ours").
- **A panel that shows individual participants' data follows the terms of the data.** It is drawn from individual participants only where the terms of the data permit publishing an individual's data and the study says so in `sourcedata/README.md`, under the heading "Personal data" (**replicating-studies**, Phase 2, says what that section holds). Otherwise it is drawn as an aggregate, and the caption says so. **Replication only:** where the terms do not permit it and the published panel shows one mark per participant, or one participant's recording, ours shows a density with the statistic and the number of participants, scored against the printed values, or the simulated trace alone.
- **Tables with a computed caption**: the `tbl-cap` cell option takes a *literal* string, so a caption holding a computed value must use Quarto's **cross-reference div** — the div's last paragraph is the caption and is ordinary markdown. `crossref_div("tbl-x", table, caption)` emits it.
- **Hand-written markdown tables**: put `: Caption text {#tbl-x}` on the line after the table.
- A caption **defines its terms**. If a column says `core`/`extended` or `mech`/`dec`, the caption says what those mean; the reader should not have to find `targets.md`.

**Print each figure as a numbered float whose caption is its sidecar.** Do it from an **`#| output: asis`** cell, as a cross-reference div. The caption is the sidecar's text without its front matter, which holds the structural metadata, and without its bold label lead, which the numbered float replaces. `FIGS` is the layout's figure directory from the setup cell, and `staged` is the path `report_figure` returned for this figure ("The A/B pair is composed by tvbo"):

    ```{python}
    #| output: asis
    import re

    sidecar = (FIGS / f"{fig.name}.caption.qmd").read_text()
    text = sidecar.split("\n---\n", 1)[-1] if sidecar.startswith("---\n") else sidecar
    caption = re.sub(r"^\s*\*\*[^*()]+\.\*\*\s*", "", text)
    print(crossref_div(f"fig-{fig.name.replace('_', '-')}", f"![]({embed_path(staged)})", caption))
    ```

The pattern removes only a bold run that ends in a period, which is the figure's label; a panel letter such as `**(a)**` opening an unlabelled caption stays. The caption reads the same in the public and the internal build.

**Use `output: asis`, not an inline `` `{python}` `` expression.** Inline-code output is inserted *verbatim* and is NOT re-parsed as markdown, so `**bold**` shows its asterisks and `$I_0$` shows its dollar signs. `output: asis` prints raw markdown that Quarto DOES process — so the description may use **LaTeX math** (`$I_0$`, `$\sigma$`, `$\Gamma$`) and computed `` `{python} M[...]` `` values, and no Unicode is needed (the whole point: LaTeX-compatible symbols, never font-fragile glyphs). Keep `Figure.description` in LaTeX + ASCII for this reason.

**Uncaptioned `longtable`s still step LaTeX's table counter**, so a Methods section that emits anonymous tables pushes the first captioned table in Results out to "Table 34" — and the floats that took the numbers are invisible, so the document simply appears to start at 34. `STUDY.report()` captions and anchors every table it emits, which removes the cause. Reach for the counter reset only where a section genuinely emits anonymous tables you cannot caption:

````markdown
```{=latex}
\setcounter{table}{0}
```
````

Verify rather than assume: count `^: .*\{#tbl-` against the tables in the rendered `.tex`, and check the first number is 1. A single uncaptioned float is enough to shift every number after it.

## Tables a reader can actually read

A replication report is mostly tables, and they are the first thing to go wrong. Build every one through `tvbo.utils.report.md_table` (never hand-write a pipe table with computed values), and hold to six rules.

- **A small grid is a sentence.** A numbered, captioned float tells the reader something has to be looked up, and journals cap how many a paper may carry — spending one on two numbers is a waste of a scarce slot. `report.table_or_prose` writes any grid below the threshold as prose instead (`Exp 50 (Duration: 2000 ms); Exp 51 (Duration: 7000 ms).`), so for a grid it builds this is automatic, and the rule matters when you are deciding whether to build a table at all. `md_table` always renders a table, so reach for it when the grid has no subject column for a sentence to name (a parameter block, a state-variable list, a scorecard). Pang2023 had a one-row float announcing that a model declares one event.
- **A table whose cells are sentences is not a table.** Four columns where two hold prose does not become readable by tuning widths — the prose columns starve the short ones and every row wraps to four lines. Write it as prose (see "Prose, not bullets" below). Reserve tables for short cells — IDs, numbers, verdicts, a clause — which is exactly what a scorecard is.
- **A column filled by a tenth of the rows belongs outside the grid.** It widens the table for every row to serve a few. Lift it into prose under the table, keyed by the row it describes. Measure before deciding: per-column fill is one pass over the rendered rows. Two under-half columns answering the *same* question merge into one that is mostly full — Schirner2023's observation table had `Sampling` at 44 % and `Pipeline` at 47 %, and merging them into `Reduction` took the grid from 34 % empty to 6 %.
- **A value every row repeats says nothing.** State it once in the caption and drop the column. `time_scale` defaults to `ms` in the schema, so `scale=ms` was printed on all 34 rows of one observation table and all 29 of another — a default nobody chose, on every line.
- **A verdict column needs its reason somewhere the reader reaches.** "T14 … out" beside the criterion it *would* have been judged against reads as a non-sequitur — the criterion explains a *failure*, never a *choice not to attempt* (see the four kinds below). Keep a `Why it falls short` register in `targets.md` with one row per non-`met` target, and join it by ID. Same rule inside the document: a scorecard row and its justification must not live in different files.
- **Give short columns a floor.** `md_table` sizes each column's separator to its content so pandoc allocates page width proportionally — with a floor, because a 3-character `ID` column beside two prose columns would otherwise get ~6 % of the text block, less than the width of the word `T14`, and its cells collide with the next column. If you build a table by hand, size its separator row the same way.
- **Spell verdicts out.** A bare status word in a narrow column is cryptic and unwrappable, so `Scorecard` prints each status under a display label: `out` reads "out of scope" and `not attempted` reads as itself. `targets.md` holds the status word and the table a reader sees holds the label, which also earns the column enough width to typeset it.

## Replication only — a target that is not met is one of four kinds, never one bucket

`met / partial / out` collapses unrelated judgements into one word, and the report then reads as if every shortfall were the same kind of shortfall. It is not. Score five verdicts, told apart by what would change them:

| Verdict | Meaning | What would change it | Is it a replication failure? |
|---|---|---|---|
| `met` | Run, and the criterion written for it is met. | — | — |
| `short` | **Run, and its criterion is not met.** | A run that meets it. | **Yes — the only one.** |
| `not attempted` | In scope, and only the study's own remaining work stands between the target and a run. For a team that maintains the framework, a capability the framework lacks that no other party owns and that generalizes is part of that work: the team builds it (**replicating-studies**, "Surface feature gaps now", "NEVER give a study its own execution machinery"). | Doing that work. | No. Nothing was run. |
| `out` | Excluded by a scope decision that Methods states, with the reason in its row. | Nothing that is pending, only a new decision. | No. Nothing was run, so nothing failed. |
| `blocked` | In scope, and something outside the study has to arrive first, named with its evidence: access to data, an answer from the authors, or a framework capability that the study's authors cannot change or that another party owns, while its issue is open, in which case the row gives the cause as `framework` and cites the issue. | Its arrival. | No. The obstacle lies outside the replication. |

The rule that catches the error: **read each reason and ask what would change the status.** If the study's own work would, the row is `not attempted`. If something asked for from outside would, the row is `blocked` and names what was asked and of whom. If nothing pending would, the row is `out`: a result the published material does not allow to be checked, with no request open, is `out` with that as its reason. A row marked `out` whose reason is really "we tried and it did not match" is the serious mistake, a failure hidden inside a scope decision, and it is `short`. And a row marked `out` only because its run has not happened yet is not `out`: `out` is a decision with its reason, never a queue.

Report the kinds separately, each opening with what its verdict means, so a scope decision can never be read as a failure; which of them a build can contain is said under "The two builds have different duties". The tally of the results summary counts the targets that are `blocked` with the cause `framework` on a line of their own. Carry the distinction into each chapter's outcome sentence, which names every outcome by its own word. And write the reason to match its own class — lead a scope decision with the judgement ("a robustness sweep of T12 rather than a new claim"), not with a blocker that happens to also apply.

## Replication only — copyright-safe internal/public split: one project, two entry files, ONE command

The report shows A/B panels (the paper's published figure beside yours), but the paper's figures are copyright-restricted and must never be committed or shared. So the report renders two ways: a **PUBLIC** `report.pdf` (your figures only, shareable) and an **INTERNAL** `report_internal.pdf` (paper © figures beside yours, git-ignored). The Quarto-native way to get both from one command is a small **project with two entry files**:

- **`report.qmd`** holds the *whole* report and carries **no YAML front matter** — every format/title/bibliography setting lives in `docs/_quarto.yml`, which `tvbo study init -t replication` seeds. This is the only file you write prose in.
- **`report_internal.qmd`** is a four-line wrapper: its front matter overrides only the `output-file` (and title), then `{{< include report.qmd >}}` pulls in the real report.
- **`docs/_quarto.yml`** lists both under `project: render:` and holds the shared `format: pdf` (+ `output-file: report.pdf`), `bibliography:`, and `execute:`.

`quarto render` (run in `docs/`, no file argument) builds the project's render list → **both PDFs in one pass**. No `--profile`, no `_quarto-internal.yml`, no post-render shell hook.

- `INTERNAL = tvbo.utils.report.is_internal()` reads `QUARTO_DOCUMENT_FILE` — the input filename, the only per-build variable Quarto exposes to the kernel, which is *why* the split is two entry files rather than two formats in one file. The A/B helper draws the paper original **only when `INTERNAL`**, so the public build never opens the copyrighted file.
- **Why `report.qmd` must have no front matter:** Quarto's `{{< include >}}` splices the included file's front matter too, and an included `output-file:` **overrides** the wrapper's — so the internal build would write `report.pdf` and clobber the public one. Keeping all front matter out of `report.qmd` (in `_quarto.yml` instead) removes the collision, and the two distinct input stems (`report`, `report_internal`) keep each build's `<stem>.pdf` intermediate from ever being the other's final. Track `report.qmd`, `report_internal.qmd`, `_quarto.yml`, and `references.bib`. The generated `.gitignore` already covers the PDFs and the whole of `sourcedata/`, composites included — never hand-edit it. Verify by rendering and confirming the public `report.pdf` embeds no © original: the internal PDF is visibly larger (it carries the paper figures) and its A/B composites are wide (~2.2+), while public figure aspect ratios stay near 1.0–1.5.

**The two builds have different duties.** The internal build renders at any time. A target that is `not attempted`, as the verdict table above defines that status, is named by that word in the targets table and in its chapter's outcome sentence, and has a labelled placeholder where its panel will be. The public build refuses to render while any target is `not attempted`, and its error lists them: what reaches a reader is complete, and until then the report is a local check. The check is one function of tvbo's report utilities: it takes the targets table and the build, and in the public build it raises with that list. `not attempted` is the only status that stops it. An `out` or a `blocked` row passes with its stated reason, and its panel keeps a placeholder that names the obstacle, in both builds. Call the check in the setup cell, before anything is printed.

### The A/B pair is composed by tvbo, not by each report

**Do not write an `ab()` that lays out matplotlib axes.** Ten reports each grew their own copy of that helper and they drifted: different widths, different title wording, one that false-coloured a greyscale scan through the default colormap. The layout lives in `tvbo.utils.report`:

```python
from tvbo.utils.report import report_figure

theirs, credit = original(fig)  # the guard above: a public build without clearance resolves neither
staged = report_figure(FIGS / f"{fig.name}.png", theirs, credit=credit, cleared=CLEARED)
```

What that buys, and what a per-report copy kept getting wrong:

- **Original LEFT, replication RIGHT, at a common height** with widths following each image's own aspect. Neither side is stretched to match the other — a squared-off original would misrepresent the very layout the A/B exists to check.
- **A missing original still holds its pane**, labelled with how to obtain it. Collapsing to a lone panel reads as a completed comparison that never happened.
- **Several scans stack into one pane** — pass a list when the paper splits one quantity over Fig 2A/2B.
- **A greyscale scan stays grey.** `imshow` on a 2-D array applies the default colormap, which silently recolours the paper's figure.
- **The stage comes from the layout record**, so no report names a directory: the composite lands in `sourcedata/original_study/fig_comparisons/`, inside the original-study directory whose figure it embeds, and the one rule that keeps the original unpublished covers it. The copyrighted original then reaches exactly one artifact and never the repository. Embed it through `embed_path`, which makes the reference relative to the render: LaTeX prefixes a bare path with `./`, so an absolute one arrives as `./Users/…` and the build fails on an image that is plainly there.
- **Our own figure is embedded where it was rendered.** With no original to compose against there is nothing to stage, and `report_figure` returns the path it was given; staging a copy would give every public figure a second location to keep current.

**Name the replication's pane only by what differs.** Leave `label` empty unless the replication's data or parcellation differ from the original's (a proxy atlas, a substitute connectome); naming the atlas the original also uses reads as a difference that is not there.

Embed the path `report_figure` returns through `crossref_div`, which gives the figure its number, its caption and a cross-reference target; the cell is printed under "Every table and every figure carries a caption". `show_report_figure` displays the image from a cell with none of the three, so a report does not use it.

### Take every figure's facts from the recipe, never a hand-written list

A list of `(stem, paper_image, caption)` tuples in the report is three things that drift from the `figures:` block. Place each figure in the chapter of the result it supports and take everything about it from the metadata: its place among the chapter's figures from `figures_in_paper_order`, its original from `reference_image_for` (the declared `reference_image:`), its caption from the sidecar. A figure renamed or re-captioned in the recipe then changes in the report with nothing typed.

Derive each chapter's outcome sentence the same way: give `targets.md` a `Result` column naming the result of the original study each target belongs to, and build the sentence from the scorecard rows whose `Result` is the chapter's, so the sentence under a chapter heading and the tally in the summary cannot disagree. `figure_targets(fig, TARGET_ROWS)` joins on the `Fig(s)` column where a figure needs the targets it supports.

## Render with LaTeX, and keep long tables out of callouts

`_quarto.yml` sets **`format: pdf`** with `pdf-engine: xelatex`. A replication report is mostly wide computed tables, and `longtable` is the only engine that breaks one across a page with its header repeated; typst restarted the table on a fresh page and left the remainder of the previous one blank. The two entry stems (`report`, `report_internal`) keep each build's `.tex` intermediate from being the other's, so the LaTeX path has no intermediate-clobber problem.

**A `longtable` cannot live inside a callout.** Quarto renders a callout as a breakable `tcolorbox`, and LaTeX cannot page-break a longtable inside one — so it ships each table to a page of its own and leaves three quarters of the preceding page blank. Wrapping the Methods experiment renders in `::: {.callout-note}` blocks cost 14 pages of whitespace in one report. **A callout is not a container**: print the Methods that `STUDY.report()` generates into the plain flow, under the headings it emits. The symptom to recognise is a PDF whose page count is two or three times what its content warrants, with a table alone at the top of each page.

Two settings earn their place in `include-in-header`:

```yaml
    include-in-header:
      text: |
        \usepackage{etoolbox}
        \usepackage{ragged2e}
        \AtBeginEnvironment{longtable}{\small\RaggedRight}
```

`\small` buys a wide scorecard the width it needs; `\RaggedRight` stops LaTeX stretching inter-word space in a narrow column until the row looks broken.

## Section references — Quarto crossrefs, never typed numbers

A reference like "(Sec. 2.8)" typed into prose is a number the build does not own: any change that renumbers the sections — a new Methods subsection, a model card the generator starts emitting, a supplementary family — silently retargets every typed reference in the document. Deco2014's report pointed "(Sec. 2.8)" at its Backend section until the equation render gained the spiking synapse families, after which the same string pointed mid-way into the Methods; a second reference ("Sec. 3.1") was stale on arrival, written for a numbering that had already moved. Give the heading an explicit id (`## Backend {#sec-backend}`) and reference it as `@sec-backend` — Quarto renders "Section 2.8" and re-resolves it on every build (`number-sections: true` required). Headings the model render emits already carry `{#sec-model-…}` / `{#sec-experiment-…}` anchors for exactly this use.

## References — Quarto's bibliography, never a hand-written list

Do not write a `# References` section and do not add a `::: {#refs}` div. Set `bibliography: references.bib` in `_quarto.yml`, cite sources inline with `@key` (a bare `@key` renders "Author (Year)", a bracketed `[@key]` renders "(Author, Year)"), and Quarto **appends the bibliography once, automatically**, listing exactly the keys you cited. A manual heading or `#refs` div only produces a second, empty section. Two consequences to hold to:

- **Cite every source in the prose at least once.** Quarto lists only keys that appear as `@key` somewhere; an entry in `references.bib` that is never cited is silently dropped. So the paper (`@<Author><Year>`), the sources of the words the report uses for the attempt and its outcome (Benureau & Rougier 2018, The Turing Way), the data/connectome/substitute sources, and every citekey the native model render emits (with `citeformat="quarto"`) each need an inline `@key` — not just a line in the `.bib`.
- **Keep `references.bib` complete.** Any unresolved `@key` (including one emitted by the model render) surfaces as "Citation key not found". When a curated model cites a key you don't yet have (e.g. `Tsodyks1998` vs an existing `Tsodyks1997`), add the missing entry rather than editing the model.

## PDF gotchas

These failures are silent, so verify them in the rendered PDF rather than in the source:

- Write all math, subscripts, and superscripts as **LaTeX**, never Unicode — xelatex drops Unicode math without a word. Use `$J_{NMDA}$`, `$\geq$`, `$w_+$`, `$\sigma$`, not σ or ≥. This applies inside `output: asis` captions too.
- **No emoji, anywhere.** xelatex has no glyph for them and drops them silently, so a ✅/🟡 status column renders as an empty column. A verdict is a word, never a colour or a glyph (`callout-icon: false` is set for the same reason).
- Write dashes as ASCII: `--` renders as an en-dash and `---` as an em-dash. Do not paste the Unicode – or — glyphs.
- A `#| label: tbl-*` on a cell that prints a table makes Quarto number it as a float — it comes out as a bare `Table 36` with no caption unless you give it one. Label it something else unless you are cross-referencing it.
- Avoid a closing `$` immediately followed by a digit; it breaks pandoc's math parser.
- Do not introduce Unicode while editing prose either.

## Write it like a human: the anti-slop standard

The report is scientific prose, and it must not read like generated filler. The standard below is distilled from the *anti-ai-slop editor* (a personal skill at `~/.claude/skills/anti-ai-slop`, not shipped with tvbo) and reproduced in full so this skill stands alone — you do not need that skill installed. Before you call the report done, scan the prose against it and fix every hit. Target a slop score of 0–2 (0–1 clean, 2–3 light, 4+ needs rework).

Scan the **recipe** as well as the `.qmd`: labels and descriptions are printed verbatim into the Methods, so their prose is the report's prose. In one study every remaining em-dash after the generated text was cleaned came from the YAML.

**Structural patterns to kill.**

- **No em-dashes at all in reader-facing text.** Nothing that reaches the PDF may contain one: not the prose, not a figure caption, not a table cell, not a title or subtitle, not a recipe `description:` or `label:`, and not a row of an analysis table the report reads and reprints. Use a period, comma, colon or parenthesis instead. Grep the rendered PDF, not the sources, because generated text has sources you will not think of: `pdftotext -layout report_internal.pdf - | grep -c '—'` must print `0`. That single check doubles as a container check, since a missing result renders as `—` too.
- **Rule-of-three fetish.** Not every list has three items and not every noun needs three adjectives. When there are genuinely three points, keep them, but drop the "threefold / First… Second… Third…" scaffolding and vary the count when you can.
- **Monotone sentence openers.** "This enables… This ensures… This provides…" or "First… Second… Third…" repeated down a paragraph. Vary the rhythm.
- **Mirror-structure paragraphs.** Every paragraph going topic → explanation → example → transition reads as a template. Vary it.
- **Enthusiasm sandwich.** A positive opener, the content, a positive closer. Cut the wrapper.
- **Hedge stacking.** "It might be worth considering that perhaps…" — commit to the statement or delete it. Hedge only where the data is genuinely uncertain.
- **"not only … but also …"** — just say both things.
- **Throat-clearing.** If a paragraph still works without its first sentence, delete the first sentence.

**Banned words and phrases** (zero tolerance in report prose; replace or cut):

> delve · leverage · navigate (metaphorical) · landscape (metaphorical) · seamless · robust (as a filler intensifier) · holistic · comprehensive · furthermore · moreover · in conclusion · to summarize · it's important to note · in other words · game-changer · cutting-edge · state-of-the-art · unlock · empower · foster · elevate · harness · streamline · revolutionize · tapestry · multifaceted · synergy · paradigm shift · in today's fast-paced world · let's dive in

Replace each with the specific thing you mean: "use", not "leverage"; "handle", not "navigate"; the named benefit, not "robust" or "seamless"; "and" or nothing, not "furthermore". Context matters: "robust to the SC normalization" backed by the numbers that show it is a real claim, not slop — the ban is on the empty intensifier, not the statistical term. Same for "attractor landscape", for *robust* as The Turing Way defines it in a replication's Discussion, and for other genuine terms of art.

**Rewriting principles.**

- Specific over abstract: "the FC fit reaches r ≈ 0.50 for FIC" beats "shows strong performance".
- Active voice: say who does what. Past-tense methods stay conventional ("the dynamics were validated against …").
- One idea per sentence; break compound slop sentences.
- Vary sentence length — short punches between longer flows.
- Cut before you rewrite; the finished prose should be roughly 20% shorter than the first draft.

**Before submitting**, re-scan: grep the rendered PDF for em-dashes (it must return none), check the banned list, check for the three-item-list habit, and confirm no Unicode math slipped in.

## Render and verify

```bash
# from docs/ — no file arg, so quarto renders the project's own `render:` list (in a replication that is BOTH PDFs)
PATH=<repo>/.venv/bin:$PATH quarto render
```

**PATH, not `QUARTO_PYTHON`, decides which interpreter executes the cells.** Quarto runs the notebook through the `python3` kernelspec, whose `argv` is a bare `python`, resolved against PATH at execution time; `QUARTO_PYTHON` only picks the Python that drives the render. Get this wrong and the report is executed by whatever `python` came first, which fails as `No module named 'tvbo.utils.study_layout'` — a missing-module message that names the module you just wrote rather than the one actually absent (`numpy`, imported by `tvbo/utils/__init__.py`). Check with `import sys; print(sys.executable)` in the first cell when a render fails on an import you are certain exists.

The render must succeed, with no warning in its log. Check the structure before the content: the six top-level sections are there in order, the first Results chapter is the summary, every `{#sec-supp-…}` anchor is referenced from the main text, no cell displays a figure or a table outside a numbered float, and no experiment id organises a heading. In a replication the public `report.pdf` must contain no copyrighted original, and the internal PDF is the larger one because it carries the paper's figures. Open the PDF and confirm the numbers rendered (an absent container shows as `—`, not a crash, so the em-dash grep above catches it), the math typeset (no dropped subscripts, no stray `\pm`), the citations resolved (no "Citation key not found", one auto-appended bibliography), and the prose passes the slop scan.
