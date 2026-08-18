# Phase 7 reference — a number of ours disagrees with theirs

Read this when the published material contains the authors' own derived arrays and your
implementation does not return them. Every section is a way to establish *which*
of the two is wrong before rewriting either. Work them roughly in order: the cheap
structural checks — what is actually in the files, what order it is stored in —
come before the expensive ones.

The default holds throughout: **a claimed discrepancy is our bug until a
falsification test says otherwise.** These checks exist to run that test, not to
assume either answer.

## Inventory the published material by CONTENT, not by filename — a mislabelled file can be the oracle

Read every published array's **shape and range** before you read its name. Pang2023 ships
`results/hcp_striatum-lh_gradient_20_variance.txt`, whose two siblings are 200-element variance
spectra; the striatum's is a **1,896 × 1,896 affinity matrix** — the *input* to the one step of
the chain the published material contains no code for, for the one structure of three that was failing.
`numpy.loadtxt` returns it without complaint and its first row reads as a plausible decreasing
spectrum, so nothing announces the mistake. Finding it converted a dead end into an exact test:
run our embedding on their own matrix and require their own gradients back.

The check is one pass and costs nothing: for each file, print `shape`, `min`, `max`, whether it
is symmetric, whether the diagonal is constant. A square matrix where the name promises a vector,
a range in `[0, 1]` with a unit diagonal where the name promises eigenvalues — these are the
tells. **Do this before concluding a step is unverifiable**, because the reason a step looks
unverifiable is usually that its oracle is filed under the wrong name.

## Find an ORDER-INVARIANT oracle when you don't know the published material's storage order

Comparing two derived arrays requires knowing how each is indexed, and published repositories rarely say.
MATLAB's `find` walks a volume column-major and numpy's `np.where` row-major, so the same voxel
set arrives in different orders; compared positionally, correct modes score |r| = 0.18 against
their own counterparts and reindexed they score 0.994. **A wrong order looks exactly like a
failed replication**, and worse, it looks like it in a way that invites you to go and "fix" a
pipeline that was right.

The way out is to score against a quantity that **does not depend on the order at all**.
Eigenvalues are the common one: `λ(P S Pᵀ) = λ(S)` for any permutation `P`, so a published
*spectrum* identifies the construction that produced it without your knowing the permutation
its *vectors* are stored in. That collapses a joint (ordering × construction) search into a 1-D
one. Pang2023's `subcortical_gradients_variance` — 200 numbers summing to exactly 100 — is what
pinned three unstated steps: it fixed the variance weight (`1/λ`, r = 0.996 where `1/λ²`,
`1 − λ` and `exp(−λ)` all fail), the normalisation window (200, because that is how many numbers
there are), and later the input field's smoothness. Only after the construction was fixed was
the permutation recovered, and then trivially — sorted values matching at r = 0.996 says "right
matrix, wrong order" in one line.

Other order-invariant handles worth reaching for: sorted value distributions, traces, matrix
norms, degree sequences (a normalised Laplacian's λ = 0 eigenvector is exactly `sqrt(degree)`,
which turns "which graph did they use?" into an integer identity test over every vertex).

## Before blaming your implementation, measure the STATISTIC's own stability

A published number your pipeline misses is *usually* your bug — that default is right and this
skill says so elsewhere. But some published statistics have no stable value to reproduce, and
attributing their spread to your code will send you hunting forever. The test is cheap and
decisive: **sweep a degree of freedom the paper never fixes, holding everything else constant,
and see how far the statistic travels.**

Pang2023's time-to-peak is a raw global `argmax` over a field reconstructed from a truncated
eigenbasis. Truncation leaves ripple *before* the wave can physically arrive, and where a
region's own peak is small the ripple wins. Swept over mode counts from 100 to 500 — same basis,
same stimulus, same parameters, same integrator — the published correlation returns
**+0.29, −0.34, −0.45, −0.51, −0.76, −0.71, −0.31, −0.71, −0.45**: a range of 1.05 in r,
straddling zero, over a choice the paper never states. Our "shortfall" of −0.45 against −0.72 was
one draw of that lottery, and so was theirs.

Read the result as follows. **If the statistic's spread under an unstated choice exceeds your
discrepancy, the discrepancy is not evidence about your implementation** — stop looking there.
Then either report the target as unreproducible-as-defined, or repair the statistic. Repairing
it is legitimate only under conditions you can state in advance:

- the correction is **principled**, derived from the model rather than from the target (here:
  a region cannot peak before the wave reaches it, at the model's own speed `γ_s·r_s`, using
  Euclidean distance — which *under*-estimates the geodesic, so the bound is strict and cannot
  exclude a real peak);
- it is applied **uniformly**, to every element, with no per-element exceptions;
- it is **stable** where the original was not (here: −0.701 to −0.712 at every mode count, and
  the same to three decimals for any fixed cut between 8 and 15 ms);
- it **does not rescue everything** — applied to the authors' own published series it *removes* a
  published significance rather than confirming it, which is what distinguishes a correction
  from a fit;
- it is declared as a **deviation** in the recipe, and the uncorrected numbers are computed and
  reported beside the corrected ones.

Never repair the statistic by dropping the elements that disagree. Excluding the one region that
carried the whole gap reproduces the paper's number exactly and is fitting, not replication.

## Prove a step inert with algebra before hunting for it with sweeps

When a pipeline stage seems not to matter, ask whether it *can* matter before sweeping its
parameters. Pang2023's connectopic mapping compares ROI voxels by their connectivity to the rest
of the grey matter, reduced by SVD to `T−1` components. But `T−1` is a **complete** orthonormal
basis of the demeaned time space, so each fingerprint row is that voxel's own z-scored time
series in a rotated frame and has unit norm — whence `η²(Cᵢ,Cⱼ) = (1 + corr(aᵢ,aⱼ))/2`
identically. The extra-ROI data never enters. Whatever `B` is — cortex, whole brain, one
hemisphere — the answer is the same.

Two lines of algebra replaced a parameter search, and the numerical confirmation took one run
(the two matrices agreed to four decimals in mean and sd and gave identical spectra). It also
redirected the whole investigation: with the fingerprint step proven inert, the remaining
difference had to be in the *input series*, which is where it was.

Corollary for the register: a step that is provably inert at the printed rank is a **class-B**
finding (the code computes a different operation than the prose claims), not a footnote. Someone
implementing the method as printed is not doing what the method is named for.

## Compare two derived matrices as a FUNCTION of a covariate, not element-wise

Element-wise agreement tells you *whether* two matrices differ, never *how*. Bin both against a
physical covariate instead. Pang2023's similarity-vs-distance profile is what turned "our
striatum embedding is worse" into a mechanism:

| distance (mm) | 0–2 | 2–4 | 4–6 | 6–8 | 8–12 | 12–16 |
|---|---|---|---|---|---|---|
| published data | 0.934 | 0.859 | 0.729 | 0.612 | 0.537 | 0.518 |
| ours | 0.681 | 0.552 | 0.500 | 0.503 | 0.502 | 0.502 |

Ours is at chance by 6 mm where theirs still carries structure at 15: their input field is
spatially smoother, which the Methods' own "volumetric voxel-wise" wording implies against our
grayordinate series. Matching the profile then *measured* the difference (≈ 4 mm FWHM), rather
than leaving it as an adjective. Element-wise r between the two matrices was 0.16 and said
nothing at all.

## A derived quantity is DEFINED by the analysis code — and the code's OWN OUTPUT outranks the prose

The single most expensive class of error in a replication with published analysis code is
rebuilding a derived quantity from its *name* and checking it against the statistic the
*manuscript* prints. Names are ambiguous, printed statistics are reachable by more than one
construction, and — this is the part that bites — **the manuscript's number is not always the
one its own pipeline produces.** A plausible reimplementation can then agree with the paper to
one decimal and disagree with the published code everywhere.

In Kadak2025 the phrase "circuit-mean synaptic weight change" denotes **three** different
constructions in the authors' own notebook, and the Methods describe none of them:

| construction | notebook name | r with inter-burst frequency | r with power |
|---|---|---|---|
| mean of the raw per-connection change | `mean_synaptic_weight_delta` | −0.086 | +0.570 |
| **`abs(col) − abs(init)` per connection, THEN a plain mean** | `scaled_syn_change` | **−0.420** | **+0.128** |
| each connection min–max scaled, then averaged | `scaled_nus_normal` | −0.312 | +0.190 |

The manuscript prints **−.31 and .196**, which selects the third row. The notebook's own executed
cells print **−0.420365 and +0.127655**, which selects the second — and the second reproduces
*five* separate printed correlations to six decimals (+0.838946 against LTP calcium, −0.027200
against LTD, −0.849526 and −0.360424 against pulse rate, +0.127655 against power). The third
reproduces none of them. This replication implemented the third, on the strength of its agreement
with the manuscript, and carried it for two sessions with a register entry asserting it was the
verified construction.

**The rule: when a study publishes both its code and the arrays that code ran on, reproduce the
number the code PRINTS, not the number the prose quotes.** A published Jupyter notebook usually stores its executed
outputs inside the `.ipynb` — mine them before writing anything:

```python
import json
nb = json.load(open("notebooks/analysis.ipynb"))
for i, c in enumerate(nb["cells"]):
    out = "".join("".join(o.get("text", "")) or "".join(o.get("data", {}).get("text/plain", ""))
                  for o in c.get("outputs", []))
    if out.strip():
        print(f"--- cell {i}", "".join(c["source"])[:80], "\n", out[:300])
```

That inventory is the replication's real target list. It gives you, per figure panel, the exact
statistic the authors saw — and it makes prose/code disagreements visible as findings rather
than absorbing them into your own error budget. In Kadak2025 it surfaced four: Fig 3D's r
(.196 printed, .128 computed), Fig 3C's r (−.31 printed, −.420 computed), a hardcoded baseline
loop gain (.4201 against .410146 implied by the authors' own published arrays), and a "model-optimal"
marker. Each had been costing a target.

Four corollaries, each of which cost a target in the same study:

- **Audit the WINDOW every published quantity is computed over, per quantity.** Published pipelines rarely
  use one interval for everything. Kadak2025's `load.py` takes spectra over a pre-window and a
  post-window, weight means over those same two, but calcium occupancy and conductance spread
  over `active_stim` — the stimulation window only. This replication averaged the calcium
  indicator over the *whole run*, which agrees to a percent for every cortical connection, whose
  calcium is back below threshold within 50 ms of the last pulse, and overstates the one
  relay-to-reticular connection **26-fold**, whose calcium lingers in the depression band for
  seconds afterwards. Grep the published code for every window variable and list which quantity uses
  which before implementing any of them; a window error is invisible wherever the signal happens
  to be transient and enormous wherever it is not.
- **The plotted quantity is not always the published column.** The same notebook does
  `df['alpha_PW_delta'] = df['alpha_PW_delta'] * -1` immediately before drawing its
  supplementary figure, so the panel and the stored column carry opposite signs. Our figure
  looked inverted against theirs while our column agreed with theirs at ρ = +0.71. **A figure
  that looks flipped is a plotting-convention mismatch until you have checked the published
  column, and the check is one correlation.**
- **A printed marker may name a fit — so reproduce the FIT, not just the number.** The paper's
  "model-optimal protocol (22)" is the centre of a Gaussian over pulse rate, not the argmax over
  the protocol plane (their arrays put that at 40). Scoring the argmax fails a target that in
  fact reproduces. And the fit's *form* is part of the definition: their cell fits
  `amp * exp(-((x-mean)**2)/(2*stddev**2))` with no constant term and prints 21.119; adding an
  offset term — a strictly more general model — moves the centre and manufactures a discrepancy.
  Copy the fitted function, its parameter count and its initial guess out of the published code.
- **Colour is an encoding and needs the same provenance as an axis.** Read `c=` and `cmap=`
  out of the plotting code and check the figure caption; do not infer the colour variable from
  the rendered image, where two correlated candidates are indistinguishable. Kadak2025's caption
  says "Point colors reflect broadband power modulation" and its notebook sets
  `color_variable = data['V_AUC_delta']`; the replication had used a correlated third variable
  throughout, which is invisible in a thumbnail and wrong in every panel.

## A criterion cannot demand more precision than the estimator has — measure the spread, and run the null

A replication target is a claim about a *circuit*, but it is scored on an *estimate*, and the
estimate has a realisation spread that the published paper — running one seed — never reports.
Score against the paper's number and you are testing their draw against yours.

Measure the spread instead. It costs one extra experiment: **the same circuit, the same
observables, the stimulus amplitude at zero, across a handful of noise seeds.** That single
control does three jobs:

- **It sets the tolerance for every criterion.** Kadak2025's alpha-peak target demanded ±0.25 Hz
  of the published 10.344 Hz. Fitting the same spectrum under eight unstimulated seeds gives a
  peak with a standard deviation of **0.90 Hz** on 0.5 Hz frequency bins. The criterion was three
  times tighter than the measurement, and our 0.30 Hz difference was a third of one spread. A
  criterion whose tolerance is not traceable to a measured spread is a coin flip.
- **It exposes nuisance offsets that ride on every cell of a sweep.** A quantity defined as a
  difference between two finite windows — `(AUC_pre − AUC_post)/AUC_pre` and every index like it —
  carries an offset belonging to the noise draw, not the stimulus. In Kadak2025 that offset was
  **−0.036** on the sweep's own draw against an inclusion threshold of 0.061, and it cost
  45 protocols out of 219: the sweep's whole map sat below the published one, uniformly, and no
  amount of looking at the model would have found it. Subtracting the control restored the count
  to 217 and left every correlation untouched, because a common offset moves subsets and not
  correlations. Identify the sweep's draw exactly — the pre-stimulus spectra should match the
  control's seed *bit for bit* — and say which seed it was.
- **It tells you which shortfalls are worth chasing.** Before adding machinery to fix a marginal
  target, sweep the nuisance parameter across its measured range and see whether the verdict even
  moves. In this study that test killed a plausible explanation for a Bonferroni-threshold
  disagreement in minutes: across the entire measured offset range only two of eleven conditions
  changed verdict, and no offset reproduced the published pattern. That is a falsification, and
  it is the cheapest kind to buy.

- **A criterion built on a THRESHOLD CROSSING needs the null run through the paper's own test,
  not through yours.** "Significant in 9 of 11 conditions, failing at these two" reads like a
  structural claim and is really eleven p values landing on one side of a line. Take the
  *published* array for the quantity, add noise at the spread your control measured for that same
  quantity, re-run the authors' own test on each draw, and count how often their own printed
  verdict set comes back. In Kadak2025 it came back in **32 % of 2000 draws**: three of the eleven
  conditions were coin flips (P(significant) = .43, .10, .84) and the other eight never moved. The
  criterion then writes itself, and it is measured rather than negotiated: score the sign
  everywhere and the verdict on the stable conditions, and say in the report which rows the study
  could not have reproduced on itself. Note what makes this legitimate rather than convenient:
  the null is built entirely from *their* data and *their* code path, so it cannot be tuned by
  anything of ours, and it is run BEFORE looking at whether our own verdicts agree.

A design note that follows from the same place: if your sweep runs every cell at a common length
so the grid can be batched, every cell's post-stimulus window is *the same stretch of the same
noise realisation*. Nuisance variation that is independent across the paper's protocols becomes
common-mode across yours, which inflates correlations among post-window quantities. That is a
real consequence of a legitimate design choice, and it belongs in the report as one, not in the
error budget as a mystery.

## "Their result" and "their printed p" are different claims — cross the sources

When a target asks for a correlation **and** its significance, score both datasets under both
nulls under both definitions, varying one thing at a time. The 2 × 2 × 2 is small and it
separates what is otherwise inseparable:

| series | statistic | null | r | P |
|---|---|---|---|---|
| ours | corrected | ours | −0.4548 | 0.089 |
| ours | corrected | theirs | −0.4548 | 0.093 |
| theirs | as published | ours | −0.4372 | 0.034 |
| theirs | as published | theirs | −0.4372 | 0.037 |
| theirs | corrected | theirs | −0.4490 | 0.098 |

Row 1 against row 5 is the honest comparison: 0.006 apart, against a null sd of 0.319. Rows 3–4
show the null is not the difference (every p moves by < 0.005 between the two permutation sets)
**and** that our statistic is sound, because their own vector through our code returns their own
published number. What is unreachable is the printed `P`, and it is unreachable *for them too*
under the same rule. So: **we meet their result, we do not meet their printed p, and the
shortfall is in the published statistic rather than in our field** — three separate sentences,
each defensible, where "T26 failed" would have been misleading in both directions.

Watch the mechanism that produces this, because it is general: the uncorrected map is spatially
*rougher* (a few elements displaced to noise), which **narrows** a spatially-constrained null and
so makes the same |r| look more significant. A significance that depends on the noise in its own
map is worth saying out loud.

## "Unrecoverable" is a measurement you haven't made yet — price it against the authors' own published scatter

When a published result and a published landscape disagree about the same model, or your value
sits off a published curve whose generator was never released, the stopping sentence "the
configuration is unrecoverable" is usually one bootstrap away from a bound. The published data that
ships a result often ships its *ensemble* too (per-subject rows, per-realisation FCDs), and that
ensemble is a draw-to-draw floor you can price every gap against — including the published material's gaps
with itself. In Pang2023, Fig 4b's stored KS and the Extended-Data-10 curve disagreed by 0.0155
at the same optimum; the authors' own published 125 realisations gave a single-draw sd of 0.0295, so the
"internal inconsistency" was 0.53 draw-sd — and our own replication's 0.008 offset was 0.27.
Three moves generalise:

1. **Recover the estimator by elimination.** The stored scalar plus the stored arrays identify
   the reduction between them: pooled-ECDF, per-subject-paired and mean-per-realisation KS gave
   0.0753 / 0.191 / 0.090 against a stored 0.0753 — one candidate is exact, the others are
   excluded. No code needed.
2. **Bound the unrecorded trial count by subsampling the published ensemble.** Pool n of the
   published data's own realisations, ask for which n the published curve value is a plausible draw.
   "Unrecorded" becomes "≤ N".
3. **Read the run configuration out of array shapes.** An FCD stored as C(w,2) pairs encodes its
   window count; two arms with different pair counts were not simulated at matched duration.

The tell that closes the case rather than opening a new one: **curve-min < full-run pooled <
mean single draw** is the argmin-selection signature of a sweep evaluated cheaply per grid point
and confirmed once at the winner — the same bias your own landscape shows if each cell is one
seed. Price it before writing "inconsistent", and price your own gap in the same units before
accepting a tolerance the published material does not hold itself to.

Two traps while you are in there: MAT-file headers embed a creation timestamp, so **md5 cannot
establish content identity between two published repositories** — compare variable names and arrays, not
bytes; and a per-figure "source data" bank (one `.mat` per published figure) is itself evidence —
the figures *missing* from it are usually exactly the unreleased pipelines.
