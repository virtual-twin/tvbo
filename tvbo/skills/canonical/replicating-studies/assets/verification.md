# Phase 7 reference — building the oracle

Read this when you reach Phase 7 of **replicating-studies** and the study has
something to verify against: a closed form, a reference implementation, or the
authors' own published arrays. The spine states the rules; this file is how you
build the instrument. If a number of yours already disagrees with a published one
and you have the published material open, read `published-artifacts.md` instead.

## When the paper published repositories its own ANALYSIS OUTPUTS, demand identity (r = 1, RMSE ~1e-15)

Many published repositories contain not just inputs but the authors' own *derived* arrays (accuracy curves,
power spectra, permutation sets). That converts verification from "do we agree roughly?" into
an exact test: run **our** implementation on **their** inputs and require machine precision.
Write it as a standing harness (`code/verify_identity.py`) that prints one table, because it
is the thing you re-run after every refactor. Classify each check up front — mixing the
classes is how a replication overclaims:

| class | meaning | criterion |
|---|---|---|
| `identity` | deterministic, same inputs, same algorithm | RMSE ≲ 1e-12. **A failure is OUR bug.** |
| `convergent` | deterministic but solver-tolerance-limited | agreement stated *with its floor* |
| `stochastic` | depends on an unpublished seed | distributional only — matching an exact number would mean we tuned to it |

Identity is a *discriminating instrument*, not a rubber stamp — it localises bugs that a
correlation would hide. Four traps it caught in one study (Pang2023), each of which would
have produced plausible, wrong figures:

- **The published data ships several versions of "the same" array.** The basis under
  `results/basis_geometric_*` differed from `template_eigenmodes/*_emode_200.txt` by 4.2e-2.
  Both look right; only one gives identity (5.6e-16 vs 2.6e-6). **Try every candidate and let
  identity pick** — never assume the obviously-named file is the one the figures used.
- **Order of a nonlinear step.** A normalised power spectrum averaged over subjects is NOT
  the spectrum of the subject-averaged map: r = 0.885 vs r = 1.0000000000. Whenever a
  statistic normalises, establish *where* the averaging happens; the paper's prose often
  won't say, and only identity distinguishes them.
- **"Improving" the reference algorithm breaks it.** Symmetrising a Gram matrix before
  solving is numerically defensible and *wrong here* — port the reference's arithmetic
  exactly (`(Ψ'Ψ)\(Ψ'y)`), because identity against it is the criterion.
- **Masked/NaN vertices silently poison a least-squares solve.** One NaN turns an entire
  reconstruction into NaN. Restrict to the analysis mask the paper uses (its cortex mask),
  and treat an all-NaN result as a convention bug, not a data problem.

Two mechanical ones worth a checklist line: when loading a `.mat`/HDF5 reference, select the
dataset **by name** (`eig_vec`), never "the first key" — sibling arrays like `eig_val` sort
first and load silently; and MATLAB HDF5 arrives **transposed**, so confirm orientation
against a known dimension rather than by eye.

**A cross-check experiment should RECORD on the grid it will be compared against.** When one
experiment exists to bound another's error, declare its observation at the *other* run's
sampling period (`iri: tvbo:SubSample`, `period: <the other run's dt>`, `reduce: streaming`)
rather than recording its own — much finer — solver step. The two then share one time
coordinate by construction, so the comparison needs no interpolation and no positional
decimation, and the container stops being an artifact in its own right: Pang2023's vertex-space
check went 2.3 GB → 151 MB and 10 min → 1m23, because the *write*, not the solve, was eight of
those ten minutes. Recording every step of a 32,492-node field "in case we need it" is also how
you stall the whole machine — that write filled the page cache and collapsed throughput for
unrelated work that followed, which reads as a hung job rather than as the disk-bound write it is.

**Report a cross-check that does not converge AS unresolved.** Do not quote a bound from a
diverged run, and do not quietly drop the target. Say what was measured (the step sizes tried,
where it left the physical range, the growth rate at each), separate what that *does* exonerate
(here: the analysis chain, verified end to end on the diverged container) from what stays open
(the discretisation), and mark the row `short` with the open question as its reason — it was
attempted and its criterion is not met, which is the one verdict that says so. An unresolved
verification honestly reported is a result; a missing one is a gap in the replication.

## A harness that ABORTS reports success for every check it never reached

`check_declared_signs` raised on the first multi-output container it met, so the sign vectors it
existed to guard had not been verified for an unknown number of sessions — and the harness
printed a clean summary the whole time, because a check that never ran is not a check that
failed. Two rules follow:

- **A check that cannot run must FAIL, not vanish.** Count the checks you expect and assert the
  count, or have each check register itself before it can raise. A summary line that says
  "50 checks, 0 failing" must mean fifty were attempted.
- **Legitimate exemptions belong IN the harness, named and reasoned** — never as an absent
  check. Two are typical: a comparison whose pairing is too weak to determine an answer
  (below |cos| 0.5 the reference's own sign is noise, and failing a vector for declining to
  assert one is backwards), and an element deliberately gauged against a different target
  (the indices a figure actually draws may follow the *published panel* rather than the
  published array). Both are one named constant with a register reference; both then show up
  in the check's own label (`8 flips, 12 of 20 determined`) so a reader sees the denominator.

Read multi-output containers by the **declared** output name. An analysis returning a dict names
its arrays by key, not by analysis name; take the key from the recipe's own `used:` edge rather
than guessing, so a rename cannot silently turn a check into a skip.

## A verification script must parse the SPEC with the loader, never with a regex

An oracle that greps YAML is one refactor from silently checking nothing. Pang2023's declared-sign
check matched analysis names with a regex; the spec then grew `!include` fragments and anchors, the
pattern matched zero rows, and the check reported every vector as fine. Load the study
(`SimulationStudy.from_file`) and walk the objects — then a renamed analysis fails loudly instead
of passing vacuously. The same applies to any check that asserts something about the recipe:
read it through the same loader the run uses, or you are testing a different document.

**Cover every artifact the report QUOTES, not just the numbers you were last debugging.** A
harness that asserts physics and polarity will happily pass while a headline deliverable the
report parses — a divergence register, a targets table — is corrupt on disk (below). If the
report reads a file to produce a number, the harness checks that file. Cheap and sufficient:
re-parse it with the *same* helper the report uses and assert the structural invariants a
corruption breaks (no duplicate row ids, more than one class present, every verdict in the
declared vocabulary).

**A standing check must compare a quantity against a reference of ITSELF.** A harness row that
pits one convention against another measures the gap between definitions and reports it as a
failure of your arithmetic. Ours compared a spin-test p computed under the reference
implementation's one-sided, direction-averaged, uncorrected rule against a hand-rolled
two-sided `(k+1)/(n+1)` p, and failed forever on a difference that was the point of having two
definitions. Where a function offers several conventions, check *each* against a hand-rolled
reference of that same convention — and treat a long-standing red row as a bug in the check
until you have re-derived it, since a permanently-failing check trains you to ignore the table.

## When NO output data is shipped, an unverified convention is an ASSUMPTION — label it

The identity checks above only exist because that published data happened to include the authors'
derived arrays. **Most do not.** The failure mode is subtle and expensive: with nothing to
test against, a plausible reading of the Methods gets written into `targets.md` as though it
were established, every downstream number inherits it, and the report states it as fact.

The tell is that the paper's prose *underdetermines* the computation. "The power spectrum of
the group-averaged maps" does not say whether the averaging precedes or follows a nonlinear
normalisation — and those differ by r = 0.885 vs 1.0. Prose almost never pins down: where an
average sits relative to a nonlinear step; which of several shipped files is "the" basis;
whether an analysis runs on all vertices or a cortex mask; 0- vs 1-based indices; whether a
"correlation" is over vertices or parcels.

So, when you cannot verify:

1. **Write the assumption down as an assumption**, in `targets.md`, next to the target it
   feeds — not as a statement of what the paper did. Phrase it "we read X as Y; not
   verifiable from the published material".
2. **Enumerate the plausible alternatives you rejected**, and say why. If you cannot name an
   alternative, you have not understood the choice well enough to make it.
3. **Test sensitivity.** Compute the target under each candidate convention. If they agree
   to within the reported precision, the ambiguity is harmless — say so and move on. If they
   disagree materially, that is a *first-class limitation* of the replication, and the
   scorecard must show the range, not one arbitrarily-chosen member of it.
4. **Never let an assumption harden into an assertion** through repetition. A convention you
   guessed in Phase 1 is still a guess in Phase 6 unless something verified it in between.

This is the same discipline as **doubting a claimed discrepancy** — default to "we may have
misread this", and make the uncertainty visible instead of resolving it silently.

## A derived object with a FREE CONVENTION: gauge it on the DISPLAY path, never in the solver

Some products are defined only up to a convention the mathematics does not fix — an
eigenvector's sign, an ICA/PCA component's sign, the ordering inside a degenerate eigengroup,
the rotation inside an NMF factorisation, a gradient's direction. Half of any independent solve
then comes out mirror-imaged against the paper, which reads to a reviewer as a wrong result. Four
rules, in order; each of them was learned by breaking it.

**1. Apply the convention only where the object is DISPLAYED.** If anything *integrates, fits or
projects in* that basis, its coefficients are defined relative to the basis **as the run saw
it**, so re-gauging afterwards projects the result through a convention the run never used. A
deterministic sign rule placed inside Pang2023's eigensolvers scrambled the wave model:
corr(field, V1 stimulus) = −0.33 where either self-consistent pairing gives +0.92, V1's response
moved from 6.1 ms to 26.8 ms, node FC fell 0.618 → 0.205 — and the mismatch was exactly the gauge
vector on all 200 modes. Structure it in the recipe as two nodes: `<name>_raw` (the solve) →
`<name>` (one declarative `apply_signs`-style transform). Panels bind the aligned node;
eigenvalues, stimulus weights, the noise covariance and every projection bind the raw one.
**One-step diagnostic** when you suspect two bases are in play: correlate the quantity the run
actually *drove with* against a freshly produced copy of it — ~+1 means one basis, a per-element
±1 pattern names the transform that got inserted between them.

**2. Measure that no principled rule exists before you write a literal one.** A hardcoded
alignment vector is either an honest record of an arbitrary convention or a bug wearing a
constant, and only a measurement tells them apart. Enumerate the candidate data-only rules
(max-|value|, sum, third moment, positive mass, first element) and score each against the
published data. Pang2023's scored at **chance** — 94–106 of 200 modes — and its three graph bases
disagree with *each other* on the leading mode, which Perron–Frobenius fixes as non-negative.
That measurement is what licenses the literal vector; without it, do not write one.

**3. Derive the constant from the exact container the recipe applies it to.** Not from a fresh
call to the same solver — same modes, different signs, because an iterative eigensolver's output
depends on restarts and thread order. Deriving Pang2023's vectors from a direct
`surface_eigenmodes` call instead of the produced `*_raw` container left 99 modes wrong and read
as a 101/200 near-chance result, which looks exactly like a failed alignment rather than a
sampling mistake.

**4. Prove the transform moves nothing scored, with a number.** Recompute every scored quantity
with and without the gauge and assert the worst |Δ| is at rounding (ours: 0.000e+00). A cosmetic
transform that changes a result is not cosmetic.

**The recipe must run without the published material; the ORACLE may read it.** This is the general split
that makes a declared alignment legitimate rather than a hidden dependency on reference data. The
literal vector goes in the spec as metadata (non-negotiable #1 — the recipe renders a figure
with no published data on disk); a `verify_identity` check re-derives it from the published arrays and
fails on drift. Same rule for any hardcoded convention: constant in the spec, derivation in the
oracle, and say in the report which it is.

## For a LINEAR model, don't fit a scale — invert the transfer function

When a replication's output has the right shape but the wrong magnitude, the instinct is to
report a best-fit scale factor. For a linear model that is the weak measurement, because the
fit absorbs every other residual — basis truncation above all — and lands on a number that is
neither the true scale nor obviously wrong. In Pang2023 the forward fit read 1.85 against a
4–8 % truncation floor, and sat unexplained for a long time.

Invert the model instead. A linear system's own transfer function is exactly invertible, so
the published OUTPUT determines the INPUT that produced it:
`Q(ω) = Φ(ω)·[−ω² + 2iωγ_s + γ_s²(1 + r_s²λ)]/γ_s²`. That returned a flat boxcar of amplitude
**10.00 ± 0.05** where the Methods said 20 — a factor of exactly 2, settled in one step.

Two reasons this beats fitting:

- **It is truncation-consistent.** The same basis appears on both sides, so the error that
  contaminates a forward comparison cancels instead of biasing the estimate.
- **The recovered input's SHAPE is a self-test of the whole model.** A flat rectangle can only
  come out if `γ_s`, the damping term, the eigenvalues and the stiffness are all right; any
  error makes the recovery frequency-shaped. So the measurement validates the model and
  quantifies the discrepancy at once — you are not merely asserting agreement.

Generalises to any linear or linearised stage: a haemodynamic convolution, a filter, a modal
projection. Where the model is nonlinear, invert around the operating point and say so.

**Port a statistical procedure from the reference implementation, not from its description.**
A spin test is the canonical example: naive nearest-neighbour matching of rotated parcels is
*not a permutation* (parcels get duplicated and dropped), which biases the null; the published
method (Váša `rotate_parcellation.m`) does a greedy "most distant minimum" assignment
**without replacement**. Also force `det = +1` — the QR of a Gaussian matrix can be a
*reflection*, which is not a rotation of the sphere. Where the published material contains its own
permutation set, use **theirs** to verify your statistic, which isolates the test from your
RNG; then check your own generator separately (every row a true permutation).

## Verify the figure, not only the number

**Measure the layout, then eyeball the shape.** Declare each figure's published counterpart
with `reference_image: original_study/img/fig_0N.png` and run `tvbo figure compare
<Study>.yaml`: it decomposes both images into panel boxes (recursive XY-cut), matches them by
overlap, and writes a per-panel offset table plus a side-by-side overlay. Page **aspect** is
the number to read first — it is exactly reproducible and it catches the whole class of "the
figure is the wrong shape" that survives every value check. A deliberate aspect difference
(a panel of the paper's you do not draw) is fine, but it belongs in the figure's
`description:` as a stated departure, not as an unexplained 1.14-against-1.75. The panel
counts often disagree because a published raster's panels touch where yours have gutters;
read the offsets only where the counts agree. Identifying the counterpart is itself worth the
few minutes: published repositories number their images `fig_01…fig_NN` with no mapping to "Extended Data
Fig 10", the offset from main-text numbering is *not* uniform, and the only reliable way is to
open the candidates — doing so is what turned Pang2023's `r_s` landscape from an
uncomparable panel into one measurable at aspect 1.272 against 1.280.

**Eyeball every reproduced panel's *shape* against the paper — the A/B internal composite is
the instrument, not a formality.** Inline-computed numbers (non-negotiable #2) catch a wrong
*value*, but a curve that plateaus where the paper's descends, a flipped monotonicity, a sign
error, or a saturated axis still *computes* a number and sails through a value check. Lay the
reproduction beside the original panel-for-panel and confirm the qualitative shape before
declaring a figure done — a mismatch there is a modelling/analysis bug the reference
integration alone won't surface. (Taher Fig 9(d): one strategy curve sat as a flat plateau
instead of the paper's staircase descent — the visible tell of a broken solitary set, invisible
in the scalar metrics.) This is also the moment a stale caption shows up: prose written before a
later fix (a "not yet wired" follow-up that since shipped) must be reconciled with what the
panel now shows.
