# Phase 7 reference — a number of ours disagrees with theirs

Read this when the deposit ships the authors' own derived arrays and your
implementation does not return them. Every section is a way to establish *which*
of the two is wrong before rewriting either. Work them roughly in order: the cheap
structural checks — what is actually in the files, what order it is stored in —
come before the expensive ones.

The default holds throughout: **a claimed discrepancy is our bug until a
falsification test says otherwise.** These checks exist to run that test, not to
assume either answer.

## Inventory the deposit by CONTENT, not by filename — a mislabelled file can be the oracle

Read every deposited array's **shape and range** before you read its name. Pang2023 ships
`results/hcp_striatum-lh_gradient_20_variance.txt`, whose two siblings are 200-element variance
spectra; the striatum's is a **1,896 × 1,896 affinity matrix** — the *input* to the one step of
the chain the deposit ships no code for, for the one structure of three that was failing.
`numpy.loadtxt` returns it without complaint and its first row reads as a plausible decreasing
spectrum, so nothing announces the mistake. Finding it converted a dead end into an exact test:
run our embedding on their own matrix and require their own gradients back.

The check is one pass and costs nothing: for each file, print `shape`, `min`, `max`, whether it
is symmetric, whether the diagonal is constant. A square matrix where the name promises a vector,
a range in `[0, 1]` with a unit diagonal where the name promises eigenvalues — these are the
tells. **Do this before concluding a step is unverifiable**, because the reason a step looks
unverifiable is usually that its oracle is filed under the wrong name.

## Find an ORDER-INVARIANT oracle when you don't know the deposit's storage order

Comparing two derived arrays requires knowing how each is indexed, and deposits rarely say.
MATLAB's `find` walks a volume column-major and numpy's `np.where` row-major, so the same voxel
set arrives in different orders; compared positionally, correct modes score |r| = 0.18 against
their own counterparts and reindexed they score 0.994. **A wrong order looks exactly like a
failed replication**, and worse, it looks like it in a way that invites you to go and "fix" a
pipeline that was right.

The way out is to score against a quantity that **does not depend on the order at all**.
Eigenvalues are the common one: `λ(P S Pᵀ) = λ(S)` for any permutation `P`, so a deposited
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
- it **does not rescue everything** — applied to the deposit's own series it *removes* a
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
| deposit | 0.934 | 0.859 | 0.729 | 0.612 | 0.537 | 0.518 |
| ours | 0.681 | 0.552 | 0.500 | 0.503 | 0.502 | 0.502 |

Ours is at chance by 6 mm where theirs still carries structure at 15: their input field is
spatially smoother, which the Methods' own "volumetric voxel-wise" wording implies against our
grayordinate series. Matching the profile then *measured* the difference (≈ 4 mm FWHM), rather
than leaving it as an adjective. Element-wise r between the two matrices was 0.16 and said
nothing at all.

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
