---
name: fitting-and-sweeps
description: "Hygiene for parameter fits, working-point tuning and sweeps: gate on convergence, never argmax onto a grid edge or into a degenerate regime, and make the selection criterion match the claim. Use when tuning a model to a working point, sweeping a coupling, or fitting parameters against data."
metadata:
  audience: maintainer
  applies_to:
    - "**"
  tags: [fitting, sweeps, numerics, validation]
---

# Fitting and Sweeps

Every rule here comes from a defect that reached a rendered document: a whole-brain model
reported for months at a working point nobody chose, because each step looked reasonable.

## 1. A tuner that did not converge has no result

A tuning loop that ends off target still returns usable-looking parameters. Score them and
you are comparing working points, not the thing you swept.

- **Raise, do not return.** Default the tuner to `strict=True` and make it raise when it
  misses its target. Accepting a best-effort tune must be an explicit opt-in.
- **Verify on an independent run.** A short probe drifts from the point a long run settles
  at. Re-check the target on a fresh, longer simulation before declaring convergence;
  a probe checking its own tuning is not evidence.
- **Gate the sweep too.** Write `NaN` where the tuner missed, so a failed point cannot win
  an `argmax`. Record the ungated value alongside if you want to show what would have been
  reported without the gate.

Seen in practice: FIC converged for G ≤ 1.2 and above that flipped between bistable
attractors (0.9 Hz ↔ 48 Hz) while still returning `J_i`. The ungated sweep's winner was a
10 Hz point scored as if it were the 3 Hz one.

## 2. The target must be the one the docstring names

If the module says it tunes a firing rate and the code tunes a gating variable, the code is
wrong even when it converges. Check that the constant being optimised is the one the cited
method uses — and that named constants defined for that purpose are not sitting unused.

Related: a constant that encodes an operating point (an offset, a centring term, a
reference level) must follow the working point, not be hardcoded. Change the working point
and a stale constant becomes a silent DC bias.

## 3. An optimum on the grid edge is truncation, not a result

If the best value sits at the first or last point you tried, you have not found an optimum;
you have found the edge of your grid. Widen it until the curve turns over, or until a gate
rejects the points beyond. Emit a warning from the sweep itself when the argmax lands on an
end point — do not rely on noticing.

## 4. A metric can improve monotonically into a degenerate state

Goodness-of-fit does not stop rising when the model stops being a model.

- Ask what the metric does as the system degenerates, and gate on that directly:
  variance floor, saturation ceiling, collapse fraction across seeds.
- **A working point that depends on the noise seed is not a working point.** Score
  candidates over several seeds and reject any where some seed collapses.

Seen in practice: FC–FC `r` rose monotonically with coupling right into full
synchronisation (`FC = 1.000 ± 0.000`, `r = 0.403` computed on the residual variance).
The rate gate passed it. Only a seed-stability and saturation gate rejected it.

## 5. Select on the quantity you are claiming

If the headline is a fit quality, do not select the reported model with an objective that
also contains a regulariser — it will trade the claim away and you will report a model
chosen partly for something else. Run the penalised variant as a **control** and report both.

A penalty that improved both terms at one working point may trade them at another. Re-test
a control after any change to the operating point; do not carry its conclusion forward.

## 6. Re-derive the whole chain, not the endpoint

When a working point changes, everything initialised from it, averaged over it, or scored
against it is stale: fits, ensembles, per-subject evaluations, figures, documents. List the
dependency order once, script it, and run it as one job. Artefacts that do not depend on the
changed quantity (a structural null, a linear reference) should be identified and left alone.

## 7. Bound and guard long runs

Fits and sweeps here are hours, not minutes. Run them under the RSS watchdog via
task-spooler rather than in the foreground, and give each stage a printed marker so a
resumed or killed run can be picked up at the right step.

---

**These rules are working if:** no reported number comes from a tune that missed its target,
no optimum sits on a grid edge, and every headline metric has a stated gate against the
degenerate regime it would otherwise climb into.
