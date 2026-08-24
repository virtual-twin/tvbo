# Phase 4 reference — the replication-pairs contract

Read this when a study must state its published-vs-reproduced numbers, or when an existing
study is being migrated onto the contract. The spine states the rule; this file is the schema,
the migration recipes by container shape, and the traps that migration hits.

A study states its findings as **pairs**: a number the paper published beside the number this
study reproduced. One analysis, one schema, portfolio-wide — so a consumer joins on numbers and
never parses prose or per-study naming. Before the contract every study invented its own layout
and every consumer carried a per-study adapter; that adapter is the thing this removes.

## The two artifacts

1. **`docs/analysis/published-values.md`** — the one transcription of the numbers the paper
   printed, each row carrying where in the paper it came from. Published values are read from
   here and **never typed in code**. A value typed into a Python literal is a value no reviewer
   can trace and no gate can check.
2. **A `replication_pairs` analysis** declared in the study spec, joining that transcription
   against this study's own containers, built with `tvbo.analysis.replication.pairs_payload`.

```python
from tvbo.analysis.replication import pairs_payload


def replication_pairs() -> dict:
    """This study's pairs in the portfolio-wide contract, one row per joined target."""
    return pairs_payload(
        {
            "quantity": key,
            "published": pub,
            "reproduced": ours,
            "kind": ...,
            "published_provenance": ...,
            "join_sound": ...,
            "units": ...,
            "published_source": ...,
            "reproduced_from": ...,
        }
        for key in targets
    )
```

```yaml
  - name: replication_pairs
    label: "Replication pairs: every published <Study> number beside the reproduced one"
    description: >-
      ... what is joined to what, and from which transcription ...
    callable: {name: replication_pairs, module: <study>_analysis}
```

## The schema

`pairs_payload` emits one parallel array per field, in row order, and validates as it goes.

| field | required | meaning |
|---|---|---|
| `quantity` | yes | the target's name; unique within the study |
| `published` | yes | the number the paper printed |
| `reproduced` | yes | the number this study produced |
| `kind` | yes | what **our** side is |
| `published_provenance` | yes | where the **paper's** side came from |
| `join_sound` | yes | false where the two sides are not established to denote the same object |
| `deviation`, `abs_deviation` | computed | **relative**, computed here so no two studies compute them differently |
| `units` | should | the unit both sides are in |
| `published_source` | should | where in the paper the value is printed |
| `reproduced_from` | should | which container key ours was read from |

`kind` ∈ `measured`, `closed_form`, `configured`, `degenerate`. A `configured` pair is an input
we set rather than derived; a `degenerate` pair cannot carry a relative deviation (the paper
published zero).

`published_provenance` ∈ `printed`, `axis_read`, `rederived`, `bound`, `released_array`,
`not_in_paper`. This bounds how far a deviation may be read: a value taken off a plotted axis
cannot be held to more precision than the axis, and a `bound` is not a claimed result at all.

A term outside either vocabulary, or a missing required field, raises rather than serialising
something a consumer would silently misread.

## Checking conformance

```python
import h5py
from tvbo.analysis.replication import conforms

with h5py.File(container) as f:
    print(conforms(f) or "CONFORMS")  # -> [] when it conforms, else the missing fields
```

`conforms` looks fields up both bare and under the `observation__` prefix the writer adds, so an
open `h5py.File` may be passed directly. **A consumer discovers a study's pairs container by
conformance, not by filename** — which is stronger than a naming convention, because a study is
onboarded by satisfying the contract and no existing container has to be renamed.

## Migration recipes, by the shape you find

Four shapes cover every study met so far. In all four the study already computes canonical rows
somewhere; only the serialisation differs. Find that row producer first — grep for `join_sound`
— and route it through `pairs_payload` rather than rewriting the study's joins.

**1. Row table already** — parallel `published`/`reproduced` arrays. Usually only the derived and
optional fields are missing. Compute them through the helper rather than by hand, so the
definition of deviation stays in one place:

```python
payload = pairs_payload([{**r, "reproduced_from": r["reproduced_key"]} for r in records])
for r, dev, adev in zip(records, payload["deviation"], payload["abs_deviation"], strict=True):
    r.update(deviation=float(dev), abs_deviation=float(adev))
```

**2. Per-key scalars** — `published__<key>` / `reproduced__<key>`. Build rows from the key list
and spread the payload beside the existing dicts. The per-key scalars may stay: flat dataset
naming means `observation__published` and `observation__published__<key>` are distinct names
that coexist, so nothing that reads the old form breaks.

```python
rows = [
    {
        "quantity": k,
        "published": paired[k][0],
        "reproduced": paired[k][1],
        "kind": TARGET_KIND[k],
        "published_provenance": PUBLISHED_PROVENANCE[k],
        "join_sound": k not in JOIN_UNSOUND,
    }
    for k in keys
]
save_result(
    "replication_targets",
    {
        "published": {k: paired[k][0] for k in keys},
        "reproduced": {k: paired[k][1] for k in keys},
        **{f: column(v) for f, v in pairs_payload(rows).items()},
    },
)
```

Wrap each array in whatever axis carrier the study already uses — `xr.DataArray(v,
dims=["<row axis>"])` — so the flattener keeps the new arrays aligned with the old ones.

**3. Nested per-pair leaves** — `pair__<name>__published`. Same move; the row producer usually
returns exactly the right dicts and needs only a field rename:

```python
rows = pairs()
return {
    "pair": {p["name"]: {...} for p in rows},
    **pairs_payload({**r, "quantity": r["name"], "reproduced_from": r["reproduced_key"]} for r in rows),
}
```

**4. No pairs container at all** — the published side lives in a Python module. This is the
largest case and the most worth fixing: add the `replication_pairs` analysis to the spec and
have it read the transcription, so the paper's numbers leave executable source and become a
tracked artifact.

## Traps

- **`deviation` is relative, portfolio-wide.** A study that already emits a field of that name
  meaning an *absolute* difference must rename its own — `difference_mV`, say — and every
  consumer of it updated. This is the same-name-different-meaning trap the register records as
  class E, and it is easy to introduce here: the report keeps printing the field with a unit
  label while the number underneath has become a ratio. Grep the study's `report.qmd` for the
  field before you take the name.
- **Label arrays are usually already per row.** `kind`, `published_provenance` and `join_sound`
  are per-row arrays in every study met so far, even where a flat key listing makes them look
  like single values. Check the shape before concluding they need re-deriving.
- **Watch the import indentation.** These row producers are often nested functions with local
  imports; inserting `from tvbo.analysis.replication import pairs_payload` at column 0 next to an
  indented `import pandas as pd` silently breaks the module. Parse the file after editing:
  `python -c "import ast;ast.parse(open(F).read())"`.
- **A spec that will not load blocks the re-emit.** Where the analysis is spec-driven and the
  spec no longer loads against the installed tvbo, the study cannot be re-run at all. Studies
  whose pairs are written by a module invoked directly are unaffected. Check every spec loads
  before planning a migration: `tvbo run <spec> --analysis __probe__` prints `No analysis named`
  when the spec is fine, and the rejected keyword when it is not.

## Runbook — closing the gap on one study

1. `tvbo run <spec> --analysis __probe__` — confirm the spec loads. If it names a rejected
   keyword, fix that first; nothing else can run.
2. Locate the row producer: `grep -rn join_sound <study>/code/`.
3. Confirm `docs/analysis/published-values.md` exists and holds the paper's numbers. If the
   published side is in Python literals, move it there first.
4. Route the rows through `pairs_payload` using the recipe matching the shape you found.
5. Check the field-name collisions above, especially `deviation`.
6. Re-run: `tvbo run <spec> --analysis replication_pairs`, or the module directly where the
   study writes its container that way.
7. `conforms(container)` must return `[]`.
8. Prove it lossless: the pairs a consumer reads before and after must be identical in count and
   in every value. A migration that changes a number is a bug, not an improvement.
