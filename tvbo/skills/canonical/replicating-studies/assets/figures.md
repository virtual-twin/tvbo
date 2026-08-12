# Phase 5 reference — sizing, binding and drawing a panel

Read this while writing a study's `figures:` block. The spine holds the rules a
figure must obey (integrity, captions, coordinate convention); this file holds the
mechanics — the layout keys, how to measure the original's size and type scale, how
a panel binds its data, and when a panel is allowed to be code.

**A `Figure` is layout + binding + style; keep compute and plotting code out of it.**
- **Layout is metadata:** `layout` (bsplot mosaic string, e.g. `aab/ccb` — letters = panel
  keys, `/` = new row, repeated letters span, `.` = empty), `width`/`height` (mm), `dpi`,
  `font_size`, `height_ratios`/`width_ratios`, `style` (`.mplstyle` paths), `spines`,
  `panel_numbers`/`panel_number_format`/`panel_number_loc`. Set the paper's physical size and
  type scale here, once — never in code.

**Size, aspect and type size are MEASURED off the original, not guessed — get them right on
the first render.** Three defaults are wrong for a replication and cost a re-render every time:

- **Aspect.** Derive `height` from the original's own pixel aspect: `height = width ×
  (h_px / w_px)` of the paper's figure scan (a Nature two-column figure is `width: 183`, a
  single-column one `88`). Then set **`trim_margins: false`** — the default trims to content
  (`bbox_inches="tight"`), which re-crops the saved PNG and makes its aspect drift away from
  the declared `width × height`, which is exactly the "why is my figure the wrong shape"
  symptom. Check it: the rendered PNG's `h/w` must equal `height/width` to ~1 %.
  **But `trim_margins: false` also exposes a layout failure that trimming used to hide.**
  A dense mosaic (many panels, a very short row, empty `.` cells) can starve matplotlib's
  constrained-layout solver — it warns `axes sizes collapsed to zero` and silently drops
  EVERY axes back to the default 12 % subplot margins, giving a figure ringed by ~8 % white
  that neither `set_layout_engine("none")` nor `subplots_adjust` nor `height_ratios` will
  shift. Diagnose by measuring the ink bounding box of the PNG; if padding is ~8 % on all
  four sides while a sibling figure is at ~1 %, that is this bug and not your margins. The
  fix is to let it trim and declare the size OVERSIZE so the trimmed result lands on the
  target (iterate twice — measure, rescale, re-render), and **re-measure after any type-size
  change**, because the trim moves with it.
- **Panel proportions.** A mosaic alone distributes rows/columns EQUALLY. If the paper's rows
  or columns are unequal (a short schematic row above tall matrix panels), measure their pixel
  extents in the original and set `height_ratios` / `width_ratios` — otherwise every panel is
  subtly the wrong shape even though the figure size is right.
- **Type size — the single most repeated defect in this skill's history. VERIFY IT IN PIXELS,
  every time, before you call a figure done.** Set **`font_size: 9`** for a 183 mm figure and
  **10** for an 88 mm one as the *starting* value, never below 8. Then MEASURE, because two
  independent traps make the declared number a lie:

  1. **A study `.mplstyle` silently overrides `font_size`.** If the study ships a style file
     that sets `font.size`, it used to be applied *after* the declared size and won — so
     `font_size: 8` in the spec rendered as whatever the style said (Pang2023: 5 pt) with
     nothing in the spec, the log, or the emitted script to show it. tvbo now applies the
     `.mplstyle` FIRST and the declared `font_size` last (regression-tested), but **any style
     file you write must still be checked**: grep it for `font.size`, `axes.labelsize`,
     `xtick.labelsize`.
  2. **A point size is meaningless without the width it was measured at.** Apparent size is
     the ratio of glyph height to figure WIDTH. Pixel forensics on a 120 mm (1.5-column)
     original that you then reproduce at 183 mm yields type ~1.5× too small — and the same
     error scales every `linewidth`, `markersize` and tick length in the file. If a
     `.mplstyle` is derived from measurements, record the width they were taken at and
     rescale by `target_mm / measured_mm`.

  **The check (do it, don't assume):** binarise the rendered PNG and the paper's own scan,
  take connected components with `5 <= h <= 40 px`, and compare the modal glyph height as a
  PERCENTAGE OF IMAGE WIDTH. That ratio is resolution-independent, so it compares a 953 px
  scan with a 2161 px render directly. Journals run ~0.7–0.85 %; land within that or slightly
  above. **Aim a little above the original** — a Nature figure's 6 pt labels are legible at
  183 mm in print and illegible on screen at 2000 px, and every replication we have shipped
  erred small, three of them after this rule was already written down.
- **Grammar panels need zero code.** A `cartesian` or `heatmap` panel binds data through its
  `layers`: `used: {iri: tvbo:exp/<Study>/exp-3, output: <var|observation__name>, sel: {dim: label}}`
  (label-keyed, never positional — this binding **is** the PROV `used` edge), plus `mark`
  (`line`/`scatter`/`rule`/`band`/`area`/`bar`; implied for heatmap) and `encoding: {x, y, color}`
  naming container dims/coords. **`band` draws a spread** (`fill_between`) and its output must
  carry a length-2 axis beside the swept one — the analysis returns `mean ± sd` as ONE
  `(n, 2)` array with a `bound: [lo, hi]` coordinate, so a figure cannot bind a lower edge
  from one run and an upper edge from another. Draw the band layer BEFORE its mean line, or
  the fill covers the curve it belongs to. **`rule` draws a reference line at a value the
  CONTAINER holds** — an ensemble mean, a published number the recipe declared as an analysis
  argument and the analysis echoed back — with the encoded channel picking the orientation
  (`x:` vertical). Prefer it to the `axvline`/`axhline` opts, which take a literal typed into
  the spec and render as subdued gridlines: a marker the figure exists to make is worth a
  styleable layer and a PROV edge. `transform:` names an optional presentation-only
  reduction. Bind an
  **in-study** experiment by id — `used: {experiment: 3}` — rather than spelling a full `iri`:
  it needs no hardcoded study key and registers the run-order dependency (that experiment runs
  before the figure). Reserve an explicit `iri` for a curated/external container.
- **Only a bespoke interior is code.** A `custom` panel sets `render: <fn>` + `opts:`, where
  `<fn>` is a `@bsplot.register_panel` callable `fn(fig, ax, ctx)` in a module named in the
  figure's `code_modules:` (a flat file in `code/`, e.g. `code/<study>_figures.py`). It reads its resolved layers with
  `bsplot.load_layer(ctx["layers"][i])` and draws. A reused reduction is a
  `@bsplot.register_transform` `fn(da)->da`. This is the escape hatch — reach for it only when
  the grammar genuinely can't express the panel (twin axes, connectome, brain surface, dense
  nested subgrids), not by default.

## Choosing WHICH point a marker marks

**A marked/sampled point must match what the paper says it IS — read the figure description, then
verify via the panel it feeds.** When a `custom` panel marks sample points on a curve (three
periodic orbits 1/2/3 on a period-vs-parameter branch), select each from the paper's figure
caption/description, not a guessed heuristic: the description names what the point is — its period
band **and its morphology** ("point 2 = the *mid* orbit, an asymmetric spike with a slow rise") —
and that fixes which branch point to take. Then verify against the panel the marker drives: the
marked orbit's waveform sub-panel must show the described shape. (An argmin-on-period heuristic put
our "2" at the bottom-corner fold — a too-symmetric spike; the description's "asymmetric spike +
slow rise" is what identified the elbow one bend up as the right orbit.) This is Phase 7's
shape-check applied to marker placement — the caption/description is the oracle for *which* point,
not just how to word it.

## Binding the paper's own published data

**External published paper data binds by IRI too.** When a panel pairs TVBO output against the
paper's own figure data, wrap that data as an external `Dataset` and bind
`used: {iri: tvbo:dataset/<Study>_source, output: <var>, sel: {figure: 6, panel: c}}` — the
same declarative path, figure/panel as coordinates you `sel` into. Until wrapped, a **flat,
label-keyed** per-panel `.nc` set (`xarray` named coords, not filesystem-keyed) is an accepted
stopgap; don't build an elaborate filename tree — it's throwaway once the `Dataset` binding lands.
