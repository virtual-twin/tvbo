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
- **Correcting a too-tall figure BREAKS its labels, and the fix is a label budget, not a taller
  figure.** A figure whose declared height is 30 % over the original has been hiding a crowding
  problem: cut it to the measured aspect and the axis labels, colourbar labels and panel numbers
  start writing over their neighbours. Do not put the height back. Spend the space instead --
  shorten a y label ("Circuit-Mean Synaptic Weight Change (Δ)" to "Weight Change (Δ)"), drop a
  colourbar label the caption already states, widen the mosaic gap column between a colourbar and
  its neighbour's tick labels, and pull `panel_number_offset` in. The tell that you are looking at
  this and not at bad data is six-decimal tick labels: a narrow panel with an automatic locator
  prints `0.00102747` where the paper prints `0.0010`, so declare `xticks`/`yticks` explicitly on
  every panel the original ticks at round values. Budget one render per fix and measure the
  aspect each time -- with `trim_margins: false` it does not move, which is the point.
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
- **A bespoke panel that builds its OWN sub-grid uses `fig.subplot_mosaic` + the compressed
  engine — never `add_gridspec` + `canvas.draw()` + `get_position()` + `fig.add_axes`/`fig.text`
  at hand-computed figure coordinates.** A custom panel takes over the whole figure and its
  single `ax` is unused, so drop it first (`for a in list(fig.axes): a.remove()`), then lay the
  whole panel out as ONE mosaic — heatmap cells, a per-row colorbar cell, and a thin spanning
  header row for group titles — with `width_ratios`/`height_ratios` and `empty_sentinel="."`
  gutters; let `layout: compressed` pack it. Put the shared axis titles on `fig.supxlabel` /
  `fig.supylabel`, not `fig.text`. The manual-coordinate approach (`canvas.draw()` to read boxes,
  then `add_axes`/`text` off `get_position()`) is fragile, breaks under a resize or a different
  DPI, and is what makes a grid figure look "off". **And do NOT rely on the base style moving the
  spines**: if the panel wants a clean data-box (a heatmap), reset each axes' spines explicitly
  (`s.set_visible(True); s.set_position(("outward", 0))`) so an offset-spine base style can't
  detach them from the axes.

## Identify each panel's QUANTITY from its axis range, not its title

**A panel's title names a column; its axis range names the quantity.** When the two disagree,
the range wins, and the disagreement is common enough to expect: Kadak's per-connection panels
are titled `coupling.xx.nu_post` in three different figures and plot three different things —
the absolute post-stimulation weight in one (its y range brackets the initial weight), the
signed change in another (±2.5e-4 where the weight itself is 1.2e-3 and never crosses zero),
and the unsigned relative magnitude in the radar. Nothing in the titles distinguishes them.

So before binding anything, **read the published panel's tick labels and derive which candidate
they can come from**, then bind and check that your rendered range lands on theirs. That check
is cheap and decisive: ten connections agreeing to within a few percent on a quantity you chose
by elimination is proof; one connection agreeing is a coincidence. Register the mismatch (class
E, convention trap) — the paper's own published data usually explains it, here a differenced frame that
kept the column names of the frame it was differenced from.

**Then bind the SAME analysis three times rather than sharing one.** Two panels wanting two
quantities from one analysis is how a rebind silently changes the other; give each its own
`sc_*` / `hm_*` analysis named for what it holds.

## A unit error hides inside a monotone quantity

`pulse_rate` was `ppb x f_ibf` where the paper's axis is pulses per 2 s TRAIN, so every axis in
five figures was wrong by exactly 2x — and **not one correlation, p-value or scorecard verdict
moved**, because a monotone rescaling leaves Pearson and Spearman untouched. Nothing in the
verification caught it; the figure's marked lines did, sitting at 20/22/30 on an axis whose data
stopped at 38.

Two habits catch this class:

- **Check a landmark the paper prints in that unit.** Canonical iTBS is "30 pulses / train"; our
  canonical cell read 15. One arithmetic check on one protocol.
- **Put the derived quantity in the published-data oracle comparison** (Phase 7), not just the outcome
  measures. `pulse_rate -> train_dose` now returns r = 1.000 over all 432 protocols, which is
  what a unit match looks like; anything else is a scale error, and a *correlation* of 1.0 with
  a *range* that disagrees is exactly the signature.

## Mark the paper's own named protocols

When a figure marks specific conditions, take them from the recipe's own declared set (a
`MARKED` mapping in `code/<study>_protocols.py`), never from a quantile of your own ranking.
Quantiles look reasonable and are a different claim: the paper marks *these* protocols because
of what they are, and the panels that repeat the marking (spectra, radars, profiles) have to
mark the same ones or the figure stops being one figure. Mark exactly the set the paper marks
per figure, too — an extra marker on the plane is a visible difference, so narrow the set with
a `marks:` opt where a figure uses fewer.

## Mosaic traps, all of them found by rendering

- **Every row of `layout:` must have the same number of columns.** Widen short rows by repeating
  letters, never by adding columns.
- **A panel may not span across an all-dot spacer row** — matplotlib reports "the label 'k'
  specifies a non-rectangular or non-contiguous area". Keep the letter in the spacer row.
- **A spacer row has to be big enough for the labels that live in it.** The x-label of the row
  above and the panel letter of the row below both land there; 0.3 of a row is usually too
  little and 0.6 is usually right. The symptom is a label crossing into the panel beneath.
- **A label wider than its column is clipped and pushed sideways into the neighbour.** Either
  break it over two lines (`"Inter-Burst\nFrequency (Hz)"` — and give the row above the height
  for a second line) or widen the column.
- **Per-cell colourbars inside a `grid` panel float** — matplotlib positions them against the
  host axes, so they land over neighbouring cells. A row of independently-scaled heatmaps is
  ten ordinary mosaic panels, not one grid.
- **Tick labels of adjacent panels collide in the gutter.** `ytick_side: right` puts a cell's
  scale on its own outer edge (which is how most published grids print them); otherwise leave a
  spacer column, since a gutter narrower than the label just moves the collision.
- **Editing a mosaic with `sed` will hit an identical row in another figure.** Edit by figure
  block or by line index, and re-render every figure afterwards.
- **The column count of `layout:` is yours to choose — widen the grid rather than rob a gutter.**
  When one block needs more width, the reflex is to take a spacer column from somewhere else, and
  the space you took was holding a colourbar's tick labels or a neighbour's axis title apart. Go
  from 20 columns to 23 and give every block its share; the letters are relative widths, not a
  fixed budget.

## A `grid` panel's spacing is subtracted from its cells

Every fraction in a `grid` is of the HOST PANEL, and this is where a block of small multiples
goes wrong:

- **A cell is `cw - wspace` wide.** `wspace` does not sit *between* cells, it is taken *out of*
  each one, so a value approaching `cw` collapses the cells instead of separating them. A
  `wspace: 0.34` on a two-column grid (`cw ≈ 0.49`) left cells a third of their width and read,
  at a glance, as "the block is too narrow".
- **The last column already has a trailing gap** — one `wspace` sits to its right — so a `right:`
  strip must hold only what that gap does not. Reserving the tick labels in BOTH is the same
  space counted twice, and it is why cells came out 13 mm wide where the budget said 19.
- **`xlabel`/`ylabel` on the grid name what the cells share, once, at the OUTER edge of the
  reserved strip.** Anchor them just outside the cells instead and they are drawn straight over
  the tick labels the same strip holds.
- **A dense grid cannot carry the figure's own tick geometry.** `tick_size` and `tick_length` are
  per-axes for exactly this: the house tick protruded 4.7 mm into a 10 mm gutter, so the labels
  it was making room for were clipped by the next cell's patch.

## Colour, scale and geometry

- **A colourbar that factors out a shared multiplier is a silently wrong axis.** A field
  spanning 3e4 prints "3, 1, 0" and one spanning 1e-4 prints "3, 0, -1", because a slim bar has
  nowhere to put the exponent. tvbo now writes every colourbar tick in full and takes
  `colorbar_decimals` where a paper prints a specific precision — but **read the bar's numbers
  against the layer's own min/max** before believing a figure.
- **A diverging field needs its neutral colour pinned, not its limits symmetrised.** `center: 0`
  in a heatmap layer's `opts` keeps the data's own limits and truncates the map to the half-range
  the data reaches, so a unit of change is the same colour distance either side of zero and the
  bar shows no colour the field never takes. (That is seaborn's `center=`, which is what most
  published repositories produce.) Symmetrising the limits instead invents headroom the data never uses.
- **A colour convention the plotting stack does not ship is a NAME, registered in the study's
  own figure module.** Read the two hues off the published colourbar, register a
  `LinearSegmentedColormap` at module import, and name it from the spec like any other map —
  reading a colour off a published figure is a style fact, not data.
- **Never compute geometry from `ax.get_position()` at draw time.** The layout pass has not run,
  so the box you read is not the box you get: a radar that corrected its aspect by a
  hand-computed `width/height` ratio drew its spoke labels inside the web. Use
  `ax.set_aspect("equal")` and let the layout engine solve it. Anything that genuinely must run
  after the tidy-up (an inset's declared frame, a colourbar's declared ticks) belongs in the
  template's post-format pass, not in the panel.
- **A declared encoding the renderer silently drops is worse than an error.** `color:` on a
  scatter means a third quantity per point; if the renderer treats `color` as a per-artist
  fan it drops the encoding and the panel looks fine. Compare against the original: a shaded
  cloud that came out one colour is the tell.
- **Calibrate a panel's axis LIMITS off the paper's own reference marks, not off your data.** Two
  marks whose values you know — the alpha lines at 5.02 and 10.05 Hz — give a linear pixel→data
  map for that panel, and everything else in it can then be read in data units. That measurement
  said the published frame ran 0–20 Hz (the protocol space) where ours auto-scaled to the
  responsive subset's 1–15.5, and it placed the paper's own marker glyph at x ≈ 0 — pinned to the
  axis start, carrying only its height, not drawn at the protocol's 5 Hz. Both were invisible to
  three rounds of eyeballing the two images side by side.
- **Do not shrink type to make a block fit until you have measured the original's.** The paper's
  E-block ticks run ~0.58 mm per character against our 1.16 at body size; "the labels don't fit"
  was a statement about type size, not about the block's width. Measure mm-per-character in both,
  then decide what to spend — and record the size you chose as a deviation, because dropping to
  the paper's own would be illegible in our layout.
- **Two tick labels printing the same number are not a scale — but check clipping first.** The
  same symptom has two causes: a formatter rounding 0.0011300 and 0.0011325 to one string, or a
  neighbour's opaque patch cutting the last two characters off both. tvbo widens the decimals
  automatically for the first, over the DRAWN labels; only the second is yours to fix, with space.

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
