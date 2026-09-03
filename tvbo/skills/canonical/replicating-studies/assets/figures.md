# Phase 5 reference — sizing, binding and drawing a panel

Read this while writing a study's `figures:` block. The spine holds the rules a figure must obey (integrity, captions, coordinate convention); this file holds the mechanics — the layout keys, how to measure the original's size and type scale, how a panel binds its data, and when a panel is allowed to be code.

## Before any of it: bind the original and stage the A/B

Do this first, before the mosaic string, before a single panel drawer. It costs one line of spec and one command, and it is the only instrument that catches a figure which is internally consistent, passes every provenance check, and is still not the figure the paper printed.

```yaml
name: Fig2_herzog2024
reference_image: sourcedata/original_study/img/fig_03.png   # the published scan this reproduces
```

Then measure both sides and make the aspect match before drawing anything:

```python
from PIL import Image

w, h = Image.open("sourcedata/original_study/img/fig_03.png").size
print(w, h, h / w)  # 1611 889 0.5518  -> a single landscape row, not a portrait
```

`height = width * (h/w)`, with `trim_margins: false`, and the mosaic laid out the way the paper lays it out. Herzog2024 declared `layout: "ab/cc"` at aspect 1.10 against an original at 0.55 and carried the wrong shape through every review, because nothing ever put the two side by side.

Stage the composites from a `code/compare_figures.py` the study runs itself, not only from the report's internal build:

```python
os.environ.setdefault("QUARTO_DOCUMENT_FILE", "report_internal.qmd")  # the internal-build marker report_figure checks
for figure in study.figures:
    report_figure(rendered / f"{figure.name}.png", reference_image_for(figure, ROOT), credit="<Authors> <year> (c)")
```

Composites land in the layout's `figures_restricted` role — `sourcedata/original_study/fig_comparisons/` — which the `sourcedata/*` ignore rule already covers, so the publisher's material never enters the repository. Run it after every `tvbo figure render` and look at the result.

**Guard the binding, because its failure is silent.** `reference_image_for` falls back to a file named after our figure at the study root; no replication has one, so an undeclared binding returns `None` and every consumer reads that as "no original to show" rather than "the binding is missing". Put it in the standing harness:

```python
unbound = [f.name for f in study.figures if (p := reference_image_for(f, ROOT)) is None or not Path(p).is_file()]
rep.check("every figure names the published original it reproduces", "identity", not unbound, ...)
```

Assert the binding, not the composite — the harness also runs inside the public build, where opening the publisher's material is exactly what must not happen. Composing stays in the command a person runs.

**Never cut the original, and always show the A/B.** `reference_image:` names the whole published figure as printed. Cropping it to the panel you happen to reproduce makes a derived work of the publisher's material and, worse, removes the only signal the comparison carries: a six-panel original beside one filled panel tells you the figure needs five more panels, and a crop tells you nothing at all. Measuring a panel's extent is fine and is how you size a partial reproduction — read the pixel coordinates, never write a cropped file.

**Mirror the original's mosaic and let the empty slots stand.** Give every slot you cannot fill a `placeholder:` naming the input that blocks it:

```yaml
  b:
    kind: custom
    placeholder: "K-S over (G, alpha), AAL90\nno empirical FCD (F5)"
```

A placeholder is not a defect to drive to zero — it states the extent of the replication in the shape of the figure, where a table of blocked targets states it somewhere a reader has to go looking. The forbidden case is a *silent* slot: empty with no stated obstacle, which nobody can tell from an oversight. Check that every placeholder carries text and every panel either binds data or says why it does not.

**Check the type size on the first render, in points read off the drawn figure.** The declaration does not enforce itself: a study `.mplstyle` can override `font_size`, a drawer passing `fontsize="small"` steps outside it, and legends and insets carry their own defaults.

```python
drawn = [t.get_fontsize() for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
modal = collections.Counter(round(x, 2) for x in drawn).most_common(1)[0][0]
assert abs(modal - declared_font_size) <= 0.5  # nothing silently overrode the declaration
assert min(drawn) >= 6.0  # nothing is below what a reader can follow in print
```

Two failures with two different fixes: a modal size that is not the declared one means an override you did not write, and a minimum below the floor means a crowding problem a smaller type size was hiding — spend a label budget, not a font step.

**Captions are the paper's, not ours.** Write each `description:` from the published caption for that display item — its claims, panel lettering, units and stated ranges — and record deviations inside it ("on the released 100-region consensus connectome rather than the paper's AAL90", "swept to 4.5 where the paper plots to 3"). A caption drafted from our own figure drifts into describing what we happened to draw, which is how a panel comes to answer a question the paper never asked while still reading as a reproduction of it.

**A `Figure` is layout + binding + style; keep compute and plotting code out of it.**
- **Layout is metadata:** `layout` (bsplot mosaic string, e.g. `aab/ccb` — letters = panel keys, `/` = new row, repeated letters span, `.` = empty), `width`/`height` (mm), `dpi`, `font_size`, `height_ratios`/`width_ratios`, `style` (`.mplstyle` paths), `spines`, `panel_numbers`/`panel_number_format`/`panel_number_loc`. Set the paper's physical size and type scale here, once — never in code.

**Size, aspect and type size are MEASURED off the original, not guessed — get them right on the first render.** Three defaults are wrong for a replication and cost a re-render every time:

- **Aspect.** Derive `height` from the original's own pixel aspect: `height = width × (h_px / w_px)` of the paper's figure scan (a Nature two-column figure is `width: 183`, a single-column one `88`). Then set **`trim_margins: false`** — the default trims to content (`bbox_inches="tight"`), which re-crops the saved PNG and makes its aspect drift away from the declared `width × height`, which is exactly the "why is my figure the wrong shape" symptom. Check it: the rendered PNG's `h/w` must equal `height/width` to ~1 %. **But `trim_margins: false` also exposes a layout failure that trimming used to hide.** A dense mosaic (many panels, a very short row, empty `.` cells) can starve matplotlib's constrained-layout solver — it warns `axes sizes collapsed to zero` and silently drops EVERY axes back to the default 12 % subplot margins, giving a figure ringed by ~8 % white that neither `set_layout_engine("none")` nor `subplots_adjust` nor `height_ratios` will shift. Diagnose by measuring the ink bounding box of the PNG; if padding is ~8 % on all four sides while a sibling figure is at ~1 %, that is this bug and not your margins. The fix is to let it trim and declare the size OVERSIZE so the trimmed result lands on the target (iterate twice — measure, rescale, re-render), and **re-measure after any type-size change**, because the trim moves with it.
- **Correcting a too-tall figure BREAKS its labels, and the fix is a label budget, not a taller figure.** A figure whose declared height is 30 % over the original has been hiding a crowding problem: cut it to the measured aspect and the axis labels, colourbar labels and panel numbers start writing over their neighbours. Do not put the height back. Spend the space instead -- shorten a y label ("Circuit-Mean Synaptic Weight Change (Δ)" to "Weight Change (Δ)"), drop a colourbar label the caption already states, widen the mosaic gap column between a colourbar and its neighbour's tick labels, and pull `panel_number_offset` in. The tell that you are looking at this and not at bad data is six-decimal tick labels: a narrow panel with an automatic locator prints `0.00102747` where the paper prints `0.0010`, so declare `xticks`/`yticks` explicitly on every panel the original ticks at round values. Budget one render per fix and measure the aspect each time -- with `trim_margins: false` it does not move, which is the point.
- **Panel proportions.** A mosaic alone distributes rows/columns EQUALLY. If the paper's rows or columns are unequal (a short schematic row above tall matrix panels), measure their pixel extents in the original and set `height_ratios` / `width_ratios` — otherwise every panel is subtly the wrong shape even though the figure size is right.
- **Type size — the single most repeated defect in this skill's history. VERIFY IT IN PIXELS, every time, before you call a figure done.** Set **`font_size: 9`** for a 183 mm figure and **10** for an 88 mm one as the *starting* value, never below 8. Then MEASURE, because two independent traps make the declared number a lie:

  1. **A study `.mplstyle` silently overrides `font_size`.** If the study ships a style file that sets `font.size`, it used to be applied *after* the declared size and won — so `font_size: 8` in the spec rendered as whatever the style said (Pang2023: 5 pt) with nothing in the spec, the log, or the emitted script to show it. tvbo now applies the `.mplstyle` FIRST and the declared `font_size` last (regression-tested), but **any style file you write must still be checked**: grep it for `font.size`, `axes.labelsize`, `xtick.labelsize`.
  2. **A point size is meaningless without the width it was measured at.** Apparent size is the ratio of glyph height to figure WIDTH. Pixel forensics on a 120 mm (1.5-column) original that you then reproduce at 183 mm yields type ~1.5× too small — and the same error scales every `linewidth`, `markersize` and tick length in the file. If a `.mplstyle` is derived from measurements, record the width they were taken at and rescale by `target_mm / measured_mm`.

  **The check (do it, don't assume):** binarise the rendered PNG and the paper's own scan, take connected components with `5 <= h <= 40 px`, and compare the modal glyph height as a PERCENTAGE OF IMAGE WIDTH. That ratio is resolution-independent, so it compares a 953 px scan with a 2161 px render directly. Journals run ~0.7–0.85 %; land within that or slightly above. **Aim a little above the original** — a Nature figure's 6 pt labels are legible at 183 mm in print and illegible on screen at 2000 px, and every replication we have shipped erred small, three of them after this rule was already written down.
- **Grammar panels need zero code.** A `cartesian` or `heatmap` panel binds data through its `layers`: `used: {iri: tvbo:exp/<Study>/exp-3, output: <var|observation__name>, sel: {dim: label}}` (label-keyed, never positional — this binding **is** the PROV `used` edge), plus `mark` (`line`/`scatter`/`rule`/`band`/`area`/`bar`; implied for heatmap) and `encoding: {x, y, color}` naming container dims/coords. **`band` draws a spread** (`fill_between`) and its output must carry a length-2 axis beside the swept one — the analysis returns `mean ± sd` as ONE `(n, 2)` array with a `bound: [lo, hi]` coordinate, so a figure cannot bind a lower edge from one run and an upper edge from another. Draw the band layer BEFORE its mean line, or the fill covers the curve it belongs to. **`rule` draws a reference line at a value the CONTAINER holds** — an ensemble mean, a published number the recipe declared as an analysis argument and the analysis echoed back — with the encoded channel picking the orientation (`x:` vertical). Prefer it to a `rules:` entry, which takes a literal typed into the spec and renders as a subdued gridline: a marker the figure exists to make is worth a styleable layer and a PROV edge. `transform:` names an optional presentation-only reduction. Bind an **in-study** experiment by id — `used: {experiment: 3}` — rather than spelling a full `iri`: it needs no hardcoded study key and registers the run-order dependency (that experiment runs before the figure). Reserve an explicit `iri` for a curated/external container.
- **Every axis directive, and every built-in kind's options, are declared slots — `opts:` is a custom callable's keywords.** Labels, limits, scales, shape, ticks, `legend`, `rules`, `regions` and `camera` sit on the panel; a built-in kind's own options are the object named after it (`surface:`, `volume:`, `network:`, `grid:`, `colorbar:`). A retired spelling is refused by name, per kind — `color`/`cmap`/`labels` stay good keywords for a `custom` panel. `python scripts/migrate_panel_marks.py <study>` converts a tree.
- **Only a bespoke interior is code.** A `custom` panel sets `render: <fn>` + `opts:`, where `<fn>` is a `@bsplot.register_panel` callable `fn(fig, ax, ctx)` in a module named in the figure's `code_modules:` (a flat file in `code/`, e.g. `code/<study>_figures.py`). It reads its resolved layers with `bsplot.load_layer(ctx["layers"][i])` and draws. A reused reduction is a `@bsplot.register_transform` `fn(da)->da`. This is the escape hatch — reach for it only when the grammar genuinely can't express the panel (twin axes, connectome, brain surface, dense nested subgrids), not by default.
- **A bespoke panel that builds its OWN sub-grid uses `fig.subplot_mosaic` + the compressed engine — never `add_gridspec` + `canvas.draw()` + `get_position()` + `fig.add_axes`/`fig.text` at hand-computed figure coordinates.** A custom panel takes over the whole figure and its single `ax` is unused, so drop it first (`for a in list(fig.axes): a.remove()`), then lay the whole panel out as ONE mosaic — heatmap cells, a per-row colorbar cell, and a thin spanning header row for group titles — with `width_ratios`/`height_ratios` and `empty_sentinel="."` gutters; let `layout: compressed` pack it. Put the shared axis titles on `fig.supxlabel` / `fig.supylabel`, not `fig.text`. The manual-coordinate approach (`canvas.draw()` to read boxes, then `add_axes`/`text` off `get_position()`) is fragile, breaks under a resize or a different DPI, and is what makes a grid figure look "off". **And do NOT rely on the base style moving the spines**: if the panel wants a clean data-box (a heatmap), reset each axes' spines explicitly (`s.set_visible(True); s.set_position(("outward", 0))`) so an offset-spine base style can't detach them from the axes.

## Identify each panel's QUANTITY from its axis range, not its title

**A panel's title names a column; its axis range names the quantity.** When the two disagree, the range wins, and the disagreement is common enough to expect: Kadak's per-connection panels are titled `coupling.xx.nu_post` in three different figures and plot three different things — the absolute post-stimulation weight in one (its y range brackets the initial weight), the signed change in another (±2.5e-4 where the weight itself is 1.2e-3 and never crosses zero), and the unsigned relative magnitude in the radar. Nothing in the titles distinguishes them.

So before binding anything, **read the published panel's tick labels and derive which candidate they can come from**, then bind and check that your rendered range lands on theirs. That check is cheap and decisive: ten connections agreeing to within a few percent on a quantity you chose by elimination is proof; one connection agreeing is a coincidence. Register the mismatch (class E, convention trap) — the paper's own published data usually explains it, here a differenced frame that kept the column names of the frame it was differenced from.

**Then bind the SAME analysis three times rather than sharing one.** Two panels wanting two quantities from one analysis is how a rebind silently changes the other; give each its own `sc_*` / `hm_*` analysis named for what it holds.

## A unit error hides inside a monotone quantity

`pulse_rate` was `ppb x f_ibf` where the paper's axis is pulses per 2 s TRAIN, so every axis in five figures was wrong by exactly 2x — and **not one correlation, p-value or scorecard verdict moved**, because a monotone rescaling leaves Pearson and Spearman untouched. Nothing in the verification caught it; the figure's marked lines did, sitting at 20/22/30 on an axis whose data stopped at 38.

Two habits catch this class:

- **Check a landmark the paper prints in that unit.** Canonical iTBS is "30 pulses / train"; our canonical cell read 15. One arithmetic check on one protocol.
- **Put the derived quantity in the published-data oracle comparison** (Phase 7), not just the outcome measures. `pulse_rate -> train_dose` now returns r = 1.000 over all 432 protocols, which is what a unit match looks like; anything else is a scale error, and a *correlation* of 1.0 with a *range* that disagrees is exactly the signature.

## Mark the paper's own named protocols

When a figure marks specific conditions, take them from the recipe's own declared set (a `MARKED` mapping in `code/<study>_protocols.py`), never from a quantile of your own ranking. Quantiles look reasonable and are a different claim: the paper marks *these* protocols because of what they are, and the panels that repeat the marking (spectra, radars, profiles) have to mark the same ones or the figure stops being one figure. Mark exactly the set the paper marks per figure, too — an extra marker on the plane is a visible difference, so narrow the set with a `marks:` opt where a figure uses fewer.

## Mosaic traps, all of them found by rendering

- **The mosaic is the layout's picture — let it carry the proportions, not a ratio list.** A panel twice as tall as its neighbour is two rows against one; `height_ratios` beside the mosaic is a second, invisible description of the same fact, and the two drift. An all-equal ratio list (`[1.0]` x 48) says nothing at all and should be deleted outright. Keep a ratio only for what the grid genuinely cannot say at a sane row count — a hairline spacer of 0.02 of a row — and check first whether the spacer is needed at all, since the layout engine already spaces adjacent panels.
- **Never pad the mosaic with an empty leading row or column.** It reads as a margin and produces nothing: `trim_margins: true` crops that whitespace away, so the cells are simply spent. Gaps BETWEEN panels are real (they hold the labels); gaps at the edge are not.
- **Every row of `layout:` must have the same number of columns.** Widen short rows by repeating letters, never by adding columns.
- **A panel may not span across an all-dot spacer row** — matplotlib reports "the label 'k' specifies a non-rectangular or non-contiguous area". Keep the letter in the spacer row.
- **A spacer row has to be big enough for the labels that live in it.** The x-label of the row above and the panel letter of the row below both land there; 0.3 of a row is usually too little and 0.6 is usually right. The symptom is a label crossing into the panel beneath.
- **A label wider than its column is clipped and pushed sideways into the neighbour.** Either break it over two lines (`"Inter-Burst\nFrequency (Hz)"` — and give the row above the height for a second line) or widen the column.
- **Per-cell colourbars inside a `grid` panel float** — matplotlib positions them against the host axes, so they land over neighbouring cells. A row of independently-scaled heatmaps is ten ordinary mosaic panels, not one grid.
- **Tick labels of adjacent panels collide in the gutter.** `ytick_side: right` puts a cell's scale on its own outer edge (which is how most published grids print them); otherwise leave a spacer column, since a gutter narrower than the label just moves the collision.
- **Editing a mosaic with `sed` will hit an identical row in another figure.** Edit by figure block or by line index, and re-render every figure afterwards.
- **The column count of `layout:` is yours to choose — widen the grid rather than rob a gutter.** When one block needs more width, the reflex is to take a spacer column from somewhere else, and the space you took was holding a colourbar's tick labels or a neighbour's axis title apart. Go from 20 columns to 23 and give every block its share; the letters are relative widths, not a fixed budget.
- **Widening a gutter shrinks the panels, and past a point that costs more than it buys.** The layout is constrained, so it re-packs rather than honouring empty cells: an extra spacer row is partly reclaimed, while the panels either side genuinely lose the width you moved. Kadak2025's densest supplementary figure went from **10 overlapping text pairs to 40** on one such widening — the panels lost enough width that their own titles reached their neighbours' ticks, a collision class that had not existed before. Widen by ONE unit and re-measure; a jump of three is how you discover this.
- **When geometry stops paying, type size is the lever that scales with the problem.** Every collision in a dense figure is text against text, so a size cut shortens every offender at once, including the rotated axis titles whose overlap is along their length. The same figure went 10 → 4 on `font_size: 5.5 → 4.4` alone, and to 0 with one gutter column and a panel-letter offset on top. Check the original's own type size first: a dense four-quadrant supplementary figure is usually set smaller than the paper's main figures, so this is often a move TOWARD fidelity rather than away from it — but verify in the rendered PDF at final scale, since the page shrinks it again.
- **Ask the reference image before inventing a fix.** Two of four collision sets in Kadak2025 were the original telling us we had the layout wrong: its three stacked spectra are SEPARATED, and ours were flush, so at every boundary the lower panel's top tick printed on top of the upper panel's bottom tick; and its two calcium panels are EQUAL width, where ours were 2 and 3 mosaic columns and the narrow one ran its own x-labels together. Matching the paper fixed the text and the fidelity in the same edit. Look before you tune.

## A `grid` panel's spacing is subtracted from its cells

Every fraction in a `grid` is of the HOST PANEL, and this is where a block of small multiples goes wrong:

- **A cell is `cw - wspace` wide.** `wspace` does not sit *between* cells, it is taken *out of* each one, so a value approaching `cw` collapses the cells instead of separating them. A `wspace: 0.34` on a two-column grid (`cw ≈ 0.49`) left cells a third of their width and read, at a glance, as "the block is too narrow".
- **The last column already has a trailing gap** — one `wspace` sits to its right — so a `right:` strip must hold only what that gap does not. Reserving the tick labels in BOTH is the same space counted twice, and it is why cells came out 13 mm wide where the budget said 19.
- **`xlabel`/`ylabel` on the grid name what the cells share, once, at the OUTER edge of the reserved strip.** Anchor them just outside the cells instead and they are drawn straight over the tick labels the same strip holds.
- **A dense grid cannot carry the figure's own tick geometry.** `tick_size` and `tick_length` are per-axes for exactly this: the house tick protruded 4.7 mm into a 10 mm gutter, so the labels it was making room for were clipped by the next cell's patch.

## Colour, scale and geometry

- **A colourbar that factors out a shared multiplier is a silently wrong axis.** A field spanning 3e4 prints "3, 1, 0" and one spanning 1e-4 prints "3, 0, -1", because a slim bar has nowhere to put the exponent. tvbo now writes every colourbar tick in full and takes `colorbar_decimals` where a paper prints a specific precision — but **read the bar's numbers against the layer's own min/max** before believing a figure.
- **A diverging field needs its neutral colour pinned, not its limits symmetrised.** `center: 0` in a heatmap layer's `opts` keeps the data's own limits and truncates the map to the half-range the data reaches, so a unit of change is the same colour distance either side of zero and the bar shows no colour the field never takes. (That is seaborn's `center=`, which is what most published repositories produce.) Symmetrising the limits instead invents headroom the data never uses.
- **A colour convention the plotting stack does not ship is a NAME, registered in the study's own figure module.** Read the two hues off the published colourbar, register a `LinearSegmentedColormap` at module import, and name it from the spec like any other map — reading a colour off a published figure is a style fact, not data.
- **Never compute geometry from `ax.get_position()` at draw time.** The layout pass has not run, so the box you read is not the box you get: a radar that corrected its aspect by a hand-computed `width/height` ratio drew its spoke labels inside the web. Use `ax.set_aspect("equal")` and let the layout engine solve it. Anything that genuinely must run after the tidy-up (an inset's declared frame, a colourbar's declared ticks) belongs in the template's post-format pass, not in the panel.
- **A declared encoding the renderer silently drops is worse than an error.** `color:` on a scatter means a third quantity per point; if the renderer treats `color` as a per-artist fan it drops the encoding and the panel looks fine. Compare against the original: a shaded cloud that came out one colour is the tell.
- **Calibrate a panel's axis LIMITS off the paper's own reference marks, not off your data.** Two marks whose values you know — the alpha lines at 5.02 and 10.05 Hz — give a linear pixel→data map for that panel, and everything else in it can then be read in data units. That measurement said the published frame ran 0–20 Hz (the protocol space) where ours auto-scaled to the responsive subset's 1–15.5, and it placed the paper's own marker glyph at x ≈ 0 — pinned to the axis start, carrying only its height, not drawn at the protocol's 5 Hz. Both were invisible to three rounds of eyeballing the two images side by side.
- **Do not shrink type to make a block fit until you have measured the original's.** The paper's E-block ticks run ~0.58 mm per character against our 1.16 at body size; "the labels don't fit" was a statement about type size, not about the block's width. Measure mm-per-character in both, then decide what to spend — and record the size you chose as a deviation, because dropping to the paper's own would be illegible in our layout.
- **Two tick labels printing the same number are not a scale — but check clipping first.** The same symptom has two causes: a formatter rounding 0.0011300 and 0.0011325 to one string, or a neighbour's opaque patch cutting the last two characters off both. tvbo widens the decimals automatically for the first, over the DRAWN labels; only the second is yours to fix, with space.

- **Hiding a panel's tick labels does not put it on another panel's scale.** A row of panels showing one quantity against different predictors is drawn once per panel, so each auto-scales to its own layer's extent — including the confidence bands, which is where they diverge. Hiding the tick labels on all but the leftmost then invites the reader to compare heights across three different ranges. Declare the group instead (`share_y: ["c,g,h"]` at figure level): every panel in it ends on the union of the group's limits, hidden labels included. Do not reach for a literal `ylim` — a spec that pins a limit to today's numbers clips tomorrow's run in silence.
- **A slot a panel blanks is un-blanked by the format pass.** Colour-scale, legend, `grid` and `image` panels switch their host axes off and draw inside it; the figure-wide tidy-up then re-derives ticks for every axes and hands the ghost frame back, with the panel's declared tick options applied to it rather than to the bar. tvbo re-blanks after the format pass and lets a scale panel return the axes its bar lives on, so `nbins`, tick formats and label padding land on the scale. Two consequences worth knowing: `Axes3D` reports itself as blanked by construction, so it is excluded by name; and an overlap check that walks every `Text` will count the ghost's labels until it skips switched-off axes.
- **An overlap checker's coordinates are the CANVAS's; the saved file is a crop of it.** `trim_margins: true` saves with `bbox_inches="tight"`, so the PNG starts at the tight box's origin while `get_window_extent(renderer)` keeps reporting canvas pixels. The overlap *counts* survive this — detection compares two extents to each other and the offset cancels — but every coordinate you print is displaced, so cropping a reported box shows blank paper. That looks exactly like a false positive, and it cost a session's confidence in a detector that was right all along: the "invisible" text turned out to be two glyphs printed squarely on top of each other, 180 px away. Convert before reporting a position:

  ```python
  tb = fig.get_tightbbox(renderer)  # inches, y-up, canvas origin
  x0, y1 = tb.x0 * fig.dpi, tb.y1 * fig.dpi
  png_xy = lambda x, y: (round(x - x0), round(y1 - y))
  ```
- **Report each colliding text's OWN box, and key them by identity, not by their string.** A figure has many texts reading `2` or `20`, so a `{label: box}` dict silently keeps the last one and prints a box belonging to a different glyph than the one that collided. Carry the pair's two extents through with the hit.

## Choosing WHICH point a marker marks

**A marked/sampled point must match what the paper says it IS — read the figure description, then verify via the panel it feeds.** When a `custom` panel marks sample points on a curve (three periodic orbits 1/2/3 on a period-vs-parameter branch), select each from the paper's figure caption/description, not a guessed heuristic: the description names what the point is — its period band **and its morphology** ("point 2 = the *mid* orbit, an asymmetric spike with a slow rise") — and that fixes which branch point to take. Then verify against the panel the marker drives: the marked orbit's waveform sub-panel must show the described shape. (An argmin-on-period heuristic put our "2" at the bottom-corner fold — a too-symmetric spike; the description's "asymmetric spike + slow rise" is what identified the elbow one bend up as the right orbit.) This is Phase 7's shape-check applied to marker placement — the caption/description is the oracle for *which* point, not just how to word it.

## Binding the paper's own published data

**External published paper data binds by IRI too.** When a panel pairs TVBO output against the paper's own figure data, wrap that data as an external `Dataset` and bind `used: {iri: tvbo:dataset/<Study>_source, output: <var>, sel: {figure: 6, panel: c}}` — the same declarative path, figure/panel as coordinates you `sel` into. Until wrapped, a **flat, label-keyed** per-panel `.nc` set (`xarray` named coords, not filesystem-keyed) is an accepted stopgap; don't build an elaborate filename tree — it's throwaway once the `Dataset` binding lands.

## Measuring type size: use the quartiles, and check whether the original is even legible

Two refinements to the pixel check above, both learned on Deco2018's Figure 3.

**The modal glyph height is unstable and the lower quartile is not.** A figure carries several type sizes at once -- tick labels, axis labels, panel letters, an annotation -- and each forms its own cluster in the height histogram. The mode is whichever cluster happens to be most populous, so a one-point change in `font_size` can move it from the tick cluster to the axis-label cluster and report a jump from 0.694 to 1.435 per cent when nothing has moved by more than 11 per cent. Report the **lower quartile** (which tracks the tick labels, the most numerous population in a filled figure) and the **median** together, and compare like against like. Print the whole histogram once when a number surprises you; two well-separated clusters are the tell.

**Measure the original's placed physical size before trusting its percentage.** A published figure's pixel width says nothing about how wide it is on the page, and the ratio you are matching is glyph height over figure width. `pdfimages -list <paper>.pdf` prints each embedded image's pixel size **and its effective dpi**, so the placed width is `px / dpi` inches -- one command, no guessing whether a figure is one column, 1.5 columns or two. Deco2018's Figure 3 is 987 x 1710 px at 300 dpi, so 83.6 mm wide: a single-column figure, and the standard 88 mm is the right declaration.

**When the original's own type is below the journal floor, declare the deviation instead of matching it.** That same figure's tick labels are 9 px at 300 dpi, an ink height of 0.76 mm, which is a body size near 3 pt -- under any journal's stated minimum and unreadable on screen at any zoom. Matching it would mean rendering illegible text in order to reproduce a defect. Set `font_size` to the floor this skill states, land at roughly twice the original's percentage, and write the measurement and the reason into `figures.md` so the number is not mistaken for carelessness later. Assert the **direction** in the harness -- our type is at least the original's -- rather than a ratio that would fail the moment a panel fills.

**A placeholder figure measures its placeholders, not its future ticks.** A figure whose panels are all held has no tick labels at all, so the small end of the glyph distribution is simply absent and every statistic reads high. Say so where the measurement is recorded, and re-measure when a panel fills; a standing harness check does this without anyone having to remember.

## Two Panel-schema names that are not what you would guess

**The per-panel letter override is `number:`, not `panel_number:`.** The figure-level settings are all prefixed (`panel_numbers`, `panel_number_format`, `panel_number_loc`, `panel_number_offset`, `panel_number_size`); the per-panel overrides drop the prefix, so a Panel takes `number:` and `number_loc:`. `panel_number:` on a panel raises `TypeError: Panel.__init__() got an unexpected keyword argument`, which names the class but not the correct spelling. Use `number: "A"` wherever the paper letters its panels in upper case and the mosaic keys are lower.

**A panel's `label:` is drawn as its title.** It is not an internal name. A descriptive `label: "Placebo fit against global coupling"` on a half-width panel prints straight over its own panel letter and into its neighbour's, which reads as a layout bug and is a naming one. Keep `label:` to what fits above the panel -- the paper's own titles are usually two or three words, and often absent -- and put the description in the `placeholder:` text or the figure's `description:`.
