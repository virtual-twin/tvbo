# Handoff — Align tvbo with LEMS+NeuroML by ingesting NeuroML-core into the tvbo ontology

**Status:** design assessed + feasibility verified; implementation not started. Intended for a
fresh, focused session (the originating session was long and Deco-replication-heavy).

---

## 1. Goal & principle

**Principle (from the user):** *tvbo must be able to represent everything its connected backends
can represent.* A shortfall where **pure NeuroML/LEMS can express X but tvbo's adapter cannot** is a
tvbo bug to fix — not a "framework limit." Only a genuine NeuroML incapability is a limit worth
showcasing (that's the story tvbo tells: switch backends).

**Goal (generalized, not a one-off patch):** ingest the **NeuroML core type definitions** into
tvbo's **ontology** so that:
1. `tvbo.owl` gains a **semantic reference to NeuroML-core** (each LEMS `ComponentType` → an OWL
   class; the `extends` hierarchy → `rdfs:subClassOf`), alongside the existing GO link.
2. `tvbo.owl` records the **semantic overlaps between tvbo-core and NeuroML-core** (e.g.
   `tvbo:Dynamics ≈ lems:ComponentType`, `tvbo:StateVariable ≈ StateVariable`, `tvbo:Parameter ≈
   Parameter`, `tvbo:DerivedVariable ≈ DerivedVariable`, `tvbo:Event ≈ EventPort`; plus curated
   domain overlaps: synapse, cell, population).
3. The tvbo **NeuroML adapter's emission is grounded in this ontology** — the hardcoded
   `_BASE_TYPE_META` (4 base types today) is replaced by contracts **derived from the ingested
   NeuroML ontology** (all 107 core types). Then any NeuroML component tvbo emits (synapses,
   inputs, …) is faithful by construction.

The immediate concrete win this unlocks: emitting **custom synapse LEMS ComponentTypes** (tvbo can't
today), which is required for Deco 2014's **saturating NMDA gate** — see §7.

---

## 2. Ground-truth locations (corrected — I initially had this wrong)

- **Schema ground-truth:** `/Users/leonmartin_bih/tools/tvbo/schema/` (LinkML). Main file
  `tvbo_datamodel.yaml`. Regenerate the datamodel after schema edits with
  `/Users/leonmartin_bih/tools/tvbo/.venv/bin/python hatch_build.py` (NOT `make gen-linkml`). The
  generated `tvbo/datamodel/{schema.py,pydantic.py}` are **untracked** (build hook).
- **Ontology ground-truth:** `/Users/leonmartin_bih/tools/tvbo/ontology/` — the mergeable source
  modules. **NOT** `tvbo/data/ontology/`.
- **Generated, do NOT hand-edit:** `tvbo/data/ontology/tvbo.owl` (runtime-loaded via
  `tvbo/ontology/owl.py`, lazy owlready2).
- **Deprecated, retire soon:** `tvbo/data/ontology/tvb-o.owl` (old name). The new name is
  `tvbo.owl`. Do not build new work on `tvb-o.owl`.

### Ontology build pipeline (`ontology/README.md`)
- `make gen-owl`  → `ontology/tvb-o-struct.owl`  (structural OWL from `schema/tvbo_datamodel.yaml`).
- `make gen-shacl`→ `ontology/tvb-o.shacl.ttl`   (SHACL shapes from the schema).
- Hand-authored / addon **mergeable modules** layered on top: `tvb-o-axioms.ttl` (OWL axioms LinkML
  can't express), `tvb-o-bifurcation.ttl`, `tvb-o-biology.ttl`, `tvb-o-clinical.ttl`,
  `tvb-o-clinical-nmm.ttl`, `tvb-o-data.ttl`; SPARQL post-merge updates `*.ru`
  (`clinical-postmerge.ru`, `fix-punning.ru`).
- These merge → `tvbo.owl` (checked in; CI fails on drift).

**➡ NeuroML ingestion = a new GENERATED mergeable module `ontology/tvb-o-neuroml.ttl`** (from the
LEMS core types) **+ mappings in `tvb-o-axioms.ttl`** (or a `tvb-o-neuroml-mappings.ttl`), mirroring
the `tvb-o-bifurcation.ttl` precedent exactly. Never hand-edit the generated `.owl`.

---

## 3. Verified feasibility (facts, already checked — don't re-derive)

- **The NeuroML core types are machine-ingestible.** They live inside the jNeuroML jar:
  `.venv/lib/python3.12/site-packages/pyneuroml/lib/jNeuroML-0.14.0-jar-with-dependencies.jar`,
  entries `NeuroML2CoreTypes/{Cells,Synapses,Inputs,Networks,Channels,PyNN,Simulation,...}.xml`.
  Extract with `unzip -o "$JAR" 'NeuroML2CoreTypes/*.xml' -d <dir>` (use an **absolute** jar path).
- **pylems parses them with the full contract.** `pyneuroml`/`pylems`(`lems`) are installed in the
  `.venv`. Verified:
  ```python
  from lems.model.model import Model
  m = Model(include_includes=True)
  m.import_from_file('<dir>/NeuroML2CoreTypes/Synapses.xml')  # → 107 component_types (with includes)
  ct = m.component_types['baseConductanceBasedSynapse']
  ct.extends            # 'baseVoltageDepSynapse'
  [e.name for e in ct.exposures]     # ['g']
  [r.name for r in ct.requirements]  # (v inherited)
  [(e.name,e.direction) for e in ct.event_ports]
  ```
  Verified synapse hierarchy: `baseSynapse`(extends `basePointCurrent`, event port `in`) →
  `baseVoltageDepSynapse`(requires `v`) → `baseConductanceBasedSynapse`(exposes `g`) →
  `expOneSynapse`; `blockingPlasticSynapse` extends `expTwoSynapse`.
- **tvbo already has the mirror infrastructure:** `tvbo/ontology/semanticweb/tvbgo.py` is the
  tvbo↔**GO** bridge (curated cross-reference + external-ontology ingest) — the pattern to copy for
  NeuroML; `tvbo/ontology/owl.py` builds/loads `tvbo.owl`; mergeable-module precedent =
  `tvb-o-bifurcation.ttl`.

---

## 4. Target architecture (three layers, one source of truth)

1. **Ingest NeuroML-core → `ontology/tvb-o-neuroml.ttl` (generated).** A generator (pylems → rdflib)
   walks the 107 `ComponentType`s: class per type, `extends`→`subClassOf`,
   `exposures`/`requirements`/`event_ports`/dimensions → annotation properties. Regenerated when the
   jNeuroML version bumps (like `gen-owl`).
2. **Bridge tvbo-core ↔ NeuroML-core** (mirror `tvbgo.py`): auto structural mappings (Dynamics ≈
   ComponentType, StateVariable/Parameter/DerivedVariable/Event ≈ their LEMS peers) + a small curated
   domain set (synapse/cell/population). Emitted into `tvb-o-axioms.ttl` (or `tvb-o-neuroml-mappings.ttl`)
   → merged into `tvbo.owl`. **This is the "tvbo.owl links to NeuroML + shows overlaps" deliverable.**
3. **Ground the emitter in the ontology.** Replace the hardcoded `_BASE_TYPE_META`
   (`tvbo/adapters/neuroml.py:1351`, only `baseCellMembPot`/`baseIonChannel`/`baseGate`/
   `baseVoltageDepRate`) with base-type contracts **derived from the ingested NeuroML ontology** (all
   types incl. `baseSynapse`/`baseVoltageDepSynapse`/`baseConductanceBasedSynapse`, inputs, …). The
   emitter (`_hier_build_dynamics` at `:1433`, driven by `exposures`/`requirements`) then handles any
   base type. **Custom-synapse emission falls out for free** → gap 1 solved generally, not special-cased.

---

## 5. tvbo NeuroML adapter internals (the code to touch)

`tvbo/adapters/neuroml.py`:
- `_BASE_TYPE_META` (`:1351`) — the hand-curated base-type registry to REPLACE with ontology-derived.
- `_hier_build_dynamics` (`:1433`) — emits a LEMS ComponentType's `Dynamics` from a tvbo `Dynamics` +
  the base-type `exposures`/`requirements`. Already general over the registry.
- `_hier_parse_select_label` (`:1414`) — parses `select:.../reduce:add` DerivedVariable labels.
- `_SYNAPSE_TYPES` (`:~542`), `CURRENT_INPUT_TYPES`, `EVENT_SOURCE_TYPES` — standard-type name sets.
- Network paths: `_build_std_network_context` (`:2011`, standard NeuroML types) and
  `_build_network_context` (`:2514`, custom-LEMS). Both currently need `network.nodes`+`network.edges`
  and emit one connection per edge.
- Templates: `tvbo/templates/neuroml/*.mako` (`tvbo-neuroml-hier-custom-lems.xml.mako` is the custom
  ComponentType emitter; `tvbo-neuroml-lems.xml.mako:131` shows the `reduce="add"` over `synapses[*]`).

---

## 6. NeuroML's genuine limits (document/showcase) vs tvbo adapter gaps (fix)

Established from the installed XSD + LEMS core types + object model (evidence-backed):
- **Genuine NeuroML limits** (no fix — these motivate backend switching):
  - **No all-to-all / connectivity-rule construct.** Every `<projection>` enumerates every
    `<connection>` (O(N²)); the `AllAll`/`EventConnectivity` primitive exists only as commented-out
    "Not yet stable" text in `Networks.xml`. At Deco's real scale (~2000 neurons/area, Wong-Wang 2006)
    → millions of connections → impractical even via the HDF5 network format.
  - **No O(1) population-summed shared gating.** `reduce="add"` sums only a cell's *own* attached
    synapses, never a foreign population. That O(1) reduction *is* the mean-field DMF (already on
    tvboptim).
- **tvbo adapter gaps** (must fix — pure NeuroML *can* express these):
  - **Custom-synapse LEMS emission** (no synapse base type in `_BASE_TYPE_META`) → Deco's saturating
    NMDA (Eqs 3-4, `ds/dt = −s/τ + α·x·(1−s)`) currently downgraded to the linear `blockingPlasticSynapse`.
    **This is the headline fix**; the ontology grounding delivers it.
  - **Idiomatic projection output**: emit `<projection>`+`<connectionWD>` grouped per receptor, not
    2450 flat `<synapticConnection>`.
- **Layering (not gaps):** FIC tuning + 1000-trial / ΔI ensembles are tvbo `Algorithm`/`Exploration`,
  realized per run on any backend — correct as-is.

---

## 7. Immediate problem context — the Deco 2014 spiking column (acceptance test)

The spiking half of the Deco replication lives in the **manuscript repo**:
`/Users/leonmartin_bih/projects/TVB-O/tvbo-manuscript/use-cases/replication_studies/Deco2014/`
- `Deco2014_spiking.yaml` — 11 tvbo `Dynamics` components mapped to NeuroML iris (iafRefCell,
  expOneSynapse, blockingPlasticSynapse+voltageConcDepBlockMechanism, poissonFiringSynapse) + a
  descriptive `network` block + (added this session) `code_source` + 3 experiments (S0 isolated,
  S1 bifurcation, S2 decorrelation).
- `code/spiking/` — the **reference Brian2 implementation** (`deco_column.py` — the *correct*
  behavior; isolated rates 2.91/7.68 Hz vs Deco 2.92/7.54), `deco_analysis.py`, `bifurcation_sweep.py`,
  `validate_isolated.py`, figure makers.

**Acceptance test for the whole ingestion work:** a faithful Deco **saturating-NMDA** synapse
(Eqs 3-4) authored as a tvbo `Dynamics` renders — via the ontology-derived `baseConductanceBasedSynapse`
contract — to **valid NeuroML/LEMS** (PyLEMS-valid), and a small E/I column runs through jNeuroML→Brian2
producing spikes. The existing suite must still pass:
`.venv/bin/python -m pytest tests/functional/test_simulation_backends_neuroml.py -q --override-ini="addopts="`.

NOTE: `test_run_brian2` currently **fails** — it's the pyNeuroML→Brian2 *runner's* success-detection
(jnml translates fine); brian2 was only just installed, so the test never ran before. Environmental,
**not** a code regression. Confirm/park it; don't chase it as part of the ingestion.

---

## 8. Staged plan

1. **Stage 1 — ingestion generator** (pylems core types → `ontology/tvb-o-neuroml.ttl`). Self-contained,
   testable in isolation (assert N classes, the synapse subClassOf chain, exposures/requirements
   annotations). Wire into the ontology build (a `gen-neuroml` step; merge into `tvbo.owl`).
2. **Stage 2 — the bridge** (auto structural mappings + curated domain overlaps → `tvb-o-axioms.ttl` /
   `tvb-o-neuroml-mappings.ttl`; merged). Verify `tvbo.owl` shows tvbo↔NeuroML links.
3. **Stage 3 — emitter grounding** (`_BASE_TYPE_META` derived from the ontology; general over all base
   types incl. synapses/inputs). Keep behavior identical for the existing cell/channel/gate cases
   (regression-guard with the neuroml test suite).
4. **Stage 4 — validate** the saturating-NMDA acceptance test (§7).

**De-risking option:** a thin vertical slice first — ingest only the *synapse* branch → bridge → emit
the NMDA — before ingesting all 107 types. Recommended.

---

## 9. Working-tree state at handoff (tvbo repo, branch `dev`, HEAD `508a75d9`)

The user commits selectively (e.g. `39e6d850` already landed the `symbol` feature: `dynamics.py`
`symbol_map` + report template). **Uncommitted `M` files** split as:

**Validated — keep (safe to commit):**
- `tvbo/analysis/linear_response.py` — parse rhs via canonical `parse_eq(..., local_dict=model.get_symbolic_elements())` so builtin-colliding names (`gamma`) stay symbols. Byte-identity suite 23/23.
- `tvbo/templates/tvboptim/utils.py` — `resolve_coupling_spec`: identity-sentinel FastLinearCoupling → incoming path (fixes 0.4.0 `pre()` contract for both-declared incoming+local). Byte-identity 23/23.
- `tvbo/utils/report.py` — `display_symbol` + cached `_symbol_latex` (part of the committed symbol feature; the user refined it).
- `schema/tvbo_datamodel.yaml` — **partly**: the `symbol` slot on `CouplingInput` (keep). See "provisional" for the rest.

**Provisional — revisit/likely revert before the ontology work (design pivoted past them):**
- `tvbo/adapters/neuroml.py` — the failed agent's all-to-all lowering: `_connectivity_pairs` +
  `Node.size`/`Edge.connectivity` expansion, but emits **flat `<synapticConnection>`** (not projections)
  and is grounded in a hardcoded registry. The ontology approach supersedes the grounding; the
  projection-output + custom-synapse pieces are unbuilt. Keep as reference, don't commit as-is.
- `schema/tvbo_datamodel.yaml` — the agent's `ConnectivityRule` enum + `Node.size` +
  `Edge.allow_self_connections`/`connectivity`. `ConnectivityRule` (rule-expansion) is *defensible*
  (matches NetPyNE/neuroConstruct tooling), but re-decide it within the ontology-grounded design before
  keeping. Datamodel was regenerated for these.

**Unclear — check before committing:** `pyproject.toml` (a neuroml/brian2 extra?),
`docs/Interoperability/TVB/Monitor.qmd` (agent-incidental), and prior-session tvboptim
`tvbo-tvboptim-cfun.py.mako` / `tvbo-tvboptim-experiment.py.mako` (from the 0.4.0-alignment session
`b3908dd4`; verify whether they should be committed).

**Deps installed into `.venv` this session (via `uv pip`, NOT in pyproject):** `pyneuroml`, `brian2`,
`libneuroml`, `pylems`, `lxml` (needed by the NeuroML backend + this ontology work).

---

## 10. Gotchas / commands
- Regenerate datamodel after schema edits: `.venv/bin/python hatch_build.py` (from repo root).
- Extract NeuroML core types from the jar with an **absolute** jar path + `-d <outdir>`.
- Never hand-edit `tvbo/data/ontology/tvbo.owl`; edit sources in `ontology/`, rebuild.
- `import tvbo` is ~20 ms (lazy owlready2/jax); test latency is JAX compile + jnml runs, not import.
- Run neuroml tests with `--override-ini="addopts="` (no xdist).
- Read-only git only unless the user explicitly asks; no `git clean`; no Claude attribution in commits.

## 11. Related memories (auto-recalled)
`reference-symbol-display-override`, `project-tvboptim-040-alignment`,
`feedback-root-cause-not-workaround`, `feedback-generalize-backend-independent`,
`feedback-tvbo-extends-backends-via-codegen`, `reference-bifurcation-ontology-taxonomy`,
and the new `project-neuroml-ontology-ingestion` (points here).
