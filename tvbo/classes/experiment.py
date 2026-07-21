#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""The `SimulationExperiment` host object that ties a whole simulation together.

A `SimulationExperiment` binds local [`Dynamics`](../classes/dynamics.qmd), a
[`Network`](../classes/network.qmd), coupling, integration settings, monitors,
and stimulation into a single, YAML-round-trippable object. It is the entry
point for constructing an experiment (from YAML, a file, the platform, or a TVB
simulator), configuring and resolving its coupling/delay metadata, rendering
backend code, and running it on any of the supported backends (`tvb`,
`tvboptim`, `jax`, `pde`, `cuda`, `python`).
"""
import copy as _copy
import logging
import os
import re
from collections import Counter
from os.path import join
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

# Apply the JAX Metal-fallback guard before any simulation touches JAX. Idempotent
# and cheap here (jax is already imported); kept out of ``import tvbo`` so the CLI
# and bare imports stay fast. See tvbo.__init__._configure_jax_backend.
from tvbo import _configure_jax_backend as _cfg_jax

_cfg_jax()

try:
    from lems.base.util import validate_lems
except ImportError:
    validate_lems = None  # PyLEMS is optional (neuroml extra)

from tvbo import templates
from tvbo.classes.network import Network
from tvbo.data.types import SimulationResult, SimulationState, TimeSeries, ExperimentResult, ExplorationResult
from tvbo.datamodel import schema as tvbo_datamodel
from linkml_runtime.utils.yamlutils import YAMLRoot
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from tvbo.codegen import templater
from tvbo.codegen.templater import format_code
from tvbo.classes.coupling import Coupling
from tvbo.classes.noise import Integrator
from tvbo.classes.continuation import Continuation
from tvbo.classes.dynamics import Dynamics
from tvbo.run.graph import GraphRunner as _Network
from tvbo.adapters.tvb import from_tvb_simulator as _from_tvb_simulator
from tvbo.utils import traverse_metadata
from tvbo.utils import Bunch
from tvbo.utils import as_list
from tvbo.log import ensure_configured

logger = logging.getLogger(__name__)

sessionid = 1


def _strip_private_yaml_keys(text: str) -> str:
    """Drop private/runtime keys (``_source_file``, a codegen ``_coupling_key`` on a
    Parameter, …) from a serialized YAML at any depth, so the spec round-trips
    through ``from_file``. A private key line and its whole value block go too. The
    value block includes deeper-indented lines and a block-sequence value whose
    ``-`` items YAML writes at the key's own indentation (not deeper); without that
    case a private list-valued cache (e.g. ``_model_labels_from_bids``) would leave
    its items orphaned as a bare list at the document root.
    """
    import re as _re

    key = _re.compile(r"^(\s*)_[A-Za-z]\w*\s*:")
    kept, drop_indent = [], None
    for line in text.splitlines():
        if drop_indent is not None:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            same_indent_seq_item = indent == drop_indent and (
                stripped == "-" or stripped.startswith("- ")
            )
            if stripped == "" or indent > drop_indent or same_indent_seq_item:
                continue
            drop_indent = None
        m = key.match(line)
        if m:
            drop_indent = len(m.group(1))
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


# Maps ``BidsEntities`` attribute names to the short entity keys that
# ``tvbo.classes.network._parse_bids_entities`` emits from a filename. ``suffix``
# is deliberately absent — it is the trailing filename component, not a key-value
# entity, and is handled separately by its consumers.
_BIDS_ENTITY_SHORT_KEYS = {
    "template": "tpl", "cohort": "cohort", "reconstruction": "rec",
    "segmentation": "seg", "scale": "scale", "atlas": "atlas",
    "acquisition": "acq", "hemi": "hemi", "desc": "desc",
}


def _bids_entities_to_short_dict(obj) -> dict:
    """Convert a ``BidsEntities`` object to ``{short_key: value}`` for the set entities."""
    return {short: str(getattr(obj, attr))
            for attr, short in _BIDS_ENTITY_SHORT_KEYS.items()
            if getattr(obj, attr, None) is not None}


def _sync_network_node_count(net):
    """Sync number_of_nodes from the nodes list.

    When Network is created via LinkML deserialization + __class__ patching,
    Network.__init__ never runs. This ensures node count is consistent.
    """
    # Migrate deprecated number_of_regions -> number_of_nodes
    if getattr(net, "number_of_regions", None) and not getattr(net, "number_of_nodes", None):
        net.number_of_nodes = net.number_of_regions
    if net.nodes:
        n = len(net.nodes)
        net.number_of_nodes = n
    elif (net.number_of_nodes or 0) > 1 and not net.nodes:
        # number_of_nodes set but no nodes list — create default nodes
        net.nodes = [tvbo_datamodel.Node(id=i, label=f"node_{i}") for i in range(net.number_of_nodes)]


def _upgrade_network_couplings(network, coupling_types=None):
    """Upgrade network.coupling entries to runtime Coupling instances and
    apply ``type``-based database/ontology fill.

    Parameters
    ----------
    network : Network
        The network whose coupling entries should be upgraded.
    coupling_types : dict, optional
        Mapping ``{coupling_key: type_ref}`` extracted from the raw YAML
        before LinkML deserialization.  ``type_ref`` is a coupling function
        name or CURIE (e.g. ``"KuramotoCoupling"`` or
        ``"tvbo:KuramotoCoupling"``).
    """
    coupling_types = coupling_types or {}
    coup_raw = getattr(network, "coupling", None)
    if not coup_raw:
        return

    # Schema allows dict (keyed by name), list, or a single Coupling object.
    if isinstance(coup_raw, dict):
        items = coup_raw.items()
    elif isinstance(coup_raw, list):
        items = enumerate(coup_raw)
    else:
        # Single Coupling object assigned directly
        if not isinstance(coup_raw, Coupling):
            coup_raw.__class__ = Coupling
        return

    for key, coup in items:
        # Upgrade __class__ to runtime Coupling (skip if already runtime)
        if not isinstance(coup, Coupling):
            coup.__class__ = Coupling
            # Always trigger ontology population (idempotent).
            if not getattr(coup, "pre_expression", None):
                coup._populate_from_ontology()
        # Apply type-based fill if a type reference was specified
        if key in coupling_types:
            coup.populate_from_type(coupling_types[key])


def _resolve_coupling(experiment):
    """Reconcile coupling declarations across experiment / network / dynamics.

    Single source of truth for both ``__init__`` and ``from_datamodel`` paths.

    Rules:
      1. If ``network.coupling`` is populated, mirror its first entry to
         ``experiment.coupling``.
      2. Else, if ``dynamics.coupling_inputs`` declares coupling targets,
         seed ``network.coupling`` from ``experiment.coupling`` (or a default
         ``Linear`` if absent) so a single canonical container exists.
      3. Auto-populate ``incoming_states`` on each coupling from the dynamics'
         ``state_variables[*].coupling_variable == True`` flag — but only when
         the coupling declares neither incoming nor local states. Couplings
         that explicitly declare ``local_states`` (e.g. vectorized matmul
         flavors) are left untouched.
    """
    dyn = getattr(experiment, "dynamics", None)
    network = getattr(experiment, "network", None)
    if network is None:
        return

    dyn_ci = getattr(dyn, "coupling_inputs", None)
    has_coupling_inputs = bool(dyn_ci)
    net_coup = getattr(network, "coupling", None)
    exp_coup = getattr(experiment, "coupling", None)

    if net_coup:
        if isinstance(net_coup, dict) and net_coup:
            experiment.coupling = next(iter(net_coup.values()))
    elif has_coupling_inputs:
        func = exp_coup
        if not func or not isinstance(func, Coupling):
            func = Coupling(name="Linear", iri="tvbo:Linear")
            experiment.coupling = func
        elif not getattr(func, "iri", None):
            func.iri = "tvbo:Linear"
        # Ensure ontology-derived fields (pre/post expressions, parameters)
        # are populated — backends require them at codegen time.
        if not getattr(func, "pre_expression", None):
            func._populate_from_ontology()
        coup_name = str(getattr(func, "name", "Linear"))
        network.coupling[coup_name] = func

    if dyn:
        cvars = [
            str(sv.name)
            for sv in (getattr(dyn, "state_variables", None) or {}).values()
            if getattr(sv, "coupling_variable", False)
        ]
        if cvars:
            for coup in (getattr(network, "coupling", None) or {}).values():
                if (not getattr(coup, "incoming_states", None)
                        and not getattr(coup, "local_states", None)):
                    coup.incoming_states = list(cvars)


def _iri_local(iri: str) -> str:
    return iri.split(":", 1)[-1] if ":" in iri else iri


def _backfill_name_from_iri(d):
    """If d is a dict with iri but no name, inject name from iri's local part."""
    if isinstance(d, dict) and d.get("iri") and not d.get("name"):
        d["name"] = _iri_local(d["iri"])


def _merge_from_registry(d, category: str):
    """Enrich an ``iri``-referenced spec from the registry, inline values winning.

    If ``d`` is a dict carrying an ``iri`` CURIE, load the registry entry it
    points to and deep-merge the inline dict on top: inline values supervene at
    the *leaf* (e.g. ``parameters: {a: {value: 1}}`` overrides only ``a.value``
    and keeps every other parameter from the registry entry), while the entry
    fills everything ``d`` did not specify. Mutates ``d`` in place. Falls back to
    a name-only backfill if the ``iri`` does not resolve to a DB entry.
    """
    if not isinstance(d, dict) or not d.get("iri"):
        return
    local = _iri_local(d["iri"])
    try:
        from tvbo.data.registry import resolve
        from tvbo.utils import yaml_loader, deep_merge

        loaded = yaml_loader.load_as_dict(str(resolve(category, local)))
        if isinstance(loaded, dict):
            merged = deep_merge(loaded, d)  # registry = base, inline `d` overrides
            d.clear()
            d.update(merged)
            return
    except (FileNotFoundError, RuntimeError, ValueError):
        pass
    d.setdefault("name", local)


class SimulationExperiment(tvbo_datamodel.SimulationExperiment):
    """The central runnable object in TVBO: a complete brain-network simulation spec.

    Bundles `dynamics`, `coupling`, `network`, `integration`, `observations`,
    and any analysis layers (stimulation, algorithms, explorations, …) into
    one declarative specification. The same instance can be:

    - **executed** in any registered backend (`run("jax")`, `run("tvb")`, …)
    - **serialized** to YAML, BIDS, openMINDS, or LEMS
    - **rendered** as code for inspection or external use (`render_code(...)`)
    - **reported** as Markdown or HTML (`report(...)`)

    Construct via direct kwargs, from a `Dynamics` instance, by name from the
    curated database (`from_db`), or by loading a YAML / BIDS export.

    Examples:
        ```python
        from tvbo import Dynamics, SimulationExperiment

        exp = SimulationExperiment(dynamics=Dynamics.from_db("Generic2dOscillator"))
        result = exp.run("jax", duration=5_000)

        # Or fully declarative
        exp = SimulationExperiment(
            dynamics={"iri": "tvbo:ReducedWongWangExcInh"},
            coupling={"iri": "tvbo:Linear"},
            network={"parcellation": {"atlas": {"iri": "tvbo:DesikanKilliany"}},
                     "tractogram":   {"iri": "tvbo:dTOR"}},
            integration={"method": "Heun", "duration": 10_000, "noise": None},
        )
        ```

    See the [Simulation experiments](/Simulation/SimulationExperiments.qmd)
    page for the full constructor surface and the
    [`running-simulations`](../../../skills/running-simulations/SKILL.md) skill
    for backend choices.
    """

    def __init__(self, **kwargs):
        """Initialize like the datamodel, but auto-assign an id when missing.

        Supports any of the following inputs:
        - A tvbo_datamodel.SimulationExperiment instance
        - A dict of fields
        - Keyword args matching the datamodel fields
        """
        global sessionid

        # Ensure an id exists (the datamodel requires it in __post_init__)
        if kwargs.get("id") is None:
            kwargs["id"] = sessionid
            sessionid += 1

        # Handle pydantic Network - convert to dict before parent __init__
        # This prevents the parent's __post_init__ from failing on pydantic objects
        if "network" in kwargs and kwargs["network"] is not None:
            net = kwargs["network"]
            # Check if it's a pydantic model (has model_dump) - convert to dict
            if hasattr(net, "model_dump"):
                kwargs["network"] = net.model_dump(exclude_none=True)
            elif hasattr(net, "dict") and not isinstance(net, dict):
                # Pydantic v1
                kwargs["network"] = net.dict(exclude_none=True)

        # Extract non-schema fields from network sub-dicts before parent init.
        # These are popped so the datamodel __init__ doesn't reject unknown kwargs.
        _coupling_types = {}
        net_kw = kwargs.get("network")
        if isinstance(net_kw, dict):
            # Extract `type` from network.coupling entries.
            # `type` references a coupling function name/CURIE.
            coup_dict = net_kw.get("coupling")
            if isinstance(coup_dict, dict):
                for key, val in coup_dict.items():
                    if isinstance(val, dict) and "type" in val:
                        _coupling_types[key] = val.pop("type")

        # Allow coupling to be specified as a plain string name (e.g. "SigmoidalJansenRit")
        # Set iri to trigger ontology population
        if "coupling" in kwargs and isinstance(kwargs["coupling"], str):
            _cname = kwargs["coupling"]
            kwargs["coupling"] = {"name": _cname, "iri": f"tvbo:{_cname}"}

        # Resolve Dynamics slot aliases (e.g. components → modes) before
        # the parent __post_init__ constructs the base-class Dynamics.
        dyn_kw = kwargs.get("dynamics")
        if isinstance(dyn_kw, dict):
            from tvbo.classes.dynamics import _resolve_dynamics_aliases

            _resolve_dynamics_aliases(dyn_kw)

        # Also resolve aliases and iri-source the keyed network.dynamics entries
        # (per-node dynamics), symmetrically with the top-level `dynamics` slot.
        net_kw = kwargs.get("network")
        if isinstance(net_kw, dict):
            net_dyn = net_kw.get("dynamics")
            if isinstance(net_dyn, dict):
                from tvbo.classes.dynamics import _resolve_dynamics_aliases

                for _key, _dv in net_dyn.items():
                    if isinstance(_dv, dict):
                        _resolve_dynamics_aliases(_dv)
                        if _dv.get("iri"):
                            _merge_from_registry(_dv, "Dynamics")
                            # Keyed collection: the population key is the
                            # identifier, so it stays the `name` (model identity
                            # is carried on `iri`).
                            _dv["name"] = _key

        # Resolve iri-only refs by loading from the registry (full population
        # for dynamics/coupling) or backfilling name (for parcellation/tractogram).
        _merge_from_registry(kwargs.get("dynamics"), "Dynamics")
        _merge_from_registry(kwargs.get("coupling"), "Coupling")
        net_kw = kwargs.get("network")
        if isinstance(net_kw, dict):
            parc = net_kw.get("parcellation")
            if isinstance(parc, dict):
                # parcellation: {"iri": "tvbo:X"} → wrap into atlas
                if parc.get("iri") and not parc.get("atlas"):
                    parc["atlas"] = {"name": _iri_local(parc["iri"])}
                _backfill_name_from_iri(parc.get("atlas"))
            _backfill_name_from_iri(net_kw.get("tractogram"))
            # Eagerly construct the wrapper Network so its __init__ runs
            # (loads normative connectivity from atlas/tractogram). The parent
            # __post_init__ accepts a Network instance as-is.
            kwargs["network"] = Network(**net_kw)

        # Delegate to the parent dataclass initializer
        super().__init__(**kwargs)

        # Normalize and coerce fields while preserving original conditions
        def _coerce(cls, obj, **extra):
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, YAMLRoot):
                # Copy dataclass fields directly — _as_dict over-serializes
                # enums and nested objects causing lossy round-trips.
                from dataclasses import fields as dc_fields

                d = {}
                for f in dc_fields(obj):
                    if f.name.startswith(("_", "class_")):
                        continue
                    val = getattr(obj, f.name, f.default)
                    # Convert enums/PermissibleValues to plain text so
                    # constructors can re-parse them via __post_init__.
                    from linkml_runtime.linkml_model.meta import PermissibleValue

                    if isinstance(val, PermissibleValue):
                        val = val.text
                    elif isinstance(val, EnumDefinitionImpl):
                        val = str(val)
                    if val is not None:
                        d[f.name] = val
                return cls(**d, **extra)
            if isinstance(obj, dict):
                return cls(**obj, **extra)
            return obj

        # Prefer `model` (name string) when `dynamics` is missing
        if getattr(self, "model", None) and not getattr(self, "dynamics", None):
            self.dynamics = Dynamics(name=self.model)

        # Coerce dynamics to enhanced Dynamics class
        if getattr(self, "dynamics", None) and not isinstance(self.dynamics, Dynamics):
            if isinstance(self.dynamics, tvbo_datamodel.Dynamics):
                self.dynamics = Dynamics.from_datamodel(self.dynamics)
            else:
                self.dynamics = _coerce(Dynamics, self.dynamics)

        # Mirror dynamics name → model (schema field is a string, not an object)
        if getattr(self, "dynamics", None):
            self.model = self.dynamics.name

        if getattr(self, "coupling", None) and not isinstance(self.coupling, Coupling):
            self.coupling = _coerce(Coupling, self.coupling)

        if getattr(self, "integration", None) and not isinstance(self.integration, Integrator):
            self.integration = _coerce(Integrator, self.integration)

        if getattr(self, "stimulation", None):
            from tvbo.classes import perturbation

            if not isinstance(self.stimulation, perturbation.Stimulus):
                self.stimulation = _coerce(perturbation.Stimulus, self.stimulation)

        # Auto-upgrade continuations to runtime Continuation class
        conts = getattr(self, "continuations", None)
        if conts and isinstance(conts, dict):
            for key, val in conts.items():
                if val is not None and not isinstance(val, Continuation):
                    val.__class__ = Continuation
                    conts[key] = val

        # Auto-upgrade events to runtime Event class (adds .plot() helper)
        evts = getattr(self, "events", None)
        if evts and hasattr(evts, "items"):
            from tvbo.classes.event import Event as _EventCls
            for val in evts.values():
                if val is not None and not isinstance(val, _EventCls):
                    val.__class__ = _EventCls

        # Resolve observations that reference a curated model by `iri` (e.g.
        # `iri: tvbo:BOLD_TVB`): merge the model's pipeline/parameters/class_reference
        # in non-destructively so locally declared source/period overrides still win.
        obss = getattr(self, "observations", None)
        if obss and hasattr(obss, "values"):
            from tvbo.classes.observation import populate_observation_from_iri
            # A curated model may ship the helper functions its pipeline calls; collect
            # them into a fresh sink so an experiment with no iri-referenced model keeps
            # its functions table untouched (reassigning it re-wraps the type downstream).
            _iri_funcs: dict = {}
            for val in obss.values():
                if val is not None and getattr(val, "iri", None):
                    populate_observation_from_iri(val, functions_sink=_iri_funcs)
            if _iri_funcs:
                # Keep experiment.functions a plain name->Function dict (the YAML-loaded
                # form). Assigning through the LinkML setter re-wraps it into a JsonObj
                # that mangles the Function values and breaks `dict(experiment.functions)`
                # in codegen — so mutate the existing dict in place, and for an
                # experiment with none, install a plain dict via object.__setattr__
                # (bypassing that setter). Add only functions the experiment lacks.
                funcs = getattr(self, "functions", None)
                if funcs is None:
                    funcs = {}
                    object.__setattr__(self, "functions", funcs)
                for k, v in _iri_funcs.items():
                    if k not in funcs:
                        funcs[k] = v

        if not getattr(self, "network", None):
            self.network = Network()
        else:
            self.network.__class__ = Network
            # Network.__init__ doesn't run when class is patched, so sync here
            _sync_network_node_count(self.network)
            if not getattr(self.network, "conduction_speed", None):
                self.network.parameters["conduction_speed"] = tvbo_datamodel.Parameter(
                    name="conduction_speed",
                    label="v",
                    value=3.0,
                    unit="mm_per_ms",
                )

        # Upgrade network.coupling entries to runtime Coupling + apply type fills
        _upgrade_network_couplings(self.network, _coupling_types)
        # NOTE: coupling resolution (incoming_states / network.coupling seeding)
        # is intentionally deferred to ``configure()`` so it runs lazily at the
        # execution boundary across all backends and so the resolved state is
        # observable in the spec only after the user explicitly prepared the
        # experiment. See ``configure()``.

        # Get source file path if loading from file (set by from_file classmethod)
        self._source_file = getattr(self.__class__, "_pending_source_file", None)

        # Materialise the network from its declarative spec. All branching
        # lives in Network._resolve; this hook only supplies the YAML
        # source directory for relative-path resolution.
        if self.network is not None and not getattr(self.network, "_resolved", False):
            source_dir = Path(self._source_file).parent if self._source_file else None
            self.network._resolve(source_dir=source_dir)

        if not getattr(self, "integration", None):
            self.integration = Integrator(method="Heun")

    def _load_network_from_data_file(self):
        """Load network matrices from a companion data file (h5/zarr/yaml sidecar).

        Resolves ``network.data_file`` as an absolute path or relative to the
        YAML source file / cwd.  The coupling and transforms defined inline in
        the experiment YAML are preserved after loading.
        """
        from pathlib import Path
        from tvbo.classes.network import Network as _Network

        data_file = Path(self.network.data_file)
        if not data_file.is_absolute():
            source_file = getattr(self, "_source_file", None)
            if source_file:
                data_file = (Path(source_file).parent / data_file).resolve()
            else:
                data_file = (Path.cwd() / data_file).resolve()

        # Accept .h5/.zarr → find sidecar, or direct .yaml sidecar path
        if data_file.suffix in (".h5", ".zarr"):
            sidecar = data_file.with_suffix(".yaml")
        else:
            sidecar = data_file

        loaded = _Network.from_file(sidecar)

        # Preserve inline experiment coupling / transforms.
        # We must use indexing on the loaded network's containers rather than
        # bulk-assigning a plain dict, because LinkML __setattr__ wraps plain
        # dicts into JsonObj which breaks .items() downstream.
        inline_coupling = dict(self.network.coupling) if self.network.coupling else {}
        inline_transforms = list(self.network.transforms) if self.network.transforms else []

        self.network = loaded
        self.network.__class__ = Network

        if inline_coupling:
            # Clear loaded network's coupling, then insert inline entries
            # using indexing on the existing LinkML container.
            if hasattr(self.network.coupling, "clear"):
                self.network.coupling.clear()
            for k, v in inline_coupling.items():
                self.network.coupling[k] = v

        if inline_transforms:
            self.network.transforms = inline_transforms

    def _load_network_from_bids(self):
        """Load network matrices from BEP017 BIDS directory.

        Uses network.bids_dir, network.structural_measures, and
        network.observational_measures to load connectivity data.
        Relative paths are resolved relative to the YAML source file.
        """
        from pathlib import Path

        bids_dir = Path(self.network.bids_dir)
        if not bids_dir.is_absolute():
            # Resolve relative to YAML source file location
            source_file = getattr(self, "_source_file", None)
            if source_file:
                bids_dir = (Path(source_file).parent / bids_dir).resolve()
            else:
                # Fallback: resolve relative to cwd
                bids_dir = (Path.cwd() / bids_dir).resolve()

        # Get measures from network attributes
        structural = getattr(self.network, "structural_measures", None) or []
        if not structural:
            from tvbo.classes.network import _discover_bids_measures

            structural = _discover_bids_measures(bids_dir)
        observational = getattr(self.network, "observational_measures", None) or []

        # Use Network.load_from_bids to load data into self.network
        self.network.load_from_bids(
            bids_dir,
            structural_measures=structural,
            observational_measures=observational,
        )

    @classmethod
    def from_datamodel(cls, dm: tvbo_datamodel.SimulationExperiment) -> "SimulationExperiment":
        """Create from a datamodel instance by copying its already-normalized
        state.

        This avoids the ``_as_dict`` → re-init round-trip which breaks on
        ``inlined_as_dict`` fields (the keyed dict is not valid ``**kwargs``
        for the inner class constructor).  Instead we directly copy the
        ``__dict__`` from the fully-normalised LinkML object and then set
        the convenience aliases that ``__init__`` would normally provide.
        """
        obj = cls.__new__(cls)
        # Copy all already-normalized state from the datamodel instance
        obj.__dict__.update(dm.__dict__)

        # -- Upgrade Dynamics via __class__ reassignment --
        dyn = getattr(obj, "dynamics", None)
        if isinstance(dyn, dict) and dyn:
            for v in dyn.values():
                if isinstance(v, tvbo_datamodel.Dynamics) and not isinstance(v, Dynamics):
                    v.__class__ = Dynamics
                    if getattr(v, "iri", None):
                        v.enrich_from_ontology()
            first = next(iter(dyn.values()))
            obj.__dict__["dynamics"] = first
            obj.__dict__["model"] = first.name
        elif isinstance(dyn, tvbo_datamodel.Dynamics):
            if not isinstance(dyn, Dynamics):
                dyn.__class__ = Dynamics
                if getattr(dyn, "iri", None):
                    dyn.enrich_from_ontology()
            obj.__dict__["model"] = dyn.name
        else:
            obj.__dict__.setdefault("dynamics", None)

        # -- Upgrade Integrator via __class__ reassignment --
        integ = getattr(obj, "integration", None)
        if integ is not None and not isinstance(integ, Integrator):
            integ.__class__ = Integrator
            # Always trigger population (idempotent: only fills missing fields).
            # Lookup is by ``self.method`` when ``iri`` is unset.
            integ._populate_from_ontology()
        if not getattr(obj, "integration", None):
            obj.__dict__["integration"] = Integrator(method="Heun")

        # -- Upgrade Stimulation via __class__ reassignment --
        stim = getattr(obj, "stimulation", None)
        if stim is not None:
            from tvbo.classes import perturbation

            if not isinstance(stim, perturbation.Stimulus):
                if isinstance(stim, tvbo_datamodel.Stimulus):
                    stim.__class__ = perturbation.Stimulus
                elif isinstance(stim, dict):
                    stim = perturbation.Stimulus(**stim)
                obj.__dict__["stimulation"] = stim

        # -- Upgrade Coupling via __class__ reassignment --
        coup = getattr(obj, "coupling", None)
        if coup is not None and not isinstance(coup, Coupling):
            coup.__class__ = Coupling
            # Always trigger population (idempotent). Lookup by ``self.name``
            # when ``iri`` is unset.
            if not getattr(coup, "pre_expression", None):
                coup._populate_from_ontology()
        if not getattr(obj, "coupling", None):
            obj.__dict__["coupling"] = Coupling(name="Linear")

        # -- Upgrade Network via __class__ reassignment --
        net = getattr(obj, "network", None)
        if net is not None and not isinstance(net, Network):
            net.__class__ = Network
            _sync_network_node_count(net)
            if not getattr(net, "conduction_speed", None):
                net.parameters["conduction_speed"] = tvbo_datamodel.Parameter(
                    name="conduction_speed",
                    label="v",
                    value=3.0,
                    unit="mm_per_ms",
                )
        if not getattr(obj, "network", None):
            obj.__dict__["network"] = Network()

        # Upgrade network coupling entries (no type refs in from_datamodel path)
        _upgrade_network_couplings(obj.network)
        # Coupling resolution deferred to ``configure()`` (see __post_init__).

        obj.__dict__["_source_file"] = getattr(cls, "_pending_source_file", None)

        # Materialise the network from its declarative spec (data_file,
        # bids_dir, parcellation, graph_generator). All branching logic lives
        # in Network._resolve; this hook only supplies the YAML source
        # directory for relative-path resolution.
        net = obj.network
        if net is not None and not getattr(net, "_resolved", False):
            source_file = getattr(obj, "_source_file", None)
            source_dir = Path(source_file).parent if source_file else None
            net._resolve(source_dir=source_dir)

        # Validate declarative network-observation sources against the
        # network's declared observational_measures (names only, no data load).
        obj._validate_network_observations()

        # Lower declarative stimulus/stimulation Event fields (target_variable,
        # target_regions, weight_distribution) into the legacy fields the shared
        # codegen consumes — resolved here in Python, no template changes.
        obj._resolve_events()

        return obj

    @classmethod
    def from_pyrates(cls, filepath: str) -> "SimulationExperiment":
        """Load a SimulationExperiment from a PyRates YAML template file.

        Parses all OperatorTemplates in the file and creates a keyed dict
        of Dynamics objects.

        Parameters
        ----------
        filepath : str
            Path to PyRates YAML file.

        Returns
        -------
        SimulationExperiment
            New instance with primary dynamics and network.dynamics for
            multi-operator files.

        Example
        -------
        >>> exp = SimulationExperiment.from_pyrates("synaptic_plasticity.yaml")
        >>> print(exp.dynamics.name)  # 'tsodyks'
        >>> print(list(exp.network.dynamics.keys()))  # ['tsodyks', 'depression', 'facilitation']
        """
        from tvbo.codegen.pyrates import from_pyrates_yaml_all

        dynamics_dicts = from_pyrates_yaml_all(filepath)

        # Convert dicts to Dynamics instances using lightweight construction
        dynamics = {}
        for name, dyn_dict in dynamics_dicts.items():
            # Create instance with _skip_ontology=True to avoid slow lookups
            dyn = Dynamics(_skip_ontology=True, **dyn_dict)
            dynamics[name] = dyn

        # Use the first dynamics as the primary model
        primary_dyn = None
        if dynamics:
            first_key = next(iter(dynamics))
            primary_dyn = dynamics[first_key]

        # Store all dynamics in network.dynamics for heterogeneous support
        network_kwargs = {}
        if len(dynamics) > 1:
            network_kwargs["dynamics"] = dynamics

        return cls(
            dynamics=primary_dyn,
            network=Network(**network_kwargs) if network_kwargs else None,
        )

    @classmethod
    def from_pydantic(cls, pyd_obj: Any) -> "SimulationExperiment":
        """Create a SimulationExperiment from a Pydantic model instance.

        Args:
            pyd_obj: A Pydantic BaseModel instance (e.g., from tvbo.datamodel.tvbopydantic)

        Returns:
            SimulationExperiment instance
        """
        if hasattr(pyd_obj, "model_dump"):
            # Pydantic v2
            return cls(**pyd_obj.model_dump(exclude_none=True))
        elif hasattr(pyd_obj, "dict"):
            # Pydantic v1
            return cls(**pyd_obj.dict(exclude_none=True))
        else:
            raise TypeError(f"Expected a Pydantic model, got {type(pyd_obj)}")

    @classmethod
    def from_tvb_simulator(cls, tvb_simulator):
        """Build a `SimulationExperiment` from a configured TVB `Simulator`.

        Delegates to the [TVB adapter](../adapters/tvb.qmd) to capture the
        simulator's model, connectivity, coupling, integrator, and monitors,
        then constructs an equivalent experiment from that datamodel.

        Args:
            tvb_simulator: A configured `tvb.simulator.simulator.Simulator`.

        Returns:
            A new `SimulationExperiment` mirroring the TVB simulator.
        """
        return cls.from_datamodel(_from_tvb_simulator(tvb_simulator))

    @classmethod
    def from_file(cls, filepath: str):
        """Load a `SimulationExperiment` from a YAML file on disk.

        The resolved source path is recorded on the instance (as
        `_source_file`) so later serialization can reference its origin.

        Args:
            filepath: Path to a YAML file defining the experiment.

        Returns:
            A new `SimulationExperiment` populated from the file.
        """
        from pathlib import Path
        from tvbo.utils import yaml_loader, register_recipe_code_paths
        import yaml

        # Store source file path BEFORE loading so __init__ can use it
        cls._pending_source_file = str(Path(filepath).resolve())
        # Make the recipe's code/ subdir importable, so custom builders/callables
        # resolve by bare module name without a PYTHONPATH prefix. Before loading:
        # construction resolves the network builder eagerly (see __init__).
        register_recipe_code_paths(cls._pending_source_file)
        try:
            with open(filepath) as file_handle:
                data_as_dict = yaml.safe_load(file_handle) or {}
            # Drop private/provenance keys (e.g. _source_file) — not schema slots,
            # so a round-tripped render_yaml() spec reloads cleanly.
            if isinstance(data_as_dict, dict):
                data_as_dict = {k: v for k, v in data_as_dict.items() if not str(k).startswith("_")}
            exp = yaml_loader.loads(yaml.safe_dump(data_as_dict), target_class=cls)
            exp._source_file = cls._pending_source_file
        finally:
            cls._pending_source_file = None
        return exp

    @classmethod
    def from_string(cls, yaml_string: str) -> "SimulationExperiment":
        """Create a SimulationExperiment from a YAML string.

        This is useful for defining experiments inline in notebooks or scripts
        using human-readable YAML syntax.

        Parameters
        ----------
        yaml_string : str
            YAML-formatted string defining the experiment.

        Returns
        -------
        SimulationExperiment
            New instance populated from the YAML definition.

        Example
        -------
        >>> exp = SimulationExperiment.from_string('''
        ... id: 1
        ... label: My Experiment
        ... dynamics:
        ...   name: JansenRit
        ...   parameters:
        ...     A: {value: 3.25}
        ... ''')
        """
        from tvbo.utils import yaml_loader
        import yaml

        data_as_dict = yaml.safe_load(yaml_string) or {}
        return yaml_loader.loads(yaml.safe_dump(data_as_dict), target_class=cls)

    # ── Platform retrieval ────────────────────────────────────────

    TVBO_PLATFORM_URL = "https://tvbo.charite.de"

    @classmethod
    def from_platform(
        cls,
        name: str,
        base_url: str = TVBO_PLATFORM_URL,
    ) -> "SimulationExperiment":
        """Load a simulation experiment from the tvbo platform API.

        Fetches the full LinkML-valid YAML definition from the platform.

        Parameters
        ----------
        name : str
            Experiment label/ID (e.g., "RWW_BOLD_FC_Optimization").
        base_url : str
            Platform base URL.

        Returns
        -------
        SimulationExperiment
            Experiment loaded from the platform.
        """
        import requests

        api = f"{base_url.rstrip('/')}/api/v1/experiments"
        resp = requests.get(f"{api}/{name}/sidecar", params={"format": "yaml"})
        resp.raise_for_status()
        return cls.from_string(resp.text)

    @classmethod
    def list_platform_experiments(
        cls,
        base_url: str = TVBO_PLATFORM_URL,
    ) -> list:
        """List available experiments on the tvbo platform.

        Parameters
        ----------
        base_url : str
            Platform base URL.

        Returns
        -------
        list[dict]
            List of experiment summaries.
        """
        import requests

        api = f"{base_url.rstrip('/')}/api/v1/experiments"
        resp = requests.get(api)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def from_db(cls, name: str) -> "SimulationExperiment":
        """Load a SimulationExperiment by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("SimulationExperiment", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available experiments in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("SimulationExperiment")

    @classmethod
    def from_bids(
        cls,
        bids_dir: str,
        subject: str = "01",
        session: str | None = None,
        run_to_verify: bool = False,
    ) -> tuple["SimulationExperiment", TimeSeries]:
        """
        Load a SimulationExperiment and TimeSeries from a BIDS BEP034 dataset.

        This method ingests data exported via `to_bids()` and reconstructs:
        - The SimulationExperiment with model, network, and integration settings
        - The TimeSeries data (with 100% fidelity for HDF5 format)

        Automatically detects the time series format (HDF5, CIFTI, or TSV).

        Parameters
        ----------
        bids_dir : str
            Path to the root BIDS dataset directory (e.g., './derivatives/tvbo')
        subject : str
            Subject identifier (with or without 'sub-' prefix). Default: '01'
        session : str, optional
            Session identifier (with or without 'ses-' prefix).
            If not specified and sessions exist, uses the first one.
        run_to_verify : bool
            If True, re-run the simulation and compare with loaded TimeSeries.
            Useful for verifying reproducibility. Default: False.

        Returns
        -------
        tuple[SimulationExperiment, TimeSeries]
            The reconstructed experiment and time series data.

        Examples
        --------
        >>> # Load from BIDS
        >>> exp, ts = SimulationExperiment.from_bids("./derivatives/tvbo", subject="01")
        >>> print(ts.shape)
        (1000, 2, 68, 1)

        >>> # Verify reproducibility
        >>> exp, ts = SimulationExperiment.from_bids(
        ...     "./derivatives/tvbo",
        ...     subject="01",
        ...     run_to_verify=True
        ... )

        >>> # Access the experiment settings
        >>> print(exp.dynamics.name)
        'Generic2dOscillator'

        Notes
        -----
        - HDF5 format preserves full dimensionality with 100% fidelity
        - CIFTI/TSV formats reconstruct from per-state-variable files
        - Model parameters are restored from eq/ sidecar if available
        - Network connectivity is restored from net/ directory

        See Also
        --------
        to_bids : Export experiment to BIDS format
        """

        from tvbo.adapters.bids import (
            ingest_bids_session,
        )

        # Ingest all data from the BIDS session
        bids_data = ingest_bids_session(bids_dir, subject, session)

        # =====================================================================
        # 1. Reconstruct Network
        # =====================================================================
        network = None
        if bids_data["network"] is not None:
            net_data = bids_data["network"]
            labels = list(net_data["region_labels"]) if net_data["region_labels"] else None
            network = Network.from_matrix(
                weights=np.asarray(net_data["weights"]),
                lengths=(np.asarray(net_data["distances"]) if net_data.get("distances") is not None else None),
                labels=labels,
            )

            # Add coordinates if available
            if bids_data["coordinates"] is not None:
                coord_data = bids_data["coordinates"]
                if coord_data["centres"] is not None:
                    network.centres = coord_data["centres"]

        # =====================================================================
        # 2. Reconstruct Dynamics model
        # =====================================================================
        dynamics = None
        if bids_data["equations"] is not None:
            eq_data = bids_data["equations"]
            model_type = eq_data.get("model_type", "Generic2dOscillator")
            params = eq_data.get("parameters", {})

            # Try to load from ontology by name
            try:
                dynamics = Dynamics.from_ontology(model_type)
                # Apply stored parameters
                for param_name, param_value in params.items():
                    if hasattr(dynamics, "parameters") and param_name in dynamics.parameters:
                        dynamics.parameters[param_name].value = param_value
            except Exception:
                # Fallback: create minimal Dynamics
                dynamics = Dynamics(name=model_type)
        else:
            # Default dynamics
            dynamics = Dynamics.from_ontology("Generic2dOscillator")

        # =====================================================================
        # 3. Reconstruct Integration settings from sidecar provenance
        # =====================================================================
        integration = Integrator(method="Heun")

        if bids_data["timeseries"] is not None:
            ts_data = bids_data["timeseries"]

            # Get sampling from time series
            if ts_data["sample_period"] is not None:
                sample_period = ts_data["sample_period"]
                # Convert to step size (usually smaller)
                # Note: step_size may not equal sample_period if subsampling was used
                integration.step_size = sample_period

            # Check sidecars for provenance info
            for sidecar in ts_data.get("sidecars", []):
                provenance = sidecar.get("Provenance") or sidecar.get("SimulationProvenance")
                if provenance:
                    if "StepSize" in provenance and provenance["StepSize"]:
                        integration.step_size = float(provenance["StepSize"])
                    if "Duration" in provenance and provenance["Duration"]:
                        integration.duration = float(provenance["Duration"])
                    if "Integrator" in provenance and provenance["Integrator"]:
                        # Try to parse integrator method from string
                        int_str = str(provenance["Integrator"])
                        for method in ["Heun", "Euler", "RungeKutta"]:
                            if method.lower() in int_str.lower():
                                integration.method = method
                                break

        # =====================================================================
        # 4. Reconstruct TimeSeries
        # =====================================================================
        timeseries = None
        if bids_data["timeseries"] is not None:
            ts_data = bids_data["timeseries"]

            # Build labels_dimensions
            labels_dimensions = ts_data.get("labels_dimensions", {})
            if not labels_dimensions:
                labels_dimensions = {
                    "State Variable": ts_data["state_variables"],
                    "Space": ts_data["region_labels"],
                }

            timeseries = TimeSeries(
                time=ts_data["time"],
                data=ts_data["data"],
                network=network,
                title=f"Loaded from BIDS: sub-{subject}",
                sample_period=ts_data["sample_period"],
                labels_dimensions=labels_dimensions,
            )
            timeseries.sample_period_unit = ts_data.get("sample_period_unit", "ms")

            # Store source info
            timeseries._bids_source = {
                "bids_dir": str(bids_dir),
                "subject": subject,
                "session": session,
                "format": ts_data["format"],
            }

        # =====================================================================
        # 5. Create SimulationExperiment
        # =====================================================================
        experiment = cls(
            label=f"Loaded from BIDS: sub-{subject}",
            description=f"Experiment reconstructed from BIDS dataset at {bids_dir}",
            dynamics=dynamics,
            network=network,
            integration=integration,
            coupling=Coupling.from_ontology("Linear"),
        )

        # Store BIDS source info
        experiment._bids_source = {
            "bids_dir": str(bids_dir),
            "subject": subject,
            "session": session,
            "loaded_at": str(np.datetime64("now")),
        }

        # =====================================================================
        # 6. Verify reproducibility if requested
        # =====================================================================
        if run_to_verify and timeseries is not None:
            logger.info("Running simulation to verify reproducibility...")
            ts_rerun = experiment.run(format="jax")

            # Compare shapes
            if ts_rerun.shape != timeseries.shape:
                logger.warning(
                    "Shape mismatch! Loaded: %s, Rerun: %s", timeseries.shape, ts_rerun.shape
                )
            else:
                # Compare data
                max_diff = np.max(np.abs(np.asarray(ts_rerun.data) - np.asarray(timeseries.data)))
                if max_diff < 1e-6:
                    logger.info("Verification passed! Max difference: %.2e", max_diff)
                else:
                    logger.warning(
                        "Data differs. Max difference: %.2e "
                        "(may be expected if noise was used or parameters differ).",
                        max_diff,
                    )

        return experiment, timeseries

    @property
    def metadata(self):
        """The experiment itself, exposed as its own metadata container."""
        return self

    def symbolic(self, integrate=False, indexed=False, delays=False):
        """Symbolic representation of the full experiment equations.

        Produces different styles of mathematical output depending on the
        combination of flags:

        +----------+--------+-------+--------------------------------------------+
        | integrate| indexed| delays| Description                                |
        +==========+========+=======+============================================+
        | False    | False  | False | Dynamics separated, coupling terms as free |
        |          |        |       | symbols.  Coupling equations shown in      |
        |          |        |       | ``'coupling'`` dict.                       |
        +----------+--------+-------+--------------------------------------------+
        | True     | False  | False | Coupling substituted into dynamics.        |
        |          |        |       | State vars remain ``y0(t)``.               |
        +----------+--------+-------+--------------------------------------------+
        | False    | True   | False | State vars indexed ``y0_i(t)``.            |
        |          |        |       | Coupling shown separately with ``[i],[j]``. |
        +----------+--------+-------+--------------------------------------------+
        | True     | True   | False | Fully integrated with node indices.        |
        |          |        |       | Ready for network presentation.            |
        +----------+--------+-------+--------------------------------------------+
        | *        | (True) | True  | Like above but incoming states carry       |
        |          |        |       | ``y1[j, t - tau[i,j]]`` time delay.        |
        |          |        |       | ``delays=True`` implies ``indexed=True``.  |
        +----------+--------+-------+--------------------------------------------+

        Parameters
        ----------
        integrate : bool
            Substitute coupling expressions into state equations.
        indexed : bool
            Add node index ``_i`` to state / derived variables.
        delays : bool
            Show time delays on incoming coupling states.
            Implies ``indexed=True``.

        Returns
        -------
        dict
            Keys: ``'state'``, ``'coupling'``, ``'functions'``,
            ``'derived_parameters'``, ``'derived'``, ``'parameters'``.
        """
        import sympy as sp
        from sympy import Symbol, Function

        if delays:
            indexed = True

        # Start from local dynamics
        dyn_sym = self.dynamics.symbolic

        # Collect coupling input → Coupling object mapping
        coupling_map = {}
        net_coup = getattr(self.network, "coupling", {}) or {}
        dyn_ci = getattr(self.dynamics, "coupling_inputs", {}) or {}

        for ci_name in dyn_ci:
            if ci_name in net_coup:
                coupling_map[ci_name] = net_coup[ci_name]
            elif self.coupling and str(getattr(self.coupling, "name", "")) != "Linear":
                # Fallback: single experiment-level coupling for any input
                coupling_map[ci_name] = self.coupling

        # If no coupling to resolve, return dynamics as-is
        if not coupling_map:
            dyn_sym["coupling"] = {}
            return dyn_sym

        # Build coupling symbolic expressions
        t = Symbol("t")
        coupling_exprs = {}
        for ct_name, coup in coupling_map.items():
            coupling_exprs[ct_name] = coup.symbolic(delays=delays)

        # Substitution maps (built conditionally)
        subs_index = {}
        subs_coupling = {}

        # Node-index substitution: y0(t) → y0_i(t)
        if indexed:
            for sv_name in self.dynamics.state_variables:
                old_f = Function(str(sv_name))
                new_f = Function(str(sv_name) + "_i")
                subs_index[old_f(t)] = new_f(t)
            for dv_name in getattr(self.dynamics, "derived_variables", {}) or {}:
                old_f = Function(str(dv_name))
                new_f = Function(str(dv_name) + "_i")
                subs_index[old_f(t)] = new_f(t)

        # Coupling substitution: Symbol(ct_name) → Sum(...)
        if integrate:
            for ct_name, expr in coupling_exprs.items():
                subs_coupling[Symbol(str(ct_name))] = expr

        # Apply substitutions to all equation lists
        with sp.evaluate(False):

            def _apply_subs(eq):
                """Apply node indexing + coupling substitution to an Eq."""
                lhs = eq.lhs.subs(subs_index)
                rhs = eq.rhs.subs(subs_index).subs(subs_coupling)
                return sp.Eq(lhs, rhs)

            state_eqs = [_apply_subs(eq) for eq in dyn_sym["state"]]
            dv_eqs = [_apply_subs(eq) for eq in dyn_sym["derived"]]
            dp_eqs = list(dyn_sym["derived_parameters"])  # no state vars here
            func_eqs = list(dyn_sym["functions"])  # no state vars here

        return {
            "state": state_eqs,
            "coupling": coupling_exprs,
            "functions": func_eqs,
            "derived_parameters": dp_eqs,
            "derived": dv_eqs,
            "parameters": dyn_sym["parameters"],
        }

    @property
    def noise_sigma_array(self) -> np.ndarray:
        """Per-state-variable noise sigma values.

        Preference order:
        1) sigma from each state variable's noise.parameters["sigma"].value
        2) fallback to integration-level noise.parameters["sigma"].value
        3) default 0.0

        Returns an array with one entry per state variable in model order.
        """
        sigmas: list[float] = []

        for sv in self.dynamics.state_variables.values():
            sigma = 0.0
            if sv.noise:
                try:
                    sigma = float(sv.noise.parameters["sigma"].value)
                except Exception as e:
                    logger.debug("Error retrieving sigma for state variable %s: %s", sv.name, e)

            if sigma == 0.0:
                try:
                    integ_meta = getattr(self.integration, "metadata", None)
                    integ_noise = getattr(integ_meta, "noise", None)
                    inparams = getattr(integ_noise, "parameters", None)
                    if (
                        inparams is not None
                        and isinstance(inparams, dict)
                        and "sigma" in inparams
                        and hasattr(inparams["sigma"], "value")
                    ):
                        sigma = float(inparams["sigma"].value)
                except Exception as e:
                    logger.debug("Error retrieving integration-level sigma: %s", e)

            sigmas.append(float(sigma))

        if np.any(np.asarray(sigmas, dtype=float) > 0):
            self.integration.state_wise_sigma = sigmas
            if not self.integration.noise:
                from tvbo.classes.noise import Noise

                self.integration.noise = Noise()

        return np.asarray(sigmas, dtype=float)

    def __str__(self):
        return self.label if self.label else f"SimulationExperiment{self.id}"

    def __repr__(self):
        return self.__str__()

    # ---- Copy utilities ----
    def copy(self, **overrides) -> "SimulationExperiment":
        """Return a deep copy of this experiment.

        Use keyword overrides to set attributes on the returned copy.

        Errors are not swallowed; if a field can't be copied, an exception is raised.
        """
        new_obj = _copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    # Python copy protocol hooks
    def __copy__(self):
        # Keep Python's copy.copy semantics: shallow copy
        cls = self.__class__
        clone = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(clone, k, v)
        return clone

    def __deepcopy__(self, memo):
        import dataclasses

        cls = self.__class__
        # For dataclasses, we need to copy all fields, not just __dict__
        # __dict__ may not include fields that are still at their default values
        data = {}
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                value = getattr(self, field.name, None)
                data[field.name] = _copy.deepcopy(value, memo)
        else:
            # Fallback for non-dataclass
            for k, v in self.__dict__.items():
                data[k] = _copy.deepcopy(v, memo)

        # Create clone using proper constructor to ensure all defaults are set
        clone = cls(**data)
        memo[id(self)] = clone
        return clone

    def to_yaml(self, filepath: str | None = None, format: str = "tvbo") -> str:
        """Export the experiment to YAML format.

        Parameters
        ----------
        filepath : str, optional
            Path to write the YAML file. If None, returns the YAML string.
        format : str
            Output format: "tvbo" (default) or "pyrates".

        Returns
        -------
        str
            YAML string or filepath if written to file.
        """
        if format.lower() == "pyrates":
            from tvbo.codegen.pyrates import to_pyrates_yaml_string
            from tvbo.classes.dynamics import Dynamics as DynamicsClass

            # Get network
            network = getattr(self, "network", None)

            # Build dynamics dict: primary model + any network dynamics
            dynamics = self.dynamics

            # Convert datamodel Dynamics to full Dynamics class with methods
            if dynamics is not None and not hasattr(dynamics, "render_equation"):
                dynamics = DynamicsClass.from_datamodel(dynamics)

            # For network case, build full dynamics dict
            if network is not None:
                dynamics_dict = {}
                if dynamics:
                    dynamics_dict[dynamics.name] = dynamics

                # Add any additional dynamics from network.dynamics
                net_dynamics = getattr(network, "dynamics", None)
                if isinstance(net_dynamics, dict):
                    for name, dyn in net_dynamics.items():
                        if name not in dynamics_dict:
                            if dyn is not None and not hasattr(dyn, "render_equation"):
                                dynamics_dict[name] = DynamicsClass.from_datamodel(dyn)
                            else:
                                dynamics_dict[name] = dyn

                return to_pyrates_yaml_string(
                    dynamics=dynamics_dict,
                    network=network,
                    filepath=filepath,
                )
            else:
                return to_pyrates_yaml_string(
                    dynamics=dynamics,
                    network=network,
                    filepath=filepath,
                )
        else:
            import re
            from pathlib import Path as _Path
            from tvbo.utils import to_yaml as _to_yaml

            text = _strip_private_yaml_keys(_to_yaml(self, None))
            if filepath:
                _Path(filepath).write_text(text, encoding="utf-8")
                return filepath
            return text

    def render_yaml(self) -> str:
        """Deprecated Render the YAML representation as a string.
        Use to_yaml(filepath=None) instead.
        """
        return self.to_yaml(filepath=None)

    def setup_monitors(self, **kwargs):
        """Populate monitors in metadata from simple inputs or runtime wrappers."""
        from tvbo.classes import observation as monitoring

        monitors_ = kwargs.get("monitors", [])
        meta_list = []
        if isinstance(monitors_, monitoring.Monitor):
            meta_list = [monitors_]
        else:
            for m in monitors_:
                if isinstance(m, monitoring.Monitor):
                    meta_list.append(m)
                elif isinstance(m, dict):
                    meta_list.append(tvbo_datamodel.Observation(**m))
        if meta_list:
            self.monitors = meta_list

    def configure(self):
        """Resolve coupling declarations and normalize delay flags.

        Runs at the execution boundary and is backend-agnostic and idempotent:
        the resolved state persists on the experiment so YAML re-serialization
        and metadata export reflect what was actually executed. Delayed
        integration/coupling is disabled when the connectome has no path
        lengths or the conduction speed is infinite (all delays zero).
        """
        # Reconcile experiment / network / dynamics coupling declarations.
        # Runs at the execution boundary — backend-agnostic, idempotent, and
        # the resolved state persists on the experiment so YAML re-serialization
        # and metadata export reflect what was actually executed.
        _resolve_coupling(self)

        # Lower declarative stimulus events (target_variable -> name, target_regions
        # -> node indices) into the fields codegen consumes — same idempotent boundary.
        self._resolve_events()

        # Disable delayed logic if the connectome has no path lengths or conduction speed is infinite
        try:
            network = getattr(self, "network", None)
            if network is None:
                return

            # Get the network as a Network (it might already be one)
            if isinstance(network, Network):
                conn = network
            else:
                conn = Network(network)

            # Try to get delays from explicit edge delays or length / speed.
            try:
                delays = conn.calculate_delays()
            except Exception:
                delays = None

            if delays is None or np.allclose(np.nan_to_num(delays, nan=0.0), 0):
                if getattr(self, "integration", None) is not None:
                    self.integration.delayed = False
                if getattr(self, "coupling", None) is not None:
                    self.coupling.delayed = False
        except Exception as e:
            # Best-effort; keep defaults if anything goes wrong
            import warnings

            warnings.warn(f"Could not configure delays: {e}")

    def add_stimulus(self, stimulus):
        """Attach a stimulus to the experiment.

        Args:
            stimulus: Either a `Stimulus` instance, or a name/ontology class
                (`str` or `owlready2.ThingClass`) resolved to a `Stimulus`
                via `Stimulus.from_ontology`.
        """
        import owlready2 as owl

        from tvbo.classes import perturbation

        if isinstance(stimulus, perturbation.Stimulus):
            self.stimulation = stimulus
        elif isinstance(stimulus, str) or isinstance(stimulus, owl.ThingClass):
            self.stimulation = perturbation.Stimulus.from_ontology(stimulus)

    def collect_state(self, initial_conditions: TimeSeries | None = None):
        """Assemble a `SimulationState` pytree for the JAX-style backends.

        Gathers the parameter collection (expanding coupling parameters with
        shape annotations and wrapping array-valued parameters as ndarrays so
        the JAX backend receives real arrays), the network, integration
        step/step-count, and the noise wrapper into a single state object.

        Args:
            initial_conditions: History to seed the state with. When omitted,
                initial conditions are collected from the experiment via
                `collect_initial_conditions`.

        Returns:
            A `SimulationState` carrying initial conditions, network, `dt`,
            step count, noise, and parameters.
        """
        _ = self.noise_sigma_array
        parameters = self.get_parameters_collection(
            keys_to_exclude=[
                "derived_parameters",
                "conduction_speed",
                "coupling_inputs",
            ]
        )
        # Expand coupling parameters with shape annotations like "(N, N)" or "(N,)"
        self._expand_coupling_parameter_shapes(parameters)
        # Wrap array-valued (per-mode) parameters as ndarrays so the JAX backend
        # receives jnp arrays. Bare Python lists would survive convert_dtype's
        # tree_map (which descends into the list and only converts the scalar
        # elements, leaving a list[Array]) and then break `scalar * param`
        # arithmetic in the generated dfun. Mirrors render_jax_default (tvboptim).
        self._arrayify_parameter_values(parameters)

        state = SimulationState(
            initial_conditions=(initial_conditions if initial_conditions is not None else self.collect_initial_conditions()),
            network=self.network,
            dt=self.integration.step_size,
            nt=int(np.ceil(self.integration.duration / self.integration.step_size)),
            # Provide a JAX-pytree-friendly Noise wrapper (or None)
            noise=self.integration.noise_wrapper,
            parameters=parameters,
            stimulus=None,
            monitor_parameters=None,
        )
        # Attach state variable names for ergonomic noise setters
        try:
            state._svar_names = list(self.dynamics.state_variables.keys())
        except Exception:
            pass
        return state

    def _expand_coupling_parameter_shapes(self, parameters: Bunch) -> None:
        """Expand coupling parameters that have shape annotations like (N, N) or (N,)."""
        if not hasattr(parameters, "coupling") or self.network is None:
            return

        N = self.network.number_of_nodes
        coupling_params = parameters.coupling

        # Get the coupling parameter definitions with shape info
        if self.coupling is None:
            return

        for param_name, param_obj in (self.coupling.parameters or {}).items():
            if param_name not in coupling_params:
                continue

            shape_str = getattr(param_obj, "shape", None)
            if not shape_str:
                continue

            current_value = coupling_params[param_name]

            # Parse shape string and expand
            if shape_str == "(N, N)" or shape_str == "(N,N)":
                # Expand scalar to NxN matrix
                if np.isscalar(current_value) or (hasattr(current_value, "shape") and current_value.shape == ()):
                    coupling_params[param_name] = np.full((N, N), float(current_value))
            elif shape_str == "(N,)" or shape_str == "(N)":
                # Expand scalar to N-vector
                if np.isscalar(current_value) or (hasattr(current_value, "shape") and current_value.shape == ()):
                    coupling_params[param_name] = np.full((N,), float(current_value))

    def _arrayify_parameter_values(self, parameters: Bunch) -> None:
        """Convert list/tuple-valued parameters in the collection to ``np.array``.

        Array-valued constants (e.g. the Stefanescu-Jirsa per-mode coupling
        vectors/matrices) arrive as nested Python lists from the metadata. The
        JAX backend evaluates the dfun with these values as ``_p`` leaves, and
        ``scalar * list`` raises ``TypeError`` under JAX. Wrapping them as
        ndarrays makes ``SimulationState.convert_dtype`` emit real ``jnp`` arrays
        (an ndarray is a single pytree leaf, whereas a list is traversed
        element-wise). Recurses through the nested ``dynamics``/``coupling``
        Bunches; scalars and existing arrays are left untouched.
        """
        from tvbo.utils import is_array_valued

        def _walk(container):
            for key, value in list(container.items()):
                if isinstance(value, dict):
                    _walk(value)
                elif is_array_valued(value) and not isinstance(value, np.ndarray):
                    try:
                        container[key] = np.asarray(value)
                    except (TypeError, ValueError):
                        pass

        _walk(parameters)

    def execute(self, format="tvb", **kwargs):
        """Render and build the executable object for a backend without running it.

        Calls `configure` to normalize coupling/delay metadata, renders the
        backend code, and executes it to produce a ready-to-run artefact. The
        return type depends on `format`:

        - `"tvb"`: a configured `tvb` `Simulator`.
        - `"tvboptim"` / `"tvb-optim"`: a `Bunch` namespace of the generated
          functions.
        - `"autodiff"` / `"jax"`: the JAX `kernel` callable (JIT-compiled
          unless `jit=False`).
        - `"pde"` / `"pde-fem"` / `"pde-python"`: the generated namespace.

        Args:
            format: Backend identifier selecting what to build.
            **kwargs: Backend-specific options. Forwarded to the TVB simulator
                factory, to `render_code`, or interpreted as JAX flags
                (`jit`, `_return_namespace`) depending on `format`.

        Returns:
            The backend-specific executable object described above.

        Raises:
            ValueError: If `format` is not one of the supported backends.
        """
        # Ensure coupling resolution / delay flags are normalized before any
        # backend code generation. Idempotent — safe if called twice via run().
        self.configure()
        if format.lower() == "tvb":
            code = self.render_code(format=format)
            namespace = templater.exec_globals
            exec(code, namespace)
            sim = namespace["define_simulation"](connectivity=self.network.execute("tvb"), **kwargs)
            sim.initial_conditions = self.collect_initial_conditions().data
            sim.configure()
            return sim

        elif format.lower() in ["tvboptim", "tvb-optim"]:
            # Return namespace with all generated functions for tvboptim workflows
            # This allows: ns = exp.execute('tvboptim')
            #              spectrum, cauchy_pdf = ns.spectrum, ns.cauchy_pdf
            namespace = {}
            code = self.render_code("tvboptim")
            exec(code, namespace)
            return Bunch(**namespace)

        elif format.lower() in ["autodiff", "jax"]:
            _return_namespace = kwargs.pop("_return_namespace", False)
            jit = kwargs.get("jit", True)
            code = self.render_code(format=format, **kwargs)
            # Use a fresh namespace each time to avoid JAX tracer leaks
            # between repeated executions (stale tracers in shared globals
            # cause UnexpectedTracerError on re-runs).
            namespace = {"TimeSeries": TimeSeries}
            exec(code, namespace)
            jax_model = namespace["kernel"]
            if jit:
                jax_model = jax.jit(jax_model)
            if _return_namespace:
                return jax_model, namespace
            return jax_model

        elif format.lower() in ["pde", "pde-fem", "pde-python"]:
            code = self.render_code(format="pde")
            namespace = templater.exec_globals
            exec(code, namespace)
            return namespace

        else:
            raise ValueError(f"Format {format} not supported. Valid formats: tvb, tvboptim, jax.")

    def _locate_source_result(self, results_root, source_id):
        """Path to the ``from_experiment`` source run's saved result HDF5.

        Globs ``results_root`` (or cwd) by the ``exp-<id>_`` stem, skipping the
        network sidecar. Raises if the source hasn't been run yet. Shared by the
        seed / branch / parameter resolvers.
        """
        from pathlib import Path

        root = Path(results_root) if results_root else Path.cwd()
        cands = [p for p in root.glob(f"**/*exp-{source_id}_*.h5") if "network" not in p.name]
        if not cands:
            raise FileNotFoundError(
                f"initial_state.from_experiment: no saved result for experiment {source_id} "
                f"under {root} (looked for '*exp-{source_id}_*.h5'). Run experiment "
                f"{source_id} first so its result is available."
            )
        return sorted(cands)[0]

    def _resolve_from_experiment_seed(self, results_root=None):
        """Load the operating point for ``initial_state.method == from_experiment``.

        The source experiment exposes its settled per-node state as observations
        named ``<state_variable>_final`` (e.g. ``theta_final``). This locates that
        experiment's saved result under ``results_root`` — matched by the
        ``exp-<id>_`` file stem, so the output-directory layout (``results/2``,
        ``output/nc/exp2``, …) does not matter — and reads one ``<sv>_final`` per
        state variable of *this* experiment. Everything is keyed by name/dim, never
        positional: the result is a ``{state_variable_name: (n_nodes,)}`` dict that
        the generated code places into its own canonical rows. For a swept source
        (an adiabatic ramp) the operating point is the last recorded point
        (``source_point``; default ``'endpoint'``).

        Returns the name-keyed IC dict, or ``None`` when this experiment does not
        use ``from_experiment``.
        For ``source_point == 'branch'`` this returns ``None`` — the whole
        recorded branch is a per-cell seed, resolved by
        :meth:`_resolve_from_experiment_branch`.
        """
        return self._read_source_final(results_root, branch=False)

    def _resolve_from_experiment_branch(self, results_root=None):
        """Load the WHOLE recorded branch for ``source_point == 'branch'``.

        Where :meth:`_resolve_from_experiment_seed` picks one settled point, this
        keeps the source run's swept dimension, so every ``<sv>_final`` observation
        loads as ``(n_cells, n_nodes)`` and the swept-axis coordinate supplies the
        per-cell parameter values ``(n_cells,)``. An independent exploration can then
        restart an analysis (Lyapunov exponents, Jacobian/Floquet spectra, basin
        sampling, perturbation kicks, first-passage, …) at every branch point in
        parallel — the per-cell, cross-experiment counterpart of the single operating
        point above. Returns ``{axis_name, axis_values, seeds, n_cells}`` (``seeds`` a
        name-keyed ``{sv: (n_cells, n_nodes)}`` dict), or ``None`` when this experiment
        does not use ``from_experiment`` with ``source_point='branch'``.
        """
        return self._read_source_final(results_root, branch=True)

    def _read_source_final(self, results_root=None, *, branch: bool):
        """Shared loader behind the two ``from_experiment`` resolvers.

        Locates the source run and reads the ``<sv>_final`` settled-state
        observations, keyed by state-variable name (never positional). With
        ``branch=False`` it selects a single point (``source_point``; default
        ``endpoint``) → ``{sv: (n_nodes,)}``; with ``branch=True`` it keeps the
        swept dimension → the branch dict described in
        :meth:`_resolve_from_experiment_branch`. Returns ``None`` when the
        experiment's ``source_point`` mode does not match ``branch``.
        """
        ini = getattr(self, "initial_state", None)
        if ini is None or str(getattr(ini, "method", "") or "") != "from_experiment":
            return None
        point = str(getattr(ini, "source_point", "") or "endpoint")
        # Each resolver owns exactly one mode, so the run() call site can invoke both.
        if branch != (point == "branch"):
            return None
        src = getattr(ini, "source_experiment", None)
        if src is None:
            raise ValueError("initial_state.method=from_experiment requires source_experiment")
        source_id = int(getattr(src, "id", src))

        # State variables of THIS experiment → which <sv>_final observations to
        # load. Keyed by name; the generated code places each into its own row.
        svs = self.dynamics.state_variables
        sv_items = svs.items() if hasattr(svs, "items") else [(getattr(s, "name", None), s) for s in svs]
        sv_names = [getattr(sv, "name", None) or key for key, sv in sv_items]

        src_h5 = self._locate_source_result(results_root, source_id)
        n_nodes = len(self.network.nodes) if self.network.nodes else None

        def _node_dim(da):
            if n_nodes is not None:
                for d in da.dims:
                    if da.sizes[d] == n_nodes:
                        return d
            for d in da.dims:
                if "node" in str(d).lower():
                    return d
            return da.dims[-1]

        def _final_key(ds, sv):
            target = f"{sv}_final"
            for k in ds.data_vars:
                if k == target or k.endswith(f"__{target}"):
                    return k
            raise KeyError(
                f"initial_state.from_experiment: source experiment {source_id} does not "
                f"record '{target}'. The source must expose its settled state as an "
                f"observation named '{sv}_final' for each state variable."
            )

        ds = xr.open_dataset(src_h5, engine="h5netcdf")
        try:
            if not branch:
                seed = {}
                sel = -1 if (point in ("", "endpoint")) else int(point)
                for sv in sv_names:
                    da = ds[_final_key(ds, sv)]
                    for d in [d for d in da.dims if d != _node_dim(da)]:
                        da = da.isel({d: sel})
                    seed[sv] = jnp.asarray(np.asarray(da.values).ravel())
                return seed

            seeds, axis_values, axis_name, n_cells = {}, None, None, None
            for sv in sv_names:
                da = ds[_final_key(ds, sv)]
                nd = _node_dim(da)
                sweep = [d for d in da.dims if d != nd]
                if len(sweep) != 1:
                    raise ValueError(
                        "initial_state.from_experiment source_point='branch' expects the source "
                        f"'{sv}_final' to have exactly one swept dimension besides nodes; got "
                        f"dims {tuple(da.dims)}. Only a single-axis branch can be restarted per cell."
                    )
                sd = sweep[0]
                da = da.transpose(sd, nd)                       # (n_cells, n_nodes), by name
                seeds[sv] = jnp.asarray(np.asarray(da.values))
                if axis_values is None:
                    axis_name = str(sd)
                    if sd not in da.coords:
                        raise ValueError(
                            "initial_state.from_experiment source_point='branch': the source's "
                            f"'{sv}_final' has no coordinate on its swept dim '{sd}', so the per-cell "
                            "parameter values are unavailable — the restart would run at wrong values. "
                            "The source experiment must record its swept axis (a warm-start scan does)."
                        )
                    axis_values = np.asarray(da.coords[sd].values)
                    n_cells = int(da.sizes[sd])
                elif int(da.sizes[sd]) != n_cells:
                    raise ValueError(
                        "initial_state.from_experiment source_point='branch': state variables "
                        "disagree on the number of recorded branch points."
                    )
        finally:
            ds.close()
        return {
            "axis_name": axis_name,
            "axis_values": jnp.asarray(np.asarray(axis_values, dtype=float)),
            "seeds": seeds,
            "n_cells": n_cells,
        }

    def _resolve_from_experiment_params(self, results_root=None):
        """Load model PARAMETERS sourced from the from_experiment operating point.

        A parameter (dynamics **or** coupling) that declares ``measure: <name>`` in a
        ``method=from_experiment`` experiment takes its value from the source run's
        operating point. Two flavours share one path: a recorded per-node observation
        (e.g. ``g`` from a ``control_mask``), or a tuned free parameter the source fit
        persisted as ``estimate__<name>`` (warm-start / prior location, e.g. ``wLRE``).
        Per-node **vectors and per-edge matrices** are both handled; when the source
        array carries node-label coordinates (``node`` / ``node_i``+``node_j``) it is
        reconciled to THIS experiment's network **by label** — alias-aware, on every
        node axis — via the same `.sel` path the dataset targets use. An unlabelled
        source array is taken in model order (legacy behaviour). Returns
        ``{param_name: ndarray}``, or ``None`` when no parameter sources a value.
        """
        ini = getattr(self, "initial_state", None)
        if ini is None or str(getattr(ini, "method", "") or "") != "from_experiment":
            return None

        # Parameters that source from the operating point — dynamics + coupling.
        def _items(params):
            if params is None:
                return []
            return list(params.items()) if hasattr(params, "items") \
                else [(getattr(p, "name", None), p) for p in params]

        def _couplings(obj):
            if obj is None:
                return []
            if hasattr(obj, "values"):
                return list(obj.values())
            return list(obj) if isinstance(obj, (list, tuple)) else [obj]

        wanted: dict = {}
        param_sets = [getattr(getattr(self, "dynamics", None), "parameters", None)]
        net = getattr(self, "network", None)
        for c in _couplings(getattr(net, "coupling", None)) + _couplings(getattr(self, "coupling", None)):
            param_sets.append(getattr(c, "parameters", None))
        for ps in param_sets:
            for key, p in _items(ps):
                m = str(getattr(p, "measure", "") or "")
                nm = getattr(p, "name", None) or key
                if m and nm and nm not in wanted:
                    wanted[nm] = m
        if not wanted:
            return None

        src = getattr(ini, "source_experiment", None)
        if src is None:
            raise ValueError("initial_state.method=from_experiment requires source_experiment")
        source_id = int(getattr(src, "id", src))
        src_h5 = self._locate_source_result(results_root, source_id)

        n_nodes = len(self.network.nodes) if self.network.nodes else None
        point = str(getattr(ini, "source_point", "") or "endpoint")
        # 'branch' has no single point for a parameter value; fall back to the settled
        # endpoint rather than crash on int('branch').
        sel = -1 if point in ("", "endpoint", "branch") else int(point)

        def _node_dims(da):
            labelled = [d for d in da.dims if str(d).startswith("node")]
            if labelled:
                return labelled
            if n_nodes is not None:
                by_size = [d for d in da.dims if da.sizes[d] == n_nodes]
                if by_size:
                    return by_size[:1]
            return [da.dims[-1]]

        def _measure_key(ds, measure):
            for k in ds.data_vars:
                if k == measure or k.endswith(f"__{measure}"):
                    return k
            raise KeyError(
                f"initial_state.from_experiment: source experiment {source_id} does not record "
                f"'{measure}' (looked for an observation or estimate__{measure}). Have the source "
                f"record it — a per-node observation, or a tuned free parameter."
            )

        _recon: dict = {}   # cache the network-invariant alias map + model labels
        def _reconcile(da, node_dims):
            # Relabel each LABELLED node axis source->canonical, then select this model's
            # node order (both axes for a matrix). Unlabelled axes are assumed already in
            # model order (legacy). Reuses the dataset-target `.sel` reconcile path. The
            # alias map + labels depend only on self.network, so compute them once (lazily,
            # to keep the all-unlabelled case free of an atlas load) and reuse across params.
            labelled = [d for d in node_dims if d in da.coords]
            if not labelled:
                return np.asarray(da.values)
            if not _recon:
                _recon["amap"] = self.network.region_alias_map()
                _recon["labels"] = self._resolve_model_node_labels()
            for d in labelled:
                da = da.assign_coords({d: [_recon["amap"].get(str(l), str(l)) for l in da[d].values]})
                da = da.sel({d: _recon["labels"]})
            return np.asarray(da.values)

        ds = xr.open_dataset(src_h5, engine="h5netcdf")
        try:
            out = {}
            for pname, measure in wanted.items():
                da = ds[_measure_key(ds, measure)]
                node_dims = _node_dims(da)
                for d in [d for d in da.dims if d not in node_dims]:
                    da = da.isel({d: sel})     # settle any swept dims to the operating point
                out[pname] = jnp.asarray(_reconcile(da, node_dims))
        finally:
            ds.close()
        return out

    def run(self, format="tvboptim", initial_conditions=None, results_root=None, **kwargs):
        """Configure, build, and run the experiment on a backend.

        Dispatches on `format` to the corresponding backend (`tvb`, `tvboptim`,
        `jax`/`autodiff`, `cuda`, `python`, or `pde`), executes the simulation,
        and wraps the output in an [`ExperimentResult`](../data/types.qmd).

        Args:
            format: Backend identifier selecting how to run the experiment.
            initial_conditions: Optional history to seed the simulation
                (used by the JAX backend); defaults to conditions collected
                from the experiment.
            results_root: Directory under which sibling experiments' saved
                results are searched when this experiment's
                `initial_state.method == from_experiment` (see
                `_resolve_from_experiment_seed`). Defaults to the current
                directory; the CLI passes the run's output-directory parent.
            **kwargs: Backend-specific run options. A `duration` value is
                applied to the integration settings before running; other
                keys (e.g. `benchmark`, `mode`) are passed through to the
                backend runner.

        Returns:
            An `ExperimentResult` holding the integrated time series and any
            observations, with `source` set to this experiment.
        """
        # Make tvbo progress visible for interactive/script runs without
        # clobbering an app's own logging — the same switch the CLI uses, so
        # ``exp.run(...)`` and ``tvbo run`` log identically. No-op if the
        # embedding application already configured logging.
        ensure_configured()

        if "duration" in kwargs:
            self.integration.duration = kwargs.pop("duration")

        # The active subject (set by the per-subject workflow fan-out) selects
        # which per-subject empirical target this run resolves. Popped here so it
        # never leaks into a backend runner's kwargs.
        active_subject = kwargs.pop("active_subject", None)
        if active_subject is None:
            active_subject = getattr(getattr(self, "dataset", None), "active_subject", None)
        self._active_subject = str(active_subject) if active_subject is not None else None

        self.configure()
        Bunch()

        if format.lower() == "tvb":
            _random_ic = kwargs.pop("random_initial_conditions", False)
            if _random_ic:
                import warnings

                warnings.warn(
                    "random_initial_conditions=True is deprecated. Set distribution on state variables instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            initial_conditions = self.collect_initial_conditions(random=_random_ic)
            simulator_ = self.execute()
            simulator_.initial_conditions = initial_conditions.data
            simulator_.configure()
            simres = simulator_.run(**kwargs)

            sv_names = list(simulator_.model.variables_of_interest)
            region_labels = [str(r) for r in simulator_.connectivity.region_labels]

            observations = {}
            sim_result = None
            for m, (tv, xv) in zip(simulator_.monitors, simres):
                m_name = m.title.split(" ")[0]
                data_np = np.asarray(xv)
                dims = ["time", "variable", "node", "mode"][: data_np.ndim]
                coords = {
                    "time": np.asarray(tv),
                    "variable": sv_names,
                    "node": region_labels,
                }
                if "mode" in dims:
                    coords["mode"] = list(range(data_np.shape[3]))
                da = xr.DataArray(data=data_np, dims=dims, coords=coords)
                if m_name == "Raw":
                    sim_result = SimulationResult(data=da)
                else:
                    # Key by the monitor's canonical name (its title's first
                    # token already equals the observation name) so tvb results
                    # expose the same keys as the jax/tvboptim backends, e.g.
                    # result.observations.BOLD_TVB rather than a lower-cased key.
                    observations[m_name] = SimulationResult(data=da)

            if sim_result is None:
                # Fallback if no Raw monitor — use first
                first_m, (tv, xv) = list(zip(simulator_.monitors, simres))[0]
                data_np = np.asarray(xv)
                dims = ["time", "variable", "node", "mode"][: data_np.ndim]
                coords = {
                    "time": np.asarray(tv),
                    "variable": sv_names,
                    "node": region_labels,
                }
                if "mode" in dims:
                    coords["mode"] = list(range(data_np.shape[3]))
                da = xr.DataArray(data=data_np, dims=dims, coords=coords)
                sim_result = SimulationResult(data=da)

            # Wrap in Bunch so tvb results support dot-access
            # (result.observations.BOLD_TVB), matching the jax/tvboptim backends;
            # assigning after construction bypasses SimulationResult's own
            # normalization.
            sim_result.observations = Bunch(observations)
            return ExperimentResult(
                integration=sim_result,
                source=self,
                name=self.label,
            )

        elif format.lower() in ["tvboptim", "tvb-optim"]:
            import time

            benchmark = kwargs.pop("benchmark", False)
            timings = Bunch() if benchmark else None

            # Get the namespace (reuse execute to avoid code duplication)
            if benchmark:
                t0 = time.perf_counter()
            ns = self.execute("tvboptim")
            if benchmark:
                timings.code_generation = time.perf_counter() - t0

            # Mode defaults to 'all' - run complete workflow
            mode = kwargs.pop("mode", "all")

            # Build node labels from network.nodes (used as xarray 'node' coord)
            node_labels = [n.label for n in self.network.nodes] if self.network.nodes else None

            delay_matrix = self.network.calculate_delays() if getattr(self.coupling, "delayed", False) else None

            # Resolve network-sourced observations (e.g. fc_target <- empirical
            # FC) to matrices and pass them alongside weights/distances. Only
            # set when present, so experiments without network observations are
            # unaffected.
            net_obs = self.resolve_network_observations()
            # Per-subject dataset-sourced targets (e.g. this subject's empirical
            # FC), reconciled to the model's node labels, merge into the same
            # network_observations injection.
            if getattr(self, "_active_subject", None):
                for obs_name, da in self.resolve_dataset_observations(self._active_subject).items():
                    net_obs = dict(net_obs or {})
                    net_obs[obs_name] = np.asarray(da.values)
            if net_obs:
                kwargs.setdefault("network_observations", net_obs)

            # Resolve InitialState.from_experiment -> the operating point another
            # experiment already reached, loaded from its saved run and handed to the
            # generated code as seed_dynamics. No-op unless method == from_experiment.
            _seed = self._resolve_from_experiment_seed(results_root)
            if _seed is not None:
                kwargs.setdefault("seed_dynamics", _seed)

            # Parameters sourced from the same operating point (Parameter.measure, e.g.
            # a control mask g <- source's control_mask), injected as seed_params.
            _pseed = self._resolve_from_experiment_params(results_root)
            if _pseed is not None:
                kwargs.setdefault("seed_params", _pseed)

            # source_point='branch': the source run's WHOLE branch is a per-cell seed
            # (axis values + settled state per point), handed over as branch_seed so an
            # independent exploration can restart an analysis at every branch point.
            _branch = self._resolve_from_experiment_branch(results_root)
            if _branch is not None:
                kwargs.setdefault("branch_seed", _branch)

            # Run the experiment with optional per-step timing
            if benchmark:
                # Run with detailed timing
                t0 = time.perf_counter()
                results = ns.run_experiment(
                    weights=self.network.weights,
                    distances=self.network.distances,
                    delays=delay_matrix,
                    region_labels=node_labels,
                    mode=mode,
                    **kwargs,
                )
                # Wait for JAX async dispatch to complete
                jax.block_until_ready(results.result.data if hasattr(results, "result") else results)
                timings.total = time.perf_counter() - t0

                # Add timings to results and wrap in ExperimentResult
                results.timings = timings
                return ExperimentResult(results, experiment_name=self.label, source=self)
            else:
                raw_results = ns.run_experiment(
                    weights=self.network.weights,
                    distances=self.network.distances,
                    delays=delay_matrix,
                    region_labels=node_labels,
                    mode=mode,
                    **kwargs,
                )
                return ExperimentResult(raw_results, experiment_name=self.label, source=self)

        elif format.lower() in ["autodiff", "jax"]:
            state = self.collect_state(initial_conditions=initial_conditions)
            if kwargs.get("enable_x64", True):
                jax.config.update("jax_enable_x64", True)
                state = state.convert_dtype(target_dtype=jnp.float64)
            else:
                jax.config.update("jax_enable_x64", False)
                state = state.convert_dtype(target_dtype=jnp.float32)

            jax_model, _jax_ns = self.execute(format="jax", _return_namespace=True, **kwargs)
            _run_fn = _jax_ns.get("run_experiment")
            if _run_fn is not None:
                # Template builds ExperimentResult directly (mirrors tvboptim backend).
                result = _run_fn(state)
                result.source = self
                result.name = result.name or self.label
                return result
            # Fallback: IC-finding kernel returns a tuple, not an ExperimentResult.
            ts = jax_model(state)
            return ExperimentResult.from_timeseries(ts, source=self, name=self.label)

        elif format.lower() == "cuda":
            from tvbo.codegen.cuda import run_cuda

            cuda_result = run_cuda(self, **kwargs)
            if isinstance(cuda_result, TimeSeries):
                return ExperimentResult.from_timeseries(cuda_result, source=self, name=self.label)
            return ExperimentResult(integration=cuda_result, source=self, name=self.label)

        elif format.lower() == "python":
            bnm = _Network(Network(self.network))
            bnm.add_local_model(self.dynamics)
            bnm.add_coupling(self.coupling)

            ts = bnm.run(
                duration=kwargs.get("duration", self.integration.duration),
                dt=self.integration.step_size,
            )
            return ExperimentResult.from_timeseries(ts, source=self, name=self.label)

        elif format.lower() in ["pde", "pde-fem", "pde-python"]:
            ns = self.execute(format="pde")
            solve = ns.get("solve_pde")
            viz = ns.get("visualize")
            meta = ns.get("meta")

            if solve is None:
                raise RuntimeError("PDE backend did not expose solve_pde.")

            steps = kwargs.get("steps", None)
            out = kwargs.get("out", None)

            # Optional node-based initial condition and source
            u0_override = None
            src = kwargs.get("source", None)

            # Accept explicit u0 or u0_override kwarg
            u0_kw = kwargs.get("u0", None)
            if u0_kw is None:
                u0_kw = kwargs.get("u0_override", None)

            if u0_kw is not None:
                arr = np.asarray(u0_kw, dtype=float).ravel()
                if meta and isinstance(meta, dict):
                    ndofs = int(meta.get("ndofs", arr.size))
                    if arr.size != ndofs:
                        raise ValueError(f"u0 length {arr.size} != ndofs {ndofs}")
                u0_override = arr
            elif initial_conditions is not None:
                # Support TimeSeries or ndarray as initial conditions
                if isinstance(initial_conditions, TimeSeries):
                    arr = np.asarray(
                        initial_conditions.data.isel(time=-1, variable=0).values,
                        dtype=float,
                    ).ravel()
                else:
                    arr = np.asarray(initial_conditions, dtype=float).ravel()
                if meta and isinstance(meta, dict):
                    ndofs = int(meta.get("ndofs", arr.size))
                    if arr.size != ndofs:
                        raise ValueError(f"initial_conditions length {arr.size} != ndofs {ndofs}")
                u0_override = arr

            # Always compute and return a full TimeSeries (save_timeseries=True)
            solve_kwargs = dict(save_timeseries=True, outpath=out)
            if u0_override is not None:
                solve_kwargs["u0_override"] = u0_override
            if src is not None:
                solve_kwargs["source"] = src

            if steps is not None:
                solve_kwargs["steps"] = int(steps)
            u, U = solve(**solve_kwargs)

            if kwargs.get("visualize", False) and viz is not None:
                try:
                    viz(u)
                except Exception:
                    pass

            T = U.shape[0] if U is not None else 1
            t = np.arange(T) * float(meta.get("dt", 1.0))
            data_raw = U if U is not None else u[np.newaxis, :]
            # Reshape: (time, 1_sv, region) — no singleton mode dim
            n_region = data_raw.shape[-1] if data_raw.ndim >= 2 else data_raw.size
            data_np = data_raw.reshape(T, 1, n_region)

            sv_name = str(meta.get("unknown", "u"))
            da = xr.DataArray(
                data=data_np,
                dims=["time", "variable", "node"],
                coords={
                    "time": t,
                    "variable": [sv_name],
                    "node": [str(i) for i in range(n_region)],
                },
            )
            sim = SimulationResult(data=da)
            return ExperimentResult(
                integration=sim,
                source=self,
                name=self.label,
            )

        elif format.lower() == "pyrates":
            return self._run_pyrates(**kwargs)

        elif format.lower() in ["networkdynamics", "nd", "networkdynamics.jl"]:
            return self._run_networkdynamics(**kwargs)

        elif format.lower() in ["mtk", "modelingtoolkit", "modelingtoolkit.jl"]:
            return self._run_modelingtoolkit(**kwargs)

        elif format.lower() in [
            "bifurcationkit",
            "bifurcationkit.jl",
            "bifurcation",
            "bifurcation-julia",
        ]:
            return self._run_bifurcation(**kwargs)

        elif format.lower() in [
            "pyrates-bifurcation",
            "pyrates-bif",
            "pycobi",
            "bifurcation-pyrates",
        ]:
            return self._run_pyrates_bifurcation(**kwargs)

        elif format.lower() in [
            "numcont",
            "auto",
            "auto-07p",
            "auto07p",
            "bifurcation-auto7p",
            "bifurcation-numcont",
        ]:
            return self._run_numcont(**kwargs)

        elif format.lower() in [
            "julia",
            "diffeq",
            "differentialequations",
            "differentialequations.jl",
        ]:
            return self._run_julia(**kwargs)

        elif format.lower() in ["neuroml", "nml", "lems"]:
            from tvbo.adapters.neuroml import NeuroMLAdapter

            return NeuroMLAdapter(self).run(**kwargs)

        elif format.lower() in ["brian2", "brian"]:
            from tvbo.adapters.brian2 import Brian2Adapter

            return Brian2Adapter(self).run(**kwargs)

        else:
            raise ValueError(
                f"Format {format} not supported. Valid formats: tvb, jax, python, pyrates, "
                "networkdynamics, mtk, modelingtoolkit, bifurcationkit.jl, "
                "pyrates-bifurcation, julia, neuroml, nml, lems, brian2"
            )

    def _run_pyrates(
        self,
        solver: str | None = None,
        inputs: dict | None = None,
        outputs: list[str] | None = None,
        matrix_edge_threshold: int = 100,
        **kwargs,
    ) -> ExperimentResult:
        """Run simulation using PyRates backend.

        Parameters
        ----------
        solver : str, optional
            ODE solver: "euler", "heun", "scipy". Defaults to mapped integration.method.
        inputs : dict, optional
            External inputs as {node/op/var: array} or will be auto-generated.
        outputs : list[str], optional
            Variables to monitor. If None, monitors all state variables.
        matrix_edge_threshold : int, optional
            For networks with N > threshold nodes, use add_edges_from_matrix instead
            of YAML edges for efficiency. Default is 100.
        **kwargs
            Additional kwargs passed to circuit.run().

        Returns
        -------
        ExperimentResult
        """
        from tvbo.adapters.pyrates import PyRatesAdapter

        adapter = PyRatesAdapter(self)
        return adapter.run(
            solver=solver,
            inputs=inputs,
            outputs=outputs,
            matrix_edge_threshold=matrix_edge_threshold,
            **kwargs,
        )

    def _run_networkdynamics(self, **kwargs) -> ExperimentResult:
        """Run simulation using NetworkDynamics.jl via pyjulia."""
        from tvbo.adapters.networkdynamics import NetworkDynamicsAdapter

        adapter = NetworkDynamicsAdapter(self)
        return adapter.run(**kwargs)

    def _run_modelingtoolkit(self, **kwargs) -> ExperimentResult:
        """Run simulation using pure ModelingToolkit.jl via pyjulia."""
        from tvbo.adapters.modelingtoolkit import ModelingToolkitAdapter

        adapter = ModelingToolkitAdapter(self)
        return adapter.run(**kwargs)

    def _run_bifurcation(self, **kwargs) -> ExperimentResult:
        """Run bifurcation analysis via BifurcationKit.jl."""
        from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

        exploration_kwargs = kwargs.pop("exploration_kwargs", {})
        adapter = BifurcationKitAdapter(self)
        bif_result = adapter.run(**kwargs)
        return ExperimentResult(
            continuations=self._wrap_bifurcation_result(bif_result),
            explorations=self._run_python_explorations(**exploration_kwargs),
            source=self,
            name=self.label,
        )

    def _run_pyrates_bifurcation(self, **kwargs) -> ExperimentResult:
        """Run bifurcation analysis via PyRates/PyCoBi (AUTO-07p)."""
        from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter

        exploration_kwargs = kwargs.pop("exploration_kwargs", {})
        adapter = PyRatesBifurcationAdapter(self)
        bif_result = adapter.run(**kwargs)
        return ExperimentResult(
            continuations=self._wrap_bifurcation_result(bif_result),
            explorations=self._run_python_explorations(**exploration_kwargs),
            source=self,
            name=self.label,
        )

    def _run_numcont(self, **kwargs) -> ExperimentResult:
        """Run bifurcation analysis via the in-tree AUTO-07p (numcont) adapter."""
        from tvbo.adapters.numcont import NumContAdapter

        exploration_kwargs = kwargs.pop("exploration_kwargs", {})
        adapter = NumContAdapter(self)
        bif_result = adapter.run(**kwargs)
        return ExperimentResult(
            continuations=self._wrap_bifurcation_result(bif_result),
            explorations=self._run_python_explorations(**exploration_kwargs),
            source=self,
            name=self.label,
        )

    def _run_python_explorations(self, **kwargs):
        """Run declared time-series explorations directly from the dynamics object."""
        explorations = getattr(self, "explorations", None) or {}
        if not explorations:
            return {}

        from itertools import product

        results = {}
        duration = kwargs.get("duration", getattr(self.integration, "duration", 1200))
        dt = kwargs.get("dt", getattr(self.integration, "step_size", 0.1))

        for name, exploration in explorations.items():
            axes = as_list(getattr(exploration, "space", None))
            axis_values = []
            axis_info = []
            for axis in axes:
                parameter = str(getattr(axis, "parameter"))
                values = list(getattr(axis, "explored_values", None) or [])
                if not values:
                    # Fall back to the axis domain (lo/hi + n or step) so a
                    # domain-based sweep works here too, not only explicit lists.
                    dom = getattr(axis, "domain", None)
                    lo = getattr(dom, "lo", None) if dom is not None else None
                    hi = getattr(dom, "hi", None) if dom is not None else None
                    if lo is not None and hi is not None:
                        lo, hi = float(lo), float(hi)
                        n = int(getattr(dom, "n", 0) or 0)
                        step = getattr(dom, "step", None)
                        if n >= 1:
                            # n==1 -> linspace(lo, hi, 1) == [lo] (a single-point axis),
                            # not the 11-point default below.
                            values = [float(v) for v in jnp.linspace(lo, hi, n)]
                        elif step:
                            values = [float(v) for v in jnp.arange(lo, hi + float(step) / 2, float(step))]
                        else:
                            values = [float(v) for v in jnp.linspace(lo, hi, 11)]
                if not values:
                    raise ValueError(
                        f"Exploration {name!r} axis {parameter!r} needs explored_values or a domain (lo/hi)"
                    )
                axis_values.append(values)
                # Harmonized axis shape shared by every backend: Bunch(name,
                # explored_values, n) — the coordinate source for ExplorationResult.as_grid().
                axis_info.append(Bunch(name=parameter, explored_values=jnp.asarray(values), n=len(values)))

            obs = getattr(exploration, "observable", None)
            output_name = getattr(obs, "output", None) or getattr(obs, "function", None) or getattr(obs, "name", None)
            output_names = [str(output_name)] if output_name else [next(iter(self.dynamics.state_variables))]
            state_names = list(self.dynamics.state_variables)
            output_indices = [state_names.index(var) for var in output_names]

            dyn = self.dynamics.copy()
            data = []
            for values in product(*axis_values):
                for axis, value in zip(axes, values):
                    parameter = str(getattr(axis, "parameter"))
                    parameter_name = parameter.split(".")[-1]
                    dyn.parameters[parameter_name].value = float(value)
                ts = dyn.run(format="python", duration=duration, dt=dt)
                data.append(ts.data[:, output_indices, 0, 0])

            results[str(name)] = ExplorationResult(
                name=str(name),
                axes=axis_info,
                results=np.asarray(data),
                observable=", ".join(output_names),
                dt=dt,
                output_names=output_names,
            )

        return results

    def _wrap_bifurcation_result(self, bif_result):
        """Map adapter return (single result or dict) to named continuations dict."""
        if isinstance(bif_result, dict):
            return bif_result
        conts = getattr(self, "continuations", None) or {}
        name = next(iter(conts.keys())) if conts else "default"
        return {name: bif_result}

    def _run_julia(self, **kwargs) -> ExperimentResult:
        """Run simulation using DifferentialEquations.jl via juliacall."""
        from tvbo.adapters.diffeq import DiffEqAdapter

        adapter = DiffEqAdapter(self)
        return adapter.run(**kwargs)

    def plot(self, layout=None, panels=None, run_kwargs=None, auto=True, **kwargs):
        """Plot experiment outputs directly or compose multi-panel layouts.

        By default (``auto=True``), this runs the experiment once, infers
        task-aware panels, and renders a flexible subplot_mosaic layout.
        """
        from tvbo.plot.experiment_layout import plot_experiment_layout

        fig = kwargs.pop("fig", None)
        axes = kwargs.pop("axes", None)
        if auto or layout is not None or panels is not None or fig is not None or axes is not None:
            return plot_experiment_layout(
                self,
                layout=layout,
                panels=panels,
                figsize=kwargs.pop("figsize", None),
                subplot_kwargs=kwargs.pop("subplot_kwargs", None),
                run_kwargs=run_kwargs,
                fig=fig,
                axes=axes,
            )

        result = self.run(**(run_kwargs or {}))
        return result.plot(**kwargs)

    def get_experiment_file_prefix(self):
        """Build a BIDS-style filename prefix for this experiment.

        Returns:
            A string of the form `ses-<id>_desc-<label>`, where the
            description falls back to the dynamics label, name, or
            `"simulation"`.
        """
        desc = self.dynamics.label or self.dynamics.name or "simulation"
        return f"ses-{self.id}_desc-{desc}"

    def freeze_yaml(self, out_dir, network_stem="network"):
        """Render a self-contained spec YAML with the connectome frozen alongside.

        When the experiment has a resolved multi-node network, its matrices are
        written as an HDF5 companion (``<network_stem>.h5`` + ``.yaml`` sidecar) in
        *out_dir* and the returned YAML references them via ``network.data_file``
        (inline coupling / transforms / parameters preserved). This makes the spec
        reproducible on reload without the original data sources — the same
        mechanism the workflow emitter uses. Without such a network the plain
        metadata YAML already round-trips and is returned unchanged.
        """
        from pathlib import Path as _Path

        import numpy as _np

        from tvbo import datamodel as _dm
        from tvbo.classes.network import Network as _Network

        net = getattr(self, "network", None)
        has_matrices = net is not None and (
            (getattr(net, "number_of_nodes", None) or 0) > 1
            or (getattr(net, "weights", None) is not None and _np.asarray(net.weights).size > 1)
        )
        if not has_matrices:
            return self.to_yaml()

        out_dir = _Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(net, _Network):
            net.__class__ = _Network
        self.bake_real_node_labels()
        sidecar = out_dir / f"{network_stem}.yaml"
        net.save(sidecar, binary_format="h5")
        # The network sidecar goes through a different serializer; strip the same
        # private runtime keys so the frozen connectome round-trips on reload.
        sidecar.write_text(_strip_private_yaml_keys(sidecar.read_text()), encoding="utf-8")

        ref = _dm.Network(data_file=f"{network_stem}.h5")
        if getattr(net, "coupling", None):
            for k, v in dict(net.coupling).items():
                ref.coupling[k] = v
        if getattr(net, "transforms", None):
            ref.transforms = list(net.transforms)
        if getattr(net, "parameters", None):
            for k, v in dict(net.parameters).items():
                ref.parameters[k] = v
        if getattr(net, "observations", None):
            for k, v in dict(net.observations).items():
                ref.observations[k] = v

        original = self.network
        self.network = ref
        try:
            return self.to_yaml()
        finally:
            self.network = original

    def get_result_stem(self):
        """BIDS result basename (no extension), generated with pybids ``build_path``.

        Returns e.g. ``exp-<id>_desc-<label>_result`` — the shared stem for this
        experiment's ``<stem>.h5`` data file and ``<stem>.yaml`` provenance sidecar,
        identical whether written by a local run or the HPC gather pass. The naming
        is driven by ``tvbo.adapters.bids.RESULT_PATTERNS`` (a pybids rule string),
        so it stays BIDS-compliant and customizable in one place.
        """
        from tvbo.adapters.bids import build_result_path

        path = build_result_path(self, extension=".h5")
        return os.path.splitext(path)[0] if path else "result"

    @property
    def max_delay(self) -> float:
        """Compute the maximum delay (ms) from the current network/connectome."""
        if self.network is None:
            return 0.0
        delays = self.network.calculate_delays()
        # Use nanmax to ignore NaN values for non-existent edges
        max_val = np.nanmax(delays)
        return float(max_val) if not np.isnan(max_val) else 0.0

    @property
    def horizon(self, dt: float | None = None) -> int:
        """Number of history steps needed given delays and dt, like the old `horizon` attribute."""
        if dt is None:
            dt = float(self.integration.step_size)
        md = self.max_delay
        return int(round(md / dt)) + 1 if dt > 0 else 1

    @property
    def network_observation_measures(self) -> dict:
        """Map each network-sourced observation to its network measure.

        Resolves the declarative ``source: [network.observations.<measure>]``
        (or ``network.edges.<measure>``) pointer once, in Python, so neither
        the codegen template nor the runtime re-parse the string. Returns e.g.
        ``{'fc_target': 'BoldCorrelation'}``; empty when no observation is
        network-sourced.
        """
        out: dict = {}
        for name, obs in (self.observations or {}).items():
            src = getattr(obs, "source", None)
            if isinstance(src, (list, tuple)):
                src = src[0] if src else None
            if src is not None and hasattr(src, "name"):
                src = src.name
            s = str(src) if src is not None else ""
            if s.startswith("network.observations.") or s.startswith("network.edges."):
                out[name] = s.split(".")[-1]
        return out

    def _validate_network_observations(self) -> None:
        """Load-time check: every ``network.observations.<measure>`` an
        observation references must be declared in the network's
        ``observational_measures``. Cheap (names only, no data load) and
        turns a runtime ``NameError`` deep in generated JAX into a clear
        message at experiment-resolution time."""
        measures = self.network_observation_measures
        if not measures or self.network is None:
            return
        declared = set(self.network.observational_measures or [])
        for name, measure in measures.items():
            if declared and measure not in declared:
                raise ValueError(
                    f"Observation '{name}' sources 'network.observations."
                    f"{measure}', but network '{getattr(self.network, 'label', '?')}' "
                    f"declares observational_measures={sorted(declared)}. "
                    f"Add '{measure}' to the network or fix the observation source."
                )

    def resolve_network_observations(self) -> dict:
        """Resolve network-sourced observations to their matrices.

        Pairs :attr:`network_observation_measures` with the data the network
        carries (:attr:`Network.observations`), yielding ``{obs_name: matrix}``
        ready to pass into the generated
        ``run_experiment(network_observations=...)``. Raises a clear error if
        a declared measure's data is absent.
        """
        measures = self.network_observation_measures
        if not measures:
            return {}
        available = self.network.observations if self.network is not None else {}
        resolved: dict = {}
        for name, measure in measures.items():
            if measure not in available:
                raise ValueError(
                    f"Observation '{name}' sources network measure "
                    f"'{measure}', but the network provides "
                    f"{sorted(available) or 'no observational data'}. "
                    f"Ensure the network's companion file carries '{measure}'."
                )
            resolved[name] = available[measure]
        return resolved

    def bake_real_node_labels(self) -> bool:
        """Replace the model network's placeholder labels with real ones in place.

        A network sourced by a ``bids:`` entity block carries only ``region_N``
        placeholders until run time; freezing drops that block, so the real labels
        (hydrated from the db) are written onto the network's nodes here. This keeps
        a frozen kit self-contained and label reconciliation (never positional)
        working on reload. Returns True when labels were applied.
        """
        net = getattr(self, "network", None)
        nodes = list(getattr(net, "nodes", None) or []) if net is not None else []
        real_labels = self._resolve_model_node_labels()
        if not real_labels or len(real_labels) != len(nodes):
            return False
        if [str(getattr(nd, "label", "")) for nd in nodes] == real_labels:
            return False
        for nd, lbl in zip(nodes, real_labels):
            nd.label = lbl
        return True

    def _resolve_model_node_labels(self) -> list:
        """Real node labels of the model network, hydrated if needed.

        A network sourced by a ``bids:`` entity block loads its weights lazily,
        so at parse time its node labels can be placeholders (``region_<i>``).
        Label reconciliation needs the true labels, so when the current ones look
        like placeholders this resolves the ``bids:`` entities against the tvbo
        database and reads the matched connectome's labels. Returns the existing
        labels unchanged when they are already real.
        """
        net = getattr(self, "network", None)
        if net is None:
            return []
        labels = [str(lbl) for lbl in (net.node_labels or [])]
        placeholder = bool(labels) and all(re.fullmatch(r"region_\d+", lbl) for lbl in labels)
        if labels and not placeholder:
            return labels
        # Placeholder path — resolving from the db is a directory glob + a sidecar
        # parse, so cache it (the bids block + db are stable for the experiment).
        cached = getattr(self, "_model_labels_from_bids", None)
        if cached is not None:
            return cached
        bids = getattr(net, "bids", None)
        if bids is None:
            return labels
        from tvbo.classes.network import _filter_networks_by_entities

        ents = _bids_entities_to_short_dict(bids)
        matches = _filter_networks_by_entities(ents) if ents else []
        if len(matches) == 1:
            resolved = [str(lbl) for lbl in Network.load(str(matches[0])).node_labels]
            self._model_labels_from_bids = resolved
            return resolved
        return labels

    @property
    def dataset_observation_targets(self) -> dict:
        """Map each dataset-sourced observation to its measure name.

        Recognises ``source: [dataset.subject.<measure>]`` pointers — the
        per-subject empirical target selected by the observation's ``query``
        under ``dataset.bids_root``. Returns e.g. ``{'empirical_fc': 'fc'}``;
        empty when no observation is dataset-sourced.
        """
        out: dict = {}
        for name, obs in (self.observations or {}).items():
            src = getattr(obs, "source", None)
            if isinstance(src, (list, tuple)):
                src = src[0] if src else None
            if src is not None and hasattr(src, "name"):
                src = src.name
            s = str(src) if src is not None else ""
            if s.startswith("dataset.subject."):
                out[name] = s.split(".")[-1]
        return out

    @staticmethod
    def _bids_query_dict(query) -> tuple[dict, str | None]:
        """Split a ``BidsEntities`` query into (key-value entities, suffix).

        Maps the schema's attribute names to the short entity keys that
        :func:`tvbo.classes.network._parse_bids_entities` emits. ``suffix`` is
        returned separately because it is the trailing filename component, not a
        ``key-value`` entity.
        """
        if query is None:
            return {}, None
        ents = _bids_entities_to_short_dict(query)
        suffix = getattr(query, "suffix", None)
        return ents, (str(suffix) if suffix is not None else None)

    @staticmethod
    def _match_subject_files(files, ents: dict, suffix: str | None):
        """Yield (subject, path) for files whose parsed entities match *ents* + *suffix*."""
        from tvbo.classes.network import _parse_bids_entities

        for f in files:
            file_ents = _parse_bids_entities(f.stem)
            if not file_ents.get("sub"):
                continue
            if suffix is not None and not f.stem.endswith(f"_{suffix}"):
                continue
            if all(file_ents.get(k) == v for k, v in ents.items()):
                yield file_ents["sub"], f

    def _find_subject_file(self, root: Path, subject: str, query) -> Path:
        """Resolve the single per-subject file under *root* matching *query*."""
        ents, suffix = self._bids_query_dict(query)
        return self._find_subject_file_by_entities(root, subject, ents, suffix)

    def _find_subject_file_by_entities(self, root: Path, subject: str,
                                       ents: dict, suffix: str | None) -> Path:
        """Resolve the single per-subject file under *root* matching *ents* + *suffix*.

        The entity-level entry point behind :meth:`_find_subject_file`: callers that
        need to tighten or override the query entities (a kit bundling one atlas
        variant out of several) build the ``(ents, suffix)`` pair themselves.
        """
        sub = str(subject).replace("sub-", "")
        files = sorted((Path(root) / f"sub-{sub}").glob("*.yaml"))
        matches = [f for s, f in self._match_subject_files(files, ents, suffix) if s == sub]
        if not matches:
            raise ValueError(
                f"No file for sub-{sub} matching query {ents} suffix={suffix!r} under {root}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Query {ents} suffix={suffix!r} is ambiguous for sub-{sub}: "
                f"{[m.name for m in matches]}. Add entities to disambiguate."
            )
        return matches[0]

    @staticmethod
    def _reconcile_mode(obs) -> str:
        """The ``reconcile`` mode of an observation as a plain string (default by_label)."""
        m = getattr(obs, "reconcile", None)
        return str(getattr(m, "value", m) or "by_label")

    def _dataset_bids_root(self) -> Path:
        """``dataset.bids_root`` as a Path, or a clear error if a dataset target needs it.

        A relative root resolves against the spec file's directory (exactly like a
        network ``data_file``), so a kit that bundles its per-subject data under
        ``spec/dataset`` and records ``bids_root: dataset`` resolves on any host
        regardless of the working directory it is run from.
        """
        ds = getattr(self, "dataset", None)
        root = getattr(ds, "bids_root", None) if ds is not None else None
        if not root:
            raise ValueError("A dataset-sourced observation needs experiment.dataset.bids_root.")
        root = Path(root)
        if not root.is_absolute() and getattr(self, "_source_file", None):
            root = Path(self._source_file).parent / root
        return root

    def _target_canonical_labels(self, target_net) -> list:
        """Each target node's label mapped to the model's canonical label, target order.

        Alias-aware: a target label that equals a model node's ``label`` or any of its
        ``alternateName`` entries maps to that node's canonical label; an unmatched
        label is returned unchanged (it will not intersect the model set and so is
        excluded from the reconciliation, counting against coverage). Mapping is by
        name only — a target whose nodes are permuted still maps correctly.
        """
        amap = self.network.region_alias_map()
        return [amap.get(str(tl), str(tl)) for tl in target_net.node_labels]

    def _shared_labels(self, model_labels: list, target_net) -> list:
        """Canonical labels shared by the model network and a target, in model order.

        Keyed and alias-aware (never positional): the target's labels are mapped to
        the model's canonical labels first, so a divergent nomenclature reconciles.
        """
        tset = set(self._target_canonical_labels(target_net))
        return [lbl for lbl in model_labels if lbl in tset]

    @staticmethod
    def _coverage_threshold(obs) -> float:
        """Minimum reconciliation coverage for a ``by_label`` dataset target.

        Absent ``min_coverage`` means require full coverage (``1.0``): silently
        fitting on a partial node subset must be an explicit opt-in.
        """
        mc = getattr(obs, "min_coverage", None)
        return 1.0 if mc is None else float(mc)

    def resolve_dataset_observations(self, active_subject: str) -> dict:
        """Resolve per-subject dataset-sourced targets for one subject.

        For each observation whose ``source`` is ``dataset.subject.<measure>``:
        query ``dataset.bids_root`` for the subject's matching file, load it as a
        Network, read ``<measure>``, and reconcile its nodes to the model network.
        ``reconcile: by_label`` maps each target node to the model's canonical label
        (alias-aware — a divergent nomenclature such as ``THALAMUS_LEFT`` for
        ``L_Thalamus`` still matches via the atlas ``alternateName`` crosswalk), then
        selects the shared labels in the model's order. Alignment is by name on both
        the empirical target and the simulated observable — never by row index — so a
        differing node count or order (or a swapped hemisphere block) cannot silently
        misalign the comparison. The realised coverage is logged, and falls back to
        requiring full coverage unless the observation sets ``min_coverage``.

        Returns ``{obs_name: xarray.DataArray}`` keyed by canonical node label on both
        axes, restricted to the labels shared with the model network. The model-side
        gather (which model nodes the shared labels are) is available via
        :meth:`dataset_reconcile_index` so the simulated observation selects the same
        sub-block.
        """
        targets = self.dataset_observation_targets
        if not targets:
            return {}
        root = self._dataset_bids_root()
        model_labels = self._resolve_model_node_labels()
        logger = logging.getLogger(__name__)
        resolved: dict = {}
        for name, measure in targets.items():
            obs = self.observations[name]
            path = self._find_subject_file(root, active_subject, getattr(obs, "query", None))
            target_net = Network.load(str(path))
            mat = np.asarray(target_net.matrix(measure))
            if mat.ndim != 2:
                raise ValueError(
                    f"Observation '{name}': measure '{measure}' is not a 2-D matrix "
                    f"in {path.name}."
                )
            tlabels = list(target_net.node_labels)
            if self._reconcile_mode(obs) == "by_label" and model_labels:
                # Relabel the target's node axes to the model's canonical labels by
                # NAME (alias-aware), then select the shared labels in model order.
                tcanon = self._target_canonical_labels(target_net)
                tcanon_set = set(tcanon)
                model_set = set(model_labels)
                counts = Counter(tcanon)
                dup = sorted(c for c in tcanon_set if c in model_set and counts[c] > 1)
                if dup:
                    raise ValueError(
                        f"Observation '{name}': target {path.name} maps multiple nodes "
                        f"onto the same model region(s) {dup} — ambiguous reconciliation."
                    )
                shared = [lbl for lbl in model_labels if lbl in tcanon_set]
                coverage = len(shared) / len(model_labels) if model_labels else 0.0
                logger.info(
                    "reconcile[%s] sub-%s: %d/%d model nodes matched (coverage %.3f) "
                    "against %s",
                    name, active_subject, len(shared), len(model_labels), coverage,
                    path.name,
                )
                if not shared:
                    raise ValueError(
                        f"Observation '{name}': no node labels shared between the model "
                        f"network and target {path.name} (after alias reconciliation). "
                        f"Check the atlas alternateName crosswalk."
                    )
                threshold = self._coverage_threshold(obs)
                if coverage < threshold:
                    unmatched = [lbl for lbl in model_labels if lbl not in tcanon_set]
                    raise ValueError(
                        f"Observation '{name}': reconciliation coverage {coverage:.3f} "
                        f"< required {threshold:.3f} for {path.name}. "
                        f"{len(unmatched)} model node(s) unmatched, e.g. {unmatched[:5]}. "
                        f"Set observation.min_coverage to accept a partial subset."
                    )
                da = xr.DataArray(
                    mat, dims=("node_i", "node_j"),
                    coords={"node_i": tcanon, "node_j": tcanon},
                )
                da = da.sel(node_i=shared, node_j=shared)
            else:
                da = xr.DataArray(
                    mat, dims=("node_i", "node_j"),
                    coords={"node_i": tlabels, "node_j": tlabels},
                )
            resolved[name] = da
        return resolved

    def dataset_reconcile_index(self, shared_labels: list, model_labels: list = None) -> np.ndarray:
        """Indices into the model network's nodes for *shared_labels* (keyed).

        The simulated observation selects this sub-block so it aligns, label for
        label, with a ``by_label``-reconciled empirical target. Pass *model_labels*
        to avoid re-resolving them.
        """
        model_labels = model_labels if model_labels is not None else self._resolve_model_node_labels()
        pos = {lbl: i for i, lbl in enumerate(model_labels)}
        return np.array([pos[lbl] for lbl in shared_labels], dtype=int)

    def dataset_reconcile_indices(self) -> dict:
        """Model-side gather index for each ``by_label`` dataset target (keyed).

        For every observation sourced from ``dataset.subject.<measure>`` with
        ``reconcile: by_label``, returns the positions of the shared node labels
        within the model network's node order — derived from the labels, never
        from position. The shared label set is identical across the cohort, so it
        is resolved once from the first cohort subject (reading only its node
        labels, not its matrix). Codegen uses this to gather the simulated
        observable onto the same shared labels as the reconciled empirical target
        before comparing them. Returns ``{}`` (no gather) when nothing is
        dataset-sourced or the data cannot be resolved.
        """
        targets = self.dataset_observation_targets
        subjects = self.dataset_subject_ids() if targets else []
        if not subjects:
            return {}
        try:
            root = self._dataset_bids_root()
            model_labels = self._resolve_model_node_labels()
            out: dict = {}
            for name, _measure in targets.items():
                obs = self.observations[name]
                if self._reconcile_mode(obs) != "by_label":
                    continue
                path = self._find_subject_file(root, subjects[0], getattr(obs, "query", None))
                shared = self._shared_labels(model_labels, Network.load(str(path)))
                out[name] = self.dataset_reconcile_index(shared, model_labels).tolist()
        except Exception:
            return {}
        return out

    def dataset_subject_ids(self) -> list:
        """Enumerate the cohort for the per-subject workflow fan-out.

        An explicit ``dataset.subjects`` list wins (a curated subset). Otherwise
        the subjects are discovered by querying ``dataset.bids_root`` with the
        first dataset-sourced observation's ``query`` — the same filter that
        resolves each shard's target, so discovery and resolution never diverge.
        Returns sorted subject IDs without the ``sub-`` prefix; empty when the
        experiment has no dataset-sourced observation.
        """
        ds = getattr(self, "dataset", None)
        if ds is None:
            return []
        subs = getattr(ds, "subjects", None)
        if subs:
            ids = list(subs.keys()) if hasattr(subs, "keys") else [
                getattr(x, "subject_id", x) for x in subs
            ]
            return [str(s).replace("sub-", "") for s in ids]
        targets = self.dataset_observation_targets
        root = getattr(ds, "bids_root", None)
        if not targets or not root:
            return []
        cached = getattr(self, "_subject_ids_from_query", None)
        if cached is not None:
            return cached
        query = getattr(self.observations[next(iter(targets))], "query", None)
        ents, suffix = self._bids_query_dict(query)
        files = sorted(self._dataset_bids_root().glob("sub-*/*.yaml"))
        found = sorted({s for s, _ in self._match_subject_files(files, ents, suffix)})
        self._subject_ids_from_query = found
        return found

    @staticmethod
    def _sidecar_companions(sidecar: Path) -> list:
        """Data files a network sidecar references, resolved next to the sidecar.

        A per-subject FC file is a YAML sidecar plus its payload (``data_file``, an
        HDF5 matrix); some networks instead reference ``nodes``/``edges`` tables or an
        ``edge_matrix_files`` map. Return every companion the sidecar points at so a
        bundle copies the sidecar together with the bytes it needs — nothing more.
        """
        import yaml

        try:
            meta = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        except Exception:
            return []
        if not isinstance(meta, dict):   # a malformed (non-mapping) sidecar has no companions
            return []
        refs: list = []
        for key in ("data_file", "nodes", "edges"):
            v = meta.get(key)
            if isinstance(v, str):
                refs.append(v)
        emf = meta.get("edge_matrix_files")
        if isinstance(emf, dict):
            refs += [v for v in emf.values() if isinstance(v, str)]
        elif isinstance(emf, list):
            refs += [v for v in emf if isinstance(v, str)]
        return [sidecar.parent / r for r in refs]

    def dataset_bundle_files(self, entity_overrides: dict | None = None) -> dict:
        """Per enumerated subject, the source file(s) a self-contained kit must carry.

        For each dataset-sourced observation, resolve the subject's matching sidecar
        under ``dataset.bids_root`` and pair it with the payload(s) it references, so a
        workflow kit can bundle exactly the empirical targets its fan-out consumes and
        drop its dependence on a machine-specific data tree. *entity_overrides* pins or
        tightens the BIDS entities used for selection (e.g. ``{'atlas': 'HCPMMP1',
        'suffix': 'relmat'}``) — the exact variant is chosen when a subject directory
        holds several. ``suffix`` overrides the trailing filename component; every
        other key overrides a ``key-value`` entity. The overrides disambiguate among
        the variants a subject already has; the cohort itself is still enumerated by
        the observation's own query (:meth:`dataset_subject_ids`).

        Returns ``{subject_id: [sidecar, payload, …]}`` (existing files, de-duplicated
        in first-seen order); empty when the experiment has no dataset target.
        """
        targets = self.dataset_observation_targets
        if not targets:
            return {}
        root = self._dataset_bids_root()
        overrides = dict(entity_overrides or {})
        suffix_override = overrides.pop("suffix", None)
        out: dict = {}
        for subject in self.dataset_subject_ids():
            picked: list = []
            for name in targets:
                obs = self.observations[name]
                ents, suffix = self._bids_query_dict(getattr(obs, "query", None))
                ents = {**ents, **overrides}
                if suffix_override is not None:
                    suffix = suffix_override
                sidecar = self._find_subject_file_by_entities(root, subject, ents, suffix)
                picked.append(sidecar)
                picked.extend(self._sidecar_companions(sidecar))
            seen: set = set()
            unique: list = []
            for f in picked:
                if f in seen:
                    continue
                seen.add(f)
                if not f.exists():
                    # A referenced payload absent from the tree would bundle a sidecar
                    # without its data — surface it now, not as a load failure on a node.
                    raise FileNotFoundError(
                        f"sub-{subject}: dataset target references {f.name}, absent under "
                        f"{f.parent} — the source tree is incomplete."
                    )
                unique.append(f)
            out[subject] = unique
        return out

    def _resolve_events(self) -> None:
        """Lower declarative stimulus/stimulation Event fields into the form the
        (shared) tvboptim codegen already consumes — done in Python at load
        time, exactly like graph-generator resolution, so no template change is
        needed and a sampled ``weight_distribution`` goes through the same
        printer-backed sampler a graph generator's `sample` step does.

        Per stimulus-type event:
        - ``event_type: stimulation`` is normalised to ``stimulus`` (synonym),
          so the codegen's ``'stimulus' in event_type`` filter matches.
        - ``target_variable`` becomes the event's effective ``name`` — the
          codegen exposes ``external_input[name]`` as the dfun variable.
        - ``target_regions: all`` (or labels/indices) is lowered to integer
          ``regions``; ``weight_distribution`` is sampled (seeded, reproducible)
          into the ``weighting`` array via the canonical resolver. Both are the
          legacy fields the codegen reads.
        - equation-level ``parameters`` are merged onto the event so the
          stimulus equation's symbols (e.g. ``T_drive``) resolve.
        """
        events = self.events
        if not events:
            return
        from tvbo.graph_generators.procedural import draw

        n_nodes = (
            getattr(self.network, "number_of_nodes", None)
            or getattr(self.network, "number_of_regions", None)
            or 1
        )
        labels = {}
        try:
            for i, node in enumerate(getattr(self.network, "nodes", None) or []):
                labels[str(getattr(node, "label", getattr(node, "id", i)))] = i
        except Exception:
            labels = {}

        items = events.items() if hasattr(events, "items") else []
        for _key, ev in items:
            et = str(getattr(ev, "event_type", "stimulus") or "stimulus").lower()
            if "stimul" not in et:
                continue
            ev.event_type = "stimulus"  # normalise synonym for the codegen filter

            tv = getattr(ev, "target_variable", None)
            if tv:
                ev.name = str(tv)

            # target_regions ('all' | labels | indices) -> integer regions
            tr = getattr(ev, "target_regions", None) or []
            if isinstance(tr, str):
                tr = [tr]
            tr = [str(x) for x in tr]
            regions = list(getattr(ev, "nodes", None) or getattr(ev, "regions", None) or [])
            if tr and not (len(tr) == 1 and tr[0].lower() == "all"):
                regions = [int(labels.get(x, x)) for x in tr]

            # weight_distribution -> weights array (canonical, seeded resolver)
            wd = getattr(ev, "weight_distribution", None)
            weighting = list(getattr(ev, "weights", None) or getattr(ev, "weighting", None) or [])
            if wd is not None and not weighting:
                n = len(regions) if regions else int(n_nodes)
                try:
                    samples = draw(wd, (n,), seed=getattr(wd, "seed", None))
                except ValueError as exc:
                    # The sampler names a graph-generator step, which says nothing about
                    # which stimulus is unresolvable. Fail — dropping the weighting
                    # silently is the bug this replaced — but name the event and field.
                    raise ValueError(
                        f"event {_key!r}: `weight_distribution` cannot be sampled: {exc}"
                    ) from exc
                weighting = [float(x) for x in samples]
                if not regions:
                    regions = list(range(int(n_nodes)))
            if regions:
                ev.nodes = regions
            if weighting:
                ev.weights = weighting

            # merge equation-level parameters onto the event (codegen reads ev.parameters)
            eq = getattr(ev, "equation", None)
            eq_params = getattr(eq, "parameters", None) if eq is not None else None
            if eq_params:
                merged = dict(getattr(ev, "parameters", None) or {})
                for pk, pv in (eq_params.items() if hasattr(eq_params, "items") else []):
                    merged.setdefault(pk, pv)
                ev.parameters = merged

    def collect_initial_conditions(self, random=False):
        """Build the initial-history `TimeSeries` for the simulation.

        Constructs a history buffer of shape `(horizon, n_state_vars, n_nodes,
        n_modes)` spanning the delay window `[-max_delay, 0]`. Values are drawn
        from each state variable's `distribution` when one is set (or when
        `random` is requested), otherwise from its scalar `initial_value`.

        Args:
            random: Deprecated. Force randomized initial values instead of
                relying on per-state-variable distributions.

        Returns:
            A `TimeSeries` whose time axis is the history window and whose data
            is the seeded initial state.
        """
        history = []
        n_modes = getattr(self.dynamics, "number_of_modes", 1) or 1
        n_nodes = getattr(self.network, "number_of_nodes", None) or 1

        if random:
            import warnings

            warnings.warn(
                "random=True is deprecated. Set distribution on state variables instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Auto-detect distributions on state variables
        has_distributions = any(getattr(sv, "distribution", None) for sv in self.dynamics.state_variables.values())

        if random or has_distributions:
            history.append(self.dynamics.get_initial_values(N=n_nodes))
        else:
            for sv in self.dynamics.state_variables.values():
                history.append(np.repeat(sv.initial_value, n_nodes).astype(float))

        history = np.vstack(history)
        history = np.repeat(history[:, :, None], repeats=n_modes, axis=2)
        # Compute horizon from max delay and dt
        H = self.horizon
        md = self.max_delay
        history = np.repeat(history[None], repeats=H, axis=0)
        t = np.linspace(-md, 0, H)
        return TimeSeries(t, history)

    def save_model_specification(self, dir):
        """Save the LEMS simulation file to *dir*.

        .. deprecated::
            Use ``NeuroMLAdapter(experiment).export(dir)`` from
            ``tvbo.adapters.neuroml`` instead.
        """
        import warnings

        warnings.warn(
            "save_model_specification() is deprecated. Use NeuroMLAdapter(experiment).export(dir) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tvbo.adapters.neuroml import NeuroMLAdapter

        paths = NeuroMLAdapter(self).export(dir, validate=False)
        return paths["simulation"]

    def to_lems(
        self,
        initial_conditions=1,
        out_path: str | None = None,
        out_file: str | None = None,
    ):
        """Export this experiment as a LEMS Model object.

        .. deprecated::
            Use ``NeuroMLAdapter(experiment).render_code()`` from
            ``tvbo.adapters.neuroml`` instead. This method returns a
            ``lems.Model`` object; the adapter produces a validated XML string
            that covers all LEMS constructs including ConditionalDerivedVariable
            and Coupling.
        """
        import warnings

        warnings.warn(
            "SimulationExperiment.to_lems() is deprecated. "
            "Use NeuroMLAdapter(experiment).render_code() from tvbo.adapters.neuroml instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import lems.api as lems
        from lems.model.component import Text
        from lems.model.simulation import DataWriter, Run

        model = self.dynamics.to_lems(initial_conditions=initial_conditions)

        base_local_ct = next(iter(model.component_types), None)
        local_comp = next(iter(model.components), None)

        local_ct = None
        if base_local_ct is not None:
            local_ct = lems.ComponentType(name="LocalDynamics", extends=base_local_ct.name)
            model.add(local_ct)
            if local_comp is not None:
                local_comp.type = local_ct.name

        sv_names = list(self.dynamics.state_variables.keys())
        target_sv = sv_names[0] if sv_names else "V"
        coupling_ct = lems.ComponentType(name="Coupling")
        coupling_ct.add(lems.Parameter(name="global_coupling", dimension="none"))

        try:
            coupl_meta = self.coupling
            params = getattr(coupl_meta, "parameters", {}) or {}
            for pname, pobj in params.items():
                pval = getattr(pobj, "value", 0)
                coupling_ct.add(lems.Constant(name=str(pname), value=str(pval), dimension="none"))
            pre_expr = getattr(getattr(coupl_meta, "pre_expression", None), "rhs", None)
            post_expr = getattr(getattr(coupl_meta, "post_expression", None), "rhs", None)
        except Exception:
            params = {}
            pre_expr = f"{target_sv}_j"
            post_expr = "a*gx + b"

        import re as _re

        if isinstance(pre_expr, str):
            for m in _re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)_j\b", pre_expr):
                pname = m.group(1) + "_j"
                if all(getattr(x, "name", None) != pname for x in list(coupling_ct.parameters) + list(coupling_ct.constants)):
                    coupling_ct.add(lems.Parameter(name=pname, dimension="none"))
        else:
            pre_expr = f"{target_sv}_j"

        coupling_ct.dynamics.add(lems.DerivedVariable(name="pre", value=str(pre_expr), dimension="none"))
        coupling_ct.dynamics.add(lems.DerivedVariable(name="gx", value="global_coupling * pre", dimension="none"))
        coupling_ct.dynamics.add(
            lems.DerivedVariable(
                name="post",
                value=str(post_expr) if post_expr else "gx",
                dimension="none",
            )
        )
        _ci = getattr(self.dynamics, "coupling_inputs", {}) or {}
        coupling_out_name = next(
            (
                str(name)
                for name, ci in _ci.items()
                if str(name) != "local_coupling" and not getattr(ci, "local", False)
            ),
            "c_pop0",
        )
        coupling_ct.add(lems.DerivedParameter(name=coupling_out_name, value="post"))
        model.add(coupling_ct)

        comp_id = local_comp.id if local_comp is not None else (local_ct.name if local_ct is not None else None)
        if local_ct is not None and comp_id is not None:
            if "out_path" not in local_ct.texts:
                local_ct.add_text(Text("out_path"))
            if "out_file" not in local_ct.texts:
                local_ct.add_text(Text("out_file"))

            if local_comp is not None:
                dir_path = out_path or "."
                file_name = out_file
                if file_name is None:
                    base = os.path.basename(out_path) if out_path else ""
                    root, ext = os.path.splitext(base)
                    if base and ext:
                        dir_path = os.path.dirname(out_path) or "."
                        file_name = base
                    else:
                        file_name = base or "tvbo_timeseries.csv"
                        if "." not in file_name:
                            file_name = f"{file_name}.csv"
                local_comp.set_parameter("out_path", dir_path)
                local_comp.set_parameter("out_file", file_name)

            def ensure_ms(x):
                """Return the value as a string with a trailing `ms` unit."""
                s = str(x).strip()
                return s if s.endswith("ms") else f"{s}ms"

            dt_ms = ensure_ms(self.integration.step_size)
            T_ms = ensure_ms(self.integration.duration)

            local_ct.simulation.add(Run(comp_id, "t", dt_ms, T_ms))
            local_ct.simulation.add(DataWriter("out_path", "out_file"))

            if comp_id not in model.targets:
                model.add_target(comp_id)

        for comp in model.components:
            for pk, pv in list(comp.parameters.items()):
                if not isinstance(pv, str):
                    comp.parameters[pk] = str(pv)

        return model

    def render_code(self, format="tvb", **kwargs):
        """Render generated code in *format* (back-compat shim around the registry)."""
        # Backend codegen requires resolved coupling/delay/stimulus metadata (configure
        # lowers coupling + stimulus events; idempotent).
        self.configure()
        from tvbo import export as _export
        return _export.render(self, format, **kwargs)

    def render(self, format="yaml", **kwargs) -> str:
        """Unified entry point for rendering the experiment in any output format.

        Dispatches via the :mod:`tvbo.export.registry`.  All supported
        formats (YAML, openMINDS, markdown/PDF report, TVB, JAX, tvboptim,
        Julia, NeuroML/LEMS, …) are looked up by canonical key or alias.

        Parameters
        ----------
        format : str
            Target output format.  See :func:`tvbo.export.list_formats`
            for the current set.
        **kwargs
            Forwarded to the underlying renderer.

        Returns
        -------
        str
        """
        from tvbo import export as _export
        return _export.render(self, format, **kwargs)

    @classmethod
    def supported_export_formats(cls) -> list[dict]:
        """Return metadata for API/UI export format dropdowns."""
        from tvbo import export as _export
        return _export.list_format_dicts()

    def save(
        self,
        path: str | Path,
        format: str | None = None,
        metadata_only: bool = True,
        **kwargs,
    ) -> str:
        """Render via :meth:`render` and persist to disk.

        Parameters
        ----------
        path : str or Path
            Output file path **or** directory.  When a directory is given the
            filename is derived from :meth:`get_experiment_file_prefix` and the
            correct extension for *format* (BIDS-style).
        format : str, optional
            Export format key (e.g. ``'yaml'``, ``'tvb'``, ``'neuroml'``).
            When omitted, inferred from *path* suffix.
        metadata_only : bool
            When ``True`` (default) only the textual/metadata artefact is
            written.  When ``False`` and the experiment has a network, the
            network arrays are also saved as an HDF5 sidecar file next to
            *path*.
        """
        from tvbo import export as _export

        path = Path(path)

        # Infer format from suffix if not provided
        if format is None:
            suffix_to_key = {
                ".yaml": "yaml", ".yml": "yaml",
                ".jsonld": "openminds", ".json": "openminds",
                ".nml": "neuroml", ".xml": "lems",
                ".py": "tvb", ".jl": "julia",
                ".md": "markdown", ".pdf": "pdf",
            }
            format = suffix_to_key.get(path.suffix.lower(), "yaml")

        fmt = _export.resolve(format)

        # If path is a directory, auto-generate BIDS-style filename
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{self.get_experiment_file_prefix()}{fmt.extension}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        if fmt.key == "pdf":
            fmt.renderer(self, outputfile=str(path), **kwargs)
        else:
            rendered = fmt.renderer(self, **kwargs)
            with path.open("w", encoding="utf-8") as f:
                f.write(rendered)

        if not metadata_only and self.network is not None:
            from tvbo.classes.network import Network

            net = self.network
            if not isinstance(net, Network):
                net.__class__ = Network
            net.save(path.with_stem(path.stem + "_network").with_suffix(".yaml"), binary_format="h5")

        return str(path)

    def save_code(self, dir, file_name=None):
        """Render the experiment as TVB Python code and write it to disk.

        Args:
            dir: Target directory (or full path when combined with a BIDS-style
                auto-generated filename).
            file_name: Optional filename to write inside `dir`; when omitted,
                the path is derived from `get_experiment_file_prefix`.

        Returns:
            The path of the written file, as a string.
        """
        if file_name is not None:
            from pathlib import Path as _Path
            return self.save(_Path(dir) / file_name, format="tvb")
        return self.save(dir, format="tvb")

    def get_parameters_collection(self, **kwargs):
        """Collect all experiment parameters into a nested `Bunch`.

        Traverses the experiment metadata, gathering parameters from the
        dynamics, coupling, network, and integration into a single container
        keyed by component.

        Args:
            **kwargs: Traversal options. `keys_to_exclude` is a list of keys to
                skip; `connectivity` and `coupling_inputs` are always excluded
                when any exclusions are given.

        Returns:
            A `Bunch` mapping component names to their parameter values.
        """
        if keys_to_exclude := kwargs.get("keys_to_exclude", []):
            keys_to_exclude = keys_to_exclude + ["connectivity", "coupling_inputs"]
        parameters = Bunch()
        traverse_metadata(
            self,
            callback_kwargs={"parameters": parameters},
            keys_to_exclude=keys_to_exclude,
        )
        return parameters

    @property
    def parameters(self):
        """The full collection of experiment parameters as a nested `Bunch`."""
        return self.get_parameters_collection()

    # ---- Reporting utilities (paralleling Dynamics) ----
    def report(
        self,
        format: str | None = "markdown",
        template_name: str = "tvbo-report-experiment",
        outputfile: str | None = None,
        derivative_notation: str = "dot",
    ) -> str:
        """Render a human-readable report for this experiment.

        - Reuses the model/dynamics report template via Mako include to avoid redundancy.
        - Summarizes integration, network/connectome, coupling, monitors, stimulation, and software info.

        Parameters
        - format: optional explicit fallback format ('markdown' or 'pdf')
        - template_name: base name of the template without extension
        - outputfile: optional path to write the rendered report;
            when provided, extension defines output format (.md or .pdf)
        """
        normalized_format = format.lower() if isinstance(format, str) else None

        if outputfile:
            ext = os.path.splitext(outputfile)[1].lower()
            ext_to_format = {
                ".md": "markdown",
                ".markdown": "markdown",
                ".pdf": "pdf",
            }
            if ext not in ext_to_format:
                raise ValueError("outputfile extension must be one of: .md, .pdf")
            normalized_format = ext_to_format[ext]

        if normalized_format is None:
            normalized_format = "markdown"

        if normalized_format not in ["markdown", "md", "pdf"]:
            raise ValueError("format must be one of: markdown, pdf")

        md_template = templates.lookup.get_template(f"report/{template_name}.md.mako")
        md_render = md_template.render(experiment=self, derivative_notation=derivative_notation)

        render = md_render

        # Persist if requested
        if outputfile:
            if normalized_format in ["pdf"]:
                from tvbo.utils import report as _report

                _report.to_pdf(md_render, outputfile)
            else:
                with open(outputfile, "w", encoding="utf-8") as f:
                    f.write(render)

        return render

    def generate_report(
        self,
        format: str | None = "markdown",
        template_name: str = "tvbo-report-experiment",
        outputfile: str | None = None,
        derivative_notation: str = "dot",
    ) -> str:
        """Backward-compatible alias for :meth:`report`."""
        return self.report(
            format=format,
            template_name=template_name,
            outputfile=outputfile,
            derivative_notation=derivative_notation,
        )

    def to_bids(
        self,
        output_dir: str,
        subject: str = "01",
        session: str | None = None,
        description: str = "tvbsim",
        run: int | None = None,
        ts_label: str = "sim",
        timeseries: TimeSeries | None = None,
        run_simulation: bool = True,
        **run_kwargs,
    ) -> str:
        """
        Export simulation experiment and results to BIDS-compliant format (BEP034 v1.0.0).

        This method creates a complete BIDS dataset containing:
        - Time series data in ts/ directory
        - Network connectivity in net/ directory
        - Model equations in eq/ directory
        - Coordinates in coord/ directory (if available)
        - JSON sidecar files with full metadata

        Parameters
        ----------
        output_dir : str
            Root directory for the BIDS dataset.
        subject : str
            Subject identifier (without 'sub-' prefix). Default: '01'.
        session : str, optional
            Session identifier (without 'ses-' prefix).
        description : str
            Description label for the output files. Default: 'tvbsim'.
        run : int, optional
            Run number.
        ts_label : str
            Time series label (e.g., 'sim', 'bold', 'eeg'). Default: 'sim'.
        timeseries : TimeSeries, optional
            Pre-computed TimeSeries. If not provided and run_simulation=True,
            the simulation will be executed.
        run_simulation : bool
            If True and no timeseries provided, run the simulation. Default: True.
        **run_kwargs
            Additional arguments passed to the run() method.

        Returns
        -------
        str
            Path to the created BIDS dataset root directory.

        Examples
        --------
        >>> experiment = SimulationExperiment(...)
        >>> experiment.to_bids("./derivatives/tvbo", subject="01", description="rest")
        './derivatives/tvbo'

        >>> # Or with pre-computed timeseries
        >>> ts = experiment.run()
        >>> experiment.to_bids("./derivatives/tvbo", timeseries=ts)
        './derivatives/tvbo'

        Notes
        -----
        Follows BIDS BEP034 Computational Modeling extension v1.0.0.
        Uses tvbo format for model equations.
        """
        # Run simulation if needed
        if timeseries is None and run_simulation:
            timeseries = self.run(**run_kwargs)

        if timeseries is None:
            raise ValueError(
                "No timeseries provided and run_simulation=False. Provide a TimeSeries or set run_simulation=True."
            )

        # Delegate to TimeSeries.to_bids with experiment reference
        return timeseries.to_bids(
            output_dir=output_dir,
            subject=subject,
            session=session,
            description=description,
            run=run,
            suffix=ts_label,
            experiment=self,
        )

    # ---- OpenMINDS JSON-LD conversion ----
    def to_openminds(
        self,
        filepath: str | None = None,
        base_id: str | None = None,
        include_context: bool = True,
    ) -> dict:
        """Export experiment to openMINDS JSON-LD format.

        Parameters
        ----------
        filepath : str, optional
            If provided, write JSON-LD to this file path.
        base_id : str, optional
            Base URI for generating @id values (e.g., "https://example.org/simulations").
        include_context : bool
            Whether to include the @context in the output. Default True.

        Returns
        -------
        dict
            OpenMINDS-compatible JSON-LD dictionary.

        Example
        -------
        >>> exp = SimulationExperiment(...)
        >>> jsonld = exp.to_openminds()
        >>> exp.to_openminds("output.jsonld", base_id="https://example.org")
        """
        from tvbo.adapters.openminds import experiment_to_openminds, save_openminds

        result = experiment_to_openminds(self, base_id=base_id, include_context=include_context)

        if filepath:
            save_openminds(self, filepath, base_id=base_id)

        return result

    @classmethod
    def from_openminds(cls, source: str | dict) -> "SimulationExperiment":
        """Create a SimulationExperiment from openMINDS JSON-LD.

        Parameters
        ----------
        source : str or dict
            Either a file path to a JSON-LD file, or a dict containing
            JSON-LD data.

        Returns
        -------
        SimulationExperiment
            New instance constructed from the openMINDS data.

        Example
        -------
        >>> exp = SimulationExperiment.from_openminds("experiment.jsonld")
        >>> exp = SimulationExperiment.from_openminds({"@type": "tvbo:SimulationExperiment", ...})
        """
        from tvbo.adapters.openminds import experiment_from_openminds, load_openminds

        if isinstance(source, str):
            # It's a file path
            data = load_openminds(source)
        elif isinstance(source, dict):
            data = experiment_from_openminds(source)
        else:
            raise TypeError(f"Expected str or dict, got {type(source)}")

        return cls(**data)
