"""The inventory of names generated code takes from frameworks, and the software entries that record it.

`tvbo.codegen.inventory` reads generated source; `scripts/update_software_interfaces.py` files what it reads under each `SimulationTool`'s `interfaces:`; `tvbo doctor` resolves those against the installed packages. The tests here hold each link: the readers find what a small module visibly uses, resolution tells a present name from a missing, an absorbed and a deprecated one, the entries record every name the golden corpus reaches, and the A-box carries them as individuals.
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import sys

import pytest
from rdflib import Graph, Literal
from rdflib.namespace import RDF, SKOS

from tvbo.codegen.inventory import (
    ABSORBED,
    ATTRIBUTE,
    BASE_CLASS,
    CALL,
    IMPORT,
    KEYWORD,
    MISSING,
    PRESENT,
    UNCHECKED,
    EmittedSymbol,
    _documented_deprecation,
    corpus_cases,
    inventory,
    julia_symbols,
    module_families,
    python_symbols,
    resolve,
    software_entries,
)
from tvbo.codegen.templater import source_language

REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS = REPO / "tests" / "reference_data" / "codegen"

PYTHON = """
import json
import numpy as np
from tvb.simulator.models.base import Model
from brian2 import *


class Generated(Model):
    def __init__(self):
        super().__init__(incoming_states=("x",))

    def dfun(self, x):
        return np.where(x > 0, np.pi * x, json.loads("0"))
"""


@pytest.mark.backend_core
def test_python_reader_finds_what_the_module_uses():
    """Imports, a leaf attribute, a call with its keywords, a base class and the keyword handed to the base's constructor; never the standard library or a star import."""
    found = {(symbol.name, symbol.kind) for symbol in python_symbols(PYTHON)}

    assert found == {
        ("numpy", IMPORT),
        ("tvb.simulator.models.base.Model", IMPORT),
        ("tvb.simulator.models.base.Model", BASE_CLASS),
        ("tvb.simulator.models.base.Model(incoming_states=)", KEYWORD),
        ("numpy.where", CALL),
        ("numpy.pi", ATTRIBUTE),
    }


JULIA = [
    (
        "a.julia",
        "julia",
        "using OrdinaryDiffEqTsit5\nimport NaNMath\nf!(dx, x, p, t) = (dx[1] = NaNMath.sqrt(x[1]))\nsolve(ODEProblem(f!, [1.0], (0, 1), ()), Tsit5(); saveat=0.1)\n",
    ),
    (
        "b.julia",
        "julia",
        "using OrdinaryDiffEqTsit5\nusing Random\nsolve(ODEProblem(g, [1.0], (0, 1), ()), Tsit5(); saveat=0.1)\nrand(MersenneTwister(0))\n",
    ),
]


@pytest.mark.backend_core
def test_an_unqualified_julia_call_is_attributed_to_the_packages_every_caller_loads():
    """`ODEProblem` is called from a file that loads `NaNMath` and one that loads `Random`; only `OrdinaryDiffEqTsit5` is loaded by both."""
    pytest.importorskip("tree_sitter_julia", reason="reading Julia needs tvbo[julia]")

    joined = inventory(JULIA)

    assert EmittedSymbol("OrdinaryDiffEqTsit5", "ODEProblem", CALL) in joined
    assert EmittedSymbol("OrdinaryDiffEqTsit5", "solve(saveat=)", KEYWORD) in joined
    assert EmittedSymbol("NaNMath", "NaNMath.sqrt", CALL) in joined
    assert not any(symbol.name in ("f!", "rand") for symbol in joined), (
        "a function the file defines, or one of Base's, is no package's"
    )
    assert {symbol.name for symbol in julia_symbols(JULIA[1][2]) if symbol.kind == IMPORT} == {"OrdinaryDiffEqTsit5", "Random"}


@pytest.mark.backend_core
@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("yaml.safe_dump", PRESENT),
        ("yaml.no_such_function", MISSING),
        ("yaml.safe_load(stream=)", PRESENT),
        ("yaml.safe_load(no_such_keyword=)", MISSING),
        ("yaml.safe_dump(sort_keys=)", ABSORBED),
    ],
)
def test_resolution_tells_present_missing_and_absorbed_apart(name: str, status: str):
    """A keyword the callee takes only through `**kwargs` is absorbed: it names data, not an interface."""
    assert resolve(EmittedSymbol("yaml", name, KEYWORD if "(" in name else CALL)).status == status


@pytest.mark.backend_core
def test_a_julia_package_is_not_asked():
    assert resolve(EmittedSymbol("OrdinaryDiffEqTsit5", "ODEProblem", CALL)).status == UNCHECKED


def _coupling(source=None, incoming_states=None):
    """A coupling.

    Args:
        source: The states sent across edges.
        incoming_states: Deprecated alias for ``source``; removed in 1.0.
    """


@pytest.mark.backend_core
def test_a_documented_deprecation_is_read_with_its_replacement():
    assert _documented_deprecation(_coupling, "incoming_states") == (True, "source")
    assert _documented_deprecation(_coupling, "source") == (False, "")


JULIA_READABLE = importlib.util.find_spec("tree_sitter_julia") is not None


@functools.cache
def _recorded() -> dict[str, tuple[list[str], set[str]]]:
    """`{tool name: (modules, recorded interface names)}` for every software entry that lists modules."""
    return {
        record["name"]: (record["modules"], set(record.get("interfaces") or {}))
        for _, record in software_entries()
        if record.get("modules")
    }


@functools.cache
def _corpus_inventory() -> dict:
    """The inventory of the golden corpus, built once; without tree-sitter, of its Python references only."""
    cases = [case for case in corpus_cases(CORPUS) if JULIA_READABLE or source_language(case[1]) != "julia"]
    return inventory(cases, module_families(software_entries()))


@pytest.mark.backend_core
@pytest.mark.parametrize("tool", sorted(_recorded()))
def test_every_name_the_corpus_reaches_is_recorded(tool: str):
    """A template that starts reaching a new name of a tool fails here until `scripts/update_software_interfaces.py` records it."""
    modules, recorded = _recorded()[tool]
    if any(module[0].isupper() for module in modules) and not JULIA_READABLE:
        pytest.skip("reading Julia needs tvbo[julia]")
    for module in modules:
        if module[0].islower():
            pytest.importorskip(module)
    reached = [symbol for symbol in _corpus_inventory() if symbol.package.split(".")[0] in modules]

    unrecorded = sorted(
        symbol.name for symbol in reached if symbol.name not in recorded and resolve(symbol).status != ABSORBED
    )

    assert not unrecorded, f"{tool} is reached through names its entry does not record: {unrecorded}"


@pytest.mark.backend_core
def test_the_abox_carries_each_interface_with_its_deprecation():
    """The recorded tvboptim keyword becomes an individual linked from the tool, deprecated and naming its replacement."""
    sys.path.insert(0, str(REPO / "scripts" / "ontology"))
    import gen_abox

    graph = Graph()
    tool = gen_abox.TVBO["software/Example"]
    gen_abox.emit_interface(
        graph,
        tool,
        "pkg.Coupling(incoming_states=)",
        {
            "kind": ["keyword"],
            "checked_version": "0.4.0",
            "deprecated": True,
            "replaced_by": "source",
            "realises": ["tvbo:Coupling"],
        },
    )

    (interface,) = graph.objects(tool, gen_abox.TVBO_SW.interfaces)
    assert (interface, RDF.type, gen_abox.TVBO_SW.SoftwareInterface) in graph
    assert (interface, gen_abox.TVBO_SW.deprecated, Literal(True)) in graph
    assert (interface, gen_abox.TVBO_SW.replaced_by, Literal("source")) in graph
    assert (interface, SKOS.exactMatch, gen_abox.TVBO.Coupling) in graph
