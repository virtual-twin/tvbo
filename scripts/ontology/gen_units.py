"""QUDT vendoring generator for TVB-O units.

Transcribes the unit facts TVBO reasons about from QUDT rather than restating them locally, and emits two artifacts from one pass:

- ``ontology/tvb-o-units.ttl`` — the semantic module merged into ``tvbo.owl``. Every
  ``UnitEnum`` value becomes an individual carrying the QUDT facts it was vendored
  from: ``qudt:conversionMultiplier``, ``qudt:hasDimensionVector``,
  ``qudt:hasQuantityKind``, ``qudt:symbol`` and ``qudt:ucumCode``. A value QUDT
  publishes directly is ``owl:sameAs`` its QUDT unit; a compound QUDT has no IRI for
  is minted under the tvbo namespace with ``qudt:hasFactorUnit`` over vendored atoms,
  and its facts are *computed* from those atoms, never invented.

- ``tvbo/data/ontology/unit_facts.json`` — the compiled projection ``tvbo.utils.units``
  loads at runtime (stdlib ``json``, no rdflib on the hot path).

Only one thing here is authored by TVBO: ``DECOMPOSITIONS``, the factor-unit reading of the values QUDT publishes no IRI for. Everything else — multiplier, dimension vector, quantity kind, symbol, UCUM code — is copied or derived arithmetically.

That is also what makes a SymPy expression unnecessary to author. A unit is exactly ``multiplier x product(base_unit ** exponent)`` over the seven SI base units, and both halves come from QUDT, so ``mV`` is ``Rational(1,1000) * kilogram*meter**2 / (second**3 * ampere)`` — exact, canonical, and distinguishable from ``V``. An earlier draft of this work authored ``mV -> milli*volt`` by hand for every value; deriving it removes that table and the chance of it disagreeing with the dimension vector beside it.

Regenerate with ``make gen-units`` whenever ``UnitEnum`` gains a value.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "schema" / "units.yaml"
TTL_OUT = REPO / "ontology" / "tvb-o-units.ttl"
JSON_OUT = REPO / "tvbo" / "data" / "ontology" / "unit_facts.json"

QUDT_UNIT = "http://qudt.org/vocab/unit/"
QUDT_FETCH = "https://qudt.org/vocab/unit/"
TVBO_UNIT = "https://w3id.org/tvbo/units/"

BASE_DIMENSIONS = {
    "A": "mole",
    "E": "ampere",
    "L": "meter",
    "I": "candela",
    "M": "kilogram",
    "H": "kelvin",
    "T": "second",
}

DIMENSION_VECTOR_RE = re.compile(r"([AELIMHT])(-?\d+(?:\.\d+)?)")

DECOMPOSITIONS = {
    "mV_per_ms": {"MilliV": 1, "MilliSEC": -1},
    "mV_per_s": {"MilliV": 1, "SEC": -1},
    "mm_per_ms": {"MilliM": 1, "MilliSEC": -1},
    "per_mV": {"MilliV": -1},
    "per_mm2": {"MilliM": -2},
    "per_nC": {"NanoC": -1},
    "per_pC": {"PicoC": -1},
    "uA_per_cm2": {"MicroA": 1, "CentiM": -2},
    "uF_per_cm2": {"MicroFARAD": 1, "CentiM": -2},
    "S_per_cm2": {"S": 1, "CentiM": -2},
    "mS_per_cm2": {"MilliS": 1, "CentiM": -2},
    "S_per_m2": {"S": 1, "M": -2},
    "nS_per_mV": {"NanoS": 1, "MilliV": -1},
    "mol_per_cm3": {"MOL": 1, "CentiM": -3},
    "mol_per_m_per_A_per_s": {"MOL": 1, "M": -1, "A": -1, "SEC": -1},
    "Hz_per_nA": {"HZ": 1, "NanoA": -1},
    "Mohm": {"MegaOHM": 1},
    "kohm_cm": {"KiloOHM": 1, "CentiM": 1},
    "rad_per_ms": {"RAD": 1, "MilliSEC": -1},
    "kg_per_s": {"KiloGM": 1, "SEC": -1},
    "arbitrary_unit": {"UNITLESS": 1},
}

UNCURATED = {
    "per_unit": "Per-unit is a power-systems normalisation, not a physical unit; QUDT "
    "publishes no IRI for it and it has no uses in the database.",
}

FIELD_PATTERNS = {
    "multiplier": re.compile(r"qudt:conversionMultiplier\s+([^\s;]+)"),
    "offset": re.compile(r"qudt:conversionOffset\s+([^\s;]+)"),
    "dimension_vector": re.compile(r"qudt:hasDimensionVector\s+qkdv:([^\s;]+)"),
    "quantity_kinds": re.compile(r"qudt:hasQuantityKind\s+quantitykind:([^\s;]+)"),
    "symbol": re.compile(r'qudt:symbol\s+"([^"]*)"'),
    "ucum_code": re.compile(r'qudt:ucumCode\s+"([^"]*)"'),
}


class VendorError(RuntimeError):
    """A QUDT record TVBO depends on could not be vendored."""


def fetch_qudt(iri: str) -> dict:
    """The QUDT record for one unit IRI, reduced to the facts TVBO stores."""
    request = urllib.request.Request(QUDT_FETCH + iri, headers={"Accept": "text/turtle"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        raise VendorError(f"qudt:{iri} → HTTP {error.code}") from error

    if f"unit:{iri}\n" not in body and f"unit:{iri} " not in body:
        raise VendorError(f"qudt:{iri} resolved but describes no such unit")

    record = {"iri": iri}
    for field, pattern in FIELD_PATTERNS.items():
        found = pattern.findall(body)
        if field == "quantity_kinds":
            record[field] = found
        else:
            record[field] = found[0].strip('"') if found else None
    return record


def parse_dimension_vector(code: str) -> dict[str, int]:
    """A qkdv code such as ``A0E-1L2I0M1H0T-3D0`` as base-dimension exponents."""
    exponents = {}
    for letter, value in DIMENSION_VECTOR_RE.findall(code):
        number = Fraction(value)
        if number:
            exponents[BASE_DIMENSIONS[letter]] = number
    return exponents


def format_dimension_vector(exponents: dict[str, Fraction]) -> str:
    """Base-dimension exponents back into a qkdv code, so compounds round-trip.

    QUDT's trailing `D` component flags a dimensionless quantity, so it is 1 exactly when no base dimension survives.
    """
    parts = [f"{letter}{_number(exponents.get(BASE_DIMENSIONS[letter], Fraction(0)))}" for letter in "AELIMHT"]
    return "".join(parts) + f"D{0 if exponents else 1}"


def _number(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(float(value))


def compose(name: str, factors: dict[str, int], atoms: dict[str, dict]) -> dict:
    """The facts for a compound, computed from its vendored atoms.

    The multiplier is the exact product of the atoms' multipliers raised to their exponents, kept as a `Fraction` so `mm/ms` is exactly 1 rather than 0.9999999.
    """
    multiplier = Fraction(1)
    exponents: dict[str, Fraction] = {}
    for atom_iri, power in factors.items():
        atom = atoms[atom_iri]
        if atom["multiplier"] is None:
            raise VendorError(f"{name}: atom qudt:{atom_iri} has no conversionMultiplier")
        multiplier *= Fraction(str(atom["multiplier"])) ** power
        for dimension, value in atom["dimensions"].items():
            exponents[dimension] = exponents.get(dimension, Fraction(0)) + value * power
    exponents = {k: v for k, v in exponents.items() if v}
    return {
        "iri": None,
        "factors": factors,
        "multiplier": multiplier,
        "offset": None,
        "dimensions": exponents,
        "quantity_kinds": [],
        "symbol": None,
        "ucum_code": None,
    }


def load_enum() -> dict[str, dict]:
    """The `UnitEnum` permissible values, with their declared QUDT meanings."""
    schema = yaml.safe_load(SCHEMA.read_text())
    values = {}
    for name, body in schema["enums"]["UnitEnum"]["permissible_values"].items():
        body = body or {}
        meaning = body.get("meaning") or ""
        values[name] = {
            "qudt": meaning.split("qudt:")[1] if meaning.startswith("qudt:") else None,
            "description": body.get("description", ""),
        }
    return values


def vendor() -> dict:
    """Every `UnitEnum` value resolved to its unit facts."""
    enum = load_enum()

    direct = {name: body["qudt"] for name, body in enum.items() if body["qudt"]}
    needed = set(direct.values()) | {atom for f in DECOMPOSITIONS.values() for atom in f}

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = dict(zip(sorted(needed), pool.map(_safe_fetch, sorted(needed)), strict=True))

    broken = {iri: r for iri, r in fetched.items() if isinstance(r, VendorError)}
    atoms = {}
    for iri, record in fetched.items():
        if iri in broken:
            continue
        record["dimensions"] = parse_dimension_vector(record["dimension_vector"] or "")
        record["multiplier"] = Fraction(str(record["multiplier"])) if record["multiplier"] else Fraction(1)
        atoms[iri] = record

    units, problems = {}, []
    for name, body in enum.items():
        iri = body["qudt"]
        if name in UNCURATED:
            units[name] = {"curated": False, "reason": UNCURATED[name], "description": body["description"]}
            continue
        if iri and iri in atoms:
            facts = dict(atoms[iri])
            facts["factors"] = {iri: 1}
        elif name in DECOMPOSITIONS:
            try:
                facts = compose(name, DECOMPOSITIONS[name], atoms)
            except (KeyError, VendorError) as error:
                problems.append(f"{name}: {error}")
                continue
            if iri:
                problems.append(f"{name}: declares qudt:{iri} which does not resolve — delete the meaning")
        else:
            problems.append(f"{name}: no resolving QUDT IRI and no entry in DECOMPOSITIONS")
            continue

        units[name] = {
            "curated": True,
            "description": body["description"],
            "qudt": facts.get("iri"),
            "factors": {k: int(v) for k, v in facts["factors"].items()},
            "multiplier": [facts["multiplier"].numerator, facts["multiplier"].denominator],
            "offset": str(facts["offset"]) if facts.get("offset") else None,
            "dimensions": {k: [v.numerator, v.denominator] for k, v in facts["dimensions"].items()},
            "quantity_kinds": facts.get("quantity_kinds") or [],
            "symbol": facts.get("symbol"),
            "ucum_code": facts.get("ucum_code"),
        }

    return {"units": units, "problems": problems, "broken_iris": sorted(broken)}


def _safe_fetch(iri: str):
    try:
        return fetch_qudt(iri)
    except VendorError as error:
        return error


def emit_ttl(units: dict) -> str:
    """The mergeable semantic module, one individual per curated unit."""
    lines = [
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix qudt: <http://qudt.org/schema/qudt/> .",
        "@prefix qkdv: <http://qudt.org/vocab/dimensionvector/> .",
        "@prefix quantitykind: <http://qudt.org/vocab/quantitykind/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix unit: <http://qudt.org/vocab/unit/> .",
        f"@prefix tvbounit: <{TVBO_UNIT}> .",
        "",
        f"<{TVBO_UNIT}> a owl:Ontology ;",
        '    rdfs:comment "Unit facts vendored from QUDT. Generated by scripts/ontology/gen_units.py — do not hand-edit." .',
        "",
    ]
    for name, facts in sorted(units.items()):
        subject = f"tvbounit:{_escape(name)}"
        lines.append(f"{subject} a qudt:Unit ;")
        lines.append(f'    rdfs:label "{name}" ;')
        if facts.get("description"):
            lines.append(f'    rdfs:comment "{_escape_literal(facts["description"])}" ;')
        if not facts["curated"]:
            lines.append(f'    rdfs:comment "uncurated: {_escape_literal(facts["reason"])}" .')
            lines.append("")
            continue
        if facts["qudt"]:
            lines.append(f"    owl:sameAs unit:{facts['qudt']} ;")
        else:
            for atom, power in sorted(facts["factors"].items()):
                lines.append(f"    qudt:hasFactorUnit [ qudt:hasUnit unit:{atom} ; qudt:exponent {power} ] ;")
        numerator, denominator = facts["multiplier"]
        lines.append(f"    qudt:conversionMultiplier {numerator / denominator:.12g} ;")
        if facts["offset"]:
            lines.append(f"    qudt:conversionOffset {facts['offset']} ;")
        lines.append(f"    qudt:hasDimensionVector qkdv:{_qkdv(facts['dimensions'])} ;")
        for kind in facts["quantity_kinds"]:
            lines.append(f"    qudt:hasQuantityKind quantitykind:{kind} ;")
        if facts["symbol"]:
            lines.append(f'    qudt:symbol "{_escape_literal(facts["symbol"])}" ;')
        if facts["ucum_code"]:
            lines.append(f'    qudt:ucumCode "{_escape_literal(facts["ucum_code"])}" ;')
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        lines.append("")
    return "\n".join(lines) + "\n"


def _qkdv(dimensions: dict) -> str:
    exponents = {name: Fraction(n, d) for name, (n, d) in dimensions.items()}
    return format_dimension_vector(exponents)


def _escape(name: str) -> str:
    return name.replace("/", "_per_")


def _escape_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the emitted artifacts differ from those on disk")
    args = parser.parse_args()

    result = vendor()
    if result["problems"]:
        for problem in result["problems"]:
            print(f"  ! {problem}", file=sys.stderr)
        print(f"✗ {len(result['problems'])} unit(s) could not be vendored", file=sys.stderr)
        return 1

    payload = {
        "source": "QUDT (http://qudt.org/vocab/unit/), vendored by scripts/ontology/gen_units.py",
        "base_dimensions": sorted(set(BASE_DIMENSIONS.values())),
        "units": result["units"],
    }
    compiled = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    turtle = emit_ttl(result["units"])

    if args.check:
        stale = [p for p, new in ((JSON_OUT, compiled), (TTL_OUT, turtle)) if not p.exists() or p.read_text() != new]
        if stale:
            print(f"✗ stale: {', '.join(str(p.relative_to(REPO)) for p in stale)} — run `make gen-units`", file=sys.stderr)
            return 1
        print(f"✓ unit artifacts current ({len(result['units'])} units)")
        return 0

    JSON_OUT.write_text(compiled)
    TTL_OUT.write_text(turtle)
    curated = sum(1 for f in result["units"].values() if f["curated"])
    print(f"✓ {curated} curated units vendored ({len(result['units'])} total) → {TTL_OUT.name}, {JSON_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
