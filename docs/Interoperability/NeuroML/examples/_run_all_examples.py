#!/usr/bin/env python
"""Run NeuroML Ex0-Ex27 QMD examples through TVBO and compare to jNeuroML.

Usage:
    python _run_all_examples.py
    python _run_all_examples.py Ex0 Ex1 Ex22

This runner parses each Ex*.qmd file to extract:
- the TVBO YAML passed to SimulationExperiment.from_string(...)
- the reference LEMS filename passed to run_lems_example(...)

It then executes:
1. SimulationExperiment.render("lems")
2. SimulationExperiment.run("neuroml")
3. jNeuroML reference run
4. Automatic best-match numeric comparison for each TVBO state variable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from tvbo import SimulationExperiment
from tvbo.adapters.neuroml import LEMS_EXAMPLES, run_lems_example

EXAMPLE_FILE_RE = re.compile(r"^Ex(?P<num>\d+)_.*\.qmd$")
LEMS_RE = re.compile(r"run_lems_example\((['\"])(?P<lems>[^'\"]+)\1\)")
FROM_STRING_RE = re.compile(
    r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*SimulationExperiment\.from_string\(\s*\"\"\"(?P<yaml>.*?)\"\"\"\s*\)",
    re.DOTALL,
)


@dataclass
class ParsedExample:
    name: str
    qmd_path: Path
    lems_file: str
    yaml_text: str
    source_var: str


def _example_order(name: str) -> int:
    return int(name[2:])


def _canonical_xml(text: str) -> str:
    try:
        return ET.canonicalize(xml_data=text)
    except Exception:
        return re.sub(r"\s+", " ", text).strip()


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _lems_signature(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)

    def is_name(elem: ET.Element, *names: str) -> bool:
        name = _tag_name(elem.tag)
        return name in names

    includes: list[str] = []
    targets: list[dict] = []
    pulses: list[dict] = []
    sims: list[dict] = []
    nets: list[dict] = []
    counts = Counter(_tag_name(e.tag) for e in root.iter())

    for elem in root.iter():
        name = _tag_name(elem.tag)

        if is_name(elem, "Include"):
            includes.append(elem.attrib.get("file", ""))

        if is_name(elem, "Target"):
            targets.append(
                {
                    "component": elem.attrib.get("component", ""),
                    "reportFile": elem.attrib.get("reportFile", ""),
                }
            )

        if "pulsegenerator" in name.lower():
            pulses.append(
                {
                    "tag": name,
                    "id": elem.attrib.get("id", ""),
                    "delay": elem.attrib.get("delay", ""),
                    "duration": elem.attrib.get("duration", ""),
                    "amplitude": elem.attrib.get("amplitude", ""),
                }
            )

        if is_name(elem, "Simulation"):
            if not elem.attrib.get("id"):
                continue

            outputs: list[dict] = []
            displays: list[dict] = []

            for child in list(elem):
                child_name = _tag_name(child.tag)
                if child_name == "OutputFile":
                    cols = []
                    for col in list(child):
                        if _tag_name(col.tag) == "OutputColumn":
                            cols.append(col.attrib.get("quantity", ""))
                    outputs.append(
                        {
                            "id": child.attrib.get("id", ""),
                            "fileName": child.attrib.get("fileName", ""),
                            "quantities": sorted(cols),
                        }
                    )

                if child_name == "Display":
                    lines = []
                    for line in list(child):
                        if _tag_name(line.tag) == "Line":
                            lines.append(line.attrib.get("quantity", ""))
                    displays.append(
                        {
                            "id": child.attrib.get("id", ""),
                            "title": child.attrib.get("title", ""),
                            "quantities": sorted(lines),
                        }
                    )

            sims.append(
                {
                    "id": elem.attrib.get("id", ""),
                    "target": elem.attrib.get("target", ""),
                    "length": elem.attrib.get("length", ""),
                    "step": elem.attrib.get("step", ""),
                    "outputs": sorted(outputs, key=lambda d: (d["id"], d["fileName"])),
                    "displays": sorted(displays, key=lambda d: (d["id"], d["title"])),
                }
            )

        if is_name(elem, "network", "Network"):
            populations = []
            explicit_inputs = []
            for child in list(elem):
                child_name = _tag_name(child.tag)
                if child_name in {"population", "Population"}:
                    populations.append(
                        {
                            "id": child.attrib.get("id", ""),
                            "component": child.attrib.get("component", ""),
                            "size": child.attrib.get("size", ""),
                        }
                    )
                if child_name in {"explicitInput", "ExplicitInput"}:
                    explicit_inputs.append(
                        {
                            "target": child.attrib.get("target", ""),
                            "input": child.attrib.get("input", ""),
                            "destination": child.attrib.get("destination", ""),
                        }
                    )

            nets.append(
                {
                    "id": elem.attrib.get("id", ""),
                    "populations": sorted(populations, key=lambda d: d["id"]),
                    "explicitInputs": sorted(
                        explicit_inputs,
                        key=lambda d: (d["target"], d["input"], d["destination"]),
                    ),
                }
            )

    return {
        "includes": sorted(includes),
        "targets": sorted(targets, key=lambda d: (d["component"], d["reportFile"])),
        "pulse_generators": sorted(pulses, key=lambda d: (d["id"], d["tag"])),
        "simulations": sorted(sims, key=lambda d: d["id"]),
        "networks": sorted(nets, key=lambda d: d["id"]),
        "element_counts": dict(sorted(counts.items())),
    }


def _compare_lems_structure(rendered_xml: str, ref_xml: str) -> dict:
    try:
        rendered = _lems_signature(rendered_xml)
        reference = _lems_signature(ref_xml)
    except Exception as exc:
        return {
            "structural_match": False,
            "mismatch_count": 1,
            "mismatches": [{"key": "parse_error", "error": str(exc)}],
        }

    mismatches = []
    keys = ["includes", "targets", "pulse_generators", "networks", "simulations"]
    for key in keys:
        ref_value = reference[key]
        rendered_value = rendered[key]
        if rendered_value != ref_value:
            mismatch: dict[str, object] = {"key": key}
            if isinstance(ref_value, list) and isinstance(rendered_value, list):
                mismatch.update(
                    {
                        "reference_count": len(ref_value),
                        "rendered_count": len(rendered_value),
                        "reference_sample": ref_value[:3],
                        "rendered_sample": rendered_value[:3],
                    }
                )
            else:
                mismatch.update(
                    {
                        "reference": ref_value,
                        "rendered": rendered_value,
                    }
                )
            mismatches.append(mismatch)

    count_diffs = []
    count_keys = sorted(set(rendered["element_counts"]) | set(reference["element_counts"]))
    for key in count_keys:
        r_count = int(reference["element_counts"].get(key, 0))
        t_count = int(rendered["element_counts"].get(key, 0))
        if r_count != t_count:
            count_diffs.append({"tag": key, "reference": r_count, "rendered": t_count})
    if count_diffs:
        count_diffs.sort(key=lambda d: abs(d["reference"] - d["rendered"]), reverse=True)
        mismatches.append(
            {
                "key": "element_counts",
                "difference_count": len(count_diffs),
                "differences": count_diffs[:25],
            }
        )

    return {
        "structural_match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 1.0 if np.allclose(a, b) else 0.0
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 and b_std < 1e-12:
        return 1.0
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(corr):
        return -1.0
    return corr


def _pick_yaml_block(example_name: str, blocks: list[tuple[str, str]]) -> tuple[str, str]:
    for var, yaml_text in blocks:
        if var == "exp":
            return var, yaml_text

    if example_name == "Ex2":
        for var, yaml_text in blocks:
            if "tonic" in var.lower():
                return var, yaml_text

    return blocks[0]


def parse_qmd_example(qmd_path: Path) -> ParsedExample:
    text = qmd_path.read_text(encoding="utf-8")

    lems_matches = [m.group("lems") for m in LEMS_RE.finditer(text)]
    if not lems_matches:
        raise ValueError(f"No run_lems_example(...) call found in {qmd_path.name}")

    blocks = [(m.group("var"), m.group("yaml")) for m in FROM_STRING_RE.finditer(text)]
    if not blocks:
        raise ValueError(f"No SimulationExperiment.from_string(...) block found in {qmd_path.name}")

    stem_match = EXAMPLE_FILE_RE.match(qmd_path.name)
    if not stem_match:
        raise ValueError(f"Unexpected example filename: {qmd_path.name}")

    name = f"Ex{int(stem_match.group('num'))}"
    source_var, yaml_text = _pick_yaml_block(name, blocks)

    return ParsedExample(
        name=name,
        qmd_path=qmd_path,
        lems_file=lems_matches[0],
        yaml_text=yaml_text.strip(),
        source_var=source_var,
    )


def discover_examples(examples_dir: Path) -> dict[str, ParsedExample]:
    parsed: dict[str, ParsedExample] = {}
    for path in sorted(examples_dir.glob("Ex*.qmd")):
        item = parse_qmd_example(path)
        parsed[item.name] = item
    return parsed


def _parse_lems_output_columns(xml_str: str) -> dict[str, list[str]]:
    """Extract declared output quantities from LEMS XML.

    Returns {fileName: [quantity_0, quantity_1, …]} preserving column order.

    Checks OutputFile/OutputColumn first.  If none are found, falls back to
    Display/Line quantities (which ``_inject_probe_output`` would write to
    ``auto.dat``).
    """
    root = ET.fromstring(xml_str)
    result: dict[str, list[str]] = {}
    for sim in root.iter("Simulation"):
        for of in sim.iter("OutputFile"):
            fname = of.get("fileName", "")
            fname = fname.replace("\\", "/")
            if fname.startswith("./"):
                fname = fname[2:]
            if fname.startswith("results/"):
                fname = fname[len("results/") :]
            quantities = []
            for oc in of.iter("OutputColumn"):
                q = oc.get("quantity", "")
                if q:
                    quantities.append(q)
            if quantities:
                result[fname] = quantities

        # Fallback: Display/Line → auto.dat (mirrors _inject_probe_output)
        if not result:
            quantities = []
            for disp in sim.iter("Display"):
                for line in disp.iter("Line"):
                    q = line.get("quantity", "")
                    if q and q not in quantities:
                        quantities.append(q)
            if quantities:
                result["auto.dat"] = quantities

    return result


def _var_name(quantity: str) -> str:
    """Extract the bare variable name from a LEMS quantity path.

    ``pop[0]/v`` → ``v``, ``izpopTonic[0]/i1/I`` → ``I``.
    """
    return quantity.rsplit("/", 1)[-1]


def _match_tvbo_to_reference(
    tvbo_arr: np.ndarray,
    tvbo_var_names: list[str],
    ref_outputs: dict[str, np.ndarray],
    tvbo_xml: str | None = None,
    ref_xml: str | None = None,
) -> list[dict]:
    """Match TVBO output columns to reference columns by variable name.

    Parses OutputColumn (or Display/Line) quantities from both LEMS XMLs
    and matches by the bare variable name (last path segment).
    No post-hoc scaling is applied: outputs must already be in matching units.
    """
    mappings: list[dict] = []

    tvbo_time = tvbo_arr[:, 0]
    if tvbo_time.size < 2:
        raise ValueError("TVBO output time axis has fewer than 2 points")

    # ── Parse output quantities from both sides ──
    tvbo_outs = _parse_lems_output_columns(tvbo_xml) if tvbo_xml else {}
    ref_outs = _parse_lems_output_columns(ref_xml) if ref_xml else {}

    # Build ref index: var_name → list of (ref_file, col_idx, full_quantity)
    # Preserves order for multi-population matching.
    ref_by_var: dict[str, list[tuple[str, int, str]]] = {}
    for ref_file, quantities in ref_outs.items():
        for col_idx, q in enumerate(quantities, start=1):
            vn = _var_name(q)
            ref_by_var.setdefault(vn, []).append((ref_file, col_idx, q))

    # Build TVBO quantity map: tvbo_var_name → col_idx (1-based)
    tvbo_quantity_map: dict[str, int] = {}
    for _tvbo_file, tvbo_quantities in tvbo_outs.items():
        for col_idx, q in enumerate(tvbo_quantities, start=1):
            vn = _var_name(q)
            tvbo_quantity_map[vn] = col_idx

    # Restrict to voltage-like variables for comparison
    comparable = {str(v).lower() for v in ("v", "vs", "vd")}

    # Track how many times each variable name has been consumed for
    # positional matching when there are multiple populations.
    _var_consumed: dict[str, int] = {}

    for tvbo_col_idx, var_name in enumerate(tvbo_var_names, start=1):
        base_var = var_name.rsplit("/", 1)[-1] if "/" in var_name else var_name
        if base_var.lower() not in comparable:
            continue

        tvbo_trace = tvbo_arr[:, tvbo_col_idx]

        # Match by variable name, positionally when duplicates exist
        ref_entries = ref_by_var.get(base_var)
        if not ref_entries:
            mappings.append(
                {
                    "tvbo_var": var_name,
                    "skip": True,
                    "reason": f"no reference column for variable '{base_var}'",
                }
            )
            continue

        pos = _var_consumed.get(base_var, 0)
        if pos >= len(ref_entries):
            mappings.append(
                {
                    "tvbo_var": var_name,
                    "skip": True,
                    "reason": f"more TVBO '{base_var}' columns than reference",
                }
            )
            continue
        _var_consumed[base_var] = pos + 1

        ref_file, ref_col_idx, _ref_q = ref_entries[pos]
        ref_arr = ref_outputs.get(ref_file)
        if ref_arr is None or ref_arr.ndim != 2 or ref_arr.shape[1] <= ref_col_idx:
            mappings.append(
                {
                    "tvbo_var": var_name,
                    "skip": True,
                    "reason": f"reference file '{ref_file}' missing or too few columns",
                }
            )
            continue

        ref_time = ref_arr[:, 0]
        ref_trace = ref_arr[:, ref_col_idx]

        # Interpolate TVBO onto reference time grid
        interp_fn = interp1d(
            tvbo_time,
            tvbo_trace,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        tvbo_on_ref = interp_fn(ref_time)

        rmse = float(np.sqrt(np.mean((ref_trace - tvbo_on_ref) ** 2)))
        ref_span = float(np.ptp(ref_trace))
        nrmse = rmse / ref_span if ref_span > 1e-12 else rmse
        corr = _safe_corr(ref_trace, tvbo_on_ref)

        mappings.append(
            {
                "tvbo_var": var_name,
                "tvbo_col": tvbo_col_idx,
                "ref_file": ref_file,
                "ref_col": ref_col_idx,
                "corr": corr,
                "rmse": rmse,
                "nrmse": nrmse,
                "match_type": "name",
            }
        )

    return mappings


def run_one(parsed: ParsedExample, strict_canonical: bool = False) -> dict:
    print(f"\n{'=' * 76}")
    print(f"{parsed.name}  ({parsed.qmd_path.name})")
    print(f"{'=' * 76}")

    exp = SimulationExperiment.from_string(parsed.yaml_text)
    dyn_name = (
        exp.dynamics.name
        if exp.dynamics
        else (next(iter(exp.network.dynamics)) if exp.network and exp.network.dynamics else "network")
    )
    print(f"TVBO model: {dyn_name}  (source var: {parsed.source_var})")

    rendered_xml = exp.render("lems")
    ref_lems_path = LEMS_EXAMPLES / parsed.lems_file
    ref_xml = ref_lems_path.read_text(encoding="utf-8")
    xml_exact_match = _canonical_xml(rendered_xml) == _canonical_xml(ref_xml)
    structure = _compare_lems_structure(rendered_xml, ref_xml)
    xml_structural_match = bool(structure["structural_match"])
    print(f"LEMS canonical exact match: {xml_exact_match}")
    print(f"LEMS structural match: {xml_structural_match}  mismatches={structure['mismatch_count']}")
    for mismatch in structure["mismatches"][:3]:
        print(f"  - structure mismatch: {mismatch['key']}")

    # Detect multicompartmental cells → need NEURON backend (jLEMS can't handle them)
    _needs_neuron = "neuroml:segment" in parsed.yaml_text
    _backend = "neuron" if _needs_neuron else "jneuroml"
    if _needs_neuron:
        print("Using NEURON backend (multicompartmental cell detected)")
    result = exp.run("neuroml", backend=_backend)
    da = result.integration.data

    tvbo_time = da.coords["time"].values
    is_multi_pop = "quantity" in da.dims
    if is_multi_pop:
        # Multi-population network: dims are (time, quantity)
        tvbo_values = da.values
        tvbo_var_names = [str(q) for q in da.coords["quantity"].values]
    else:
        # Single-population: dims are (time, variable, node)
        tvbo_values = da.values
        if tvbo_values.ndim == 1:
            tvbo_values = tvbo_values[:, None]
        if "variable" in da.coords:
            tvbo_var_names = [str(v) for v in da.coords["variable"].values]
        else:
            tvbo_var_names = [f"sv_{i}" for i in range(tvbo_values.shape[1])]

    tvbo_arr = np.column_stack([tvbo_time, tvbo_values])
    print(f"TVBO output shape: {tvbo_arr.shape}, vars={tvbo_var_names}")

    ref_outputs = run_lems_example(parsed.lems_file)
    print(f"Reference outputs: {', '.join(ref_outputs.keys())}")

    mappings = _match_tvbo_to_reference(
        tvbo_arr,
        tvbo_var_names,
        ref_outputs,
        tvbo_xml=rendered_xml,
        ref_xml=ref_xml,
    )

    corr_values: list[float] = []
    nrmse_values: list[float] = []
    for m in mappings:
        if "error" in m:
            print(f"  {m['tvbo_var']}: ERROR ({m['error']})")
            continue
        if m.get("skip"):
            print(f"  {m['tvbo_var']}: SKIP  ({m.get('reason', 'extra output')})")
            continue
        corr_values.append(float(m["corr"]))
        nrmse_values.append(float(m["nrmse"]))
        status = "PASS" if m["corr"] >= 0.99 else ("WARN" if m["corr"] >= 0.95 else "FAIL")
        match_type = m.get("match_type", "?")
        print(
            f"  {m['tvbo_var']}: {status}  corr={m['corr']:.6f}  "
            f"nrmse={m['nrmse']:.6f}  rmse={m['rmse']:.6g}  "
            f"ref={m['ref_file']}[col {m['ref_col']}]  "
            f"match={match_type}"
        )

    if corr_values:
        # For multi-pop: TVBO may record more voltage variables than the
        # reference (e.g. pre-cell + post-cells while ref only has post-cells).
        # Only score the top N best-matching variables where N = total ref cols.
        if is_multi_pop:
            n_ref_cols = sum(max(0, arr.shape[1] - 1) for arr in ref_outputs.values() if arr.ndim == 2)
            if n_ref_cols > 0 and len(corr_values) > n_ref_cols:
                scored = sorted(zip(corr_values, nrmse_values, strict=True), key=lambda x: -x[0])[:n_ref_cols]
                corr_values = [c for c, _ in scored]
                nrmse_values = [n for _, n in scored]
        worst_corr = float(min(corr_values))
        worst_nrmse = float(max(nrmse_values))
    else:
        worst_corr = -1.0
        worst_nrmse = float("inf")

    status = "PASS" if worst_corr >= 0.99 else "FAIL"
    if strict_canonical and not (xml_exact_match and xml_structural_match):
        status = "FAIL"

    return {
        "name": parsed.name,
        "qmd": parsed.qmd_path.name,
        "source_var": parsed.source_var,
        "lems_file": parsed.lems_file,
        "xml_exact_match": xml_exact_match,
        "xml_structural_match": xml_structural_match,
        "xml_structure_mismatches": structure["mismatches"],
        "tvbo_var_names": tvbo_var_names,
        "mappings": mappings,
        "worst_corr": worst_corr,
        "worst_nrmse": worst_nrmse,
        "status": status,
    }


def write_markdown_report(results: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# NeuroML Ex0-Ex27 Full Run Report")
    lines.append("")
    lines.append("| Example | Status | Worst Corr | Worst NRMSE | Canonical XML Exact | Canonical Structure Match |")
    lines.append("|---|---|---:|---:|---|---|")

    for item in results:
        if item.get("status") == "ERROR":
            lines.append(f"| {item['name']} | ERROR | - | - | - | - |")
            continue

        lines.append(
            f"| {item['name']} | {item['status']} | "
            f"{item['worst_corr']:.6f} | {item['worst_nrmse']:.6f} | "
            f"{str(item['xml_exact_match'])} | {str(item.get('xml_structural_match', False))} |"
        )

    lines.append("")
    lines.append("## Per-Variable Mapping")
    lines.append("")

    for item in results:
        lines.append(f"### {item['name']}")
        if item.get("status") == "ERROR":
            lines.append(f"- Error: {item['error']}")
            lines.append("")
            continue

        lines.append(f"- QMD: {item['qmd']}")
        lines.append(f"- LEMS: {item['lems_file']}")
        lines.append(f"- XML exact canonical match: {item['xml_exact_match']}")
        lines.append(f"- XML structural canonical match: {item.get('xml_structural_match', False)}")
        for mismatch in item.get("xml_structure_mismatches", []):
            lines.append(f"- Structural mismatch: {mismatch['key']}")

        for mapping in item["mappings"]:
            if "error" in mapping:
                lines.append(f"- {mapping['tvbo_var']}: ERROR ({mapping['error']})")
            elif mapping.get("skip"):
                lines.append(f"- {mapping['tvbo_var']}: SKIP ({mapping.get('reason', 'extra output')})")
            else:
                lines.append(
                    f"- {mapping['tvbo_var']}: corr={mapping['corr']:.6f}, "
                    f"nrmse={mapping['nrmse']:.6f}, "
                    f"ref={mapping['ref_file']}[col {mapping['ref_col']}]"
                )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        help="Subset to run, e.g. Ex0 Ex1 Ex22 (default: all Ex0-Ex27)",
    )
    parser.add_argument(
        "--json",
        default="_run_all_examples_results.json",
        help="Path for JSON report (default: _run_all_examples_results.json)",
    )
    parser.add_argument(
        "--md",
        default="_run_all_examples_results.md",
        help="Path for markdown report (default: _run_all_examples_results.md)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first failing example",
    )
    parser.add_argument(
        "--strict-canonical",
        action="store_true",
        help="Require both exact and structural canonical XML parity for PASS",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    here = Path(__file__).resolve().parent
    all_examples = discover_examples(here)

    if args.examples:
        selected: list[str] = []
        for token in args.examples:
            key = token.split("_")[0]
            if key not in all_examples:
                print(f"Unknown example: {token}")
                return 2
            selected.append(key)
        run_keys = sorted(set(selected), key=_example_order)
    else:
        run_keys = sorted(all_examples.keys(), key=_example_order)

    results: list[dict] = []

    for key in run_keys:
        parsed = all_examples[key]
        try:
            result = run_one(parsed, strict_canonical=args.strict_canonical)
            results.append(result)
            if args.fail_fast and result["status"] != "PASS":
                print("Fail-fast enabled: stopping after first failure.")
                break
        except Exception as exc:
            traceback.print_exc()
            error_item = {
                "name": key,
                "qmd": parsed.qmd_path.name,
                "status": "ERROR",
                "error": str(exc),
            }
            results.append(error_item)
            if args.fail_fast:
                print("Fail-fast enabled: stopping after first exception.")
                break

    json_path = (here / args.json).resolve()
    md_path = (here / args.md).resolve()

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown_report(results, md_path)

    print(f"\nJSON report written to: {json_path}")
    print(f"Markdown report written to: {md_path}")

    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_err = sum(1 for r in results if r.get("status") == "ERROR")

    print("\nSummary")
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    print(f"  ERROR: {n_err}")

    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
