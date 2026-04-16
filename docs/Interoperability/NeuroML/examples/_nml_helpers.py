"""Shared helpers for NeuroML comparison QMD notebooks.

Running a NeuroML reference example via jNeuroML and comparing against TVBO.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# NeuroML2 examples — auto-resolved from env, local checkout, or GitHub clone.
_NML2_REPO = "https://github.com/NeuroML/NeuroML2.git"
_NML2_BRANCH = "master"
_CACHE_DIR = Path(os.environ.get("TVBO_CACHE_DIR", Path.home() / ".cache" / "tvbo"))


def _resolve_nml2_root() -> Path:
    """Find or fetch the NeuroML2 repository.

    Resolution order:
    1. ``NEUROML2_DIR`` environment variable (explicit override)
    2. Auto-clone to ``~/.cache/tvbo/NeuroML2`` (works anywhere with git)
    """
    env_dir = os.environ.get("NEUROML2_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "LEMSexamples").is_dir():
            return p
        raise FileNotFoundError(
            f"NEUROML2_DIR={env_dir} does not contain LEMSexamples/"
        )

    cached = _CACHE_DIR / "NeuroML2"
    if (cached / "LEMSexamples").is_dir():
        return cached

    print(f"Cloning NeuroML2 reference repo to {cached} ...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", _NML2_BRANCH, _NML2_REPO, str(cached)],
        check=True, capture_output=True, text=True,
    )
    return cached


NML2_ROOT = _resolve_nml2_root()
LEMS_EXAMPLES = NML2_ROOT / "LEMSexamples"
NML_EXAMPLES = NML2_ROOT / "examples"


def run_lems_example(lems_file: str, cwd: str | Path | None = None) -> dict[str, np.ndarray]:
    """Run a LEMS XML file via jNeuroML and return {filename: array} for each .dat output.

    Parameters
    ----------
    lems_file : str
        Name of the LEMS file (e.g., 'LEMS_NML2_Ex9_FN.xml').
    cwd : path, optional
        Working directory.  Defaults to the LEMSexamples directory.

    Returns
    -------
    dict mapping output filename to (n_time, n_cols) numpy arrays.
    """
    from pyneuroml import JNEUROML_VERSION
    import pyneuroml
    jar_dir = Path(pyneuroml.__file__).parent / "lib"
    jar = jar_dir / f"jNeuroML-{JNEUROML_VERSION}-jar-with-dependencies.jar"

    if cwd is None:
        cwd = LEMS_EXAMPLES

    cwd = Path(cwd)

    # Ensure results directory exists and clear stale outputs from prior runs.
    results_dir = cwd / "results"
    results_dir.mkdir(exist_ok=True)
    output_globs = ("*.dat", "*.v.dat", "*.h5", "*.csv")
    for pat in output_globs:
        for f in results_dir.glob(pat):
            f.unlink()

    start_time = time.time()

    def _discover_neuron_home() -> str | None:
        """Resolve a usable NEURON home directory for jNeuroML NEURON mode."""
        env_home = os.environ.get("NEURON_HOME") or os.environ.get("NRNHOME")
        if env_home:
            return env_home

        try:
            import neuron  # type: ignore

            root = Path(neuron.__file__).resolve().parent / ".data"
            if (root / "bin" / "nrniv").exists():
                return str(root)
        except Exception:
            pass

        # Fall back to sys.prefix (venv root) when nrniv lives there
        candidate = Path(sys.prefix) / "bin" / "nrniv"
        if candidate.exists():
            return str(Path(sys.prefix))

        return None

    def _build_neuron_env() -> dict[str, str]:
        env = dict(os.environ)
        neuron_home = _discover_neuron_home()
        if neuron_home:
            env.setdefault("NEURON_HOME", neuron_home)
            env.setdefault("NRNHOME", neuron_home)
        # Ensure venv bin/ is on PATH so jnml and nrniv are discoverable
        venv_bin = str(Path(sys.prefix) / "bin")
        cur_path = env.get("PATH", "")
        if venv_bin not in cur_path.split(os.pathsep):
            env["PATH"] = venv_bin + os.pathsep + cur_path
        return env

    def _run(lems_name: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["java", "-jar", str(jar), str(lems_name), "-nogui"],
            capture_output=True, text=True, cwd=str(cwd), timeout=600,
        )

    def _run_neuron_backend(lems_name: str) -> subprocess.CompletedProcess:
        """Generate and execute a LEMS model through NEURON backend."""
        env = _build_neuron_env()

        jnml = shutil.which("jnml") or shutil.which(
            "jnml", path=str(Path(sys.prefix) / "bin")
        )
        if not jnml:
            return subprocess.CompletedProcess(
                args=["jnml", lems_name, "-neuron", "-nogui"],
                returncode=127,
                stdout="",
                stderr="jnml command not found in PATH",
            )

        generated = subprocess.run(
            [jnml, str(lems_name), "-neuron", "-nogui"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=600,
            env=env,
        )
        if generated.returncode != 0:
            return generated

        nrnivmodl = None
        nrnhome = env.get("NRNHOME") or env.get("NEURON_HOME")
        if nrnhome:
            candidate = Path(nrnhome) / "bin" / "nrnivmodl"
            if candidate.exists():
                nrnivmodl = str(candidate)
        if nrnivmodl is None:
            nrnivmodl = shutil.which("nrnivmodl")

        compile_stdout = ""
        compile_stderr = ""
        if nrnivmodl:
            compiled = subprocess.run(
                [nrnivmodl],
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=600,
                env=env,
            )
            compile_stdout = compiled.stdout
            compile_stderr = compiled.stderr
            if compiled.returncode != 0:
                return subprocess.CompletedProcess(
                    args=compiled.args,
                    returncode=compiled.returncode,
                    stdout="\n".join([generated.stdout, compile_stdout]).strip(),
                    stderr="\n".join([generated.stderr, compile_stderr]).strip(),
                )

        nrn_script = cwd / f"{Path(lems_name).stem}_nrn.py"
        if not nrn_script.exists():
            return subprocess.CompletedProcess(
                args=[sys.executable, nrn_script.name],
                returncode=1,
                stdout="\n".join([generated.stdout, compile_stdout]).strip(),
                stderr="\n".join(
                    [generated.stderr, compile_stderr, f"Generated NEURON script not found: {nrn_script.name}"]
                ).strip(),
            )

        ran = subprocess.run(
            [sys.executable, nrn_script.name],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=600,
            env=env,
        )
        return subprocess.CompletedProcess(
            args=ran.args,
            returncode=ran.returncode,
            stdout="\n".join([generated.stdout, compile_stdout, ran.stdout]).strip(),
            stderr="\n".join([generated.stderr, compile_stderr, ran.stderr]).strip(),
        )

    def _collect_outputs() -> dict[str, np.ndarray]:
        outputs = {}
        candidates = []
        for pat in output_globs:
            candidates.extend(results_dir.glob(pat))
            candidates.extend(cwd.glob(pat))

        fresh = [p for p in candidates if p.stat().st_mtime >= (start_time - 0.5)]
        if not fresh:
            for pat in output_globs:
                fresh.extend(results_dir.glob(pat))

        for out_file in sorted(fresh):
            # Skip binary outputs for now; this helper is trace-oriented.
            if out_file.suffix.lower() in {".h5"}:
                continue
            try:
                arr = np.loadtxt(str(out_file))
            except Exception:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            outputs[out_file.name] = arr

        return outputs

    def _inject_probe_output(src: Path, dst: Path) -> bool:
        """Create a temporary LEMS file with OutputFile from Display lines."""
        tree = ET.parse(src)
        root = tree.getroot()

        sim = root.find("Simulation")
        if sim is None:
            return False

        if sim.find("OutputFile") is not None or sim.find("EventOutputFile") is not None:
            return False

        quantities = []
        for disp in sim.findall("Display"):
            for line in disp.findall("Line"):
                q = line.attrib.get("quantity")
                if q and q not in quantities:
                    quantities.append(q)

        if not quantities:
            return False

        of = ET.SubElement(sim, "OutputFile", id="of_auto", fileName="results/auto.dat")
        for i, q in enumerate(quantities):
            ET.SubElement(of, "OutputColumn", id=f"c{i}", quantity=q)

        tree.write(dst, encoding="unicode")
        return True

    result = _run(lems_file)
    if result.returncode != 0:
        text = "\n".join([result.stdout or "", result.stderr or ""])
        needs_neuron = (
            "MULTICOMPARTMENTAL_CELL_MODEL" in text
            or "requires Neuron" in text
            or "Ex25" in lems_file
        )

        if needs_neuron:
            result = _run_neuron_backend(lems_file)

        if result.returncode != 0:
            merged = "\n".join([result.stdout or "", result.stderr or ""])
            raise RuntimeError(
                f"jNeuroML failed (rc={result.returncode}):\n{merged[-4000:]}"
            )

    outputs = _collect_outputs()

    # Some canonical examples only define Display lines and no OutputFile.
    # For those, auto-inject an OutputFile and rerun to extract traces.
    if not outputs:
        src = cwd / lems_file
        probe_name = f"__tvbo_probe__{Path(lems_file).name}"
        probe_path = cwd / probe_name
        try:
            if _inject_probe_output(src, probe_path):
                result = _run(probe_name)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"jNeuroML probe run failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
                    )
                outputs = _collect_outputs()
        finally:
            if probe_path.exists():
                probe_path.unlink()

    if not outputs:
        raise RuntimeError(f"jNeuroML produced no output .dat files for {lems_file}")

    return outputs



def compare_traces(
    ref_data: np.ndarray,
    tvbo_data: np.ndarray,
    ref_cols: list[str],
    tvbo_cols: list[str],
    time_col: int = 0,
    rtol: float = 0.05,
    atol: float = 1e-4,
):
    """Compare reference and TVBO traces, print metrics.

    Parameters
    ----------
    ref_data, tvbo_data : (n_time, n_cols) arrays
    ref_cols, tvbo_cols : column names (index 0 is time)
    time_col : which column is time (default 0)
    rtol, atol : tolerances for np.allclose
    """
    # Interpolate TVBO onto reference time grid
    from scipy.interpolate import interp1d

    ref_time = ref_data[:, time_col]
    tvbo_time = tvbo_data[:, time_col]

    results = {}
    for col_name in ref_cols:
        if col_name == 'time':
            continue
        ref_idx = ref_cols.index(col_name)
        if col_name not in tvbo_cols:
            print(f"  {col_name}: not found in TVBO output, skipping")
            continue
        tvbo_idx = tvbo_cols.index(col_name)

        ref_trace = ref_data[:, ref_idx]
        tvbo_trace_raw = tvbo_data[:, tvbo_idx]

        # Interpolate to common grid
        f_tvbo = interp1d(tvbo_time, tvbo_trace_raw, kind='linear',
                          fill_value='extrapolate')
        tvbo_trace = f_tvbo(ref_time)

        # Metrics
        rmse = np.sqrt(np.mean((ref_trace - tvbo_trace) ** 2))
        max_err = np.max(np.abs(ref_trace - tvbo_trace))
        corr = np.corrcoef(ref_trace, tvbo_trace)[0, 1] if np.std(ref_trace) > 0 else 1.0
        close = np.allclose(ref_trace, tvbo_trace, rtol=rtol, atol=atol)

        results[col_name] = {
            'rmse': rmse, 'max_err': max_err, 'corr': corr, 'close': close,
        }
        status = "✅" if close else "⚠️"
        print(f"  {col_name}: RMSE={rmse:.6f}  max_err={max_err:.6f}  "
              f"corr={corr:.6f}  {status}")

    return results


def plot_comparison(
    ref_data: np.ndarray,
    tvbo_data: np.ndarray,
    ref_cols: list[str],
    tvbo_cols: list[str],
    title: str = "",
    time_scale: float = 1.0,
    time_unit: str = "s",
):
    """Plot overlaid traces: reference vs TVBO.

    Parameters
    ----------
    ref_data, tvbo_data : arrays with time in col 0
    ref_cols, tvbo_cols : column names
    title : plot title
    time_scale : multiply time by this factor for display
    time_unit : label for x axis
    """
    import matplotlib.pyplot as plt

    sv_names = [c for c in ref_cols if c != 'time' and c in tvbo_cols]
    n = len(sv_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False, sharex=True)

    ref_time = ref_data[:, 0] * time_scale
    tvbo_time = tvbo_data[:, 0] * time_scale

    for i, name in enumerate(sv_names):
        ax = axes[i, 0]
        ref_idx = ref_cols.index(name)
        tvbo_idx = tvbo_cols.index(name)

        ax.plot(ref_time, ref_data[:, ref_idx], label=f'NeuroML (ref)', alpha=0.8)
        ax.plot(tvbo_time, tvbo_data[:, tvbo_idx], '--', label='TVBO', alpha=0.8)
        ax.set_ylabel(name)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(f"Time ({time_unit})")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


# ── Display-aware comparison plotting ────────────────────────────────


@dataclass
class _DisplayLine:
    """One <Line> inside a <Display>."""
    line_id: str
    quantity: str
    color: str
    scale: str


@dataclass
class _Display:
    """One <Display> element from a LEMS Simulation."""
    display_id: str
    title: str
    time_scale: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    lines: list[_DisplayLine]


def parse_lems_displays(lems_file: str) -> list[_Display]:
    """Parse Display + Line elements from a LEMS XML file.

    Parameters
    ----------
    lems_file : str
        LEMS filename (e.g. 'LEMS_NML2_Ex9_FN.xml') resolved relative
        to the LEMSexamples directory.

    Returns
    -------
    list of _Display objects with their Line children.
    """
    path = LEMS_EXAMPLES / lems_file
    tree = ET.parse(path)
    root = tree.getroot()
    sim = root.find("Simulation")
    if sim is None:
        return []

    displays = []
    for disp in sim.findall("Display"):
        lines = []
        for ln in disp.findall("Line"):
            lines.append(_DisplayLine(
                line_id=ln.attrib.get("id", ""),
                quantity=ln.attrib.get("quantity", ""),
                color=ln.attrib.get("color", "#000000"),
                scale=ln.attrib.get("scale", "1"),
            ))
        displays.append(_Display(
            display_id=disp.attrib.get("id", ""),
            title=disp.attrib.get("title", ""),
            time_scale=disp.attrib.get("timeScale", "1ms"),
            xmin=float(disp.attrib.get("xmin", "0")),
            xmax=float(disp.attrib.get("xmax", "100")),
            ymin=float(disp.attrib.get("ymin", "-80")),
            ymax=float(disp.attrib.get("ymax", "40")),
            lines=lines,
        ))
    return displays


def _match_quantity_to_col(quantity: str, tvbo_cols: list[str]) -> str | None:
    """Best-effort match a LEMS quantity path to a TVBO column name.

    LEMS uses e.g. ``izpopBurst[0]/v`` while TVBO produces
    ``izBurst_pop[0]/v``.  We try progressively looser matching.
    """
    # Direct match
    if quantity in tvbo_cols:
        return quantity

    # Extract the variable part after last /
    parts = quantity.rsplit("/", 1)
    var_suffix = parts[-1] if len(parts) > 1 else quantity

    # Try matching by suffix
    candidates = [c for c in tvbo_cols if c.endswith("/" + var_suffix)]
    if len(candidates) == 1:
        return candidates[0]

    # Try matching by population index pattern: pop[N]/var
    idx_match = re.search(r'\[(\d+)\]', quantity)
    if idx_match:
        idx = idx_match.group(0)
        candidates = [c for c in tvbo_cols
                      if idx in c and c.endswith("/" + var_suffix)]
        if len(candidates) == 1:
            return candidates[0]

    return None


def _scale_factor(scale_str: str) -> float:
    """Convert a LEMS scale string like '1mV' to a numeric factor."""
    scale_str = scale_str.strip()
    unit_factors = {
        'V': 1.0, 'mV': 1e3, 'uV': 1e6,
        'A': 1.0, 'nA': 1e9, 'pA': 1e12, 'uA': 1e6,
        'S': 1.0, 'nS': 1e9, 'uS': 1e6,
        'ms': 1e3, 's': 1.0,
        'Hz': 1.0,
    }
    for unit, factor in sorted(unit_factors.items(), key=lambda x: -len(x[0])):
        if scale_str.endswith(unit):
            num = scale_str[:-len(unit)].strip()
            num_val = float(num) if num else 1.0
            return num_val * factor
    try:
        return float(scale_str)
    except ValueError:
        return 1.0


def _find_ref_column(quantity: str, ref_outputs: dict,
                     output_columns: dict | None = None
                     ) -> tuple[str, int] | None:
    """Find which reference .dat file and column index contains a quantity.

    Parameters
    ----------
    quantity : str
        LEMS quantity path e.g. ``iafPop[0]/v``
    ref_outputs : dict
        {filename: array} from run_lems_example()
    output_columns : dict, optional
        {filename: [quantity_strings]} parsed from OutputFile/OutputColumn
        If None, uses positional order.

    Returns
    -------
    (filename, col_index) or None
    """
    if output_columns:
        for fname, cols in output_columns.items():
            if quantity in cols:
                idx = cols.index(quantity) + 1  # +1 because col 0 is time
                if fname in ref_outputs and idx < ref_outputs[fname].shape[1]:
                    return (fname, idx)

    # Fallback: try to match by position across all output files
    return None


def parse_lems_output_columns(lems_file: str) -> dict[str, list[str]]:
    """Parse OutputFile → OutputColumn quantities from a LEMS file.

    Returns
    -------
    dict mapping output filename (e.g. 'ex14.dat') to list of quantity strings.
    """
    path = LEMS_EXAMPLES / lems_file
    tree = ET.parse(path)
    root = tree.getroot()
    sim = root.find("Simulation")
    if sim is None:
        return {}

    result = {}
    for of in sim.findall("OutputFile"):
        fname = of.attrib.get("fileName", "")
        # Strip path prefixes
        if fname.startswith("./"):
            fname = fname[2:]
        if fname.startswith("results/"):
            fname = fname[len("results/"):]
        quantities = [oc.attrib.get("quantity", "")
                      for oc in of.findall("OutputColumn")]
        result[fname] = quantities
    return result


def plot_lems_comparison(
    lems_file: str,
    ref_outputs: dict[str, np.ndarray],
    tvbo_result=None,
    title_prefix: str = "",
):
    """Create publication-quality comparison plots mirroring LEMS Display layout.

    For each Display in the LEMS file, creates one subplot panel with:
    - Reference traces as solid lines (using original colors from LEMS)
    - TVBO traces as dashed lines (same color, slightly transparent)

    Parameters
    ----------
    lems_file : str
        Reference LEMS filename (e.g. 'LEMS_NML2_Ex2_Izh.xml')
    ref_outputs : dict
        {filename: array} from run_lems_example()
    tvbo_result : xarray.DataArray, optional
        ``result.integration.data`` from ``exp.run("neuroml")``.
        Expects dims ``(time, quantity)``.  If None, only reference is plotted.
    title_prefix : str, optional
        Prefix for figure titles (e.g. 'Ex2')
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    displays = parse_lems_displays(lems_file)
    output_cols = parse_lems_output_columns(lems_file)
    if not displays:
        return

    # Build auto.dat positional map: Display lines → column indices
    # When no explicit OutputFile exists, jNeuroML writes Display quantities
    # to auto.dat in order of appearance.
    auto_position = {}
    col_counter = 1  # col 0 is time
    for d in displays:
        for ln in d.lines:
            q = ln.quantity
            if q not in auto_position:
                auto_position[q] = col_counter
                col_counter += 1

    # Build ref lookup: quantity → (filename, col_idx)
    ref_lookup: dict[str, tuple[str, int]] = {}
    if output_cols:
        for fname, qs in output_cols.items():
            for i, q in enumerate(qs):
                ref_lookup[q] = (fname, i + 1)
    # For auto.dat fallback, use positional mapping
    if not ref_lookup and "auto.dat" in ref_outputs:
        for q, idx in auto_position.items():
            if idx < ref_outputs["auto.dat"].shape[1]:
                ref_lookup[q] = ("auto.dat", idx)

    # Extract TVBO data from xarray DataArray
    tvbo_time = None
    tvbo_lookup: dict[str, np.ndarray] = {}  # col_name → 1-D array
    if tvbo_result is not None:
        tvbo_time = tvbo_result.coords['time'].values
        qty_dim = [d for d in tvbo_result.dims if d != 'time'][0]
        for col_name in tvbo_result.coords[qty_dim].values:
            tvbo_lookup[str(col_name)] = tvbo_result.sel({qty_dim: col_name}).values

    # Positional TVBO matching: for each variable name (e.g. 'v'), track
    # consumption order to handle multiple populations
    _tvbo_by_var: dict[str, list[str]] = {}
    for col_name in tvbo_lookup:
        var = col_name.rsplit("/", 1)[-1] if "/" in col_name else col_name
        _tvbo_by_var.setdefault(var, []).append(col_name)
    _tvbo_consumed: dict[str, int] = {}

    for display in displays:
        if not display.lines:
            continue

        fig, ax = plt.subplots(figsize=(10, 3.5))
        has_ref = False
        has_tvbo = False

        for line in display.lines:
            color = line.color
            if not color.startswith('#'):
                color = f'#{color}'
            scale = _scale_factor(line.scale)
            label = line.line_id or line.quantity.split("/")[-1]

            # Find reference data
            ref_info = ref_lookup.get(line.quantity)
            if ref_info:
                fname, col_idx = ref_info
                ref_arr = ref_outputs[fname]
                t_ref = ref_arr[:, 0] * _scale_factor(display.time_scale)
                y_ref = ref_arr[:, col_idx] * scale
                ax.plot(t_ref, y_ref, color=color, alpha=0.9,
                        linewidth=1.5, label=f'{label} (ref)')
                has_ref = True

            # Find matching TVBO column — try direct match, then positional
            tvbo_col = _match_quantity_to_col(line.quantity, list(tvbo_lookup.keys()))
            if not tvbo_col:
                var = line.quantity.rsplit("/", 1)[-1] if "/" in line.quantity else line.quantity
                candidates = _tvbo_by_var.get(var, [])
                pos = _tvbo_consumed.get(var, 0)
                if pos < len(candidates):
                    tvbo_col = candidates[pos]
                    _tvbo_consumed[var] = pos + 1

            if tvbo_col and tvbo_time is not None:
                t_tvbo = tvbo_time * _scale_factor(display.time_scale)
                y_tvbo = tvbo_lookup[tvbo_col] * scale
                ax.plot(t_tvbo, y_tvbo, color=color, alpha=0.5,
                        linewidth=1.5, linestyle='--', label=f'{label} (TVBO)')
                has_tvbo = True

        title = display.title
        if title_prefix:
            title = f"{title_prefix}: {title}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"Time ({display.time_scale})")
        ax.set_ylim(display.ymin, display.ymax)
        ax.set_xlim(display.xmin, display.xmax)
        ax.grid(True, alpha=0.2)

        legend_handles = []
        if has_ref:
            legend_handles.append(
                Line2D([0], [0], color='gray', linewidth=1.5,
                       label='NeuroML ref (solid)'))
        if has_tvbo:
            legend_handles.append(
                Line2D([0], [0], color='gray', linewidth=1.5,
                       linestyle='--', alpha=0.6, label='TVBO (dashed)'))
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

        fig.tight_layout()
        plt.show()
