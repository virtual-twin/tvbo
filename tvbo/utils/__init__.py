#  utils.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
Utilities Module for TVB-O
==========================

Core utilities: ``Bunch`` container, PyTree formatting, YAML I/O,
and metadata traversal helpers.

Plotting utilities (colors, colormaps, ``multiview``) have moved to
``tvbo.plot.utils`` and are re-exported here for backward compatibility.

Analysis functions (``per_window_fc``, ``ttest_correlation_strength``) have
moved to ``tvbo.analysis``.
"""

from os.path import abspath, dirname, join

import numpy as np

cm = 1 / 2.54
ROOT_DIR = abspath(dirname(__file__))


def domain_enforcement(domain) -> str:
    """Normalise a state-variable domain's enforcement mode to a plain string.

    Returns one of ``'none'`` (default — descriptive metadata only), ``'clamp'``
    (hard-clip to [lo, hi]) or ``'wrap'`` (periodic). Accepts a Range/domain
    object (reads its ``enforce`` slot), a bare ``DomainEnforcement`` value, or
    ``None``. Normalises across both generated representations of the enum — the
    pydantic ``(str, Enum)`` (compare via ``.value``) and the gen-python
    permissible value (compare via ``str()``) — so callers can simply test
    ``domain_enforcement(sv.domain) == 'clamp'``.
    """
    enf = getattr(domain, "enforce", domain)
    if enf is None:
        return "none"
    val = getattr(enf, "value", None)
    if isinstance(val, str):
        return val
    return str(enf).rsplit(".", 1)[-1]


def register_recipe_code_paths(source_file, code_source=None) -> list:
    """Make a recipe's callable code importable — the ``code/`` convention, or a
    declared :class:`CodeSource` (a local directory or a git repository).

    A recipe references custom builders and analysis callables by bare module
    name (e.g. ``module: taher2019_analysis``); their directory must be on
    ``sys.path`` for ``import`` to resolve them. Resolution:

    1. **Explicit ``code_source``** (a ``CodeSource`` or dict on the study) —
       decouples the specification from where its code lives:
         * ``path`` — a directory (relative to the recipe YAML, or absolute); or
         * ``git`` — a repository shallow-cloned and cached under
           ``~/.cache/tvbo/code_sources/<url+ref hash>``, checked out at ``ref``.
       An optional ``subdir`` narrows which directory of the source is used.
    2. **Convention** (no ``code_source``) — the ``code/`` subdir beside the
       recipe YAML.

    Registering at load time, once and left in place (callables resolve lazily
    during a run), lets ``tvbo run`` / ``tvbo workflow`` and notebooks load a
    recipe without a ``PYTHONPATH`` prefix. The dir goes to the front of
    ``sys.path`` (matching ``PYTHONPATH``) and is skipped when already present.
    Returns the paths newly inserted.
    """
    import sys
    from pathlib import Path

    entries = []
    if code_source is not None:
        resolved = _resolve_code_source(code_source, source_file)
        if resolved:
            entries = [str(resolved)]
    if not entries and source_file:  # no (or empty) code_source -> code/ convention
        code_dir = Path(source_file).resolve().parent / "code"
        if code_dir.is_dir():
            entries = [str(code_dir)]

    inserted = []
    for entry in entries:
        if entry not in sys.path:
            sys.path.insert(0, entry)
            inserted.append(entry)
    return inserted


def _resolve_code_source(code_source, source_file):
    """Resolve a ``CodeSource`` (local ``path`` or ``git`` repo) to a directory on
    disk, applying an optional ``subdir``. ``path``/``git`` are mutually exclusive.
    """
    from pathlib import Path

    get = code_source.get if isinstance(code_source, dict) else (lambda k: getattr(code_source, k, None))
    path, git, ref, subdir = get("path"), get("git"), get("ref"), get("subdir")
    if path and git:
        raise ValueError("CodeSource: 'path' and 'git' are mutually exclusive")
    if git:
        base = _fetch_git_code_source(git, ref)
    elif path:
        base = Path(path)
        if not base.is_absolute() and source_file:
            base = Path(source_file).resolve().parent / base
    else:
        return None
    resolved = (Path(base) / subdir if subdir else Path(base)).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"CodeSource resolved to a non-directory: {resolved}")
    return resolved


def _fetch_git_code_source(url, ref=None):
    """Shallow-clone (and cache) a git code source; return the local clone dir.

    Cached by ``sha1(url@ref)`` under ``$TVBO_CACHE`` (default ``~/.cache/tvbo``)
    so a re-run reuses the clone. A branch/tag uses ``--branch``; a bare commit
    (which ``--branch`` rejects) falls back to a full clone + ``checkout``.
    """
    import hashlib, os, shutil, subprocess
    from pathlib import Path

    key = hashlib.sha1(f"{url}@{ref or 'HEAD'}".encode()).hexdigest()[:16]
    cache = Path(os.environ.get("TVBO_CACHE", Path.home() / ".cache" / "tvbo")) / "code_sources" / key
    if (cache / ".git").is_dir():
        return cache
    shutil.rmtree(cache, ignore_errors=True)  # clear any partial/corrupt leftover
    cache.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else []) + [url, str(cache)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return cache
    except FileNotFoundError as e:
        raise RuntimeError(f"CodeSource git fetch needs `git` on PATH ({url})") from e
    except subprocess.CalledProcessError as e:
        shutil.rmtree(cache, ignore_errors=True)
        if not ref:
            raise RuntimeError(f"CodeSource git fetch failed ({url}): {e.stderr}") from e
        try:  # ref may be a bare commit — clone then checkout
            subprocess.run(["git", "clone", url, str(cache)], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(cache), "checkout", ref], check=True, capture_output=True, text=True)
            return cache
        except subprocess.CalledProcessError as e2:
            shutil.rmtree(cache, ignore_errors=True)
            raise RuntimeError(f"CodeSource git fetch failed ({url}@{ref}): {e2.stderr}") from e2


def as_list(obj) -> list:
    """Normalize a keyed-dict-or-list collection to a list of its members.

    TVBO keyed collections (``parameters``, ``space``, …) are dicts keyed by
    each member's identifier, but may also appear as plain lists. Returns the
    member values in either case (``None`` -> ``[]``).
    """
    if obj is None:
        return []
    if hasattr(obj, "values"):
        return list(obj.values())
    return list(obj)


def is_array_valued(value) -> bool:
    """Return True if a parameter value is an array constant rather than a scalar.

    Array-valued parameters (e.g. mode-coupling matrices, Gaussian-quadrature
    vectors) are stored as nested lists/tuples in YAML or as ``np.ndarray`` when
    set programmatically. Scalar-only call sites (``float(p.value)`` substitution,
    sympy ``subs``) must skip them. Single source of truth so list/tuple *and*
    ndarray are treated consistently everywhere.
    """
    return isinstance(value, (list, tuple, np.ndarray))


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested dicts are merged key-by-key, so an override can replace a single leaf
    while inheriting its siblings from ``base`` — e.g. ``{parameters: {a: {value:
    1}}}`` overrides only ``a.value`` and keeps every other parameter from
    ``base``. Any key whose two sides are not both dicts is taken from
    ``override``. Neither input is mutated.

    This is the field-level precedence used when a spec sourced by ``iri`` is
    refined by inline metadata: the inline value supervenes and the source
    (registry entry / ontology default) fills the gaps.
    """
    out = dict(base)
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Backward-compatible re-exports (moved to tvbo.plot.utils)
def __getattr__(name):
    _plot_names = {
        "get_logo",
        "hex2rgba",
        "get_cmap",
        "get_continuous_cmap",
        "multiview",
        "tvb_colors",
    }
    if name in _plot_names:
        from tvbo.plot import utils as _plot_utils

        if name == "tvb_colors":
            return _plot_utils.tvb_colors_simple
        return getattr(_plot_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Bunch(dict):
    """Dictionary with attribute access and optional JAX PyTree support.

    Extends dict to allow both ``bunch["key"]`` and ``bunch.key`` access.
    If JAX is installed, registered as a PyTree via ``register_pytree_node_class``
    with deterministic (sorted-key) traversal order.

    Based on scikit-learn's ``sklearn.utils.Bunch``.

    See Also:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __dir__(self):
        return list(super().__dir__()) + list(self.keys())

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"{type(self).__name__}({items})"

    def copy(self):
        """Return a shallow copy as a new `Bunch`.

        Returns:
            A `Bunch` containing the same key/value pairs as this instance.
        """
        return Bunch(self)

    def tree_flatten(self):
        """Flatten the `Bunch` into JAX pytree (children, aux_data).

        Keys are sorted so traversal order is deterministic across calls.
        """
        keys = tuple(sorted(self.keys()))
        values = tuple(self[k] for k in keys)
        return values, keys

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a `Bunch` from JAX pytree aux_data and children."""
        return cls(zip(aux_data, children))


try:
    from jax.tree_util import register_pytree_node_class

    register_pytree_node_class(Bunch)
except ImportError:
    pass


def numbered_print(text):
    """Print `text` with each line prefixed by a zero-padded line number.

    Line numbers start at 1 and are padded to the width of the largest number
    so the printed numbers stay aligned.

    Args:
        text: The multi-line string to print.
    """
    lines = text.splitlines()
    max_line_num = len(str(len(lines)))
    for i, line in enumerate(lines, start=1):
        print(f"{i:0{max_line_num}} {line}")


def format_pytree_as_string(
    pytree,
    name: str = "root",
    prefix: str = "",
    is_last: bool = False,
    show_numerical_only: bool = False,
    is_root: bool = True,
    hide_none: bool = False,
    show_array_values: bool = False,
) -> str:
    """
    Recursively formats a JAX pytree structure as a string with Unicode box-drawing characters.

    Args:
        pytree (Any): The pytree to format.
        name (str): The name of the current node.
        prefix (str): Current line prefix.
        is_last (bool): Whether the current node is the last child of its parent.
        show_numerical_only (bool): If True, only show arrays and numerical types (float, int, etc.).
        is_root (bool): Whether this node is the root of the tree.
        hide_none (bool): If True, fields with None values will be hidden.
        show_array_values (bool): If True, print full array values instead of summaries.

    Returns:
        str: The formatted string representation of the pytree.
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    # Unicode box-drawing characters for the tree structure
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    # Initialize the result string
    result = []

    # Special handling for root element
    if is_root:
        current_prefix = ""  # Root has no prefix
        next_prefix = ""  # Children of root start without vertical bars
    else:
        # Determine the current line prefix
        current_prefix = prefix + (last if is_last else tee)
        # Determine the prefix for children
        next_prefix = prefix + (space if is_last else branch)

    # Check if the object is a string
    if isinstance(pytree, str):
        if not show_numerical_only:
            result.append(f'{current_prefix}{name}: "{pytree}"')
        return "\n".join(result)

    # Check if the object is None
    if pytree is None:
        if not hide_none and not show_numerical_only:
            result.append(f"{current_prefix}{name}: NoneType")
        return "\n".join(result)

    # Check if the object is a JAX array
    if isinstance(pytree, (jnp.ndarray, np.ndarray)):
        # result.append(f"{current_prefix}{name}: Array({shape_str}, {dtype_str})")
        if show_array_values:
            # result.append(f"{current_prefix}{name}: Array({shape_str}, {dtype_str})")
            # result.append(f"{current_prefix}{name}: No({shape_str}, {dtype_str})")
            result.append(f"{current_prefix}{name}: {pytree}")
        else:
            result.append(f"{current_prefix}{name}: {eqx.tree_pformat(pytree)}")
        return "\n".join(result)

    # Try to flatten the pytree
    try:
        leaves, _ = jax.tree_util.tree_flatten(pytree)
        # If it's a leaf (i.e., it has no children), format its type
        if not leaves or (len(leaves) == 1 and pytree is leaves[0]):
            # For numerical types, always display the value
            if isinstance(pytree, (int, float, bool, complex)):
                result.append(f"{current_prefix}{name}: {pytree}")
            # For other types, check filter setting
            elif not show_numerical_only:
                result.append(f"{current_prefix}{name}: {type(pytree).__name__}")
            return "\n".join(result)

        # Otherwise, format it as a container and process its children
        result.append(f"{current_prefix}{name}")

        # If it's a dictionary, iterate through its key-value pairs
        if isinstance(pytree, dict):
            items = list(pytree.items())
            for i, (key, value) in enumerate(items):
                child_result = format_pytree_as_string(
                    value,
                    str(key),
                    next_prefix,
                    i == len(items) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:  # Only append if there's content (might be empty with show_numerical_only)
                    result.append(child_result)

        # If it's a dataclass or a custom class with __dict__ attribute
        elif hasattr(pytree, "__dict__"):
            items = list(pytree.__dict__.items())
            for i, (key, value) in enumerate(items):
                child_result = format_pytree_as_string(
                    value,
                    key,
                    next_prefix,
                    i == len(items) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:
                    result.append(child_result)

        # If it's a sequence (like list or tuple)
        elif hasattr(pytree, "__len__") and not isinstance(pytree, (str, bytes, bytearray)):
            for i, item in enumerate(pytree):
                child_result = format_pytree_as_string(
                    item,
                    f"[{i}]",
                    next_prefix,
                    i == len(pytree) - 1,
                    show_numerical_only,
                    False,
                    hide_none,
                    show_array_values,
                )
                if child_result:
                    result.append(child_result)

        # For other types of containers
        else:
            result.append(f"{current_prefix}{name}: {type(pytree).__name__} (unknown structure)")

    except Exception:
        # If we can't flatten it as a pytree, treat it as a leaf
        # For strings, display the string value if not filtering
        if isinstance(pytree, str):
            if not show_numerical_only:
                result.append(f'{current_prefix}{name}: "{pytree}"')
        # For numerical types, always display the value
        elif isinstance(pytree, (int, float, bool, complex)):
            result.append(f"{current_prefix}{name}: {pytree}")
        # For other types, check filter setting
        elif not show_numerical_only:
            result.append(f"{current_prefix}{name}: {type(pytree).__name__}")

    return "\n".join(result)


def pretty_print_pytree(
    pytree,
    name: str = "root",
    prefix: str = "",
    show_numerical_only: bool = False,
    hide_none: bool = False,
) -> None:
    """
    Prints a pretty formatted representation of a JAX pytree structure.

    Args:
        pytree (Any): The pytree to print.
        name (str): The name of the current node.
        prefix (str): Current line prefix.
        show_numerical_only (bool): If True, only show arrays and numerical types (float, int, etc.).
        hide_none (bool): If True, fields with None values will be hidden.

    Returns:
        None
    """
    formatted_string = format_pytree_as_string(pytree, name, prefix, False, show_numerical_only, True, hide_none)
    print(formatted_string)


# ---- YAML utilities ----
def to_yaml(obj, filepath: str | None = None) -> str:
    """Dump a LinkML datamodel object to YAML.

    - If filepath is provided, write YAML to that file and return the path.
    - If filepath is None, return the YAML string.

    Args:
        obj (object): Datamodel object to serialize.
        filepath (str | None): Optional path to write YAML.

    Returns:
        str: File path when written to disk, otherwise the YAML string.
    """
    try:
        from linkml_runtime.dumpers import yaml_dumper
    except Exception as e:
        raise RuntimeError("linkml_runtime is required for YAML dumping") from e

    if filepath:
        yaml_dumper.dump(obj, filepath)
        return filepath
    return yaml_dumper.dumps(obj)


def from_yaml(filepath: str, cls) -> object:
    """Load a LinkML datamodel object from a YAML file.

    Parameters:
        filepath (str): Path to the YAML file.
        cls (type): The datamodel class to instantiate.

    Returns:
        object: An instance of the datamodel class populated with data from the YAML file.
    """
    try:
        from linkml_runtime.loaders import yaml_loader
    except Exception as e:
        raise RuntimeError("linkml_runtime is required for YAML loading") from e
    md = yaml_loader.load(filepath, target_class=cls)
    return md


def add_to_parameters_collection(key, value, path, parameters):
    """Adds a value to a Bunch object using the provided path, without inserting a redundant sub-level."""
    current_level = parameters
    for part in path:
        if part == "parameters":
            continue
        part_key = str(part) if isinstance(part, int) else part
        if part_key not in current_level:
            current_level[part_key] = Bunch()
        if part != key:
            current_level = current_level[part_key]
    final_key = str(key) if isinstance(key, int) else key
    current_level[final_key] = value.value


def traverse_metadata(
    metadata,
    target_instance=None,
    path=None,
    callback=None,
    callback_kwargs=None,
    keys_to_exclude=(),
):
    """Recursively traverses the attributes of a metadata object, calling a callback on each Parameter."""
    if target_instance is None:
        from tvbo.datamodel import schema as tvbo_datamodel

        target_instance = tvbo_datamodel.Parameter
    if callback is None:
        callback = add_to_parameters_collection
    if callback_kwargs is None:
        callback_kwargs = {}
    if path is None:
        path = []

    def _is_datamodel_like(obj):
        try:
            mod = getattr(type(obj), "__module__", "")
            if mod.startswith("tvbo.datamodel."):
                return True
            for base in type(obj).mro():
                if getattr(base, "__module__", "").startswith("tvbo.datamodel."):
                    return True
        except Exception:
            pass
        return False

    if hasattr(metadata, "__dict__"):
        if isinstance(metadata, target_instance):
            if callback:
                callback(path[-1], metadata, path, **callback_kwargs)

        for attr_name, attr_value in metadata.__dict__.items():
            if attr_name in keys_to_exclude or attr_value is None:
                continue

            current_path = path + [attr_name]
            if _is_datamodel_like(attr_value):
                traverse_metadata(
                    attr_value,
                    target_instance,
                    current_path,
                    callback,
                    callback_kwargs,
                    keys_to_exclude,
                )
            elif isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    traverse_metadata(
                        item,
                        target_instance,
                        current_path + [i],
                        callback,
                        callback_kwargs,
                        keys_to_exclude,
                    )
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    traverse_metadata(
                        value,
                        target_instance,
                        current_path + [key],
                        callback,
                        callback_kwargs,
                        keys_to_exclude,
                    )
