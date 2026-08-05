"""What :mod:`tvbo.codegen.prune` may and may not remove from generated source.

The passes exist so backend templates can emit the imports and scaffolding a feature
*may* need without each one growing a condition that drifts. That only works if they are
trusted, and they are only trustworthy if the cases where removing something would
change behaviour are pinned. Those cases are the point of this module: side-effecting
imports and right-hand sides, names reached through a string, ``__future__``, star
imports, closures, rebinding, and the ordering that ``JAX_PLATFORMS`` depends on.
"""

from __future__ import annotations

import pytest

from tvbo.codegen.prune import (
    prune,
    prune_dead_assignments,
    prune_unused_imports,
    unused_import_names,
)


def test_drops_an_unreferenced_import():
    src = "import os\nimport sys\n\nprint(sys.argv)\n"
    assert prune_unused_imports(src) == "import sys\n\nprint(sys.argv)\n"


def test_keeps_only_the_used_names_of_a_multi_name_import():
    src = "from typing import Any, Dict, List, Optional\n\nx: Dict[str, Any] = {}\n"
    out = prune_unused_imports(src)
    assert "Dict" in out and "Any" in out
    assert "List" not in out and "Optional" not in out


def test_attribute_root_counts_as_a_use():
    """``import jax.numpy as jnp`` is used by ``jnp.exp``, an Attribute, not a Name."""
    src = "import jax.numpy as jnp\n\ny = jnp.exp(1.0)\n"
    assert prune_unused_imports(src) == src


def test_dotted_import_binds_its_root():
    """``import jax.scipy.signal`` binds ``jax``; ``jax.jit`` keeps it."""
    src = "import jax.scipy.signal\n\nf = jax.jit(lambda x: x)\n"
    assert prune_unused_imports(src) == src


def test_a_name_used_only_inside_a_string_is_kept():
    """A class reached by name must survive; the pass errs toward keeping."""
    src = 'from tvbo.data.types import TimeSeries\n\ncls = registry["TimeSeries"]\n'
    assert prune_unused_imports(src) == src


def test_a_name_mentioned_only_in_a_docstring_is_dropped():
    """Prose naming a class is not a use of it — the distinction from the test above."""
    src = '"""Builds a TimeSeries."""\n\nfrom tvbo.data.types import TimeSeries\n\nx = 1\n'
    assert "TimeSeries" not in prune_unused_imports(src).split('"""')[2]


def test_a_name_mentioned_only_in_prose_is_dropped():
    """A ``doc=`` string is prose, not a reference.

    ``"Additive coefficient for the second state-variable"`` does not parse as Python,
    which is exactly what distinguishes it from ``registry["TimeSeries"]`` above. Word
    matching cannot tell them apart, and treating prose as a use kept ``Additive`` and
    ``Coupling`` imported into every generated TVB model.
    """
    src = (
        "from tvb.simulator.noise import Additive\n"
        '\nx = NArray(doc="Additive coefficient for the second state-variable")\n'
    )
    assert "import Additive" not in prune_unused_imports(src)


def test_a_string_that_parses_as_code_still_counts():
    """The safety net that matters: an eval-ed expression keeps what it names."""
    src = 'import jax.numpy as jnp\n\nf = compile_expr("jnp.mean(data)")\n'
    assert prune_unused_imports(src) == src


def test_drops_a_dead_binding_with_a_pure_right_hand_side():
    src = "def f(weights):\n    n_nodes = weights.shape[0]\n    return 1\n"
    assert "n_nodes" not in prune_dead_assignments(src)


def test_class_attributes_are_never_dropped():
    """A class body's assignments are its interface, read from outside the module.

    ``COUPLING_INPUTS = {...}`` is unread by the module that defines it and looks exactly
    like dead scaffolding. Removing it left the generated dynamics advertising no
    coupling inputs, so building the network failed with "Unknown coupling names".
    """
    src = (
        "class Kuramoto:\n"
        '    COUPLING_INPUTS = {"c": 1}\n'
        '    STATE_NAMES = ("theta",)\n'
        "\n    def dfun(self, s):\n        return s\n"
    )
    assert prune_dead_assignments(src) == src


def test_module_level_bindings_are_never_dropped():
    """Another module may import them; only function locals are private enough."""
    src = "DEFAULTS = {}\nN_NODES = 4\n\n\ndef f():\n    return 1\n"
    assert prune_dead_assignments(src) == src


def test_a_class_body_inside_a_function_is_still_a_class_body():
    src = "def make():\n    class C:\n        FIELD = 1\n\n    return C\n"
    assert prune_dead_assignments(src) == src


def test_never_drops_a_binding_whose_right_hand_side_is_a_call():
    """``copy.deepcopy`` is the point: dropping it would skip the copy, not just a name."""
    src = "import copy\n\ndef f(state):\n    initial = copy.deepcopy(state)\n    return 1\n"
    assert prune_dead_assignments(src) == src


def test_binding_counts_are_per_scope():
    """The same name is dead in one function and live in another; only the first goes."""
    src = (
        "def a(w):\n    n = w.shape[0]\n    return 1\n"
        "\ndef b(w):\n    n = w.shape[0]\n    return n\n"
    )
    out = prune_dead_assignments(src)
    assert out.count("n = w.shape[0]") == 1
    assert "return n" in out


def test_a_binding_read_by_a_nested_closure_is_kept():
    src = "def outer(w):\n    n = w.shape[0]\n\n    def inner():\n        return n\n\n    return inner\n"
    assert prune_dead_assignments(src) == src


def test_a_rebound_name_is_kept():
    """Two bindings mean the first may feed the second; the pass does not reason about it."""
    src = "def f():\n    d = 1\n    d = d + 1\n    return d\n"
    assert prune_dead_assignments(src) == src


def test_tuple_unpacking_is_never_touched():
    """One element may be needed even when the other is not."""
    src = "def f(pair):\n    a, b = pair\n    return b\n"
    assert prune_dead_assignments(src) == src


def test_attribute_and_subscript_targets_are_never_touched():
    src = "def f(obj, arr):\n    obj.x = 1\n    arr[0] = 2\n    return 3\n"
    assert prune_dead_assignments(src) == src


def test_pruning_an_assignment_can_free_its_import():
    """Assignments are pruned before imports so the cascade resolves in one pass."""
    src = "import os\n\n\ndef f():\n    sep = os.sep\n    return 1\n"
    out = prune(src)
    assert "import os" not in out and "sep" not in out


def test_future_import_is_never_dropped():
    src = "from __future__ import annotations\n\nx = 1\n"
    assert prune_unused_imports(src) == src


def test_star_import_is_never_dropped():
    src = "from math import *\n\nx = 1\n"
    assert prune_unused_imports(src) == src


def test_noqa_marks_an_import_as_deliberate():
    src = "import tvbo.plugins  # noqa: F401\n\nx = 1\n"
    assert prune_unused_imports(src) == src


def test_import_order_is_preserved():
    """Pruning never hoists.

    The tvboptim module sets ``JAX_PLATFORMS`` before importing jax, so an import moved
    above that assignment would silently change the device the experiment runs on.
    """
    src = (
        "import os\n"
        '\nos.environ.setdefault("JAX_PLATFORMS", "cpu")\n'
        "import jax\n"
        "\ny = jax.jit(lambda x: x)\n"
    )
    out = prune_unused_imports(src)
    assert out.index("os.environ") < out.index("import jax")


def test_unparseable_source_is_returned_unchanged():
    """Reporting a syntax error is the formatter's job, and its message is better."""
    src = "import os\n\ndef broken(\n"
    assert prune_unused_imports(src) == src
    assert unused_import_names(src) == set()


def test_a_module_import_shadowed_by_a_local_one_is_dropped():
    """The read resolves to the local import, so the module-level one is dead.

    Python decides this per function: one nested ``import os`` makes every ``os`` in that
    function local. Counting those reads against the module kept a top-level ``import
    os`` that only the ``JAX_PLATFORMS`` line — itself not emitted — would have used.
    """
    src = "import os\n\n\ndef f():\n    import os\n\n    return os.sep\n"
    out = prune_unused_imports(src)
    assert out.lstrip().startswith("def f()")
    assert "    import os" in out


def test_a_module_import_read_from_module_scope_is_kept():
    """The other half of the pair: no local binding, so the read reaches the module."""
    src = "import os\n\n\ndef f():\n    return os.sep\n"
    assert prune_unused_imports(src) == src


def test_indented_import_keeps_its_indentation():
    src = "def f():\n    import os\n    import sys\n\n    return sys.argv\n"
    out = prune_unused_imports(src)
    assert "    import sys" in out and "import os" not in out


def test_unparseable_source_is_returned_unchanged_by_the_assignment_pass():
    src = "x = 1\n\ndef broken(\n"
    assert prune_dead_assignments(src) == src


@pytest.mark.parametrize("fmt", ["tvb", "jax", "tvboptim"])
def test_backends_emit_no_unused_imports(fmt):
    """The end-to-end claim: a rendered experiment carries no import it does not use."""
    from tvbo import SimulationExperiment

    from tests.test_codegen_style_contract import KURAMOTO_FACTORED

    try:
        code = SimulationExperiment(**KURAMOTO_FACTORED).render_code(format=fmt)
    except ImportError as exc:
        pytest.skip(f"{fmt} backend unavailable: {exc}")
    leftover = unused_import_names(code)
    assert not leftover, f"{fmt} still imports unused {sorted(leftover)}"
