"""What :mod:`tvbo.codegen.imports` may and may not remove from generated source.

The pass exists so backend templates can emit the imports a feature *may* need without
each one growing a condition that drifts. That only works if it is trusted, and it is
only trustworthy if the cases where removing an import would change behaviour are
pinned. Those cases are the point of this module: side-effecting imports, names reached
through a string, ``__future__``, star imports, and the ordering that ``JAX_PLATFORMS``
depends on.
"""

from __future__ import annotations

import pytest

from tvbo.codegen.imports import prune_unused_imports, unused_import_names


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


def test_indented_import_keeps_its_indentation():
    src = "def f():\n    import os\n    import sys\n\n    return sys.argv\n"
    out = prune_unused_imports(src)
    assert "    import sys" in out and "import os" not in out


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
