"""A behaviour mixin attaches to the schema class it is named after, discovered where the datamodel is generated.

These exercise ``hatch_build.py`` itself, so they need LinkML, the build requirement the installed package does not depend on; the ``Schema validation`` job, which installs it, runs them.
"""

import pathlib
import sys

import pytest

pytest.importorskip("linkml_runtime")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCHEMA = _ROOT / "schema" / "tvbo_datamodel.yaml"

sys.path.insert(0, str(_ROOT))
from hatch_build import _behaviour_mixins  # noqa: E402


def test_a_mixin_is_discovered_by_its_own_name():
    """``EventBehaviour`` attaches to ``Event``; there is nothing to register."""
    assert _behaviour_mixins(_ROOT, _SCHEMA)["Event"] == "tvbo.behaviour.event.EventBehaviour"


def test_a_mixin_naming_no_schema_class_fails_the_build(tmp_path):
    """Otherwise it attaches to nothing and the helpers vanish with no error."""
    package = tmp_path / "tvbo" / "behaviour"
    package.mkdir(parents=True)
    (package / "bogus.py").write_text("class NotAClassBehaviour:\n    pass\n")

    with pytest.raises(RuntimeError, match="no class 'NotAClass'"):
        _behaviour_mixins(tmp_path, _SCHEMA)
