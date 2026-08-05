"""Shared pydantic base for the runtime classes in :mod:`tvbo.classes`.

The generated datamodel configures its classes for a leaf specification: every
assignment is re-validated, and an unknown field is an error. Those defaults are
right at the YAML boundary — an authored recipe with a typo'd slot should fail
loudly — and wrong for a subclass that carries behaviour, because codegen mutates
these objects in hot paths and hangs runtime caches off them.

This base relaxes exactly those two settings and changes nothing else, so
``extra="forbid"`` keeps guarding authored YAML on the datamodel classes while the
runtime classes above them stay ergonomic. Validation belongs at load and at save,
not on every intermediate mutation.
"""

from pydantic import ConfigDict

from tvbo.datamodel.pydantic import ConfiguredBaseModel


class TVBOModel(ConfiguredBaseModel):
    """Base for the behaviour-carrying runtime classes."""

    model_config = ConfigDict(
        **{
            **ConfiguredBaseModel.model_config,
            "extra": "allow",
            "validate_assignment": False,
        }
    )

    def as_dict(self, **kwargs):
        """This object as a plain dict, dropping unset slots.

        A compatibility shim for the call sites that still expect the
        ``linkml_runtime`` / ``jsonasobj2`` dataclass API; it retires with the last
        of them.
        """
        return self.model_dump(exclude_none=True, **kwargs)
