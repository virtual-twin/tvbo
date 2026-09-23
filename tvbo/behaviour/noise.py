#
# Module: behaviour/noise.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""What a ``Noise`` record does: its two derived dispersions, its symbolic form, its code generation, and its pytree.

A noise process is declared once and read by every backend, so ``sigma`` and ``nsig`` are two spellings of one quantity and each is derived from the other where only one is written. The record carries no cache: everything here is computed from the declared parameters, which is what makes a noise safe to serialize back out unchanged.
"""

from __future__ import annotations

from typing import ClassVar

from tvbo.behaviour._runtime import RuntimeAttributes
from tvbo.utils.pytree import Pytree, static_spec


class NoiseBehaviour(RuntimeAttributes, Pytree):
    """Everything a declared noise process does, on both generated forms.

    A JAX leaf as well as a record: a runtime per-state ``sigma_vec`` is the one child, so a sweep can `vmap` over it while the declared spec stays static metadata.
    """

    LEAVES: ClassVar[tuple[str, ...]] = ("sigma_vec",)
    """The runtime per-state sigma vector, exposed as the single array child so it can participate in `vmap` batching."""

    @property
    def sigma_vec(self):
        """The per-state sigma a run is using, or ``None`` before one sets it.

        Runtime state, not a slot: a record declares its sigma as a parameter, and this is the vector a solver builds from it and may `vmap` over. Held under a leading underscore so the dumpers hide it by rule — assigned as a plain attribute it is an array in the record, which the YAML dumper cannot represent at all.
        """
        return getattr(self, "_sigma_vec", None)

    @sigma_vec.setter
    def sigma_vec(self, value):
        object.__setattr__(self, "_sigma_vec", value)

    def _pytree_static(self) -> str:
        """The declared spec as canonical JSON, without the runtime leaves."""
        return static_spec(self, held_out=self.LEAVES)

    @classmethod
    def _pytree_build(cls, static, leaves):
        """The record again, with its runtime sigma vector reattached.

        Reattached through the `sigma_vec` property, which holds it as runtime state rather than a slot: the Pydantic form declares no such field and would refuse it, leaving that form flattenable and never unflattenable.
        """
        import json

        obj = cls(**json.loads(static))
        obj.sigma_vec = leaves.get("sigma_vec")
        return obj

    @property
    def parameters_dict(self):
        """The noise parameters as a dict-like view, empty when unset."""
        params = getattr(self, "parameters", None)
        return params if isinstance(params, dict) else (params or {})

    STANDARD_EQUATIONS: ClassVar[dict[str, tuple[str, str]]] = {
        "gaussian": ("N", "sqrt(dt) * sigma * xi"),
        "ou": ("dN/dt", "-N/tau + sigma * xi"),
    }
    """The process each canonical noise type names, as ``(lhs, rhs)``.

    Derived rather than written into the record: a recipe stating ``noise_type: gaussian`` has said which process it means, and filling the slot on its behalf would put content into the published record that its author did not write.
    """

    _CANONICAL_TYPE: ClassVar[dict[str, str]] = {"white": "gaussian", "ornstein-uhlenbeck": "ou"}
    """The other spellings a noise type is written under, each mapped onto the type it means."""

    @property
    def canonical_type(self) -> str | None:
        """The declared noise type under its canonical spelling, ``None`` when none is declared."""
        declared = str(self.noise_type or "").lower()
        return self._CANONICAL_TYPE.get(declared, declared) or None

    @property
    def equation_of_record(self):
        """The equation this noise runs, declared or standard.

        A record that writes its own equation — one with a right-hand side — means that one; a record that only names a type means the standard form of that type. One reader, so a backend cannot integrate a different process from the one the record states.
        """
        if getattr(self.equation, "rhs", None):
            return self.equation
        standard = self.STANDARD_EQUATIONS.get(self.canonical_type)
        if standard is None:
            return None
        from tvbo.datamodel.dialect import peer_module

        lhs, rhs = standard
        return peer_module(self).Equation(lhs=lhs, rhs=rhs)

    @property
    def symbolic(self):
        r"""The right-hand side of the process this noise runs, as a SymPy expression.

        Read from `equation_of_record`, so it cannot disagree with it: $\sqrt{dt}\,\sigma\,\xi$ for gaussian/white noise, $-N/\tau + \sigma\,\xi$ for an Ornstein-Uhlenbeck process, the declared equation when the record writes one, and `None` when it names no process. ``dt`` and ``sigma`` carry their positivity, so a root over them simplifies.
        """
        import sympy as sp

        from tvbo.parse.symbols import BUILTIN_SHADOW

        equation = self.equation_of_record
        if equation is None:
            return None
        dt = sp.symbols("dt", real=True, positive=True)
        sigma_sym = sp.symbols("sigma", real=True, positive=True)
        xi = sp.symbols("xi", real=True)
        return BUILTIN_SHADOW.extend(dt=dt, sigma=sigma_sym, xi=xi).parse(str(equation.rhs))

    @property
    def nsig(self):
        r"""The noise dispersion `nsig`, derived from `sigma` as $0.5\,\sigma^2$ if needed.

        Prefers an explicit `nsig` parameter; otherwise computes it from `sigma`. Returns `None` when neither is available.
        """
        value = _declared(self.parameters_dict, "nsig")
        if value is not None:
            return value
        sigma = _declared(self.parameters_dict, "sigma")
        return None if sigma is None else 0.5 * (sigma**2)

    @property
    def sigma(self):
        r"""The noise standard deviation `sigma`, derived from `nsig` as $\sqrt{2\,nsig}$ if needed.

        Prefers an explicit `sigma` parameter; otherwise computes it from `nsig`. Returns `None` when neither is available.
        """
        import numpy as np

        value = _declared(self.parameters_dict, "sigma")
        if value is not None:
            return value
        nsig = _declared(self.parameters_dict, "nsig")
        return None if nsig is None else np.sqrt(2 * nsig)

    def render_code(self, format="tvb"):
        """Render the noise as source code for the requested backend.

        Args:
            format: Target backend; `"tvb"` selects the TVB template, while `"autodiff"` or `"jax"` selects the JAX template.

        Returns:
            The rendered source code as a string.
        """
        from tvbo import templates

        name = "tvbo-tvb-noise.py.mako" if format == "tvb" else "jax-noise.py.mako"
        return templates.lookup.get_template(name).render(noise=self)

    def execute(self, format="tvb"):
        """Render, execute, and return the backend's noise class.

        Args:
            format: Target backend passed through to code generation.

        Returns:
            The executed backend noise object.
        """
        from tvbo.codegen import templater

        local_vars = {}
        exec(self.render_code(format=format), templater.exec_globals, local_vars)
        self._tvb = local_vars["Noise"]
        return self._tvb


def _declared(parameters, name):
    """The value a noise parameter states, or ``None`` when it does not state one.

    A parameter arrives as a record with a ``value`` slot or as the plain mapping a terse recipe writes; one reader for both, so ``sigma`` and ``nsig`` cannot mean different things on the two spellings.
    """
    entry = parameters.get(name)
    if entry is None:
        return None
    return entry.get("value") if isinstance(entry, dict) else getattr(entry, "value", None)
