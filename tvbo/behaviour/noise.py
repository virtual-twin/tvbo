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

from tvbo.utils.pytree import Pytree, static_spec


class NoiseBehaviour(Pytree):
    """Everything a declared noise process does, on both generated forms.

    A JAX leaf as well as a record: a runtime per-state ``sigma_vec`` is the one child, so a sweep can `vmap` over it while the declared spec stays static metadata.
    """

    LEAVES = ("sigma_vec",)
    """The runtime per-state sigma vector, exposed as the single array child so it can participate in `vmap` batching."""

    def __post_init__(self, *args, **kwargs):
        """The dataclass form's construction hook."""
        self._fill_default_equation()
        super().__post_init__(*args, **kwargs)

    def model_post_init(self, context, /):
        """The Pydantic form's construction hook."""
        super().model_post_init(context)
        self._fill_default_equation()

    def _fill_default_equation(self) -> None:
        r"""Give a noise that declares no equation the standard form of its own type.

        Gaussian/white noise is $\sqrt{dt}\,\sigma\,\xi$ and an Ornstein-Uhlenbeck process is $-N/\tau + \sigma\,\xi$; a record naming a type and no equation means the standard one, and writing it out is what lets every backend read the process from the record alone.
        """
        if self.equation:
            return
        from tvbo.datamodel.schema import Equation

        if self.noise_type in ("gaussian", "white"):
            self.equation = Equation(lhs="N", rhs="sqrt(dt) * sigma * xi")
        elif self.noise_type in ("ou", "ornstein-uhlenbeck"):
            self.equation = Equation(lhs="dN/dt", rhs="-N/tau + sigma * xi")

    def _pytree_static(self) -> str:
        """The declared spec as canonical JSON, without the runtime leaves."""
        return static_spec(self, held_out=self.LEAVES)

    @classmethod
    def _pytree_build(cls, static, leaves):
        """The record again, with its runtime sigma vector reattached."""
        import json

        obj = cls(**json.loads(static))
        obj.sigma_vec = leaves.get("sigma_vec")
        return obj

    @property
    def parameters_dict(self):
        """The noise parameters as a dict-like view, empty when unset."""
        params = getattr(self, "parameters", None)
        return params if isinstance(params, dict) else (params or {})

    @property
    def symbolic(self):
        r"""The symbolic noise term $\sqrt{dt}\,\sigma\,\xi$ for gaussian/white noise.

        Returns `None` for noise types other than `gaussian`/`white`.
        """
        import sympy as sp

        dt = sp.symbols("dt", real=True, positive=True)
        sigma_sym = sp.symbols("sigma", real=True, positive=True)
        xi = sp.symbols("xi", real=True)
        if isinstance(self.noise_type, str) and self.noise_type.lower() in ("gaussian", "white"):
            return sp.sqrt(dt) * sigma_sym * xi
        return None

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
