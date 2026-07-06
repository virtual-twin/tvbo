"""Regression tests for cross-backend observation consistency and TVB networks.

These guard two defects:

1. **Observation sample-count divergence.** ``Bold_TVB`` is a declarative YAML
   observation model. The same YAML MUST resolve to the same sampling on every
   Python-based backend (jax / tvboptim / tvb): a BOLD monitor with ``TR=720ms``
   over a ``duration`` produces ``floor(duration / TR)`` samples regardless of
   the integration ``step_size``. The bug: the jax monitor template applied the
   *literal* pre-baked ``subsample`` stepsize (720/stock_dt = 180) to the raw
   ``dt=0.1`` grid instead of ``TR/dt``, yielding ``n_steps / 180`` BOLD samples
   (e.g. 112 for a 2000ms run) while tvboptim/tvb correctly yielded 2. tvboptim
   resolves the period as ``TR/dt`` and is the reference (see
   ``tvbo/templates/tvboptim/observations.py``).

2. **TVB rejects a standard weights network.** ``exp.run("tvb")`` on a
   connectivity-only ``Network`` (weights / tract_lengths / labels / centres —
   e.g. ``atlas="Lobar8", rec="avgMatrix"``) raised
   ``ValueError("Atlas ... is not available in the dataset")`` because the TVB
   adapter tried to resolve a parcellation-volume ``.nii`` it never needs for a
   ``tvb.Connectivity``.

Conventions match ``tests/functional/test_simulation_backends_*.py`` and
``simulation_backends_shared.py``.
"""

import numpy as np
import pytest

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment
from tests.functional.simulation_backends_shared import _HAVE_TVB, _HAVE_TVBOPTIM

# ``Bold_TVB`` monitor period. Read from the DB so the expected sample count is
# derived from the same declarative source the backends consume, not a magic
# number.
_BOLD_TR = Observation.from_db("Bold_TVB").parameters["TR"].value  # 720.0 ms

# Short but >= 2 BOLD samples: floor(2000 / 720) == 2.
_DURATION = 2000.0
_STEP_SIZE = 0.1
_EXPECTED_BOLD_SAMPLES = int(_DURATION // _BOLD_TR)


def _matrix_bold_experiment(duration=_DURATION):
    """Two-node, tvb-safe experiment (Network.from_matrix) with BOLD + TA.

    ``from_matrix`` builds a pure connectivity network that every backend
    (including TVB) already accepts, so this isolates the observation
    sample-count bug from the TVB-network-acceptance bug.
    """
    network = Network.from_matrix(
        weights=np.array([[0.0, 0.2], [0.1, 0.0]]),
        lengths=np.zeros((2, 2)),
        labels=["left", "right"],
    )
    network.coupling["Linear"] = Coupling.from_db("Linear")
    return SimulationExperiment(
        dynamics=Dynamics.from_db("ReducedWongWang"),
        network=network,
        integration={"method": "Heun", "duration": duration, "step_size": _STEP_SIZE},
        observations=[
            Observation.from_db("TemporalAverage"),
            Observation.from_db("Bold_TVB"),
        ],
    )


def _find_observation(result, needle):
    """Return the observation data whose key contains *needle* (case-insensitive).

    All backends now expose the same canonical observation keys (e.g.
    ``BOLD_TVB``); the case-insensitive substring match is kept as a
    lenient safety net.
    """
    observations = result.integration.observations
    for key, value in observations.items():
        if needle.lower() in key.lower():
            return value.data
    raise AssertionError(f"No observation matching {needle!r} in {list(observations)}")


def _sample_count(result, needle):
    """First-axis length (number of monitor samples) of an observation."""
    return _find_observation(result, needle).shape[0]


def _run(backend, duration=_DURATION):
    return _matrix_bold_experiment(duration).run(backend)


def _align(left, right):
    """Squeeze trailing singleton axes and drop a leading off-by-one sample.

    TVB carries an extra trailing mode axis; backends may emit a boundary
    sample the others don't. Mirrors the alignment in
    ``tests/test_tvboptim_observation_codegen.py``.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    while left.ndim > right.ndim and left.shape[-1] == 1:
        left = left[..., 0]
    while right.ndim > left.ndim and right.shape[-1] == 1:
        right = right[..., 0]
    if left.shape[0] == right.shape[0] + 1:
        left = left[1:]
    elif right.shape[0] == left.shape[0] + 1:
        right = right[1:]
    return left, right


class TestObservationSampleCountConsistency:
    """The same BOLD YAML must sample identically on every backend."""

    @pytest.mark.backend_core
    def test_bold_sample_count_matches_tr_formula_per_backend(self):
        """Each available backend emits floor(duration / TR) BOLD samples.

        Catches the jax defect directly: jax produced ``n_steps / 180`` (=112)
        instead of ``duration / TR`` (=2). A single backend suffices to fail,
        so this holds even when only jax is installed.
        """
        counts = {}

        # jax is a core dependency and always present.
        counts["jax"] = _sample_count(_run("jax"), "bold")

        if _HAVE_TVBOPTIM:
            counts["tvboptim"] = _sample_count(_run("tvboptim"), "bold")
        if _HAVE_TVB:
            counts["tvb"] = _sample_count(_run("tvb"), "bold")

        for backend, count in counts.items():
            assert count == _EXPECTED_BOLD_SAMPLES, (
                f"{backend} produced {count} BOLD samples; expected "
                f"{_EXPECTED_BOLD_SAMPLES} = floor({_DURATION}/{_BOLD_TR}). "
                f"All backends: {counts}"
            )

    @pytest.mark.backend_core
    def test_bold_sample_count_identical_across_backends(self):
        """Cross-backend agreement: BOLD sample count is identical everywhere.

        Requires at least two backends to be meaningful; skips otherwise.
        """
        counts = {"jax": _sample_count(_run("jax"), "bold")}
        if _HAVE_TVBOPTIM:
            counts["tvboptim"] = _sample_count(_run("tvboptim"), "bold")
        if _HAVE_TVB:
            counts["tvb"] = _sample_count(_run("tvb"), "bold")

        if len(counts) < 2:
            pytest.skip("Need >= 2 backends installed to compare BOLD sampling")

        unique = set(counts.values())
        assert len(unique) == 1, f"BOLD sample count diverges across backends: {counts}"
        assert unique.pop() >= 2, f"Expected >= 2 BOLD samples, got {counts}"

    @pytest.mark.backend_core
    def test_temporal_average_sample_count_identical_across_backends(self):
        """TemporalAverage (dt-period monitor) must also agree across backends."""
        counts = {"jax": _sample_count(_run("jax"), "temporal")}
        if _HAVE_TVBOPTIM:
            counts["tvboptim"] = _sample_count(_run("tvboptim"), "temporal")
        if _HAVE_TVB:
            counts["tvb"] = _sample_count(_run("tvb"), "temporal")

        if len(counts) < 2:
            pytest.skip("Need >= 2 backends installed to compare sampling")

        assert len(set(counts.values())) == 1, (
            f"TemporalAverage sample count diverges across backends: {counts}"
        )

    @pytest.mark.backend_tvboptim
    @pytest.mark.skipif(not _HAVE_TVBOPTIM, reason="tvboptim not installed")
    def test_bold_values_close_jax_vs_tvboptim(self):
        """Same YAML on jax and tvboptim should be numerically close, not just
        equal in shape. Guards against a same-count-but-wrong-values fix."""
        jax_bold = _find_observation(_run("jax"), "bold")
        tvboptim_bold = _find_observation(_run("tvboptim"), "bold")

        jax_bold, tvboptim_bold = _align(jax_bold, tvboptim_bold)
        assert jax_bold.shape == tvboptim_bold.shape, (
            f"BOLD shapes differ after alignment: {jax_bold.shape} vs {tvboptim_bold.shape}"
        )
        # Two independent float64 Heun/JAX paths accumulate ULP-level drift over
        # 20000 dt=0.1 steps; a loose tolerance still catches a wrong monitor.
        np.testing.assert_allclose(jax_bold, tvboptim_bold, rtol=5e-2, atol=1e-3)


@pytest.mark.backend_tvb
@pytest.mark.skipif(not _HAVE_TVB, reason="tvb-library not installed")
class TestTVBNetworkAcceptance:
    """run("tvb") must accept a plain weights Network (no parcellation volume)."""

    def _weights_network(self):
        matches = Network.list_db(atlas="Lobar8", rec="avgMatrix")
        if not matches:
            pytest.skip("Lobar8/avgMatrix network not in database")
        network = Network.from_db(atlas="Lobar8", rec="avgMatrix")
        network.coupling["Linear"] = Coupling.from_db("Linear")
        return network

    def _weights_experiment(self, duration=_DURATION):
        return SimulationExperiment(
            dynamics=Dynamics.from_db("ReducedWongWang"),
            network=self._weights_network(),
            integration={"method": "Heun", "duration": duration, "step_size": _STEP_SIZE},
            observations=[
                Observation.from_db("TemporalAverage"),
                Observation.from_db("Bold_TVB"),
            ],
        )

    def test_tvb_runs_weights_network_without_atlas_volume_error(self):
        """A connectivity-only Network must run on TVB.

        Regression: the TVB adapter demanded a parcellation ``.nii`` volume and
        raised ``ValueError('Atlas Lobar8 is not available in the dataset...')``.
        A ``tvb.Connectivity`` only needs weights / tract_lengths / labels /
        centres, none of which require the volume.
        """
        experiment = self._weights_experiment()
        try:
            result = experiment.run("tvb")
        except ValueError as exc:
            if "is not available in the dataset" in str(exc):
                pytest.fail(
                    "TVB backend rejected a standard weights Network by demanding "
                    f"a parcellation volume: {exc}"
                )
            raise

        assert result is not None
        assert hasattr(result, "data")
        assert hasattr(result, "time")
        bold_samples = _sample_count(result, "bold")
        assert bold_samples == _EXPECTED_BOLD_SAMPLES, (
            f"Expected {_EXPECTED_BOLD_SAMPLES} BOLD samples on the weights "
            f"network, got {bold_samples}"
        )
