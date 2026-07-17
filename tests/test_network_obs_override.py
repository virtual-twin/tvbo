"""`compute_all_observations(network_obs=...)` overrides the module-level target.

Network-observation targets (an empirical FC to fit against) are materialised into
module-level constants and bound once at ``run_experiment`` time. That is correct
for one process per subject, but a batched cohort fit scores many subjects in one
traced call, each against its own target carried as a traced leaf — so the scoring
path must honour a per-call ``network_obs`` override instead of the process-wide
constant. Without it every subject is silently scored against whichever target was
bound last. This renders the real FIC+EIB experiment and checks the override end to
end, so reverting the binding to the bare module constant fails here.
"""

import types

import numpy as np
import pytest

from tvbo.classes.experiment import SimulationExperiment

_EXPERIMENT = "tvbo/database/experiments/EI_Tuning_FIC_EIB_Optimization.yaml"


@pytest.fixture(scope="module")
def gen_module():
    """Render the experiment to a tvboptim module and exec it in isolation."""
    code = SimulationExperiment.from_file(_EXPERIMENT).render_code(format="tvboptim")
    mod = types.ModuleType("gen_fic_eib")
    exec(compile(code, "gen_fic_eib.py", "exec"), mod.__dict__)
    return mod


def _fc_target_scalar(obs):
    return float(np.asarray(obs.fc_target).ravel()[0])


def test_network_obs_override_wins_over_module_constant(gen_module):
    """A `network_obs` entry is used in place of the bound module constant."""
    bound = np.full((3, 3), 7.0)
    override = np.full((3, 3), 9.0)
    gen_module._bind_network_observations({"fc_target": bound})

    # `only=set()` skips every simulated observation, so result/state are unused and
    # this isolates the network-observation binding.
    default = gen_module.compute_all_observations(None, None, only=set())
    overridden = gen_module.compute_all_observations(
        None, None, only=set(), network_obs={"fc_target": override}
    )

    assert _fc_target_scalar(default) == 7.0, "module constant not honoured by default"
    assert _fc_target_scalar(overridden) == 9.0, "network_obs override was ignored"


def test_absent_override_falls_back_to_module_constant(gen_module):
    """An override dict without this observation leaves the module constant in place."""
    bound = np.full((3, 3), 5.0)
    gen_module._bind_network_observations({"fc_target": bound})

    # An unrelated key must not shadow fc_target — the fallback is per-observation.
    obs = gen_module.compute_all_observations(
        None, None, only=set(), network_obs={"something_else": np.zeros((3, 3))}
    )
    assert _fc_target_scalar(obs) == 5.0
