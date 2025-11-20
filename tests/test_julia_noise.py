import pytest

from tvbo.data import db
from tvbo import SimulationStudy
from tvbo.datamodel.tvbo_datamodel import Noise, Parameter

@pytest.mark.julia
def test_julia_run_with_state_noise():
    study = SimulationStudy.from_file(db.Cortes2013)
    exp = study.get_experiment(0)
    # Attach simple per-state noise intensity
    for sv in exp.model.state_variables.values():
        sv.noise = Noise(intensity=Parameter(name=f"sigma_{sv.name}", value=0.05))
    ts = exp.model.run('julia', save=False)
    assert ts.data.shape[1] == len(exp.model.state_variables)
    # Basic stochastic variability heuristic: successive differences should not all be zero
    diffs = (ts.data[1:10,:,0,0] - ts.data[:9,:,0,0]).std()
    assert diffs > 0.0
