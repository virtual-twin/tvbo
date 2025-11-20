#
# Module: test_all_backends.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
import numpy as np
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import integrators
from tvb.simulator.backend import ReferenceBackend
from tvb.simulator.backend.nb import NbBackend
from tvb_autodiff.jax import JaxBackend

from tvbo.export import templater
from tvbo.knowledge import ontology
from tvbo.benchmark.bench_against_mpr_backend import Bench

get_logger("tvb.datatypes").setLevel("ERROR")
get_logger("tvb.datatypes.Connectivity").setLevel("ERROR")
get_logger("tvb.simulator").setLevel("ERROR")
get_logger("tvb.simulator.simulator").setLevel("ERROR")
get_logger("tvb.simulator.history").setLevel("ERROR")
get_logger("tvb.simulator.noise").setLevel("ERROR")
get_logger("tvb.simulator.integrators").setLevel("ERROR")
get_logger("tvb.simulator.models").setLevel("ERROR")

# %%
models = list()
for m in ontology.get_models():
    models.append(
        templater.model2class(m, print_source=False, split_nonintegrated_variables=True)
    )
conn = Connectivity.from_file()
connectivities = [conn]

test = Bench(
    backends=[ReferenceBackend, NbBackend, JaxBackend],
    models=models,
    integrators=[
        integrators.HeunDeterministic(),
        integrators.EulerDeterministic(),
        integrators.HeunStochastic(),
        integrators.EulerStochastic(),
    ],
    connectivities=connectivities,
    conductions=[np.Inf],
    int_dts=[0.001],
    sim_lengths=[2],  # list(np.arange(100, 10000, 1000)),
)
results = test.run()
# test.report()
# test.plot_report()
import pickle

with open("tests/backend_test.pkl", "wb") as f:
    pickle.dump(results, f)
