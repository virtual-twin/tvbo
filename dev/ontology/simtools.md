# Simulation Tools for Computational Neuroscience

A curated catalog of brain simulation tools, standards, and interoperability layers.
Each entry maps to a `SimulationTool` instance in the TVBO LinkML schema (`schema/software.yaml`),
with corresponding YAML files in `tvbo/database/software/`.

Spans scales from `channel → neuron → neural_network → neural_mass → network_system`.

## Schema mapping

| Field | LinkML slot | Schema.org mapping |
|---|---|---|
| Name | `name` | `schema:name` |
| Description | `description` | `schema:description` |
| Homepage | `homepage` | `schema:url` |
| Repository | `repository` | `schema:codeRepository` |
| DOI | `doi` | `schema:identifier` |
| Programming language | `programming_language` | `schema:programmingLanguage` |
| Runtime / backend | `runtime_platform` | `schema:runtimePlatform` |
| License (SPDX) | `license` | `schema:license` |
| Application category | `application_category` | `schema:applicationCategory` |
| Scale | `scale` | `tvbo:scale` |
| Model paradigm | `model_paradigm` | `tvbo:modelParadigm` |
| Tool role | `tool_role` | `tvbo:toolRole` |
| Interoperates with | `interoperates_with` | `tvbo:interoperatesWith` |
| Ecosystem | `ecosystem` | — |

## Enum values

### SimulationScale
`channel`, `neuron`, `neural_network`, `neural_mass`, `network_system`

### ToolRole
`simulator`, `framework`, `backend_runtime`, `optimization_framework`, `specification_language`,
`workflow_framework`, `analysis_tool`, `visualization_tool`, `model_repository`, `continuation_tool`

### ModelParadigm
`neural_mass`, `mean_field`, `spiking`, `conductance_based`, `compartmental`, `rate_based`,
`phase_oscillator`, `reaction_diffusion`, `plasticity`, `generic`, `multiscale`,
`dynamic_mean_field`, `data_standard`, `model_description`, `bifurcation_analysis`

---

## Full table

> Notes:
> - Scale, tool role, model paradigm, and interoperates_with are multi-valued.
> - Licenses use SPDX identifiers where possible.
> - Repository links point to canonical sources.

| Tool | Repository | Category | Scale | Paradigm | Role | Language | Runtime | Interoperates with | License |
|---|---|---|---|---|---|---|---|---|---|
| The Virtual Brain (TVB) | https://github.com/the-virtual-brain/tvb-root | simulation | neural_mass, network_system | neural_mass, mean_field | simulator | Python, C | Python, NumPy | NEST, NEURON, Arbor, NeuroML, PyRates, neurolib | GPL-3.0-or-later |
| TVB-Optim | https://github.com/virtual-twin/tvboptim | optimization | neural_mass, network_system | neural_mass, dynamic_mean_field | optimization_framework | Python | JAX | TVB | EUPL-1.2 |
| neurolib | https://github.com/neurolib-dev/neurolib | simulation | neural_mass, network_system | neural_mass, mean_field | simulator | Python | NumPy | TVB | MIT |
| PyRates | https://github.com/pyrates-neuroscience/PyRates | simulation | neural_mass, network_system | rate_based, neural_mass | simulator, framework | Python | Python, NumPy, TensorFlow | TVB | BSD-3-Clause |
| NEST | https://github.com/nest/nest-simulator | simulation | neuron, neural_network, network_system | spiking, plasticity | simulator | C++, Python | MPI, OpenMP | PyNN, NeuroML, TVB, SONATA | GPL-2.0-or-later |
| NEURON | https://github.com/neuronsimulator/nrn | simulation | channel, neuron, neural_network, network_system | conductance_based, compartmental, spiking | simulator | Python, C++, HOC | MPI, CoreNEURON | CoreNEURON, NeuroML, PyNN, TVB, NetPyNE, SONATA | BSD-3-Clause |
| CoreNEURON | https://github.com/BlueBrain/CoreNeuron | simulation | channel, neuron, network_system | conductance_based, compartmental | backend_runtime | C++ | MPI, GPU, NMODL | NEURON | BSD-3-Clause |
| Arbor | https://github.com/arbor-sim/arbor | simulation | channel, neuron, network_system | compartmental, spiking | simulator | C++, Python | GPU, MPI, SIMD | NeuroML, NEURON, TVB, SONATA | BSD-2-Clause |
| Brian2 | https://github.com/brian-team/brian2 | simulation | neuron, neural_network, network_system | spiking | simulator | Python, C++ | C++ standalone, Python, OpenMP | PyNN, NeuroML, Brian2GeNN | CeCILL-2.1 |
| BrainPy | https://github.com/brainpy/BrainPy | simulation | channel, neuron, neural_mass, neural_network, network_system | spiking, neural_mass, mean_field | framework | Python | JAX, XLA | TVB, NEURON, NEST, Brian2 | Apache-2.0 |
| MOOSE | https://github.com/BhallaLab/moose-core | simulation | channel, neuron, network_system | reaction_diffusion, compartmental, spiking | simulator | C++, Python | Python | NeuroML, NEURON | GPL-3.0-only |
| LFPy | https://github.com/LFPy/LFPy | analysis | neuron, network_system | compartmental | analysis_tool | Python | NEURON backend | NEURON, HNN-core, TVB | GPL-3.0-only |
| HNN-core | https://github.com/jonescompneurolab/hnn-core | simulation | neuron, network_system | spiking, compartmental | simulator | Python | NEURON backend | TVB, LFPy, NWB, MNE-Python | BSD-3-Clause |
| GeNN | https://github.com/genn-team/genn | simulation | neuron, neural_network, network_system | spiking | simulator | C++, Python | CUDA, OpenCL | PyNN, Brian2, NEURON | GPL-2.0-only |
| pynn_genn | https://github.com/genn-team/pynn_genn | interface | neuron, neural_network, network_system | spiking | workflow_framework | Python | CUDA | PyNN, GeNN | GPL-2.0-only |
| CARLsim | https://github.com/UCI-CARL/CARLsim6 | simulation | neuron, neural_network, network_system | spiking, plasticity | simulator | C++, Python | CUDA | PyNN | MIT |
| sPyNNaker | https://github.com/SpiNNakerManchester/sPyNNaker | simulation | neuron, neural_network, network_system | spiking | backend_runtime | Python | SpiNNaker hardware | PyNN | GPL-3.0-only |
| PyNN | https://github.com/NeuralEnsemble/PyNN | interoperability | neuron, neural_network, network_system | spiking, generic | specification_language, workflow_framework | Python | backend-dependent | NEST, NEURON, Brian2, sPyNNaker, GeNN | CeCILL-2.0 |
| NeuroML | https://github.com/NeuroML/NeuroML2 | specification | channel, neuron, neural_network, network_system | conductance_based, spiking, neural_mass | specification_language | XML, Python, Java | backend-dependent | NEURON, NEST, Arbor, Brian2, jNeuroML, pyNeuroML | LGPL-3.0-only |
| jNeuroML | https://github.com/NeuroML/jNeuroML | specification | channel, neuron, neural_network, network_system | conductance_based, spiking | simulator, specification_language | Java | JVM | NeuroML, LEMS, NEURON, NEST, Brian2 | LGPL-3.0-only |
| pyNeuroML | https://github.com/NeuroML/pyNeuroML | specification | channel, neuron, neural_network, network_system | conductance_based, spiking | workflow_framework | Python | Python | NeuroML, LEMS, jNeuroML, NEURON, NEST | LGPL-3.0-only |
| LEMS | https://github.com/LEMS/jLEMS | specification | channel, neuron, neural_network, network_system | model_description | specification_language | XML, Java | JVM | NeuroML | MIT |
| SpineCreator | https://github.com/SpineML/SpineCreator | editor | neuron, neural_network, network_system | spiking | workflow_framework | C++ | Qt | SpineML, BRAHMS | GPL-3.0-only |
| SpineML | https://github.com/SpineML | specification | neuron, neural_network, network_system | spiking | specification_language | XML | backend-dependent | SpineCreator, BRAHMS, PyNN | BSD-3-Clause |
| NineML | https://github.com/INCF/nineml-spec | specification | neuron, neural_network, network_system | spiking, generic | specification_language | Python, XML | backend-dependent | PyNN, NeuroML | BSD-2-Clause |
| neuroConstruct | https://github.com/NeuralEnsemble/neuroConstruct | model-building | channel, neuron, network_system | compartmental | workflow_framework | Java | NEURON, GENESIS | NEURON, PyNN, NeuroML | GPL-2.0-only |
| Neurofitter | https://github.com/ModelDBRepository/64261 | optimization | neuron, network_system | spiking | optimization_framework | C++ | MPI | NEURON, GENESIS | GPL-2.0-only |
| PSICS | https://github.com/BorgwardtLab/PSICS | simulation | channel, neuron | conductance_based, compartmental | simulator | Java | JVM | NeuroML | MPL-2.0 |
| MCell | https://github.com/mcellteam/mcell | simulation | channel, neuron | reaction_diffusion | simulator | C++, Python | Python | CellBlender, NeuroML | GPL-2.0-only |
| STEPS | https://github.com/CNS-OIST/STEPS | simulation | channel, neuron | reaction_diffusion | simulator | C++, Python | MPI, Python | NeuroML | GPL-3.0-only |
| PyRhO | https://github.com/ProjectPyRhO/PyRhO | modeling | neuron | conductance_based | analysis_tool | Python | Python | NEURON, Brian2 | BSD-3-Clause |
| NetPyNE | https://github.com/Neurosim-lab/netpyne | simulation | neuron, neural_network, network_system | spiking, multiscale | workflow_framework | Python | NEURON backend | NEURON, TVB, PyNN, NeuroML, SONATA | MIT |
| NetPyNE-UI | https://github.com/MetaCell/NetPyNE-UI | visualization | neuron, neural_network, network_system | spiking | visualization_tool | JavaScript, Python | browser | NetPyNE | MIT |
| BSB | https://github.com/dbbs-lab/bsb-core | scaffold | neuron, network_system | multiscale | workflow_framework | Python | Python, HPC | SONATA, NEURON, Arbor, NEST | Apache-2.0 |
| BluePyOpt | https://github.com/BlueBrain/BluePyOpt | optimization | channel, neuron | conductance_based | optimization_framework | Python | Python, iPyParallel | NEURON, eFEL | LGPL-3.0-only |
| SONATA | https://github.com/AllenInstitute/sonata | specification | neuron, neural_network, network_system | generic | specification_language | Python | backend-dependent | NEST, NEURON, Arbor, NetPyNE | BSD-3-Clause |
| MNE-Python | https://github.com/mne-tools/mne-python | analysis | network_system | generic | analysis_tool | Python | Python, NumPy | NWB, FieldTrip, EEGLAB | BSD-3-Clause |
| FieldTrip | https://github.com/fieldtrip/fieldtrip | analysis | network_system | generic | analysis_tool | MATLAB | MATLAB | MNE-Python, SPM, EEGLAB | GPL-3.0-only |
| EEGLAB | https://github.com/sccn/eeglab | analysis | network_system | generic | analysis_tool | MATLAB | MATLAB | FieldTrip, MNE-Python | BSD-2-Clause |
| SPM | https://github.com/spm/spm | analysis | network_system | generic | analysis_tool | MATLAB | MATLAB | FieldTrip, FSL, FreeSurfer | GPL-2.0-or-later |
| Brainstorm | https://github.com/brainstorm-tools/brainstorm3 | analysis | network_system | generic | analysis_tool | MATLAB | MATLAB | MNE-Python, FieldTrip, SPM | GPL-3.0-only |
| NWB | https://github.com/NeurodataWithoutBorders/pynwb | specification | network_system | data_standard | specification_language | Python | Python, HDF5 | HNN-core, TVB, LFPy, MNE-Python, SpikeInterface | BSD-3-Clause |
| Elephant | https://github.com/NeuralEnsemble/elephant | analysis | neuron, network_system | generic | analysis_tool | Python | Python, NumPy | Neo, NEST, Brian2, NWB | BSD-3-Clause |
| Neo | https://github.com/NeuralEnsemble/python-neo | data | neuron, network_system | data_standard | specification_language | Python | Python | Elephant, NEST, Brian2, NWB, SpikeInterface | BSD-3-Clause |
| SpikeInterface | https://github.com/SpikeInterface/spikeinterface | analysis | neuron, network_system | generic | analysis_tool | Python | Python | NWB, Neo, MNE-Python | MIT |
| Open Source Brain | https://github.com/OpenSourceBrain | repository | channel, neuron, network_system | generic | model_repository | Python, Java | browser | NeuroML, NEURON, jNeuroML | LGPL-3.0-only |
| ModelDB | https://modeldb.science/ | repository | channel, neuron, network_system | generic | model_repository | — | browser | NEURON, GENESIS, NeuroML | — |
| OpenWorm | https://github.com/openworm/OpenWorm | simulation | neuron, network_system | compartmental, spiking | simulator | Python, C++ | Python | NEURON, NeuroML, c302 | MIT |
| CxSystem2 | https://github.com/VisualNeuroscience-UH/CxSystem2 | simulation | neuron, network_system | spiking | simulator | Python | Brian2 backend | Brian2, NEST | MIT |
| FastDMF | https://github.com/Picardian14/FastDMF | simulation | neural_mass, network_system | dynamic_mean_field | simulator | C++, Python | C++ | TVB | MIT |
| Nengo | https://github.com/nengo/nengo | simulation | neuron, neural_network, network_system | spiking, rate_based | framework | Python | Python, NumPy, TensorFlow, NengoLoihi | NengoLoihi, NengoDL, SpiNNaker | MIT |
| GENESIS | https://github.com/genesis-sim/genesis-2.4 | simulation | channel, neuron, network_system | compartmental, conductance_based, spiking | simulator | C | C | NEURON, NeuroML, neuroConstruct | GPL-2.0-only |
| NetworkDynamics.jl | https://github.com/PIK-ICoNe/NetworkDynamics.jl | simulation | neural_mass, network_system | neural_mass, generic | simulator, framework | Julia | DifferentialEquations.jl | DifferentialEquations.jl, Graphs.jl | MIT |
| Neuroblox.jl | https://github.com/Neuroblox/Neuroblox.jl | simulation | neural_mass, neural_network, network_system | neural_mass, spiking, mean_field | framework | Julia | ModelingToolkit.jl, DifferentialEquations.jl | NetworkDynamics.jl, TVB | MIT |
| jaxley | https://github.com/jaxleyverse/jaxley | simulation | channel, neuron, network_system | compartmental, conductance_based | simulator | Python | JAX | NEURON, SWC morphologies | Apache-2.0 |
| BifurcationKit.jl | https://github.com/bifurcationkit/BifurcationKit.jl | analysis | neural_mass, network_system | bifurcation_analysis | continuation_tool | Julia | DifferentialEquations.jl | NetworkDynamics.jl, AUTO-07p | MIT |
| AUTO-07p | https://github.com/auto-07p/auto-07p | analysis | neural_mass, network_system | bifurcation_analysis | continuation_tool | Fortran, C | Fortran runtime | MatCont, BifurcationKit.jl | BSD-2-Clause |
| MatCont | https://sourceforge.net/projects/matcont/ | analysis | neural_mass, network_system | bifurcation_analysis | continuation_tool | MATLAB | MATLAB | AUTO-07p | GPL-3.0-only |
| DifferentialEquations.jl | https://github.com/SciML/DifferentialEquations.jl | simulation | neural_mass, network_system | generic | framework | Julia | Julia, GPU | NetworkDynamics.jl, Neuroblox.jl, BifurcationKit.jl | MIT |
| BrainBrowser | https://github.com/aces/brainbrowser | visualization | network_system | generic | visualization_tool | JavaScript | browser, WebGL | neuroimaging formats | MIT |
| BrainSimII | https://github.com/FutureAIGuru/BrainSimII | simulation | neuron, network_system | spiking | simulator | C# | Windows | — | MIT |
| Snudda | https://github.com/Hjorthmedansen/Snudda | simulation | neuron, network_system | spiking, compartmental | workflow_framework | Python | NEURON backend | NEURON, SONATA | GPL-3.0-only |
