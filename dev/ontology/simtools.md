# awesome-brain-simulation

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Schema.org](https://img.shields.io/badge/schema.org-reused-blue)](https://schema.org/SoftwareApplication)
[![LinkML](https://img.shields.io/badge/LinkML-ready-purple)](https://linkml.io)
[![FAIR](https://img.shields.io/badge/FAIR-metadata-green)](https://www.fairsharing.org)

A curated, **schema.org + LinkML** list of brain simulation tools, standards, and interoperability layers, spanning `channel → neuron → neural-network → neural-mass → network-system`.  
The schema reuses standard software metadata from `schema.org` and only adds TVBO-specific fields for scale, paradigm, and interoperability.

## Schema mapping

| README field | Schema / ontology term |
|---|---|
| Name | `schema:name` |
| Description | `schema:description` |
| Homepage | `schema:url` |
| Repository | `schema:codeRepository` |
| Programming language | `schema:programmingLanguage` |
| Runtime / backend | `schema:runtimePlatform` |
| License | `schema:license` |
| Maintainer | `schema:maintainer` |
| Application category | `schema:applicationCategory` |
| Scale | `tvbo:scale` |
| Paradigm | `tvbo:modelParadigm` |
| Tool role | `tvbo:toolRole` |
| Interoperates with | `tvbo:interoperatesWith` |
| Input/output formats | `tvbo:inputFormats` / `tvbo:outputFormats` |

---

## Full table

> Notes:
> - `Scale` is multi-valued.
> - `Tool role` is multi-valued where needed.
> - Official repositories are used where available.
> - Repository links are to the canonical project, not mirrors or forks.

| Tool | Official repo / home | `schema:applicationCategory` | `tvbo:scale` | `tvbo:modelParadigm` | `tvbo:toolRole` | `schema:programmingLanguage` | `schema:runtimePlatform` | `tvbo:interoperatesWith` | `schema:license` |
|---|---|---|---|---|---|---|---|---|---|
| The Virtual Brain (TVB) | https://github.com/the-virtual-brain/tvb-root | simulation; whole-brain | neural-mass, network-system | neural-mass, mean-field, whole-brain | simulator | Python | Python, HPC | NEST, NEURON, Arbor, NeuroML | open-source |
| TVB-Collab | https://github.com/the-virtual-brain/tvb-collab | collaboration; simulation | network-system | whole-brain | workflow-framework | Python | Python | TVB | open-source |
| TVB-Optim | https://virtual-twin.github.io/tvboptim/ | optimization; simulation | neural-mass, network-system | neural-mass, optimization | optimization-framework | Python, JAX | JAX | TVB | open-source |
| neurolib | https://github.com/neurolib-dev/neurolib | simulation; whole-brain | neural-mass, network-system | neural-mass | simulator | Python | Python | TVB, BIDS, fMRI workflows | open-source |
| PyRates | https://github.com/pyrates-neuroscience/PyRates | simulation; neural-mass | neural-mass, network-system | rate-based, neural-mass | simulator | Python | Python | TVB, dynamical-systems workflows | open-source |
| NEST | https://github.com/nest/nest-simulator | simulation; spiking | neuron, neural-network, network-system | spiking, plasticity | simulator | C++, Python | MPI, OpenMP | PyNN, NeuroML, TVB | GPL-3.0 |
| NEURON | https://github.com/neuronsimulator/nrn | simulation; biophysical | channel, neuron, network-system | conductance-based, compartmental, spiking | simulator | Python, C++ | HPC, MPI | CoreNEURON, NeuroML, PyNN, TVB | open-source |
| CoreNEURON | https://github.com/bluebrain/CoreNeuron | backend-runtime; simulation | channel, neuron, network-system | conductance-based, compartmental | backend-runtime | C++ | MPI, GPU | NEURON | open-source |
| Arbor | https://github.com/arbor-sim/arbor | simulation; HPC | channel, neuron, network-system | compartmental, spiking | simulator | C++, Python | GPU, MPI | NeuroML, NEURON, TVB | open-source |
| Brian2 | https://brian2.readthedocs.io | simulation; spiking | neuron, neural-network, network-system | spiking | simulator | Python | C++ standalone, Python | PyNN, NeuroML, TVB | GPL-3.0 |
| BrainPy | https://github.com/brainpy/BrainPy | simulation; framework | channel, neuron, neural-mass, neural-network, network-system | spiking, neural-mass, mean-field | framework | Python | JAX, XLA | TVB, NEURON, NEST, Brian2 | GPL-3.0 |
| MOOSE | https://github.com/BhallaLab/moose | simulation; multi-scale | channel, neuron, network-system | reaction-diffusion, compartmental, spiking | simulator | C++, Python | HPC | NeuroML, NEURON, channel modeling | GPL-3.0 |
| LFPy | https://github.com/LFPy/LFPy | analysis; simulation | neuron, network-system | compartmental, forward-modeling | analysis-tool | Python | NEURON | NEURON, HNN, TVB | GPL-3.0 |
| HNN-core | https://github.com/jonescompneurolab/hnn-core | simulation; MEG/EEG | neuron, network-system | neocortical, spiking | simulator | Python | NEURON | TVB, LFPy, NWB | BSD-3-Clause |
| GeNN | https://github.com/genn-team/genn | simulation; GPU-SNN | neuron, neural-network, network-system | spiking | simulator | Python, C++ | CUDA | PyNN, Brian2, NEURON | LGPL/GPL family |
| pynn_genn | https://github.com/genn-team/pynn_genn | interface; simulation | neuron, neural-network, network-system | spiking | workflow-framework | Python | CUDA | PyNN, GeNN | open-source |
| CARLsim | https://uci-carl.github.io/CARLsim3/ | simulation; GPU-SNN | neuron, neural-network, network-system | spiking | simulator | C++ | CUDA, GPU | PyNN, Brian2 | open-source |
| sPyNNaker | https://github.com/SpiNNakerManchester/sPyNNaker | runtime; neuromorphic | neuron, neural-network, network-system | spiking | backend-runtime | Python | SpiNNaker hardware | PyNN | open-source |
| PyNN | https://github.com/NeuralEnsemble/PyNN | interoperability; simulation | neuron, neural-network, network-system | spiking, generic | specification-language, workflow-framework | Python | backend-dependent | NEST, NEURON, Brian2, sPyNNaker, GeNN | GPL-3.0 |
| NeuroML | https://github.com/NeuroML | specification | channel, neuron, neural-network, network-system | conductance-based, spiking, neural-mass | specification-language | XML, Python, Java | backend-dependent | NEURON, NEST, Arbor, Brian2, jNeuroML | LGPL/GPL family |
| jNeuroML | https://github.com/NeuroML/jNeuroML | specification; simulation | channel, neuron, neural-network, network-system | conductance-based, spiking | simulator, specification-language | Java | Java | NeuroML, LEMS | LGPL-3.0 |
| pyNeuroML | https://github.com/NeuroML/pyNeuroML | specification; simulation | channel, neuron, neural-network, network-system | conductance-based, spiking | workflow-framework | Python | Python | NeuroML, LEMS, jNeuroML | LGPL-3.0 |
| LEMS | https://github.com/NeuroML/Documentation | specification | channel, neuron, neural-network, network-system | model-description | specification-language | XML | backend-independent | NeuroML | open-source |
| SpineCreator | https://spineml.github.io/spinecreator/ | editor; simulation | neuron, neural-network, network-system | layered neural models | workflow-framework | C++ | Qt/OpenGL | SpineML, SpineML toolchain | open-source |
| SpineML | https://github.com/SpineML | specification | neuron, neural-network, network-system | spiking, layered networks | specification-language | XML | backend-dependent | SpineCreator, BRAHMS, PyNN backend | open-source |
| NineML | https://github.com/nineml/nineml | specification | neuron, neural-network, network-system | spiking, generic | specification-language | Python, XML | backend-dependent | simulator-independent model exchange | open-source |
| neuroConstruct | https://github.com/NeuralEnsemble/neuroConstruct | model-building; simulation | channel, neuron, network-system | morphologically detailed, compartmental | workflow-framework | Java | Java3D, NEURON, GENESIS | NEURON, PyNN, NeuroML | open-source |
| Neurofitter | https://www.neuro-dynamix.com | optimization | neuron, network-system | parameter tuning | optimization-framework | software tool | backend-dependent | NEURON, GENESIS | legacy/open |
| PSICS | https://github.com/nachtigi/PSIC | simulation | channel, neuron, network-system | conductance-based, compartmental | simulator | Java, C++ | Java, HPC | neuron models, morphological workflows | MPL-2.0 |
| MCell | https://github.com/mcellteam/mcell | simulation; reaction-diffusion | channel, neuron | reaction-diffusion | simulator | C++, Python | HPC | morphology, biochemical signaling | open-source |
| STEPS | https://groups.oist.jp/cnu/software | simulation; reaction-diffusion | channel, neuron | stochastic reaction-diffusion | simulator | C++, Python | MPI, HPC | NeuroML, morphology workflows | open-source |
| PyRhO | https://github.com/berenslab/PyRhO | modeling; optogenetics | neuron | channel-based, conductance-based | analysis-tool | Python | Python | NEURON, optogenetics workflows | open-source |
| NetPyNE | https://github.com/Neurosim-lab/netpyne | simulation; workflow | neuron, neural-network, network-system | spiking, multiscale | workflow-framework | Python | NEURON backend | NEURON, TVB, PyNN, NeuroML | BSD-3-Clause |
| NetPyNE-UI | https://github.com/MetaCell/NetPyNE-UI | UI; workflow | neuron, neural-network, network-system | spiking | visualization-tool | JavaScript, Python | browser | NetPyNE | open-source |
| BSB | https://github.com/BlueBrain/bsb-core | scaffold-builder | neuron, network-system | multi-scale scaffold | workflow-framework | Python | HPC | SONATA, NEURON, BluePyOpt | open-source |
| BluePyOpt | https://github.com/BlueBrain/BluePyOpt | optimization | channel, neuron | parameter optimization | optimization-framework | Python | Python | NEURON, optimization workflows | GPL-3.0 |
| Blue Brain atlas / tools | https://github.com/BlueBrain | data; modeling | network-system | multi-scale | workflow-framework | Python, C++ | HPC | NEURON, Arbor, SONATA | open-source |
| SONATA tools | https://github.com/AllenInstitute/sonata | data format; simulation | neuron, neural-network, network-system | network exchange | specification-language | Python | backend-dependent | NEST, NEURON, Brain modeling ecosystem | BSD-3-Clause |
| BrainBrowser | https://github.com/aces/brainbrowser | visualization | network-system | visualization | visualization-tool | JavaScript, WebGL | browser | neuroimaging workflows | MIT |
| MNE-Python | https://github.com/mne-tools/mne-python | analysis; visualization | network-system | electrophysiology | analysis-tool | Python | Python | EEG/MEG, NWB | BSD-3-Clause |
| FieldTrip | https://github.com/fieldtrip/fieldtrip | analysis; visualization | network-system | EEG/MEG | analysis-tool | MATLAB | MATLAB | EEG/MEG workflows | GPL |
| EEGLAB | https://sccn.ucsd.edu/eeglab/ | analysis; visualization | network-system | EEG | analysis-tool | MATLAB | MATLAB | EEG workflows | GPL |
| SPM | https://github.com/spm/spm12 | analysis; imaging | network-system | neuroimaging | analysis-tool | MATLAB | MATLAB | fMRI, structural MRI | GPL |
| Brainstorm | https://github.com/brainstorm-tools/brainstorm3 | analysis; visualization | network-system | EEG/MEG | analysis-tool | MATLAB | MATLAB | EEG/MEG workflows | BSD |
| Neurodata Without Borders (NWB) | https://github.com/NeurodataWithoutBorders | format; interoperability | network-system | data standard | specification-language | Python | backend-independent | HNN, TVB, LFPy, MNE | open-source |
| Elephant | https://github.com/NeuralEnsemble/elephant | analysis | neuron, network-system | spike-train analysis | analysis-tool | Python | Python | Neo, NEST, Brian2 | BSD-3-Clause |
| Neo | https://github.com/NeuralEnsemble/python-neo | analysis; format | neuron, network-system | electrophysiology | specification-language | Python | Python | Elephant, NEST, Brian2 | BSD-3-Clause |
| SpikeInterface | https://github.com/SpikeInterface/spikeinterface | analysis | network-system | electrophysiology | analysis-tool | Python | Python | NWB, Neo, MNE | MIT |
| Open Source Brain | https://github.com/OpenSourceBrain | repository; interoperability | channel, neuron, network-system | model exchange | model-repository | GitHub org | browser | NeuroML, NEURON, jNeuroML | open-source |
| ModelDB | https://modeldb.science/ | repository; interoperability | channel, neuron, network-system | model exchange | model-repository | web | browser | NEURON, GENESIS, NeuroML | open |
| OpenWorm | https://github.com/openworm | simulation; connectomics | neuron, network-system | whole-organism | simulator | Python, C++ | HPC | NEURON, TVB | open-source |
| CxSystem2 | https://www.ebrains.eu/tools/cxsystem2 | simulation | neuron, network-system | cortical microcircuit | simulator | Python | HPC | TVB, NEST | open-source |
| FastDMF | https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.866517/full | simulation; neural-mass | neural-mass, network-system | dynamic mean field | simulator | C++/Python | HPC | TVB, fMRI workflows | open-source |
| BrainSimII | https://github.com/FutureAIGuru/BrainSimII | simulation | neuron, network-system | AGI-inspired neural simulation | simulator | C++/Python | HPC | unspecified | open-source |
| neuroConstruct tutorials / assets | https://neuroconstruct.org | model-building | channel, neuron | morphologically detailed | workflow-framework | Java | Java | NEURON, NeuroML | open-source |
