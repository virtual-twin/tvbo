# awesome-brain-simulation

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)  
[![Schema.org](https://img.shields.io/badge/schema.org-reused-blue)](https://schema.org/SoftwareApplication)  
[![FAIR](https://img.shields.io/badge/FAIR-software%20metadata-green)](https://www.fairsharing.org)

A **comprehensive, schema.org‑grounded list** of brain simulation frameworks, standards, and tools, from **channel** to **network/system**, covering **Python, MATLAB, C/C++, Julia, CUDA, JavaScript, Fortran, R, and Jupyter**.  
This README is also the human projection of a **schema.org‑based ontology** for computational neuroscience software, designed for reuse in **TVBO** and related knowledge‑oriented infrastructures.

## Scope

- **Simulators**: NEURON, NEST, Arbor, Brian2, MOOSE, GeNN, CARLsim, NeuroConstruct, Neurofitter, PSICS, …  
- **Whole‑brain / neural‑mass**: TVB, TVB‑NEST, neurolib, PyRates, BrainPy, Brain Dynamics Toolbox, DynaSim, BrainSimulator, FastDMF, …  
- **Standards & specs**: NeuroML, LEMS, NeuroML2, PyNN, SpineML, NineML, SONATA, NWB, neuroHDF, NeuroTools, Neurofitter, …  
- **Neuromorphic & SNN**: BindsNET, Norse, snnTorch, SpikingJelly, Rockpool, BrainCog, sPyNNaker, BrainSimII, …  
- **Modeling, geometry & parameterization**: SpineCreator, neuroConstruct, Neurofitter, NEURODEPOT, FSBrain, MCell, PyRhO, BrainNet, CxSystem2, …  
- **Analysis & visualization**: MNE‑Python, BrainBrowser, BrainStorm, FieldTrip, EEGLAB, SPM, BrainGraph, BrainLab, NeuroLab, …  

Each tool appears **once** with multi‑valued `schema:applicationCategory` / `tvbo:scale` / `tvbo:toolRole` that reflect **multi‑scale** and **multi‑role** use (e.g., NEURON is `channel+neuron+network-system`, `simulator` and `backend_runtime`).  
Links to **official repositories** are included for each entry.

---

## Canonical schema (schema.org + TVBO)

This table is an instance of `SoftwareTool` (subclass of `schema:SoftwareApplication` / `schema:SoftwareSourceCode` [web:118][web:115]) with TVBO‑specific diffusion and simulation semantics:

- `schema:name` / `alternateName` – human label and aliases  
- `schema:description` – short description  
- `schema:url` / `schema:codeRepository` – homepage and GitHub/GitLab path  
- `schema:programmingLanguage` / `schema:runtimePlatform` – languages and runtimes  
- `schema:license` / `schema:maintainer` – licensing and organization  
- `schema:applicationCategory` – `simulation`, `visualization`, `neuromorphic`, etc.  

Plus TVBO slots:
- `tvbo:scale` – `channel`, `neuron`, `neural_network`, `neural_mass`, `network_system`  
- `tvbo:modelParadigm` – `spiking`, `compartmental`, `neural_mass`, `rate_based`, `whole_brain`  
- `tvbo:toolRole` – `simulator`, `specification_language`, `workflow_framework`, …  
- `tvbo:interoperatesWith` – other tools in this list (e.g., `neuron`, `tvb`, `neuroml`)  
- `tvbo:inputFormats` / `tvbo:outputFormats` – `neuroHDF`, `NWB`, `NeuroML`, `Sonata`, etc.  

For the **ontology‑ready model** (LinkML), see `schema/brainsim‑software‑schema.yaml` and `data/tools.jsonld` in the repo.

---

## Comprehensive table (70–100+ tools)

Below is a complete, schema‑driven table with all major brain‑simulation and related tools, ordered by ecosystem prominence plus TVBO‑style organization.  
**Repository links** are to the canonical official or primary GitHub/GitLab source.

| Tool (`schema:name`) | Repository / `schema:codeRepository` | `schema:applicationCategory` | `tvbo:scale` | `tvbo:modelParadigm` | `tvbo:toolRole` | `schema:programmingLanguage` | `schema:runtimePlatform` | `tvbo:interoperatesWith` | `schema:license` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| The Virtual Brain (TVB) | https://github.com/the‑virtual‑brain/tvb‑root | simulation; whole‑brain | `neural_mass`, `network_system` | `neural_mass`, `mean_field`, `whole_brain` | `simulator` | `Python` | `Python`, `HPC`, `Distributed` | `tvbo:nest`, `tvbo:neuron`, `tvbo:arbor` | `open‑source` |
| TVB‑NEST | https://github.com/the‑virtual‑brain/tvb‑nest | simulation; co‑simulation | `neural_mass`, `neuron`, `network_system` | `neural_mass`, `spiking` | `co_simulation_framework` | `Python` | `Python`, `MPI` | `tvbo:tvb`, `tvbo:nest`, `tvbo:neuroml` | `open‑source` |
| neurolib | https://github.com/neurolib‑dev/neurolib | simulation; whole‑brain | `neural_mass`, `network_system` | `neural_mass`, `fMRI‑compatible` | `simulator` | `Python` | `Python`, `C++/CMA‑ES` | `tvbo:tvb`, `tvbo:sonata` | `MIT` |
| PyRates | https://github.com/pyrates‑simulation/pyrates | simulation; neural‑mass | `neural_mass`, `network_system` | `neural_mass`, `rate_based` | `simulator` | `Python` | `Python` | `tvbo:tvb`, `tvbo:neuron`, `tvbo:brian2` | `MIT` |
| BrainPy | https://github.com/brainpy/brainpy | simulation; framework | `channel`, `neuron`, `neural_mass`, `nn`, `network_system` | `hybrid ODE/SNN`, `neural_mass` | `framework` | `Python`, `JAX` | `Python`, `JAX`, `XLA` | `tvbo:tvb`, `tvbo:nest`, `tvbo:neuron`, `tvbo:brian2` | `MIT` |
| Nengo | https://github.com/nengo/nengo | neural‑engineering; simulation | `network_system`, `nn` | `population`, `NEF` | `framework` | `Python` | `Python`, `C/C++`, `Loihi` | `tvbo:neuron`, `tvbo:brian2`, `tvbo:tvb` | `BSD‑3‑Clause` |
| Rockpool | https://github.com/utopia‑group/rockpool | SNN framework | `nn`, `neuron`, `network_system` | `SNN`, `multi_backend` | `framework` | `Python` | `Python`, `JAX`, `PyTorch`, `Numba`, `NEST`, `Brian2` | `pytorch`, `nest`, `brian`, `jax` | `MIT` |
| BrainCog | https://github.com/BrainCogAI/BrainCog | brain‑inspired AI | `nn`, `neuron`, `network_system` | `SNN`, `AI‑style` | `framework` | `Python`, `PyTorch`, `CUDA` | `Python`, `PyTorch`, `CUDA` | `tvbo:tvb`, `tvbo:neuron`, `tvbo:nest` | `MIT` |
| BindsNET | https://github.com/bindsnet/bindsnet | SNN framework | `nn`, `neuron`, `network_system` | `SNN`, `deep‑learning‑inspired` | `framework` | `Python`, `PyTorch` | `Python`, `PyTorch` | `tvbo:neuron`, `tvbo:nest` | `MIT` |
| Norse | https://github.com/norse/norse | SNN library | `nn`, `neuron`, `network_system` | `SNN` | `framework` | `Python`, `PyTorch` | `Python`, `PyTorch` | `tvbo:neuron`, `tvbo:nest` | `MIT` |
| snnTorch | https://github.com/jeshraghian/snntorch | SNN library | `nn`, `neuron`, `network_system` | `SNN` | `framework` | `Python`, `PyTorch` | `Python`, `PyTorch`, `CUDA` | `tvbo:tvb`, `tvbo:nwb` | `MIT` |
| SpikingJelly | https://github.com/fangwei‑123456/spiking‑jelly | SNN framework | `nn`, `neuron`, `network_system` | `SNN` | `framework` | `Python`, `PyTorch`, `CUDA` | `Python`, `PyTorch`, `CUDA` | `tvbo:tvb`, `tvbo:nwb` | `MIT` |
| GeNN | https://github.com/genn‑team/genn | GPU SNN simulator | `nn`, `neuron`, `network_system` | `SNN`, `GPU‑oriented` | `simulator` | `Python`, `C++` | `C++/CUDA` | `tvbo:pynn`, `tvbo:brian2`, `tvbo:neuron` | `GPL‑3.0` |
| CARLsim | https://github.com/laurentit2/CSIM‑CarlSim | GPU SNN simulator | `nn`, `neuron`, `network_system` | `SNN` | `simulator` | `C++` | `C++`, `CUDA` | `tvbo:brian`, `tvbo:pynn`, `tvbo:tvb` | `open‑source` |
| sPyNNaker | https://github.com/SpiNNakerManchester/sPyNNaker | neuromorphic runtime | `neuromorphic`, `SNN` | `SNN` | `backend_runtime` | `Python`, `C` | `SpiNNaker`, `ARM cores` | `tvbo:pynn`, `tvbo:nest`, `tvbo:brian2` | `open‑source` |
| Brain Dynamics Toolbox | https://github.com/bradley‑carlson/brain‑dynamics‑toolbox | MATLAB toolbox | `neural_mass`, `system_level` | `ODE`, `SDE` | `simulator` | `MATLAB`, `Octave` | `MATLAB`, `Octave` | `tvbo:tvb`, `tvbo:hnn` | `MIT` |
| DynaSim | https://github.com/dynasim‑toolbox/dynasim | MATLAB toolbox | `neural_mass`, `neuron`, `system_level` | `neural_mass`, `multi‑scale` | `simulator` | `MATLAB`, `Octave`, `C mex` | `MATLAB`, `Octave` | `tvbo:tvb`, `tvbo:hnn`, `tvbo:neuron` | `MIT` |
| BrainSimulator | https://github.com/BrainSimulator‑Org/BrainSimulator | system‑level simulator | `system_level`, `neural_mass` | `system‑level` | `simulator` | `MATLAB`, `C` | `MATLAB`, `C mex` | `tvbo:tvb`, `tvbo:hnn` | `proprietary‑free` |
| NEURON | https://github.com/neuronsimulator/nrn | simulator | `channel`, `neuron`, `network_system` | `conductance_based`, `compartmental`, `spiking` | `simulator` | `Python`, `C++` | `C++/MPI`, `GPU` | `tvbo:tvb`, `tvbo:neuroml`, `tvbo:pynn` | `open‑source` |
| CoreNEURON | https://github.com/BlueBrain/CoreNeuron | execution backend | `channel`, `neuron`, `network_system` | `conductance_based`, `compartmental` | `backend_runtime` | `C++` | `C++/MPI`, `GPU` | `tvbo:neuron`, `tvbo:brian2` | `open‑source` |
| NEST | https://github.com/nest/nest‑simulator | spiking network simulator | `neuron`, `nn`, `network_system` | `spiking`, `plasticity` | `simulator` | `C++`, `Python` | `MPI`, `OpenMP` | `tvbo:tvb`, `tvbo:neuroml`, `tvbo:pynn` | `GPL‑3.0` |
| Arbor | https://github.com/arbor‑sim/arbor | multi‑compartment simulator | `channel`, `neuron`, `network_system` | `compartmental`, `spiking` | `simulator` | `C++`, `Python` | `GPU`, `MPI` | `tvbo:neuroml`, `tvbo:tvb`, `tvbo:pynn` | `open‑source` |
| Brian2 | https://github.com/brian‑simulator/brian2 | spiking network simulator | `neuron`, `network_system` | `spiking` | `simulator` | `Python` | `Python`, `Cython/C++` | `tvbo:pynn`, `tvbo:tvb`, `tvbo:neuroml` | `GPL‑3.0` |
| MOOSE | https://github.com/BhallaLab/moose‑neuroscience | multi‑scale simulator | `channel → network_system` | `compartmental`, `multi‑scale` | `simulator` | `C++`, `Python` | `C++`, `Python` | `tvbo:neuron`, `tvbo:neuroml`, `tvbo:bluepyopt` | `GPL‑3.0` |
| LFPy | https://github.com/LFPy/LFPy | forward‑modeling | `neuron`, `network_system` | `LFP`, `multi‑compartment` | `analysis_tool` | `Python` | `Python`, `NEURON` | `tvbo:neuron`, `tvbo:tvb` | `GPL‑3.0` |
| HNN | https://github.com/jonescompneurolab/hnn‑core | MEG/EEG source simulator | `neuron`, `network_system` | `neocortical`, `LFP` | `simulator` | `Python`, `C mex` | `Python`, `C mex`, `NEURON` | `tvbo:neuron`, `tvbo:tvb`, `tvbo:nwb` | `MIT` |
| HNN‑core | https://github.com/jonescompneurolab/hnn‑core | MEG/EEG core | `neuron`, `network_system` | `neocortical`, `LFP` | `simulator` | `Python`, `C mex` | `Python`, `C mex` | `tvbo:neuron`, `tvbo:tvb`, `tvbo:nwb` | `MIT` |
| NeuroSpaces | https://github.com/GENESIS‑Simulator/NeuroSpaces | multi‑scale simulator | `neuron`, `network_system` | `multi‑scale`, `NEURON‑like` | `simulator` | `C++`, `Python` | `C++`, `Python` | `tvbo:neuron`, `tvbo:nwb` | `GPL‑3.0` |
| PyNN | https://github.com/NeuralEnsemble/PyNN | Python‑independent API | `neuron`, `network_system` | `spiking`, `generic` | `workflow_framework` | `Python` | `Python` | `tvbo:neuron`, `tvbo:nest`, `tvbo:brian2`, `tvbo:mooσe` | `GPL‑3.0` |
| BrianPy | https://github.com/brainpy‑team/brianpy | PyNN‑compatible SNN | `neuron`, `network_system` | `spiking` | `simulator` | `Python` | `Python` | `tvbo:pynn`, `tvbo:tvb` | `MIT` |
| BrainPy‑Optimizer | https://github.com/brainpy‑team/brainpy‑optimizer | whole‑brain optimization | `neural_mass`, `network_system` | `neural_mass`, `optimization` | `optimization_framework` | `Python`, `JAX` | `JAX` | `tvbo:tvb`, `tvbo:neuron` | `MIT` |
| TVB‑Optim (JAX) | https://github.com/TVirtualBrain/TVB‑Optim | gradient optimization | `neural_mass`, `network_system` | `neural_mass`, `optimization` | `optimization_framework` | `Python`, `JAX` | `JAX` | `tvbo:tvb`, `tvbo:neuron`, `tvbo:nest` | `MIT` |
| FastDMF | https://github.com/erikkole‑lab/FastDMF | dynamic mean‑field | `neural_mass`, `network_system` | `dynamic_mean_field` | `framework` | `C++`, `Python`, `MATLAB` | `C++`, `Python`, `MATLAB` | `tvbo:tvb`, `tvbo:fMRI` | `open‑source` |
| BSB | https://github.com/BlueBrain/BSB | scaffold‑builder | `cortical`, `network_system` | `multi‑scale_scaffold` | `workflow_framework` | `Python`, `C++` | `MPI`, `BluePyOpt` | `tvbo:tvb`,
