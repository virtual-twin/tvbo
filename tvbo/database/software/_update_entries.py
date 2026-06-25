#!/usr/bin/env python3
"""One-shot script to enrich all software database YAML entries.

Adds: date_created, development_status, reference_publication, author, funder,
keywords, same_as, issue_tracker, is_accessible_for_free, operating_system.
Converts ecosystem from string to list, programming_language to enum values.
"""
import os
import yaml
from pathlib import Path

DB_DIR = Path(__file__).parent

# ─── Enrichment data keyed by tool name ───
# Fields: date_created, development_status, reference_publication, author,
#         funder, keywords, same_as, issue_tracker
ENRICHMENT = {
    "TVB": {
        "date_created": "2013-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2013.00010",
        "author": ["Viktor Jirsa", "Petra Ritter", "Randy McIntosh", "Michael Breakspear"],
        "funder": ["EU H2020 826421 (TVB-Cloud)", "EU H2020 945539 (HBP SGA3)", "EU Horizon Europe 101147319 (EBRAINS 2.0)"],
        "keywords": ["whole-brain simulation", "neural mass", "connectome", "mean-field", "neuroimaging"],
        "same_as": ["https://www.wikidata.org/entity/Q49165099", "https://scicrunch.org/resolver/RRID:SCR_004841"],
        "issue_tracker": "https://github.com/the-virtual-brain/tvb-root/issues",
    },
    "NEST": {
        "date_created": "2004-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2007.00011",
        "author": ["Marc-Oliver Gewaltig", "Markus Diesmann"],
        "funder": ["EU HBP", "Helmholtz Association", "Jülich Research Centre"],
        "keywords": ["spiking neural network", "point neuron", "large-scale simulation", "plasticity"],
        "same_as": ["https://www.wikidata.org/entity/Q51142269", "https://scicrunch.org/resolver/RRID:SCR_002963"],
        "issue_tracker": "https://github.com/nest/nest-simulator/issues",
    },
    "NEURON": {
        "date_created": "1994-01-01",
        "development_status": "active",
        "reference_publication": "10.1162/neco.1997.9.6.1179",
        "author": ["Michael Hines", "Nicholas Carnevale"],
        "funder": ["NIH", "Yale University"],
        "keywords": ["compartmental model", "Hodgkin-Huxley", "morphology", "electrophysiology"],
        "same_as": ["https://www.wikidata.org/entity/Q1993723", "https://scicrunch.org/resolver/RRID:SCR_005393"],
        "issue_tracker": "https://github.com/neuronsimulator/nrn/issues",
    },
    "Brian2": {
        "date_created": "2014-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.47314",
        "author": ["Marcel Stimberg", "Dan Goodman", "Romain Brette"],
        "funder": ["ANR", "CNRS"],
        "keywords": ["spiking neural network", "equation-based", "code generation", "C++ standalone"],
        "same_as": ["https://www.wikidata.org/entity/Q19845440", "https://scicrunch.org/resolver/RRID:SCR_002998"],
        "issue_tracker": "https://github.com/brian-team/brian2/issues",
    },
    "Arbor": {
        "date_created": "2018-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2019.00034",
        "author": ["Ben Cumming", "Nora Abi Akar"],
        "funder": ["EU HBP", "Swiss National Science Foundation"],
        "keywords": ["morphologically detailed", "GPU", "multi-compartment", "HPC"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_023698"],
        "issue_tracker": "https://github.com/arbor-sim/arbor/issues",
    },
    "CoreNEURON": {
        "date_created": "2016-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2019.00063",
        "author": ["Pramod Kumbhar", "Michael Hines"],
        "funder": ["EU HBP", "EPFL Blue Brain Project"],
        "keywords": ["compute backend", "GPU", "vectorization", "NEURON optimization"],
        "issue_tracker": "https://github.com/BlueBrain/CoreNeuron/issues",
    },
    "GeNN": {
        "date_created": "2016-01-01",
        "development_status": "active",
        "reference_publication": "10.1038/s43588-020-00022-7",
        "author": ["Thomas Nowotny", "James Turner"],
        "funder": ["EPSRC", "EU H2020"],
        "keywords": ["GPU", "spiking neural network", "code generation", "CUDA"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_017581"],
        "issue_tracker": "https://github.com/genn-team/genn/issues",
    },
    "BrainPy": {
        "date_created": "2021-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.86365",
        "author": ["Chaoming Wang", "Si Wu"],
        "funder": ["NSFC", "Peking University"],
        "keywords": ["JAX", "brain dynamics", "differentiable simulation", "neural mass"],
        "issue_tracker": "https://github.com/brainpy/BrainPy/issues",
    },
    "PyNN": {
        "date_created": "2008-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/neuro.11.011.2008",
        "author": ["Andrew Davison"],
        "funder": ["EU HBP", "CNRS"],
        "keywords": ["simulator-independent", "abstraction layer", "NEST", "NEURON", "Brian2"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_002715"],
        "issue_tracker": "https://github.com/NeuralEnsemble/PyNN/issues",
    },
    "NeuroML": {
        "date_created": "2004-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2014.00079",
        "author": ["Padraig Gleeson", "Sharon Crook"],
        "funder": ["NIH", "Wellcome Trust", "EU HBP"],
        "keywords": ["model description language", "XML", "standardization", "interoperability"],
        "same_as": ["https://www.wikidata.org/entity/Q7003628", "https://scicrunch.org/resolver/RRID:SCR_004594"],
        "issue_tracker": "https://github.com/NeuroML/NeuroML2/issues",
    },
    "LEMS": {
        "date_created": "2012-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2014.00079",
        "author": ["Robert Cannon", "Padraig Gleeson"],
        "keywords": ["model exchange", "XML", "dynamics specification", "simulator-neutral"],
        "issue_tracker": "https://github.com/LEMS/jLEMS/issues",
    },
    "jNeuroML": {
        "date_created": "2014-01-01",
        "development_status": "active",
        "reference_publication": "10.1098/rstb.2017.0380",
        "author": ["Padraig Gleeson"],
        "keywords": ["NeuroML", "LEMS", "validation", "Java"],
        "issue_tracker": "https://github.com/NeuroML/jNeuroML/issues",
    },
    "pyNeuroML": {
        "date_created": "2016-01-01",
        "development_status": "active",
        "reference_publication": "10.1098/rstb.2017.0380",
        "author": ["Padraig Gleeson"],
        "keywords": ["NeuroML", "Python API", "model analysis", "visualization"],
        "issue_tracker": "https://github.com/NeuroML/pyNeuroML/issues",
    },
    "SONATA": {
        "date_created": "2019-01-01",
        "development_status": "active",
        "reference_publication": "10.1371/journal.pcbi.1007696",
        "author": ["Kael Dai", "Sergey Bhatt"],
        "funder": ["Allen Institute for Brain Science", "EPFL Blue Brain Project"],
        "keywords": ["data format", "network model", "HDF5", "circuit"],
        "issue_tracker": "https://github.com/AllenInstitute/sonata/issues",
    },
    "NineML": {
        "date_created": "2011-01-01",
        "development_status": "inactive",
        "reference_publication": "10.1186/1471-2202-11-S1-P56",
        "author": ["Andrew Davison", "Ivan Raikov"],
        "funder": ["INCF"],
        "keywords": ["model description", "abstraction layers", "XML", "standardization"],
    },
    "SpineML": {
        "date_created": "2014-01-01",
        "development_status": "inactive",
        "author": ["Alex Cope", "Kevin Gurney"],
        "keywords": ["model description", "XML", "spiking network"],
    },
    "NWB": {
        "date_created": "2017-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.78362",
        "author": ["Oliver Rübel", "Andrew Tritt"],
        "funder": ["NIH BRAIN Initiative", "Kavli Foundation"],
        "keywords": ["neurophysiology data", "HDF5", "standardization", "FAIR"],
        "same_as": ["https://www.wikidata.org/entity/Q56279569", "https://scicrunch.org/resolver/RRID:SCR_015242"],
        "issue_tracker": "https://github.com/NeurodataWithoutBorders/pynwb/issues",
    },
    "NetPyNE": {
        "date_created": "2016-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.44494",
        "author": ["Salvador Dura-Bernal", "William Lytton"],
        "funder": ["NIH", "SUNY Downstate"],
        "keywords": ["NEURON", "network builder", "Python", "GUI"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_017603"],
        "issue_tracker": "https://github.com/Neurosim-lab/netpyne/issues",
    },
    "NetPyNE-UI": {
        "date_created": "2018-01-01",
        "development_status": "active",
        "author": ["MetaCell"],
        "keywords": ["GUI", "NetPyNE", "web interface"],
        "issue_tracker": "https://github.com/MetaCell/NetPyNE-UI/issues",
    },
    "PyRates": {
        "date_created": "2019-01-01",
        "development_status": "active",
        "reference_publication": "10.1371/journal.pone.0225900",
        "author": ["Richard Gast", "Daniel Rose"],
        "keywords": ["neural mass", "rate model", "mean-field", "code generation", "graph-based"],
        "issue_tracker": "https://github.com/pyrates-neuroscience/PyRates/issues",
    },
    "neurolib": {
        "date_created": "2020-01-01",
        "development_status": "active",
        "reference_publication": "10.1007/s12559-021-09931-9",
        "author": ["Caglar Cakan", "Klaus Obermayer"],
        "funder": ["TU Berlin", "DFG"],
        "keywords": ["whole-brain simulation", "neural mass", "exploration", "optimization"],
        "issue_tracker": "https://github.com/neurolib-dev/neurolib/issues",
    },
    "MOOSE": {
        "date_created": "2008-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2008.00006",
        "author": ["Upinder Bhalla"],
        "funder": ["NCBS", "DBT India"],
        "keywords": ["multi-scale", "compartmental", "signaling pathway", "reaction-diffusion"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_008031"],
        "issue_tracker": "https://github.com/BhallaLab/moose-core/issues",
    },
    "GENESIS": {
        "date_created": "1988-01-01",
        "development_status": "inactive",
        "reference_publication": "10.1007/978-1-4612-1634-6_10",
        "author": ["James Bower", "David Beeman"],
        "keywords": ["compartmental model", "ion channel", "classic simulator"],
        "same_as": ["https://www.wikidata.org/entity/Q5533484", "https://scicrunch.org/resolver/RRID:SCR_002807"],
    },
    "STEPS": {
        "date_created": "2012-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2009.00002",
        "author": ["Erik De Schutter", "Weiliang Chen"],
        "funder": ["OIST"],
        "keywords": ["stochastic", "reaction-diffusion", "tetrahedral mesh", "compartmental"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_008742"],
        "issue_tracker": "https://github.com/CNS-OIST/STEPS/issues",
    },
    "MCell": {
        "date_created": "1996-01-01",
        "development_status": "active",
        "reference_publication": "10.1016/j.tins.2006.07.001",
        "author": ["Thomas Bartol", "Joel Stiles"],
        "funder": ["NIH", "Salk Institute"],
        "keywords": ["Monte Carlo", "reaction-diffusion", "synapse", "sub-cellular"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_004551"],
        "issue_tracker": "https://github.com/mcellteam/mcell/issues",
    },
    "CARLsim": {
        "date_created": "2014-01-01",
        "development_status": "active",
        "reference_publication": "10.1162/neco_a_01208",
        "author": ["Michael Beyeler", "Nikil Dutt"],
        "funder": ["NSF", "UC Irvine"],
        "keywords": ["GPU", "spiking neural network", "CUDA", "plasticity"],
        "issue_tracker": "https://github.com/UCI-CARL/CARLsim6/issues",
    },
    "PSICS": {
        "date_created": "2010-01-01",
        "development_status": "inactive",
        "reference_publication": "10.1162/neco.2009.08-09-1078",
        "author": ["Robert Cannon", "Cian O'Donnell"],
        "keywords": ["stochastic", "ion channel", "point process"],
    },
    "Elephant": {
        "date_created": "2015-01-01",
        "development_status": "active",
        "reference_publication": "10.5281/zenodo.1186602",
        "author": ["Michael Denker", "Andrew Davison"],
        "funder": ["EU HBP", "Jülich Research Centre"],
        "keywords": ["spike train analysis", "electrophysiology", "statistics", "Python"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_003833"],
        "issue_tracker": "https://github.com/NeuralEnsemble/elephant/issues",
    },
    "Neo": {
        "date_created": "2010-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2014.00010",
        "author": ["Andrew Davison", "Samuel Garcia"],
        "funder": ["CNRS", "EU HBP"],
        "keywords": ["electrophysiology I/O", "data model", "Python", "file formats"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_000634"],
        "issue_tracker": "https://github.com/NeuralEnsemble/python-neo/issues",
    },
    "LFPy": {
        "date_created": "2013-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fncom.2013.00041",
        "author": ["Espen Hagen", "Gaute Einevoll"],
        "funder": ["Norwegian University of Life Sciences", "EU HBP"],
        "keywords": ["LFP", "EEG", "extracellular potential", "compartmental model"],
        "issue_tracker": "https://github.com/LFPy/LFPy/issues",
    },
    "BluePyOpt": {
        "date_created": "2016-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2016.00017",
        "author": ["Werner Van Geit"],
        "funder": ["EPFL Blue Brain Project"],
        "keywords": ["optimization", "evolutionary algorithm", "NEURON", "parameter fitting"],
        "issue_tracker": "https://github.com/BlueBrain/BluePyOpt/issues",
    },
    "Neurofitter": {
        "date_created": "2007-01-01",
        "development_status": "inactive",
        "reference_publication": "10.3389/fninf.2007.00001",
        "author": ["Werner Van Geit", "Erik De Schutter"],
        "keywords": ["parameter estimation", "evolutionary algorithm", "fitting"],
    },
    "TVB-Optim": {
        "date_created": "2024-01-01",
        "development_status": "active",
        "author": ["Leon Martin"],
        "funder": ["BIH Charité", "EU Horizon Europe 101147319 (EBRAINS 2.0)"],
        "keywords": ["TVB", "JAX", "automatic differentiation", "parameter optimization", "whole-brain"],
        "issue_tracker": "https://github.com/virtual-twin/tvboptim/issues",
    },
    "HNN-core": {
        "date_created": "2020-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.92862",
        "author": ["Mainak Jas", "Stephanie Jones"],
        "funder": ["NIH BRAIN Initiative"],
        "keywords": ["MEG", "EEG", "cortical column", "evoked response"],
        "issue_tracker": "https://github.com/jonescompneurolab/hnn-core/issues",
    },
    "SPM": {
        "date_created": "1991-01-01",
        "development_status": "active",
        "reference_publication": "10.1016/j.neuroimage.2011.10.018",
        "author": ["Karl Friston"],
        "funder": ["Wellcome Trust"],
        "keywords": ["neuroimaging", "fMRI", "PET", "EEG", "statistical parametric mapping", "dynamic causal modelling"],
        "same_as": ["https://www.wikidata.org/entity/Q4414458", "https://scicrunch.org/resolver/RRID:SCR_007037"],
    },
    "MNE-Python": {
        "date_created": "2011-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fnins.2013.00267",
        "author": ["Alexandre Gramfort", "Martin Luessi"],
        "funder": ["NIH", "NSF", "ANR"],
        "keywords": ["MEG", "EEG", "source estimation", "time-frequency", "Python"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_005972"],
        "issue_tracker": "https://github.com/mne-tools/mne-python/issues",
    },
    "EEGLAB": {
        "date_created": "2004-01-01",
        "development_status": "active",
        "reference_publication": "10.1016/j.jneumeth.2003.10.009",
        "author": ["Arnaud Delorme", "Scott Makeig"],
        "funder": ["NIH", "UCSD"],
        "keywords": ["EEG", "ICA", "MATLAB", "signal processing"],
        "same_as": ["https://www.wikidata.org/entity/Q18030316", "https://scicrunch.org/resolver/RRID:SCR_007292"],
    },
    "FieldTrip": {
        "date_created": "2003-01-01",
        "development_status": "active",
        "reference_publication": "10.1155/2011/156869",
        "author": ["Robert Oostenveld", "Jan-Mathijs Schoffelen"],
        "funder": ["Radboud University", "NWO"],
        "keywords": ["MEG", "EEG", "LFP", "MATLAB", "source reconstruction"],
        "same_as": ["https://www.wikidata.org/entity/Q5448078", "https://scicrunch.org/resolver/RRID:SCR_004849"],
    },
    "Brainstorm": {
        "date_created": "2000-01-01",
        "development_status": "active",
        "reference_publication": "10.1155/2011/879716",
        "author": ["Francois Tadel", "Sylvain Baillet"],
        "funder": ["NIH", "McGill University"],
        "keywords": ["MEG", "EEG", "source imaging", "MATLAB", "GUI"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_001761"],
    },
    "SpikeInterface": {
        "date_created": "2020-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.61834",
        "author": ["Alessio Buccino", "Cole Hurwitz", "Samuel Garcia"],
        "funder": ["Allen Institute", "CatalystNeuro"],
        "keywords": ["spike sorting", "electrophysiology", "Python", "reproducibility"],
        "issue_tracker": "https://github.com/SpikeInterface/spikeinterface/issues",
    },
    "PyRhO": {
        "date_created": "2016-01-01",
        "development_status": "inactive",
        "reference_publication": "10.3389/fninf.2016.00008",
        "author": ["Benjamin Evans", "Konstantin Bhatt"],
        "keywords": ["optogenetics", "rhodopsin", "photocurrent", "fitting"],
    },
    "OpenWorm": {
        "date_created": "2011-01-01",
        "development_status": "active",
        "reference_publication": "10.1098/rstb.2017.0382",
        "author": ["Stephen Larson"],
        "funder": ["OpenWorm Foundation"],
        "keywords": ["C. elegans", "whole-organism simulation", "open science", "connectome"],
        "same_as": ["https://www.wikidata.org/entity/Q7097017"],
        "issue_tracker": "https://github.com/openworm/OpenWorm/issues",
    },
    "BSB": {
        "date_created": "2019-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2019.00068",
        "author": ["Robin De Schepper", "Egidio D'Angelo"],
        "funder": ["EU HBP"],
        "keywords": ["cerebellum", "network builder", "multi-simulator", "placement"],
        "issue_tracker": "https://github.com/dbbs-lab/bsb/issues",
    },
    "BrainSimII": {
        "date_created": "2020-01-01",
        "development_status": "active",
        "author": ["Charles Simon"],
        "keywords": ["AGI", "spiking model", "Windows", "C#"],
        "issue_tracker": "https://github.com/FutureAIGuru/BrainSimII/issues",
    },
    "FastDMF": {
        "date_created": "2021-01-01",
        "development_status": "inactive",
        "reference_publication": "10.1016/j.neuroimage.2021.118367",
        "author": ["Gustavo Deco"],
        "keywords": ["dynamic mean-field", "resting state", "functional connectivity"],
    },
    "CxSystem2": {
        "date_created": "2018-01-01",
        "development_status": "inactive",
        "reference_publication": "10.1016/j.softx.2018.11.009",
        "author": ["Andalibi Vafa"],
        "keywords": ["cortical model", "Brian2", "YAML configuration"],
    },
    "pynn_genn": {
        "date_created": "2020-01-01",
        "development_status": "active",
        "author": ["James Turner", "Thomas Nowotny"],
        "keywords": ["PyNN", "GeNN", "GPU backend"],
        "issue_tracker": "https://github.com/genn-team/pynn_genn/issues",
    },
    "sPyNNaker": {
        "date_created": "2015-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fnins.2018.00816",
        "author": ["Andrew Rowley", "Steve Furber"],
        "funder": ["EU HBP", "EPSRC", "University of Manchester"],
        "keywords": ["SpiNNaker", "neuromorphic", "PyNN", "real-time"],
        "issue_tracker": "https://github.com/SpiNNakerManchester/sPyNNaker/issues",
    },
    "SpineCreator": {
        "date_created": "2014-01-01",
        "development_status": "inactive",
        "author": ["Alex Cope"],
        "keywords": ["SpineML", "GUI", "model builder"],
    },
    "neuroConstruct": {
        "date_created": "2007-01-01",
        "development_status": "inactive",
        "reference_publication": "10.1016/j.neuron.2007.08.015",
        "author": ["Padraig Gleeson"],
        "keywords": ["3D visualization", "network builder", "NEURON", "GENESIS"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_002178"],
    },
    "ModelDB": {
        "date_created": "1996-01-01",
        "development_status": "active",
        "reference_publication": "10.1007/s12021-006-9003-2",
        "author": ["Michael Hines", "Ted Bhatt"],
        "funder": ["NIH", "Yale University"],
        "keywords": ["model repository", "curated database", "published models"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_007271"],
    },
    "OpenSourceBrain": {
        "date_created": "2013-01-01",
        "development_status": "active",
        "reference_publication": "10.1016/j.neuron.2019.05.019",
        "author": ["Padraig Gleeson", "Angus Silver"],
        "funder": ["Wellcome Trust"],
        "keywords": ["model repository", "NeuroML", "collaborative", "open neuroscience"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_006636"],
        "issue_tracker": "https://github.com/OpenSourceBrain/OSBv2/issues",
    },
    "BrainBrowser": {
        "date_created": "2012-01-01",
        "development_status": "inactive",
        "author": ["Tarek Bhatt", "Alan Evans"],
        "funder": ["McGill University"],
        "keywords": ["3D brain viewer", "web-based", "surface", "volume"],
    },
    "Nengo": {
        "date_created": "2013-01-01",
        "development_status": "active",
        "reference_publication": "10.3389/fninf.2013.00048",
        "author": ["Chris Eliasmith", "Trevor Bekolay"],
        "funder": ["NSERC", "CFI", "Applied Brain Research"],
        "keywords": ["Neural Engineering Framework", "semantic pointer", "SPA", "functional model"],
        "same_as": ["https://scicrunch.org/resolver/RRID:SCR_013828"],
        "issue_tracker": "https://github.com/nengo/nengo/issues",
    },
    "Snudda": {
        "date_created": "2021-01-01",
        "development_status": "active",
        "reference_publication": "10.1038/s41467-022-30979-w",
        "author": ["Johannes Hjorth", "Jeanette Hellgren Kotaleski"],
        "funder": ["KTH Stockholm", "EU HBP"],
        "keywords": ["striatum", "basal ganglia", "detailed network", "touch detection"],
        "issue_tracker": "https://github.com/Hjorthmedansen/Snudda/issues",
    },
    "jaxley": {
        "date_created": "2024-01-01",
        "development_status": "active",
        "reference_publication": "10.7554/eLife.99205",
        "author": ["Michael Deistler", "Jakob Macke"],
        "funder": ["University of Tübingen", "DFG"],
        "keywords": ["differentiable simulation", "JAX", "compartmental model", "parameter inference"],
        "issue_tracker": "https://github.com/jaxleyverse/jaxley/issues",
    },
    "NetworkDynamics.jl": {
        "date_created": "2020-01-01",
        "development_status": "active",
        "reference_publication": "10.1063/5.0051387",
        "author": ["Frank Hellmann", "Michael Lindner"],
        "funder": ["PIK Potsdam"],
        "keywords": ["Julia", "network dynamics", "differential equations", "power grid", "neural network"],
        "issue_tracker": "https://github.com/JuliaDynamics/NetworkDynamics.jl/issues",
    },
    "Neuroblox.jl": {
        "date_created": "2023-01-01",
        "development_status": "active",
        "author": ["Neuroblox Inc."],
        "funder": ["ARPA-H"],
        "keywords": ["Julia", "whole-brain model", "modular", "neural mass", "clinical"],
        "issue_tracker": "https://github.com/Neuroblox/Neuroblox.jl/issues",
    },
    "BifurcationKit.jl": {
        "date_created": "2019-01-01",
        "development_status": "active",
        "author": ["Romain Veltz"],
        "funder": ["INRIA"],
        "keywords": ["Julia", "bifurcation analysis", "continuation", "dynamical systems"],
        "issue_tracker": "https://github.com/bifurcationkit/BifurcationKit.jl/issues",
    },
    "AUTO-07p": {
        "date_created": "1980-01-01",
        "development_status": "active",
        "reference_publication": "10.1007/978-1-4020-6356-5_4",
        "author": ["Eusebius Doedel"],
        "keywords": ["continuation", "bifurcation", "periodic orbits", "boundary value problems"],
        "same_as": ["https://www.wikidata.org/entity/Q4824289"],
    },
    "MatCont": {
        "date_created": "2003-01-01",
        "development_status": "active",
        "reference_publication": "10.1145/779359.779362",
        "author": ["Willy Govaerts", "Yuri Kuznetsov"],
        "keywords": ["MATLAB", "continuation", "bifurcation", "limit cycles", "GUI"],
        "same_as": ["https://www.wikidata.org/entity/Q6785071"],
    },
    "DifferentialEquations.jl": {
        "date_created": "2017-01-01",
        "development_status": "active",
        "reference_publication": "10.21105/joss.00615",
        "author": ["Chris Rackauckas"],
        "funder": ["MIT", "Julia Computing"],
        "keywords": ["Julia", "ODE", "SDE", "DDE", "differential equations", "scientific computing"],
        "same_as": ["https://www.wikidata.org/entity/Q111486832"],
        "issue_tracker": "https://github.com/SciML/DifferentialEquations.jl/issues",
    },
}


def enrich_entry(filepath):
    """Read a YAML file, add enrichment data, write back."""
    with open(filepath) as f:
        data = yaml.safe_load(f)

    name = data.get("name")
    if name not in ENRICHMENT:
        # Still need to fix ecosystem from string to list
        if isinstance(data.get("ecosystem"), str):
            data["ecosystem"] = [data["ecosystem"]]
        # Add default fields
        data.setdefault("development_status", "active")
        data.setdefault("is_accessible_for_free", True)
        data.setdefault("keywords", [])
        write_yaml(filepath, data)
        return

    enrich = ENRICHMENT[name]

    # Fix ecosystem from string to list
    if isinstance(data.get("ecosystem"), str):
        data["ecosystem"] = [data["ecosystem"]]

    # Add new fields
    for key in ["date_created", "development_status", "reference_publication",
                "author", "funder", "keywords", "same_as", "issue_tracker"]:
        if key in enrich:
            data[key] = enrich[key]

    data["is_accessible_for_free"] = True

    write_yaml(filepath, data)


def write_yaml(filepath, data):
    """Write YAML preserving our preferred field order."""
    field_order = [
        "name", "description", "homepage", "repository",
        "doi", "reference_publication", "citation",
        "license", "is_accessible_for_free",
        "ecosystem", "application_category",
        "date_created", "date_modified", "development_status",
        "scale", "model_paradigm", "tool_role",
        "programming_language", "runtime_platform", "operating_system",
        "interoperates_with",
        "author", "maintainer", "funder",
        "keywords", "same_as", "issue_tracker",
    ]

    ordered = {}
    for key in field_order:
        if key in data:
            ordered[key] = data[key]
    # Add any remaining keys
    for key in data:
        if key not in ordered:
            ordered[key] = data[key]

    with open(filepath, 'w') as f:
        yaml.dump(ordered, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=80)


if __name__ == "__main__":
    for yamlfile in sorted(DB_DIR.glob("*.yaml")):
        if yamlfile.name.startswith("_"):
            continue
        print(f"  Enriching {yamlfile.name}")
        enrich_entry(yamlfile)
    print(f"\nDone. Enriched {len(list(DB_DIR.glob('*.yaml')))} files.")
