#  bids.py
#
# Created on Thu Nov 02 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#

"""
Brain Imaging Data Structure (BIDS) Export for TVB-O
====================================================

This module provides functionality to export data and metadata from TVB simulations into a format that is compliant
with the Brain Imaging Data Structure (BIDS) Extension for Computational Models (BEP034). BIDS is a standard for
organizing and describing neuroimaging and associated data that TVB-O supports to facilitate the interchange of data
within the research community.

The integration with BIDS enables users to structure their simulation output data and associated metadata in a way
that enhances interoperability and reusability. This can be particularly advantageous for users aiming to share their
simulation results with the wider neuroscience community, or for those who wish to integrate TVB-O outputs with other
BIDS-compliant datasets for combined analyses.

By adhering to the `BIDS Computational Model Specification (BEP034)
<https://docs.google.com/document/d/1NT1ERdL41oz3NibIFRyVQ2iR8xH-dKY-lRCB4eyVeRo/edit>`_,
TVB-O ensures consistency in data formatting, which is crucial for collaborative research, data sharing, and the
replicability of computational studies.

Features -------- - **BIDS Compatibility**: Seamlessly export simulation configurations, connectivity data,
and time series outputs in BIDS-compliant format. - **Metadata Organization**: Automatic generation of BIDS metadata
files to describe the simulations, including model specifications and integration details. - **Visualization
Support**: Functions to visualize the BIDS directory structure, aiding in the verification and presentation of the
export process.

Components
----------
The module comprises functions that handle various aspects of the BIDS export process:

- Conversion of time series data to BIDS-compliant time series files (TSV).
- Generation of model description files in JSON format that align with BIDS specifications.
- Organization of connectivity data in accordance with BIDS modality-specific extensions.
- Utility functions for directory listing and visualization, enhancing user interaction with the file system.

See Also
--------
For more information on BIDS and BEP034, visit the official BIDS website: https://bids.neuroimaging.io/
For comprehensive details on BEP034, refer to the `BIDS Computational Model Specification document
<https://docs.google.com/document/d/1NT1ERdL41oz3NibIFRyVQ2iR8xH-dKY-lRCB4eyVeRo/edit>`_.

"""

import importlib.metadata
import json
import os
from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import owlready2 as owl
import pandas as pd
from bids.layout.writing import build_path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tvbo.export import lemsgenerator
from tvbo.knowledge import ontology, config

bids_dir = "./bids_output"  # TODO: not used
subid = "test"  # TODO: not used
sesid = "1"  # TODO: not used

try:
    version = importlib.metadata.version("tvb-library")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

metadata = dict(
    SoftwareName="TVB",
    SoftwareDescription="The Virtual Brain",
    SoftwareVersion=importlib.metadata.version("tvb-framework"),
    SoftwareRepository=[
        importlib.metadata.metadata("tvb-framework")["Download-URL"],
    ],
    SourceCode=[
        importlib.metadata.metadata("tvb-framework")["Download-URL"],
    ],
    SourceCodeVersion=version,
)

# TODO: not used
kvpairs = dict(
    id="hashedJSON",
    net=["distances", "weights"],
    coord=["vertices", "faces", "normals", "meg", "eeg", "stim", "nodes"],
    ts=["sim", "emp", "stim", "art"],
    map=["lfm-eeg", "lfm-meg", "area", "volume", "fc-sim", "fc-emp"],
)

# TODO: not used
patterns = [
    # base pattern
    "sub-{subject}/ses-{session}/{modality}/sub-{subject}_ses-{session}_desc-{description}_{suffix}.{extension}",
    "sub-{subject}/{modality}/sub-{subject}_desc-{description}_{suffix}.{extension}",
    # modality-specific patterns
    ## Equations
    "sub-{subject}/ses-{session}/{modality}/sub-{subject}_ses-{session}_eq-{equation}_desc-{description}_{suffix}.{extension}",
    "sub-{subject}/{modality}/sub-{subject}_eq-{equation}_desc-{description}_{suffix}.{extension}",
    ## Networks
    "sub-{subject}/ses-{session}/{modality}/sub-{subject}_ses-{session}_nt-{network}_desc-{description}_{suffix}.{extension}",
    "sub-{subject}/{modality}/sub-{subject}_nt-{network}_desc-{description}_{suffix}.{extension}",
]


def generate_bids_path(**entities):
    """
    Generate a BIDS-compliant path for different modalities based on provided entities.

    Parameters
    ----------
    **entities : dict
        Arbitrary keyword arguments that represent BIDS entities
        (e.g., `subject`, `session`, `modality`, `suffix`, `extension`).

    Returns
    -------
    tuple of (str, dict)
        A tuple containing the BIDS-compliant file path and the entities used for generating it.

    Raises
    ------
    ValueError
        If any required entities are missing from the keyword arguments or no valid path
        could be built from the provided entities.

    Examples
    --------
    >>> generate_bids_path(subject='01', session='01', modality='anat', suffix='T1w', extension='nii.gz')
    ('sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz',
    {'subject': '01', 'session': '01', 'modality': 'anat', 'suffix': 'T1w', 'extension': 'nii.gz'})
    """
    # Your provided patterns
    patterns = [
        # Base pattern
        "sub-{subject}/ses-{session}/{modality}/sub-{subject}_ses-{session}_desc-{description}_{suffix}.{extension}",
        "sub-{subject}/{modality}/sub-{subject}_desc-{description}_{suffix}.{extension}",
        # Modality-specific patterns
        ## Equations
        "sub-{subject}/ses-{session}/{modality}/sub-{subject}_ses-{session}_eq-{equation}_desc-{description}_{suffix}.{extension}",
        "sub-{subject}/{modality}/sub-{subject}_eq-{equation}_desc-{description}_{suffix}.{extension}",
        ## Networks
        "sub-{subject}/ses-{session}/{modality}/sub-{subject}_ses-{session}_nt-{network}_desc-{description}_{suffix}.{extension}",
        "sub-{subject}/{modality}/sub-{subject}_nt-{network}_desc-{description}_{suffix}.{extension}",
    ]

    # Check if all required entities are included
    required_entities = {"subject", "modality", "description", "suffix", "extension"}
    missing_entities = required_entities - entities.keys()
    if missing_entities:
        raise ValueError(f"Missing required entities: {', '.join(missing_entities)}")

    # Filter patterns based on provided entities
    matching_patterns = [
        pattern
        for pattern in patterns
        if all(f"{{{entity}}}" in pattern for entity in entities)
    ]

    if not matching_patterns:
        raise ValueError("No pattern matches the provided entities")

    # Build the path using the first matching pattern
    path = build_path(entities, matching_patterns)
    if path is None:
        raise ValueError("No valid path found for the given entities and patterns")
    return path, entities


def generate_json_sidecar(dataset, metadata, **entities):
    """
    Generate a JSON sidecar file for a dataset with provided metadata and entities.

    Parameters
    ----------
    dataset : str
        The path to the dataset directory where the JSON sidecar will be created.
    metadata : dict
        A dictionary containing metadata to be written into the JSON sidecar file.
    **entities : dict
        Arbitrary keyword arguments that represent BIDS entities for file naming.

    Returns
    -------
    dict
        The metadata dictionary that was written to the JSON sidecar.

    Examples
    --------
    >>> generate_json_sidecar('path/to/dataset', {'Author': 'Leon K. Martin'}, subject='01', modality='anat')
    {'Author': 'Leon K. Martin'}
    """
    entities.update(dict(extension="json"))
    json_path, entities = generate_bids_path(**entities)
    with open(join(dataset, json_path), "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


def ts_metadata(model, param_config=None):
    """
    Generate metadata for time series data based on a computational model and optional parameter configuration.

    Parameters
    ----------
    model : str or owl.ThingClass
        The model identifier or an instance from the ontology to create metadata for.
    param_config : str or dict, optional
        The parameter configuration identifier or a dictionary of the parameter settings.

    Returns
    -------
    dict
        A dictionary containing the metadata for the time series based on the model.

    Raises
    ------
    ValueError
        If the provided model is not found in the ontology or is not a recognized type.

    Examples
    --------
    >>> ts_metadata('JansenRit')
    {'Model': {'ModelName': 'Jansen-Rit Model', 'ModelID': 'TVBO:001', ...}}
    """
    if isinstance(model, str):
        model = ontology.search_class("JansenRit")
        if model == []:
            raise ValueError("Model not found in ontology.")
    elif not isinstance(model, owl.ThingClass):
        raise ValueError("Model must be a string or owl.ThingClass.")

    model_name = model.label.first()

    if isinstance(param_config, str):
        param_setting = config.update_default(model, param_config)
    elif isinstance(param_config, dict):
        param_setting = config.get_default(model)

    nmm_metadata = dict(
        Model=dict(
            ModelName=model_name,
            ModelID=f"TVBO:{model.identifier.first()}",
            ModelIRI=model.iri,
            ModelType=[
                c.label.first()
                for c in model.is_a
                if not isinstance(c, owl.Restriction)
            ],
            Equations="file",
            # Network="file",
            # Code="file",
            Parameters=param_setting,
        ),
    )
    return nmm_metadata


def model2bids(dataset, model, subject="1", param_config=None, **entities):
    """
    Export a computational model in TVB format to BIDS-compliant structure.

    Parameters
    ----------
    dataset : str
        Path to the BIDS dataset directory where model files will be stored.
    model : str
        Name of the model to be exported.
    subject : str, optional
        The subject identifier.
    param_config : str or dict, optional
        The parameter configuration for the model to be exported.
    **entities : dict
        Additional BIDS entities for file naming.

    Returns
    -------
    str
        Path to the exported XML LEMS model file.

    Examples
    --------
    >>> model2bids('path/to/bids/dataset', 'JansenRit', subject='02', description='ExcitatoryInhibitory')
    'path/to/bids/dataset/sub-02/eq/sub-02_excitatoryinhibitory_lems.xml'
    """
    model = ontology.search_class(model)
    model_name = model.label.first()
    if isinstance(param_config, type(None)):
        param_config = config.get_default(model)

    xml_path, entities = generate_bids_path(
        subject=subject,
        description=model_name,
        modality="eq",
        suffix="lems",
        extension="xml",
        **entities,
    )
    os.makedirs(join(dataset, dirname(xml_path)), exist_ok=True)
    lemsgenerator.export_lems_model(
        "JansenRit",  # param_config=param_config, #TODO: Fix bug with k,v pairs
        fpath=join(dataset, xml_path),
    )

    model_metadata = ts_metadata(model, param_config=param_config)
    model_metadata["Model"]["Equations"] = xml_path

    metadata.update(model_metadata)

    generate_json_sidecar(
        dataset,
        metadata=metadata,
        **entities,
    )
    return xml_path


def connectivity2bids(
    dataset, connectivity, subject="1", compression="gzip", **entities
):
    """
    Export connectivity data to BIDS-compliant files.

    Parameters
    ----------
    dataset : str
        Path to the BIDS dataset directory where connectivity files will be stored.
    connectivity : Connectivity class instance
        The Connectivity instance containing data to be exported.
    subject : str, optional
        The subject identifier.
    compression : str, optional
        Compression method for the output files.
    **entities : dict
        Additional BIDS entities for file naming.

    Returns
    -------
    list of str
        Paths to the created BIDS-compliant connectivity files.

    Examples
    --------
    >>> connectivity2bids('path/to/bids/dataset', conn_instance, subject='02')
    ['path/to/bids/dataset/sub-02/net/sub-02_weights.tsv.gz', 'path/to/bids/dataset/sub-02/net/sub-02_tracts.tsv.gz']
    """
    entities.update(
        dict(subject=subject, modality="net", suffix="net", extension="tsv")
    )

    conn_metadata = dict(
        SubjectID=subject,
        Description="Structural Connectivity",
        VolumeUnit="mm^3",
        LengthUnit="mm",
        Nodes=dict(
            Count=len(connectivity.centres),
            ID=list(connectivity.region_labels),
            # Volume=[area for area in connectivity.areas if len(connectivity.areas) > 0],
            CenterX=list(connectivity.centres[:, 0]),
            CenterY=list(connectivity.centres[:, 1]),
            CenterZ=list(connectivity.centres[:, 2]),
        ),
    )

    paths = []
    for network, data in zip(
        ["weights", "distances"], [connectivity.weights, connectivity.tract_lengths]
    ):
        entities.update(dict(description="SC", network=network))
        xml_path, entities = generate_bids_path(
            **entities,
        )
        os.makedirs(join(dataset, dirname(xml_path)), exist_ok=True)
        pd.DataFrame(data).to_csv(
            join(dataset, xml_path), sep="\t", index=False, compression=compression
        )
        generate_json_sidecar(
            dataset,
            metadata=conn_metadata,
            **entities,
        )
        paths.append(xml_path)
    return paths


def timeseries2bids(
    dataset,
    sim,
    time,
    data,
    model_file,
    network_file,
    subject="1",
    compression="gzip",
    **entities,
):
    """
    Converts time series data from simulations to BIDS-compliant format.

    Parameters
    ----------
    dataset : str
        Path to the BIDS dataset directory where the time series files will be stored.
    sim : Simulator object
        The simulator object that contains the model and connectivity information.
    time : array_like
        The time points for which the data is available.
    data : ndarray
        The time series data to be exported.
    model_file : str
        The filename where the model description is stored.
    network_file : str
        The filename where the network description is stored.
    subject : str, optional
        The subject identifier. Defaults to "1".
    compression : str, optional
        Compression method for storage. Defaults to "gzip".
    **entities : dict, optional
        Additional BIDS entities to be included.

    Raises
    ------
    ValueError
        If the number of variables in data does not match the number of variables of interest.

    See Also
    --------
    generate_bids_path : Helper function to generate BIDS-compliant paths.

    Notes
    -----
    This function assumes that the simulation data is organized with time as the last dimension.

    Examples
    --------
    >>> timeseries2bids('path/to/bids_dataset', sim_object, np.linspace(0, 1000, 1001), sim_data, 'model.json', 'network.json')
    """
    entities.update(
        dict(
            subject=subject,
            description="sim",
            modality="ts",
            suffix="ts",
            extension="tsv",
        )
    )
    vois = sim.model.variables_of_interest
    if not data.shape[1] == len(vois):
        raise ValueError(
            f"Data shape ({data.shape[1]}) does not match number of variables of interest ({len(vois)})."
        )

    for i, voi in enumerate(vois):
        entities.update(dict(description=f"simVOI{voi}"))
        xml_path, entities = generate_bids_path(
            **entities,
        )
        os.makedirs(join(dataset, dirname(xml_path)), exist_ok=True)

        voi_data = data[:, i].squeeze()
        pd.DataFrame(voi_data).to_csv(
            join(dataset, xml_path), sep="\t", index=False, compression=compression
        )

        ts_metadata = dict(
            SubjectID=subject,
            Description=dict(Name="Time Series", Type="Simulated"),
            Units=dict(Time="ms", Data="mV"),
            Timesteps=len(time),
            DataFiles=xml_path,
            Model=dict(
                Equations=model_file,
                Network=network_file,
                Integrator=dict(
                    Type=sim.integration.title.split(" ")[0],
                    Identifier=ontology.search_class(
                        sim.integration.title.split(" ")[0]
                    ).identifier.first(),
                    Timestep=sim.integration.dt,
                ),
            ),
        )

        generate_json_sidecar(
            dataset,
            metadata=ts_metadata,
            **entities,
        )


def save_sim2bids(sim, time, data, dataset, subject, model_file=None, **entities):
    """
    Save simulation outputs in a BIDS-compatible format.

    Parameters
    ----------
    sim : Simulator object
        The simulation object containing the model and connectivity data.
    time : ndarray
        A 1D array containing the time points at which the simulation was sampled.
    data : ndarray
        A 2D array (time points x variables) of data from the simulation.
    dataset : str
        The path to the BIDS dataset directory.
    subject : str
        The subject identifier to be used in the BIDS dataset.
    model_file : str, optional
        The path to the JSON file containing the model description. If not provided, it
        will be generated from the simulation object.
    **entities : dict, optional
        Additional BIDS entities to include in the filenames.

    See Also
    --------
    timeseries2bids : Function to convert time series data to BIDS format.
    connectivity2bids : Function to convert connectivity data to BIDS format.

    Notes
    -----
    This function acts as a high-level interface to convert and save all necessary
    components of a simulation into the BIDS format. It organizes metadata, models,
    and time series data according to BIDS specifications.
    """
    # sim.model.ontology
    network_paths = connectivity2bids(
        dataset, sim.network, subject=subject, **entities
    )
    timeseries2bids(
        dataset,
        sim,
        time,
        data,
        model_file=model_file,
        network_file=network_paths,
        subject="1",
        compression="gzip",
        **entities,
    )


############
# Plotting #
############
ROOT_DIR = realpath(dirname(__file__))

# You will need to download the emoji images and specify their paths here
folder_icon_path = join(ROOT_DIR, "data", "media", "file-folder.png")
file_icon_path = join(ROOT_DIR, "data", "media", "chart-increasing.png")
json_icon_path = join(ROOT_DIR, "data", "media", "page-facing-up.png")


def getImage(path):
    """
    Load and return an OffsetImage object from the given image file path.

    Parameters:
    path (str): The file path of the image.

    Returns:
    OffsetImage: An OffsetImage object representing the loaded image.
    """
    return OffsetImage(plt.imread(path), zoom=0.05)


def list_files(startpath, render_image=False):
    """
    Visualize the file hierarchy of a given directory with options to render images.

    Parameters
    ----------
    startpath : str
        The root directory from which the file hierarchy is generated.
    render_image : bool, optional
        If True, renders the file hierarchy as images using matplotlib. If False,
        prints the hierarchy to the console with emojis.

    Raises
    ------
    FileNotFoundError
        If the specified `startpath` does not exist.

    Notes
    -----
    When `render_image` is True, the function uses matplotlib to create a visual
    representation of the file structure with icons representing different file types.
    This is particularly useful for presentations or documentation purposes.
    """
    folder_emoji = "\U0001F4C1"  # Folder emoji
    file_emoji = "\U0001F4C8"  # File emoji
    json_emoji = "\U0001F4C4"  # JSON emoji

    tree = []
    if not os.path.exists(startpath):
        print(f"The directory {startpath} does not exist.")
        return

    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = (
            " " * 4 * level
        )  # you can adjust the number of spaces for indentation
        tree.append(("DIR", f"{indent}{os.path.basename(root)}/", level))
        if not render_image:
            print(f"{indent}{folder_emoji} {os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in sorted(files):
            if f.endswith(".json"):
                emoji = json_emoji
                tree.append(("JSON", f"{subindent}{f}", level + 1))
            else:
                emoji = file_emoji
                tree.append(("FILE", f"{subindent}{f}", level + 1))

            if not render_image:
                print(f"{subindent}{emoji} {f}")

    if render_image and tree:
        fig, ax = plt.subplots(figsize=(7, len(tree) * 0.3), dpi=100)
        for i, (type, line, level) in enumerate(tree):
            y = 1 - i * 0.05  # adjust spacing to your needs
            x = 0.02 * level  # Indentation step for each level; adjust as needed
            if type == "DIR":
                ab = AnnotationBbox(
                    getImage(folder_icon_path),
                    (x, y),
                    frameon=False,
                    box_alignment=(0, 0.5),
                )
                ax.add_artist(ab)
            elif type == "FILE":
                ab = AnnotationBbox(
                    getImage(file_icon_path),
                    (x, y),
                    frameon=False,
                    box_alignment=(0, 0.5),
                )
                ax.add_artist(ab)
            elif type == "JSON":
                ab = AnnotationBbox(
                    getImage(json_icon_path),
                    (x, y),
                    frameon=False,
                    box_alignment=(0, 0.5),
                )
                ax.add_artist(ab)
            ax.text(
                x + 0.03,
                y,
                line.strip(),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=10,
                family="monospace",
            )

        ax.axis("off")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        return fig
