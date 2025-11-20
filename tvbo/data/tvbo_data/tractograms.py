import subprocess
from os.path import basename, isfile

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram

from tvbo.data.tvbo_data.atlases import atlas_data
from tvbo.data.tvbo_data.connectomes import connectome_data


def convert_trk_to_tck(trk_file, tck_file, template_file, reset_affine=False):

    template = nib.load(template_file)
    print("loading tractogram...")
    dtor = load_tractogram(
        trk_file,
        template,
        trk_header_check=False,
        bbox_valid_check=False,
    )
    affine = template.affine.copy()
    if reset_affine:
        affine = np.array(
            [
                [
                    0,
                    0,
                    0,
                    0,
                ],  # No scaling, no rotation, and no translation for the x-axis
                [
                    0,
                    0,
                    0,
                    0,
                ],  # No scaling, no rotation, and no translation for the y-axis
                [
                    0,
                    0,
                    0,
                    0,
                ],  # No scaling, no rotation, and no translation for the z-axis
                [0, 0, 0, 1],  # Homogeneous coordinate for affine transformations
            ]
        )

        dtor.to_rasmm()

    print("saving tractogram...")
    # Create a new StatefulTractogram with the updated affine
    new_dtor = StatefulTractogram(
        dtor.streamlines,
        reference=template,  # This assumes the affine matrix is being used as the reference
        space=Space.RASMM,
        # origin=dtor.origin,  # Use the same origin as the original tractogram
        # data_per_point=dtor.data_per_point,
        # data_per_streamline=dtor.data_per_streamline,
    )

    save_tractogram(new_dtor, tck_file, bbox_valid_check=False)


def create_connectome(ftractogram, atlas, overwrite=False, out_dir=None):

    if not isfile(atlas):
        versions = atlas_data.get(atlas=atlas, extension="nii.gz", return_type="file")
        if len(versions) > 1:
            versions = atlas_data.get(
                atlas=atlas, extension="nii.gz", return_type="file", desc="ranked"
            )

        fatlas = versions[0]
    else:
        fatlas = atlas
        atlas = basename(fatlas).split(".")[0]

    entities = atlas_data.parse_file_entities(fatlas)
    tckname = atlas_data.parse_file_entities(ftractogram)["suffix"]

    entities_weights = entities.copy()
    entities_weights["suffix"] = "weights"
    entities_weights["desc"] = tckname
    entities_weights["extension"] = ".csv"
    fweights_out = connectome_data.build_path(
        entities_weights,
        path_patterns=["space-{space}_atlas-{atlas}_desc-{desc}_{suffix}{extension}"],
        validate=False,
    )
    entities_lengths = entities_weights.copy()
    entities_lengths["suffix"] = "lengths"
    flengths_out = connectome_data.build_path(
        entities_lengths,
        path_patterns=["space-{space}_atlas-{atlas}_desc-{desc}_{suffix}{extension}"],
        validate=False,
    )

    entities_assign = entities_weights.copy()
    entities_assign["suffix"] = "assignments"
    fassign_out = connectome_data.build_path(
        entities_assign,
        path_patterns=["space-{space}_atlas-{atlas}_desc-{desc}_{suffix}{extension}"],
        validate=False,
    )

    if out_dir is not None:
        fweights_out = fweights_out.replace(connectome_data.root, out_dir)
        flengths_out = flengths_out.replace(connectome_data.root, out_dir)
        fassign_out = fassign_out.replace(connectome_data.root, out_dir)

    if overwrite or (not isfile(fweights_out) or not isfile(fassign_out)):

        subprocess.run(
            [
                "tck2connectome",
                ftractogram,
                fatlas,
                fweights_out,
                "-out_assignments",
                fassign_out,
                "-symmetric",
                "-zero_diagonal",
                "-force",
            ]
        )
    else:
        print(f"Weights already exists for {atlas} and {tckname}.")

    if overwrite or not isfile(flengths_out):
        subprocess.run(
            [
                "tck2connectome",
                ftractogram,
                fatlas,
                flengths_out,
                "-scale_length",
                "-stat_edge",
                "mean",
                "-symmetric",
                "-zero_diagonal",
                "-force",
            ]
        )

    else:
        print(f"Lengths already exists for {atlas} and {tckname}.")
