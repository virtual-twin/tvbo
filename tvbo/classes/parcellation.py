#
# Module: parcellation.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
This module provides functions for loading and accessing parcellation data.

Author: Leon Martin
"""

from typing import Any
import requests
import nibabel as nib
from nibabel.loadsave import load as nib_load
import tempfile


def load_from_url(url: str) -> Any:
    """
    Load a NIFTI file from a given URL.

    Args:
        url (str): The URL of the NIFTI file.

    Returns:
        nibabel.nifti1.Nifti1Image: The loaded NIFTI image.

    Raises:
        Exception: If the download request fails or the file cannot be loaded.
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write content to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Load the NIFTI file
        # Using nibabel's load function (imported explicitly for clearer typing surface)
        img = nib_load(temp_file_path)
        return img
    else:
        raise Exception(f"Failed to load NIFTI file from URL: {url}")
