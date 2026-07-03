"""Locate the bundled TVBO data assets shipped inside this package.

Exposes filesystem paths to the packaged data directories, notably the atlas
and assignments folders, resolved relative to this module. Network and atlas
loading has moved to `tvbo.classes.network`, `tvbo.classes.atlas`, and
`tvbo.data.bids_utils`.
"""

import glob
import importlib
from os.path import basename, dirname, isfile, join

ROOT = dirname(__file__)
ATLAS_DIR = join(ROOT, "atlas")
ASSIGNMENTS_DIR = join(ROOT, "assignments")

# Modules moved to tvbo.classes.network, tvbo.classes.atlas, tvbo.data.bids_utils
