"""Locate the bundled TVBO data assets shipped inside this package.

Exposes filesystem paths to the packaged data directories, notably the atlas and assignments folders, resolved relative to this module. Network and atlas loading has moved to `tvbo.classes.network`, `tvbo.classes.atlas`, and `tvbo.data.bids_utils`.
"""

import glob
import importlib
from os.path import basename, dirname, isfile, join

ROOT = dirname(__file__)
# Atlases are consolidated under tvbo/database/atlases (the single source shared with networks, models, coordinate_spaces, …). ROOT is tvbo/data/tvbo_data, so the package root tvbo/ is two levels up. The legacy tvbo/data/tvbo_data/atlas copy is deprecated.
_PACKAGE_ROOT = dirname(dirname(ROOT))
ATLAS_DIR = join(_PACKAGE_ROOT, "database", "atlases")
ASSIGNMENTS_DIR = join(ROOT, "assignments")

# Modules moved to tvbo.classes.network, tvbo.classes.atlas, tvbo.data.bids_utils
