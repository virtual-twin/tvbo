import glob
import importlib
from os.path import basename, dirname, isfile, join

ROOT = dirname(__file__)
ATLAS_DIR = join(ROOT, "atlas")
ASSIGNMENTS_DIR = join(ROOT, "assignments")

# Modules moved to tvbo.classes.network, tvbo.classes.atlas, tvbo.data.bids_utils
