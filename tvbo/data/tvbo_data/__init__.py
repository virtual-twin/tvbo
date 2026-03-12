import glob
import importlib
from os.path import basename, dirname, isfile, join

ROOT = dirname(__file__)
ATLAS_DIR = join(ROOT, "atlas")
CONNECTOME_DIR = join(ROOT, "_archive_connectome")  # Legacy CSV data (archived)
ASSIGNMENTS_DIR = join(ROOT, "assignments")

from . import atlases, connectomes, bids_utils
