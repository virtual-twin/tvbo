#!/usr/bin/env python
"""Migrate all database networks from plural edge labels to singular.

Renames HDF5 groups and YAML sidecar edge labels:
  weights → weight lengths → length

Uses Network load/save round-trip: loads each network, renames internal arrays and template edge labels, then saves back to the same path.
"""

from pathlib import Path

from tvbo import database_path
from tvbo.data.network_io import load_network, save_network

NETWORK_DIR = database_path / "networks"

# Edge label renames: old → new
RENAMES = {
    "weights": "weight",
    "lengths": "length",
}


def migrate_network(sidecar: Path) -> bool:
    """Load a network, rename edge labels, and re-save. Returns True if changed."""
    net = load_network(sidecar)

    # Rename in _arrays dict
    arrays = getattr(net, "_arrays", None) or {}
    # Also pull from lazy store if arrays empty
    store = getattr(net, "_store", None)
    if store is not None:
        for old in RENAMES:
            if old not in arrays and old in store.arrays:
                arrays[old] = store.arrays[old]

    changed = False
    for old, new in RENAMES.items():
        if old in arrays and new not in arrays:
            arrays[new] = arrays.pop(old)
            changed = True

    # Rename template edge labels
    for e in net.edges or []:
        lbl = getattr(e, "label", None)
        if lbl in RENAMES:
            e.label = RENAMES[lbl]
            changed = True

    if not changed:
        return False

    # Ensure _arrays is set on the object
    object.__setattr__(net, "_arrays", arrays)

    # Re-save
    save_network(net, sidecar)
    return True


def main():
    sidecars = sorted(NETWORK_DIR.glob("*.yaml"))
    print(f"Found {len(sidecars)} network sidecars in {NETWORK_DIR}")

    migrated = 0
    for sc in sidecars:
        try:
            if migrate_network(sc):
                print(f"  migrated: {sc.name}")
                migrated += 1
            else:
                print(f"  skipped (already singular): {sc.name}")
        except Exception as exc:
            print(f"  ERROR: {sc.name}: {exc}")

    print(f"\nDone. Migrated {migrated}/{len(sidecars)} networks.")


if __name__ == "__main__":
    main()
