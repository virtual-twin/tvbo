#!/usr/bin/env python3
"""Check if colliding file pairs have identical data."""

import h5py
import numpy as np
from tvbo import database_path

pairs = [
    (
        "space-FSLMNI152_atlas-Schaefer100017Networks_desc-MghUscHcp32",
        "space-FSLMNI152_tpl-MghUscHcp32_atlas-Schaefer1000_desc-17Networks",
    ),
    (
        "space-MNI152Nlin2009c_atlas-DesikanKillianyranked_desc-MghUscHcp32",
        "space-MNI152Nlin2009c_tpl-MghUscHcp32_atlas-DesikanKilliany_desc-ranked",
    ),
    (
        "space-MNI152Nlin2009c_atlas-Destrieuxranked_desc-MghUscHcp32",
        "space-MNI152Nlin2009c_tpl-MghUscHcp32_atlas-Destrieux_desc-ranked",
    ),
]

NET_DIR = database_path / "networks"

for a, b in pairs:
    fa = str(NET_DIR / (a + ".h5"))
    fb = str(NET_DIR / (b + ".h5"))
    with h5py.File(fa) as ha, h5py.File(fb) as hb:
        datasets_a = {}
        datasets_b = {}

        def visit_a(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets_a[name] = obj[:]

        def visit_b(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets_b[name] = obj[:]

        ha.visititems(visit_a)
        hb.visititems(visit_b)
        print(a)
        print(b)
        print("  keys A:", sorted(datasets_a.keys()))
        print("  keys B:", sorted(datasets_b.keys()))
        for k in sorted(set(datasets_a.keys()) & set(datasets_b.keys())):
            same = np.array_equal(datasets_a[k], datasets_b[k])
            print("  %s: same=%s, shapes=%s vs %s" % (k, same, datasets_a[k].shape, datasets_b[k].shape))
        print()
