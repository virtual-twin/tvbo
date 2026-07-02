"""Bundle the brain atlas parcellation data shipped with TVBO.

This package directory holds the packaged atlas segmentation lookup tables
(`*_dseg.yaml`) and region-center coordinate files (`*_centers.txt`) for the
parcellations TVBO ships — including Schaefer2018, Yeo17, HCP-MMP1,
Desikan-Killiany, and Destrieux. It contains no importable code; the empty
`__init__` marks the directory as a Python package so these assets can be
located as package data (see [`tvbo.data.tvbo_data`](../__init__.py) and its
`ATLAS_DIR` path).
"""
