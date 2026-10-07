"""The public import location for ``Phenotype``.

A cohort's per-subject phenotype scores (cognitive, clinical, behavioral, demographic, physiological, derived) for multi-subject studies that correlate simulated quantities with empirical measurements. What a phenotype does — the h5 companion beside its YAML descriptor, one measure or one subject answered for — lives in :mod:`tvbo.behaviour.phenotype`, attached where the class is generated.
"""

from tvbo.datamodel import pydantic as tvbo_datamodel

Phenotype = tvbo_datamodel.Phenotype
