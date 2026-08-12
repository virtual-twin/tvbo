#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
TVB-O's plotting utilities
==========================
This module contains utilities for plotting TVB-O models.


```python
from tvbo import plot

plot.ontology.plot_model('JansenRit')
```
"""

from . import (
    network,
    ontology,
    functions,
    timeseries,
    phase,
    animate,
    dynamics,
    layout_mosaic,
    dynamics_layout,
    experiment_layout,
)
