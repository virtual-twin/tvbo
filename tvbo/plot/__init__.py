# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""TVB-O's plotting utilities.

This module contains utilities for plotting TVB-O models.


```python
from tvbo import plot

plot.ontology.plot_model('JansenRit')
```
"""

from . import (
    animate,
    dynamics,
    dynamics_layout,
    experiment_layout,
    layout_mosaic,
    network,
    ontology,
    phase,
    timeseries,
)
