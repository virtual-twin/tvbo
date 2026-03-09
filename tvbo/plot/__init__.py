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


```{python}
from tvbo.plot import graph

graph.plot_model('JansenRit')
```
"""
from . import functions, network
