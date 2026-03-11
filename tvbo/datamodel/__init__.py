"""
TVB-O Data Model
================
This module contains the data model for TVB-O.

```{seealso}
https://bss.git-pages.bihealth.org/tvb-o/tvbo-datamodel
```
"""

from tvbo.datamodel.tvbo_datamodel import Network  # noqa: E402

# number_of_regions is a deprecated alias for number_of_nodes.
# Defined here (not in the generated file) so it survives make gen-linkml.
Network.number_of_regions = property(
    lambda self: self.number_of_nodes,
    lambda self, v: setattr(self, 'number_of_nodes', v),
)

from .tvbo_datamodel import *
