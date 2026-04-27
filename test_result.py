#!/usr/bin/env python
"""Quick test: run EI tuning experiment and print result structure."""

import traceback
from tvbo import SimulationExperiment

exp = SimulationExperiment.from_db("EI_Tuning_FIC_EIB_Optimization")
try:
    result = exp.run("tvboptim")
    print("=== SUCCESS ===")
    print("TYPE:", type(result).__name__)
    print(result)
    print()
    print("integration:", type(result.integration).__name__ if result.integration else None)
    if result.integration:
        print("  data shape:", result.integration.data.shape if result.integration.data is not None else None)
    print("algorithms:", list(result.algorithms.keys()) if result.algorithms else [])
    print("optimizations:", list(result.optimizations.keys()) if result.optimizations else [])
except Exception:
    print("=== FAILED ===")
    traceback.print_exc()
