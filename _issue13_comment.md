## Fix: Wrong node type (#13)

**Root cause:** The `get_type()` function in `tvbo/knowledge/ontology.py` used `onto.integration` (a property/namespace reference) instead of `onto.IntegrationMethod` (the actual ontology class). When the ancestor check failed, the function fell through to the default `return onto.TheVirtualBrain` — which is why the Heun integrator showed type "TheVirtualBrain" instead of "IntegrationMethod".

### Changes

1. **`tvbo/knowledge/ontology.py`** — Fixed `onto.integration` → `onto.IntegrationMethod` in the `get_type()` function
2. **`schema/tvbo_datamodel.yaml`** — Added missing `class_uri` to `Integrator`, `Equation`, `Stimulus`, `Network`, `SimulationStudy`, `TimeSeries` for correct RDF/JSON-LD type serialization
3. **`tvbo/datamodel/tvbo_datamodel.py`** — Regenerated from updated schema

### Test

```python
from tvbo.knowledge import ontology
onto = ontology.onto

# Test integrators
for i in onto.IntegrationMethod.descendants(include_self=False):
    t = ontology.get_type(i)
    tname = t.label.first() if hasattr(t, 'label') and t.label.first() else t.name
    print(f'{i.name:40s} -> type: {tname}')

# Test couplings and models
for name in ['Linear', 'JansenRit', 'Generic2dOscillator']:
    c = onto.search(label=name)[0]
    t = ontology.get_type(c)
    tname = t.label.first() if hasattr(t, 'label') and t.label.first() else t.name
    print(f'{name:40s} -> type: {tname}')
```

### Result

```
Dopri853                                 -> type: IntegrationMethod
Dopri5                                   -> type: IntegrationMethod
Euler                                    -> type: IntegrationMethod
Heun                                     -> type: IntegrationMethod
Identity                                 -> type: IntegrationMethod
RungeKutta4thOrder                       -> type: IntegrationMethod
VODE                                     -> type: IntegrationMethod
Linear                                   -> type: Coupling
JansenRit                                -> type: Neural Mass Model
Generic2dOscillator                      -> type: Neural Mass Model
```

All integrators now correctly resolve to **IntegrationMethod** instead of **TheVirtualBrain**. Couplings and Neural Mass Models continue to resolve correctly.
