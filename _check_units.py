from tvbo.export.experiment import SimulationExperiment

for name in ['gas_network', 'init_tutorial']:
    exp = SimulationExperiment.from_file(f'docs/Interoperability/NetworkDynamics.jl/yaml/{name}.yaml')
    model = exp.dynamics
    if model:
        units = {}
        for p_name, p in model.parameters.items():
            if getattr(p, 'unit', None):
                units[p_name] = p.unit
        for sv_name, sv in model.state_variables.items():
            if getattr(sv, 'unit', None):
                units[sv_name] = sv.unit
        if units:
            print(f'{name} ({model.name}): {len(units)} units')
            for k, v in units.items():
                print(f'  {k}: {v}')
        else:
            print(f'{name} ({model.name}): no units')
    else:
        print(f'{name}: no top-level dynamics')

    # Check node-specific dynamics
    if hasattr(exp, 'network') and exp.network and hasattr(exp.network, 'nodes'):
        for node in exp.network.nodes:
            dyn = getattr(node, 'dynamics', None) or getattr(node, 'model', None)
            if dyn and hasattr(dyn, 'parameters'):
                node_units = {}
                for pn, pv in (dyn.parameters or {}).items():
                    if getattr(pv, 'unit', None):
                        node_units[pn] = pv.unit
                for sn, sv in (dyn.state_variables or {}).items():
                    if getattr(sv, 'unit', None):
                        node_units[sn] = sv.unit
                if node_units:
                    nid = getattr(node, 'id', '?')
                    dname = getattr(dyn, 'name', '?')
                    print(f'  node {nid} ({dname}): {node_units}')
