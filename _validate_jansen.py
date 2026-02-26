import yaml
from tvbo.datamodel.tvbopydantic import SimulationStudy

with open('database/studies/Jansen1995/Jansen1995_extracted.yaml') as f:
    data = yaml.safe_load(f)

# Remove YAML anchors (underscore-prefixed keys not in schema)
for key in list(data.keys()):
    if key.startswith('_'):
        del data[key]

study = SimulationStudy.model_validate(data)
print("Valid: %s, %d experiments" % (study.key, len(study.simulation_experiments)))
for exp in study.simulation_experiments:
    parts = ["  Exp %d: %s" % (exp.id, exp.label)]
    if exp.explorations:
        for name, expl in exp.explorations.items():
            parts.append("    Exploration: %s (%d params)" % (name, len(expl.parameters)))
    if exp.connectivity:
        n = exp.connectivity
        parts.append("    Network: %d regions" % n.number_of_regions)
        if n.nodes:
            parts.append("    Nodes: %d" % len(n.nodes))
        if n.edges:
            parts.append("    Edges: %d" % len(n.edges))
        if n.coupling:
            parts.append("    Coupling: %s" % list(n.coupling.keys()))
    if exp.dynamics:
        parts.append("    Dynamics: %s" % exp.dynamics.name)
    if exp.stimulation:
        parts.append("    Stimulation: %s" % exp.stimulation.label)
    print("\n".join(parts))
