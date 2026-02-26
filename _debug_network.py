from tvbo import SimulationStudy
from tvbo.export.experiment import SimulationExperiment
from tvbo.data.tvbo_data.connectomes import Network

s = SimulationStudy.from_file('database/studies/Jansen1995/Jansen1995_extracted.yaml')
exp_dm_list = s.experiments or []
print('n experiments:', len(exp_dm_list))
exp_dm = next((e for e in exp_dm_list if getattr(e,'id',None)==3), None)
print('exp_dm module:', type(exp_dm).__module__, type(exp_dm).__qualname__)
print('exp_dm.network module:', type(exp_dm.network).__module__)
print('exp_dm.network isinstance RuntimeNetwork?', isinstance(exp_dm.network, Network))

exp = SimulationExperiment.from_datamodel(exp_dm)
print('from_datamodel network module:', type(exp.network).__module__)
print('from_datamodel isinstance RuntimeNetwork?', isinstance(exp.network, Network))

# Also check get_experiment
exp2 = s.get_experiment(3)
print('get_experiment network module:', type(exp2.network).__module__)
print('get_experiment isinstance RuntimeNetwork?', isinstance(exp2.network, Network))
