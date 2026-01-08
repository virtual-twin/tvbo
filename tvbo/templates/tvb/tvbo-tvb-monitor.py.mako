<%
# Get observations from experiment - map to TVB monitors
# observations use our schema, monitors are TVB's concept
observations = getattr(experiment, 'observations', None) or {}

# Map observation types to TVB monitor names
obs_to_monitor = {
    'bold': 'Bold',
    'eeg': 'EEG', 
    'meg': 'MEG',
    'seeg': 'iEEG',
    'lfp': 'Raw',
    'raw': 'Raw',
}

monitors = []
for name, obs in observations.items() if hasattr(observations, 'items') else []:
    monitor_name = obs_to_monitor.get(name.lower(), 'Raw')
    period = getattr(obs, 'period', None) or getattr(obs, 'sampling_period', 1.0)
    if monitor_name != 'Raw':
        monitors.append(f"{monitor_name}(period={period})")
    else:
        monitors.append(f"{monitor_name}()")

# Default to Raw() if no observations defined
if not monitors:
    monitors = ["Raw()"]
%>
##
from tvb.simulator.monitors import *

monitors = [${', '.join(monitors)}]
