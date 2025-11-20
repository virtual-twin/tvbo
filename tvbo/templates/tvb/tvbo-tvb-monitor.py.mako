<%
monitors = [f"{m.name}(period={m.period})" if m.name != "Raw" else f"{m.name}()" for k, m in experiment.metadata.monitors.items()]
%>
##
from tvb.simulator.monitors import *

monitors = [${', '.join(monitors) if monitors else "Raw()"}]
