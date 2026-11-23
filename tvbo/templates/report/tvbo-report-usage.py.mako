<%doc>
Study Document Usage
====================

The Python a reader types for one usage operation of a study document. `tvbo/utils/document.py` resolves every value and decides which imports and which load the document still owes; this template only lays out the lines.

Context Variables:
- operation: load | generate | run | figure | report | share
- imports: the operations whose import is still owed, in the order they are written
- load: whether to write the line that loads the study
- path: the study file, relative to the rendered document
- experiment: the experiment as a Python literal (`1`, `"baseline"`)
- backend: the backend the code is generated for
- figure: the declared figure's name
</%doc>\
% for key in imports:
% if key == "load":
from tvbo import SimulationStudy
% elif key == "share":
from tvbo.platform import TVBOPlatform
% endif
% endfor
% if imports:

% endif
% if load:
study = SimulationStudy.from_file("${path}")
% endif
% if operation == "generate":
print(study.get_experiment(${experiment}).render_code("${backend}"))
% elif operation == "run":
result = study.run()
% elif operation == "figure":
result.figure("${figure}")
% elif operation == "report":
result.report("pdf", outputfile="report.pdf")
% elif operation == "share":
platform = TVBOPlatform("https://tvbo.charite.de", api_key="<your API key>")
platform.push_experiment(study.get_experiment(${experiment}), visibility="shared")
% endif
