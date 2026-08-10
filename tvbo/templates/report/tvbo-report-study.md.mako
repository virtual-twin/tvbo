<%doc>
Study Methods Report
====================

One Methods section for a whole study, not one per experiment. Experiments that share a
model share its equations and its symbol table; a model that only varies its sibling
contributes its delta alone. What every experiment holds in common is stated once, and
the table compares only what actually differs.

Context Variables:
- experiments: the SimulationExperiment list to describe, in declared order
- part: 'main' | 'supplementary' | 'all' — which experiments carry their full paragraph
- level: heading depth the block is inserted at, so it nests under its host section
- fmt: 'qmd' (Quarto anchors) or 'markdown' (\tag numbering)
- eqs: a report.Equations carrying the numbering across the whole render
- derivative_notation, mul_symbol: equation rendering, as for a single experiment
</%doc>
<%
from tvbo.utils import report

families = report.model_families(experiments)
in_part = lambda e: part == 'all' or str(report.slot(e, 'part', 'main')) == part
heading = '#' * max(1, int(level))
%>\
<%def name="equations(model, keys=None)">\
% for name, ltx in report.model_equations(model, 'state', derivative_notation, mul_symbol):
% if keys is None or name in keys:
${eqs.block(ltx, model, name)}
% endif
% endfor
</%def>\
<%def name="derived(model, keys=None)">\
<% items = report.model_equations(model, 'derived', derivative_notation, mul_symbol) %>\
% if items and (keys is None or any(n in keys for n, _ in items)):

where

% for name, ltx in items:
% if keys is None or name in keys:
${eqs.block(ltx, model, name)}
% endif
% endfor
% endif
</%def>\
<%def name="functions(model)">\
% for name, ltx in report.model_functions(model, derivative_notation, mul_symbol):

${eqs.block(ltx, model, name)}
% endfor
</%def>\
% for family in families:
<% members = [e for e in family.experiments if in_part(e)] %>\
% if members:

${heading} ${family.label} {#${eqs.unique_anchor('sec-model-' + report.section_slug(report.slot(family.base.model, 'name', family.label)))}}

% if report.slot(family.base.model, 'description', None):
${report.slot(family.base.model, 'description').strip()}

% endif
${equations(family.base.model)}\
${derived(family.base.model)}\
${functions(family.base.model)}\
% if report.slot(family.base.model, 'events', None):

${report.captioned(report.event_table(report.slot(family.base.model, 'events'), derivative_notation),
                   f"Events of the {family.label}: stimuli, resets and the conditions that fire them.",
                   f"events-{report.slot(family.base.model, 'name', family.label)}", fmt, eqs)}\
% endif

${report.captioned(report.symbol_table(family.base.model, report.study_sweeps(family.experiments),
                                       report.coupling_of(family.experiments)),
                   f"Symbols of the {family.label}, including those its coupling introduces.",
                   f"model-{report.slot(family.base.model, 'name', family.label)}", fmt, eqs)}\
% for variant in family.variants:
<% changed = variant.delta.eq_svars | variant.delta.dvars %>\

${report.variant_sentence(variant, eqs, family.base.model)}
% if changed:

${equations(variant.model, changed)}\
${derived(variant.model, changed)}\
% endif
% endfor

${report.captioned(report.variant_parameter_table(family),
                   f"Parameters the variants of the {family.label} change; everything else is as above.",
                   f"variants-{report.slot(family.base.model, 'name', family.label)}", fmt, eqs)}\
% if report.coupling_of(family.experiments):

${report.coupling_prose(family.experiments, eqs)}
% endif

${report.captioned(report.experiment_table(family.experiments, family.shared_parameters, orient),
                   f"Experiments using the {family.label}. Quantities identical across them are stated above and omitted here.",
                   f"experiments-{report.slot(family.base.model, 'name', family.label)}", fmt, eqs)}\
% for exp in members:

${heading}# Experiment ${report.slot(exp, 'id', '')}: ${report.experiment_title(exp)} {#${eqs.unique_anchor('sec-experiment-' + report.section_slug(report.slot(exp, 'id', '')))}}

${report.settings_sentence(exp)}
% if report.slot(exp, 'description', None):

${report.slot(exp, 'description').strip()}
% endif
% endfor
% endif
% endfor
<% obs = report.observation_table([e for e in experiments if in_part(e)]) %>\
% if obs.table:

${heading} Recorded output

${report.captioned(obs.table, "What each experiment records, and how it is reduced."
                              + (f" Throughout, {obs.shared}." if obs.shared else ""),
                   "observations", fmt, eqs)}\
% if obs.notes:

${obs.notes}
% endif
% endif
