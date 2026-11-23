<%doc>The events a model declares, applied to a stepped state by a backend that integrates with jax: `${events.apply_model_events(model, resets)}` emits `_apply_model_events(state, params)`, where *resets* is `tvbo.adapters.base.post_step_resets(model, backend)`.</%doc>
<%!
from tvbo.codegen import render_expression
from tvbo.templates.base.utils import referenced
from tvbo.utils import normalize_params
%>
<%def name="apply_model_events(model, resets)">
<%
    states = list(model.state_variables)
    parameters = list(normalize_params(model.parameters))
    jax = lambda text: render_expression(text, format='jax', parameters=states + parameters)
    sources = [reset.condition for reset in resets] + [rhs for reset in resets for _, rhs in reset.assignments]
%>\
def _apply_model_events(state, params):
    """The model's events on a stepped state: each event, in declared order, makes its assignments, in declared order, where its condition holds."""
% for i, name in enumerate(states):
    ${name} = state[${i}]
% endfor
% for name in referenced(parameters, *sources):
    ${name} = params.${name}
% endfor
% for reset in resets:
    _fired = ${jax(reset.condition)}  # event ${reset.name}
% for lhs, rhs in reset.assignments:
    ${lhs} = jnp.where(_fired, ${jax(rhs)}, ${lhs})
% endfor
% endfor
    return jnp.stack([${', '.join(states)}])
</%def>
