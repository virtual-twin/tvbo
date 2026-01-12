# -*- coding: utf-8 -*-
<%doc>
dfun Template - Uses base/dfun.mako for generating dynamics functions.
Supports: jax, numpy, tvboptim (standard signature) with optional auxiliaries.
</%doc>
<%namespace name="dfun" file="/base/dfun.mako"/>
<%
model = experiment.local_dynamics if 'experiment' in context.keys() else context['model']
_format = context['format'] if 'format' in context.keys() else 'jax'
_fmt = 'jax' if _format in ('jax', 'autodiff', 'tvboptim') else 'numpy'
_return_aux = context['return_aux'] if 'return_aux' in context.keys() else (_format == 'tvboptim')
%>
${dfun.imports(fmt=_fmt)}

${dfun.full_dfun(model, fmt=_fmt, signature='standard', return_aux=_return_aux)}
