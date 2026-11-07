## -*- coding: utf-8 -*-
<%doc>
scipy-compatible dfun Template
==============================

Generates a scipy.odeint-compatible function with parameters as kwargs.
Uses base/dfun.mako with signature='scipy'.
</%doc>
<%namespace name="dfun" file="/base/dfun.mako"/>
<%
# scipy always uses numpy; coupling_as_argument switches to the network runtime's single-vector coupling signature.
_fmt = 'numpy'
_coupling_arg = context.get('coupling_as_argument', False)
%>
${dfun.imports(model=model, fmt=_fmt)}

${dfun.full_dfun(model, fmt=_fmt, signature='scipy', coupling_as_argument=_coupling_arg)}
