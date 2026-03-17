## -*- coding: utf-8 -*-
<%doc>
scipy-compatible dfun Template
==============================

Generates a scipy.odeint-compatible function with parameters as kwargs.
Uses base/dfun.mako with signature='scipy'.
</%doc>
<%namespace name="dfun" file="/base/dfun.mako"/>
<%
# Get model from context - handle both Dynamics and wrapped model objects
if hasattr(model, 'metadata'):
    _model = model
else:
    _model = model

# scipy always uses numpy
_fmt = 'numpy'
%>
${dfun.imports(fmt=_fmt)}

${dfun.full_dfun(_model, fmt=_fmt, signature='scipy')}
