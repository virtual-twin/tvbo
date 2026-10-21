## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl EdgeModel from tvbo Coupling.

A tvbo coupling gives node i the input post(sum_j w_ij * pre(x_i, x_j)). The edge j -> i evaluates the weighted pre-expression over the source's outputs (v_src) and the destination's (v_dst), the states the coupling transmits; NetworkDynamics sums the edges into each node, and the vertex f! applies the post-expression to that sum through the function post_function defines, which the experiment template emits ahead of the vertex models that call it. Every graph is a SimpleDiGraph, so the edge is always Directed: its output goes to the destination alone.

Every statement is resolved by coupling_split (tvbo.adapters.networkdynamics); this template only emits them.

Context: coupling (Coupling instance), split (its coupling_split: the edge body and observables, the edge parameters and output symbols, the post-expression)
</%doc>
<%page args="coupling, split"/>
<%!
from tvbo.codegen import render_expression
%>
<% edge_names = [name for name, _ in split['edge_parameters']] %>

## ── Edge coupling function ──────────────────────────────────────────────────
function ${coupling.name}_edge_g!(e_dst, v_src, v_dst, (${", ".join(edge_names)},), t)
% for line in split['edge_body']:
    ${line}
% endfor
    nothing
end
% if split['observed']:

## ── Edge observed function ──────────────────────────────────────────────────
function ${coupling.name}_obsf!(obsout, u, v_src, v_dst, (${", ".join(edge_names)},), t)
% for _, lines in split['observed']:
% for line in lines:
    ${line}
% endfor
% endfor
    nothing
end
% endif

edge_${coupling.name} = EdgeModel(;
    g = Directed(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in split['outsym'])}],
    psym = [${", ".join(f':{name} => {value}' for name, value in split['edge_parameters'])}],
    % if split['observed']:
    obsf = ${coupling.name}_obsf!,
    obssym = [${", ".join(f':{name}' for name, _ in split['observed'])}],
    % endif
    name = :${coupling.name},
)
<%def name="post_function(split)">\
<% post_names = [name for name, _, _ in split['post_parameters']] %>\
% if split['post'] is not None and post_names:
${split['post_function']}(gx, (${", ".join(post_names)},)) = ${render_expression(split['post'], format='julia', parameters=post_names + ['gx'])}
% elif split['post'] is not None:
${split['post_function']}(gx) = ${render_expression(split['post'], format='julia', parameters=['gx'])}
% endif
</%def>
