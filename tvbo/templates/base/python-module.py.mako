## Layout for a generated Python module: one deduplicated import header, then the body.
##
## A template renders through this by inheriting from it and declaring the collector:
##
##     <%inherit file="/base/python-module.py.mako"/>\
##     <%page args="imports"/>\
##     <% imports.add("import jax.numpy as jnp") %>\
##
## Every fragment that needs a name adds the import line that provides it, wherever it
## happens to be, and the header is assembled once from the union. Fragments carrying
## their own import blocks is what produced `jnp` three times and an unused `jsp` in the
## JAX output; a fragment that stops being emitted now takes its imports with it, which
## is what keeps the header free of F401 rather than a reviewer keeping it so.
##
## The collector is created here and passed into `self.body(imports=...)`, so nothing a
## caller does is required to make it exist. `capture` renders the body FIRST — the adds
## are a side effect of rendering — and the header is written above the captured text.
<%! from mako.runtime import capture %>\
<%
    collected = set()
    body = capture(context, self.body, imports=collected)
    # `import x` before `from x import y`, each group alphabetical.
    header = sorted(collected, key=lambda line: (line.startswith("from"), line))
%>\
% for line in header:
${line}
% endfor

${body}\
