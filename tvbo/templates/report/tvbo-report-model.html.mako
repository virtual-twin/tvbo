<%
from sympy import latex, Eq, symbols, sympify, pretty
import pydot
import networkx as nx

state_equations = [eq for k, eq in model.equations.items() if k in model.metadata.state_variables]
derived_variables = [eq for k, eq in model.equations.items() if k in model.metadata.derived_variables]
output_transforms = [eq for k, eq in model.equations.items() if k in model.metadata.output_transforms]
derived_parameters = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.metadata.derived_parameters.values()
]

# Prepare the dependency tree with MathJax-ready labels
dependency_tree = model.get_dependency_tree()
for node in dependency_tree.nodes:
    dependency_tree.nodes[node]['label'] = f"{node}"

# Generate the DOT graph
pydot_graph = nx.drawing.nx_pydot.to_pydot(dependency_tree)

# Convert DOT to SVG
svg_content = pydot_graph.create_svg().decode("utf-8")
%>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${model.metadata.name}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { width: 90%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
        th { background-color: #f4f4f4; }
        .equation { margin: 20px 0; }
        .plot { margin-top: 30px; text-align: center; }
        .plot svg { max-width: 100%; height: auto; }
    </style>
    <!-- Include MathJax -->
    <script type="text/javascript" id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            MathJax.typesetPromise();
        });
    </script>
</head>
<body>
    <h1>${model.metadata.name}</h1>
    <p>${model.metadata.description if model.metadata.description else ""}</p>

    <h2>Equations</h2>
    % if derived_parameters:
    <h3>Derived Parameters</h3>
    <div class="equations">
        % for eq in derived_parameters:
        <div class="equation">
            <p>$$ ${latex(eq, mul_symbol='*')} $$</p>
        </div>
        % endfor
    </div>
    % endif

    % if derived_variables:
    <h3>Derived Variables</h3>
    <div class="equations">
        % for eq in derived_variables:
        <div class="equation">
            <p>$$ ${latex(eq, mul_symbol='*')} $$</p>
        </div>
        % endfor
    </div>
    % endif

    <h3>State Equations</h3>
    <div class="equations">
        % for eq in state_equations:
        <div class="equation">
            <p>$$ ${latex(eq, mul_symbol='*')} $$</p>
        </div>
        % endfor
    </div>

    % if output_transforms:
    <h3>Output Transforms</h3>
    <div class="equations">
        % for eq in output_transforms:
        <div class="equation">
            <p>$$ ${latex(eq, mul_symbol='*')} $$</p>
        </div>
        % endfor
    </div>
    % endif

    <h2>Parameters</h2>
    <table>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Unit</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            % for p in model.metadata.parameters.values():
            <tr>
                <td>\(${latex(symbols(p.name))}\)</td>
                <td>${p.value}</td>
                <td>${p.unit if p.unit else "N/A"}</td>
                <td>${p.description}</td>
            </tr>
            % endfor
        </tbody>
    </table>

    <h2>Dependency Tree</h2>
    <div class="plot">
        ${svg_content | n}
    </div>
</body>
</html>
