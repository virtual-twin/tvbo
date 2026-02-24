## -*- coding: utf-8 -*-
##
## PyRates Model Template (OperatorTemplate only)
## ================================================
## Generates PyRates OperatorTemplate from TVBO Dynamics model.
## This template defines ONLY the dynamics (equations and variables).
## Use tvbo-pyrates-network.yaml.mako for Node/Circuit topology.
## Use tvbo-pyrates-experiment.yaml.mako for complete runnable experiments.
##
## Note: Custom functions (e.g., Sigm) are automatically inlined into equations
## since PyRates doesn't support user-defined functions in YAML templates.
## We use format='sympy' since PyRates parses equations with SymPy internally.
##
## This template can be used standalone or included via <%namespace>.
##

<%def name="render_operator(m, op_name=None)">
<%
    # Replace reserved names that conflict with SymPy/PyRates built-ins.
    # These names cannot be used as-is because SymPy's sympify (used by PyRates
    # internally) resolves them to built-in objects instead of symbols:
    # - 'I': imaginary unit
    # - 'E': Euler's number (exp(1))
    # - 'S': SymPy's SingletonRegistry (sympify shorthand)
    # - 'N': SymPy's numerical evaluation function
    # - 'O': big-O notation class
    # - 'Q': SymPy's AssumptionKeys object
    # - 'gamma', 'beta', 'zeta': SymPy special functions
    # - 'lambda': Python keyword
    # - 'epsilon': PyRates internal parsing conflict
    # Suffix '_' avoids implying the symbol is a parameter (it may be a state var).
    repl = {
        "I": "I_",
        "gamma": "gamma_",
        "beta": "beta_",
        "zeta": "zeta_",
        "lambda": "lambda_",
        "E": "E_",
        "N": "N_",
        "S": "S_",
        "O": "O_",
        "Q": "Q_",
        "epsilon": "epsilon_",
        "y": "y_",
        "dy": "dy_",
    }

    # Get model name
    name = m.name or "tvbo_model"
    _op_name = op_name or f"{name}_op"

    # --- PyRates-compatibility helpers ---
    # PyRates does not support Piecewise or Abs in equations.
    # Convert Piecewise((a, cond), (b, True)) -> (a+b)/2 + (a-b)/2 * sign(cond_expr)
    # Convert Abs(x) -> sign(x)*x
    import sympy
    from tvbo.knowledge.simulation.equations import sympify as tvbo_sympify

    def _pyrates_compat(eq_str):
        """Post-process a rendered equation string for PyRates compatibility."""
        expr = tvbo_sympify(eq_str)
        expr = _piecewise_to_sign(expr)
        expr = _abs_to_sign(expr)
        expr = _mod_to_fmod(expr)
        return str(expr)

    def _piecewise_to_sign(expr):
        """Recursively replace Piecewise with sign-based arithmetic."""
        if not expr.args:
            return expr
        # First recurse into children
        new_args = [_piecewise_to_sign(a) for a in expr.args]
        expr = expr.func(*new_args) if new_args != list(expr.args) else expr
        if isinstance(expr, sympy.Piecewise):
            return _convert_piecewise(expr)
        return expr

    def _convert_piecewise(pw):
        """Convert a 2-branch Piecewise to sign-based expression.

        Piecewise((a, x > c), (b, True))
          -> (a + b)/2 + (a - b)/2 * sign(x - c)

        For Piecewise((a, x < c), (b, True))
          -> (a + b)/2 - (a - b)/2 * sign(x - c)

        Multi-branch: nest from last to first.
        """
        pieces = list(pw.args)
        # Start with default (last piece, condition=True)
        result = pieces[-1][0]
        for val, cond in reversed(pieces[:-1]):
            # Extract comparison: cond is a relational like x > c or x < c
            sign_arg, negate = _extract_sign_arg(cond)
            if sign_arg is not None:
                s = sympy.Function('sign')(sign_arg)
                if negate:
                    s = -s
                # result = old_default; new = (val + result)/2 + (val - result)/2 * s
                result = (val + result) / 2 + (val - result) / 2 * s
            else:
                # Fallback: cannot convert, keep as-is (will likely fail in PyRates)
                return pw
        return result

    def _extract_sign_arg(cond):
        """Extract sign argument from a relational condition.

        Returns (sign_arg, negate) where:
        - sign(sign_arg) is positive when condition is True
        - negate=True means we should flip the sign
        """
        if isinstance(cond, (sympy.StrictGreaterThan, sympy.GreaterThan)):  # x > c
            return cond.lhs - cond.rhs, False
        elif isinstance(cond, (sympy.StrictLessThan, sympy.LessThan)):  # x < c
            return cond.lhs - cond.rhs, True
        return None, False

    def _abs_to_sign(expr):
        """Replace Abs(x) with sign(x)*x throughout expression."""
        return expr.replace(
            lambda e: isinstance(e, sympy.Abs),
            lambda e: sympy.Function('sign')(e.args[0]) * e.args[0]
        )

    def _mod_to_fmod(expr):
        """Replace Mod(a, b) with fmod(a, b) for numpy compatibility."""
        return expr.replace(
            lambda e: isinstance(e, sympy.Mod),
            lambda e: sympy.Function('fmod')(e.args[0], e.args[1])
        )

    # Collect equations and variables
    equations = []
    variables = {}

    # Build list of terms to remove - only remove local_coupling if not explicitly defined
    # as a coupling input or coupling term
    coupling_input_names = list((m.coupling_inputs or {}).keys()) if hasattr(m, 'coupling_inputs') else []
    coupling_term_names = list((m.coupling_terms or {}).keys())
    defined_coupling_names = coupling_input_names + coupling_term_names

    remove_terms = []
    if 'local_coupling' not in defined_coupling_names:
        remove_terms.append('local_coupling')

    # Add derived variables (algebraic equations) — apply repl to keys too
    for k, dv in m.derived_variables.items():
        display_k = repl.get(k, k)
        raw_eq = m.render_equation(dv, format='sympy', inline_functions=True, replace=repl, remove=remove_terms)
        equations.append(f"{display_k} = {_pyrates_compat(raw_eq)}")
        variables[display_k] = "variable"

    # Add state variable equations (differential equations) — apply repl to keys/LHS
    for k, sv in (m.state_variables or {}).items():
        display_k = repl.get(k, k)
        raw_eq = m.render_equation(sv, format='sympy', inline_functions=True, replace=repl, remove=remove_terms)
        equations.append(f"{display_k}' = {_pyrates_compat(raw_eq)}")
        iv = sv.initial_value
        variables[display_k] = f"variable({iv})"

    # For non-autonomous models, add time variable 't' so PyRates can resolve it
    if getattr(m, 'autonomous', True) is False:
        variables["t"] = "variable"

    # Add parameters as constants — apply repl to keys
    for param_name, param in (m.parameters or {}).items():
        if param_name in repl:
            param_name = repl[param_name]

        val = param.value
        variables[param_name] = float(val)

    # Add derived parameters as equations — apply repl to keys
    for dp_name, dp in (m.derived_parameters or {}).items():
        display_dp = repl.get(dp_name, dp_name)
        eq_str = m.render_equation(dp, format='sympy', inline_functions=True, replace=repl, remove=remove_terms)
        equations.append(f"{display_dp} = {eq_str}")
        variables[display_dp] = "variable"

    # Add coupling terms as inputs
    for ct_name in (m.coupling_terms or {}).keys():
        variables[ct_name] = "input"

    description = (m.description or f"TVBO model: {name}").replace('\\', '\\\\').replace('"', "'")
%>\
${_op_name}:
  base: OperatorTemplate
  description: "${description}"
% if len(equations) == 1:
  equations: "${equations[0]}"
% else:
  equations:
% for eq in equations:
    - "${eq}"
% endfor
% endif
  variables:
% for var_name, var_spec in variables.items():
% if isinstance(var_spec, float):
    ${var_name}: ${var_spec}
% else:
    ${var_name}: ${var_spec}
% endif
% endfor
</%def>\
##
## Standalone rendering when used directly (not via namespace)
##
% if 'model' in context.keys() and context['model'] is not None:
<%
model = context['model']
name = model.name or "tvbo_model"
op_name = f"{name}_op"
%>\
# PyRates OperatorTemplate: ${name}
# Generated from TVBO Dynamics model

${render_operator(model, op_name)}
% endif
