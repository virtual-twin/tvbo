<%!
    import numpy as np
%>
<%
if 'experiment' in context.keys():
    integration = context['experiment'].metadata.integration
    # consider state-wise noise as stochastic, too
    sw = getattr(context['experiment'], 'noise_sigma_array', None)
    try:
        has_state_noise = (sw is not None) and np.any(np.asarray(sw) > 0)
    except Exception:
        has_state_noise = bool(sw)
else:
    integration = context['integrator'].metadata
    has_state_noise = False
# Select the base class for the integrator based on whether it is stochastic and/or uses SciPy ODE solvers.
stochastic = (integration.noise is not None) or has_state_noise

if stochastic:
    if scipy_ode_base:
        base_class = 'SciPySDE, IntegratorStochastic'
    else:
        base_class = 'IntegratorStochastic'
else:
    if scipy_ode_base:
        base_class = 'SciPyODE, Integrator'
    else:
        base_class = 'Integrator'

# Define the time step variables
dt = 1 if integration.method == 'Identity' else integration.step_size

# Set noise expressions based on the stochastic parameter
if stochastic:
    noise_expr = "self.noise.generate(X.shape)"
    noise_gfun_expr = "self.noise.gfun(X)"
else:
    noise_expr = "0"
    noise_gfun_expr = "1"

# Set the number of integration steps
n_dx = len(integration.intermediate_expressions) + 1
%>
################################################################################
# TVB Integrator
class ${integration.method + ('Stochastic' if stochastic else '')}(${base_class}):
    """
    This is a custom Integrator class generated from a template.
    It supports both deterministic and stochastic schemes with variable intermediate steps.
    """
    % if not scipy_ode_base:

    n_dx = ${n_dx}

    def scheme(self, X, dfun, coupling, local_coupling=0.0, stimulus=0.0):
        """
        Defines the integration scheme for the custom integrator.
        """

        # Define the key expressions and time steps
        dt = self.dt

        # Calculate the noise based on whether the integrator is stochastic
        noise = ${noise_expr}
        noise_gfun = ${noise_gfun_expr}
        noise *= noise_gfun

        dX0 = dfun(X, coupling, local_coupling)

        % for k, step in integration.intermediate_expressions.items():
        # Calculate intermediate step ${k}
        ${k} = ${step.equation.rhs}
        self.integration_bound_and_clamp(${k})

        # Calculate derivative ${k}
        d${k} = dfun(${k}, coupling, local_coupling)

        % endfor
        # Calculate the state change dX
        dX = ${integration.update_expression.equation.rhs}

        # Calculate the next state of X, including noise
        X_next = X + dX + noise + stimulus * dt
        self.integration_bound_and_clamp(X_next)

        return X_next
    % else:
    _scipy_ode_integrator_name = "${class_name.lower()}"
    % endif
