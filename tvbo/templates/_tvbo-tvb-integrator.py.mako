<%!
    import numpy as np
%>

# Custom Integrator Template for TVB
<%
# Select the base class for the integrator based on whether it is stochastic and/or uses SciPy ODE solvers.
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
dt = 1 if identity else "self.dt"

# Set noise expressions based on the stochastic parameter
if stochastic:
    noise_expr = "self.noise.generate(X.shape)"
    noise_gfun_expr = "self.noise.gfun(X)"
else:
    noise_expr = "0"
    noise_gfun_expr = "1"

# Set the number of intermediate steps
num_steps = n_dx
%>

class ${class_name}(${base_class}):
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
        dt = ${dt}

        # Calculate the noise based on whether the integrator is stochastic
        noise = ${noise_expr}
        noise_gfun = ${noise_gfun_expr}
        noise *= noise_gfun

        k1 = dfun(X, coupling, local_coupling)

        % for i, step in enumerate(intermediate_steps):
        # Calculate intermediate step inter_k${i+1}
        inter_k${i+1} = ${step}
        self.integration_bound_and_clamp(inter_k${i+1})

        # Calculate derivative k${i+2}
        k${i+2} = dfun(inter_k${i+1}, coupling, local_coupling)

        % endfor
        # Calculate the state change dX
        dX = ${dX_expr}

        # Calculate the next state of X, including noise
        X_next = X + dX + noise + stimulus * dt
        self.integration_bound_and_clamp(X_next)

        return X_next
    % else:
    _scipy_ode_integrator_name = "${class_name.lower()}"
    % endif
