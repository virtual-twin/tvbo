<%!
    import numpy as np
%>
<%
if 'experiment' in context.keys():
    noise = context['experiment'].integration.noise
    nsig = 0.5 * context['experiment'].noise_sigma_array**2
else:
    noise = context['noise']
    nsig = [noise.nsig] if getattr(noise, 'nsig', None) is not None else []

# Select the base class for the integrator based on whether it is stochastic
if noise or (len(nsig) > 0):
    additive = False
    try:
        additive = bool(getattr(noise, 'additive', False))
    except Exception:
        additive = False
    base_class = "Additive" if additive else "Multiplicative"

    extra_params = []
    def _scalarize(val, as_int=False):
        try:
            if isinstance(val, (list, tuple, set)):
                val = list(val)[0] if len(val) > 0 else 0
            elif hasattr(val, 'shape'):
                # numpy array or similar
                try:
                    val = np.ravel(val)[0]
                except Exception:
                    pass
            # Convert to desired numeric type
            if as_int:
                return int(val)
            return float(val)
        except Exception:
            # best-effort fallback
            return 0 if as_int else 0.0
    try:
        params = getattr(noise, 'parameters', {}) or {}
        # map schema names to TVB expected names
        for p in params.values() if hasattr(params, 'values') else []:
            pname = getattr(p, 'name', None) if not isinstance(p, dict) else p.get('name')
            pval = getattr(p, 'value', None) if not isinstance(p, dict) else p.get('value')
            # Only pass scalar TVB traits here; nsig handled separately
            if pname in ['tau', 'ntau', 'noise_seed']:
                if pname == 'noise_seed':
                    extra_params.append(f"noise_seed={_scalarize(pval, as_int=True)}")
                else:
                    extra_params.append(f"{pname}={_scalarize(pval)}")
    except Exception:
        pass

    nsig_arr = f"np.array([{', '.join(map(str, nsig))}])" if len(nsig) > 0 else "np.array([0.0])"
    param_str = ", ".join([f"nsig={nsig_arr}"] + extra_params)
    noise_print = f"{base_class}({param_str})"
else:
    noise_print = "None"
%>
################################################################################
# Noise TVB
noise = ${noise_print}



