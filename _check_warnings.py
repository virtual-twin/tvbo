import sys
print('Initializing Julia...', flush=True)
from tvbo.adapters.julia import get_julia
jl, Main = get_julia()
print('Julia initialized.', flush=True)

# Verify Julia actually works
from tvbo.run.julia import run_julia_code
result = run_julia_code("1 + 1")
print(f'Julia 1+1 = {result}', flush=True)
assert result == 2, f"Expected 2, got {result}"
print('All good!', flush=True)
