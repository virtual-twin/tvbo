from importlib import import_module

_julia_instance = None
_julia_main = None

# tvbo/utils/julia.py

def install():
    import julia
    julia.install()

def get_julia(compiled_modules=True):
    global _julia_instance, _julia_main
    if _julia_instance is None or _julia_main is None:
        jl = import_module("julia")
        _julia_instance = jl.Julia(compiled_modules=compiled_modules)
        _julia_main = import_module("julia.Main")
    return _julia_instance, _julia_main
