"""JAX's 64-bit switch under the one spelling every supported JAX release accepts."""


def enable_x64(enabled: bool = True):
    """A context in which JAX computes in 64 bits when *enabled*, in 32 bits otherwise.

    JAX 0.5 moved the context manager from ``jax.experimental`` to the top level, and Intel Macs stay on 0.4.x, where only the experimental spelling exists.
    """
    try:
        from jax import enable_x64 as switch
    except ImportError:
        from jax.experimental import enable_x64 as switch
    return switch(enabled)
