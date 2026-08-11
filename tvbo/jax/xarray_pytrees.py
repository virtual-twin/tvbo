# Copyright © Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2.
#
# Registers xarray.Variable, xarray.DataArray, and xarray.Dataset as JAX pytree nodes so they pass through jit, grad, vmap, etc.
#
# Adapted from the pytree registration in google-deepmind/xarray_jax (Apache-2.0) and the PyPI xarray-jax package (MIT).  Both ultimately trace back to google-deepmind/graphcast/xarray_jax.py.
"""Register xarray data structures as JAX pytree nodes."""

import collections.abc
from collections.abc import Hashable, Iterator, Mapping

import jax
import jax.tree_util
import xarray

# _HashableCoords – makes coordinate dicts hashable so JAX can treat them as static auxiliary data in pytree flatten/unflatten.


class _HashableCoords(collections.abc.Mapping):
    """Hashable wrapper around a dict of xarray Variables (coordinates).

    JAX requires auxiliary pytree data to be hashable.  Standard xarray coordinate dicts are not, so we wrap them here.  When coordinates change between calls to a ``jax.jit``-ed function JAX will re-trace.
    """

    __slots__ = ("_variables", "_hash")

    def __init__(self, coord_vars: Mapping[Hashable, xarray.Variable]):
        self._variables = coord_vars

    def __repr__(self) -> str:
        return f"_HashableCoords({self._variables!r})"

    def __getitem__(self, key: Hashable) -> xarray.Variable:
        return self._variables[key]

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._variables)

    def __hash__(self):
        if not hasattr(self, "_hash") or self._hash is None:
            object.__setattr__(
                self,
                "_hash",
                hash(frozenset((name, var.data.tobytes()) for name, var in self._variables.items())),
            )
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return NotImplemented
        if self._variables is other._variables:
            return True
        return self._variables.keys() == other._variables.keys() and all(
            variable.equals(other._variables[name]) for name, variable in self._variables.items()
        )


def _maybe_hash_coords(coords):
    if isinstance(coords, _HashableCoords):
        return coords
    return _HashableCoords(coords)


# xarray.Variable


def _flatten_variable(v: xarray.Variable):
    children = (v._data,)
    aux = (
        v._dims,
        {} if v._attrs is None else v._attrs,
    )
    return children, aux


def _unflatten_variable(aux, children):
    dims, attrs = aux
    (data,) = children
    var = object.__new__(xarray.Variable)
    var._dims = dims
    var._data = data
    var._attrs = attrs
    var._encoding = None
    return var


# xarray.DataArray


def _flatten_data_array(da: xarray.DataArray):
    children = (da._variable,)
    aux = (da._name, _maybe_hash_coords(da._coords), da._indexes)
    return children, aux


def _unflatten_data_array(aux, children):
    name, coords, indexes = aux
    (variable,) = children
    da = object.__new__(xarray.DataArray)
    da._variable = variable
    da._name = name
    da._coords = dict(coords)
    da._indexes = indexes
    return da


# xarray.Dataset


def _flatten_dataset(ds: xarray.Dataset):
    variables = ds._variables
    coord_names = ds._coord_names
    coords = {name: variables[name] for name in coord_names}
    data_vars = {name: variables[name] for name in variables if name not in coord_names}

    children = (data_vars,)
    aux = (_maybe_hash_coords(coords), ds._indexes, ds._dims, ds._attrs)
    return children, aux


def _unflatten_dataset(aux, children):
    (data_vars,) = children
    coords, indexes, dims, attrs = aux

    ds = object.__new__(xarray.Dataset)
    ds._dims = dims
    ds._variables = data_vars | dict(coords)
    ds._coord_names = set(coords.keys())
    ds._attrs = attrs
    ds._indexes = indexes
    ds._encoding = None
    ds._close = None
    return ds


# Registration (runs on import)

jax.tree_util.register_pytree_node(xarray.Variable, _flatten_variable, _unflatten_variable)
jax.tree_util.register_static(xarray.IndexVariable)
jax.tree_util.register_pytree_node(xarray.DataArray, _flatten_data_array, _unflatten_data_array)
jax.tree_util.register_pytree_node(xarray.Dataset, _flatten_dataset, _unflatten_dataset)
