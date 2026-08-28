"""The container's flat data-variables, read back in the shape the recipe declares them in.

The result is an `xarray.DataTree`, so the container stays an xarray object end to end: coordinates are declared once at the root and inherited, `.sel` applies across the whole tree, and it writes back out. A `dict`-of-`Bunch` would give the dotted access and none of the rest.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo.data.experiment_result_io import result_tree


@pytest.fixture
def container():
    """A container carrying one output of every shape `ExperimentResult.save` writes."""
    node = ["L.LOG", "L.SFG"]
    return xr.Dataset(
        {
            "integration__peak_frequencies": xr.DataArray([11.0, 7.0], dims=["node"], coords={"node": node}),
            "optimization__spectral_gradient_fit__final_loss": xr.DataArray(np.float64(0.062)),
            "optimization__spectral_gradient_fit__fitted__dynamics__a": xr.DataArray([0.06, 0.05], dims=["node"]),
            "optimization__spectral_gradient_fit__observation__peak_frequencies": xr.DataArray(
                [11.0, 7.0], dims=["node"], coords={"node": node}
            ),
            "results": xr.DataArray(np.zeros((2, 2)), dims=["a", "b"]),
        }
    )


def test_the_path_is_the_one_the_recipe_declares(container):
    """The whole point: an output is reached along the path it was written at, not by its flattened name."""
    tree = result_tree(container)
    fitted = tree.optimizations.spectral_gradient_fit.observations.peak_frequencies
    assert fitted.sel(node="L.LOG") == 11.0


def test_structural_segments_take_the_spec_spelling(container):
    """The writer flattens in the singular, a recipe declares in the plural, and the tree is read beside the recipe."""
    tree = result_tree(container)
    assert sorted(tree.children) == ["integration", "optimizations"]
    assert sorted(tree.optimizations.spectral_gradient_fit.children) == ["fitted", "observations"]
    assert "final_loss" in tree.optimizations.spectral_gradient_fit.dataset.data_vars
    assert sorted(tree.optimizations.spectral_gradient_fit) == ["final_loss", "fitted", "observations"]


def test_a_name_the_author_chose_is_left_alone(container):
    """Only structure is re-spelled; pluralising an author's observation would rename their data."""
    tree = result_tree(container)
    assert "peak_frequencies" in tree.integration
    assert "fitted" in tree.optimizations.spectral_gradient_fit


def test_leaves_keep_their_labels(container):
    """A tree that dropped the coordinates would buy dotted access at the price of label-by-design."""
    assert list(result_tree(container).integration.peak_frequencies.coords["node"].values) == ["L.LOG", "L.SFG"]


def test_both_access_styles_reach_the_same_leaf(container):
    """Groups are addressable by path too, so a segment that is not an identifier is still reachable."""
    tree = result_tree(container)
    assert tree["optimizations/spectral_gradient_fit"]["final_loss"] == tree.optimizations.spectral_gradient_fit.final_loss


def test_coordinates_are_declared_once_at_the_root(container):
    """Labels-by-design as structure: the node labels belong to the run, so every group inherits them instead of each output repeating them."""
    tree = result_tree(container)
    assert "node" in tree["/"].coords
    assert "node" not in tree["/optimizations/spectral_gradient_fit/observations"].to_dataset(inherit=False).coords
    assert list(tree.optimizations.spectral_gradient_fit.observations.peak_frequencies.coords["node"].values) == [
        "L.LOG",
        "L.SFG",
    ]


def test_the_tree_selects_by_label_across_every_group(container):
    """What a `Bunch` could not do: one `.sel` reaches every per-node output at once."""
    picked = result_tree(container).sel(node=["L.LOG"])
    assert picked.integration.peak_frequencies.values.tolist() == [11.0]
    assert picked.optimizations.spectral_gradient_fit.observations.peak_frequencies.values.tolist() == [11.0]


def test_the_tree_writes_back_out(container, tmp_path):
    """A container read into a tree is still a container, so an analysis can persist what it derived."""
    import xarray as xr

    out = tmp_path / "tree.h5"
    result_tree(container).to_netcdf(out, engine="h5netcdf")
    assert xr.open_datatree(out, engine="h5netcdf").optimizations.spectral_gradient_fit.observations.peak_frequencies.size == 2


def test_an_output_that_is_both_a_value_and_a_group_is_refused(container):
    """A group and a variable cannot share a name; choosing one would hide the collision rather than report it."""
    import numpy as np
    import xarray as xr

    clash = container.assign({"integration": xr.DataArray(np.zeros(2), dims=["node"])})
    with pytest.raises(ValueError, match="both a value and a group"):
        result_tree(clash)


def test_an_unnested_output_stays_at_the_root(container):
    """`results` is a whole exploration grid and carries no path; nesting it under a segment would invent one."""
    assert result_tree(container).results.dims == ("a", "b")


def test_a_name_cannot_sanitise_into_the_path_separator(tmp_path):
    """`peak freq (Hz)` once became `peak_freq__Hz_`, which reads back as a group nothing declared.

    The hierarchy is encoded in the variable name, so the separator has to mean only what the writer put there.
    """
    from types import SimpleNamespace

    import numpy as np

    from tvbo.data.types import ExperimentResult, SimulationResult

    labels = ["L.LOG", "L.SFG"]
    source = SimpleNamespace(network=SimpleNamespace(node_labels=labels), dynamics=None, coupling=None)
    integration = SimulationResult(observations={"peak freq (Hz)": np.zeros(2)}, nodes=labels)
    written = ExperimentResult(integration=integration, source=source).save(str(tmp_path), compress=False, record_only=False)

    import xarray as xr

    with xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf") as ds:
        names = [str(v) for v in ds.data_vars]
    assert not [n for n in names if n.count("__") > 1], names
    assert "integration__peak_freq_Hz_" in names
