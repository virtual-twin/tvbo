(
    """What a study run produces, and the public import location for ``SimulationStudy``.

`StudyResult` and `FigureImage` are what a run hands back — plain objects, not records. The study class itself is the generated one; what it does lives in :mod:`tvbo.behaviour.study`.
"""
    ""
)

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tvbo.datamodel import schema as tvbo_datamodel


@dataclass(frozen=True)
class FigureImage:
    """One figure a study run rendered: the image, the self-contained script that drew it, and an inline representation.

    ``_repr_png_`` / ``_repr_svg_`` are what a notebook or Quarto cell calls, so the cell that ran the study also shows what it drew and a page never writes its own image reference. The object is a :class:`os.PathLike`, so it still passes anywhere a path does.
    """

    name: str
    path: Path
    script: Path | None = None
    caption: str | None = None

    def __fspath__(self) -> str:
        return str(self.path)

    def _repr_png_(self):
        return self.path.read_bytes() if self.path.suffix.lower() == ".png" and self.path.is_file() else None

    def _repr_svg_(self):
        return self.path.read_text(encoding="utf-8") if self.path.suffix.lower() == ".svg" and self.path.is_file() else None

    def _repr_html_(self):
        """An animated figure inline, for the movie kinds a notebook cannot show as a still image.

        Embedded rather than linked: the page that displays it is rendered once and read from somewhere else, so a relative path out of a cached notebook would resolve against whatever directory the reader happens to be in.
        """
        import base64
        from html import escape

        suffix = self.path.suffix.lower()
        if suffix not in (".gif", ".mp4") or not self.path.is_file():
            return None
        data = base64.b64encode(self.path.read_bytes()).decode("ascii")
        if suffix == ".gif":
            return f'<img src="data:image/gif;base64,{data}" alt="{escape(self.name)}" style="max-width:100%;height:auto" />'
        return (
            '<video controls loop muted playsinline style="max-width:100%;height:auto">'
            f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'
        )


class StudyResult(Mapping):
    """What one :meth:`SimulationStudy.run` produced — the study-level counterpart of the :class:`~tvbo.data.types.ExperimentResult` a single experiment's ``run()`` returns.

    A mapping from experiment to that experiment's outputs, read back from the same container the study's figures read, so a page and the figure it shows cannot report different numbers. Keys are the ``exp-<id>`` spelling the recipe's ``id:`` gives each experiment, which is also the name of the container on disk; the experiment's ``label`` reaches it too, and so does the bare id a figure layer writes (``used: {experiment: 2}``). A bare integer is accepted for that last reason and no other: it is the id, never a position in the list.

    Each entry is an :class:`xarray.DataTree` in the recipe's own shape rather than the container's flat one, so an output is reached along the path it was declared at::

        results["exp-1"].optimizations.spectral_gradient_fit.observations.peak_frequencies

    Being a tree rather than a bag of attributes is what keeps the run an xarray object end to end: the node labels are declared once at the root and inherited by every group, one ``.sel`` reaches every per-node output at once, and an analysis can write back out what it derived.

    :meth:`dataset` hands back the underlying :class:`xarray.Dataset` for anyone who wants the flat names, :meth:`analysis` reaches an analysis container by name, :meth:`figure` a rendered figure by name, and :meth:`report` the study's Methods section. Containers are read on first access and cached.
    """

    def __init__(self, study, root: Path, results_root: Path, experiments, figures, studies=None):
        from tvbo.data.dataref import experiment_id

        self.study = study
        self.studies = dict(studies or {})
        """Each nested study's own result, keyed by the label it goes by — empty unless this is a study-of-studies.

        Nested rather than merged into this mapping, because that is the shape the recipe has and the shape the results have on disk: a sub-study keeps its own results root, and two sub-studies both declaring ``exp-1`` are two different experiments.
        """
        self.root = Path(root)
        self.results_root = Path(results_root)
        self._figures = {f.name: f for f in figures}
        self._open: dict[str, object] = {}
        self._keys: list[str] = []
        self._by_alias: dict[str, str] = {}
        for exp in experiments or []:
            eid = experiment_id(getattr(exp, "id", exp))
            if eid is None:
                continue
            key = f"exp-{eid}"
            self._keys.append(key)
            for alias in (key, eid, getattr(exp, "label", None)):
                if alias:
                    self._by_alias[str(alias)] = key

    def _dataset(self, path: Path):
        """The container at *path*, read into memory once and kept.

        Read rather than opened: an open HDF5 handle holds a lock, and a result object that outlives its cell — which every notebook one does — would block the next run from rewriting the container it is holding.
        """
        import xarray as xr

        key = str(path)
        if key not in self._open:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                self._open[key] = ds.load()
        return self._open[key]

    def _container(self, key) -> Path:
        """Path to the container of the experiment *key* names, in any spelling the recipe uses."""
        from tvbo.data.dataref import locate_exp_container

        resolved = self._by_alias.get(str(getattr(key, "id", key)))
        if resolved is None:
            raise KeyError(
                f"{key!r} names no experiment in this study. It declares {self._keys}, "
                f"reachable by that key, by its label, or by the bare id a figure layer writes."
            )
        return locate_exp_container(self.results_root, resolved.removeprefix("exp-"))

    def __getitem__(self, key):
        """Experiment *key*'s outputs as an :class:`xarray.DataTree` shaped like the recipe that declared them."""
        from tvbo.data.experiment_result_io import result_tree

        return result_tree(self._dataset(self._container(key)))

    def dataset(self, key):
        """The raw :class:`xarray.Dataset` behind experiment *key*, with the container's flat variable names."""
        return self._dataset(self._container(key))

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def analysis(self, name: str):
        """The container the analysis *name* wrote, as a labelled :class:`xarray.Dataset`."""
        from tvbo.data.dataref import locate_analysis_container

        return self._dataset(locate_analysis_container(self.results_root, name))

    @property
    def figures(self) -> dict:
        """The rendered figures, keyed by their declared ``name``."""
        return dict(self._figures)

    def _undrawn(self) -> str:
        """The declared figures this run did not draw, named so a reader is not left to conclude the recipe declared none."""
        from tvbo.utils import as_list

        declared = {str(f.name) for f in as_list(getattr(self.study, "figures", None)) if getattr(f, "name", None)}
        missing = sorted(declared - set(self._figures))
        if not missing:
            return ""
        return f"; declared but not drawn: {', '.join(missing)} — rendering failed, and the run logged why at WARNING"

    def figure(self, name: str | None = None) -> FigureImage:
        """The rendered figure *name*, or the only one when the study declares a single figure."""
        if name is None:
            if len(self._figures) != 1:
                raise KeyError(
                    f"this run drew {len(self._figures)} figures; name one of: {sorted(self._figures)}{self._undrawn()}"
                )
            return next(iter(self._figures.values()))
        try:
            return self._figures[name]
        except KeyError:
            raise KeyError(
                f"no figure {name!r} was rendered; this run drew: {sorted(self._figures)}{self._undrawn()}"
            ) from None

    def report(self, *args, **kwargs) -> str:
        """The study's Methods section — :meth:`SimulationStudy.report`, reached from the run that produced the numbers it describes."""
        return self.study.report(*args, **kwargs)

    def __repr__(self) -> str:
        # Relative where that is shorter: this repr is cell output on a docs page, and an absolute path would publish one machine's directory layout and change under every other.
        root = str(self.results_root)
        try:
            rel = os.path.relpath(self.results_root)
            root = rel if len(rel) < len(root) else root
        except ValueError:
            pass
        nested = f", studies={sorted(self.studies)}" if self.studies else ""
        return f"StudyResult(experiments={self._keys}, figures={sorted(self._figures)}{nested}, results_root={root!r})"


SimulationStudy = tvbo_datamodel.SimulationStudy
"""The generated class itself. What a study does lives in :mod:`tvbo.behaviour.study`, attached where
the class is generated, so a study nested inside another is already a study — no promotion step, and
none of the class-rebinding that used to stand in for one."""
