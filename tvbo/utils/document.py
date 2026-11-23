# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Render a study's declared document: ordered sections of prose and of blocks TVB-O produces from the study itself.

A ``StudyDocument`` (``schema/document.yaml``) is what ``SimulationStudy.report()`` renders when a study declares one. Every block other than ``text`` is produced here from the study or the installed package — an excerpt of the study file cut with its comments, the Python a reader types, a unit of generated code, a rendered figure, the Methods, counts from the curated database, the backend matrix, the tree a run wrote, the lifecycle wheel — so a document cannot drift from the code it describes. Every selector is keyed (a dotted path by declared key, code by symbol, records by IRI) and a selector that no longer matches fails the render by name.

Blocks that show a run (``figure``, ``tree``) read the :class:`~tvbo.classes.study.StudyResult` the report is rendered from, and write what they show into the document's ``assets`` directory; rendered with ``reuse_assets=True`` and no result, they read those files back instead, which is how a check regenerates everything else without running the study.
"""

from __future__ import annotations

import ast
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FORMATS = ("gfm", "markdown", "md", "qmd", "pdf")

READS = {
    "text": {"text"},
    "spec": {"study", "select"},
    "usage": {"study", "operation", "experiment", "backend", "figure"},
    "generated_code": {"study", "experiment", "backend", "symbol"},
    "figure": {"study", "figure"},
    "methods": {"study"},
    "catalogue": {"of", "entries"},
    "backends": {"form"},
    "tree": {"study", "depth"},
    "lifecycle": {"band", "icons"},
}
"""The slots each block kind reads; a block setting any other is refused."""

USAGE = {
    "load": (),
    "generate": ("experiment", "backend"),
    "run": (),
    "figure": ("figure",),
    "report": (),
    "share": ("experiment",),
}
"""Each usage operation and the block slots its snippet needs; `report/tvbo-report-usage.py.mako` lays the snippet out."""

USAGE_IMPORTS = ("load", "share")
"""The operations whose snippet opens with an import of its own, in the order the imports are written. Usage blocks are cumulative: an import is written once, the first time it is owed, and a study is loaded once."""


@dataclass
class _Context:
    """What one render carries from block to block."""

    study: Any
    result: Any
    fmt: str
    out_dir: Path
    assets: Path | None
    asset_url: str | None
    reuse_assets: bool
    imports: set = field(default_factory=set)
    bound: dict = field(default_factory=dict)


def render(
    study, format: str = "gfm", *, outputfile: str | os.PathLike | None = None, result=None, reuse_assets: bool = False
) -> str:
    """The study's document, rendered.

    Args:
        study: A ``SimulationStudy`` that declares a ``document``.
        format: ``gfm`` (GitHub, raw HTML kept: ``<picture>`` for the wheel, ``<details>`` for a folded section), ``markdown``/``md`` (the same text), ``qmd`` (Quarto, Methods with Quarto anchors), or ``pdf`` (through pandoc); these two show the light wheel and matrix only.
        outputfile: Write the render here. Relative asset links and usage paths are written relative to its directory, else to the study file's.
        result: The study's run, for ``figure`` and ``tree`` blocks.
        reuse_assets: Without a *result*, read ``figure`` and ``tree`` blocks from the files a previous render left in ``assets``.

    Raises:
        ValueError: The document is malformed, a selector matches nothing, or a block needs a run that was not given.
    """
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, not {format!r}")
    document = study.document
    source = Path(study._source_file).resolve().parent
    out_dir = Path(outputfile).resolve().parent if outputfile else source
    assets = (source / document.assets).resolve() if document.assets else None
    asset_url = str(document.asset_url).rstrip("/") if document.asset_url else None
    ctx = _Context(
        study, result, {"markdown": "gfm", "md": "gfm"}.get(format, format), out_dir, assets, asset_url, reuse_assets
    )

    parts = [f"# {document.title}"] if document.title else []
    for section in document.sections or []:
        body = "\n\n".join(filter(None, (_block(ctx, block, section.level) for block in section.blocks or [])))
        icon = _stage_icon(ctx, str(section.stage)) if section.stage else ""
        if section.collapsed and ctx.fmt != "pdf":
            parts.append(f"<details>\n<summary>{icon}<b>{section.title}</b></summary>\n\n{body}\n\n</details>")
        elif section.title:
            parts.append(f"{'#' * section.level} {icon}{section.title}\n\n{body}")
        else:
            parts.append(body)
    text = "\n\n".join(p.strip("\n") for p in parts if p.strip()) + "\n"

    if outputfile:
        if format == "pdf":
            from tvbo.utils import report

            report.to_pdf(text, str(outputfile))
        else:
            Path(outputfile).write_text(text, encoding="utf-8")
    return text


def _block(ctx: _Context, block, level: int) -> str:
    """The block rendered, or nothing when it names its formats and this render's is not among them."""
    kind = str(block.kind)
    stray = sorted(
        k
        for k, v in vars(block).items()
        if not k.startswith("_") and k not in ("kind", "formats") and v not in (None, [], {}) and k not in READS[kind]
    )
    if stray:
        raise ValueError(f"a {kind} block reads {sorted(READS[kind])}, not {stray}")
    if block.formats and ctx.fmt not in {str(f) for f in block.formats}:
        return ""
    return _RENDERERS[kind](ctx, block, level).strip("\n")


def _target(ctx: _Context, block) -> tuple[Any, Any, str | None]:
    """The study a block addresses, its result when the render has one, and its label (``None`` for the document's own study)."""
    if not block.study:
        return ctx.study, ctx.result, None
    label = str(block.study)
    found = dict(ctx.study.walk_studies(include_self=False))
    if label not in found:
        raise ValueError(f"no nested study is labelled {label!r}; this study holds {sorted(found)}")
    return found[label], _nested_result(ctx.result, label), label


def _nested_result(result, label: str):
    if result is None:
        return None
    if label in result.studies:
        return result.studies[label]
    return next((r for r in (_nested_result(sub, label) for sub in result.studies.values()) if r is not None), None)


def _experiment_id(study, block) -> Any:
    """The declared id of the one experiment of *study* that *block*'s ``experiment`` selects, matched as ``tvbo run --experiment`` matches it (:func:`tvbo.run.study.select_experiments`); a refusal names the block."""
    from tvbo.run.study import StudyRunError, select_experiments

    try:
        records = select_experiments(study, str(block.experiment))
    except StudyRunError as exc:
        raise ValueError(f"a {block.kind} block: {exc}") from None
    if len(records) != 1:
        raise ValueError(
            f"a {block.kind} block shows one experiment, but {block.experiment!r} selects {[r.id for r in records]}"
        )
    return records[0].id


def _link(ctx: _Context, path: Path) -> str:
    """How the document links an asset: Markdown by the declared ``asset_url``, read where relative paths do not resolve; pandoc's PDF by absolute path; Quarto, and Markdown without a URL, relative to the rendered file."""
    if ctx.fmt == "gfm" and ctx.asset_url:
        return f"{ctx.asset_url}/{path.relative_to(ctx.assets).as_posix()}"
    if ctx.fmt == "pdf":
        return path.as_posix()
    return Path(os.path.relpath(path, ctx.out_dir)).as_posix()


def _asset(ctx: _Context, name: str) -> Path:
    if ctx.assets is None:
        raise ValueError(f"this document writes {name}, but declares no `assets` directory to write it into")
    if not ctx.reuse_assets:
        ctx.assets.mkdir(parents=True, exist_ok=True)
    return ctx.assets / name


def _pair(ctx: _Context, stem: str, draw) -> tuple[Path, Path]:
    """*draw* in the default and the dark theme, as the assets ``<stem>-light.svg`` and ``<stem>-dark.svg``.

    A render from a run writes them. A render that reuses its assets writes nothing: it checks that the files on disk are what *draw* produces now, and fails on one that is missing or stale, so reading a document never edits the tree it is read from.
    """
    pair = []
    for suffix, theme in (("light", "tvbo:themes/default"), ("dark", "tvbo:themes/dark")):
        path, svg = _asset(ctx, f"{stem}-{suffix}.svg"), draw(theme)
        if not ctx.reuse_assets:
            path.write_text(svg, encoding="utf-8")
        elif not path.is_file() or path.read_text(encoding="utf-8") != svg:
            raise ValueError(f"{path} is missing or stale; render the document from a run (`make readme`)")
        pair.append(path)
    return pair[0], pair[1]


def _dark_light(ctx: _Context, dark: Path, img: str, *, block: bool = False) -> str:
    """*img*, the light rendering's tag, in the ``<picture>`` that has GitHub show *dark* to a reader in dark mode: on lines of its own for a *block* figure, else inline, as a heading's icon is."""
    source = f'<source media="(prefers-color-scheme: dark)" srcset="{_link(ctx, dark)}">'
    if block:
        return f"<picture>\n  {source}\n  {img}\n</picture>"
    return f"<picture>{source}{img}</picture>"


def _fenced(lang: str, text: str) -> str:
    """*text* as a fenced code block in *lang*."""
    return f"```{lang}\n{text}\n```"


def _picture(ctx: _Context, stem: str, draw, alt: str) -> str:
    """A light and a dark rendering written as assets; GitHub picks by colour scheme, every other reader gets the light one."""
    light, dark = _pair(ctx, stem, draw)
    if ctx.fmt != "gfm":
        return f"![{alt}]({_link(ctx, light)})"
    return _dark_light(ctx, dark, f'<img src="{_link(ctx, light)}" alt="{alt}" width="100%">', block=True)


def _stage_icon(ctx: _Context, stage: str) -> str:
    """The inline icon that opens a stage's heading: light and dark on GitHub, the light one in Quarto, none in a PDF, whose headings carry no HTML."""
    from tvbo.plot import lifecycle

    if ctx.fmt == "pdf":
        return ""
    light, dark = _pair(ctx, f"stage-{stage}", lambda theme: lifecycle.icon(stage, theme))
    img = f'<img src="{_link(ctx, light)}" alt="" width="20" height="20" align="top">'
    if ctx.fmt != "gfm":
        return img + " "
    return _dark_light(ctx, dark, img) + " "


def _text(ctx, block, level) -> str:
    return str(block.text or "")


def _spec(ctx, block, level) -> str:
    study, _, _ = _target(ctx, block)
    return _fenced("yaml", cut(Path(study._source_file), str(block.select or "")))


def cut(path: Path, select: str) -> str:
    """The author's own lines of the YAML file *path* at the dotted path *select*, comments and order kept.

    A segment names a mapping key, or a list item by its declared ``id``, ``name``, ``citekey`` or ``label``; an ``!include`` on the way is followed into the file it names. A key's lines include the key itself; a list item's are the item; a value that is a whole included file is that file. An empty *select* is the whole file.
    """
    from tvbo.utils.yaml_loader import _include_path

    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    node, key = yaml.compose(text), None
    for seg in [s for s in select.split(".") if s]:
        node, followed = _follow(node, path)
        key = None
        if followed != path:
            path, text = followed, followed.read_text(encoding="utf-8")
        if isinstance(node, yaml.MappingNode):
            pair = next(((k, v) for k, v in node.value if k.value == seg), None)
            if pair is None:
                raise ValueError(f"{select!r}: {path} has no key {seg!r} there; it has {[k.value for k, _ in node.value]}")
            key, node = pair
        elif isinstance(node, yaml.SequenceNode):
            node = next((item for item in node.value if _item_key(item, path) == seg), None)
            if node is None:
                raise ValueError(
                    f"{select!r}: no item of that list in {path} declares {seg!r} as its id, name, citekey or label"
                )
        else:
            raise ValueError(f"{select!r}: {seg!r} goes below a scalar in {path}")
    if node.tag == "!include":
        path = _include_path(str(node.value), path.parent)
        return path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    start, end = (key or node).start_mark, node.end_mark
    last = end.line if end.line < len(lines) and lines[end.line][: end.column].strip() else end.line - 1
    col = start.column
    out = [lines[start.line][col:]] + [
        ln[col:] if not ln[:col].strip() else ln.strip() for ln in lines[start.line + 1 : last + 1]
    ]
    while out and (not out[-1].strip() or out[-1].lstrip().startswith("#")):
        out.pop()  # PyYAML ends a block at the next token, past the comments that open the next sibling
    return "\n".join(out)


def _follow(node, path: Path):
    """*node*, or the root of the file it includes, and the file it now lives in."""
    from tvbo.utils.yaml_loader import _include_path

    if node.tag != "!include":
        return node, path
    target = _include_path(str(node.value), path.parent)
    return yaml.compose(target.read_text(encoding="utf-8")), target


def _item_key(item, path: Path) -> str | None:
    item, _ = _follow(item, path)
    if not isinstance(item, yaml.MappingNode):
        return None
    keys = {k.value: v.value for k, v in item.value if isinstance(v, yaml.ScalarNode)}
    return next((str(keys[k]) for k in ("id", "name", "citekey", "label") if k in keys), None)


def usage(
    operation: str, *, path: str, experiment=None, backend=None, figure=None, shown: tuple[set, dict] | None = None
) -> str:
    """The Python a reader types for *operation* on the study at *path*, opened by whatever import and load it still owes.

    *shown* is what the snippets before this one left behind, and is updated: the imports written, and which study file the names ``study`` and ``result`` are bound to. A snippet loads its study whenever ``study`` holds another one, and a ``figure`` or ``report`` snippet needs a ``run`` of the same study before it, since ``result`` is what it reads.

    Raises:
        ValueError: A slot the operation needs is unset, or ``result`` holds no run of this study.
    """
    import json

    from tvbo import templates

    imported, bound = shown if shown is not None else (set(), {})
    values = {"experiment": experiment, "backend": backend, "figure": figure}
    missing = [k for k in USAGE[operation] if values[k] is None]
    if missing:
        raise ValueError(f"a {operation} usage needs {missing}")
    if operation in ("figure", "report") and bound.get("result") != path:
        raise ValueError(f"a {operation} usage reads `result`; put a run usage of {path} before it")
    load = bound.get("study") != path
    owed = [k for k in USAGE_IMPORTS if k not in imported and (k == operation or (k == "load" and load))]
    imported.update(owed)
    bound["study"] = path
    if operation == "run":
        bound["result"] = path
    return (
        templates.lookup.get_template("report/tvbo-report-usage.py.mako")
        .render(
            operation=operation,
            imports=owed,
            load=load,
            path=path,
            experiment=json.dumps(experiment),
            backend=backend,
            figure=figure,
        )
        .strip("\n")
    )


def _usage(ctx, block, level) -> str:
    study, _, _ = _target(ctx, block)
    path = Path(os.path.relpath(Path(study._source_file).resolve(), ctx.out_dir)).as_posix()
    code = usage(
        str(block.operation),
        path=path,
        experiment=_experiment_id(study, block) if block.experiment is not None else None,
        backend=block.backend,
        figure=block.figure,
        shown=(ctx.imports, ctx.bound),
    )
    return _fenced("python", code)


def _generated_code(ctx, block, level) -> str:
    study, _, _ = _target(ctx, block)
    code = study.get_experiment(_experiment_id(study, block)).render_code(str(block.backend))
    return _fenced("python", symbol_source(code, str(block.symbol), str(block.backend)))


def symbol_source(code: str, symbol: str, backend: str = "") -> str:
    """The source of the function or class *symbol* (dotted for a method) in the Python *code*, decorators included."""
    try:
        node = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"a symbol is selected from Python, but the {backend} code is not Python ({exc.msg})") from None
    for name in symbol.split("."):
        node = next(
            (
                n
                for n in getattr(node, "body", [])
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
            ),
            None,
        )
        if node is None:
            raise ValueError(f"the generated {backend} code defines no {symbol!r}")
    lines = code.splitlines()
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    return "\n".join(ln[indent:] for ln in lines[start : node.end_lineno])


def _saved(ctx, kind: str, pattern: str) -> Path:
    """The one asset matching *pattern* that an earlier render left for a *kind* block, which shows a run, read back when the render reuses assets instead of having a result."""
    if not ctx.reuse_assets:
        raise ValueError(f"a {kind} block shows a run; render the report from the study's result (`study.run().report(...)`)")
    found = sorted(ctx.assets.glob(pattern)) if ctx.assets else []
    if len(found) != 1:
        raise ValueError(f"expected one {pattern} in {ctx.assets} to reuse, found {[p.name for p in found]}")
    return found[0]


def _figure(ctx, block, level) -> str:
    from tvbo.run.study import figure_still
    from tvbo.utils import as_list

    study, result, label = _target(ctx, block)
    declared = {str(f.name): f for f in as_list(getattr(study, "figures", None))}
    if block.figure:
        name = str(block.figure)
    elif len(declared) == 1:
        name = next(iter(declared))
    else:
        raise ValueError(f"name the figure to show; this study declares {sorted(declared)}")
    stem = "-".join(p for p in ("figure", label, name) if p)
    if result is None:
        target = _saved(ctx, "figure", f"{stem}.*")
    else:
        image = result.figure(name).path
        source = figure_still(declared[name], image) or image
        target = _asset(ctx, f"{stem}{source.suffix}")
        for stale in sorted(ctx.assets.glob(f"{stem}.*")):
            stale.unlink()
        shutil.copyfile(source, target)
    alt = getattr(declared.get(name), "label", None) or name
    return f"![{alt}]({_link(ctx, target)})"


def _methods(ctx, block, level) -> str:
    study, _, _ = _target(ctx, block)
    return study.methods("qmd" if ctx.fmt == "qmd" else "markdown", level=level + 1)


def _catalogue(ctx, block, level) -> str:
    from tvbo.data import registry
    from tvbo.utils.study_layout import _sentence
    from tvbo.utils.yaml_loader import load_as_dict, strip_envelope

    rows = []
    if block.of:
        rows += ["| Class | Curated records |", "|---|--:|"]
        for cls in block.of:
            count = len(registry.list_entries(str(cls)))
            if not count:
                raise ValueError(f"the curated database holds no records of class {cls!r}")
            rows.append(f"| `{cls}` | {count} |")
    if block.entries:
        rows += [""] if rows else []
        rows += ["| Record | Description |", "|---|---|"]
        for iri in block.entries:
            record = strip_envelope(load_as_dict(str(registry.resolve_iri(str(iri)))))
            label = record.get("label") or record.get("title") or record.get("name") or registry.local_name(str(iri))
            description = _sentence(record.get("description"))
            rows.append(f"| {label}<br><sub><code>{iri}</code></sub> | {description + '.' if description else ''} |")
    return "\n".join(rows)


def _backends(ctx, block, level) -> str:
    from tvbo.plot import backends

    if str(block.form or "figure") == "table":
        return backends.markdown()
    return _picture(ctx, "backends", backends.svg, "What each simulation backend can do")


def _tree(ctx, block, level) -> str:
    study, result, label = _target(ctx, block)
    stem = "-".join(p for p in ("tree", label) if p)
    if result is None:
        return _fenced("text", _saved(ctx, "tree", f"{stem}.txt").read_text(encoding="utf-8").rstrip())
    root = Path(result.root)
    text = "\n".join([root.name + "/"] + _tree_lines(root, int(block.depth or 3), ""))
    _asset(ctx, f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    return _fenced("text", text)


def _tree_lines(path: Path, depth: int, prefix: str) -> list[str]:
    if depth <= 0:
        return []
    children = sorted((p for p in path.iterdir() if not p.name.startswith(".")), key=lambda p: (p.is_file(), p.name))
    out = []
    for i, child in enumerate(children):
        last = i == len(children) - 1
        out.append(f"{prefix}{'└── ' if last else '├── '}{child.name}{'/' if child.is_dir() else ''}")
        if child.is_dir():
            out += _tree_lines(child, depth - 1, prefix + ("    " if last else "│   "))
    return out


def _lifecycle(ctx, block, level) -> str:
    from tvbo.plot import lifecycle

    def draw(theme):
        return lifecycle.wheel(theme, band=str(block.band or "icons"), icons=str(block.icons or "line"))

    return _picture(ctx, "lifecycle", draw, "The Simulation Study Lifecycle: Explore, Specify, Generate, Run, Report, Share")


_RENDERERS = {
    "text": _text,
    "spec": _spec,
    "usage": _usage,
    "generated_code": _generated_code,
    "figure": _figure,
    "methods": _methods,
    "catalogue": _catalogue,
    "backends": _backends,
    "tree": _tree,
    "lifecycle": _lifecycle,
}
