#
# Module: imports.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Drop the imports a generated module does not use.

A backend template cannot know which imports its output will need: whether
``BoundedSolver`` appears depends on a state variable declaring ``domain.enforce``,
whether ``optax`` appears depends on the experiment carrying an optimization. Writing
that out as one ``% if`` per import means every new feature has to remember to extend a
condition it does not otherwise touch, and the ones already written drift — the tvb and
tvboptim headers between them carried nineteen names no render referenced.

So the templates emit the imports their features *may* need and this pass removes the
ones the assembled module does not reference. The decision is made from the finished
source, which is the only place the answer is actually known.

Imports are pruned **in place**, never hoisted. Order is load-bearing in generated code:
the tvboptim module sets ``JAX_PLATFORMS`` in ``os.environ`` and only then imports jax,
so moving that import above the assignment would silently change which device the
experiment runs on.

The pass is deliberately conservative — it drops a name only when the module cannot
plausibly refer to it, treating a name mentioned inside any string literal as used, so a
name reached by ``getattr`` or an ``eval``-ed expression survives.
"""
from __future__ import annotations

import ast
import re

__all__ = ["prune_unused_imports", "unused_import_names"]


def _bound_names(alias: ast.alias) -> str:
    """The name an ``import`` binds: ``import a.b`` binds ``a``, ``as c`` binds ``c``."""
    return alias.asname or alias.name.split(".")[0]


def _docstrings(tree: ast.AST) -> set[int]:
    """``id()`` of every docstring node, which describes the code rather than running it."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            out.add(id(first.value))
    return out


def _referenced(tree: ast.AST, source: str) -> set[str]:
    """Every identifier the module could be referring to, outside its import statements.

    Attribute chains contribute their root (``jnp`` for ``jnp.exp``), which is the name
    an import binds. Non-docstring string literals contribute every identifier-shaped
    word they contain, so a class reached by name through ``getattr`` keeps its import.
    Docstrings are excluded: prose naming a class is not a use of it, and counting it as
    one is what kept ``AbstractMonitor`` imported into modules that never touch it.

    Decorators, base classes and annotations are already ``Name``/``Attribute`` nodes.
    Comments are not part of the AST, so a name mentioned only in one cannot keep its
    import alive — deliberate, since that is what ``# noqa`` is for.
    """
    names: set[str] = set()
    strings: list[str] = []
    prose = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            strings.append(node.value)
    for text in strings:
        names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return names


def _import_nodes(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    """Top-level and nested import statements, in source order."""
    return sorted(
        (n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
        key=lambda n: (n.lineno, n.col_offset),
    )


def unused_import_names(source: str) -> set[str]:
    """Names *source* imports but never refers to. Empty when *source* does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    used = _referenced(tree, source)
    unused = set()
    for node in _import_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        for alias in node.names:
            if alias.name != "*" and _bound_names(alias) not in used:
                unused.add(_bound_names(alias))
    return unused


def prune_unused_imports(source: str) -> str:
    """Return *source* with unreferenced imports removed, everything else untouched.

    A statement importing several names keeps the ones that are used; a statement whose
    names are all unused is dropped whole. ``from __future__`` imports, star imports and
    lines carrying a ``noqa`` comment are always kept — the first two because dropping
    them changes semantics, the last because it is how a template says an import is
    deliberate.

    Source that does not parse is returned unchanged: reporting a syntax error is
    :func:`tvbo.codegen.style.format_source`'s job, and it gives a better message.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    used = _referenced(tree, source)
    lines = source.split("\n")
    replacements: dict[int, list[str]] = {}

    for node in _import_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if any(a.name == "*" for a in node.names):
            continue
        end = node.end_lineno or node.lineno
        span = "\n".join(lines[node.lineno - 1 : end])
        if "noqa" in span:
            continue
        keep = [a for a in node.names if _bound_names(a) in used]
        if len(keep) == len(node.names):
            continue
        indent = " " * node.col_offset
        if keep:
            kept = ast.Import(names=keep) if isinstance(node, ast.Import) else ast.ImportFrom(
                module=node.module, names=keep, level=node.level
            )
            text = [indent + ast.unparse(ast.fix_missing_locations(kept))]
        else:
            text = []
        replacements[node.lineno - 1] = text
        for i in range(node.lineno, end):
            replacements[i] = []

    if not replacements:
        return source
    out: list[str] = []
    for i, line in enumerate(lines):
        out.extend(replacements[i] if i in replacements else [line])
    return "\n".join(out)
