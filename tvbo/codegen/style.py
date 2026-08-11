#
# Module: style.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""House style for generated source, one formatter per output language.

Generated code is read, reviewed and attached to papers, so it is held to the same bar as the rest of tvbo. Every backend declares the language it emits (:attr:`~tvbo.export.registry.ExportFormat.language`) and
:func:`tvbo.export.registry.render` routes the assembled source through the matching formatter here — once, centrally, rather than each renderer remembering to do it.

The gate is deliberately a *parse* gate, not only a cosmetic one. Source that a formatter cannot parse is source tvbo would have handed the user as a runnable program, and it fails at import with a worse message than the one raised here. So ``python`` and ``xml`` raise :class:`GeneratedSourceError` rather than passing the text through: an emitter that produces unparseable output has a bug, and silence is what let one live in the JAX templates.

Languages divide by how much a formatter can safely change:

``python``
    Reformatted with ``black``. Its output is canonical, so equality with it is a
    testable contract (see ``tests/test_codegen_style_contract.py``).
``xml``, ``yaml``
    Checked for well-formedness, then normalised only. Pretty-printing is *not*
    applied: ``ElementTree`` drops comments, and re-emitting YAML rewrites quoting
    and key order. Both would change content to fix whitespace.
``julia``, ``c``
    Normalised only. Their real formatters (JuliaFormatter.jl, clang-format) are
    not dependencies of tvbo and would make codegen require a foreign toolchain.
"""

from __future__ import annotations

import re

__all__ = ["GeneratedSourceError", "LANGUAGES", "format_source", "normalize"]


class GeneratedSourceError(ValueError):
    """Raised when generated source does not parse as the language it claims to be."""


def normalize(code: str) -> str:
    """Apply the language-independent house rules to *code*.

    Converts line endings to ``\\n``, strips trailing whitespace from every line, collapses runs of blank lines to at most two, drops leading blank lines, and ends the text with exactly one newline.
    """
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.lstrip("\n").rstrip("\n") + "\n"


def _excerpt(code: str, lineno: int | None, context: int = 2) -> str:
    """Numbered source lines around *lineno*, for a parse-failure message."""
    lines = code.split("\n")
    lineno = lineno or 1
    lo = max(1, lineno - context)
    hi = min(len(lines), lineno + context)
    return "\n".join(f"{'->' if i == lineno else '  '} {i:4d} | {lines[i - 1]}" for i in range(lo, hi + 1))


def _format_python(code: str) -> str:
    """Reformat with black, raising :class:`GeneratedSourceError` if it will not parse.

    The offending line comes from :func:`ast.parse` rather than from black's message, so the excerpt does not depend on how black happens to word a parse failure.
    """
    import ast

    import black

    try:
        return black.format_str(code, mode=black.FileMode())
    except black.InvalidInput as exc:
        try:
            ast.parse(code)
            lineno = None
        except SyntaxError as syn:
            lineno = syn.lineno
        raise GeneratedSourceError(f"generated Python does not parse: {exc}\n{_excerpt(code, lineno)}") from exc


def _validated(code: str, parse, errors, message: str, lineno_of) -> str:
    """Normalise *code*, then check the NORMALISED text parses — never the other way round.

    The normalised text is what gets written, so it is what has to be valid and what a reported line number has to refer to. Checking first also rejects documents that normalisation is about to make valid: an XML declaration counts as one only at the very start of the document, so a template's stray leading newline would fail an export that every reader accepts.
    """
    text = normalize(code)
    try:
        parse(text)
    except errors as exc:
        raise GeneratedSourceError(f"{message}: {exc}\n{_excerpt(text, lineno_of(exc))}") from exc
    return text


def _format_xml(code: str) -> str:
    """Normalise, then check well-formedness. Never re-serialises (comments are content)."""
    from xml.etree import ElementTree

    return _validated(
        code,
        ElementTree.fromstring,
        ElementTree.ParseError,
        "generated XML is not well-formed",
        lambda exc: exc.position[0] if getattr(exc, "position", None) else None,
    )


def _format_yaml(code: str) -> str:
    """Normalise, then check the document parses. Never re-emits (quoting is content)."""
    import yaml

    def _lineno(exc):
        mark = getattr(exc, "problem_mark", None)
        return mark.line + 1 if mark is not None else None

    return _validated(code, lambda t: list(yaml.safe_load_all(t)), yaml.YAMLError, "generated YAML does not parse", _lineno)


_FORMATTERS = {
    "python": _format_python,
    "xml": _format_xml,
    "yaml": _format_yaml,
    "julia": normalize,
    "c": normalize,
}

LANGUAGES = frozenset(_FORMATTERS)
"""Languages with a formatter. A format whose ``language`` is empty is left alone."""


def format_source(code: str, language: str | None) -> str:
    """Return *code* formatted as *language*.

    Args:
        code: Assembled source as rendered by a backend's templates.
        language: One of :data:`LANGUAGES`, or empty/``None`` to leave *code* untouched.

    Raises:
        GeneratedSourceError: *code* does not parse as *language*.
    """
    if not language:
        return code
    formatter = _FORMATTERS.get(language)
    if formatter is None:
        raise ValueError(f"Unknown source language {language!r}. Known: {', '.join(sorted(LANGUAGES))}.")
    return formatter(code)
