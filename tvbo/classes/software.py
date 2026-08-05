#
# Module: software.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
SimulationTool
==============
Runtime wrapper around the auto-generated ``SimulationTool`` dataclass, adding
the factory constructors the other database entities have and the lookup that
answers *how does this tool write this unit*.

A tool's unit vocabulary is a fact about the tool. LEMS calls ``Hz`` a
``per_time`` quantity and writes it ``per_s``; nothing else need agree, and the
same unit means the same thing regardless. Keeping the vocabulary on the tool's
own database entry is what lets a second backend declare its own without
touching a shared table — and what retired ``PhysicalDimension``, which was
LEMS's dimension names restated in TVBO's schema as though they were universal.

Usage
-----
>>> from tvbo.classes.software import SimulationTool
>>> lems = SimulationTool.for_format("neuroml")
>>> lems.name
'LEMS'
>>> lems.dimension_of("mV"), lems.symbol_of("Hz")
('voltage', 'per_s')
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.utils import yaml_loader

_CODEGEN_FORMAT_RE = re.compile(r"^codegen_format:\s*(\S+)\s*$", re.MULTILINE)


class SimulationTool(tvbo_datamodel.SimulationTool):
    """A software tool's database entry, with its code-generation capabilities."""

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "SimulationTool":
        """Load a SimulationTool from a YAML file."""
        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_db(cls, name: str) -> "SimulationTool":
        """Load a SimulationTool by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("SimulationTool", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available software entries in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("SimulationTool")

    @classmethod
    def for_format(cls, format_key: str) -> "SimulationTool | None":
        """The tool TVBO emits *format_key* for, or `None` if no entry claims it.

        `format_key` is an export-registry key (`tvbo formats`). `None` rather than
        a raise: most formats have no tool entry, and a caller asking how a tool
        spells a unit wants the same answer — "it does not say" — whether the entry
        is absent or merely silent on that unit.
        """
        path = _format_index().get(format_key.lower())
        return None if path is None else cls.from_file(path)

    def dimension_of(self, unit) -> str:
        """This tool's name for the dimension of *unit*, or `"none"`.

        `"none"` is LEMS's own spelling of dimensionless, and is what a tool that
        has no dimension for the unit must be told.
        """
        entry = self._unit_entry(unit)
        return getattr(entry, "dimension", None) or "none"

    def symbol_of(self, unit) -> str:
        """This tool's spelling of *unit* itself, or `""` to write the value bare."""
        entry = self._unit_entry(unit)
        return getattr(entry, "symbol", None) or ""

    def _unit_entry(self, unit):
        if unit is None or not self.units:
            return None
        return self.units.get(str(getattr(unit, "text", unit)).strip())


@lru_cache(maxsize=1)
def _format_index() -> dict[str, str]:
    """Export-format key → path of the software entry declaring it.

    Read straight from the file text rather than by loading all 60-odd entries
    through LinkML, which would cost seconds to answer a question about one line.
    """
    from tvbo.data.registry import database_dir

    index = {}
    for path in sorted(database_dir("SimulationTool").rglob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        match = _CODEGEN_FORMAT_RE.search(path.read_text(encoding="utf-8"))
        if match:
            index[match.group(1).lower()] = str(path)
    return index
