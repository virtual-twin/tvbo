"""``tvbo doctor`` — check the installed frameworks still provide every name TVBO's generated code reaches."""

from __future__ import annotations

import typer


def doctor(
    tool: list[str] = typer.Option(None, "--tool", "-t", help="Check only this software entry; repeat for several."),
    strict: bool = typer.Option(False, "--strict", help="Fail on a deprecated name too, as the weekly drift job does."),
) -> None:
    """Resolve each software entry's `interfaces:` against the installed package, and exit non-zero on a name it no longer provides.

    A name the installed version documents as deprecated is reported, with its replacement; it fails the check only under `--strict`, since the generated code still runs. A package that is not installed, or is a Julia package, is reported as unchecked.
    """
    from tvbo.codegen.inventory import CALL, MISSING, UNCHECKED, EmittedSymbol, resolve, software_entries

    failed = False
    for _, record in software_entries():
        interfaces = record.get("interfaces") or {}
        if not interfaces or (tool and record.get("name") not in tool):
            continue
        counts = {"present": 0, "deprecated": 0, "missing": 0, "unchecked": 0}
        lines, versions = [], set()
        modules = record.get("modules") or []
        for name in interfaces:
            head = name.split("(")[0].split(".")[0]
            resolution = resolve(EmittedSymbol(head if head in modules else "|".join(modules), name, CALL))
            versions.add(resolution.version)
            if resolution.status == MISSING:
                counts["missing"] += 1
                failed = True
                lines.append(f"  missing     {name}: {resolution.detail}")
            elif resolution.status == UNCHECKED:
                counts["unchecked"] += 1
            elif resolution.deprecated:
                counts["deprecated"] += 1
                failed = failed or strict
                replacement = f", use `{resolution.replaced_by}`" if resolution.replaced_by else ""
                lines.append(f"  deprecated  {name}{replacement}")
            else:
                counts["present"] += 1
        checked = {str(recorded.get("checked_version")) for recorded in interfaces.values() if recorded.get("checked_version")}
        installed = ", ".join(sorted(v for v in versions if v))
        summary = ", ".join(f"{n} {status}" for status, n in counts.items() if n)
        against = ", ".join(sorted(checked)) or "none"
        state = f"installed {installed}, recorded against {against}" if installed else "not reachable from Python"
        typer.echo(f"{record.get('name')} ({state}): {summary}")
        for line in lines:
            typer.echo(line)
    if failed:
        raise typer.Exit(code=1)
