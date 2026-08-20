"""Inspect the curated unit vocabulary, and promote a unit into it.

TVBO reasons about a unit only when it holds that unit's facts — its scale, its base dimensions, its quantity kind. Those are vendored from QUDT rather than written here, so "curating a unit" means pulling its authoritative record in, not inventing one.

The `unit` slot itself is open: an uncurated unit is recorded as written and carries no dimensional claim, so nothing is blocked while it waits. `add` is what turns it from recorded into reasoned-about.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Inspect and curate the unit vocabulary.", no_args_is_help=True)


def _generator():
    """The vendoring generator, imported from `scripts/` which is not an installed package."""
    import importlib.util
    from pathlib import Path

    for root in (Path(__file__).resolve().parents[2], Path.cwd()):
        script = root / "scripts" / "ontology" / "gen_units.py"
        if script.exists():
            spec = importlib.util.spec_from_file_location("tvbo_gen_units", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise typer.BadParameter(
        "scripts/ontology/gen_units.py not found — `units add` curates the repository's "
        "vendored ontology, so it runs from a source checkout, not an installed wheel."
    )


@app.command("list")
def list_units(
    uncurated: bool = typer.Option(False, "--uncurated", help="Show only units carrying no facts."),
) -> None:
    """List the unit vocabulary with the QUDT facts behind each value."""
    from tvbo.utils.units import _vendored

    rows = _vendored()
    shown = 0
    for name, facts in sorted(rows.items()):
        if not facts.get("curated"):
            typer.echo(f"{name:24s} uncurated — {facts.get('reason', 'no QUDT record')}")
            shown += 1
        elif not uncurated:
            numerator, denominator = facts["multiplier"]
            dimensions = " ".join(
                f"{b}^{n}/{d}" if d != 1 else f"{b}^{n}" for b, (n, d) in sorted(facts["dimensions"].items())
            )
            typer.echo(f"{name:24s} x{numerator}/{denominator:<18} {dimensions or 'dimensionless'}")
            shown += 1
    typer.echo(f"\n{shown} unit(s)")


@app.command("show")
def show(unit: str = typer.Argument(..., help="A UnitEnum value, or any unit string.")) -> None:
    """Print everything TVBO knows about one unit, and say plainly when that is nothing."""
    from tvbo.utils.units import unit_dimensions, unit_expression, unit_facts

    facts = unit_facts(unit)
    if facts is None:
        typer.echo(f"{unit}: uncurated — recorded as written, but carrying no dimensional claim.")
        typer.echo(f"Promote it with:  tvbo units add {unit} --qudt <QUDT-IRI>")
        raise typer.Exit(code=0)

    typer.echo(f"{unit}: {facts.get('description') or ''}".rstrip())
    typer.echo(f"  qudt        {facts.get('qudt') or 'minted from factor units ' + ', '.join(facts['factors'])}")
    typer.echo(f"  scale       x{facts['multiplier'][0]}/{facts['multiplier'][1]}")
    typer.echo(f"  dimensions  {unit_dimensions(unit) or 'dimensionless'}")
    if facts.get("offset"):
        typer.echo(f"  offset      {facts['offset']} (affine — no multiplicative expression)")
    else:
        typer.echo(f"  expression  {unit_expression(unit)}")
    for kind in facts.get("quantity_kinds") or []:
        typer.echo(f"  kind        {kind}")


@app.command("add")
def add(
    unit: str = typer.Argument(..., help="The UnitEnum value to curate."),
    qudt: str | None = typer.Option(None, "--qudt", help="QUDT unit IRI, e.g. NanoSEC."),
    factors: list[str] | None = typer.Option(
        None,
        "--factor",
        help="Factor unit as QUDT_IRI:EXPONENT, repeatable — for a compound QUDT has no IRI for.",
    ),
) -> None:
    """Pull a unit's QUDT record into the vendored ontology.

    Either name the QUDT unit directly, or decompose it into factor units TVBO already vendors. The facts are transcribed or computed from those atoms — nothing about the unit is authored here, which is what keeps the vocabulary authoritative rather than locally parsed.
    """
    if bool(qudt) == bool(factors):
        raise typer.BadParameter("give exactly one of --qudt or one or more --factor")

    generator = _generator()
    if qudt:
        try:
            generator.fetch_qudt(qudt)
        except generator.VendorError as error:
            raise typer.BadParameter(f"{error}. An IRI that resolves to nothing is worse than none.") from error
        entry = (
            f'            {unit}:\n                meaning: qudt:{qudt}\n                description: "TODO — describe {unit}"'
        )
        where = "schema/units.yaml, under the matching section"
        recipe = f"add `{unit}` to UnitEnum with that meaning"
    else:
        decomposition = {}
        for factor in factors:
            atom, _, exponent = factor.partition(":")
            if not exponent:
                raise typer.BadParameter(f"--factor {factor!r} must be QUDT_IRI:EXPONENT")
            decomposition[atom] = int(exponent)
        entry = f"    {unit!r}: {decomposition!r},"
        where = "scripts/ontology/gen_units.py, in DECOMPOSITIONS"
        recipe = f"add `{unit}` to UnitEnum (no `meaning:`) and record its factors"

    typer.echo(f"To curate {unit}, {recipe}:\n")
    typer.echo(f"  in {where}:")
    typer.echo(entry)
    typer.echo("\nthen regenerate and review:\n  make gen-units && git diff ontology/ tvbo/data/ontology/")
