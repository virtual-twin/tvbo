"""``tvbo skills`` — manage TVBO skill files for AI coding assistants.

Three subcommands:

* ``tvbo skills sync``       — repo-side: regenerate ``.claude/skills/``,
  ``.github/instructions/`` and the index region of ``AGENTS.md`` from the
  canonical sources in ``skills/`` and ``tvbo/skills/canonical/``.
* ``tvbo skills install``    — user-side: render canonical user skills onto
  a user's machine (``~/.claude/skills/``, ``./.cursor/rules/``, stdout, …).
* ``tvbo skills uninstall``  — user-side: remove files previously written by
  ``install`` (detected via the ``managed-by: tvbo`` frontmatter marker).

See :mod:`tvbo.skills._render` for the canonical schema.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

import tvbo
from tvbo.skills import (
    CANONICAL_PACKAGE_DIR,
    CANONICAL_REPO_DIR,
    Skill,
    is_managed_file,
    load_canonical,
    render_agents_md,
    render_claude_code,
    render_copilot,
    render_cursor,
    render_prompt,
)

app = typer.Typer(name="skills", no_args_is_help=True)


class Target(str, Enum):
    claude_code = "claude-code"
    copilot = "copilot"
    cursor = "cursor"
    agents_md = "agents-md"
    prompt = "prompt"
    all = "all"


class Audience(str, Enum):
    user = "user"
    maintainer = "maintainer"
    all = "all"


class Scope(str, Enum):
    user = "user"
    project = "project"


# --------------------------------------------------------------------------- helpers


def _filter_by_audience(skills: list[Skill], audience: Audience) -> list[Skill]:
    if audience is Audience.all:
        return skills
    wanted = audience.value  # "user" | "maintainer"
    return [s for s in skills if s.audience in {wanted, "both"}]


def _filter_by_name(skills: list[Skill], names: list[str] | None) -> list[Skill]:
    if not names:
        return skills
    wanted = set(names)
    return [s for s in skills if s.name in wanted or s.install_name in wanted]


def _scope_dir(target: Target, scope: Scope) -> Path:
    if target is Target.claude_code:
        if scope is Scope.user:
            return Path.home() / ".claude" / "skills"
        return Path.cwd() / ".claude" / "skills"
    if target is Target.cursor:
        if scope is Scope.user:
            return Path.home() / ".cursor" / "rules"
        return Path.cwd() / ".cursor" / "rules"
    raise typer.BadParameter(f"--scope does not apply to target {target.value!r}")


def _existing_blocks_install(path: Path, force: bool) -> bool:
    """Return True if *path* exists and is not safe to overwrite."""
    if not path.exists():
        return False
    if force:
        return False
    return not is_managed_file(path)


# --------------------------------------------------------------------------- sync


@app.command("sync", help="Regenerate adapter files from canonical sources (maintainer).")
def sync(
    check: bool = typer.Option(
        False, "--check", help="Exit non-zero if any adapter file would change. CI guard."
    ),
    repo_root: Path = typer.Option(
        Path.cwd(), "--repo-root", help="Repo root containing skills/ and AGENTS.md."
    ),
) -> None:
    """Regenerate ``.claude/skills/``, ``.github/instructions/`` and ``AGENTS.md``."""
    skills = load_canonical([repo_root / CANONICAL_REPO_DIR, CANONICAL_PACKAGE_DIR])
    if not skills:
        typer.echo("no canonical skills found", err=True)
        raise typer.Exit(1)

    claude_dir = repo_root / ".claude" / "skills"
    copilot_dir = repo_root / ".github" / "instructions"
    agents_md = repo_root / "AGENTS.md"

    if check:
        return _sync_check(skills, claude_dir, copilot_dir, agents_md)

    written: list[Path] = []
    for skill in skills:
        written.append(render_claude_code(skill, claude_dir))
        # Copilot files only make sense in-repo for maintainer-tagged skills.
        if skill.audience in {"maintainer", "both"}:
            written.append(render_copilot(skill, copilot_dir))
    written.append(render_agents_md(skills, agents_md))

    typer.echo(f"synced {len(skills)} skills → {len(written)} files")


def _sync_check(skills: list[Skill], claude_dir: Path, copilot_dir: Path, agents_md: Path) -> None:
    """Render in-memory, compare to on-disk; non-zero exit if anything differs."""
    import io
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for skill in skills:
            render_claude_code(skill, tmp_path / "claude")
            if skill.audience in {"maintainer", "both"}:
                render_copilot(skill, tmp_path / "copilot")
        if agents_md.exists():
            shutil.copy(agents_md, tmp_path / "AGENTS.md")
        render_agents_md(skills, tmp_path / "AGENTS.md")

        drift: list[str] = []

        def _cmp(a: Path, b: Path, label: str) -> None:
            ax = a.read_text(encoding="utf-8") if a.exists() else ""
            bx = b.read_text(encoding="utf-8") if b.exists() else ""
            if ax != bx:
                drift.append(label)

        for skill in skills:
            _cmp(
                tmp_path / "claude" / skill.name / "SKILL.md",
                claude_dir / skill.name / "SKILL.md",
                f".claude/skills/{skill.name}/SKILL.md",
            )
            if skill.audience in {"maintainer", "both"}:
                _cmp(
                    tmp_path / "copilot" / f"{skill.name}.instructions.md",
                    copilot_dir / f"{skill.name}.instructions.md",
                    f".github/instructions/{skill.name}.instructions.md",
                )
        _cmp(tmp_path / "AGENTS.md", agents_md, "AGENTS.md")

    if drift:
        typer.echo("skills sync drift detected:", err=True)
        for d in drift:
            typer.echo(f"  - {d}", err=True)
        typer.echo("\nRun `tvbo skills sync` to regenerate.", err=True)
        raise typer.Exit(1)
    typer.echo("skills sync: up to date")


# --------------------------------------------------------------------------- install


@app.command("install", help="Install user skills onto your machine.")
def install(
    target: Target = typer.Option(Target.claude_code, "--target", "-t", help="Adapter format."),
    audience: Audience = typer.Option(
        Audience.user, "--audience", help="Which skills to install."
    ),
    scope: Scope = typer.Option(
        Scope.user, "--scope", help="user (~/.claude/) or project (./.claude/)."
    ),
    skill: list[str] = typer.Option(
        None, "--skill", "-s", help="Limit to specific skill(s). Repeatable."
    ),
    out: Path = typer.Option(
        None, "--out", "-o", help="Override the install directory or output path."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite unmanaged files."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen; write nothing."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if any skill is refused/skipped."
    ),
) -> None:
    """Render user skills to ``--target`` and write them to disk (or stdout)."""
    skills = load_canonical([CANONICAL_PACKAGE_DIR])
    skills = _filter_by_audience(skills, audience)
    skills = _filter_by_name(skills, skill)
    if not skills:
        typer.echo("no skills match the given filters", err=True)
        raise typer.Exit(0 if not strict else 1)

    version = tvbo.__version__

    if target is Target.prompt:
        text = render_prompt(skills)
        if out is None:
            typer.echo(text)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            typer.echo(f"wrote {out} ({len(skills)} skills)")
        return

    if target is Target.agents_md:
        dest = out or (Path.cwd() / "AGENTS.md")
        if dry_run:
            typer.echo(f"would update skills-index region in {dest}")
            return
        render_agents_md(skills, dest)
        typer.echo(f"updated skills index in {dest}")
        return

    if target is Target.all:
        targets = [Target.claude_code, Target.cursor, Target.agents_md]
    else:
        targets = [target]

    counts = {"installed": 0, "updated": 0, "refused": 0, "skipped": 0}
    refused_paths: list[Path] = []

    for t in targets:
        dest_dir = out or _scope_dir(t, scope) if t in {Target.claude_code, Target.cursor} else out
        if t is Target.agents_md:
            agents_dest = out or (Path.cwd() / "AGENTS.md")
            if not dry_run:
                render_agents_md(skills, agents_dest)
            counts["updated"] += 1
            continue

        for s in skills:
            target_path = _target_path(t, dest_dir, s)
            existed = target_path.exists()
            if _existing_blocks_install(target_path, force):
                counts["refused"] += 1
                refused_paths.append(target_path)
                typer.echo(f"  refused (not managed-by tvbo): {target_path}", err=True)
                continue
            if dry_run:
                action = "would update" if existed else "would install"
                typer.echo(f"  {action}: {target_path}")
                counts["installed" if not existed else "updated"] += 1
                continue
            _render_target(t, s, dest_dir, version)
            counts["updated" if existed else "installed"] += 1

    typer.echo(
        "skills install: "
        f"installed={counts['installed']} updated={counts['updated']} "
        f"refused={counts['refused']} skipped={counts['skipped']}"
    )
    if refused_paths:
        typer.echo("  (pass --force to overwrite, or remove the files manually)", err=True)
    if strict and counts["refused"]:
        raise typer.Exit(1)


def _target_path(target: Target, dest_dir: Path, skill: Skill) -> Path:
    if target is Target.claude_code:
        return dest_dir / skill.install_name / "SKILL.md"
    if target is Target.cursor:
        return dest_dir / f"{skill.install_name}.mdc"
    raise typer.BadParameter(f"no target-path mapping for {target.value!r}")


def _render_target(target: Target, skill: Skill, dest_dir: Path, version: str) -> Path:
    if target is Target.claude_code:
        return render_claude_code(
            skill, dest_dir, managed=True, tvbo_version=version, use_install_name=True
        )
    if target is Target.cursor:
        return render_cursor(
            skill, dest_dir, managed=True, tvbo_version=version, use_install_name=True
        )
    raise typer.BadParameter(f"cannot render to {target.value!r}")


# --------------------------------------------------------------------------- uninstall


@app.command("uninstall", help="Remove skills previously written by `tvbo skills install`.")
def uninstall(
    target: Target = typer.Option(Target.claude_code, "--target", "-t", help="Adapter format."),
    scope: Scope = typer.Option(Scope.user, "--scope", help="user or project."),
    out: Path = typer.Option(None, "--out", "-o", help="Override the install directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed."),
) -> None:
    """Remove files in *target* whose frontmatter carries ``managed-by: tvbo``."""
    if target in {Target.prompt, Target.agents_md}:
        typer.echo(f"uninstall is not meaningful for target {target.value!r}", err=True)
        raise typer.Exit(2)

    if target is Target.all:
        targets = [Target.claude_code, Target.cursor]
    else:
        targets = [target]

    removed = 0
    for t in targets:
        base = out or _scope_dir(t, scope)
        if not base.exists():
            continue
        if t is Target.claude_code:
            candidates = list(base.glob("*/SKILL.md"))
        else:  # cursor
            candidates = list(base.glob("*.mdc"))
        for path in candidates:
            if not is_managed_file(path):
                continue
            if dry_run:
                typer.echo(f"  would remove: {path}")
            else:
                path.unlink()
                # for claude-code, also drop the now-empty parent dir
                if t is Target.claude_code:
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
                typer.echo(f"  removed: {path}")
            removed += 1
    typer.echo(f"skills uninstall: {removed} files {'would be ' if dry_run else ''}removed")
