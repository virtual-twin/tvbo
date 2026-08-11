"""``tvbo skills`` — manage TVBO skill files for AI coding assistants.

Three subcommands:

* ``tvbo skills sync``       — repo-side: regenerate ``.claude/skills/``,
  ``.github/instructions/`` and the index region of ``AGENTS.md`` from the canonical sources in ``skills/`` and ``tvbo/skills/canonical/``.
* ``tvbo skills install``    — user-side: render canonical user skills onto
  a user's machine (``~/.claude/skills/``, ``./.cursor/rules/``, stdout, …).
* ``tvbo skills uninstall``  — user-side: remove files previously written by
  ``install`` (detected via the ``managed-by: tvbo`` frontmatter marker).

See :mod:`tvbo.skills._render` for the canonical schema.
"""

from __future__ import annotations

import re
import shutil
import tomllib
from enum import StrEnum
from pathlib import Path

import typer

import tvbo
from tvbo.skills import (
    CANONICAL_PACKAGE_DIR,
    CANONICAL_REPO_DIR,
    Skill,
    asset_refs,
    is_asset_noise,
    is_managed_file,
    load_canonical,
    render_agents_md,
    render_claude_code,
    render_copilot,
    render_cursor,
    render_prompt,
)

app = typer.Typer(name="skills", no_args_is_help=True)


class Target(StrEnum):
    """Output format a skill is rendered to.

    Selects which assistant-specific artifact `install` or `sync` produces:
    Claude Code skill files, GitHub Copilot instructions, Cursor rules, the `AGENTS.md` index, a plain prompt on stdout, or `all` of them at once.
    """

    claude_code = "claude-code"
    copilot = "copilot"
    cursor = "cursor"
    agents_md = "agents-md"
    prompt = "prompt"
    all = "all"


class Audience(StrEnum):
    """Intended reader of a skill, used to filter which skills apply.

    A skill declares an `audience` of `user`, `maintainer`, or `both`;
    this enum selects the subset to act on, where `all` keeps every skill regardless of its declared audience.
    """

    user = "user"
    maintainer = "maintainer"
    all = "all"


class Scope(StrEnum):
    """Install location for rendered skill files.

    `user` writes to the per-user config directory (e.g. `~/.claude/skills`), while `project` writes to the current working directory (e.g.
    `./.claude/skills`).
    """

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
    check: bool = typer.Option(False, "--check", help="Exit non-zero if any adapter file would change. CI guard."),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", help="Repo root containing skills/ and AGENTS.md."),
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
        return _sync_check(skills, claude_dir, copilot_dir, agents_md, repo_root)

    written: list[Path] = []
    for skill in skills:
        written.append(render_claude_code(skill, claude_dir))
        # Copilot files only make sense in-repo for maintainer-tagged skills.
        if skill.audience in {"maintainer", "both"}:
            written.append(render_copilot(skill, copilot_dir))
    written.append(render_agents_md(skills, agents_md))

    typer.echo(f"synced {len(skills)} skills → {len(written)} files")

    # Warn, never repair: an unrecognised file may be someone's local work, and a bad reference or extra is a content decision the maintainer has to make.
    _report(_lint(skills, claude_dir, copilot_dir, repo_root))


def _references(body: str, name: str) -> bool:
    """True if *body* points at the skill *name*.

    Matches the three forms the corpus actually uses — ``**name**``, ``/name``, and ``` `name` skill ``` — rather than a bare substring, so a skill named ``git`` is not matched by every mention of ``gitignore``.
    """
    n = re.escape(name)
    return any(re.search(p, body) for p in (rf"\*\*{n}\*\*", rf"/{n}\b", rf"`{n}`\s+skill"))


def _find_leaked_refs(skills: list[Skill]) -> list[str]:
    """Shipped user skills that point at maintainer skills.

    ``install`` renders only the user root, so a maintainer skill named in a shipped body is a dead pointer for everyone outside this repo. It looks fine here because ``sync`` renders both audiences side by side.
    """
    maintainer = {s.name for s in skills if s.audience == "maintainer"}
    return [
        f"{skill.source}: references maintainer skill {name!r}"
        for skill in skills
        if skill.audience != "maintainer"
        for name in sorted(maintainer)
        if _references(skill.body, name)
    ]


def _find_dead_asset_refs(skills: list[Skill]) -> list[str]:
    """``assets/…`` paths a body points at that do not exist on disk.

    Same class as :func:`_find_leaked_refs`: a pointer the reader cannot follow. It matters most for a body that *defers* detail to a reference chapter, where a dead pointer silently drops that content instead of erroring.
    """
    return [
        f"{skill.source}: references missing asset {ref!r}"
        for skill in skills
        for ref in asset_refs(skill.body)
        if skill.assets_dir is None or not (skill.assets_dir / ref).exists()
    ]


def _find_bad_extras(skills: list[Skill], repo_root: Path) -> list[str]:
    """``requires_extras`` entries naming no real optional-dependency group."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    real = set((data.get("project") or {}).get("optional-dependencies") or {})
    return [
        f"{skill.source}: requires_extras {extra!r} is not a group in [project.optional-dependencies]"
        for skill in skills
        for extra in skill.requires_extras
        if extra not in real
    ]


def _lint(skills: list[Skill], claude_dir: Path, copilot_dir: Path, repo_root: Path) -> list[tuple[str, list[str], str]]:
    """Content problems that re-rendering cannot fix, as (title, items, hint).

    Distinct from drift: these live in the canonical sources, or in what surrounds the rendered output, so they are reported for a human to resolve rather than silently repaired.
    """
    findings: list[tuple[str, list[str], str]] = []
    if stray := _find_orphans(skills, claude_dir, copilot_dir):
        findings.append(
            (
                "orphaned rendered skills (no canonical source):",
                stray,
                "Either add a canonical source under skills/ (maintainer) or "
                "tvbo/skills/canonical/ (user), or delete the rendered file. "
                "Personal skills belong in ~/.claude/skills/, not the project dir.",
            )
        )
    if leaks := _find_leaked_refs(skills):
        findings.append(
            (
                "shipped user skills referencing maintainer skills:",
                leaks,
                "`tvbo skills install` ships only the user root, so these pointers are "
                "dead outside this repo. Inline the guidance, or drop the reference.",
            )
        )
    if dead := _find_dead_asset_refs(skills):
        findings.append(
            (
                "skills referencing assets that do not exist:",
                dead,
                "Add the file under the skill's assets/, or drop the reference. A body that "
                "defers detail to a missing chapter loses it silently.",
            )
        )
    if bad := _find_bad_extras(skills, repo_root):
        findings.append(
            (
                "requires_extras naming no real optional-dependency group:",
                bad,
                "Name a group from [project.optional-dependencies] in pyproject.toml, or use [] if the skill needs no extra.",
            )
        )
    return findings


def _report(findings: list[tuple[str, list[str], str]]) -> None:
    """Print lint findings. Never deletes — removal is the maintainer's call."""
    for title, items, hint in findings:
        typer.echo(title, err=True)
        for item in items:
            typer.echo(f"  - {item}", err=True)
        typer.echo(f"\n{hint}\n", err=True)


def _find_orphans(skills: list[Skill], claude_dir: Path, copilot_dir: Path) -> list[str]:
    """Rendered skill files that no canonical source accounts for.

    The rendered directories hold generated output only, so anything in them without a canonical source behind it is a stray — typically a personal skill committed by accident. Those belong in a user-scope directory (``~/.claude/skills/``) instead.
    """
    claude_known = {s.name for s in skills}
    copilot_known = {s.name for s in skills if s.audience in {"maintainer", "both"}}

    stray = [
        f".claude/skills/{md.parent.name}/"
        for md in sorted(claude_dir.glob("*/SKILL.md"))
        if md.parent.name not in claude_known
    ]
    stray += [
        f".github/instructions/{inst.name}"
        for inst in sorted(copilot_dir.glob("*.instructions.md"))
        if inst.name.removesuffix(".instructions.md") not in copilot_known
    ]
    return stray


def _sync_check(
    skills: list[Skill],
    claude_dir: Path,
    copilot_dir: Path,
    agents_md: Path,
    repo_root: Path,
) -> None:
    """Render in-memory, compare to on-disk; non-zero exit if anything differs.

    Two independent failure modes: *drift*, where a committed copy no longer matches what its canonical source renders to (fix by re-running sync), and the :func:`_lint` findings, which sync cannot fix.

    ``.claude/skills/`` is a local render and is not committed — the skills live in ``skills/`` and ``tvbo/skills/canonical/`` — so it is gated only where it exists, which is exactly where a stale copy could mislead someone.
    """
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

        def _cmp_tree(a: Path, b: Path, label: str) -> None:
            """Compare two directory trees byte-for-byte, recording per-file drift.

            Byte-compiled and OS-noise files are ignored on both sides (they are never rendered), so a locally-run asset script cannot spoof drift.
            """

            def _files(root: Path) -> dict[Path, bytes]:
                if not root.exists():
                    return {}
                return {
                    rel: p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file() and not is_asset_noise(rel := p.relative_to(root))
                }

            af, bf = _files(a), _files(b)
            for rel in sorted(set(af) | set(bf), key=str):
                if af.get(rel) != bf.get(rel):
                    drift.append(f"{label}/{rel}")

        check_claude = claude_dir.exists()
        for skill in skills:
            if check_claude:
                _cmp(
                    tmp_path / "claude" / skill.name / "SKILL.md",
                    claude_dir / skill.name / "SKILL.md",
                    f".claude/skills/{skill.name}/SKILL.md",
                )
                _cmp_tree(
                    tmp_path / "claude" / skill.name / "assets",
                    claude_dir / skill.name / "assets",
                    f".claude/skills/{skill.name}/assets",
                )
            if skill.audience in {"maintainer", "both"}:
                _cmp(
                    tmp_path / "copilot" / f"{skill.name}.instructions.md",
                    copilot_dir / f"{skill.name}.instructions.md",
                    f".github/instructions/{skill.name}.instructions.md",
                )
        _cmp(tmp_path / "AGENTS.md", agents_md, "AGENTS.md")

    findings = _lint(skills, claude_dir, copilot_dir, repo_root)
    if drift:
        findings.insert(
            0,
            ("skills sync drift detected:", drift, "Run `tvbo skills sync` to regenerate."),
        )
    if findings:
        _report(findings)
        raise typer.Exit(1)
    typer.echo("skills sync: up to date")


# --------------------------------------------------------------------------- install


@app.command("install", help="Install user skills onto your machine.")
def install(
    target: Target = typer.Option(Target.claude_code, "--target", "-t", help="Adapter format."),
    audience: Audience = typer.Option(Audience.user, "--audience", help="Which skills to install."),
    scope: Scope = typer.Option(Scope.user, "--scope", help="user (~/.claude/) or project (./.claude/)."),
    skill: list[str] = typer.Option(None, "--skill", "-s", help="Limit to specific skill(s). Repeatable."),
    out: Path = typer.Option(None, "--out", "-o", help="Override the install directory or output path."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite unmanaged files."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen; write nothing."),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if any skill is refused/skipped."),
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
        return render_claude_code(skill, dest_dir, managed=True, tvbo_version=version, use_install_name=True)
    if target is Target.cursor:
        return render_cursor(skill, dest_dir, managed=True, tvbo_version=version, use_install_name=True)
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
            assets = path.parent / "assets"  # only meaningful for claude-code
            if dry_run:
                typer.echo(f"  would remove: {path}")
                if t is Target.claude_code and assets.is_dir():
                    typer.echo(f"  would remove: {assets}/")
            else:
                path.unlink()
                # for claude-code, also drop our assets/ and the now-empty parent dir
                if t is Target.claude_code:
                    if assets.is_dir():
                        shutil.rmtree(assets)
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
                typer.echo(f"  removed: {path}")
            removed += 1
    typer.echo(f"skills uninstall: {removed} files {'would be ' if dry_run else ''}removed")
