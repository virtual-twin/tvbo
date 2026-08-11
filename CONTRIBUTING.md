# Contributing to TVBO

The Virtual Brain Ontology (TVBO) is a community driven project. Here's a step-by-step guide to help you contribute in a meaningful and efficient way.

## Getting Started

### Clone the Repository

First, you'll need to clone the repository to your local machine:

```
git clone https://github.com/virtual-twin/tvbo.git
cd tvbo
pip install -e ".[all]"
pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

The editable install matters: `tvbo/datamodel/` is **generated** from `schema/tvbo_datamodel.yaml` by a build hook and is not tracked, so a fresh clone (or a branch switch) has no datamodel until something triggers a build. Around a hundred import failures in `pytest` almost always mean exactly this.

The last line makes `git blame` skip the commits listed in `.git-blame-ignore-revs`, so a tree-wide reformat does not sit on top of every line's history.

### Work on the Ontology

You can make changes to the ontology using one of the following ways:

- **Protégé Desktop**: Open the ontology file in [Protégé Desktop](https://protege.stanford.edu/).
- **WebProtégé**: Use [WebProtégé](https://webprotege.stanford.edu/) to edit the ontology online.
- **Code Editor**: If you prefer, you can directly work with the OWL file using a code editor that supports OWL syntax.

## Submitting a Pull Request

When you're ready to submit your changes, here's the process to follow:

1. **Stage your changes**: Only stage the changes that are relevant to the data you are adding or the issue you are fixing.

2. **Create a Meaningful Commit**: Commit your changes with a meaningful commit message. This should clearly explain what has been done.

3. **Create a Branch**: Create a branch for your update. This makes it easier for our maintainers to understand and review your contribution. Please give the branch a meaningful way.

4. **Push to Your Fork**: Push the changes to your fork of the repository.

5. **Start a Pull Request (PR)**: Go to the original TVBO repository and click on the "New Pull Request" button. Make sure to describe your changes clearly, why they are necessary, and how they fit into the existing codebase. If you have used references please add them here.

6. **Follow Good PR Practices**:
   - Make sure the PR is concise and only addresses one issue.
   - Include any necessary documentation or tests.
   - Respond to our review feedback - in case it comes up - in a timely manner.

7. **Submit the Pull Request**: Click the "Create pull request" button to submit your PR.

## What is public, and what SemVer covers

From 1.0, TVBO follows [Semantic Versioning](https://semver.org). Three things are the public API, and a breaking change to any of them waits for a major release:

1. **`tvbo.__all__`** — the names `import tvbo` exports. Anything else in the package, whatever its spelling, is internal and may change in a patch.
2. **The schema** — `schema/tvbo_datamodel.yaml`, and so the YAML a recipe is written in. Slots are added freely; a slot is never removed or repurposed without a deprecation cycle, and `aliases:` keeps the old spelling working.
3. **The CLI** — the commands, their options, and the shape of what they write to `output/`.

Not public, and free to change at any time: `tvbo/datamodel/**` (generated — edit the schema instead), any name starting with `_`, the Mako templates and the exact text of generated code, and the contents of `dev/`.

**Deprecations.** A deprecated name keeps working, warns with `DeprecationWarning`, and its message states the version that removes it — at least one full minor of overlap. New deprecations are recorded in `CHANGELOG.md`.

1.0 starts from zero: every name that warned in 0.5.x was removed rather than carried forward, and `tvbo/` contains no `DeprecationWarning` today. That is the state to hold — a deprecation added now is a promise to delete it, not a permanent second spelling.

## Code quality

Three gates run in CI and in `pre-commit`; all three are blocking.

- `ruff check .` — `E`, `F`, `W`, `I`, `UP`, `B`, `D` (Google docstring convention).
- `ruff format --check .`
- `python scripts/check_prose.py` — the house rules a formatter cannot enforce: a standalone `#` run is **at most one line** (longer explanations belong in the docstring of the thing they describe, where quartodoc renders them), docstring prose is not hand-wrapped (`E501` is off, so a paragraph may be one long line), and no commented-out code.

`scripts/unwrap_prose.py --apply` fixes the two mechanical prose rules for you. It is safe to run unread: every rewrite is checked against an invariant — each docstring's whitespace-normalized text must be byte-identical before and after — and a single mismatch aborts the file rather than writing it.

## CI notes

Four jobs behave in ways that are not obvious from `.github/workflows/ci.yml`.

**Skills sync guard.** Canonical skills live in `skills/` (maintainer, repo-only) and `tvbo/skills/canonical/` (user, shipped in the wheel). The rendered copies — `.claude/skills/`, `.github/instructions/`, `AGENTS.md` — are **generated** by `tvbo skills sync`. The guard fails on four things: *drift*, a rendered copy that no longer matches its source (run `tvbo skills sync` and commit the result); an *orphan*, a rendered copy with no canonical source, which is usually a personal skill committed by accident and belongs in `~/.claude/skills/`; a *leak*, a shipped user skill referencing a maintainer skill that `install` never ships, leaving external users a dead pointer; and a *bad extra*, `requires_extras` naming no real optional-dependency group. Only drift is repairable by `sync` — the other three are content problems it reports but never fixes.

**A stacked PR gets no CI at all.** The `pull_request` trigger lists `main` and `dev`, so a PR based on another feature branch never fires it, and the whole stack stays unvalidated until the bottom one retargets. Run it by hand:

```
gh workflow run ci.yml --ref <your-branch>
gh run list --branch <your-branch> --limit 1
```

A dispatched run is the full thing, native backend shards and Julia included — those are gated to `pull_request`, `push` to `main`, and `workflow_dispatch`, and nothing else.

**The lint job installs nothing,** which makes it fast and makes it the one job that sees the repo exactly as a fresh clone does — no `tvbo/datamodel/`, because that is generated. That matters for import sorting: ruff resolves first-party by path, so an unbuilt tree would sort `tvbo.datamodel.*` as third-party while a built one sorts it first-party. `known-first-party = ["tvbo"]` in `pyproject.toml` declares it instead, and must stay. To check a gate the way CI sees it rather than the way your built worktree does, lint an export: `git archive HEAD | tar -x -C "$(mktemp -d)"`.

**Schema validation** runs on every PR because it is fast (~20 s) and needs only `linkml` + `pyyaml`, so schema/database drift surfaces without waiting for the full install matrix.

**Ontology reasoning** (ELK, and HermiT for full OWL-DL) runs ROBOT over the generated `tvb-o-struct.owl` to catch unsatisfiable classes and inverse/functional/cardinality regressions. It is `continue-on-error` while the generated ontology still has known cleanup pending.

**The Julia depot cache** is the subtle one. The Julia *project* (`Project.toml`/`Manifest.toml` under `.venv/julia_env`) and the Julia *depot* (sources and precompiled artifacts under `.julia-depot`) are cached under two independent keys — the `.venv` cache keys on `pyproject.toml` + `ci.yml`, the depot on `juliapkg.json`. A restore can therefore pair a Manifest with a depot that predates it, leaving the Manifest referencing a transitive source the depot lacks; precompile then dies with *"Package X is required but does not seem to be installed"*. A forced `juliapkg` resolve is the one operation guaranteed to reconcile them, and it runs as a plain Julia subprocess without loading PythonCall.jl so it cannot trip over the inconsistency it repairs. Two related rules: the depot is saved **only on success and only when the exact key was absent** (GitHub caches are immutable, so a half-built depot written once would be served forever), and the key carries a `-v2` suffix to escape exactly that situation from before the rule existed.

## Raising an Issue on GitHub

If you encounter a problem, have a question, or want to suggest a new feature, you can raise an issue on the [TVBO GitHub repository](https://github.com/virtual-twin/tvbo/issues). A **security** vulnerability is the exception — report it privately, following [SECURITY.md](SECURITY.md).

1. **Go to the Issues tab**: On the TVBO repository page, click on the "Issues" tab.

2. **Create a New Issue**: Click the "New Issue" button.

3. **Fill in the Issue Details**: Provide a clear title and detailed description of the issue. Include any relevant code snippets, error messages, or screenshots.

4. **Submit the Issue**: Click the "Submit new issue" button.

## Thank you!

Your contributions to the TVBO project are highly appreciated. Following these guidelines will help us collaborate more efficiently. Thank you for your interest and support!
