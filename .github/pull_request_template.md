## What this changes

<!-- One or two sentences. The reviewer reads this before the diff. -->

## Why

<!-- The problem, not the patch. Link the issue if there is one. -->

## How it was verified

<!-- The command you ran and what it printed. "Tests pass" is not verification; `pytest tests/test_x.py -q` → 42 passed is. -->

## Checklist

- [ ] `ruff check .`, `ruff format --check .` and `python scripts/check_prose.py` pass. (Install the pre-commit hooks and they run for you.)
- [ ] Schema changes edit `schema/tvbo_datamodel.yaml`, never the generated `tvbo/datamodel/**`.
- [ ] A backend behaviour change carries a test that would fail without it.
- [ ] User-visible changes are in `CHANGELOG.md` under `## Unreleased`.
- [ ] A new deprecation states the version that removes it.
