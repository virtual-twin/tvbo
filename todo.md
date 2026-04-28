# TODO

## LinkML 1.11.0 release follow-up

**Status:** workaround in place. Triggered by missing PyPI release of LinkML
1.11, which contains the SimpleDict-shorthand fixes we depend on.

### Why the workaround exists

`tvbo/database/models/*.yaml` uses the `Function` class with the SimpleDict
shorthand for `arguments`:

```yaml
H:
  arguments:
    - name: x
  expression: 1 / (1 + exp(-x))
```

LinkML 1.10.0 (current PyPI release) has two bugs that break this:

1. `linkml_runtime.utils.yamlutils._normalize_inlined` raises on
   `[{name: v}]` items for inlined-as-list slots inheriting from
   `Function`.
2. The jsonschema generator emits a schema that rejects the SimpleDict
   shorthand at validation time.

Both are fixed by these commits, all included in tag `v1.11.0-rc1`
(`0fa6f931e9edfc10a6410885e88eed496f6b9249`):

- `a5938c6d3` (runtime: scalar inlined-as-list keys)
- `3f2445d3b` (runtime: kwargs for inherited classes)
- `c3e37be08`, `c43a220ec` (jsonschema generator: SimpleDict support)

### Current pin

We install `linkml` and `linkml-runtime` from the LinkML monorepo at tag
`v1.11.0-rc1`, subdirectories `packages/linkml` and
`packages/linkml_runtime`.

The pin lives in three places:

1. `pyproject.toml` — `[tool.uv.sources]` block (used by `uv sync` /
   `uv pip install` of the project tree).
2. CI workflows — explicit `git+https://...@v1.11.0-rc1` URLs in
   every install step (the `uv pip` interface does not honor
   `[tool.uv.sources]`):
   - `.github/workflows/ci.yml` — `schema-validate`, `schema-artifacts`,
     `reasoner`, `compat` (cache-miss), `test-native` (cache-miss).
   - `.github/workflows/publish-pypi.yml` — `tests` job.
   - `.github/workflows/docker.yml` — `release-ready` wheel verification.
3. Docker images — `docker/Dockerfile` and `docker/Dockerfile.dev`
   install the same git URLs before reading `pyproject.toml`.

### Action items when LinkML 1.11.0 ships on PyPI

1. Bump version floors in `pyproject.toml`:
   - `"linkml-runtime>=1.8.5"` -> `"linkml-runtime>=1.11.0"`
   - `"linkml>=1.8.5"` -> `"linkml>=1.11.0"`
2. Remove the `[tool.uv.sources]` block from `pyproject.toml`.
3. Remove every `git+https://github.com/linkml/linkml.git@v1.11.0-rc1`
   install line from the workflow files listed above. Replace with a
   plain `uv pip install linkml linkml-runtime` (where needed) or rely
   on the version floor through `uv pip install -e .`.
4. Remove the `linkml @ git+...` / `linkml-runtime @ git+...` install
   step from `docker/Dockerfile` and `docker/Dockerfile.dev`.
5. Drop the `# NOTE: ... v1.11.0-rc1` comment blocks added next to each
   workaround.
6. Re-run `make gen-linkml` to confirm no regenerated drift, then
   `pytest tests/test_database_validation.py -q` (must remain 299/299).
7. Delete this section of `todo.md`.

### Verification command after removal

```bash
source .venv/bin/activate
uv pip install -e . --reinstall
pytest tests/test_database_validation.py -q
```
