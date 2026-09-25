# Security Policy

## Supported versions

Security fixes land on the latest minor release. Older minors are not patched — upgrade to the current `1.x` before reporting.

| Version | Supported |
|---|---|
| 1.x | yes |
| < 1.0 | no |

## Reporting a vulnerability

Report privately, not as a public issue:

- **Preferred** — [open a draft advisory](https://github.com/virtual-twin/tvbo/security/advisories/new) on this repository.
- **Alternative** — email <l.martin@brainmodes.com> and <petra.ritter@charite.de>.

Tell us what you can reproduce: the affected version, the shortest input that triggers it (a recipe YAML, a network file, a CLI invocation), what happens, and what you expected. A proof of concept is welcome but not required to report.

We aim to acknowledge within five working days and to state a fix or a rejection within thirty. You will be credited in the advisory unless you ask otherwise.

## Scope

TVBO reads simulation recipes and writes generated code. Two consequences are by design, not vulnerabilities:

- **A recipe is executable.** `code_source`, `module:` callables and symbolic expressions all become code that runs in your interpreter, so a recipe from an untrusted source has the same reach as a script from one. Read it first.
- **Generated code is written to disk and executed** by `tvbo run`. The generator's output is only as trustworthy as its input recipe.

In scope: anything that escapes those two — a path traversal out of an output directory, code execution from data that is not a recipe (an HDF5 network, a BIDS sidecar, an ontology file), a dependency pin with a known advisory, or a credential written to a log or an emitted artifact.
