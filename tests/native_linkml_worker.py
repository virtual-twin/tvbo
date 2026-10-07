"""Validate instances with LinkML's own closed validator in a fresh interpreter, for ``test_native_linkml``.

Reads ``{"schema": path, "cases": [[id, source, target_class], ...]}`` as JSON on stdin, where ``source`` is ``{"path": ...}`` (a YAML file, loaded here the way ``linkml-validate`` loads it) or ``{"instance": ...}``, and writes ``{id: [message, ...]}`` on stdout. It runs as its own process because LinkML's enums become unhashable once ``tvbo`` is imported anywhere in the interpreter, and a test process cannot promise that it never was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Validate every case on stdin and print the messages per case id."""
    import yaml
    from linkml.validator import Validator
    from linkml.validator.plugins import JsonschemaValidationPlugin
    from linkml_runtime.utils.schemaview import SchemaView

    request = json.load(sys.stdin)
    validator = Validator(SchemaView(request["schema"]).schema, validation_plugins=[JsonschemaValidationPlugin(closed=True)])
    results = {}
    for case_id, source, target_class in request["cases"]:
        instance = yaml.safe_load(Path(source["path"]).read_text()) if "path" in source else source["instance"]
        results[case_id] = [r.message for r in validator.iter_results(instance or {}, target_class)]
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
