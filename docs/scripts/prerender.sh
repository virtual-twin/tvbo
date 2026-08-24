#!/bin/bash
# Pre-render script for Quarto docs.
# Set TVBO_SKIP_PRERENDER=1 to skip (used by `make docs-preview` after initial render).

if [ "$TVBO_SKIP_PRERENDER" = "1" ]; then
    echo "Skipping pre-render"
    exit 0
fi

set -e

# Ensure we're in the docs/ directory (Quarto sets this automatically,
# but manual invocation may not).
cd "$(dirname "$0")/.."

# ── Sync master bibliography from iCloud ──
BIB_SRC="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Literature/TVBO.bib"
if [[ -f "$BIB_SRC" ]]; then
    rsync -avh "$BIB_SRC" references.bib
    echo "[pre-render] bib-sync OK"
else
    echo "[pre-render][WARN] source bib not found: $BIB_SRC" >&2
fi

# Replication results are embargoed until the source work is published: locally the gate only warns about `publish: false` drafts so they can be worked on, while CI and any deploy job run --strict so a withheld study fails the build instead of reaching a published site. A page missing `publish:` always hard-fails.
REPL_GATE_ARGS=""
if [ -n "$CI" ] || [ "$TVBO_DOCS_STRICT" = "1" ]; then
    REPL_GATE_ARGS="--strict"
    echo "[pre-render] replication gate: strict (CI)"
fi

python scripts/tvbo_package_struct.py
python -m quartodoc build --config api/_quartodoc_config.yml
python scripts/update_toc_api.py
python scripts/update_toc_replication.py $REPL_GATE_ARGS
python scripts/generate_datamodel_docs.py
python scripts/update_toc_datamodel.py
