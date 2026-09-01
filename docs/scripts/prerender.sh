#!/bin/bash
# Pre-render script for Quarto docs. TVBO_SKIP_PRERENDER=1 skips it entirely (used by `make docs-preview` after the initial render).

if [ "$TVBO_SKIP_PRERENDER" = "1" ]; then
    echo "Skipping pre-render"
    exit 0
fi

set -e

# Quarto sets this, manual invocation may not.
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

# Regenerating the two references costs more than the rest of the build together, so each is built only when this render can reach it. Their _toc.yml regions are committed and survive a skip, which is what lets a narrow render still show a complete sidebar.
BUILD_API=1
BUILD_DATAMODEL=1
case "${QUARTO_PROFILE:-}" in
    guide) BUILD_API=0; BUILD_DATAMODEL=0 ;;
    api) BUILD_DATAMODEL=0 ;;
    datamodel) BUILD_API=0 ;;
esac
echo "[pre-render] profile: ${QUARTO_PROFILE:-full} (api=$BUILD_API datamodel=$BUILD_DATAMODEL)"

if [ "$BUILD_API" = 1 ]; then
    if ! python -c "import quartodoc" 2>/dev/null; then
        echo "FATAL: quartodoc is required to build the API reference. Install the docs extra: pip install -e '.[docs]'." >&2
        exit 1
    fi
    python scripts/tvbo_package_struct.py
    python -m quartodoc build --config api/_quartodoc_config.yml
    python scripts/update_toc_api.py
fi

if [ "$BUILD_DATAMODEL" = 1 ]; then
    python scripts/generate_datamodel_docs.py
    python scripts/update_toc_datamodel.py
fi

python scripts/update_toc_replication.py $REPL_GATE_ARGS
python scripts/build_palette.py
python scripts/build_phase_map.py
python scripts/build_thumbnails.py
