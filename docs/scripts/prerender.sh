#!/bin/bash
# Pre-render script for Quarto docs.
# Set TVBO_SKIP_PRERENDER=1 to skip (used by `make docs-preview` after initial render).

if [ "$TVBO_SKIP_PRERENDER" = "1" ]; then
    echo "Skipping pre-render"
    exit 0
fi

set -e
python scripts/tvbo_package_struct.py
python -m quartodoc build --config api/_quartodoc_config.yml
python scripts/update_toc_api.py
python scripts/update_toc_replication.py
python scripts/generate_datamodel_docs.py
python scripts/update_toc_datamodel.py
python scripts/fix_mermaid_blocks.py
