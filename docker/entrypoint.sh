#!/bin/bash
set -e

MODE=${MODE:-api}
echo "Starting TVBO container in $MODE mode..."

# Install from mounted source if available (dev mode)
if [ -f /app/pyproject.toml ]; then
    echo "Installing tvbo from mounted dev source..."
    pip install -e /app 2>&1 | tail -5
fi

case "$MODE" in
    api)
        echo "Launching API server on port 8000..."
        exec uvicorn tvbo.api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    jupyter)
        echo "Launching Jupyter Lab on port 8888..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
    *)
        echo "ERROR: Mode '$MODE' not recognized. Use MODE=api or MODE=jupyter"
        exit 1
        ;;
esac
