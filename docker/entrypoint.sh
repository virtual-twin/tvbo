#!/bin/bash
set -e

# Honour an explicit command. `docker run <image> <cmd...>` passes <cmd...> here
# as "$@"; run it verbatim instead of the built-in service. Two shipped features
# rely on this: the docs-render pipeline (`bash -euxc '...'`, see
# .github/workflows/docs-deploy.yml) and `tvbo run --container`, which re-execs
# `tvbo <argv>` inside the image (tvbo/cli/run.py). Falling through to the MODE
# launcher for these would (a) ignore the requested command and (b) start uvicorn,
# importing tvbo from the mounted /work checkout — whose generated
# tvbo/datamodel/schema.py is untracked and absent, hence ModuleNotFoundError.
# With no command, "$@" is empty and we fall through to the default service.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

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
        # --reload is a dev-only feature (file watcher + reloader subprocess); enable it
        # only when DEV=1 so the shipped image runs a lean single-process server.
        RELOAD=""
        [ "${DEV:-0}" = "1" ] && RELOAD="--reload"
        exec uvicorn tvbo.api.main:app --host 0.0.0.0 --port 8000 ${RELOAD}
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
