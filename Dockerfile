# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.
FROM python:3.13-slim

LABEL maintainer="Leon Martin <leon.martin@bih-charite.de>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran make git perl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/auto-07p/auto-07p /auto-07p && \
    cd /auto-07p && ./configure && make && pip install .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jupyterlab uvicorn fastapi tvbo tvboptim

ENV AUTO_DIR=/auto-07p
ENV PYTHONPATH=/app
WORKDIR /app

EXPOSE 8000
EXPOSE 8888

ENTRYPOINT ["bash", "-c", "\
    MODE=${MODE:-api}; \
    echo 'Starting TVBO container in '$MODE' mode...'; \
    # Install from mounted source if available (dev mode) \
    if [ -f /app/pyproject.toml ]; then \
        echo 'Installing tvbo from mounted dev source...'; \
        pip install -e /app 2>&1 | tail -5; \
    fi; \
    if [ \"$MODE\" = \"api\" ]; then \
        echo 'Launching API server on port 8000...'; \
        exec uvicorn tvbo.api.main:app --host 0.0.0.0 --port 8000 --reload; \
    elif [ \"$MODE\" = \"jupyter\" ]; then \
        echo 'Launching Jupyter Lab on port 8888...'; \
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''; \
    else \
        echo 'ERROR: Mode not recognized. Use MODE=api or MODE=jupyter'; \
        exit 1; \
    fi"]
