# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.
# Stage 1: Build
FROM python:3.12-slim AS builder

LABEL maintainer="Leon Martin <leon.martin@bih-charite.de>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran make git bash build-essential \
    libopenblas-dev libhdf5-dev perl llvm-dev liblcms2-dev \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

SHELL ["/bin/bash", "-c"]

RUN git clone https://github.com/auto-07p/auto-07p && \
    cd auto-07p && ./configure && make && pip install .

ARG CACHEBUST=1
RUN --mount=type=secret,id=gitlab_token \
    bash -c '\
    GIT_TOKEN=$(cat /run/secrets/gitlab_token) && \
    pip install --no-cache-dir \
      jupyterlab uvicorn fastapi \
      git+https://${GIT_TOKEN}:${GIT_TOKEN}@git.bihealth.org/taherh/dynamicalsystems@dev_LM \
      git+https://${GIT_TOKEN}:${GIT_TOKEN}@git.bihealth.org/taherh/numcont@dev_LM \
      git+https://${GIT_TOKEN}:${GIT_TOKEN}@git.bihealth.org/bss/tvb-o/tvbo-python@main \
    '

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir git+https://github.com/virtual-brain-twins/tvb-ext-ontology.git

# Perform a lightweight import validation of tvbo. If we're cross-building on
# hardware/emulation lacking AVX, jax/jaxlib may raise an AVX-related RuntimeError.
# We treat that case as a warning (so the build can still succeed) while letting
# any other exceptions fail the build normally.
RUN python - <<'PY'
try:
    from tvbo.datamodel import tvbo_datamodel  # noqa: F401
    print("TVBO import OK")
except RuntimeError as e:
    if "AVX" in str(e):
        print("WARNING: AVX not available in build environment; skipping tvbo import validation.")
    else:
        raise
PY


# Stage 2: Runtime image
FROM python:3.12-slim AS runtime

LABEL maintainer="Leon Martin <leon.martin@bih-charite.de>"

COPY --from=builder /usr/local /usr/local

ENV AUTO_DIR=/auto-07p
WORKDIR /app

EXPOSE 8000
EXPOSE 8888

ENTRYPOINT ["bash", "-c", "\
    MODE=${MODE:-api}; \
    if [ \"$MODE\" = \"api\" ]; then \
    exec uvicorn tvbo.api.main:app --host 0.0.0.0 --port 8000; \
    elif [ \"$MODE\" = \"jupyter\" ]; then \
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''; \
    else \
    echo 'Mode not recognized. Exiting.'; \
    exit 1; fi"]
